#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitrev/vk_bitrev.h"
#include "common/math.h"
#include "common/vk_context.h"
#include "common/vk_device.h"

static int test_bitrev_device(VkDeviceContext* ctx, int logN, size_t elem_bytes, size_t bs) {
    VkFftLaunchConfig cfg = vk_fft_launch_config_default();
    cfg.autotune = VK_AUTOTUNE_OFF;
    VkBitRevEngine eng;
    if (!vk_bitrev_init(&eng, ctx, logN, elem_bytes, "f", &cfg)) {
        fprintf(stderr, "FAIL: vk_bitrev_init(logN=%d, eb=%zu)\n", logN, elem_bytes);
        return 1;
    }

    size_t N = (size_t)1u << logN;
    size_t total_elems = bs * N;
    size_t total_bytes = total_elems * elem_bytes;
    uint32_t words_per_elem = (uint32_t)(elem_bytes / 4u);

    uint32_t* h_in = (uint32_t*)malloc(total_bytes);
    uint32_t* h_out = (uint32_t*)malloc(total_bytes);
    for (size_t b = 0; b < bs; ++b) {
        for (size_t i = 0; i < N; ++i) {
            size_t idx = (b * N + i) * words_per_elem;
            for (uint32_t w = 0; w < words_per_elem; ++w) {
                h_in[idx + w] = (uint32_t)((b << 20) ^ (i << 3) ^ (w + 1u) * 0x9e3779b9u);
            }
        }
    }

    VkBufferAlloc d_src = {0}, d_dst = {0};
    VkCmdBatch batch = {0};
    int fails = 0;
    if (!vk_buf_create(ctx, total_bytes, VK_MEM_DEVICE_LOCAL, &d_src) || !vk_buf_create(ctx, total_bytes, VK_MEM_DEVICE_LOCAL, &d_dst) ||
        !vk_batch_init(&batch, ctx, 0)) {
        fprintf(stderr, "FAIL: alloc/batch init\n");
        fails = 1;
        goto done;
    }

    for (int ip_i = 0; ip_i < 2; ++ip_i) {
        bool in_place = (ip_i == 1);
        const VkBitRevCandidate* cands = in_place ? eng.valid_ip : eng.valid_oop;
        size_t ncands = in_place ? eng.num_ip : eng.num_oop;
        for (size_t ci = 0; ci < ncands; ++ci) {
            if (!vk_batch_upload_sync(&batch, h_in, vk_buf_slice(in_place ? &d_dst : &d_src, 0, total_bytes), total_bytes)) {
                fprintf(stderr, "FAIL: upload\n");
                ++fails;
                continue;
            }
            if (!vk_batch_begin(&batch) ||
                !vk_bitrev_record_candidate(&eng, &batch, in_place, cands[ci].variant_id, vk_buf_slice(&d_src, 0, total_bytes),
                                            vk_buf_slice(&d_dst, 0, total_bytes), bs, 0) ||
                !vk_batch_submit_and_wait(&batch) || !vk_batch_download_sync(&batch, vk_buf_slice(&d_dst, 0, total_bytes), h_out, total_bytes)) {
                fprintf(stderr, "FAIL: exec %s (%s)\n", cands[ci].variant_id, in_place ? "IP" : "OOP");
                ++fails;
                continue;
            }
            bool mismatch = false;
            for (size_t b = 0; b < bs && !mismatch; ++b) {
                for (size_t i = 0; i < N && !mismatch; ++i) {
                    size_t r = vk_bit_reverse((uint32_t)i, logN);
                    size_t dst_idx = (b * N + i) * words_per_elem;
                    size_t src_idx = (b * N + r) * words_per_elem;
                    for (uint32_t w = 0; w < words_per_elem; ++w) {
                        if (h_out[dst_idx + w] != h_in[src_idx + w]) {
                            fprintf(stderr, "MISMATCH %s (%s) logN=%d eb=%zu b=%zu i=%zu r=%zu\n", cands[ci].variant_id, in_place ? "IP" : "OOP", logN,
                                    elem_bytes, b, i, r);
                            mismatch = true;
                            ++fails;
                            break;
                        }
                    }
                }
            }
        }
    }

done:
    vk_batch_destroy(&batch);
    vk_buf_destroy(&d_src);
    vk_buf_destroy(&d_dst);
    vk_bitrev_destroy(&eng);
    free(h_in);
    free(h_out);
    return fails;
}

int main(void) {
    const VkDeviceDescriptor* devs = nullptr;
    size_t ndevs = vk_enumerate_devices(&devs);
    printf("Found %zu Vulkan devices:\n", ndevs);
    for (size_t i = 0; i < ndevs; ++i) {
        printf("  [%d] %s (%s/%s, queues=%zu, %.2f GiB, fp16=%d, fp64=%d)\n", devs[i].id, devs[i].name, devs[i].backend, devs[i].device_type,
               devs[i].max_compute_units, (double)devs[i].global_mem_size / (1024.0 * 1024.0 * 1024.0), devs[i].has_fp16 ? 1 : 0, devs[i].has_fp64 ? 1 : 0);
    }
    if (ndevs == 0) return 1;

    int total_fails = 0;
    for (size_t i = 0; i < ndevs; ++i) {
        char err[256] = {0};
        VkDeviceContext* ctx = vk_ctx_get_shared((int)i, err, sizeof(err));
        if (!ctx) {
            fprintf(stderr, "FAIL: ctx[%zu]: %s\n", i, err);
            ++total_fails;
            continue;
        }
        const VkDeviceFftConfig* cfg = vk_ctx_config(ctx);
        VkDeviceProbeResult sg = vk_ctx_probe_subgroup(ctx);
        VkDeviceProbeResult df = vk_ctx_probe_df_kat(ctx);
        VkDeviceProbeResult h16 = vk_ctx_probe_fp16_kat(ctx);
        size_t streams = vk_device_recommended_stream_count(cfg);
        size_t budget = vk_device_per_stream_mem_budget(cfg, streams);
        printf(
            "Device [%zu] %s: smem=%zu, sg=%zu (eff=%zu), streams=%zu, budget=%.1f MiB/stream, "
            "df_kat=%s (veltkamp=%d), fp16_kat=%s\n",
            i, cfg->device_name, cfg->local_mem_size, sg.raw_sg_size, sg.effective_sg_size, streams, (double)budget / (1024.0 * 1024.0),
            df.df_kat_passed ? "PASS" : df.df_kat_error, df.df_use_veltkamp ? 1 : 0, h16.fp16_kat_passed ? "PASS" : h16.fp16_kat_error);
        if (!df.df_kat_passed) ++total_fails;
        if (cfg->has_fp16 && !h16.fp16_kat_passed) ++total_fails;

        /* Test all bitrev strategies across 4B, 8B, 16B and logN=10, 12 */
        int logNs[2] = {10, 12};
        size_t ebs[3] = {4, 8, 16};
        for (int li = 0; li < 2; ++li) {
            for (int ei = 0; ei < 3; ++ei) {
                total_fails += test_bitrev_device(ctx, logNs[li], ebs[ei], cfg->is_gpu ? 8 : 2);
            }
        }
    }

    printf("\nMilestone 1 summary: %s (fails=%d)\n", total_fails == 0 ? "ALL PASS" : "FAILED", total_fails);
    return total_fails == 0 ? 0 : 1;
}
