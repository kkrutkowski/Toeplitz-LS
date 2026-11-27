/* Milestone 3 Verification: Quadruple-Precision Vulkan Type-1 NuFFT
 * Tests PSWF21 & PSWF43, SlotBitReversed (fused epilogue) & Natural orders,
 * SinglePass (N=1024) & TwoPass/multi-window (N=8192), Sign=+1/-1, all 4 precisions
 * across every detected Vulkan device (NVIDIA RTX 5060, AMD Radeon 610M RADV, llvmpipe). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nufft/vk_nufft1.h"

static void ref_nufft1(const double* x, const double* yr, const double* yi, int M, int nout, int nfft, double df, int sign, double* out_r, double* out_i) {
    const double inv_nfft = 1.0 / (double)nfft;
    const double two_pi_s = (double)sign * 2.0 * M_PI * df;
    for (int k = 0; k < nout; ++k) {
        double sr = 0.0, si = 0.0;
        const double wk = two_pi_s * (double)k;
        for (int m = 0; m < M; ++m) {
            double ang = wk * x[m];
            double c = cos(ang), s = sin(ang);
            sr += yr[m] * c - yi[m] * s;
            si += yr[m] * s + yi[m] * c;
        }
        out_r[k] = sr * inv_nfft;
        out_i[k] = si * inv_nfft;
    }
}

static int run_nufft_case(VkDeviceContext* ctx, VkCmdBatch* batch, VkPrecKind prec, int sign, int M, int N, int w, VkPswfMode mode, VkNufftOutOrder order,
                          size_t bs, double tol) {
    VkNufft1LaunchConfig cfg = vk_nufft1_launch_config_default();
    cfg.fft_cfg.autotune = VK_AUTOTUNE_OFF; /* deterministic fast test */

    VkNufft1Plan plan = {0};
    if (!vk_nufft1_plan_init(&plan, ctx, prec, sign, M, N, w, mode, bs, &cfg)) {
        printf("    FAIL: plan_init prec=%s N=%d w=%d mode=%d\n", vk_prec_tag(prec), N, w, (int)mode);
        return 1;
    }

    int nout = plan.nout;
    double df = 1.0 / (double)plan.nfft;
    double* hx = (double*)malloc((size_t)M * sizeof(double));
    double* hy_r = (double*)malloc(bs * (size_t)M * sizeof(double));
    double* hy_i = (double*)malloc(bs * (size_t)M * sizeof(double));
    for (int m = 0; m < M; ++m) {
        hx[m] = ((double)((m * 73 + 19) % 997) / 997.0) * 113.0 - 20.0;
    }
    for (size_t b = 0; b < bs; ++b) {
        for (int m = 0; m < M; ++m) {
            uint64_t s = (uint64_t)(b * 1009u + (size_t)m * 37u + 11u);
            hy_r[b * (size_t)M + (size_t)m] = -0.5 + (double)((s * 2654435761ULL) & 0xFFFFu) / 65535.0;
            hy_i[b * (size_t)M + (size_t)m] = -0.5 + (double)(((s + 7u) * 40503ULL) & 0xFFFFu) / 65535.0;
        }
    }

    size_t x_bytes = (size_t)M * ((prec == VK_PREC_DOUBLE) ? sizeof(double) : sizeof(VkDf));
    size_t y_bytes = bs * (size_t)M * plan.complex_bytes;
    size_t o_bytes = bs * (size_t)nout * plan.complex_bytes;

    void* h_x_dev = malloc(x_bytes);
    void* h_y_dev = malloc(y_bytes);
    void* h_o_raw = malloc(o_bytes);
    void* h_o_nat = malloc(o_bytes);

    for (int m = 0; m < M; ++m) {
        if (prec == VK_PREC_DOUBLE)
            ((double*)h_x_dev)[m] = hx[m];
        else
            ((VkDf*)h_x_dev)[m] = vk_df_from_double(hx[m]);
    }
    for (size_t i = 0; i < bs * (size_t)M; ++i) {
        if (prec == VK_PREC_HALF)
            ((VkComplexh*)h_y_dev)[i] = vk_complexh_from_doubles(hy_r[i], hy_i[i]);
        else if (prec == VK_PREC_FLOAT)
            ((VkComplexf*)h_y_dev)[i] = (VkComplexf){(float)hy_r[i], (float)hy_i[i]};
        else if (prec == VK_PREC_DF)
            ((VkComplexdf*)h_y_dev)[i] = (VkComplexdf){vk_df_from_double(hy_r[i]), vk_df_from_double(hy_i[i])};
        else
            ((VkComplex*)h_y_dev)[i] = (VkComplex){hy_r[i], hy_i[i]};
    }

    VkBufferAlloc d_x = {0}, d_y = {0}, d_out = {0};
    bool ok = vk_buf_create(ctx, x_bytes, VK_MEM_DEVICE_LOCAL, &d_x) && vk_buf_create(ctx, y_bytes, VK_MEM_DEVICE_LOCAL, &d_y) &&
              vk_buf_create(ctx, o_bytes, VK_MEM_DEVICE_LOCAL, &d_out) && vk_batch_upload_sync(batch, h_x_dev, vk_buf_slice(&d_x, 0, VK_WHOLE_SIZE), x_bytes) &&
              vk_batch_upload_sync(batch, h_y_dev, vk_buf_slice(&d_y, 0, VK_WHOLE_SIZE), y_bytes) && vk_batch_begin(batch) &&
              vk_nufft1_record_set_points(&plan, batch, vk_buf_slice(&d_x, 0, VK_WHOLE_SIZE), M, df, 1) &&
              vk_nufft1_record_execute(&plan, batch, vk_buf_slice(&d_y, 0, VK_WHOLE_SIZE), 0u, vk_buf_slice(&d_out, 0, VK_WHOLE_SIZE), 0u, bs, order) &&
              vk_batch_submit_and_wait(batch) && vk_batch_download_sync(batch, vk_buf_slice(&d_out, 0, VK_WHOLE_SIZE), h_o_raw, o_bytes);

    int fail = 0;
    double max_rel = 0.0;
    if (!ok) {
        printf("    FAIL: execute failed prec=%s N=%d w=%d mode=%d ord=%d\n", vk_prec_tag(prec), N, w, (int)mode, (int)order);
        fail = 1;
    } else {
        if (order == VK_NUFFT_ORDER_SLOT_BIT_REVERSED) {
            vk_nufft1_unpermute_slots_to_natural_host(h_o_raw, h_o_nat, plan.complex_bytes, N, mode, bs);
        } else {
            memcpy(h_o_nat, h_o_raw, o_bytes);
        }

        double* rr = (double*)malloc((size_t)nout * sizeof(double));
        double* ri = (double*)malloc((size_t)nout * sizeof(double));
        /* Verify first and last batch elements against exact direct DFT */
        size_t check_bs[2] = {0u, bs - 1u};
        size_t ncheck = (bs > 1u) ? 2u : 1u;
        for (size_t ci = 0; ci < ncheck; ++ci) {
            size_t b = check_bs[ci];
            ref_nufft1(hx, hy_r + b * (size_t)M, hy_i + b * (size_t)M, M, nout, plan.nfft, df, sign, rr, ri);
            double err2 = 0.0, ref2 = 0.0;
            for (int k = 0; k < nout; ++k) {
                size_t idx = b * (size_t)nout + (size_t)k;
                double gr = 0.0, gi = 0.0;
                if (prec == VK_PREC_HALF) {
                    VkComplexh v = ((VkComplexh*)h_o_nat)[idx];
                    gr = vk_f16_bits_to_f32(v.re);
                    gi = vk_f16_bits_to_f32(v.im);
                } else if (prec == VK_PREC_FLOAT) {
                    VkComplexf v = ((VkComplexf*)h_o_nat)[idx];
                    gr = v.re;
                    gi = v.im;
                } else if (prec == VK_PREC_DF) {
                    VkComplexdf v = ((VkComplexdf*)h_o_nat)[idx];
                    gr = vk_df_to_double(v.re);
                    gi = vk_df_to_double(v.im);
                } else {
                    VkComplex v = ((VkComplex*)h_o_nat)[idx];
                    gr = v.re;
                    gi = v.im;
                }
                double dr = gr - rr[k], di = gi - ri[k];
                err2 += dr * dr + di * di;
                ref2 += rr[k] * rr[k] + ri[k] * ri[k];
            }
            double rel = sqrt(err2 / (ref2 + 1e-30));
            if (rel > max_rel) max_rel = rel;
        }
        free(rr);
        free(ri);

        if (!isfinite(max_rel) || max_rel > tol) {
            printf("    FAIL: prec=%-2s sign=%+d N=%5d w=%2d mode=%s ord=%s spread=%-22s rel_l2=%.3e (tol=%.1e)\n", vk_prec_tag(prec), sign, N, w,
                   mode == VK_PSWF43 ? "43" : "21", order == VK_NUFFT_ORDER_NATURAL ? "nat" : "slot", plan.spread_variant_name, max_rel, tol);
            fail = 1;
        } else {
            printf("    PASS: prec=%-2s sign=%+d N=%5d w=%2d mode=%s ord=%s spread=%-22s rel_l2=%.3e\n", vk_prec_tag(prec), sign, N, w,
                   mode == VK_PSWF43 ? "43" : "21", order == VK_NUFFT_ORDER_NATURAL ? "nat" : "slot", plan.spread_variant_name, max_rel);
        }
    }

    vk_buf_destroy(&d_x);
    vk_buf_destroy(&d_y);
    vk_buf_destroy(&d_out);
    free(hx);
    free(hy_r);
    free(hy_i);
    free(h_x_dev);
    free(h_y_dev);
    free(h_o_raw);
    free(h_o_nat);
    vk_nufft1_plan_destroy(&plan);
    return fail;
}

int main(void) {
    const VkDeviceDescriptor* devs = nullptr;
    size_t ndevs = vk_enumerate_devices(&devs);
    if (ndevs == 0) {
        fprintf(stderr, "No Vulkan devices found!\n");
        return 1;
    }

    int total_fails = 0;
    for (size_t d = 0; d < ndevs; ++d) {
        char err[256] = {0};
        VkDeviceContext* ctx = vk_ctx_get_shared((int)d, err, sizeof(err));
        if (!ctx) {
            fprintf(stderr, "Failed to init dev %zu: %s\n", d, err);
            ++total_fails;
            continue;
        }
        const VkDeviceDescriptor* desc = vk_ctx_desc(ctx);
        printf("=== Device [%zu] %s (%s) ===\n", d, desc->name, desc->backend);

        VkCmdBatch batch = {0};
        if (!vk_batch_init(&batch, ctx, 0u)) {
            ++total_fails;
            continue;
        }

        const struct {
            VkPrecKind prec;
            int w;
            double tol;
        } precs[4] = {
            {VK_PREC_HALF, 5, 1.5e-2},
            {VK_PREC_FLOAT, 9, 2.0e-5},
            {VK_PREC_DF, 16, 1.0e-11},
            {VK_PREC_DOUBLE, 18, 5.0e-13},
        };

        for (int pi = 0; pi < 4; ++pi) {
            if (precs[pi].prec == VK_PREC_HALF && !desc->has_fp16) continue;
            if (precs[pi].prec == VK_PREC_DOUBLE && !desc->has_fp64) continue;

            /* SinglePass PSWF43 SlotBitReversed (primary fast path, sign=+1) */
            total_fails +=
                run_nufft_case(ctx, &batch, precs[pi].prec, +1, 257, 1024, precs[pi].w, VK_PSWF43, VK_NUFFT_ORDER_SLOT_BIT_REVERSED, 5, precs[pi].tol);
            /* SinglePass PSWF21 Natural (sign=-1) */
            total_fails += run_nufft_case(ctx, &batch, precs[pi].prec, -1, 193, 1024, precs[pi].w, VK_PSWF21, VK_NUFFT_ORDER_NATURAL, 3, precs[pi].tol);
            /* TwoPass + multi-window buckets PSWF43 SlotBitReversed (N=8192 -> nfft=16384) */
            if (desc->is_gpu) {
                total_fails +=
                    run_nufft_case(ctx, &batch, precs[pi].prec, +1, 513, 8192, precs[pi].w, VK_PSWF43, VK_NUFFT_ORDER_SLOT_BIT_REVERSED, 3, precs[pi].tol);
            }
        }

        vk_batch_destroy(&batch);
    }

    printf("\nMilestone 3 Test Summary: total_fails=%d\n", total_fails);
    return total_fails == 0 ? 0 : 1;
}
