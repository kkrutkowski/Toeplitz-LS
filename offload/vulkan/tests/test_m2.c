#include "common/math.h"
#include "fft/vk_dif.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Reference host Cooley-Tukey / DIF scaled-by-1/N FFT in double precision */
static void ref_fft_scaled(double* re, double* im, int logN, int sign) {
    size_t N = (size_t)1u << logN;
    for (size_t i = 0; i < N; ++i) {
        size_t j = vk_bit_reverse((uint32_t)i, logN);
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int s = 1; s <= logN; ++s) {
        size_t m = (size_t)1u << s;
        size_t half = m >> 1;
        double ang_step = (double)sign * 2.0 * M_PI / (double)m;
        for (size_t k = 0; k < N; k += m) {
            for (size_t j = 0; j < half; ++j) {
                double ang = ang_step * (double)j;
                double wr = cos(ang), wi = sin(ang);
                double ur = re[k + j], ui = im[k + j];
                double vr = re[k + j + half], vi = im[k + j + half];
                double tr = wr * vr - wi * vi;
                double ti = wr * vi + wi * vr;
                re[k + j]        = 0.5 * (ur + tr);
                im[k + j]        = 0.5 * (ui + ti);
                re[k + j + half] = 0.5 * (ur - tr);
                im[k + j + half] = 0.5 * (ui - ti);
            }
        }
    }
}

static int test_fft_case(VkDeviceContext* ctx, VkPrecKind prec, int logN, size_t bs, int sign) {
    VkFftLaunchConfig cfg = vk_fft_launch_config_default();
    cfg.autotune = VK_AUTOTUNE_OFF;
    VkDifFftPlan plan;
    if (!vk_dif_plan_init(&plan, ctx, prec, sign, logN, bs, &cfg)) {
        fprintf(stderr, "FAIL: vk_dif_plan_init(prec=%s, logN=%d)\n", vk_prec_tag(prec), logN);
        return 1;
    }

    size_t N = (size_t)1u << logN;
    size_t total = bs * N;
    size_t bytes = total * plan.complex_bytes;

    double* ref_re = (double*)malloc(total * sizeof(double));
    double* ref_im = (double*)malloc(total * sizeof(double));
    uint8_t* h_in = (uint8_t*)calloc(1, bytes);
    uint8_t* h_out = (uint8_t*)calloc(1, bytes);

    for (size_t b = 0; b < bs; ++b) {
        for (size_t i = 0; i < N; ++i) {
            double x = sin(0.013 * (double)(i + 1) + 0.7 * (double)b) * 0.5
                     + cos(0.037 * (double)(i + 3) - 0.3 * (double)b) * 0.4;
            double y = cos(0.019 * (double)(i + 2) + 0.5 * (double)b) * 0.3
                     - sin(0.029 * (double)(i + 5) + 0.2 * (double)b) * 0.4;
            size_t idx = b * N + i;
            if (prec == VK_PREC_HALF) {
                VkComplexh c = vk_complexh_from_doubles(x, y);
                ((VkComplexh*)h_in)[idx] = c;
                ref_re[idx] = vk_f16_bits_to_f32(c.re);
                ref_im[idx] = vk_f16_bits_to_f32(c.im);
            } else if (prec == VK_PREC_FLOAT) {
                float fx = (float)x, fy = (float)y;
                ((VkComplexf*)h_in)[idx] = (VkComplexf){fx, fy};
                ref_re[idx] = fx; ref_im[idx] = fy;
            } else if (prec == VK_PREC_DF) {
                VkDf dx = vk_df_from_double(x), dy = vk_df_from_double(y);
                ((VkComplexdf*)h_in)[idx] = (VkComplexdf){dx, dy};
                ref_re[idx] = vk_df_to_double(dx); ref_im[idx] = vk_df_to_double(dy);
            } else {
                ((VkComplex*)h_in)[idx] = (VkComplex){x, y};
                ref_re[idx] = x; ref_im[idx] = y;
            }
        }
        ref_fft_scaled(ref_re + b * N, ref_im + b * N, logN, sign);
    }

    VkBufferAlloc d_in = {0}, d_out = {0};
    VkCmdBatch batch = {0};
    int fail = 0;
    if (!vk_buf_create(ctx, bytes, VK_MEM_DEVICE_LOCAL, &d_in) ||
        !vk_buf_create(ctx, bytes, VK_MEM_DEVICE_LOCAL, &d_out) ||
        !vk_batch_init(&batch, ctx, 0) ||
        !vk_batch_upload_sync(&batch, h_in, vk_buf_slice(&d_in, 0, bytes), bytes) ||
        !vk_batch_begin(&batch) ||
        !vk_dif_record_execute(&plan, &batch, vk_buf_slice(&d_in, 0, bytes),
                               vk_buf_slice(&d_out, 0, bytes), bs, 0, 0, 0) ||
        !vk_batch_submit_and_wait(&batch) ||
        !vk_batch_download_sync(&batch, vk_buf_slice(&d_out, 0, bytes), h_out, bytes)) {
        fprintf(stderr, "FAIL: exec prec=%s logN=%d\n", vk_prec_tag(prec), logN);
        fail = 1;
        goto cleanup;
    }

    double err_sq = 0.0, ref_sq = 0.0;
    for (size_t idx = 0; idx < total; ++idx) {
        double gr = 0.0, gi = 0.0;
        if (prec == VK_PREC_HALF) {
            VkComplexh c = ((VkComplexh*)h_out)[idx];
            gr = vk_f16_bits_to_f32(c.re);
            gi = vk_f16_bits_to_f32(c.im);
        } else if (prec == VK_PREC_FLOAT) {
            VkComplexf c = ((VkComplexf*)h_out)[idx];
            gr = c.re; gi = c.im;
        } else if (prec == VK_PREC_DF) {
            VkComplexdf c = ((VkComplexdf*)h_out)[idx];
            gr = vk_df_to_double(c.re); gi = vk_df_to_double(c.im);
        } else {
            VkComplex c = ((VkComplex*)h_out)[idx];
            gr = c.re; gi = c.im;
        }
        double dr = gr - ref_re[idx], di = gi - ref_im[idx];
        err_sq += dr * dr + di * di;
        ref_sq += ref_re[idx] * ref_re[idx] + ref_im[idx] * ref_im[idx];
    }
    double rel_l2 = sqrt(err_sq / (ref_sq + 1e-30));
    double tol = (prec == VK_PREC_HALF) ? 2e-2 :
                 (prec == VK_PREC_FLOAT ? 2e-6 :
                 (prec == VK_PREC_DF ? 5e-13 : 5e-15));
    if (!isfinite(rel_l2) || rel_l2 > tol) {
        fprintf(stderr, "FAIL: dev=%d prec=%s logN=%d (%s) rel_l2=%.3e > tol=%.3e\n",
                vk_ctx_desc(ctx)->id, vk_prec_tag(prec), logN, plan.variant_name, rel_l2, tol);
        fail = 1;
    } else {
        printf("  PASS dev=%d prec=%-2s logN=%2d mode=%d var=%-24s rel_l2=%.2e\n",
               vk_ctx_desc(ctx)->id, vk_prec_tag(prec), logN, (int)plan.exec_mode,
               plan.variant_name, rel_l2);
    }

    /* Also test compact PSWF43 epilogue when not MultiStageGlobal */
    if (!fail && plan.exec_mode != VK_PLAN_MULTISTAGE_GLOBAL && logN <= 14) {
        size_t nout = (N >> 2) * 3u;
        size_t cbytes = bs * nout * plan.complex_bytes;
        VkBufferAlloc d_cout = {0};
        uint8_t* h_cout = (uint8_t*)calloc(1, cbytes);
        if (vk_buf_create(ctx, cbytes, VK_MEM_DEVICE_LOCAL, &d_cout) &&
            vk_batch_begin(&batch) &&
            vk_dif_record_execute_raw_compact(&plan, &batch, vk_buf_slice(&d_in, 0, bytes),
                                              vk_buf_slice(&d_cout, 0, cbytes), bs, 0, 0,
                                              2, vk_ctx_dummy_buffer(ctx), false, 0) &&
            vk_batch_submit_and_wait(&batch) &&
            vk_batch_download_sync(&batch, vk_buf_slice(&d_cout, 0, cbytes), h_cout, cbytes)) {
            double cerr_sq = 0.0, cref_sq = 0.0;
            for (size_t b = 0; b < bs; ++b) {
                for (size_t ci = 0; ci < nout; ++ci) {
                    size_t q = ci / 6u, r = ci % 6u;
                    size_t slot = (q << 3u) + r + (r >= 1u ? 1u : 0u) + (r >= 5u ? 1u : 0u);
                    size_t nat = vk_bit_reverse((uint32_t)slot, logN);
                    size_t oidx = b * nout + ci;
                    size_t ridx = b * N + nat;
                    double gr = 0.0, gi = 0.0;
                    if (prec == VK_PREC_HALF) {
                        VkComplexh c = ((VkComplexh*)h_cout)[oidx];
                        gr = vk_f16_bits_to_f32(c.re); gi = vk_f16_bits_to_f32(c.im);
                    } else if (prec == VK_PREC_FLOAT) {
                        VkComplexf c = ((VkComplexf*)h_cout)[oidx];
                        gr = c.re; gi = c.im;
                    } else if (prec == VK_PREC_DF) {
                        VkComplexdf c = ((VkComplexdf*)h_cout)[oidx];
                        gr = vk_df_to_double(c.re); gi = vk_df_to_double(c.im);
                    } else {
                        VkComplex c = ((VkComplex*)h_cout)[oidx];
                        gr = c.re; gi = c.im;
                    }
                    double dr = gr - ref_re[ridx], di = gi - ref_im[ridx];
                    cerr_sq += dr * dr + di * di;
                    cref_sq += ref_re[ridx] * ref_re[ridx] + ref_im[ridx] * ref_im[ridx];
                }
            }
            double crel = sqrt(cerr_sq / (cref_sq + 1e-30));
            if (!isfinite(crel) || crel > tol) {
                fprintf(stderr, "FAIL compact PSWF43: dev=%d prec=%s logN=%d crel=%.3e\n",
                        vk_ctx_desc(ctx)->id, vk_prec_tag(prec), logN, crel);
                fail = 1;
            }
        } else {
            fprintf(stderr, "FAIL compact exec: dev=%d prec=%s logN=%d\n",
                    vk_ctx_desc(ctx)->id, vk_prec_tag(prec), logN);
            fail = 1;
        }
        free(h_cout);
        vk_buf_destroy(&d_cout);
    }

cleanup:
    vk_batch_destroy(&batch);
    vk_buf_destroy(&d_in);
    vk_buf_destroy(&d_out);
    vk_dif_plan_destroy(&plan);
    free(ref_re); free(ref_im); free(h_in); free(h_out);
    return fail;
}

int main(void) {
    const VkDeviceDescriptor* devs = nullptr;
    size_t ndevs = vk_enumerate_devices(&devs);
    int fails = 0;
    const VkPrecKind precs[4] = {VK_PREC_HALF, VK_PREC_FLOAT, VK_PREC_DF, VK_PREC_DOUBLE};
    const int logNs_gpu[5] = {10, 11, 12, 13, 15};
    const int logNs_cpu[2] = {10, 12};

    for (size_t di = 0; di < ndevs; ++di) {
        VkDeviceContext* ctx = vk_ctx_get_shared((int)di, nullptr, 0);
        if (!ctx) { ++fails; continue; }
        printf("=== Device [%zu] %s ===\n", di, devs[di].name);
        const int* lns = devs[di].is_gpu ? logNs_gpu : logNs_cpu;
        int nl = devs[di].is_gpu ? 5 : 2;
        for (int pi = 0; pi < 4; ++pi) {
            for (int li = 0; li < nl; ++li) {
                fails += test_fft_case(ctx, precs[pi], lns[li], devs[di].is_gpu ? 4 : 2, +1);
            }
        }
    }
    printf("\nMilestone 2 summary: %s (fails=%d)\n", fails == 0 ? "ALL PASS" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
