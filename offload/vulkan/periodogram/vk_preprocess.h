#ifndef TLS_VK_PREPROCESS_H
#define TLS_VK_PREPROCESS_H

/* Plain C23 Lock-Free CPU Host Preprocessing for Vulkan FastChi2 & IHS Periodograms.
 * Mirrors offload/sycl/periodogram/preprocess.hpp & src/chi2per.c identically,
 * using reusable _Thread_local scratch buffers for zero steady-state heap allocations. */

#include "common/math.h"
#include "common/vk_device.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum VkTlsStatus {
    TLS_VK_OK             =  0,
    TLS_VK_ERR_ARGUMENT   = -1,
    TLS_VK_ERR_BACKEND    = -2,
    TLS_VK_ERR_ALLOC      = -3,
    TLS_VK_ERR_DEGENERATE = -4,
    TLS_VK_ERR_SOLVER     = -5,
    TLS_VK_ERR_DEVICE     = -6,
    TLS_VK_ERR_RUNTIME    = -7,
    TLS_VK_ERR_NO_FP16    = -8,
    TLS_VK_ERR_NO_FP64    = -9
} VkTlsStatus;

typedef enum VkTlsBackend {
    TLS_VK_BACKEND_PSWF43 = 1,
    TLS_VK_BACKEND_PSWF21 = 2
} VkTlsBackend;

typedef enum VkNormKind {
    VK_NORM_STANDARD     = 0,
    VK_NORM_ANOVA        = 1,
    VK_NORM_NLL          = 2,
    VK_NORM_REGULARIZED  = 3,
    VK_NORM_UNNORMALIZED = 4
} VkNormKind;

typedef struct VkCybenkoModel {
    double var_S;
    double mean_S;
    double k_eff;
    double theta_eff;
    double lgamma_k;
    double inv_2d;
    double sum_N_eff;
} VkCybenkoModel;

static inline VkCybenkoModel vk_init_cybenko_model(int d, double neff_w) {
    VkCybenkoModel m = {0};
    double d_dbl = (double)d;
    double M_dbl = neff_w;
    double inv_M = 1.0 / M_dbl;
    double inv_M2 = inv_M * inv_M;

    double term1 = 2.0 * d_dbl;
    double term2 = (8.0 * d_dbl * d_dbl - 14.0 * d_dbl) * inv_M;
    double term3 = ((184.0 / 3.0) * d_dbl * d_dbl * d_dbl - 98.0 * d_dbl * d_dbl + (158.0 / 3.0) * d_dbl) * inv_M2;
    m.var_S = term1 + term2 + term3;
    m.mean_S = 2.0 * d_dbl;
    m.k_eff = (m.mean_S * m.mean_S) / m.var_S;
    m.theta_eff = m.var_S / m.mean_S;
    m.lgamma_k = lgamma(m.k_eff);
    m.inv_2d = 1.0 / (2.0 * d_dbl);
    double n = 2.0 * d_dbl;
    m.sum_N_eff = n * M_dbl - n * (n + 1.0) + (n * (n + 1.0) * (2.0 * n + 1.0)) / (6.0 * M_dbl);
    return m;
}

typedef struct VkHostFastChi2Data {
    void*   x;        /* M * 8B (VkDf or double) */
    void*   w;        /* M * scalar_bytes */
    void*   yw;       /* M * scalar_bytes */
    double* w_tmp;    /* M * 8B reusable scratch */
    size_t  cap_M;
    double  ws_fft;
    double  yws_fft;
    double  chi2_ref;
    double  neff_w;
    double  b_param;
    double  row_scale;
    VkCybenkoModel cyb;
} VkHostFastChi2Data;

typedef struct VkHostIhsData {
    void*   x;        /* M * 8B (VkDf or double) */
    void*   wy;       /* M * scalar_bytes */
    double* w_tmp;    /* M * 8B reusable scratch */
    size_t  cap_M;
    double  standard_scale;
    double  nll_scale;
    double  norm_neff;
} VkHostIhsData;

static inline bool vk_host_fc2_ensure(VkHostFastChi2Data* d, size_t M) {
    if (M <= d->cap_M) return true;
    size_t nc = d->cap_M + (d->cap_M >> 1);
    if (nc < M) nc = M;
    void* nx = realloc(d->x, nc * 8u);
    void* nw = realloc(d->w, nc * 8u);
    void* ny = realloc(d->yw, nc * 8u);
    double* nt = (double*)realloc(d->w_tmp, nc * sizeof(double));
    if (!nx || !nw || !ny || !nt) {
        if (nx) d->x = nx;
        if (nw) d->w = nw;
        if (ny) d->yw = ny;
        if (nt) d->w_tmp = nt;
        return false;
    }
    d->x = nx; d->w = nw; d->yw = ny; d->w_tmp = nt;
    d->cap_M = nc;
    return true;
}

static inline bool vk_host_ihs_ensure(VkHostIhsData* d, size_t M) {
    if (M <= d->cap_M) return true;
    size_t nc = d->cap_M + (d->cap_M >> 1);
    if (nc < M) nc = M;
    void* nx = realloc(d->x, nc * 8u);
    void* nwy = realloc(d->wy, nc * 8u);
    double* nt = (double*)realloc(d->w_tmp, nc * sizeof(double));
    if (!nx || !nwy || !nt) {
        if (nx) d->x = nx;
        if (nwy) d->wy = nwy;
        if (nt) d->w_tmp = nt;
        return false;
    }
    d->x = nx; d->wy = nwy; d->w_tmp = nt;
    d->cap_M = nc;
    return true;
}

static inline int vk_preprocess_fastchi2(
    VkPrecKind prec,
    const double* t,
    const float* y_f,
    const double* y_d,
    const float* dy_f,
    const double* dy_d,
    int M,
    int degree,
    int nfft,
    VkHostFastChi2Data* out) {
    if (!vk_host_fc2_ensure(out, (size_t)M)) return TLS_VK_ERR_ALLOC;
    const bool use_d = (y_d != nullptr && dy_d != nullptr);
    double* w_d = out->w_tmp;

    const double t_mid = 0.5 * (t[0] + t[M - 1]);
    double dy_min = use_d ? dy_d[0] : (double)dy_f[0];
    for (int m = 0; m < M; ++m) {
        double dm = use_d ? dy_d[m] : (double)dy_f[m];
        if (!isfinite(dm) || dm <= 0.0) return TLS_VK_ERR_ARGUMENT;
        if (dm < dy_min) dy_min = dm;
        double xm = t[m] - t_mid;
        if (prec == VK_PREC_DOUBLE) ((double*)out->x)[m] = xm;
        else                        ((VkDf*)out->x)[m] = vk_df_from_double(xm);
    }

    double ws_raw = 0.0;
    for (int m = 0; m < M; ++m) {
        double dm = use_d ? dy_d[m] : (double)dy_f[m];
        double inv_dy = dy_min / dm;
        w_d[m] = inv_dy * inv_dy;
        ws_raw += w_d[m];
    }
    if (ws_raw == 0.0 || !isfinite(ws_raw)) return TLS_VK_ERR_DEGENERATE;

    double w_target = sqrt((double)M);
    if (prec == VK_PREC_HALF) {
        double sq_nfft = sqrt((double)nfft);
        if (sq_nfft > w_target) w_target = sq_nfft;
        if (w_target > 1024.0) w_target = 1024.0;
    }
    const double w_scale = w_target / ws_raw;
    const double y0 = use_d ? y_d[0] : (double)y_f[0];

    double ws = 0.0, sum_w2 = 0.0, wy = 0.0;
    for (int m = 0; m < M; ++m) {
        double ym = use_d ? y_d[m] : (double)y_f[m];
        if (!isfinite(ym)) return TLS_VK_ERR_ARGUMENT;
        double wm = w_d[m] * w_scale;
        w_d[m] = wm;
        ws += wm;
        sum_w2 += wm * wm;
        wy += wm * (ym - y0);
    }
    if (ws == 0.0 || !isfinite(ws)) return TLS_VK_ERR_DEGENERATE;

    const double dy_mean = wy / ws;
    double yc_max = 0.0;
    for (int m = 0; m < M; ++m) {
        double ym = use_d ? y_d[m] : (double)y_f[m];
        double a = fabs((ym - y0) - dy_mean);
        if (a > yc_max) yc_max = a;
    }
    if (yc_max == 0.0 || !isfinite(yc_max)) return TLS_VK_ERR_DEGENERATE;

    const double inv_yc_max = 1.0 / yc_max;
    double chi2_tmp = 0.0;
    for (int m = 0; m < M; ++m) {
        double ym = use_d ? y_d[m] : (double)y_f[m];
        double yc_u = ((ym - y0) - dy_mean) * inv_yc_max;
        chi2_tmp += w_d[m] * yc_u * yc_u;
    }
    if (chi2_tmp == 0.0 || !isfinite(chi2_tmp)) return TLS_VK_ERR_DEGENERATE;

    const double yc_scale = inv_yc_max * sqrt(ws / chi2_tmp);
    double chi2_ref = 0.0, yws = 0.0;
    if (prec == VK_PREC_HALF) {
        uint16_t* restrict ow = (uint16_t*)out->w;
        uint16_t* restrict oyw = (uint16_t*)out->yw;
        for (int m = 0; m < M; ++m) {
            double ym = use_d ? y_d[m] : (double)y_f[m];
            double yc = ((ym - y0) - dy_mean) * yc_scale;
            double wm = w_d[m];
            double ywm = yc * wm;
            yws += ywm;
            chi2_ref += yc * ywm;
            ow[m]  = vk_f32_to_f16_bits((float)wm);
            oyw[m] = vk_f32_to_f16_bits((float)ywm);
        }
    } else if (prec == VK_PREC_FLOAT) {
        float* restrict ow = (float*)out->w;
        float* restrict oyw = (float*)out->yw;
        for (int m = 0; m < M; ++m) {
            double ym = use_d ? y_d[m] : (double)y_f[m];
            double yc = ((ym - y0) - dy_mean) * yc_scale;
            double wm = w_d[m];
            double ywm = yc * wm;
            yws += ywm;
            chi2_ref += yc * ywm;
            ow[m]  = (float)wm;
            oyw[m] = (float)ywm;
        }
    } else if (prec == VK_PREC_DF) {
        VkDf* restrict ow = (VkDf*)out->w;
        VkDf* restrict oyw = (VkDf*)out->yw;
        for (int m = 0; m < M; ++m) {
            double ym = use_d ? y_d[m] : (double)y_f[m];
            double yc = ((ym - y0) - dy_mean) * yc_scale;
            double wm = w_d[m];
            double ywm = yc * wm;
            yws += ywm;
            chi2_ref += yc * ywm;
            ow[m]  = vk_df_from_double(wm);
            oyw[m] = vk_df_from_double(ywm);
        }
    } else {
        double* restrict ow = (double*)out->w;
        double* restrict oyw = (double*)out->yw;
        for (int m = 0; m < M; ++m) {
            double ym = use_d ? y_d[m] : (double)y_f[m];
            double yc = ((ym - y0) - dy_mean) * yc_scale;
            double wm = w_d[m];
            double ywm = yc * wm;
            yws += ywm;
            chi2_ref += yc * ywm;
            ow[m]  = wm;
            oyw[m] = ywm;
        }
    }
    if (chi2_ref == 0.0 || !isfinite(chi2_ref)) return TLS_VK_ERR_DEGENERATE;

    out->neff_w  = (ws * ws) / sum_w2;
    out->cyb     = vk_init_cybenko_model(degree, out->neff_w);
    out->b_param = 0.5 * (out->neff_w - 1.0 - 2.0 * (double)degree);
    if (prec == VK_PREC_HALF) {
        /* Use exact w_target (not FP64-summed ws) so row_scale is bit-identical across
         * lightcurves and never triggers spurious d_slot_deconv LUT rebuilds. */
        double inv_wt  = 1.0 / w_target;
        out->row_scale = (double)nfft * inv_wt;
        out->ws_fft    = ws * inv_wt;
        out->yws_fft   = yws * inv_wt;
        out->chi2_ref  = chi2_ref * inv_wt;
    } else {
        double inv_nfft = 1.0 / (double)nfft;
        out->row_scale = 1.0;
        out->ws_fft    = ws * inv_nfft;
        out->yws_fft   = yws * inv_nfft;
        out->chi2_ref  = chi2_ref * inv_nfft;
    }
    return TLS_VK_OK;
}

static inline int vk_preprocess_ihs(
    VkPrecKind prec,
    const double* t,
    const float* y_f,
    const double* y_d,
    const float* dy_f,
    const double* dy_d,
    int M,
    int nfft,
    VkHostIhsData* out) {
    if (!vk_host_ihs_ensure(out, (size_t)M)) return TLS_VK_ERR_ALLOC;
    const bool use_d = (y_d != nullptr && dy_d != nullptr);
    double* w_d = out->w_tmp;

    const double t_mid = 0.5 * (t[0] + t[M - 1]);
    double dy_min = use_d ? dy_d[0] : (double)dy_f[0];
    for (int m = 0; m < M; ++m) {
        double dm = use_d ? dy_d[m] : (double)dy_f[m];
        if (!isfinite(dm) || dm <= 0.0) return TLS_VK_ERR_ARGUMENT;
        if (dm < dy_min) dy_min = dm;
        double xm = t[m] - t_mid;
        if (prec == VK_PREC_DOUBLE) ((double*)out->x)[m] = xm;
        else                        ((VkDf*)out->x)[m] = vk_df_from_double(xm);
    }

    double ws_raw = 0.0;
    for (int m = 0; m < M; ++m) {
        double dm = use_d ? dy_d[m] : (double)dy_f[m];
        double inv_dy = dy_min / dm;
        w_d[m] = inv_dy * inv_dy;
        ws_raw += w_d[m];
    }
    if (ws_raw == 0.0 || !isfinite(ws_raw)) return TLS_VK_ERR_DEGENERATE;

    double w_target = sqrt((double)M);
    if (prec == VK_PREC_HALF) {
        double sq_nfft = sqrt((double)nfft);
        if (sq_nfft > w_target) w_target = sq_nfft;
        if (w_target > 1024.0) w_target = 1024.0;
    }
    const double w_scale = w_target / ws_raw;
    const double y0 = use_d ? y_d[0] : (double)y_f[0];

    double ws = 0.0, wy_mean = 0.0;
    for (int m = 0; m < M; ++m) {
        double ym = use_d ? y_d[m] : (double)y_f[m];
        if (!isfinite(ym)) return TLS_VK_ERR_ARGUMENT;
        double wm = w_d[m] * w_scale;
        w_d[m] = wm;
        ws += wm;
        wy_mean += wm * (ym - y0);
    }
    if (ws == 0.0 || !isfinite(ws)) return TLS_VK_ERR_DEGENERATE;

    const double dy_mean = wy_mean / ws;
    double yc_max = 0.0;
    for (int m = 0; m < M; ++m) {
        double ym = use_d ? y_d[m] : (double)y_f[m];
        double a = fabs((ym - y0) - dy_mean);
        if (a > yc_max) yc_max = a;
    }
    if (yc_max == 0.0 || !isfinite(yc_max)) return TLS_VK_ERR_DEGENERATE;

    const double inv_yc_max = 1.0 / yc_max;
    double chi2_tmp = 0.0;
    for (int m = 0; m < M; ++m) {
        double ym = use_d ? y_d[m] : (double)y_f[m];
        double yc_u = ((ym - y0) - dy_mean) * inv_yc_max;
        chi2_tmp += w_d[m] * yc_u * yc_u;
    }
    if (chi2_tmp == 0.0 || !isfinite(chi2_tmp)) return TLS_VK_ERR_DEGENERATE;

    const double yc_scale = inv_yc_max * sqrt(ws / chi2_tmp);
    double wy_sum_abs = 0.0, wy_sq_sum = 0.0, chi2_ref = 0.0;
    for (int m = 0; m < M; ++m) {
        double ym = use_d ? y_d[m] : (double)y_f[m];
        double yc = ((ym - y0) - dy_mean) * yc_scale;
        double wym = w_d[m] * yc;
        wy_sum_abs += fabs(wym);
        wy_sq_sum += wym * wym;
        chi2_ref += wym * yc;
        if (prec == VK_PREC_HALF)       ((uint16_t*)out->wy)[m] = vk_f32_to_f16_bits((float)wym);
        else if (prec == VK_PREC_FLOAT) ((float*)out->wy)[m] = (float)wym;
        else if (prec == VK_PREC_DF)    ((VkDf*)out->wy)[m] = vk_df_from_double(wym);
        else                            ((double*)out->wy)[m] = wym;
    }
    if (chi2_ref == 0.0 || wy_sq_sum == 0.0 || wy_sum_abs == 0.0) return TLS_VK_ERR_DEGENERATE;

    double inv_nfft = 1.0 / (double)nfft;
    double ws_fft = ws * inv_nfft;
    double chi2_ref_fft = chi2_ref * inv_nfft;
    out->standard_scale = 2.0 / (ws_fft * chi2_ref_fft);
    out->norm_neff = (wy_sum_abs * wy_sum_abs) / wy_sq_sum;
    double nll_scale_d = 0.0;
    if (out->norm_neff > 0.0) {
        double nfft_over_sum = (double)nfft / wy_sum_abs;
        nll_scale_d = out->norm_neff * nfft_over_sum * nfft_over_sum;
    }
    if (!isfinite(nll_scale_d) || nll_scale_d < 0.0) nll_scale_d = 0.0;
    out->nll_scale = nll_scale_d;
    return TLS_VK_OK;
}

#endif /* TLS_VK_PREPROCESS_H */
