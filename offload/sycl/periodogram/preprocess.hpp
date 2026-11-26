#pragma once

// Lock-free CPU host preprocessing for SYCL FastChi2 and IHS periodograms.
// Faithfully mirrors src/chi2per.c:
//   - t centering around t_mid = 0.5 * (t[0] + t[M - 1])
//   - overflow-safe inverse-variance weight scaling (dy_min / dy)^2
//   - weighted mean subtraction and unit weighted variance scaling
//   - chi2_ref, raw-weight effective sample size neff_w, Cybenko model, Pochhammer b
//   - FP16 double-precision preprocessing with w_target & row_scale = nfft / ws_d
//   - IHS weighted-residual effective sample size and standard / NLL scales
//
// All scratch vectors (including w_tmp) live inside the thread_local HostData
// structs so steady-state lightcurve preprocessing performs zero heap allocations.

#include "common/math.hpp"
#include "linalg/levinson.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace sycl_periodogram {

enum Status : int {
    TLS_SYCL_OK             =  0,
    TLS_SYCL_ERR_ARGUMENT   = -1,
    TLS_SYCL_ERR_BACKEND    = -2,
    TLS_SYCL_ERR_ALLOC      = -3,
    TLS_SYCL_ERR_DEGENERATE = -4,
    TLS_SYCL_ERR_SOLVER     = -5,
    TLS_SYCL_ERR_DEVICE     = -6,
    TLS_SYCL_ERR_RUNTIME    = -7
};

enum Backend : int {
    TLS_SYCL_BACKEND_PSWF43 = 1,
    TLS_SYCL_BACKEND_PSWF21 = 2
};

enum PrecKind : int {
    TLS_SYCL_PREC_HALF   = 0,
    TLS_SYCL_PREC_FLOAT  = 1,
    TLS_SYCL_PREC_DF     = 2,
    TLS_SYCL_PREC_DOUBLE = 3
};

inline sycl_linalg::CybenkoModel init_cybenko_model(int d, double neff_w) {
    sycl_linalg::CybenkoModel m{};
    const double d_dbl = static_cast<double>(d);
    const double M_dbl = neff_w;
    const double inv_M = 1.0 / M_dbl;
    const double inv_M2 = inv_M * inv_M;

    const double term1 = 2.0 * d_dbl;
    const double term2 = (8.0 * d_dbl * d_dbl - 14.0 * d_dbl) * inv_M;
    const double term3 = ((184.0 / 3.0) * d_dbl * d_dbl * d_dbl - 98.0 * d_dbl * d_dbl + (158.0 / 3.0) * d_dbl) * inv_M2;
    m.var_S = term1 + term2 + term3;

    m.mean_S = 2.0 * d_dbl;
    m.k_eff = (m.mean_S * m.mean_S) / m.var_S;
    m.theta_eff = m.var_S / m.mean_S;
    m.lgamma_k = std::lgamma(m.k_eff);
    m.inv_2d = 1.0 / (2.0 * d_dbl);

    const double n = 2.0 * d_dbl;
    m.sum_N_eff = n * M_dbl - n * (n + 1.0) + (n * (n + 1.0) * (2.0 * n + 1.0)) / (6.0 * M_dbl);
    return m;
}

template <typename XCoordT, typename ScalarT>
struct FastChi2HostData {
    std::vector<XCoordT> x;
    std::vector<ScalarT> w;
    std::vector<ScalarT> yw;
    std::vector<double>  w_tmp; // reusable scratch (zero per-call malloc)
    double ws_fft    = 0.0;
    double yws_fft   = 0.0;
    double chi2_ref  = 0.0;
    double neff_w    = 0.0;
    double b_param   = 0.0;
    double row_scale = 1.0; // Used only by FP16 fused-epilogue deconvolution
    sycl_linalg::CybenkoModel cyb{};
};

template <typename XCoordT, typename ScalarT>
struct IhsHostData {
    std::vector<XCoordT> x;
    std::vector<ScalarT> wy;
    std::vector<double>  w_tmp; // reusable scratch (zero per-call malloc)
    double standard_scale = 0.0;
    double nll_scale      = 0.0;
    double norm_neff      = 0.0;
};

// --- FP16 FastChi2 preprocessing (double precision, matching src/chi2per.c) ---
template <typename InFloatT>
inline int preprocess_fastchi2_half_typed(
    const double* t,
    const InFloatT* y,
    const InFloatT* dy,
    int M,
    int degree,
    int nfft,
    FastChi2HostData<sycl_fft::df_t, sycl_fft::f16_scalar_t>& out) {
    const size_t M_sz = static_cast<size_t>(M);
    out.x.resize(M_sz);
    out.w.resize(M_sz);
    out.yw.resize(M_sz);
    out.w_tmp.resize(M_sz);
    double* w_d = out.w_tmp.data();

    const double t_mid = 0.5 * (t[0] + t[M - 1]);
    double dy_min_d = static_cast<double>(dy[0]);
    for (int m = 0; m < M; ++m) {
        const double dm = static_cast<double>(dy[m]);
        if (!std::isfinite(dm) || dm <= 0.0) return TLS_SYCL_ERR_ARGUMENT;
        if (dm < dy_min_d) dy_min_d = dm;
        out.x[static_cast<size_t>(m)] = sycl_fft::df_from_double(t[m] - t_mid);
    }

    double ws_raw_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double inv_dy = dy_min_d / static_cast<double>(dy[m]);
        w_d[m] = inv_dy * inv_dy;
        ws_raw_d += w_d[m];
    }
    if (ws_raw_d == 0.0 || !std::isfinite(ws_raw_d)) return TLS_SYCL_ERR_DEGENERATE;

    double w_target = std::sqrt(static_cast<double>(nfft));
    const double sqrt_m = std::sqrt(static_cast<double>(M));
    if (sqrt_m > w_target) w_target = sqrt_m;
    if (w_target > 1024.0) w_target = 1024.0;
    const double w_scale_d = w_target / ws_raw_d;

    const double y0_d = static_cast<double>(y[0]);
    double ws_d = 0.0, sum_w2_d = 0.0, wy_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = static_cast<double>(y[m]);
        if (!std::isfinite(ym)) return TLS_SYCL_ERR_ARGUMENT;
        const double wm = w_d[m] * w_scale_d;
        w_d[m] = wm;
        ws_d += wm;
        sum_w2_d += wm * wm;
        wy_d += wm * (ym - y0_d);
    }
    if (ws_d == 0.0 || !std::isfinite(ws_d)) return TLS_SYCL_ERR_DEGENERATE;

    const double dy_mean_d = wy_d / ws_d;
    double yc_max_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double a = std::fabs((static_cast<double>(y[m]) - y0_d) - dy_mean_d);
        if (a > yc_max_d) yc_max_d = a;
    }
    if (yc_max_d == 0.0 || !std::isfinite(yc_max_d)) return TLS_SYCL_ERR_DEGENERATE;

    const double inv_yc_max_d = 1.0 / yc_max_d;
    double chi2_tmp_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double yc_u = ((static_cast<double>(y[m]) - y0_d) - dy_mean_d) * inv_yc_max_d;
        chi2_tmp_d += w_d[m] * yc_u * yc_u;
    }
    if (chi2_tmp_d == 0.0 || !std::isfinite(chi2_tmp_d)) return TLS_SYCL_ERR_DEGENERATE;

    const double yc_scale_d = inv_yc_max_d * std::sqrt(ws_d / chi2_tmp_d);
    double chi2_ref_d = 0.0, yws_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double yc = ((static_cast<double>(y[m]) - y0_d) - dy_mean_d) * yc_scale_d;
        const double wm = w_d[m];
        const double ywm = yc * wm;
        out.w[static_cast<size_t>(m)]  = static_cast<sycl_fft::f16_scalar_t>(static_cast<float>(wm));
        out.yw[static_cast<size_t>(m)] = static_cast<sycl_fft::f16_scalar_t>(static_cast<float>(ywm));
        yws_d += ywm;
        chi2_ref_d += yc * ywm;
    }
    if (chi2_ref_d == 0.0 || !std::isfinite(chi2_ref_d)) return TLS_SYCL_ERR_DEGENERATE;

    out.neff_w    = (ws_d * ws_d) / sum_w2_d;
    out.cyb       = init_cybenko_model(degree, out.neff_w);
    out.b_param   = 0.5 * (out.neff_w - 1.0 - 2.0 * static_cast<double>(degree));
    out.row_scale = static_cast<double>(nfft) / ws_d;
    out.ws_fft    = 1.0;
    out.yws_fft   = yws_d / ws_d;
    out.chi2_ref  = chi2_ref_d / ws_d;
    return TLS_SYCL_OK;
}

inline int preprocess_fastchi2_half(
    const double* t,
    const float* y_f,
    const double* y_d,
    const float* dy_f,
    const double* dy_d,
    int M,
    int degree,
    int nfft,
    FastChi2HostData<sycl_fft::df_t, sycl_fft::f16_scalar_t>& out) {
    if (y_d && dy_d) {
        return preprocess_fastchi2_half_typed<double>(t, y_d, dy_d, M, degree, nfft, out);
    }
    return preprocess_fastchi2_half_typed<float>(t, y_f, dy_f, M, degree, nfft, out);
}

// --- FP32 / DF / FP64 FastChi2 preprocessing (matching src/chi2per.c) ---
template <typename InFloatT, typename XCoordT, typename ScalarT>
inline int preprocess_fastchi2_generic(
    const double* t,
    const InFloatT* y,
    const InFloatT* dy,
    int M,
    int degree,
    int nfft,
    FastChi2HostData<XCoordT, ScalarT>& out) {
    const size_t M_sz = static_cast<size_t>(M);
    out.x.resize(M_sz);
    out.w.resize(M_sz);
    out.yw.resize(M_sz);
    out.w_tmp.resize(M_sz);
    double* w_d = out.w_tmp.data();

    const double t_mid = 0.5 * (t[0] + t[M - 1]);
    double dy_min = static_cast<double>(dy[0]);
    for (int m = 0; m < M; ++m) {
        const double dm = static_cast<double>(dy[m]);
        if (!std::isfinite(dm) || dm <= 0.0) return TLS_SYCL_ERR_ARGUMENT;
        if (dm < dy_min) dy_min = dm;
        const double xm = t[m] - t_mid;
        if constexpr (std::is_same_v<XCoordT, sycl_fft::df_t>) {
            out.x[static_cast<size_t>(m)] = sycl_fft::df_from_double(xm);
        } else {
            out.x[static_cast<size_t>(m)] = xm;
        }
    }

    double ws_raw = 0.0;
    for (int m = 0; m < M; ++m) {
        const double inv_dy = dy_min / static_cast<double>(dy[m]);
        w_d[m] = inv_dy * inv_dy;
        ws_raw += w_d[m];
    }
    if (ws_raw == 0.0 || !std::isfinite(ws_raw)) return TLS_SYCL_ERR_DEGENERATE;

    const double w_scale = std::sqrt(static_cast<double>(M)) / ws_raw;
    const double y0 = static_cast<double>(y[0]);
    double ws = 0.0, sum_w2 = 0.0, wy = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = static_cast<double>(y[m]);
        if (!std::isfinite(ym)) return TLS_SYCL_ERR_ARGUMENT;
        const double wm = w_d[m] * w_scale;
        w_d[m] = wm;
        ws += wm;
        sum_w2 += wm * wm;
        wy += wm * (ym - y0);
    }
    if (ws == 0.0 || !std::isfinite(ws)) return TLS_SYCL_ERR_DEGENERATE;

    const double dy_mean = wy / ws;
    double yc_max = 0.0;
    for (int m = 0; m < M; ++m) {
        const double a = std::fabs((static_cast<double>(y[m]) - y0) - dy_mean);
        if (a > yc_max) yc_max = a;
    }
    if (yc_max == 0.0 || !std::isfinite(yc_max)) return TLS_SYCL_ERR_DEGENERATE;

    const double inv_yc_max = 1.0 / yc_max;
    double chi2_tmp = 0.0;
    for (int m = 0; m < M; ++m) {
        const double yc_u = ((static_cast<double>(y[m]) - y0) - dy_mean) * inv_yc_max;
        chi2_tmp += w_d[m] * yc_u * yc_u;
    }
    if (chi2_tmp == 0.0 || !std::isfinite(chi2_tmp)) return TLS_SYCL_ERR_DEGENERATE;

    const double yc_scale = inv_yc_max * std::sqrt(ws / chi2_tmp);
    double chi2_ref = 0.0, yws = 0.0;
    for (int m = 0; m < M; ++m) {
        const double yc = ((static_cast<double>(y[m]) - y0) - dy_mean) * yc_scale;
        const double wm = w_d[m];
        const double ywm = yc * wm;
        yws += ywm;
        chi2_ref += yc * ywm;
        if constexpr (std::is_same_v<ScalarT, sycl_fft::df_t>) {
            out.w[static_cast<size_t>(m)]  = sycl_fft::df_from_double(wm);
            out.yw[static_cast<size_t>(m)] = sycl_fft::df_from_double(ywm);
        } else {
            out.w[static_cast<size_t>(m)]  = static_cast<ScalarT>(wm);
            out.yw[static_cast<size_t>(m)] = static_cast<ScalarT>(ywm);
        }
    }
    if (chi2_ref == 0.0 || !std::isfinite(chi2_ref)) return TLS_SYCL_ERR_DEGENERATE;

    const double inv_nfft = 1.0 / static_cast<double>(nfft);
    out.neff_w    = (ws * ws) / sum_w2;
    out.cyb       = init_cybenko_model(degree, out.neff_w);
    out.b_param   = 0.5 * (out.neff_w - 1.0 - 2.0 * static_cast<double>(degree));
    out.row_scale = 1.0;
    out.ws_fft    = ws * inv_nfft;
    out.yws_fft   = yws * inv_nfft;
    out.chi2_ref  = chi2_ref * inv_nfft;
    return TLS_SYCL_OK;
}

// --- IHS preprocessing (all 4 precisions, matching src/chi2per.c) ---
template <typename InFloatT, typename XCoordT, typename ScalarT, bool IsHalf>
inline int preprocess_ihs_typed(
    const double* t,
    const InFloatT* y,
    const InFloatT* dy,
    int M,
    int nfft,
    IhsHostData<XCoordT, ScalarT>& out) {
    const size_t M_sz = static_cast<size_t>(M);
    out.x.resize(M_sz);
    out.wy.resize(M_sz);
    out.w_tmp.resize(M_sz);
    double* w_d = out.w_tmp.data();

    const double t_mid = 0.5 * (t[0] + t[M - 1]);
    double dy_min = static_cast<double>(dy[0]);
    for (int m = 0; m < M; ++m) {
        const double dm = static_cast<double>(dy[m]);
        if (!std::isfinite(dm) || dm <= 0.0) return TLS_SYCL_ERR_ARGUMENT;
        if (dm < dy_min) dy_min = dm;
        const double xm = t[m] - t_mid;
        if constexpr (std::is_same_v<XCoordT, sycl_fft::df_t>) {
            out.x[static_cast<size_t>(m)] = sycl_fft::df_from_double(xm);
        } else {
            out.x[static_cast<size_t>(m)] = xm;
        }
    }

    double ws_raw = 0.0;
    for (int m = 0; m < M; ++m) {
        const double inv_dy = dy_min / static_cast<double>(dy[m]);
        w_d[m] = inv_dy * inv_dy;
        ws_raw += w_d[m];
    }
    if (ws_raw == 0.0 || !std::isfinite(ws_raw)) return TLS_SYCL_ERR_DEGENERATE;

    double w_target = std::sqrt(static_cast<double>(M));
    if constexpr (IsHalf) {
        const double sqrt_nfft = std::sqrt(static_cast<double>(nfft));
        if (sqrt_nfft > w_target) w_target = sqrt_nfft;
        if (w_target > 1024.0) w_target = 1024.0;
    }
    const double w_scale = w_target / ws_raw;

    const double y0 = static_cast<double>(y[0]);
    double ws = 0.0, wy_mean = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = static_cast<double>(y[m]);
        if (!std::isfinite(ym)) return TLS_SYCL_ERR_ARGUMENT;
        const double wm = w_d[m] * w_scale;
        w_d[m] = wm;
        ws += wm;
        wy_mean += wm * (ym - y0);
    }
    if (ws == 0.0 || !std::isfinite(ws)) return TLS_SYCL_ERR_DEGENERATE;

    const double dy_mean = wy_mean / ws;
    double yc_max = 0.0;
    for (int m = 0; m < M; ++m) {
        const double a = std::fabs((static_cast<double>(y[m]) - y0) - dy_mean);
        if (a > yc_max) yc_max = a;
    }
    if (yc_max == 0.0 || !std::isfinite(yc_max)) return TLS_SYCL_ERR_DEGENERATE;

    const double inv_yc_max = 1.0 / yc_max;
    double chi2_tmp = 0.0;
    for (int m = 0; m < M; ++m) {
        const double yc_u = ((static_cast<double>(y[m]) - y0) - dy_mean) * inv_yc_max;
        chi2_tmp += w_d[m] * yc_u * yc_u;
    }
    if (chi2_tmp == 0.0 || !std::isfinite(chi2_tmp)) return TLS_SYCL_ERR_DEGENERATE;

    const double yc_scale = inv_yc_max * std::sqrt(ws / chi2_tmp);
    double wy_sum_abs = 0.0, wy_sq_sum = 0.0, chi2_ref = 0.0;
    for (int m = 0; m < M; ++m) {
        const double yc = ((static_cast<double>(y[m]) - y0) - dy_mean) * yc_scale;
        const double wym = w_d[m] * yc;
        wy_sum_abs += std::fabs(wym);
        wy_sq_sum += wym * wym;
        chi2_ref += wym * yc;
        if constexpr (IsHalf) {
            out.wy[static_cast<size_t>(m)] = static_cast<sycl_fft::f16_scalar_t>(static_cast<float>(wym));
        } else if constexpr (std::is_same_v<ScalarT, sycl_fft::df_t>) {
            out.wy[static_cast<size_t>(m)] = sycl_fft::df_from_double(wym);
        } else {
            out.wy[static_cast<size_t>(m)] = static_cast<ScalarT>(wym);
        }
    }
    if (chi2_ref == 0.0 || wy_sq_sum == 0.0 || wy_sum_abs == 0.0) return TLS_SYCL_ERR_DEGENERATE;

    const double inv_nfft = 1.0 / static_cast<double>(nfft);
    const double ws_fft = ws * inv_nfft;
    const double chi2_ref_fft = chi2_ref * inv_nfft;
    out.standard_scale = 2.0 / (ws_fft * chi2_ref_fft);
    out.norm_neff = (wy_sum_abs * wy_sum_abs) / wy_sq_sum;
    double nll_scale_d = 0.0;
    if (out.norm_neff > 0.0) {
        const double nfft_over_sum = static_cast<double>(nfft) / wy_sum_abs;
        nll_scale_d = out.norm_neff * nfft_over_sum * nfft_over_sum;
    }
    if (!std::isfinite(nll_scale_d) || nll_scale_d < 0.0) nll_scale_d = 0.0;
    out.nll_scale = nll_scale_d;
    return TLS_SYCL_OK;
}

template <typename XCoordT, typename ScalarT, bool IsHalf>
inline int preprocess_ihs_impl(
    const double* t,
    const float* y_f,
    const double* y_d,
    const float* dy_f,
    const double* dy_d,
    int M,
    int nfft,
    IhsHostData<XCoordT, ScalarT>& out) {
    if (y_d && dy_d) {
        return preprocess_ihs_typed<double, XCoordT, ScalarT, IsHalf>(t, y_d, dy_d, M, nfft, out);
    }
    return preprocess_ihs_typed<float, XCoordT, ScalarT, IsHalf>(t, y_f, dy_f, M, nfft, out);
}

} // namespace sycl_periodogram
