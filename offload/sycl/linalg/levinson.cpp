// Decoupled SYCL Fused GLS (d=1), Levinson-Durbin (d>1), and IHS Solver Module.
// Compiled 4x via -D flags (isolated HCF per precision):
//   -DSYCL_FFT_HALF | -DSYCL_FFT_FLOAT | -DSYCL_FFT_DOUBLE | -DSYCL_FFT_DF
//
// Zero double-precision instructions in FP16 / FP32 / DF HCF modules so they run
// cleanly on GPUs without FP64 hardware.

#if !defined(SYCL_FFT_HALF) && !defined(SYCL_FFT_FLOAT) && !defined(SYCL_FFT_DOUBLE) && !defined(SYCL_FFT_DF)
#    define SYCL_FFT_DF 1
#endif

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/precision.hpp"
#include "linalg/levinson.hpp"

namespace sycl_linalg {
namespace detail {

using namespace sycl_fft;
using namespace sycl_nufft;

#if defined(SYCL_FFT_HALF)
using SolveComplexT = sycl_fft::Complexh;
using SolveScalarT = sycl_fft::f16_scalar_t;
using SolveRealT = sycl_fft::f16x2_t;
using OutScalarT = float;
using IhsAccT = float;
using IhsDeconvT = float;
constexpr float kSingularityThreshF = 500.0f;
#elif defined(SYCL_FFT_DOUBLE)
using SolveComplexT = sycl_fft::Complex;
using SolveScalarT = double;
using SolveRealT = double;
using OutScalarT = double;
using IhsAccT = double;
using IhsDeconvT = double;
constexpr double kSingularityThreshD = 1e13;
#elif defined(SYCL_FFT_DF)
using SolveComplexT = sycl_fft::Complexdf;
using SolveScalarT = sycl_fft::df_t;
using SolveRealT = sycl_fft::df_t;
using OutScalarT = sycl_fft::df_t;
using IhsAccT = sycl_fft::df_t;
using IhsDeconvT = sycl_fft::df_t;
constexpr float kSingularityThreshF = 1e13f;
#else
using SolveComplexT = sycl_fft::Complexf;
using SolveScalarT = float;
using SolveRealT = float;
using OutScalarT = float;
using IhsAccT = float;
using IhsDeconvT = float;
constexpr float kSingularityThreshF = 5e5f;
#endif

// ============================================================================
// Precision-Specific Arithmetic Helpers for GLS & Levinson
// ============================================================================
SYCL_FFT_FORCE_INLINE SolveRealT l_div(SolveRealT a, SolveRealT b) {
#if defined(SYCL_FFT_HALF)
    return f16x2_div(a, b);
#elif defined(SYCL_FFT_DF)
    return df_div(a, b);
#else
    return a / b;
#endif
}

SYCL_FFT_FORCE_INLINE SolveRealT l_sqrt(SolveRealT a) {
#if defined(SYCL_FFT_HALF)
    return f16x2_sqrt(a);
#elif defined(SYCL_FFT_DF)
    return df_sqrt(a);
#else
    return sycl::sqrt(a);
#endif
}

SYCL_FFT_FORCE_INLINE SolveRealT l_fma(SolveRealT a, SolveRealT b, SolveRealT c) {
#if defined(SYCL_FFT_HALF)
    return f16x2_fma(a, b, c);
#elif defined(SYCL_FFT_DF)
    return df_add(df_mul(a, b), c);
#else
    return sycl::fma(a, b, c);
#endif
}

SYCL_FFT_FORCE_INLINE SolveRealT l_abs2(SolveRealT r, SolveRealT i) {
#if defined(SYCL_FFT_DF)
    return df_add(df_sqr(r), df_sqr(i));
#else
    return l_fma(r, r, SYCL_FFT_MUL(i, i));
#endif
}

SYCL_FFT_FORCE_INLINE void l_cmac(SolveRealT ar, SolveRealT ai, SolveRealT br, SolveRealT bi, SolveRealT& acc_r, SolveRealT& acc_i) {
#if defined(SYCL_FFT_DF)
    df_t pr, pi;
    df_cmul_fused(ar, ai, br, bi, pr, pi);
    acc_r = df_add(acc_r, pr);
    acc_i = df_add(acc_i, pi);
#else
    acc_r = l_fma(ar, br, l_fma(SYCL_FFT_NEG(ai), bi, acc_r));
    acc_i = l_fma(ar, bi, l_fma(ai, br, acc_i));
#endif
}

SYCL_FFT_FORCE_INLINE void l_cmac_conj(SolveRealT gr, SolveRealT gi, SolveRealT ar, SolveRealT ai, SolveRealT base_r, SolveRealT base_i, SolveRealT& nr,
                                       SolveRealT& ni) {
#if defined(SYCL_FFT_DF)
    df_t pr, pi;
    df_cmul_fused(gr, gi, ar, df_neg(ai), pr, pi);
    nr = df_add(base_r, pr);
    ni = df_add(base_i, pi);
#else
    nr = l_fma(gr, ar, l_fma(gi, ai, base_r));
    ni = l_fma(gi, ar, l_fma(SYCL_FFT_NEG(gr), ai, base_i));
#endif
}

SYCL_FFT_FORCE_INLINE size_t trig_w_row(int q, int degree) { return (q <= degree) ? static_cast<size_t>(2 * (q - 1)) : static_cast<size_t>(degree + q - 1); }

SYCL_FFT_FORCE_INLINE size_t trig_yw_row(int q) { return static_cast<size_t>(2 * (q - 1) + 1); }

// ============================================================================
// Normalization, Pochhammer NLL, Cybenko NLL, Bayes-Laplace & IHS Helpers
// ============================================================================
#if defined(SYCL_FFT_DOUBLE)

SYCL_FFT_FORCE_INLINE double d_nan() { return SYCL_FFT_BIT_CAST(double, UINT64_C(0x7ff8000000000000)); }

SYCL_FFT_FORCE_INLINE double d_inf() { return SYCL_FFT_BIT_CAST(double, UINT64_C(0x7ff0000000000000)); }

SYCL_FFT_FORCE_INLINE bool d_is_finite(double v) {
    const uint64_t u = SYCL_FFT_BIT_CAST(uint64_t, v);
    return (u & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

SYCL_FFT_FORCE_INLINE bool cond_is_singular_d(double cb) { return !d_is_finite(cb) || cb < 0.0 || cb > kSingularityThreshD; }

SYCL_FFT_FORCE_INLINE double cybenko_kappa_to_S_d(double kappa, const CybenkoModel& m) {
    if (kappa <= 1.0 || !d_is_finite(kappa)) return 0.0;
    if (kappa > 5000.0) kappa = 5000.0;
    const double pow_k = sycl::pow(kappa, m.inv_2d);
    const double k_val = (pow_k - 1.0) / (pow_k + 1.0);
    return m.sum_N_eff * (k_val * k_val);
}

SYCL_FFT_FORCE_INLINE double nll_exact_pochhammer_d(int d, double r2, double b) {
    if (!d_is_finite(r2) || r2 <= 0.0) return 0.0;
    if (r2 > 0.9999999999999) r2 = 0.9999999999999;
    const double term0 = -b * sycl::log1p(-r2);
    if (d <= 1) return (term0 > 0.0 && d_is_finite(term0)) ? term0 : 0.0;
    double c = 1.0;
    double s_minus_1 = 0.0;
    for (int k = 1; k < d; ++k) {
        c *= ((b + static_cast<double>(k - 1)) / static_cast<double>(k)) * r2;
        s_minus_1 += c;
    }
    const double nll = term0 - sycl::log1p(s_minus_1);
    return (nll > 0.0 && d_is_finite(nll)) ? nll : 0.0;
}

SYCL_FFT_FORCE_INLINE double nll_cybenko_d(double kappa, const CybenkoModel& m, int d) {
    const double s = cybenko_kappa_to_S_d(kappa, m);
    if (s <= 0.0 || !d_is_finite(s)) return 0.0;
    const double k_eff = m.k_eff;
    const double th_eff = m.theta_eff;
    const double lg_k = m.lgamma_k;
    if (d <= 2) {
        const double z = s / th_eff;
        const int order = 2 * d;
        const double delta = k_eff - static_cast<double>(order);
        double c = 1.0, poly = 1.0, d_poly = 0.0;
        for (int k = 1; k < order; ++k) {
            c *= (z / static_cast<double>(k));
            poly += c;
            if (sycl::fabs(delta) > 1.0e-5) {
                double H_k = 0.0;
                for (int i = 1; i <= k; ++i) H_k += 1.0 / static_cast<double>(i);
                d_poly -= c * H_k;
            }
        }
        double nll = z - sycl::log(poly);
        if (sycl::fabs(delta) > 1.0e-5) {
            nll -= delta * (sycl::log(z) + d_poly / poly);
        }
        return (nll > 0.0 && d_is_finite(nll)) ? nll : 0.0;
    } else {
        const double a = k_eff;
        const double z = s / th_eff;
        if (z < a + 3.0) {
            double c = 1.0, sum = 1.0;
            for (int k = 1; k < 60; ++k) {
                c *= z / (a + static_cast<double>(k - 1));
                sum += c;
                if (c < 1.0e-14 * sum) break;
            }
            const double lg_inc = a * sycl::log(z) - z - sycl::log(a) + sycl::log(sum);
            const double nll = lg_k - lg_inc;
            return (nll > 0.0 && d_is_finite(nll)) ? nll : 0.0;
        } else {
            const double inv_z = 1.0 / z;
            const double t1 = (a - 1.0) * inv_z;
            const double t2 = t1 * ((a - 2.0) * inv_z);
            const double t3 = t2 * ((a - 3.0) * inv_z);
            const double t4 = t3 * ((a - 4.0) * inv_z);
            const double poly = 1.0 + t1 + t2 + t3 + t4;
            const double nll = z - (a - 1.0) * sycl::log(z) + lg_k - sycl::log(poly);
            return (nll > 0.0 && d_is_finite(nll)) ? nll : 0.0;
        }
    }
}

SYCL_FFT_FORCE_INLINE double compute_bayes_laplace_d(double delta) {
    if (delta >= 0.0) {
        return delta + 0.693147180559945309417;
    }
    const double inner = 1.0 - 0.5 * sycl::exp(delta);
    if (inner <= 0.0) return 0.0;
    return -sycl::log(inner);
}

SYCL_FFT_FORCE_INLINE double normalize_power_d(double dot, double inv_chi2_ref, double cond_bound, NormKind norm, int degree, int M, const CybenkoModel& cyb,
                                               double b) {
    if (!d_is_finite(dot)) return d_nan();
    if (dot <= 0.0) return 0.0;
    const double r2 = dot * inv_chi2_ref;
    if (!d_is_finite(r2) || r2 >= 1.0) return d_nan();
    if (norm == NormKind::Standard) {
        return r2;
    } else if (norm == NormKind::Anova) {
        const double d1 = 2.0 * static_cast<double>(degree) + 1.0;
        const double d2 = static_cast<double>(M) - 1.0 - d1;
        double residual = 1.0 - r2;
        if (residual < 1e-32) residual = 1e-32;
        return (d2 * r2) / (d1 * residual);
    } else if (norm == NormKind::Nll) {
        return nll_exact_pochhammer_d(degree, r2, b);
    } else {
        const double nll_r2 = nll_exact_pochhammer_d(degree, r2, b);
        const double kappa = cond_bound / static_cast<double>(2 * degree + 1);
        const double nll_cond = nll_cybenko_d(kappa, cyb, degree);
        return compute_bayes_laplace_d(nll_r2 - nll_cond);
    }
}

SYCL_FFT_FORCE_INLINE double ihs_wilkie_d(double z, double inv_n, double inv_n2_288) {
    if (!(inv_n > 0.0)) return z;
    if (!(z > 0.0) || !d_is_finite(z)) return 0.0;
    double p2 = sycl::fma(-9.0, z, 76.0);
    p2 = sycl::fma(p2, z, -132.0);
    p2 = sycl::fma(p2, z, 24.0);
    const double delta = sycl::fma(z * (2.0 - z), 0.25 * inv_n, -(z * p2) * inv_n2_288);
    if (!(delta > -1.0) || !d_is_finite(delta)) return z;
    const double corr = z - sycl::log1p(delta);
    return (corr > 0.0 && d_is_finite(corr)) ? corr : z;
}

SYCL_FFT_FORCE_INLINE double ihs_erlang_d(double s, int degree) {
    if (!(s > 0.0) || !d_is_finite(s)) return 0.0;
    double log_sum = 1.0, term = 1.0;
    for (int i = 1; i < degree; ++i) {
        term *= s / static_cast<double>(i);
        log_sum += term;
    }
    if (!(log_sum > 0.0) || !d_is_finite(log_sum)) return 0.0;
    const double nll = s - sycl::log(log_sum);
    return (nll > 0.0 && d_is_finite(nll)) ? nll : 0.0;
}

#else  // FP16, FP32, and DF (ZERO double ops in device code!)

SYCL_FFT_FORCE_INLINE float f_nan() { return SYCL_FFT_BIT_CAST(float, UINT32_C(0x7fc00000)); }

SYCL_FFT_FORCE_INLINE float f_inf() { return SYCL_FFT_BIT_CAST(float, UINT32_C(0x7f800000)); }

SYCL_FFT_FORCE_INLINE bool f_is_finite(float v) {
    const uint32_t u = SYCL_FFT_BIT_CAST(uint32_t, v);
    return (u & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

SYCL_FFT_FORCE_INLINE bool cond_is_singular_f(float cb) { return !f_is_finite(cb) || cb < 0.0f || cb > kSingularityThreshF; }

SYCL_FFT_FORCE_INLINE float cybenko_kappa_to_S_f(float kappa, const CybenkoModelF& m) {
    if (kappa <= 1.0f || !f_is_finite(kappa)) return 0.0f;
    if (kappa > 5000.0f) kappa = 5000.0f;
    const float pow_k = sycl::pow(kappa, m.inv_2d);
    const float k_val = (pow_k - 1.0f) / (pow_k + 1.0f);
    return m.sum_N_eff * (k_val * k_val);
}

SYCL_FFT_FORCE_INLINE float nll_cybenko_f(float kappa, const CybenkoModelF& m, int d) {
    const float s = cybenko_kappa_to_S_f(kappa, m);
    if (s <= 0.0f || !f_is_finite(s)) return 0.0f;
    const float k_eff = m.k_eff;
    const float th_eff = m.theta_eff;
    const float lg_k = m.lgamma_k;
    if (d <= 2) {
        const float z = s / th_eff;
        const int order = 2 * d;
        const float delta = k_eff - static_cast<float>(order);
        float c = 1.0f, poly = 1.0f, d_poly = 0.0f;
        for (int k = 1; k < order; ++k) {
            c *= (z / static_cast<float>(k));
            poly += c;
            if (sycl::fabs(delta) > 1.0e-5f) {
                float H_k = 0.0f;
                for (int i = 1; i <= k; ++i) H_k += 1.0f / static_cast<float>(i);
                d_poly -= c * H_k;
            }
        }
        float nll = z - sycl::log(poly);
        if (sycl::fabs(delta) > 1.0e-5f) {
            nll -= delta * (sycl::log(z) + d_poly / poly);
        }
        return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
    } else {
        const float a = k_eff;
        const float z = s / th_eff;
        if (z < a + 3.0f) {
            float c = 1.0f, sum = 1.0f;
            for (int k = 1; k < 60; ++k) {
                c *= z / (a + static_cast<float>(k - 1));
                sum += c;
                if (c < 1.0e-7f * sum) break;
            }
            const float lg_inc = a * sycl::log(z) - z - sycl::log(a) + sycl::log(sum);
            const float nll = lg_k - lg_inc;
            return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
        } else {
            const float inv_z = 1.0f / z;
            const float t1 = (a - 1.0f) * inv_z;
            const float t2 = t1 * ((a - 2.0f) * inv_z);
            const float t3 = t2 * ((a - 3.0f) * inv_z);
            const float t4 = t3 * ((a - 4.0f) * inv_z);
            const float poly = 1.0f + t1 + t2 + t3 + t4;
            const float nll = z - (a - 1.0f) * sycl::log(z) + lg_k - sycl::log(poly);
            return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
        }
    }
}

SYCL_FFT_FORCE_INLINE float compute_bayes_laplace_f(float delta) {
    if (delta >= 0.0f) {
        return delta + 0.6931471805599453f;
    }
    const float inner = 1.0f - 0.5f * sycl::exp(delta);
    if (inner <= 0.0f) return 0.0f;
    return -sycl::log(inner);
}

#    if defined(SYCL_FFT_DF)
struct CybenkoModelDf {
    df_t sum_N_eff;
    df_t inv_2d;
    df_t k_eff;
    df_t theta_eff;
    df_t lgamma_k;
};

inline CybenkoModelDf to_cybenko_df(const CybenkoModel& m) {
    return CybenkoModelDf{
        df_from_double(m.sum_N_eff), df_from_double(m.inv_2d), df_from_double(m.k_eff), df_from_double(m.theta_eff), df_from_double(m.lgamma_k),
    };
}

// ~46-47 bit DF natural logarithm using range reduction + atanh series (zero FP64 ops!)
SYCL_FFT_DF_INLINE df_t df_log(df_t x) {
    SYCL_FFT_DF_EXACT_FP
    if (x.hi <= 0.0f) return df_make(0.0f, 0.0f);
    // Extract binary exponent e so m = x * 2^(-e) lies in [sqrt(1/2), sqrt(2)]
    const uint32_t u = SYCL_FFT_BIT_CAST(uint32_t, x.hi);
    int e = static_cast<int>((u >> 23) & 0xFFu) - 127;
    // Scale by 2^(-e) exactly (exponent bit manipulation)
    const uint32_t scale_u = static_cast<uint32_t>(127 - e) << 23;
    df_t m = df_scale_pow2(x, SYCL_FFT_BIT_CAST(float, scale_u));
    if (m.hi > 1.41421356f) {
        m = df_scale_pow2(m, 0.5f);
        ++e;
    }
    // ln(m) = 2 * atanh((m - 1) / (m + 1)), |t| <= (sqrt(2)-1)/(sqrt(2)+1) = 0.17157
    const df_t t = df_div(df_sub(m, df_make(1.0f, 0.0f)), df_add(m, df_make(1.0f, 0.0f)));
    const df_t t2 = df_sqr(t);
    // Odd series up to t^15: t^17/17 <= 5e-15 (~47.5 bits)
    df_t poly = df_make(1.0f / 15.0f, 0.0f);
    poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 13.0));
    poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 11.0));
    poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 9.0));
    poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 7.0));
    poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 5.0));
    poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 3.0));
    poly = df_add(df_mul(poly, t2), df_make(1.0f, 0.0f));
    const df_t ln_m = df_scale_pow2(df_mul(t, poly), 2.0f);
    constexpr df_t kLn2Df = df_from_double(0.693147180559945309417);
    return df_add(ln_m, df_mul_f(kLn2Df, static_cast<float>(e)));
}

SYCL_FFT_DF_INLINE df_t df_log1p(df_t x) {
    SYCL_FFT_DF_EXACT_FP
    if (sycl::fabs(x.hi) < 0.15f) {
        const df_t t = df_div(x, df_add(df_make(2.0f, 0.0f), x));
        const df_t t2 = df_sqr(t);
        df_t poly = df_from_double(1.0 / 11.0);
        poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 9.0));
        poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 7.0));
        poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 5.0));
        poly = df_add(df_mul(poly, t2), df_from_double(1.0 / 3.0));
        poly = df_add(df_mul(poly, t2), df_make(1.0f, 0.0f));
        return df_scale_pow2(df_mul(t, poly), 2.0f);
    }
    return df_log(df_add(df_make(1.0f, 0.0f), x));
}

// ~46-47 bit DF exponential exp(x) via range reduction x = k*ln(2) + r, |r| <= ln(2)/2 (zero FP64 ops!)
SYCL_FFT_DF_INLINE df_t df_exp(df_t x) {
    SYCL_FFT_DF_EXACT_FP
    const float xf = x.hi + x.lo;
    if (xf < -87.0f) return df_make(0.0f, 0.0f);
    if (xf > 88.0f) return df_make(3.4028234e38f, 0.0f);
    constexpr df_t kLn2Df = df_from_double(0.693147180559945309417232121458);
    constexpr float kInvLn2 = 1.4426950408889634f;
    const float kf = sycl::rint(xf * kInvLn2);
    const int k = static_cast<int>(kf);
    const df_t r = df_sub(x, df_mul_f(kLn2Df, kf));
    // Taylor series of exp(r) up to r^11 / 11!: (0.3466)^12 / 12! < 6e-15
    df_t p = df_from_double(1.0 / 39916800.0);
    p = df_add(df_mul(p, r), df_from_double(1.0 / 3628800.0));
    p = df_add(df_mul(p, r), df_from_double(1.0 / 362880.0));
    p = df_add(df_mul(p, r), df_from_double(1.0 / 40320.0));
    p = df_add(df_mul(p, r), df_from_double(1.0 / 5040.0));
    p = df_add(df_mul(p, r), df_from_double(1.0 / 720.0));
    p = df_add(df_mul(p, r), df_from_double(1.0 / 120.0));
    p = df_add(df_mul(p, r), df_from_double(1.0 / 24.0));
    p = df_add(df_mul(p, r), df_from_double(1.0 / 6.0));
    p = df_add(df_mul(p, r), df_make(0.5f, 0.0f));
    p = df_add(df_mul(p, r), df_make(1.0f, 0.0f));
    p = df_add(df_mul(p, r), df_make(1.0f, 0.0f));
    const uint32_t scale_u = static_cast<uint32_t>(127 + k) << 23;
    return df_scale_pow2(p, SYCL_FFT_BIT_CAST(float, scale_u));
}

SYCL_FFT_DF_INLINE df_t cybenko_kappa_to_S_df(df_t kappa, const CybenkoModelDf& m) {
    if (!f_is_finite(kappa.hi) || kappa.hi < 1.0f || (kappa.hi == 1.0f && kappa.lo <= 0.0f)) {
        return df_make(0.0f, 0.0f);
    }
    if (kappa.hi > 5000.0f) kappa = df_make(5000.0f, 0.0f);
    const df_t pow_k = df_exp(df_mul(df_log(kappa), m.inv_2d));
    const df_t one = df_make(1.0f, 0.0f);
    const df_t k_val = df_div(df_sub(pow_k, one), df_add(pow_k, one));
    return df_mul(m.sum_N_eff, df_sqr(k_val));
}

SYCL_FFT_DF_INLINE df_t nll_cybenko_df(df_t kappa, const CybenkoModelDf& m, int d) {
    const df_t s = cybenko_kappa_to_S_df(kappa, m);
    if (!f_is_finite(s.hi) || s.hi <= 0.0f) return df_make(0.0f, 0.0f);
    const df_t k_eff = m.k_eff;
    const df_t th_eff = m.theta_eff;
    const df_t lg_k = m.lgamma_k;
    const df_t z = df_div(s, th_eff);
    const df_t one = df_make(1.0f, 0.0f);
    if (d <= 2) {
        const int order = 2 * d;
        const df_t delta = df_sub(k_eff, df_make(static_cast<float>(order), 0.0f));
        const bool need_delta = (sycl::fabs(delta.hi) > 1.0e-5f);
        df_t c = one, poly = one, d_poly = df_make(0.0f, 0.0f);
        for (int k = 1; k < order; ++k) {
            c = df_mul(c, df_div(z, df_make(static_cast<float>(k), 0.0f)));
            poly = df_add(poly, c);
            if (need_delta) {
                df_t H_k = df_make(0.0f, 0.0f);
                for (int i = 1; i <= k; ++i) {
                    H_k = df_add(H_k, df_div(one, df_make(static_cast<float>(i), 0.0f)));
                }
                d_poly = df_sub(d_poly, df_mul(c, H_k));
            }
        }
        df_t nll = df_sub(z, df_log(poly));
        if (need_delta) {
            nll = df_sub(nll, df_mul(delta, df_add(df_log(z), df_div(d_poly, poly))));
        }
        return (nll.hi > 0.0f && f_is_finite(nll.hi)) ? nll : df_make(0.0f, 0.0f);
    } else {
        const df_t a = k_eff;
        if (z.hi < a.hi + 3.0f) {
            df_t c = one, sum = one;
            for (int k = 1; k < 60; ++k) {
                c = df_mul(c, df_div(z, df_add(a, df_make(static_cast<float>(k - 1), 0.0f))));
                sum = df_add(sum, c);
                if (c.hi < 1.0e-14f * sum.hi) break;
            }
            const df_t lg_inc = df_add(df_sub(df_sub(df_mul(a, df_log(z)), z), df_log(a)), df_log(sum));
            const df_t nll = df_sub(lg_k, lg_inc);
            return (nll.hi > 0.0f && f_is_finite(nll.hi)) ? nll : df_make(0.0f, 0.0f);
        } else {
            const df_t inv_z = df_div(one, z);
            const df_t t1 = df_mul(df_sub(a, one), inv_z);
            const df_t t2 = df_mul(t1, df_mul(df_sub(a, df_make(2.0f, 0.0f)), inv_z));
            const df_t t3 = df_mul(t2, df_mul(df_sub(a, df_make(3.0f, 0.0f)), inv_z));
            const df_t t4 = df_mul(t3, df_mul(df_sub(a, df_make(4.0f, 0.0f)), inv_z));
            const df_t poly = df_add(df_add(one, t1), df_add(df_add(t2, t3), t4));
            const df_t nll = df_sub(df_add(df_sub(z, df_mul(df_sub(a, one), df_log(z))), lg_k), df_log(poly));
            return (nll.hi > 0.0f && f_is_finite(nll.hi)) ? nll : df_make(0.0f, 0.0f);
        }
    }
}

SYCL_FFT_DF_INLINE df_t compute_bayes_laplace_df(df_t delta) {
    constexpr df_t kLn2Df = df_from_double(0.693147180559945309417232121458);
    if (delta.hi > 0.0f || (delta.hi == 0.0f && delta.lo >= 0.0f)) {
        return df_add(delta, kLn2Df);
    }
    const df_t inner = df_sub(df_make(1.0f, 0.0f), df_scale_pow2(df_exp(delta), 0.5f));
    if (inner.hi <= 0.0f) return df_make(0.0f, 0.0f);
    return df_neg(df_log(inner));
}

SYCL_FFT_DF_INLINE df_t nll_exact_pochhammer_df(int d, df_t r2, df_t b) {
    if (!f_is_finite(r2.hi) || r2.hi <= 0.0f) return df_make(0.0f, 0.0f);
    constexpr df_t kMaxR2 = df_from_double(0.999999999999);
    if (r2.hi > kMaxR2.hi || (r2.hi == kMaxR2.hi && r2.lo > kMaxR2.lo)) r2 = kMaxR2;
    const df_t one = df_make(1.0f, 0.0f);
    const df_t term0 = df_mul(df_neg(b), df_log1p(df_neg(r2)));
    if (d <= 1) return (term0.hi > 0.0f && f_is_finite(term0.hi)) ? term0 : df_make(0.0f, 0.0f);
    df_t c = one;
    df_t s_minus_1 = df_make(0.0f, 0.0f);
    for (int k = 1; k < d; ++k) {
        const df_t factor = df_div(df_add(b, df_make(static_cast<float>(k - 1), 0.0f)), df_make(static_cast<float>(k), 0.0f));
        c = df_mul(df_mul(c, factor), r2);
        s_minus_1 = df_add(s_minus_1, c);
    }
    const df_t nll = df_sub(term0, df_log1p(s_minus_1));
    return (nll.hi > 0.0f && f_is_finite(nll.hi)) ? nll : df_make(0.0f, 0.0f);
}

SYCL_FFT_DF_INLINE df_t normalize_power_df(df_t dot, df_t inv_chi2_ref, df_t cond_bound, NormKind norm, int degree, int M, const CybenkoModelDf& cyb, df_t b) {
    if (!f_is_finite(dot.hi) || !f_is_finite(dot.lo)) return df_make(f_nan(), f_nan());
    if (dot.hi < 0.0f || (dot.hi == 0.0f && dot.lo <= 0.0f)) return df_make(0.0f, 0.0f);
    const df_t r2 = df_mul(dot, inv_chi2_ref);
    if (!f_is_finite(r2.hi) || !f_is_finite(r2.lo) || r2.hi > 1.0f || (r2.hi == 1.0f && r2.lo >= 0.0f)) {
        return df_make(f_nan(), f_nan());
    }
    if (norm == NormKind::Standard) {
        return r2;
    } else if (norm == NormKind::Anova) {
        const df_t d1 = df_make(static_cast<float>(2 * degree + 1), 0.0f);
        const df_t d2 = df_make(static_cast<float>(M - 1 - (2 * degree + 1)), 0.0f);
        df_t residual = df_sub(df_make(1.0f, 0.0f), r2);
        if (residual.hi < 1e-32f) residual = df_make(1e-32f, 0.0f);
        return df_div(df_mul(d2, r2), df_mul(d1, residual));
    } else if (norm == NormKind::Nll) {
        return nll_exact_pochhammer_df(degree, r2, b);
    } else {
        const df_t nll_r2 = nll_exact_pochhammer_df(degree, r2, b);
        const df_t kappa = df_div(cond_bound, df_make(static_cast<float>(2 * degree + 1), 0.0f));
        const df_t nll_cond = nll_cybenko_df(kappa, cyb, degree);
        return compute_bayes_laplace_df(df_sub(nll_r2, nll_cond));
    }
}

SYCL_FFT_DF_INLINE df_t ihs_wilkie_df(df_t z, df_t inv_n, df_t inv_n2_288) {
    if (!(inv_n.hi > 0.0f)) return z;
    if (!(z.hi > 0.0f) || !f_is_finite(z.hi)) return df_make(0.0f, 0.0f);
    df_t p2 = df_add(df_mul_f(z, -9.0f), df_make(76.0f, 0.0f));
    p2 = df_add(df_mul(p2, z), df_make(-132.0f, 0.0f));
    p2 = df_add(df_mul(p2, z), df_make(24.0f, 0.0f));
    const df_t t1 = df_mul(df_mul(z, df_sub(df_make(2.0f, 0.0f), z)), df_scale_pow2(inv_n, 0.25f));
    const df_t delta = df_sub(t1, df_mul(df_mul(z, p2), inv_n2_288));
    if (!(delta.hi > -1.0f) || !f_is_finite(delta.hi)) return z;
    const df_t corr = df_sub(z, df_log1p(delta));
    return (corr.hi > 0.0f && f_is_finite(corr.hi)) ? corr : z;
}

SYCL_FFT_DF_INLINE df_t ihs_erlang_df(df_t s, int degree) {
    if (!(s.hi > 0.0f) || !f_is_finite(s.hi)) return df_make(0.0f, 0.0f);
    df_t log_sum = df_make(1.0f, 0.0f);
    df_t term = df_make(1.0f, 0.0f);
    for (int i = 1; i < degree; ++i) {
        term = df_mul(term, df_div(s, df_make(static_cast<float>(i), 0.0f)));
        log_sum = df_add(log_sum, term);
    }
    if (!(log_sum.hi > 0.0f) || !f_is_finite(log_sum.hi)) return df_make(0.0f, 0.0f);
    const df_t nll = df_sub(s, df_log(log_sum));
    return (nll.hi > 0.0f && f_is_finite(nll.hi)) ? nll : df_make(0.0f, 0.0f);
}

#    else  // FP16 & FP32

SYCL_FFT_FORCE_INLINE float nll_exact_pochhammer_f(int d, float r2, float b) {
    if (!f_is_finite(r2) || r2 <= 0.0f) return 0.0f;
    if (r2 > 0.9999999f) r2 = 0.9999999f;
    const float term0 = -b * sycl::log1p(-r2);
    if (d <= 1) return (term0 > 0.0f && f_is_finite(term0)) ? term0 : 0.0f;
    float c = 1.0f;
    float s_minus_1 = 0.0f;
    for (int k = 1; k < d; ++k) {
        c *= ((b + static_cast<float>(k - 1)) / static_cast<float>(k)) * r2;
        s_minus_1 += c;
    }
    const float nll = term0 - sycl::log1p(s_minus_1);
    return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
}

SYCL_FFT_FORCE_INLINE float normalize_power_f(float dot, float inv_chi2_ref, float cond_bound, NormKind norm, int degree, int M, const CybenkoModelF& cyb,
                                              float b) {
    if (!f_is_finite(dot)) return f_nan();
    if (dot <= 0.0f) return 0.0f;
    const float r2 = dot * inv_chi2_ref;
    if (!f_is_finite(r2) || r2 >= 1.0f) return f_nan();
    if (norm == NormKind::Standard) {
        return r2;
    } else if (norm == NormKind::Anova) {
        const float d1 = 2.0f * static_cast<float>(degree) + 1.0f;
        const float d2 = static_cast<float>(M) - 1.0f - d1;
        float residual = 1.0f - r2;
        if (residual < 1e-32f) residual = 1e-32f;
        return (d2 * r2) / (d1 * residual);
    } else if (norm == NormKind::Nll) {
        return nll_exact_pochhammer_f(degree, r2, b);
    } else {
        const float nll_r2 = nll_exact_pochhammer_f(degree, r2, b);
        const float kappa = cond_bound / static_cast<float>(2 * degree + 1);
        const float nll_cond = nll_cybenko_f(kappa, cyb, degree);
        return compute_bayes_laplace_f(nll_r2 - nll_cond);
    }
}

SYCL_FFT_FORCE_INLINE float ihs_wilkie_f(float z, float inv_n, float inv_n2_288) {
    if (!(inv_n > 0.0f)) return z;
    if (!(z > 0.0f) || !f_is_finite(z)) return 0.0f;
    float p2 = sycl::fma(-9.0f, z, 76.0f);
    p2 = sycl::fma(p2, z, -132.0f);
    p2 = sycl::fma(p2, z, 24.0f);
    const float delta = sycl::fma(z * (2.0f - z), 0.25f * inv_n, -(z * p2) * inv_n2_288);
    if (!(delta > -1.0f) || !f_is_finite(delta)) return z;
    const float corr = z - sycl::log1p(delta);
    return (corr > 0.0f && f_is_finite(corr)) ? corr : z;
}

SYCL_FFT_FORCE_INLINE float ihs_erlang_f(float s, int degree) {
    if (!(s > 0.0f) || !f_is_finite(s)) return 0.0f;
    float log_sum = 1.0f, term = 1.0f;
    for (int i = 1; i < degree; ++i) {
        term *= s / static_cast<float>(i);
        log_sum += term;
    }
    if (!(log_sum > 0.0f) || !f_is_finite(log_sum)) return 0.0f;
    const float nll = s - sycl::log(log_sum);
    return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
}

#    endif  // SYCL_FFT_DF vs FP16/FP32
#endif      // SYCL_FFT_DOUBLE

// Helper: load compact-slot complex value(s) from a trig row
SYCL_FFT_FORCE_INLINE void load_trig_slot(const SolveComplexT* d_trig, size_t row_off, size_t bin_idx, SolveRealT& out_re, SolveRealT& out_im) {
#if defined(SYCL_FFT_HALF)
    // Load 2 adjacent compact slots (bin_idx, bin_idx + 1) as packed f16x2_t
    const SolveComplexT* ptr = d_trig + row_off + bin_idx;
    const uint32_t u0 = sycl_fft_bit_cast<uint32_t>(ptr[0]);
    const uint32_t u1 = sycl_fft_bit_cast<uint32_t>(ptr[1]);
    out_re = SYCL_FFT_BIT_CAST(f16x2_t, (u0 & 0xFFFFu) | ((u1 & 0xFFFFu) << 16));
    out_im = SYCL_FFT_BIT_CAST(f16x2_t, (u0 >> 16) | (u1 & 0xFFFF0000u));
#elif defined(SYCL_FFT_DF)
    const Complexdf v = d_trig[row_off + bin_idx];
    out_re = v.re;
    out_im = v.im;
#else
    const SolveComplexT v = d_trig[row_off + bin_idx];
    out_re = v.x();
    out_im = v.y();
#endif
}

SYCL_FFT_FORCE_INLINE SolveRealT load_slot_scalar(const SolveScalarT* ptr, int slot_i) {
#if defined(SYCL_FFT_HALF)
    return f16x2_make(ptr[slot_i], ptr[slot_i + 1]);
#else
    return ptr[slot_i];
#endif
}

SYCL_FFT_FORCE_INLINE SolveRealT splat_from_double(double v) {
#if defined(SYCL_FFT_HALF)
    const f16_scalar_t h = static_cast<f16_scalar_t>(static_cast<float>(v));
    return f16x2_make(h, h);
#elif defined(SYCL_FFT_DF)
    return df_from_double(v);
#elif defined(SYCL_FFT_DOUBLE)
    return v;
#else
    return static_cast<float>(v);
#endif
}

// ============================================================================
// 1. Fused Generalized Lomb-Scargle (degree == 1) Kernel
//    Matches src/chi2per.c:gls_impl + deferred deconvolution + natural-order write
// ============================================================================
inline void SYCL_FFT_SUFFIX(launch_gls_kernel)(sycl::queue& q, const SolveComplexT* d_trig, const SolveScalarT* d_slot_phi_hat,
                                               const SolveScalarT* d_slot_deconv, OutScalarT* d_power, OutScalarT* d_cond, const FastChi2SolveParams& p) {
    const int nout = p.nout;
    const int nfft = p.nfft;
    const int log_nfft = p.log_nfft;
    const int output_shift = p.output_shift;
    const int b_start = p.b_start;
    const int b_count = p.b_count;
    const int Nf = p.Nf;
    const int M = p.M;
    const PswfMode mode = p.mode;
    const NormKind norm = p.norm;
    const bool has_cond = (d_cond != nullptr);
    const bool need_cond = (norm == NormKind::Regularized) || has_cond;
    const bool use_raw_dec = (d_slot_phi_hat != nullptr && d_slot_deconv != nullptr);

    const size_t chunk_bins = static_cast<size_t>(b_count) * static_cast<size_t>(nout);
    const size_t row_w1 = 0;
    const size_t row_yw1 = chunk_bins;
    const size_t row_w2 = 2 * chunk_bins;

#if defined(SYCL_FFT_HALF)
    const int items_per_blk = nout >> 1;  // 2 adjacent slots per thread
#else
    const int items_per_blk = nout;
#endif
    const size_t total_items = static_cast<size_t>(b_count) * static_cast<size_t>(items_per_blk);
    const size_t wg = std::max<size_t>(1, std::min<size_t>(p.wg_size, 256));
    const size_t num_wg = (total_items + wg - 1) / wg;

    const SolveRealT ws_base = splat_from_double(p.ws_fft);
    const SolveRealT yws_base = splat_from_double(p.yws_fft);
    const SolveRealT inv_ws_base = splat_from_double(1.0 / p.ws_fft);
    const SolveRealT half_v = splat_from_double(0.5);
    const SolveRealT sqrt_half_v = splat_from_double(0.7071067811865475244);
    const SolveRealT two_v = splat_from_double(2.0);
    const SolveRealT three_v = splat_from_double(3.0);

#if defined(SYCL_FFT_DOUBLE)
    const double chi2_ref_out = 1.0 / p.chi2_ref;
    const double b_out = p.b_param;
    const CybenkoModel cyb = p.cyb;
#elif defined(SYCL_FFT_DF)
    const df_t chi2_ref_df = df_from_double(1.0 / p.chi2_ref);
    const df_t b_df = df_from_double(p.b_param);
    const CybenkoModelDf cyb = to_cybenko_df(p.cyb);
#else
    const float chi2_ref_f = static_cast<float>(1.0 / p.chi2_ref);
    const float b_f = static_cast<float>(p.b_param);
    const CybenkoModelF cyb = to_cybenko_f(p.cyb);
#endif

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_wg * wg, wg), [=](sycl::nd_item<1> item) {
            const size_t gid = item.get_global_linear_id();
            if (gid >= total_items) return;

            const int b = static_cast<int>(gid / static_cast<size_t>(items_per_blk));
            const int local_i = static_cast<int>(gid - static_cast<size_t>(b) * static_cast<size_t>(items_per_blk));
#if defined(SYCL_FFT_HALF)
            const int slot_i = local_i << 1;
#else
                const int slot_i = local_i;
#endif
            const size_t bin_idx = static_cast<size_t>(b) * static_cast<size_t>(nout) + static_cast<size_t>(slot_i);

            SolveRealT C, S, C2, S2, Ch, Sh;
            load_trig_slot(d_trig, row_w1, bin_idx, C, S);
            load_trig_slot(d_trig, row_w2, bin_idx, C2, S2);
            load_trig_slot(d_trig, row_yw1, bin_idx, Ch, Sh);

            SolveRealT ws_k = ws_base;
            SolveRealT yws_k = yws_base;
            SolveRealT inv_ws = inv_ws_base;
            SolveRealT dscale = SYCL_FFT_ONE;
            if (use_raw_dec) {
                const SolveRealT pk = load_slot_scalar(d_slot_phi_hat, slot_i);
                dscale = load_slot_scalar(d_slot_deconv, slot_i);
                ws_k = SYCL_FFT_MUL(ws_base, pk);
                yws_k = SYCL_FFT_MUL(yws_base, pk);
                inv_ws = SYCL_FFT_MUL(inv_ws_base, dscale);
            }

            const SolveRealT num = SYCL_FFT_SUB(S2, SYCL_FFT_MUL(SYCL_FFT_MUL(two_v, S), SYCL_FFT_MUL(C, inv_ws)));
            const SolveRealT den = SYCL_FFT_SUB(C2, SYCL_FFT_MUL(SYCL_FFT_SUB(SYCL_FFT_MUL(C, C), SYCL_FFT_MUL(S, S)), inv_ws));
            const SolveRealT radius = l_sqrt(SYCL_FFT_ADD(SYCL_FFT_MUL(num, num), SYCL_FFT_MUL(den, den)));

            // Branchless per-lane GLS tau angle computation
            auto compute_gls_trig = [&](auto num_l, auto den_l, auto rad_l, auto& S2w_l, auto& C2w_l, auto& Ctau_l, auto& Stau_l) {
#if defined(SYCL_FFT_DF)
                const bool nz = (rad_l.hi != 0.0f || rad_l.lo != 0.0f);
                const bool den_neg = (den_l.hi < 0.0f || (den_l.hi == 0.0f && den_l.lo < 0.0f));
                const df_t den_abs = den_neg ? df_neg(den_l) : den_l;
                const df_t inv_rad = nz ? df_div(df_make(1.0f, 0.0f), rad_l) : df_make(0.0f, 0.0f);
                S2w_l = nz ? (den_neg ? df_neg(df_mul(num_l, inv_rad)) : df_mul(num_l, inv_rad)) : df_make(0.0f, 0.0f);
                C2w_l = nz ? df_mul(den_abs, inv_rad) : df_make(1.0f, 0.0f);
                df_t op = df_add(df_make(1.0f, 0.0f), C2w_l);
                df_t om = df_sub(df_make(1.0f, 0.0f), C2w_l);
                if (op.hi < 0.0f) op = df_make(0.0f, 0.0f);
                if (om.hi < 0.0f) om = df_make(0.0f, 0.0f);
                Ctau_l = df_mul(kSqrt1_2df, df_sqrt(op));
                const df_t st_mag = df_mul(kSqrt1_2df, df_sqrt(om));
                const bool s2_pos = (S2w_l.hi > 0.0f || (S2w_l.hi == 0.0f && S2w_l.lo > 0.0f));
                const bool s2_neg = (S2w_l.hi < 0.0f || (S2w_l.hi == 0.0f && S2w_l.lo < 0.0f));
                Stau_l = s2_pos ? st_mag : (s2_neg ? df_neg(st_mag) : df_make(0.0f, 0.0f));
#else
                    using T = decltype(num_l);
                    const bool nz = (rad_l != static_cast<T>(0));
                    const bool den_neg = (den_l < static_cast<T>(0));
                    const T den_abs = den_neg ? -den_l : den_l;
                    S2w_l = nz ? ((den_neg ? -num_l : num_l) / rad_l) : static_cast<T>(0);
                    C2w_l = nz ? (den_abs / rad_l) : static_cast<T>(1);
                    T op = static_cast<T>(1) + C2w_l;
                    T om = static_cast<T>(1) - C2w_l;
                    if (op < static_cast<T>(0)) op = static_cast<T>(0);
                    if (om < static_cast<T>(0)) om = static_cast<T>(0);
                    const T sqh = static_cast<T>(0.7071067811865475244);
                    Ctau_l = sqh * sycl::sqrt(op);
                    const T st_mag = sqh * sycl::sqrt(om);
                    Stau_l = (S2w_l > static_cast<T>(0)) ? st_mag : ((S2w_l < static_cast<T>(0)) ? -st_mag : static_cast<T>(0));
#endif
            };

            SolveRealT S2w, C2w, Ctau, Stau;
#if defined(SYCL_FFT_HALF)
            float s2w0, c2w0, ct0, st0, s2w1, c2w1, ct1, st1;
            compute_gls_trig(static_cast<float>(f16x2_lo(num)), static_cast<float>(f16x2_lo(den)), static_cast<float>(f16x2_lo(radius)), s2w0, c2w0, ct0, st0);
            compute_gls_trig(static_cast<float>(f16x2_hi(num)), static_cast<float>(f16x2_hi(den)), static_cast<float>(f16x2_hi(radius)), s2w1, c2w1, ct1, st1);
            S2w = f16x2_make(static_cast<f16_scalar_t>(s2w0), static_cast<f16_scalar_t>(s2w1));
            C2w = f16x2_make(static_cast<f16_scalar_t>(c2w0), static_cast<f16_scalar_t>(c2w1));
            Ctau = f16x2_make(static_cast<f16_scalar_t>(ct0), static_cast<f16_scalar_t>(ct1));
            Stau = f16x2_make(static_cast<f16_scalar_t>(st0), static_cast<f16_scalar_t>(st1));
#else
                compute_gls_trig(num, den, radius, S2w, C2w, Ctau, Stau);
#endif

            SolveRealT YC = SYCL_FFT_ADD(SYCL_FFT_MUL(Ch, Ctau), SYCL_FFT_MUL(Sh, Stau));
            SolveRealT YS = SYCL_FFT_SUB(SYCL_FFT_MUL(Sh, Ctau), SYCL_FFT_MUL(Ch, Stau));
            const SolveRealT XC = SYCL_FFT_ADD(SYCL_FFT_MUL(C, Ctau), SYCL_FFT_MUL(S, Stau));
            const SolveRealT XS = SYCL_FFT_SUB(SYCL_FFT_MUL(S, Ctau), SYCL_FFT_MUL(C, Stau));

            const SolveRealT c2_rot = SYCL_FFT_ADD(SYCL_FFT_MUL(C2, C2w), SYCL_FFT_MUL(S2, S2w));
            SolveRealT CC = SYCL_FFT_SUB(SYCL_FFT_MUL(half_v, SYCL_FFT_ADD(ws_k, c2_rot)), SYCL_FFT_MUL(SYCL_FFT_MUL(XC, XC), inv_ws));
            SolveRealT SS = SYCL_FFT_SUB(SYCL_FFT_MUL(half_v, SYCL_FFT_SUB(ws_k, c2_rot)), SYCL_FFT_MUL(SYCL_FFT_MUL(XS, XS), inv_ws));
            YC = SYCL_FFT_SUB(YC, SYCL_FFT_MUL(SYCL_FFT_MUL(yws_k, XC), inv_ws));
            YS = SYCL_FFT_SUB(YS, SYCL_FFT_MUL(SYCL_FFT_MUL(yws_k, XS), inv_ws));

            SolveRealT dot = SYCL_FFT_ADD(l_div(SYCL_FFT_MUL(YC, YC), CC), l_div(SYCL_FFT_MUL(YS, YS), SS));
            if (use_raw_dec) dot = SYCL_FFT_MUL(dot, dscale);

            SolveRealT cond_bound = SYCL_FFT_ZERO;
            if (need_cond) {
                const SolveRealT c = SYCL_FFT_MUL(C, inv_ws);
                const SolveRealT s = SYCL_FFT_MUL(S, inv_ws);
                const SolveRealT c2 = SYCL_FFT_MUL(C2, inv_ws);
                const SolveRealT s2 = SYCL_FFT_MUL(S2, inv_ws);
                const SolveRealT c2_plus_s2 = SYCL_FFT_ADD(SYCL_FFT_MUL(c, c), SYCL_FFT_MUL(s, s));
                const SolveRealT k2_nr = SYCL_FFT_SUB(SYCL_FFT_ADD(c2, SYCL_FFT_MUL(s, s)), SYCL_FFT_MUL(c, c));
                const SolveRealT k2_ni = SYCL_FFT_SUB(s2, SYCL_FFT_MUL(two_v, SYCL_FFT_MUL(c, s)));
                const SolveRealT k2_den = SYCL_FFT_SUB(SYCL_FFT_ONE, c2_plus_s2);

                auto gls_cond_lane = [&](auto c2ps2, auto nr, auto ni, auto den_k) {
#if defined(SYCL_FFT_DOUBLE)
                    const double lim = 0.999999999999999;
                    const double k1_a2 = (!(c2ps2 >= 0.0) || c2ps2 > lim) ? lim : c2ps2;
                    double k2_a2 = lim;
                    if (den_k > 0.0) {
                        const double raw = (nr * nr + ni * ni) / (den_k * den_k);
                        k2_a2 = (!(raw >= 0.0) || raw > lim) ? lim : raw;
                    }
                    const double r1 = sycl::sqrt(k1_a2);
                    const double r2 = sycl::sqrt(k2_a2);
                    return 3.0 * ((1.0 + r1) / (1.0 - r1)) * ((1.0 + r2) / (1.0 - r2));
#elif defined(SYCL_FFT_DF)
                        constexpr df_t lim = df_from_double(0.999999999999);
                        const df_t one = df_make(1.0f, 0.0f);
                        const bool c1_ok = (c2ps2.hi > 0.0f || (c2ps2.hi == 0.0f && c2ps2.lo >= 0.0f));
                        const bool c1_gt = (c2ps2.hi > lim.hi || (c2ps2.hi == lim.hi && c2ps2.lo > lim.lo));
                        const df_t k1_a2 = (!c1_ok || c1_gt) ? lim : c2ps2;
                        df_t k2_a2 = lim;
                        if (den_k.hi > 0.0f || (den_k.hi == 0.0f && den_k.lo > 0.0f)) {
                            const df_t raw = df_div(df_add(df_sqr(nr), df_sqr(ni)), df_sqr(den_k));
                            const bool r_ok = (raw.hi > 0.0f || (raw.hi == 0.0f && raw.lo >= 0.0f));
                            const bool r_gt = (raw.hi > lim.hi || (raw.hi == lim.hi && raw.lo > lim.lo));
                            k2_a2 = (!r_ok || r_gt) ? lim : raw;
                        }
                        const df_t r1 = df_sqrt(k1_a2);
                        const df_t r2 = df_sqrt(k2_a2);
                        const df_t f1 = df_div(df_add(one, r1), df_sub(one, r1));
                        const df_t f2 = df_div(df_add(one, r2), df_sub(one, r2));
                        return df_mul_f(df_mul(f1, f2), 3.0f);
#elif defined(SYCL_FFT_HALF)
                        const float lim = 0.999f;
                        const float k1_a2 = (!(c2ps2 >= 0.0f) || c2ps2 > lim) ? lim : c2ps2;
                        float k2_a2 = lim;
                        if (den_k > 0.0f) {
                            const float raw = (nr * nr + ni * ni) / (den_k * den_k);
                            k2_a2 = (!(raw >= 0.0f) || raw > lim) ? lim : raw;
                        }
                        const float r1 = sycl::sqrt(k1_a2);
                        const float r2 = sycl::sqrt(k2_a2);
                        return static_cast<f16_scalar_t>(
                            3.0f * ((1.0f + r1) / (1.0f - r1)) * ((1.0f + r2) / (1.0f - r2)));
#else
                        const float lim = 0.9999999f;
                        const float k1_a2 = (!(c2ps2 >= 0.0f) || c2ps2 > lim) ? lim : c2ps2;
                        float k2_a2 = lim;
                        if (den_k > 0.0f) {
                            const float raw = (nr * nr + ni * ni) / (den_k * den_k);
                            k2_a2 = (!(raw >= 0.0f) || raw > lim) ? lim : raw;
                        }
                        const float r1 = sycl::sqrt(k1_a2);
                        const float r2 = sycl::sqrt(k2_a2);
                        return 3.0f * ((1.0f + r1) / (1.0f - r1)) * ((1.0f + r2) / (1.0f - r2));
#endif
                };
#if defined(SYCL_FFT_HALF)
                cond_bound = f16x2_make(gls_cond_lane(static_cast<float>(f16x2_lo(c2_plus_s2)), static_cast<float>(f16x2_lo(k2_nr)),
                                                      static_cast<float>(f16x2_lo(k2_ni)), static_cast<float>(f16x2_lo(k2_den))),
                                        gls_cond_lane(static_cast<float>(f16x2_hi(c2_plus_s2)), static_cast<float>(f16x2_hi(k2_nr)),
                                                      static_cast<float>(f16x2_hi(k2_ni)), static_cast<float>(f16x2_hi(k2_den))));
#else
                    cond_bound = gls_cond_lane(c2_plus_s2, k2_nr, k2_ni, k2_den);
#endif
            }
            (void)three_v;
            (void)sqrt_half_v;

            // Emit directly to natural frequency index (deferred bit-reversal epilogue!)
            const int blk_base = (b_start + b) * nout;
#if defined(SYCL_FFT_HALF)
            for (int lane = 0; lane < 2; ++lane) {
                const int s_i = slot_i + lane;
                const int slot_p = nufft1_compact_idx_to_slot(s_i, mode);
                const int nat_k = nufft1_slot_to_natural_idx(slot_p, log_nfft, nfft, output_shift);
                const int gf = blk_base + nat_k;
                if (gf < Nf) {
                    const float dot_l = static_cast<float>(lane == 0 ? f16x2_lo(dot) : f16x2_hi(dot));
                    const float cb_l = static_cast<float>(lane == 0 ? f16x2_lo(cond_bound) : f16x2_hi(cond_bound));
                    if (has_cond) d_cond[gf] = cb_l;
                    d_power[gf] = normalize_power_f(dot_l, chi2_ref_f, cb_l, norm, 1, M, cyb, b_f);
                }
            }
#else
                const int slot_p = nufft1_compact_idx_to_slot(slot_i, mode);
                const int nat_k = nufft1_slot_to_natural_idx(slot_p, log_nfft, nfft, output_shift);
                const int gf = blk_base + nat_k;
                if (gf < Nf) {
#    if defined(SYCL_FFT_DOUBLE)
                    if (has_cond) d_cond[gf] = cond_bound;
                    d_power[gf] = normalize_power_d(dot, chi2_ref_out, cond_bound, norm, 1, M, cyb, b_out);
#    elif defined(SYCL_FFT_DF)
                    if (has_cond) d_cond[gf] = cond_bound;
                    d_power[gf] = normalize_power_df(dot, chi2_ref_df, cond_bound, norm, 1, M, cyb, b_df);
#    else
                    if (has_cond) d_cond[gf] = cond_bound;
                    d_power[gf] = normalize_power_f(dot, chi2_ref_f, cond_bound, norm, 1, M, cyb, b_f);
#    endif
                }
#endif
        });
    });
}

// ============================================================================
// 2. Fused Hermitian Toeplitz Levinson-Durbin Kernel (degree > 1)
//    Port of src/linalg.c:solve_levinson + deferred deconv + natural-order write.
//    ExactDeg > 0: compile-time NOrder = 2*ExactDeg+1 + unrolled predictor a[]
//                  -> 100% register-resident (0 bytes local stack / 0 spills!).
//    MaxOrder > 0, ExactDeg == 0: bounded local array for d <= 12.
//    MaxOrder == 0: general dynamic-degree fallback using d_a_scratch.
// ============================================================================
template <int MaxOrder, int ExactDeg = 0>
inline void SYCL_FFT_SUFFIX(launch_levinson_kernel)(sycl::queue& q, const SolveComplexT* d_trig, const SolveScalarT* d_slot_phi_hat,
                                                    const SolveScalarT* d_slot_deconv, SolveRealT* d_a_scratch, OutScalarT* d_power, OutScalarT* d_cond,
                                                    const FastChi2SolveParams& p) {
    const int degree_rt = p.degree;
    const int norder_rt = 2 * degree_rt + 1;
    const int nout = p.nout;
    const int nfft = p.nfft;
    const int log_nfft = p.log_nfft;
    const int output_shift = p.output_shift;
    const int b_start = p.b_start;
    const int b_count = p.b_count;
    const int Nf = p.Nf;
    const int M = p.M;
    const PswfMode mode = p.mode;
    const NormKind norm = p.norm;
    const bool has_cond = (d_cond != nullptr);
    const bool use_raw_dec = (d_slot_phi_hat != nullptr && d_slot_deconv != nullptr);

    const size_t chunk_bins = static_cast<size_t>(b_count) * static_cast<size_t>(nout);
#if defined(SYCL_FFT_HALF)
    const int items_per_blk = nout >> 1;
#else
    const int items_per_blk = nout;
#endif
    const size_t total_items = static_cast<size_t>(b_count) * static_cast<size_t>(items_per_blk);
    const size_t wg = std::max<size_t>(1, std::min<size_t>(p.wg_size, 256));
    const size_t num_wg = (total_items + wg - 1) / wg;

    const SolveRealT ws_base = splat_from_double(p.ws_fft);
    const SolveRealT inv_ws_base = splat_from_double(1.0 / p.ws_fft);
    const SolveRealT yws_base = splat_from_double(p.yws_fft);
    const SolveRealT norder_val = splat_from_double(static_cast<double>(norder_rt));

#if defined(SYCL_FFT_DOUBLE)
    const double chi2_ref_out = 1.0 / p.chi2_ref;
    const double b_out = p.b_param;
    const CybenkoModel cyb = p.cyb;
#elif defined(SYCL_FFT_DF)
    const df_t chi2_ref_df = df_from_double(1.0 / p.chi2_ref);
    const df_t b_df = df_from_double(p.b_param);
    const CybenkoModelDf cyb = to_cybenko_df(p.cyb);
#else
    const float chi2_ref_f = static_cast<float>(1.0 / p.chi2_ref);
    const float b_f = static_cast<float>(p.b_param);
    const CybenkoModelF cyb = to_cybenko_f(p.cyb);
#endif

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_wg * wg, wg), [=](sycl::nd_item<1> item) {
            const size_t gid = item.get_global_linear_id();
            if (gid >= total_items) return;

            const int degree = (ExactDeg > 0) ? ExactDeg : degree_rt;
            const int norder = (ExactDeg > 0) ? (2 * ExactDeg + 1) : norder_rt;

            const int b = static_cast<int>(gid / static_cast<size_t>(items_per_blk));
            const int local_i = static_cast<int>(gid - static_cast<size_t>(b) * static_cast<size_t>(items_per_blk));
#if defined(SYCL_FFT_HALF)
            const int slot_i = local_i << 1;
#else
                const int slot_i = local_i;
#endif
            const size_t bin_idx = static_cast<size_t>(b) * static_cast<size_t>(nout) + static_cast<size_t>(slot_i);

            SolveRealT ws_k = ws_base;
            SolveRealT yws_k = yws_base;
            SolveRealT inv_E = inv_ws_base;
            SolveRealT dscale = SYCL_FFT_ONE;
            if (use_raw_dec) {
                const SolveRealT pk = load_slot_scalar(d_slot_phi_hat, slot_i);
                dscale = load_slot_scalar(d_slot_deconv, slot_i);
                ws_k = SYCL_FFT_MUL(ws_base, pk);
                yws_k = SYCL_FFT_MUL(yws_base, pk);
                inv_E = SYCL_FFT_MUL(inv_ws_base, dscale);
            }

            // Helpers to fetch R[idx] and Y[idx] directly from coalesced L1-cached d_trig
            auto get_R = [&](int k, SolveRealT& rr, SolveRealT& ri) {
                if (k == 0) {
                    rr = ws_k;
                    ri = SYCL_FFT_ZERO;
                } else {
                    const size_t row_off = trig_w_row(k, degree) * chunk_bins;
                    load_trig_slot(d_trig, row_off, bin_idx, rr, ri);
                    ri = SYCL_FFT_NEG(ri);
                }
            };
            auto get_Y = [&](int k, SolveRealT& yr, SolveRealT& yi) {
                const int h = k - degree;
                if (h == 0) {
                    yr = yws_k;
                    yi = SYCL_FFT_ZERO;
                } else if (h > 0) {
                    const size_t row_off = trig_yw_row(h) * chunk_bins;
                    load_trig_slot(d_trig, row_off, bin_idx, yr, yi);
                    yi = SYCL_FFT_NEG(yi);
                } else {
                    const size_t row_off = trig_yw_row(-h) * chunk_bins;
                    load_trig_slot(d_trig, row_off, bin_idx, yr, yi);
                }
            };

            // Predictor storage: register array for specialized MaxOrder > 0, 32-lane warp-tiled scratch otherwise
            constexpr int kRegLen = (MaxOrder > 0) ? MaxOrder : 1;
            SolveRealT a_r_reg[kRegLen];
            SolveRealT a_i_reg[kRegLen];
            const size_t scr_base = (MaxOrder == 0) ? ((((gid >> 5) * static_cast<size_t>(2 * (norder - 2))) << 5) | (gid & size_t{31})) : size_t{0};

            auto load_a = [&](int idx, SolveRealT& ar, SolveRealT& ai) {
                if constexpr (MaxOrder > 0) {
                    ar = a_r_reg[idx];
                    ai = a_i_reg[idx];
                } else {
                    const size_t off = scr_base + (static_cast<size_t>(2 * (idx - 1)) << 5);
                    ar = d_a_scratch[off];
                    ai = d_a_scratch[off + 32];
                }
            };
            auto store_a = [&](int idx, SolveRealT ar, SolveRealT ai) {
                if constexpr (MaxOrder > 0) {
                    a_r_reg[idx] = ar;
                    a_i_reg[idx] = ai;
                } else {
                    const size_t off = scr_base + (static_cast<size_t>(2 * (idx - 1)) << 5);
                    d_a_scratch[off] = ar;
                    d_a_scratch[off + 32] = ai;
                }
            };

            // --- Fused in-place Levinson-Durbin recursion (mirrors src/linalg.c) ---
            SolveRealT E = ws_k;
            SolveRealT y0r, y0i;
            get_Y(0, y0r, y0i);
            SolveRealT r2 = SYCL_FFT_MUL(l_abs2(y0r, y0i), inv_E);
            SolveRealT cond_num = norder_val;
            SolveRealT cond_den = SYCL_FFT_ONE;

            auto levinson_step = [&](int k) {
                // lambda = R[k] + sum_{i=1}^{k-1} R[k - i] * a[i]
                SolveRealT lam_r, lam_i;
                get_R(k, lam_r, lam_i);
                for (int i = 1; i < k; ++i) {
                    SolveRealT rr, ri, ar, ai;
                    get_R(k - i, rr, ri);
                    load_a(i, ar, ai);
                    l_cmac(rr, ri, ar, ai, lam_r, lam_i);
                }

                const SolveRealT gamma_r = SYCL_FFT_NEG(SYCL_FFT_MUL(lam_r, inv_E));
                const SolveRealT gamma_i = SYCL_FFT_NEG(SYCL_FFT_MUL(lam_i, inv_E));

                const SolveRealT abs2_gamma = l_abs2(gamma_r, gamma_i);
                const SolveRealT abs_gamma = l_sqrt(abs2_gamma);
                E = SYCL_FFT_MUL(E, SYCL_FFT_SUB(SYCL_FFT_ONE, abs2_gamma));
                inv_E = l_div(SYCL_FFT_ONE, E);

                const bool save_a = (k + 1 < norder);
                if (save_a) store_a(k, gamma_r, gamma_i);

                SolveRealT xr, xi;
                get_Y(k, xr, xi);
                l_cmac(gamma_r, gamma_i, y0r, y0i, xr, xi);

                // Paired in-place symmetric update of (a[lo], a[hi]) fused with xi accumulation
                int lo = 1, hi = k - 1;
                for (; lo < hi; ++lo, --hi) {
                    SolveRealT alr, ali, ahr, ahi;
                    load_a(lo, alr, ali);
                    load_a(hi, ahr, ahi);

                    SolveRealT nlr, nli, nhr, nhi;
                    l_cmac_conj(gamma_r, gamma_i, ahr, ahi, alr, ali, nlr, nli);
                    l_cmac_conj(gamma_r, gamma_i, alr, ali, ahr, ahi, nhr, nhi);

                    if (save_a) {
                        store_a(lo, nlr, nli);
                        store_a(hi, nhr, nhi);
                    }

                    SolveRealT yhr, yhi, ylr, yli;
                    get_Y(hi, yhr, yhi);
                    get_Y(lo, ylr, yli);

                    l_cmac(nlr, nli, yhr, yhi, xr, xi);
                    l_cmac(nhr, nhi, ylr, yli, xr, xi);
                }
                if (lo == hi) {
                    SolveRealT alr, ali, nlr, nli;
                    load_a(lo, alr, ali);
                    l_cmac_conj(gamma_r, gamma_i, alr, ali, alr, ali, nlr, nli);
                    if (save_a) store_a(lo, nlr, nli);

                    SolveRealT ylr, yli;
                    get_Y(lo, ylr, yli);
                    l_cmac(nlr, nli, ylr, yli, xr, xi);
                }

                cond_num = SYCL_FFT_MUL(cond_num, SYCL_FFT_ADD(SYCL_FFT_ONE, abs_gamma));
                cond_den = SYCL_FFT_MUL(cond_den, SYCL_FFT_SUB(SYCL_FFT_ONE, abs_gamma));

                const SolveRealT abs2_xi = l_abs2(xr, xi);
                r2 = l_fma(abs2_xi, inv_E, r2);
            };

            if constexpr (ExactDeg > 0) {
#if defined(SYCL_FFT_HALF) || defined(SYCL_FFT_FLOAT)
                constexpr bool kPreloadRY = (ExactDeg <= 6);
#else
                    constexpr bool kPreloadRY = (ExactDeg <= 3);
#endif
                constexpr int kRLen = kPreloadRY ? (2 * ExactDeg) : 1;
                constexpr int kYLen = kPreloadRY ? ExactDeg : 1;
                SolveRealT R_r_reg[kRLen], R_i_reg[kRLen];
                SolveRealT Y_r_reg[kYLen], Y_i_reg[kYLen];
                if constexpr (kPreloadRY) {
                    unroll_for<2 * ExactDeg>([&](auto q_ic) {
                        constexpr int q = decltype(q_ic)::value + 1;
                        const size_t row_off = trig_w_row(q, ExactDeg) * chunk_bins;
                        load_trig_slot(d_trig, row_off, bin_idx, R_r_reg[q - 1], R_i_reg[q - 1]);
                        R_i_reg[q - 1] = SYCL_FFT_NEG(R_i_reg[q - 1]);
                    });
                    unroll_for<ExactDeg>([&](auto q_ic) {
                        constexpr int q = decltype(q_ic)::value + 1;
                        const size_t row_off = trig_yw_row(q) * chunk_bins;
                        load_trig_slot(d_trig, row_off, bin_idx, Y_r_reg[q - 1], Y_i_reg[q - 1]);
                    });
                }
                auto fast_R = [&](auto k_tag, SolveRealT& rr, SolveRealT& ri) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int k_idx = decltype(k_tag)::value;
                    if constexpr (kPreloadRY) {
                        if constexpr (k_idx == 0) {
                            rr = ws_k;
                            ri = SYCL_FFT_ZERO;
                        } else {
                            rr = R_r_reg[k_idx - 1];
                            ri = R_i_reg[k_idx - 1];
                        }
                    } else {
                        get_R(k_idx, rr, ri);
                    }
                };
                auto fast_Y = [&](auto k_tag, SolveRealT& yr, SolveRealT& yi) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int k_idx = decltype(k_tag)::value;
                    if constexpr (kPreloadRY) {
                        constexpr int h = k_idx - ExactDeg;
                        if constexpr (h == 0) {
                            yr = yws_k;
                            yi = SYCL_FFT_ZERO;
                        } else if constexpr (h > 0) {
                            yr = Y_r_reg[h - 1];
                            yi = SYCL_FFT_NEG(Y_i_reg[h - 1]);
                        } else {
                            yr = Y_r_reg[-h - 1];
                            yi = Y_i_reg[-h - 1];
                        }
                    } else {
                        get_Y(k_idx, yr, yi);
                    }
                };

                unroll_for<2 * ExactDeg>([&](auto k_ic) {
                    constexpr int k = decltype(k_ic)::value + 1;
                    constexpr bool kSaveA = (k < 2 * ExactDeg);
                    SolveRealT lam_r, lam_i;
                    fast_R(std::integral_constant<int, k>{}, lam_r, lam_i);
                    unroll_for<k - 1>([&](auto i_ic) {
                        constexpr int i = decltype(i_ic)::value + 1;
                        SolveRealT rr, ri;
                        fast_R(std::integral_constant<int, k - i>{}, rr, ri);
                        l_cmac(rr, ri, a_r_reg[i], a_i_reg[i], lam_r, lam_i);
                    });

                    const SolveRealT gamma_r = SYCL_FFT_NEG(SYCL_FFT_MUL(lam_r, inv_E));
                    const SolveRealT gamma_i = SYCL_FFT_NEG(SYCL_FFT_MUL(lam_i, inv_E));

                    const SolveRealT abs2_gamma = l_abs2(gamma_r, gamma_i);
                    const SolveRealT abs_gamma = l_sqrt(abs2_gamma);
                    E = SYCL_FFT_MUL(E, SYCL_FFT_SUB(SYCL_FFT_ONE, abs2_gamma));
                    inv_E = l_div(SYCL_FFT_ONE, E);

                    if constexpr (kSaveA) {
                        a_r_reg[k] = gamma_r;
                        a_i_reg[k] = gamma_i;
                    }

                    SolveRealT xr, xi;
                    fast_Y(std::integral_constant<int, k>{}, xr, xi);
                    l_cmac(gamma_r, gamma_i, y0r, y0i, xr, xi);

                    constexpr int kPairs = (k - 1) / 2;
                    unroll_for<kPairs>([&](auto p_ic) {
                        constexpr int lo = decltype(p_ic)::value + 1;
                        constexpr int hi = k - lo;
                        const SolveRealT alr = a_r_reg[lo];
                        const SolveRealT ali = a_i_reg[lo];
                        const SolveRealT ahr = a_r_reg[hi];
                        const SolveRealT ahi = a_i_reg[hi];

                        SolveRealT nlr, nli, nhr, nhi;
                        l_cmac_conj(gamma_r, gamma_i, ahr, ahi, alr, ali, nlr, nli);
                        l_cmac_conj(gamma_r, gamma_i, alr, ali, ahr, ahi, nhr, nhi);

                        if constexpr (kSaveA) {
                            a_r_reg[lo] = nlr;
                            a_i_reg[lo] = nli;
                            a_r_reg[hi] = nhr;
                            a_i_reg[hi] = nhi;
                        }

                        SolveRealT yhr, yhi, ylr, yli;
                        fast_Y(std::integral_constant<int, hi>{}, yhr, yhi);
                        fast_Y(std::integral_constant<int, lo>{}, ylr, yli);

                        l_cmac(nlr, nli, yhr, yhi, xr, xi);
                        l_cmac(nhr, nhi, ylr, yli, xr, xi);
                    });
                    if constexpr ((k & 1) == 0) {
                        constexpr int mid = k / 2;
                        const SolveRealT alr = a_r_reg[mid];
                        const SolveRealT ali = a_i_reg[mid];
                        SolveRealT nlr, nli;
                        l_cmac_conj(gamma_r, gamma_i, alr, ali, alr, ali, nlr, nli);
                        if constexpr (kSaveA) {
                            a_r_reg[mid] = nlr;
                            a_i_reg[mid] = nli;
                        }

                        SolveRealT ylr, yli;
                        fast_Y(std::integral_constant<int, mid>{}, ylr, yli);
                        l_cmac(nlr, nli, ylr, yli, xr, xi);
                    }

                    cond_num = SYCL_FFT_MUL(cond_num, SYCL_FFT_ADD(SYCL_FFT_ONE, abs_gamma));
                    cond_den = SYCL_FFT_MUL(cond_den, SYCL_FFT_SUB(SYCL_FFT_ONE, abs_gamma));

                    const SolveRealT abs2_xi = l_abs2(xr, xi);
                    r2 = l_fma(abs2_xi, inv_E, r2);
                });
            } else {
                for (int k = 1; k < norder; ++k) {
                    levinson_step(k);
                }
            }
            (void)levinson_step;

            const SolveRealT cond_bound = l_div(cond_num, cond_den);
            SolveRealT dot = r2;
            if (use_raw_dec) dot = SYCL_FFT_MUL(dot, dscale);

            // Emit directly to natural frequency index (deferred bit-reversal epilogue!)
            const int blk_base = (b_start + b) * nout;
#if defined(SYCL_FFT_HALF)
            for (int lane = 0; lane < 2; ++lane) {
                const int s_i = slot_i + lane;
                const int slot_p = nufft1_compact_idx_to_slot(s_i, mode);
                const int nat_k = nufft1_slot_to_natural_idx(slot_p, log_nfft, nfft, output_shift);
                const int gf = blk_base + nat_k;
                if (gf < Nf) {
                    const float cd_l = static_cast<float>(lane == 0 ? f16x2_lo(cond_den) : f16x2_hi(cond_den));
                    float cb_l = static_cast<float>(lane == 0 ? f16x2_lo(cond_bound) : f16x2_hi(cond_bound));
                    if (!(cd_l > 0.0f) || !f_is_finite(cb_l) || cb_l < 1.0f) cb_l = f_inf();
                    const float dot_l = static_cast<float>(lane == 0 ? f16x2_lo(dot) : f16x2_hi(dot));
                    if (has_cond) {
                        d_cond[gf] = cb_l;
                        d_power[gf] = normalize_power_f(dot_l, chi2_ref_f, cb_l, norm, degree, M, cyb, b_f);
                    } else if (cond_is_singular_f(cb_l)) {
                        d_power[gf] = f_nan();
                    } else {
                        d_power[gf] = normalize_power_f(dot_l, chi2_ref_f, cb_l, norm, degree, M, cyb, b_f);
                    }
                }
            }
#else
                const int slot_p = nufft1_compact_idx_to_slot(slot_i, mode);
                const int nat_k = nufft1_slot_to_natural_idx(slot_p, log_nfft, nfft, output_shift);
                const int gf = blk_base + nat_k;
                if (gf < Nf) {
#    if defined(SYCL_FFT_DOUBLE)
                    const double cb_clean = (!(cond_den > 0.0) || !d_is_finite(cond_bound) || cond_bound < 1.0)
                        ? d_inf() : cond_bound;
                    if (has_cond) {
                        d_cond[gf] = cb_clean;
                        d_power[gf] = normalize_power_d(dot, chi2_ref_out, cb_clean, norm, degree, M, cyb, b_out);
                    } else if (cond_is_singular_d(cb_clean)) {
                        d_power[gf] = d_nan();
                    } else {
                        d_power[gf] = normalize_power_d(dot, chi2_ref_out, cb_clean, norm, degree, M, cyb, b_out);
                    }
#    elif defined(SYCL_FFT_DF)
                    const float cd_f = cond_den.hi + cond_den.lo;
                    const float cb_f_raw = cond_bound.hi + cond_bound.lo;
                    const bool bad_cb = (!(cd_f > 0.0f) || !f_is_finite(cb_f_raw) || cb_f_raw < 1.0f);
                    const df_t cb_clean = bad_cb ? df_make(f_inf(), 0.0f) : cond_bound;
                    const float cb_f = bad_cb ? f_inf() : cb_f_raw;
                    if (has_cond) {
                        d_cond[gf] = cb_clean;
                        d_power[gf] = normalize_power_df(dot, chi2_ref_df, cb_clean, norm, degree, M, cyb, b_df);
                    } else if (cond_is_singular_f(cb_f)) {
                        d_power[gf] = df_make(f_nan(), f_nan());
                    } else {
                        d_power[gf] = normalize_power_df(dot, chi2_ref_df, cb_clean, norm, degree, M, cyb, b_df);
                    }
#    else
                    const float cb_clean = (!(cond_den > 0.0f) || !f_is_finite(cond_bound) || cond_bound < 1.0f)
                        ? f_inf() : cond_bound;
                    if (has_cond) {
                        d_cond[gf] = cb_clean;
                        d_power[gf] = normalize_power_f(dot, chi2_ref_f, cb_clean, norm, degree, M, cyb, b_f);
                    } else if (cond_is_singular_f(cb_clean)) {
                        d_power[gf] = f_nan();
                    } else {
                        d_power[gf] = normalize_power_f(dot, chi2_ref_f, cb_clean, norm, degree, M, cyb, b_f);
                    }
#    endif
                }
#endif
        });
    });
}

void SYCL_FFT_SUFFIX(launch_fastchi2_solve)(sycl::queue& q, const SolveComplexT* d_trig, const SolveScalarT* d_slot_phi_hat, const SolveScalarT* d_slot_deconv,
                                            SolveRealT* d_a_scratch, OutScalarT* d_power, OutScalarT* d_cond, const FastChi2SolveParams& p) {
    if (p.degree == 1) {
        SYCL_FFT_SUFFIX(launch_gls_kernel)(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_power, d_cond, p);
        return;
    }
    switch (p.degree) {
        case 2:
            SYCL_FFT_SUFFIX(launch_levinson_kernel)<5, 2>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
            return;
        case 3:
            SYCL_FFT_SUFFIX(launch_levinson_kernel)<7, 3>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
            return;
        case 4:
            SYCL_FFT_SUFFIX(launch_levinson_kernel)<9, 4>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
            return;
        case 5:
            SYCL_FFT_SUFFIX(launch_levinson_kernel)<11, 5>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
            return;
#if defined(SYCL_FFT_HALF) || defined(SYCL_FFT_FLOAT)
        case 6:
            SYCL_FFT_SUFFIX(launch_levinson_kernel)<13, 6>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
            return;
        case 7:
            SYCL_FFT_SUFFIX(launch_levinson_kernel)<15, 7>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
            return;
        case 8:
            SYCL_FFT_SUFFIX(launch_levinson_kernel)<17, 8>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
            return;
#endif
        default:
            break;
    }
    const int norder = 2 * p.degree + 1;
    if (norder <= 17) {
        SYCL_FFT_SUFFIX(launch_levinson_kernel)<17, 0>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
    } else if (norder <= 25) {
        SYCL_FFT_SUFFIX(launch_levinson_kernel)<25, 0>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
    } else {
        SYCL_FFT_SUFFIX(launch_levinson_kernel)<0, 0>(q, d_trig, d_slot_phi_hat, d_slot_deconv, d_a_scratch, d_power, d_cond, p);
    }
}

// ============================================================================
// 3. Fused IHS Accumulation & Finalization Kernels
//    Mirrors src/chi2per.c:ihs_power_sweep_depth + ihs_finalize_block
// ============================================================================
void SYCL_FFT_SUFFIX(launch_ihs_accum)(sycl::queue& q, const SolveComplexT* d_z, const IhsDeconvT* d_slot_deconv, IhsAccT* d_acc, bool first_harmonic,
                                       const IhsSolveParams& p) {
    const int nout = p.nout;
    const size_t total_bins = static_cast<size_t>(p.b_count) * static_cast<size_t>(nout);
    const size_t wg = std::max<size_t>(1, std::min<size_t>(p.wg_size, 256));
    const size_t num_wg = (total_bins + wg - 1) / wg;
    const bool nll_multi = (p.norm == NormKind::Nll && p.degree > 1);

    const double inv_n_eff_d = (p.n_eff > 0.0 && std::isfinite(p.n_eff)) ? (1.0 / p.n_eff) : 0.0;
    const double inv_n2_288_d = (inv_n_eff_d * inv_n_eff_d) / 288.0;
#if defined(SYCL_FFT_DOUBLE)
    const double nll_scale = p.nll_scale;
    const double inv_n_eff = inv_n_eff_d;
    const double inv_n2_288 = inv_n2_288_d;
#elif defined(SYCL_FFT_DF)
    const df_t nll_scale = df_from_double(p.nll_scale);
    const df_t inv_n_eff = df_from_double(inv_n_eff_d);
    const df_t inv_n2_288 = df_from_double(inv_n2_288_d);
#else
    const float nll_scale = static_cast<float>(p.nll_scale);
    const float inv_n_eff = static_cast<float>(inv_n_eff_d);
    const float inv_n2_288 = static_cast<float>(inv_n2_288_d);
#endif

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_wg * wg, wg), [=](sycl::nd_item<1> item) {
            const size_t idx = item.get_global_linear_id();
            if (idx >= total_bins) return;

#if defined(SYCL_FFT_HALF)
            const uint32_t u = sycl_fft_bit_cast<uint32_t>(d_z[idx]);
            const f16x2_t pair = SYCL_FFT_BIT_CAST(f16x2_t, u);
            const float re = static_cast<float>(f16x2_lo(pair));
            const float im = static_cast<float>(f16x2_hi(pair));
            float val = re * re + im * im;
            if (nll_multi) {
                const int slot_i = static_cast<int>(idx % static_cast<size_t>(nout));
                const float d = d_slot_deconv[slot_i];
                val = ihs_wilkie_f(val * (nll_scale * d * d), inv_n_eff, inv_n2_288);
            }
            d_acc[idx] = first_harmonic ? val : (d_acc[idx] + val);
#elif defined(SYCL_FFT_DOUBLE)
                const Complex z = d_z[idx];
                double val = z.x() * z.x() + z.y() * z.y();
                if (nll_multi) {
                    const int slot_i = static_cast<int>(idx % static_cast<size_t>(nout));
                    const double d = d_slot_deconv[slot_i];
                    val = ihs_wilkie_d(val * (nll_scale * d * d), inv_n_eff, inv_n2_288);
                }
                d_acc[idx] = first_harmonic ? val : (d_acc[idx] + val);
#elif defined(SYCL_FFT_DF)
                const Complexdf z = d_z[idx];
                df_t val = df_add(df_sqr(z.re), df_sqr(z.im));
                if (nll_multi) {
                    const int slot_i = static_cast<int>(idx % static_cast<size_t>(nout));
                    const df_t d = d_slot_deconv[slot_i];
                    val = ihs_wilkie_df(df_mul(val, df_mul(nll_scale, df_sqr(d))), inv_n_eff, inv_n2_288);
                }
                d_acc[idx] = first_harmonic ? val : df_add(d_acc[idx], val);
#else
                const Complexf z = d_z[idx];
                float val = z.x() * z.x() + z.y() * z.y();
                if (nll_multi) {
                    const int slot_i = static_cast<int>(idx % static_cast<size_t>(nout));
                    const float d = d_slot_deconv[slot_i];
                    val = ihs_wilkie_f(val * (nll_scale * d * d), inv_n_eff, inv_n2_288);
                }
                d_acc[idx] = first_harmonic ? val : (d_acc[idx] + val);
#endif
        });
    });
}

void SYCL_FFT_SUFFIX(launch_ihs_finalize)(sycl::queue& q, const IhsAccT* d_acc, const IhsDeconvT* d_slot_deconv, OutScalarT* d_power, const IhsSolveParams& p) {
    const int nout = p.nout;
    const int nfft = p.nfft;
    const int log_nfft = p.log_nfft;
    const int output_shift = p.output_shift;
    const int b_start = p.b_start;
    const int b_count = p.b_count;
    const int Nf = p.Nf;
    const int degree = p.degree;
    const PswfMode mode = p.mode;
    const NormKind norm = p.norm;
    const bool nll_multi = (norm == NormKind::Nll && degree > 1);

    const size_t total_bins = static_cast<size_t>(b_count) * static_cast<size_t>(nout);
    const size_t wg = std::max<size_t>(1, std::min<size_t>(p.wg_size, 256));
    const size_t num_wg = (total_bins + wg - 1) / wg;

    const double base_scale_d = (norm == NormKind::Standard) ? p.standard_scale : p.nll_scale;
    const double inv_n_eff_d = (p.n_eff > 0.0 && std::isfinite(p.n_eff)) ? (1.0 / p.n_eff) : 0.0;
    const double inv_n2_288_d = (inv_n_eff_d * inv_n_eff_d) / 288.0;
#if defined(SYCL_FFT_DOUBLE)
    const double base_scale = base_scale_d;
    const double inv_n_eff = inv_n_eff_d;
    const double inv_n2_288 = inv_n2_288_d;
#elif defined(SYCL_FFT_DF)
    const df_t base_scale = df_from_double(base_scale_d);
    const df_t inv_n_eff = df_from_double(inv_n_eff_d);
    const df_t inv_n2_288 = df_from_double(inv_n2_288_d);
#else
    const float base_scale = static_cast<float>(base_scale_d);
    const float inv_n_eff = static_cast<float>(inv_n_eff_d);
    const float inv_n2_288 = static_cast<float>(inv_n2_288_d);
#endif

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_wg * wg, wg), [=](sycl::nd_item<1> item) {
            const size_t idx = item.get_global_linear_id();
            if (idx >= total_bins) return;

            const int b = static_cast<int>(idx / static_cast<size_t>(nout));
            const int slot_i = static_cast<int>(idx - static_cast<size_t>(b) * static_cast<size_t>(nout));
            const int slot_p = nufft1_compact_idx_to_slot(slot_i, mode);
            const int nat_k = nufft1_slot_to_natural_idx(slot_p, log_nfft, nfft, output_shift);
            const int gf = (b_start + b) * nout + nat_k;
            if (gf >= Nf) return;

#if defined(SYCL_FFT_DOUBLE)
            double val = d_acc[idx];
            if (!nll_multi) {
                const double d = d_slot_deconv[slot_i];
                val *= base_scale * d * d;
            }
            if (norm == NormKind::Standard) {
                if (degree == 1 && (!d_is_finite(val) || val >= 1.0)) val = d_nan();
                d_power[gf] = val;
            } else if (norm == NormKind::Nll && degree == 1) {
                const double v = ihs_wilkie_d(val, inv_n_eff, inv_n2_288);
                d_power[gf] = (d_is_finite(v) && v > 0.0) ? v : 0.0;
            } else {
                const double v = ihs_erlang_d(val, degree);
                d_power[gf] = (v >= 0.0) ? v : 0.0;
            }
#elif defined(SYCL_FFT_DF)
                df_t val = d_acc[idx];
                if (!nll_multi) {
                    const df_t d = d_slot_deconv[slot_i];
                    val = df_mul(val, df_mul(base_scale, df_sqr(d)));
                }
                if (norm == NormKind::Standard) {
                    const bool ge_one = (val.hi > 1.0f || (val.hi == 1.0f && val.lo >= 0.0f));
                    if (degree == 1 && (!f_is_finite(val.hi) || ge_one)) {
                        d_power[gf] = df_make(f_nan(), f_nan());
                    } else {
                        d_power[gf] = val;
                    }
                } else if (norm == NormKind::Nll && degree == 1) {
                    const df_t v = ihs_wilkie_df(val, inv_n_eff, inv_n2_288);
                    d_power[gf] = (f_is_finite(v.hi) && (v.hi > 0.0f || (v.hi == 0.0f && v.lo > 0.0f))) ? v : df_make(0.0f, 0.0f);
                } else {
                    const df_t v = ihs_erlang_df(val, degree);
                    d_power[gf] = (v.hi > 0.0f || (v.hi == 0.0f && v.lo > 0.0f)) ? v : df_make(0.0f, 0.0f);
                }
#else
                float val = d_acc[idx];
                if (!nll_multi) {
                    const float d = d_slot_deconv[slot_i];
                    val *= base_scale * d * d;
                }
                if (norm == NormKind::Standard) {
                    if (degree == 1 && (!f_is_finite(val) || val >= 1.0f)) val = f_nan();
                    d_power[gf] = val;
                } else if (norm == NormKind::Nll && degree == 1) {
                    const float v = ihs_wilkie_f(val, inv_n_eff, inv_n2_288);
                    d_power[gf] = (f_is_finite(v) && v > 0.0f) ? v : 0.0f;
                } else {
                    const float v = ihs_erlang_f(val, degree);
                    d_power[gf] = (v >= 0.0f) ? v : 0.0f;
                }
#endif
        });
    });
}

// Fused single-harmonic (degree == 1) IHS kernel: reads d_z directly, computes |z|^2,
// applies deferred deconvolution + normalization, and writes directly to natural frequency order.
void SYCL_FFT_SUFFIX(launch_ihs_single)(sycl::queue& q, const SolveComplexT* d_z, const IhsDeconvT* d_slot_deconv, OutScalarT* d_power,
                                        const IhsSolveParams& p) {
    const int nout = p.nout;
    const int nfft = p.nfft;
    const int log_nfft = p.log_nfft;
    const int output_shift = p.output_shift;
    const int b_start = p.b_start;
    const int b_count = p.b_count;
    const int Nf = p.Nf;
    const PswfMode mode = p.mode;
    const NormKind norm = p.norm;

    const size_t total_bins = static_cast<size_t>(b_count) * static_cast<size_t>(nout);
    const size_t wg = std::max<size_t>(1, std::min<size_t>(p.wg_size, 256));
    const size_t num_wg = (total_bins + wg - 1) / wg;

    const double base_scale_d = (norm == NormKind::Standard) ? p.standard_scale : p.nll_scale;
    const double inv_n_eff_d = (p.n_eff > 0.0 && std::isfinite(p.n_eff)) ? (1.0 / p.n_eff) : 0.0;
    const double inv_n2_288_d = (inv_n_eff_d * inv_n_eff_d) / 288.0;
#if defined(SYCL_FFT_DOUBLE)
    const double base_scale = base_scale_d;
    const double inv_n_eff = inv_n_eff_d;
    const double inv_n2_288 = inv_n2_288_d;
#elif defined(SYCL_FFT_DF)
    const df_t base_scale = df_from_double(base_scale_d);
    const df_t inv_n_eff = df_from_double(inv_n_eff_d);
    const df_t inv_n2_288 = df_from_double(inv_n2_288_d);
#else
    const float base_scale = static_cast<float>(base_scale_d);
    const float inv_n_eff = static_cast<float>(inv_n_eff_d);
    const float inv_n2_288 = static_cast<float>(inv_n2_288_d);
#endif

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_wg * wg, wg), [=](sycl::nd_item<1> item) {
            const size_t idx = item.get_global_linear_id();
            if (idx >= total_bins) return;

            const int b = static_cast<int>(idx / static_cast<size_t>(nout));
            const int slot_i = static_cast<int>(idx - static_cast<size_t>(b) * static_cast<size_t>(nout));
            const int slot_p = nufft1_compact_idx_to_slot(slot_i, mode);
            const int nat_k = nufft1_slot_to_natural_idx(slot_p, log_nfft, nfft, output_shift);
            const int gf = (b_start + b) * nout + nat_k;
            if (gf >= Nf) return;

#if defined(SYCL_FFT_DOUBLE)
            const Complex z = d_z[idx];
            const double d = d_slot_deconv[slot_i];
            double val = (z.x() * z.x() + z.y() * z.y()) * (base_scale * d * d);
            if (norm == NormKind::Standard) {
                if (!d_is_finite(val) || val >= 1.0) val = d_nan();
                d_power[gf] = val;
            } else if (norm == NormKind::Nll) {
                const double v = ihs_wilkie_d(val, inv_n_eff, inv_n2_288);
                d_power[gf] = (d_is_finite(v) && v > 0.0) ? v : 0.0;
            } else {
                d_power[gf] = (d_is_finite(val) && val > 0.0) ? val : 0.0;
            }
#elif defined(SYCL_FFT_DF)
                const Complexdf z = d_z[idx];
                const df_t d = d_slot_deconv[slot_i];
                df_t val = df_mul(df_add(df_sqr(z.re), df_sqr(z.im)), df_mul(base_scale, df_sqr(d)));
                if (norm == NormKind::Standard) {
                    const bool ge_one = (val.hi > 1.0f || (val.hi == 1.0f && val.lo >= 0.0f));
                    d_power[gf] = (!f_is_finite(val.hi) || ge_one) ? df_make(f_nan(), f_nan()) : val;
                } else if (norm == NormKind::Nll) {
                    const df_t v = ihs_wilkie_df(val, inv_n_eff, inv_n2_288);
                    d_power[gf] = (f_is_finite(v.hi) && (v.hi > 0.0f || (v.hi == 0.0f && v.lo > 0.0f))) ? v : df_make(0.0f, 0.0f);
                } else {
                    d_power[gf] = (f_is_finite(val.hi) && (val.hi > 0.0f || (val.hi == 0.0f && val.lo > 0.0f))) ? val : df_make(0.0f, 0.0f);
                }
#elif defined(SYCL_FFT_HALF)
                const uint32_t u = sycl_fft_bit_cast<uint32_t>(d_z[idx]);
                const f16x2_t pair = SYCL_FFT_BIT_CAST(f16x2_t, u);
                const float re = static_cast<float>(f16x2_lo(pair));
                const float im = static_cast<float>(f16x2_hi(pair));
                const float d = d_slot_deconv[slot_i];
                float val = (re * re + im * im) * (base_scale * d * d);
                if (norm == NormKind::Standard) {
                    if (!f_is_finite(val) || val >= 1.0f) val = f_nan();
                    d_power[gf] = val;
                } else if (norm == NormKind::Nll) {
                    const float v = ihs_wilkie_f(val, inv_n_eff, inv_n2_288);
                    d_power[gf] = (f_is_finite(v) && v > 0.0f) ? v : 0.0f;
                } else {
                    d_power[gf] = (f_is_finite(val) && val > 0.0f) ? val : 0.0f;
                }
#else
                const Complexf z = d_z[idx];
                const float d = d_slot_deconv[slot_i];
                float val = (z.x() * z.x() + z.y() * z.y()) * (base_scale * d * d);
                if (norm == NormKind::Standard) {
                    if (!f_is_finite(val) || val >= 1.0f) val = f_nan();
                    d_power[gf] = val;
                } else if (norm == NormKind::Nll) {
                    const float v = ihs_wilkie_f(val, inv_n_eff, inv_n2_288);
                    d_power[gf] = (f_is_finite(v) && v > 0.0f) ? v : 0.0f;
                } else {
                    d_power[gf] = (f_is_finite(val) && val > 0.0f) ? val : 0.0f;
                }
#endif
        });
    });
}

}  // namespace detail
}  // namespace sycl_linalg
