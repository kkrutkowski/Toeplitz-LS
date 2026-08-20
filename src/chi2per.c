#include <linalg.h>
#define NANOFFT_NEEDS_INTERNAL_VEC
#include <math.h>
#include <nanofft_precision.h>
#include <nufft1.h>
#include <scaling.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <tls.h>
#include <utils.h>

/* Return codes from the public fastchi2 entry points. */
enum { CHI2PER_OK = 0, CHI2PER_ERR_ARGUMENT = -1, CHI2PER_ERR_BACKEND = -2, CHI2PER_ERR_ALLOC = -3, CHI2PER_ERR_DEGENERATE = -4, CHI2PER_ERR_SOLVER = -5 };

enum { CHI2PER_BACKEND_PSWF43 = 1, CHI2PER_BACKEND_PSWF21 = 2, CHI2PER_BACKEND_LRA = 3 };

enum { CHI2PER_SOLVER_LEVINSON = 1, CHI2PER_SOLVER_ZOHAR = 2, CHI2PER_SOLVER_BAREISS = 3, CHI2PER_SOLVER_LDLT = 4, CHI2PER_SOLVER_SVD = 5 };

static void *checked_malloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return NULL;
    if (count > SIZE_MAX / size) return NULL;
    return malloc(count * size);
}

static void *checked_aligned_malloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return NULL;
    if (count > SIZE_MAX / size) return NULL;

    size_t bytes = count * size;
    size_t align = 64;
    size_t padded = (bytes + align - 1) & ~(align - 1);
    return aligned_alloc(align, padded);
}

#if defined(DOUBLE_DOUBLE)
#    define CHI2_PREFIX(name) tlsdd_##name
#    define TIME_INPUT_T dd_t
#    define NUFFT_INPUT_T dd_t
#    define NUFFT_LRA_INIT tlsdd_nufft_lra_initialize
#    define NUFFT_LRA_PRE tlsdd_nufft_lra_precompute
#    define NUFFT_LRA_EXEC tlsdd_nufft_lra_execute
#    define NUFFT_LRA_FREE tlsdd_nufft_free_lra_plan
#    define NUFFT_PSWF_INIT tlsdd_nufft_pswf_initialize
#    define NUFFT_PSWF_PRE tlsdd_nufft_pswf_precompute
#    define NUFFT_PSWF_EXEC tlsdd_nufft_pswf_execute
#    define NUFFT_PSWF_FREE tlsdd_nufft_free_pswf_plan
#    define LRA_PLAN_T tlsdd_nufft_lra_plan
#    define PSWF_PLAN_T tlsdd_nufft_pswf_plan
#    define SOLVE_LDLT tlsdd_solve_ldlt
#    define SOLVE_LEVINSON tlsdd_solve_levinson
#    define SOLVE_BAREISS tlsdd_solve_bareiss
#    define SOLVE_ZOHAR tlsdd_solve_zohar
#    define SOLVE_SVD tlsdd_solve_svd
#    define NUFFT_RANK 27
#    define NUFFT_W21 32
#    define NUFFT_W43 36
#    define CHI2_ALPHA DD_ALPHA
#    define CHI2_LRA_BETA DD_LRA_BETA
#    define CHI2_LRA_GAMMA DD_LRA_GAMMA
#    define CHI2_PSWF21_BETA DD_PSWF21_BETA
#    define CHI2_PSWF21_GAMMA DD_PSWF21_GAMMA
#    define CHI2_PSWF43_BETA DD_PSWF43_BETA
#    define CHI2_PSWF43_GAMMA DD_PSWF43_GAMMA
#    define COND_SINGULARITY_THRESHOLD_LEVINSON 1e22
#    define COND_SINGULARITY_THRESHOLD_ZOHAR 1e22
#    define COND_SINGULARITY_THRESHOLD_BAREISS 1e22
#    define COND_SINGULARITY_THRESHOLD_LDLT 1e10
#    define COND_SINGULARITY_THRESHOLD_SVD 1e16
static inline FLOAT time_to_float(TIME_INPUT_T x) { return x; }
static inline NUFFT_INPUT_T time_to_nufft_input(TIME_INPUT_T x) { return x; }
#else
#    define TIME_INPUT_T double
#    define NUFFT_INPUT_T double
#    if defined(DOUBLE)
#        define CHI2_PREFIX(name) tls_##name
#        define NUFFT_LRA_INIT tls_nufft_lra_initialize
#        define NUFFT_LRA_PRE tls_nufft_lra_precompute
#        define NUFFT_LRA_EXEC tls_nufft_lra_execute
#        define NUFFT_LRA_FREE tls_nufft_free_lra_plan
#        define NUFFT_PSWF_INIT tls_nufft_pswf_initialize
#        define NUFFT_PSWF_PRE tls_nufft_pswf_precompute
#        define NUFFT_PSWF_EXEC tls_nufft_pswf_execute
#        define NUFFT_PSWF_FREE tls_nufft_free_pswf_plan
#        define LRA_PLAN_T tls_nufft_lra_plan
#        define PSWF_PLAN_T tls_nufft_pswf_plan
#        define SOLVE_LDLT tls_solve_ldlt
#        define SOLVE_LEVINSON tls_solve_levinson
#        define SOLVE_BAREISS tls_solve_bareiss
#        define SOLVE_ZOHAR tls_solve_zohar
#        define SOLVE_SVD tls_solve_svd
#        define NUFFT_RANK 16
#        define NUFFT_W21 16
#        define NUFFT_W43 18
#        define CHI2_ALPHA D_ALPHA
#        define CHI2_LRA_BETA D_LRA_BETA
#        define CHI2_LRA_GAMMA D_LRA_GAMMA
#        define CHI2_PSWF21_BETA D_PSWF21_BETA
#        define CHI2_PSWF21_GAMMA D_PSWF21_GAMMA
#        define CHI2_PSWF43_BETA D_PSWF43_BETA
#        define CHI2_PSWF43_GAMMA D_PSWF43_GAMMA
#        define COND_SINGULARITY_THRESHOLD_LEVINSON 1e11
#        define COND_SINGULARITY_THRESHOLD_ZOHAR 1e11
#        define COND_SINGULARITY_THRESHOLD_BAREISS 1e11
#        define COND_SINGULARITY_THRESHOLD_LDLT 1e5
#        define COND_SINGULARITY_THRESHOLD_SVD 1e8
#    else
#        define CHI2_PREFIX(name) tlsf_##name
#        define NUFFT_LRA_INIT tlsf_nufft_lra_initialize
#        define NUFFT_LRA_PRE tlsf_nufft_lra_precompute
#        define NUFFT_LRA_EXEC tlsf_nufft_lra_execute
#        define NUFFT_LRA_FREE tlsf_nufft_free_lra_plan
#        define NUFFT_PSWF_INIT tlsf_nufft_pswf_initialize
#        define NUFFT_PSWF_PRE tlsf_nufft_pswf_precompute
#        define NUFFT_PSWF_EXEC tlsf_nufft_pswf_execute
#        define NUFFT_PSWF_FREE tlsf_nufft_free_pswf_plan
#        define LRA_PLAN_T tlsf_nufft_lra_plan
#        define PSWF_PLAN_T tlsf_nufft_pswf_plan
#        define SOLVE_LDLT tlsf_solve_ldlt
#        define SOLVE_LEVINSON tlsf_solve_levinson
#        define SOLVE_BAREISS tlsf_solve_bareiss
#        define SOLVE_ZOHAR tlsf_solve_zohar
#        define SOLVE_SVD tlsf_solve_svd
#        define NUFFT_RANK 9
#        define NUFFT_W21 8
#        define NUFFT_W43 9
#        define CHI2_ALPHA F_ALPHA
#        define CHI2_LRA_BETA F_LRA_BETA
#        define CHI2_LRA_GAMMA F_LRA_GAMMA
#        define CHI2_PSWF21_BETA F_PSWF21_BETA
#        define CHI2_PSWF21_GAMMA F_PSWF21_GAMMA
#        define CHI2_PSWF43_BETA F_PSWF43_BETA
#        define CHI2_PSWF43_GAMMA F_PSWF43_GAMMA
#        define COND_SINGULARITY_THRESHOLD_LEVINSON 5e4
#        define COND_SINGULARITY_THRESHOLD_ZOHAR 5e4
#        define COND_SINGULARITY_THRESHOLD_BAREISS 5e4
#        define COND_SINGULARITY_THRESHOLD_LDLT 1e2
#        define COND_SINGULARITY_THRESHOLD_SVD 1e3
#    endif
static inline FLOAT time_to_float(TIME_INPUT_T x) { return FCAST(x); }
static inline NUFFT_INPUT_T time_to_nufft_input(TIME_INPUT_T x) { return x; }
#endif

static inline int double_is_nan_bits(double value) {
    union {
        double f;
        uint64_t u;
    } bits = {value};
    return (bits.u & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000);
}

static inline int float_is_nan_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return (bits.u & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
}

static const double condition_singularity_thresholds[] = {
    0.0,
    COND_SINGULARITY_THRESHOLD_LEVINSON,
    COND_SINGULARITY_THRESHOLD_ZOHAR,
    COND_SINGULARITY_THRESHOLD_BAREISS,
    COND_SINGULARITY_THRESHOLD_LDLT,
    COND_SINGULARITY_THRESHOLD_SVD,
};

static inline double condition_singularity_threshold(int solver) {
    if (solver < CHI2PER_SOLVER_LEVINSON || solver > CHI2PER_SOLVER_SVD) return 0.0;
    return condition_singularity_thresholds[solver];
}

static inline int condition_bound_is_singular(FLOAT bound, int solver) {
    double threshold = condition_singularity_threshold(solver);
#if defined(DOUBLE_DOUBLE)
    double value = TO_DOUBLE(bound);
    return double_is_nan_bits(bound.hi) || double_is_nan_bits(bound.lo) || value < 0.0 || value > threshold;
#elif defined(DOUBLE)
    return double_is_nan_bits(bound) || bound < 0.0 || bound > threshold;
#else
    return float_is_nan_bits(bound) || bound < 0.0f || (double)bound > threshold;
#endif
}

static inline int condition_bound_is_invalid(FLOAT bound) {
#if defined(DOUBLE_DOUBLE)
    double value = TO_DOUBLE(bound);
    return double_is_nan_bits(bound.hi) || double_is_nan_bits(bound.lo) || value < 0.0;
#elif defined(DOUBLE)
    return double_is_nan_bits(bound) || bound < 0.0;
#else
    return float_is_nan_bits(bound) || bound < 0.0f;
#endif
}

static inline int double_is_finite_bits(double value) {
    union {
        double f;
        uint64_t u;
    } bits = {value};
    return (bits.u & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

static inline int float_is_finite_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return (bits.u & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

typedef struct {
    double var_S;
    double mean_S;
    double k_eff;     /* shape parameter a */
    double theta_eff; /* scale parameter theta */
    double lgamma_k;  /* ln(Gamma(k_eff)) cached */
    double sum_N_eff; /* sum_{j=1}^{2d} (M-j)^2 / M */
    double inv_2d;    /* 1.0 / (2.0 * d) */
} cybenko_model_t;

static inline cybenko_model_t cybenko_model_init(int d, int M) {
    cybenko_model_t m;
    double d_dbl = (double)d;
    double M_dbl = (double)M;
    double inv_M = 1.0 / M_dbl;
    double inv_M2 = inv_M * inv_M;

    /* Exact 3rd-order Edgeworth expansion for Var(S) */
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

static inline double cybenko_kappa_to_S(double kappa, const cybenko_model_t *m) {
    if (kappa <= 1.0 || !double_is_finite_bits(kappa)) return 0.0;
    if (kappa > 5000.0) kappa = 5000.0;
    double pow_k = pow(kappa, m->inv_2d);
    double k_val = (pow_k - 1.0) / (pow_k + 1.0);
    double k2 = k_val * k_val;
    return m->sum_N_eff * k2;
}

#if defined(DOUBLE_DOUBLE)
static inline dd_t cybenko_kappa_to_S_dd(dd_t kappa, const cybenko_model_t *m) {
    double kh = kappa.hi;
    if (kh <= 1.0 || !isfinite(kh)) return dd_make(0.0, 0.0);
    if (kh > 5000.0) kh = 5000.0;
    double pow_k = pow(kh, m->inv_2d);
    double k_val = (pow_k - 1.0) / (pow_k + 1.0);
    double k2 = k_val * k_val;
    return dd_mul(dd_make(m->sum_N_eff, 0.0), dd_make(k2, 0.0));
}

static inline dd_t nll_exact_pochhammer_dd(int d, dd_t r2, dd_t b) {
    if (r2.hi <= 0.0 || !isfinite(r2.hi)) return dd_make(0.0, 0.0);
    if (r2.hi > 0.999999999999999) r2 = dd_make(0.999999999999999, 0.0);

    dd_t one = dd_make(1.0, 0.0);
    dd_t one_minus_r2 = dd_sub(one, r2);
    dd_t ln_one_minus_r2 = dd_log(one_minus_r2);
    dd_t term0 = dd_mul(dd_make(-b.hi, -b.lo), ln_one_minus_r2);
    if (d <= 1) {
        return (term0.hi > 0.0 && isfinite(term0.hi)) ? term0 : dd_make(0.0, 0.0);
    }

    dd_t c = one;
    dd_t s_minus_1 = dd_make(0.0, 0.0);
    for (int k = 1; k < d; ++k) {
        dd_t num = dd_add(b, dd_make((double)(k - 1), 0.0));
        dd_t factor = dd_div(num, dd_make((double)k, 0.0));
        c = dd_mul(dd_mul(c, factor), r2);
        s_minus_1 = dd_add(s_minus_1, c);
    }

    dd_t s = dd_add(one, s_minus_1);
    dd_t ln_s = dd_log(s);
    dd_t nll = dd_sub(term0, ln_s);
    return (nll.hi > 0.0 && isfinite(nll.hi)) ? nll : dd_make(0.0, 0.0);
}

static inline dd_t nll_cybenko_dd(dd_t kappa, const cybenko_model_t *m, int d) {
    dd_t s = cybenko_kappa_to_S_dd(kappa, m);
    if (s.hi <= 0.0 || !isfinite(s.hi)) return dd_make(0.0, 0.0);

    double k_eff = m->k_eff;
    double th_eff = m->theta_eff;
    double lg_k = m->lgamma_k;

    if (d <= 2) {
        dd_t z = dd_div(s, dd_make(th_eff, 0.0));
        int order = 2 * d;
        double delta_val = k_eff - (double)order;

        dd_t c = dd_make(1.0, 0.0);
        dd_t poly = dd_make(1.0, 0.0);
        dd_t d_poly = dd_make(0.0, 0.0);

        for (int k = 1; k < order; ++k) {
            dd_t factor = dd_make(1.0 / (double)k, 0.0);
            c = dd_mul(dd_mul(c, z), factor);
            poly = dd_add(poly, c);
            if (fabs(delta_val) > 1.0e-5) {
                double H_k = 0.0;
                for (int i = 1; i <= k; ++i) H_k += 1.0 / (double)i;
                d_poly = dd_sub(d_poly, dd_mul(c, dd_make(H_k, 0.0)));
            }
        }

        dd_t ln_poly = dd_log(poly);
        dd_t nll = dd_sub(z, ln_poly);
        if (fabs(delta_val) > 1.0e-5) {
            dd_t ln_z = dd_log(z);
            dd_t term = dd_add(ln_z, dd_div(d_poly, poly));
            nll = dd_sub(nll, dd_mul(dd_make(delta_val, 0.0), term));
        }
        return (nll.hi > 0.0 && isfinite(nll.hi)) ? nll : dd_make(0.0, 0.0);
    } else {
        double s_d = s.hi;
        double a = k_eff;
        double z_d = s_d / th_eff;
        if (z_d < a + 3.0) {
            double c = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 60; ++k) {
                c *= z_d / (a + (double)(k - 1));
                sum += c;
                if (c < 1.0e-14 * sum) break;
            }
            double lg_inc = a * log(z_d) - z_d - log(a) + log(sum);
            double nll = lg_k - lg_inc;
            return dd_make((nll > 0.0 && isfinite(nll)) ? nll : 0.0, 0.0);
        } else {
            double inv_z = 1.0 / z_d;
            double t1 = (a - 1.0) * inv_z;
            double t2 = t1 * ((a - 2.0) * inv_z);
            double t3 = t2 * ((a - 3.0) * inv_z);
            double t4 = t3 * ((a - 4.0) * inv_z);
            double poly_val = 1.0 + t1 + t2 + t3 + t4;
            double nll = z_d - (a - 1.0) * log(z_d) + lg_k - log(poly_val);
            return dd_make((nll > 0.0 && isfinite(nll)) ? nll : 0.0, 0.0);
        }
    }
}

static inline dd_t compute_bayes_laplace_dd(dd_t delta) {
    static const dd_t ln2 = {0.693147180559945309417, 2.31904681384629955841e-17};
    static const dd_t half = {0.5, 0.0};
    static const dd_t one = {1.0, 0.0};

    if (delta.hi >= 0.0) {
        return dd_add(delta, ln2);
    } else {
        dd_t exp_delta = dd_exp(delta);
        dd_t inner = dd_sub(one, dd_mul(half, exp_delta));
        if (inner.hi <= 0.0) return dd_make(0.0, 0.0);
        dd_t log_inner = dd_log(inner);
        return dd_make(-log_inner.hi, -log_inner.lo);
    }
}

static inline FLOAT normalize_power_entry(FLOAT dot, FLOAT chi2_ref, FLOAT condition_bound, int normalization, int degree, int M, const cybenko_model_t *cyb, FLOAT b) {
    if (normalization == TLS_NORM_STANDARD) {
        return DIV(dot, chi2_ref);
    } else if (normalization == TLS_NORM_ASYMPTOTIC) {
        double d1 = 2.0 * (double)degree + 1.0;
        double d2 = (double)M - 1.0 - d1;
        dd_t residual = dd_sub(chi2_ref, dot);
        if (residual.hi <= 1e-32) residual = dd_make(1e-32, 0.0);
        return dd_div(dd_mul(dd_make(d2, 0.0), dot), dd_mul(dd_make(d1, 0.0), residual));
    } else if (normalization == TLS_NORM_NLL) {
        dd_t r2 = DIV(dot, chi2_ref);
        return nll_exact_pochhammer_dd(degree, r2, b);
    } else { /* TLS_NORM_BAYES */
        dd_t r2 = DIV(dot, chi2_ref);
        dd_t nll_r2 = nll_exact_pochhammer_dd(degree, r2, b);
        dd_t nll_cond = nll_cybenko_dd(condition_bound, cyb, degree);
        dd_t delta = dd_sub(nll_r2, nll_cond);
        return compute_bayes_laplace_dd(delta);
    }
}
#elif defined(DOUBLE)
static inline double nll_exact_pochhammer_scalar_d(int d, double r2, double b) {
    if (!double_is_finite_bits(r2) || r2 <= 0.0) return 0.0;
    if (r2 > 0.9999999999999) r2 = 0.9999999999999;
    double term0 = -b * log1p(-r2);
    if (d <= 1) return (term0 > 0.0 && double_is_finite_bits(term0)) ? term0 : 0.0;
    double c = 1.0;
    double s_minus_1 = 0.0;
    for (int k = 1; k < d; ++k) {
        c *= (b + (double)(k - 1)) / (double)k * r2;
        s_minus_1 += c;
    }
    double ln_s = log1p(s_minus_1);
    double nll = term0 - ln_s;
    return (nll > 0.0 && double_is_finite_bits(nll)) ? nll : 0.0;
}

static inline double nll_cybenko_scalar_d(double kappa, const cybenko_model_t *m, int d) {
    double s = cybenko_kappa_to_S(kappa, m);
    if (s <= 0.0 || !double_is_finite_bits(s)) return 0.0;
    double k_eff = m->k_eff;
    double th_eff = m->theta_eff;
    double lg_k = m->lgamma_k;

    if (d <= 2) {
        double z = s / th_eff;
        int order = 2 * d;
        double delta = k_eff - (double)order;
        double c = 1.0;
        double poly = 1.0;
        double d_poly = 0.0;
        for (int k = 1; k < order; ++k) {
            c *= (z / (double)k);
            poly += c;
            if (fabs(delta) > 1.0e-5) {
                double H_k = 0.0;
                for (int i = 1; i <= k; ++i) H_k += 1.0 / (double)i;
                d_poly += c * (-H_k);
            }
        }
        double nll = z - log(poly);
        if (fabs(delta) > 1.0e-5) {
            nll -= delta * (log(z) + d_poly / poly);
        }
        return (nll > 0.0 && double_is_finite_bits(nll)) ? nll : 0.0;
    } else {
        double a = k_eff;
        double z = s / th_eff;
        if (z < a + 3.0) {
            double c = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 60; ++k) {
                c *= z / (a + (double)(k - 1));
                sum += c;
                if (c < 1.0e-14 * sum) break;
            }
            double lg_inc = a * log(z) - z - log(a) + log(sum);
            double nll = lg_k - lg_inc;
            return (nll > 0.0 && double_is_finite_bits(nll)) ? nll : 0.0;
        } else {
            double inv_z = 1.0 / z;
            double t1 = (a - 1.0) * inv_z;
            double t2 = t1 * ((a - 2.0) * inv_z);
            double t3 = t2 * ((a - 3.0) * inv_z);
            double t4 = t3 * ((a - 4.0) * inv_z);
            double poly = 1.0 + t1 + t2 + t3 + t4;
            double nll = z - (a - 1.0) * log(z) + lg_k - log(poly);
            return (nll > 0.0 && double_is_finite_bits(nll)) ? nll : 0.0;
        }
    }
}

static inline double compute_bayes_laplace_d(double delta) {
    if (delta >= 0.0) {
        return delta + 0.693147180559945309417; // ln(2)
    } else {
        double exp_delta = exp(delta);
        double inner = 1.0 - 0.5 * exp_delta;
        if (inner <= 0.0) return 0.0;
        return -log(inner);
    }
}

static inline FLOAT normalize_power_entry(FLOAT dot, FLOAT chi2_ref, FLOAT condition_bound, int normalization, int degree, int M, const cybenko_model_t *cyb, FLOAT b) {
    if (normalization == TLS_NORM_STANDARD) {
        return dot / chi2_ref;
    } else if (normalization == TLS_NORM_ASYMPTOTIC) {
        double d1 = 2.0 * (double)degree + 1.0;
        double d2 = (double)M - 1.0 - d1;
        double residual = chi2_ref - dot;
        if (residual < 1e-32) residual = 1e-32;
        return (d2 * dot) / (d1 * residual);
    } else if (normalization == TLS_NORM_NLL) {
        double r2 = dot / chi2_ref;
        return nll_exact_pochhammer_scalar_d(degree, r2, b);
    } else { /* TLS_NORM_BAYES */
        double r2 = dot / chi2_ref;
        double nll_r2 = nll_exact_pochhammer_scalar_d(degree, r2, b);
        double nll_cond = nll_cybenko_scalar_d(condition_bound, cyb, degree);
        double delta = nll_r2 - nll_cond;
        return compute_bayes_laplace_d(delta);
    }
}
#else
static inline float nll_exact_pochhammer_scalar_f(int d, float r2, float b) {
    if (!float_is_finite_bits(r2) || r2 <= 0.0f) return 0.0f;
    if (r2 > 0.9999999f) r2 = 0.9999999f;
    float term0 = -b * log1pf(-r2);
    if (d <= 1) return (term0 > 0.0f && float_is_finite_bits(term0)) ? term0 : 0.0f;
    double c = 1.0;
    double s_minus_1 = 0.0;
    double b_d = (double)b;
    double r2_d = (double)r2;
    for (int k = 1; k < d; ++k) {
        c *= (b_d + (double)(k - 1)) / (double)k * r2_d;
        s_minus_1 += c;
    }
    double ln_s = log1p(s_minus_1);
    double nll = (double)term0 - ln_s;
    return (nll > 0.0 && double_is_finite_bits(nll)) ? (float)nll : 0.0f;
}

static inline float nll_cybenko_scalar_f(float kappa, const cybenko_model_t *m, int d) {
    double s = cybenko_kappa_to_S((double)kappa, m);
    if (s <= 0.0 || !double_is_finite_bits(s)) return 0.0f;
    float k_eff_f = (float)m->k_eff;
    float th_eff_f = (float)m->theta_eff;
    float lg_k_f = (float)m->lgamma_k;
    float s_f = (float)s;

    if (d <= 2) {
        float z = s_f / th_eff_f;
        int order = 2 * d;
        float delta = k_eff_f - (float)order;
        float c = 1.0f;
        float poly = 1.0f;
        float d_poly = 0.0f;
        for (int k = 1; k < order; ++k) {
            c *= (z / (float)k);
            poly += c;
            if (fabsf(delta) > 1.0e-5f) {
                float H_k = 0.0f;
                for (int i = 1; i <= k; ++i) H_k += 1.0f / (float)i;
                d_poly += c * (-H_k);
            }
        }
        float nll = z - logf(poly);
        if (fabsf(delta) > 1.0e-5f) {
            nll -= delta * (logf(z) + d_poly / poly);
        }
        return (nll > 0.0f && float_is_finite_bits(nll)) ? nll : 0.0f;
    } else {
        float a = k_eff_f;
        float z = s_f / th_eff_f;
        if (z < a + 3.0f) {
            double c = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 60; ++k) {
                c *= (double)z / ((double)a + (double)(k - 1));
                sum += c;
                if (c < 1.0e-7 * sum) break;
            }
            double lg_inc = (double)a * log((double)z) - (double)z - log((double)a) + log(sum);
            double nll = (double)lg_k_f - lg_inc;
            return (nll > 0.0 && double_is_finite_bits(nll)) ? (float)nll : 0.0f;
        } else {
            float inv_z = 1.0f / z;
            float t1 = (a - 1.0f) * inv_z;
            float t2 = t1 * ((a - 2.0f) * inv_z);
            float t3 = t2 * ((a - 3.0f) * inv_z);
            float t4 = t3 * ((a - 4.0f) * inv_z);
            float poly = 1.0f + t1 + t2 + t3 + t4;
            float nll = z - (a - 1.0f) * logf(z) + lg_k_f - logf(poly);
            return (nll > 0.0f && float_is_finite_bits(nll)) ? nll : 0.0f;
        }
    }
}

static inline float compute_bayes_laplace_f(float delta) {
    if (delta >= 0.0f) {
        return delta + 0.6931471805599453f; // ln(2)
    } else {
        float exp_delta = expf(delta);
        float inner = 1.0f - 0.5f * exp_delta;
        if (inner <= 0.0f) return 0.0f;
        return -logf(inner);
    }
}

static inline FLOAT normalize_power_entry(FLOAT dot, FLOAT chi2_ref, FLOAT condition_bound, int normalization, int degree, int M, const cybenko_model_t *cyb, FLOAT b) {
    if (normalization == TLS_NORM_STANDARD) {
        return dot / chi2_ref;
    } else if (normalization == TLS_NORM_ASYMPTOTIC) {
        float d1 = 2.0f * (float)degree + 1.0f;
        float d2 = (float)M - 1.0f - d1;
        float residual = chi2_ref - dot;
        if (residual < 1e-32f) residual = 1e-32f;
        return (d2 * dot) / (d1 * residual);
    } else if (normalization == TLS_NORM_NLL) {
        float r2 = dot / chi2_ref;
        return nll_exact_pochhammer_scalar_f(degree, r2, b);
    } else { /* TLS_NORM_BAYES */
        float r2 = dot / chi2_ref;
        float nll_r2 = nll_exact_pochhammer_scalar_f(degree, r2, b);
        float nll_cond = nll_cybenko_scalar_f(condition_bound, cyb, degree);
        float delta = nll_r2 - nll_cond;
        return compute_bayes_laplace_f(delta);
    }
}
#endif

static inline int real_term_harmonic(int term) { return term == 0 ? 0 : (term + 1) / 2; }

static inline int real_term_is_sin(int term) { return term > 0 && (term & 1); }

static inline FLOAT signed_float_value(FLOAT value, int sign) {
    if (sign < 0) return NEG(value);
    if (sign > 0) return value;
    return FCAST(0.0);
}

static inline FLOAT real_gram_value(const FLOAT *Sw, const FLOAT *Cw, int N, int idx, int row, int col) {
    int m = real_term_harmonic(col);
    int n = real_term_harmonic(row);
    int col_sin = real_term_is_sin(col);
    int row_sin = real_term_is_sin(row);
    int diff = abs(m - n);
    int sum = m + n;
    FLOAT half = FCAST(0.5);

    if (col_sin && row_sin) {
        return MUL(half, SUB(Cw[(size_t)diff * N + idx], Cw[(size_t)sum * N + idx]));
    }
    if (!col_sin && !row_sin) {
        return MUL(half, ADD(Cw[(size_t)diff * N + idx], Cw[(size_t)sum * N + idx]));
    }
    if (col_sin) {
        return MUL(half, ADD(signed_float_value(Sw[(size_t)diff * N + idx], m - n), Sw[(size_t)sum * N + idx]));
    }
    return MUL(half, ADD(signed_float_value(Sw[(size_t)diff * N + idx], n - m), Sw[(size_t)sum * N + idx]));
}

static inline FLOAT real_rhs_value(const FLOAT *Syw, const FLOAT *Cyw, int N, int idx, int term) {
    int h = real_term_harmonic(term);
    if (real_term_is_sin(term)) return Syw[(size_t)h * N + idx];
    return Cyw[(size_t)h * N + idx];
}

static int execute_nufft_block(void *plan, int backend, const FLOAT *src_r, const FLOAT *src_i, FLOAT *out_r, FLOAT *out_i, int freq_factor) {
    if (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21) {
        NUFFT_PSWF_EXEC((PSWF_PLAN_T *)plan, src_r, src_i, out_r, out_i, freq_factor);
    } else if (backend == CHI2PER_BACKEND_LRA) {
        NUFFT_LRA_EXEC((const LRA_PLAN_T *)plan, src_r, src_i, out_r, out_i, freq_factor);
    } else {
        return CHI2PER_ERR_BACKEND;
    }
    return CHI2PER_OK;
}

static void compute_source_phase(NUFFT_INPUT_T tm, double qf0, FLOAT *phase_r, FLOAT *phase_i) {
#if defined(DOUBLE_DOUBLE)
    FLOAT phase = MUL(tm, FCAST(qf0));
    *phase_r = M_COS2PI(phase);
    *phase_i = M_SIN2PI(phase);
#else
    double phase = (double)tm * qf0;
    *phase_r = FCAST(cos2pi(phase));
    *phase_i = FCAST(sin2pi(phase));
#endif
}

static void compute_twiddle_delta(int backend, NUFFT_INPUT_T tm, double qdf, double advance, FLOAT *delta_r, FLOAT *delta_i) {
    (void)backend;
#if defined(DOUBLE_DOUBLE)
    if (backend == CHI2PER_BACKEND_PSWF43) {
        FLOAT phase_delta = MUL(MUL(tm, FCAST(qdf)), FCAST(advance / 3.0));
        FLOAT c = M_COS2PI(phase_delta);
        FLOAT s = M_SIN2PI(phase_delta);
        nanofft_triple_angle(c, s, delta_r, delta_i);
        return;
    }
#elif defined(DOUBLE)
    if (backend == CHI2PER_BACKEND_PSWF43) {
        double phase_delta = tm * qdf * (advance / 3.0);
        double c = cos2pi(phase_delta);
        double s = sin2pi(phase_delta);
        nanofft_triple_angle(c, s, delta_r, delta_i);
        return;
    }
#endif

#if defined(DOUBLE_DOUBLE)
    FLOAT phase_delta = MUL(MUL(tm, FCAST(qdf)), FCAST(advance));
    *delta_r = M_COS2PI(phase_delta);
    *delta_i = M_SIN2PI(phase_delta);
#else
    double phase_delta = (double)tm * qdf * advance;
    *delta_r = FCAST(cos2pi(phase_delta));
    *delta_i = FCAST(sin2pi(phase_delta));
#endif
}

static void rotate_source_level(FLOAT *src_r, FLOAT *src_i, const FLOAT *delta_r, const FLOAT *delta_i, int M) {
    for (int m = 0; m < M; ++m) {
        FLOAT yr = src_r[m];
        FLOAT yi = src_i[m];
        FLOAT dr = delta_r[m];
        FLOAT di = delta_i[m];
        src_r[m] = SUB(MUL(yr, dr), MUL(yi, di));
        src_i[m] = ADD(MUL(yr, di), MUL(yi, dr));
    }
}

static void copy_twiddle_level_down(FLOAT *src_r_levels, FLOAT *src_i_levels, int M, int src_level) {
    for (int level = src_level; level > 0; --level) {
        size_t dst = (size_t)(level - 1) * (size_t)M;
        size_t src = (size_t)level * (size_t)M;
        memcpy(src_r_levels + dst, src_r_levels + src, (size_t)M * sizeof(FLOAT));
        memcpy(src_i_levels + dst, src_i_levels + src, (size_t)M * sizeof(FLOAT));
    }
}

static int compute_trig_sums(const NUFFT_INPUT_T *x, const FLOAT *h, int M, double f0, double df, int N, int max_factor, int block, int backend, void *plan,
                             FLOAT *S, FLOAT *C, FLOAT *src_r_levels, FLOAT *src_i_levels, FLOAT *delta_r_levels, FLOAT *delta_i_levels, int ladder_levels,
                             FLOAT *out_r, FLOAT *out_i) {
    size_t num_blocks = ((size_t)N + (size_t)block - 1) / (size_t)block;

    for (int q = 1; q <= max_factor; ++q) {
        FLOAT qf0 = FCAST((double)q * f0);
        double qdf = (double)q * df;

        for (int m = 0; m < M; ++m) {
            NUFFT_INPUT_T xm = x[m];
            FLOAT c0;
            FLOAT s0;
            compute_source_phase(xm, TO_DOUBLE(qf0), &c0, &s0);
            FLOAT src_r0 = MUL(h[m], c0);
            FLOAT src_i0 = MUL(h[m], s0);

            for (int level = 0; level < ladder_levels; ++level) {
                size_t idx = (size_t)level * (size_t)M + (size_t)m;
                src_r_levels[idx] = src_r0;
                src_i_levels[idx] = src_i0;
                double advance = tls_twiddle_ladder_advance(block, level);
                compute_twiddle_delta(backend, xm, qdf, advance, &delta_r_levels[idx], &delta_i_levels[idx]);
            }
        }

        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            size_t base = block_idx * (size_t)block;
            int status = execute_nufft_block(plan, backend, src_r_levels, src_i_levels, out_r, out_i, q);
            if (status != CHI2PER_OK) {
                return status;
            }

            int count = (base + (size_t)block <= (size_t)N) ? block : (int)((size_t)N - base);
            for (int k = 0; k < count; ++k) {
                C[(size_t)q * (size_t)N + base + (size_t)k] = out_r[k];
                S[(size_t)q * (size_t)N + base + (size_t)k] = out_i[k];
            }

            if (block_idx + 1 < num_blocks) {
                int level = tls_twiddle_ladder_carry_level(block_idx + 1, ladder_levels);
                size_t offset = (size_t)level * (size_t)M;
                rotate_source_level(src_r_levels + offset, src_i_levels + offset, delta_r_levels + offset, delta_i_levels + offset, M);
                copy_twiddle_level_down(src_r_levels, src_i_levels, M, level);
            }
        }
    }

    return CHI2PER_OK;
}

// Implementation of Generalized Scargle Periodogram using 3d NuFFTs instead of
// 4 used for generic AoV Solution by: Zechmeister and M. Kurster, A&A 496,
// 577-584 (2009) Trig identity: Press W.H. and Rybicki, G.B, "Fast algorithm
// for spectral analysis of unevenly sampled data". ApJ 1:338, p277, 1989
static int gls_impl(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, FLOAT ws, FLOAT yws, FLOAT chi2_ref,
                    int normalization, int M, const cybenko_model_t *cyb, FLOAT b, FLOAT *power, FLOAT *cond) {
    FLOAT half = FCAST(0.5);
    FLOAT sqrt_half = M_SQRT(half);
    FLOAT inv_ws = DIV(FCAST(1.0), ws);

    for (int idx = 0; idx < N; ++idx) {
        FLOAT S = Sw[(size_t)N + idx];
        FLOAT C = Cw[(size_t)N + idx];
        FLOAT S2 = Sw[(size_t)2 * N + idx];
        FLOAT C2 = Cw[(size_t)2 * N + idx];
        FLOAT Sh = Syw[(size_t)N + idx];
        FLOAT Ch = Cyw[(size_t)N + idx];

        FLOAT num = SUB(S2, MUL(MUL(FCAST(2.0), S), MUL(C, inv_ws)));
        FLOAT den = SUB(C2, MUL(SUB(MUL(C, C), MUL(S, S)), inv_ws));
        FLOAT radius = M_SQRT(ADD(MUL(num, num), MUL(den, den)));

        FLOAT S2w = FCAST(0.0);
        FLOAT C2w = FCAST(1.0);
        if (TO_DOUBLE(radius) != 0.0) {
            FLOAT den_abs = den;
            FLOAT den_sign = FCAST(1.0);
            if (TO_DOUBLE(den) < 0.0) {
                den_abs = NEG(den);
                den_sign = FCAST(-1.0);
            }
            S2w = MUL(den_sign, DIV(num, radius));
            C2w = DIV(den_abs, radius);
        }

        FLOAT one_plus_c2 = ADD(FCAST(1.0), C2w);
        FLOAT one_minus_c2 = SUB(FCAST(1.0), C2w);
        if (TO_DOUBLE(one_plus_c2) < 0.0) one_plus_c2 = FCAST(0.0);
        if (TO_DOUBLE(one_minus_c2) < 0.0) one_minus_c2 = FCAST(0.0);

        FLOAT Ctau = MUL(sqrt_half, M_SQRT(one_plus_c2));
        FLOAT Stau = FCAST(0.0);
        if (TO_DOUBLE(S2w) > 0.0)
            Stau = MUL(sqrt_half, M_SQRT(one_minus_c2));
        else if (TO_DOUBLE(S2w) < 0.0)
            Stau = NEG(MUL(sqrt_half, M_SQRT(one_minus_c2)));

        FLOAT YC = ADD(MUL(Ch, Ctau), MUL(Sh, Stau));
        FLOAT YS = SUB(MUL(Sh, Ctau), MUL(Ch, Stau));
        FLOAT XC = ADD(MUL(C, Ctau), MUL(S, Stau));
        FLOAT XS = SUB(MUL(S, Ctau), MUL(C, Stau));

        FLOAT CC = MUL(half, ADD(ws, ADD(MUL(C2, C2w), MUL(S2, S2w))));
        FLOAT SS = MUL(half, SUB(ws, ADD(MUL(C2, C2w), MUL(S2, S2w))));
        CC = SUB(CC, MUL(MUL(XC, XC), inv_ws));
        SS = SUB(SS, MUL(MUL(XS, XS), inv_ws));
        YC = SUB(YC, MUL(MUL(yws, XC), inv_ws));
        YS = SUB(YS, MUL(MUL(yws, XS), inv_ws));

        FLOAT dot = ADD(DIV(MUL(YC, YC), CC), DIV(MUL(YS, YS), SS));
        FLOAT abs2 = MUL(ADD(MUL(C, C), MUL(S, S)), MUL(inv_ws, inv_ws));
        FLOAT condition_bound;
#if defined(DOUBLE_DOUBLE)
        double abs2_d = TO_DOUBLE(abs2);
        if (abs2_d >= 1.0) abs2_d = 0.999999999999999;
        double abs_refl = sqrt(abs2_d);
        condition_bound = dd_div(dd_add(dd_make(1.0, 0.0), dd_make(abs_refl, 0.0)),
                                 dd_sub(dd_make(1.0, 0.0), dd_make(abs_refl, 0.0)));
#elif defined(DOUBLE)
        double abs2_d = abs2;
        if (abs2_d >= 1.0) abs2_d = 0.999999999999999;
        double abs_refl = sqrt(abs2_d);
        condition_bound = (1.0 + abs_refl) / (1.0 - abs_refl);
#else
        float abs2_f = abs2;
        if (abs2_f >= 1.0f) abs2_f = 0.9999999f;
        float abs_refl = sqrtf(abs2_f);
        condition_bound = (1.0f + abs_refl) / (1.0f - abs_refl);
#endif
        power[idx] = normalize_power_entry(dot, chi2_ref, condition_bound, normalization, 1, M, cyb, b);
        if (cond) cond[idx] = condition_bound;
    }

    return CHI2PER_OK;
}

#if defined(DOUBLE_DOUBLE)
static int solve_periodogram_ldlt_dd(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, FLOAT chi2_ref,
                                     int normalization, int M, const cybenko_model_t *cyb, FLOAT b, FLOAT *power, FLOAT *cond) {
    if (normalization == TLS_NORM_BAYES) return CHI2PER_ERR_SOLVER;
    const int norder = 2 * degree + 1;
    size_t norder_sq = (size_t)norder * (size_t)norder;

    FLOAT *XTX = (FLOAT *)checked_malloc(norder_sq, sizeof(FLOAT));
    FLOAT *XTy = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *X = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *L = (FLOAT *)checked_malloc(norder_sq, sizeof(FLOAT));
    FLOAT *D = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Y = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Z = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));

    if (!XTX || !XTy || !X || !L || !D || !Y || !Z) {
        free(XTX);
        free(XTy);
        free(X);
        free(L);
        free(D);
        free(Y);
        free(Z);
        return CHI2PER_ERR_ALLOC;
    }

    int nan_count = 0;

    for (int idx = 0; idx < N; ++idx) {
        for (int row = 0; row < norder; ++row) {
            XTy[row] = real_rhs_value(Syw, Cyw, N, idx, row);
            for (int col = 0; col < norder; ++col) {
                XTX[(size_t)row * norder + col] = real_gram_value(Sw, Cw, N, idx, row, col);
            }
        }

        FLOAT condition_bound = SOLVE_LDLT((size_t)norder, XTX, XTy, X, L, D, Y, Z);
        if (cond) cond[idx] = condition_bound;
        if (!cond && condition_bound_is_singular(condition_bound, CHI2PER_SOLVER_LDLT)) {
            power[idx] = FCAST(NAN);
            ++nan_count;
            continue;
        }

        FLOAT dot = FCAST(0.0);
        for (int k = 0; k < norder; ++k) dot = ADD(dot, MUL(XTy[k], X[k]));
        power[idx] = normalize_power_entry(dot, chi2_ref, condition_bound, normalization, degree, M, cyb, b);
    }

    free(XTX);
    free(XTy);
    free(X);
    free(L);
    free(D);
    free(Y);
    free(Z);
    return cond ? CHI2PER_OK : nan_count;
}

static int solve_periodogram_svd_dd(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, FLOAT chi2_ref,
                                    int normalization, int M, const cybenko_model_t *cyb, FLOAT b, FLOAT *power, FLOAT *cond) {
    if (normalization == TLS_NORM_BAYES) return CHI2PER_ERR_SOLVER;
    const int norder = 2 * degree + 1;
    size_t norder_sq = (size_t)norder * (size_t)norder;

    FLOAT *XTX = (FLOAT *)checked_malloc(norder_sq, sizeof(FLOAT));
    FLOAT *XTy = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *X = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Q = (FLOAT *)checked_malloc(norder_sq, sizeof(FLOAT));
    FLOAT *S = (FLOAT *)checked_malloc(norder_sq, sizeof(FLOAT));
    FLOAT *D = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Y = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));

    if (!XTX || !XTy || !X || !Q || !S || !D || !Y) {
        free(XTX);
        free(XTy);
        free(X);
        free(Q);
        free(S);
        free(D);
        free(Y);
        return CHI2PER_ERR_ALLOC;
    }

    FLOAT max_cond = cond ? FCAST(0.0) : FCAST(COND_SINGULARITY_THRESHOLD_SVD);
    int nan_count = 0;

    for (int idx = 0; idx < N; ++idx) {
        for (int row = 0; row < norder; ++row) {
            XTy[row] = real_rhs_value(Syw, Cyw, N, idx, row);
            for (int col = 0; col < norder; ++col) {
                XTX[(size_t)row * norder + col] = real_gram_value(Sw, Cw, N, idx, row, col);
            }
        }

        FLOAT condition_bound = SOLVE_SVD((size_t)norder, XTX, XTy, X, Q, S, D, Y, max_cond);
        if (cond) cond[idx] = condition_bound;
        if (!cond && condition_bound_is_invalid(condition_bound)) {
            power[idx] = FCAST(NAN);
            ++nan_count;
            continue;
        }

        FLOAT dot = FCAST(0.0);
        for (int k = 0; k < norder; ++k) dot = ADD(dot, MUL(XTy[k], X[k]));
        power[idx] = normalize_power_entry(dot, chi2_ref, condition_bound, normalization, degree, M, cyb, b);
    }

    free(XTX);
    free(XTy);
    free(X);
    free(Q);
    free(S);
    free(D);
    free(Y);
    return cond ? CHI2PER_OK : nan_count;
}

static int solve_periodogram_dd(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, int solver, FLOAT chi2_ref,
                                int normalization, int M, const cybenko_model_t *cyb, FLOAT b, FLOAT *power, FLOAT *cond) {
    if (solver == CHI2PER_SOLVER_LDLT) return solve_periodogram_ldlt_dd(Sw, Cw, Syw, Cyw, N, degree, chi2_ref, normalization, M, cyb, b, power, cond);
    if (solver == CHI2PER_SOLVER_SVD) return solve_periodogram_svd_dd(Sw, Cw, Syw, Cyw, N, degree, chi2_ref, normalization, M, cyb, b, power, cond);

    const int norder = 2 * degree + 1;

    FLOAT *Rr = (FLOAT *)checked_malloc((size_t)norder + 1, sizeof(FLOAT));
    FLOAT *Ri = (FLOAT *)checked_malloc((size_t)norder + 1, sizeof(FLOAT));
    FLOAT *Yr = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Yi = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Xr = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Xi = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Ar = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Ai = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Apr = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *Api = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
    FLOAT *BUr = NULL;
    FLOAT *BUi = NULL;
    FLOAT *BD = NULL;
    FLOAT *Bur = NULL;
    FLOAT *Bui = NULL;
    FLOAT *Bvr = NULL;
    FLOAT *Bvi = NULL;
    FLOAT *Bwr = NULL;
    FLOAT *Bwi = NULL;

    if (!Rr || !Ri || !Yr || !Yi || !Xr || !Xi || !Ar || !Ai || !Apr || !Api) {
        free(Rr);
        free(Ri);
        free(Yr);
        free(Yi);
        free(Xr);
        free(Xi);
        free(Ar);
        free(Ai);
        free(Apr);
        free(Api);
        return CHI2PER_ERR_ALLOC;
    }

    if (solver == CHI2PER_SOLVER_BAREISS) {
        size_t norder_sq = (size_t)norder * (size_t)norder;
        BUr = (FLOAT *)checked_malloc(norder_sq, sizeof(FLOAT));
        BUi = (FLOAT *)checked_malloc(norder_sq, sizeof(FLOAT));
        BD = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
        Bur = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
        Bui = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
        Bvr = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
        Bvi = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
        Bwr = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));
        Bwi = (FLOAT *)checked_malloc((size_t)norder, sizeof(FLOAT));

        if (!BUr || !BUi || !BD || !Bur || !Bui || !Bvr || !Bvi || !Bwr || !Bwi) {
            free(Rr);
            free(Ri);
            free(Yr);
            free(Yi);
            free(Xr);
            free(Xi);
            free(Ar);
            free(Ai);
            free(Apr);
            free(Api);
            free(BUr);
            free(BUi);
            free(BD);
            free(Bur);
            free(Bui);
            free(Bvr);
            free(Bvi);
            free(Bwr);
            free(Bwi);
            return CHI2PER_ERR_ALLOC;
        }
    }

    int nan_count = 0;

    for (int idx = 0; idx < N; ++idx) {
        for (int k = 0; k < norder; ++k) {
            Rr[k] = Cw[(size_t)k * N + idx];
            Ri[k] = NEG(Sw[(size_t)k * N + idx]);

            int h = k - degree;
            if (h == 0) {
                Yr[k] = Cyw[idx];
                Yi[k] = FCAST(0.0);
            } else if (h > 0) {
                Yr[k] = Cyw[(size_t)h * N + idx];
                Yi[k] = NEG(Syw[(size_t)h * N + idx]);
            } else {
                int ah = -h;
                Yr[k] = Cyw[(size_t)ah * N + idx];
                Yi[k] = Syw[(size_t)ah * N + idx];
            }
        }
        Rr[norder] = FCAST(0.0);
        Ri[norder] = FCAST(0.0);

        FLOAT condition_bound;
        if (solver == CHI2PER_SOLVER_LEVINSON) {
            condition_bound = SOLVE_LEVINSON((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ar, Ai, Apr, Api);
        } else if (solver == CHI2PER_SOLVER_ZOHAR) {
            condition_bound = SOLVE_ZOHAR((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ar, Ai, Apr, Api);
        } else if (solver == CHI2PER_SOLVER_BAREISS) {
            condition_bound = SOLVE_BAREISS((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, BUr, BUi, BD, Bur, Bui, Bvr, Bvi, Bwr, Bwi);
        } else {
            free(Rr);
            free(Ri);
            free(Yr);
            free(Yi);
            free(Xr);
            free(Xi);
            free(Ar);
            free(Ai);
            free(Apr);
            free(Api);
            free(BUr);
            free(BUi);
            free(BD);
            free(Bur);
            free(Bui);
            free(Bvr);
            free(Bvi);
            free(Bwr);
            free(Bwi);
            return CHI2PER_ERR_SOLVER;
        }

        if (cond) cond[idx] = condition_bound;
        if (!cond && condition_bound_is_singular(condition_bound, solver)) {
            power[idx] = FCAST(NAN);
            ++nan_count;
            continue;
        }

        FLOAT dot = FCAST(0.0);
        for (int k = 0; k < norder; ++k) dot = ADD(dot, ADD(MUL(Yr[k], Xr[k]), MUL(Yi[k], Xi[k])));
        power[idx] = normalize_power_entry(dot, chi2_ref, condition_bound, normalization, degree, M, cyb, b);
    }

    free(Rr);
    free(Ri);
    free(Yr);
    free(Yi);
    free(Xr);
    free(Xi);
    free(Ar);
    free(Ai);
    free(Apr);
    free(Api);
    free(BUr);
    free(BUi);
    free(BD);
    free(Bur);
    free(Bui);
    free(Bvr);
    free(Bvi);
    free(Bwr);
    free(Bwi);
    return cond ? CHI2PER_OK : nan_count;
}
#else
static int solve_periodogram_ldlt_vec(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, FLOAT chi2_ref,
                                      int normalization, int M, const cybenko_model_t *cyb, FLOAT b, FLOAT *power, FLOAT *cond) {
    if (normalization == TLS_NORM_BAYES) return CHI2PER_ERR_SOLVER;
    const int norder = 2 * degree + 1;
    size_t norder_sq = (size_t)norder * (size_t)norder;

    INTERNAL_VEC *XTX = (INTERNAL_VEC *)checked_aligned_malloc(norder_sq, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *XTy = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *X = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *L = (INTERNAL_VEC *)checked_aligned_malloc(norder_sq, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *D = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Y = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Z = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));

    if (!XTX || !XTy || !X || !L || !D || !Y || !Z) {
        free(XTX);
        free(XTy);
        free(X);
        free(L);
        free(D);
        free(Y);
        free(Z);
        return CHI2PER_ERR_ALLOC;
    }

    int nan_count = 0;

    for (int base = 0; base < N; base += INTERNAL_VEC_LEN) {
        for (int row = 0; row < norder; ++row) {
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) idx = N - 1;
                XTy[row][lane] = real_rhs_value(Syw, Cyw, N, idx, row);
            }

            for (int col = 0; col < norder; ++col) {
                for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                    int idx = base + lane;
                    if (idx >= N) idx = N - 1;
                    XTX[(size_t)row * norder + col][lane] = real_gram_value(Sw, Cw, N, idx, row, col);
                }
            }
        }

        INTERNAL_VEC condition_bound = SOLVE_LDLT((size_t)norder, XTX, XTy, X, L, D, Y, Z);

        for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
            int idx = base + lane;
            if (idx >= N) continue;
            if (cond) cond[idx] = condition_bound[lane];
            if (!cond && condition_bound_is_singular(condition_bound[lane], CHI2PER_SOLVER_LDLT)) {
                power[idx] = FCAST(NAN);
                ++nan_count;
                continue;
            }

            FLOAT dot = FCAST(0.0);
            for (int k = 0; k < norder; ++k) {
                dot = ADD(dot, MUL(XTy[k][lane], X[k][lane]));
            }
            power[idx] = normalize_power_entry(dot, chi2_ref, condition_bound[lane], normalization, degree, M, cyb, b);
        }
    }

    free(XTX);
    free(XTy);
    free(X);
    free(L);
    free(D);
    free(Y);
    free(Z);
    return cond ? CHI2PER_OK : nan_count;
}

static int solve_periodogram_svd_vec(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, FLOAT chi2_ref,
                                     int normalization, int M, const cybenko_model_t *cyb, FLOAT b, FLOAT *power, FLOAT *cond) {
    if (normalization == TLS_NORM_BAYES) return CHI2PER_ERR_SOLVER;
    const int norder = 2 * degree + 1;
    size_t norder_sq = (size_t)norder * (size_t)norder;

    INTERNAL_VEC *XTX = (INTERNAL_VEC *)checked_aligned_malloc(norder_sq, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *XTy = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *X = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Q = (INTERNAL_VEC *)checked_aligned_malloc(norder_sq, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *S = (INTERNAL_VEC *)checked_aligned_malloc(norder_sq, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *D = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Y = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));

    if (!XTX || !XTy || !X || !Q || !S || !D || !Y) {
        free(XTX);
        free(XTy);
        free(X);
        free(Q);
        free(S);
        free(D);
        free(Y);
        return CHI2PER_ERR_ALLOC;
    }

    INTERNAL_VEC max_cond;
    for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) max_cond[lane] = cond ? FCAST(0.0) : FCAST(COND_SINGULARITY_THRESHOLD_SVD);
    int nan_count = 0;

    for (int base = 0; base < N; base += INTERNAL_VEC_LEN) {
        for (int row = 0; row < norder; ++row) {
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) idx = N - 1;
                XTy[row][lane] = real_rhs_value(Syw, Cyw, N, idx, row);
            }

            for (int col = 0; col < norder; ++col) {
                for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                    int idx = base + lane;
                    if (idx >= N) idx = N - 1;
                    XTX[(size_t)row * norder + col][lane] = real_gram_value(Sw, Cw, N, idx, row, col);
                }
            }
        }

        INTERNAL_VEC condition_bound = SOLVE_SVD((size_t)norder, XTX, XTy, X, Q, S, D, Y, max_cond);

        for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
            int idx = base + lane;
            if (idx >= N) continue;
            if (cond) cond[idx] = condition_bound[lane];
            if (!cond && condition_bound_is_invalid(condition_bound[lane])) {
                power[idx] = FCAST(NAN);
                ++nan_count;
                continue;
            }

            FLOAT dot = FCAST(0.0);
            for (int k = 0; k < norder; ++k) {
                dot = ADD(dot, MUL(XTy[k][lane], X[k][lane]));
            }
            power[idx] = normalize_power_entry(dot, chi2_ref, condition_bound[lane], normalization, degree, M, cyb, b);
        }
    }

    free(XTX);
    free(XTy);
    free(X);
    free(Q);
    free(S);
    free(D);
    free(Y);
    return cond ? CHI2PER_OK : nan_count;
}

static int solve_periodogram_vec(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, int solver, FLOAT chi2_ref,
                                 int normalization, int M, const cybenko_model_t *cyb, FLOAT b, FLOAT *power, FLOAT *cond) {
    if (solver == CHI2PER_SOLVER_LDLT) return solve_periodogram_ldlt_vec(Sw, Cw, Syw, Cyw, N, degree, chi2_ref, normalization, M, cyb, b, power, cond);
    if (solver == CHI2PER_SOLVER_SVD) return solve_periodogram_svd_vec(Sw, Cw, Syw, Cyw, N, degree, chi2_ref, normalization, M, cyb, b, power, cond);

    const int norder = 2 * degree + 1;

    INTERNAL_VEC *Rr = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder + 1, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Ri = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder + 1, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Yr = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Yi = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Xr = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Xi = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Ehr = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Ehi = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Ehpr = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *Ehpi = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
    INTERNAL_VEC *BUr = NULL;
    INTERNAL_VEC *BUi = NULL;
    INTERNAL_VEC *BD = NULL;
    INTERNAL_VEC *Bur = NULL;
    INTERNAL_VEC *Bui = NULL;
    INTERNAL_VEC *Bvr = NULL;
    INTERNAL_VEC *Bvi = NULL;
    INTERNAL_VEC *Bwr = NULL;
    INTERNAL_VEC *Bwi = NULL;

    if (!Rr || !Ri || !Yr || !Yi || !Xr || !Xi || !Ehr || !Ehi || !Ehpr || !Ehpi) {
        free(Rr);
        free(Ri);
        free(Yr);
        free(Yi);
        free(Xr);
        free(Xi);
        free(Ehr);
        free(Ehi);
        free(Ehpr);
        free(Ehpi);
        return CHI2PER_ERR_ALLOC;
    }

    if (solver == CHI2PER_SOLVER_BAREISS) {
        size_t norder_sq = (size_t)norder * (size_t)norder;
        BUr = (INTERNAL_VEC *)checked_aligned_malloc(norder_sq, sizeof(INTERNAL_VEC));
        BUi = (INTERNAL_VEC *)checked_aligned_malloc(norder_sq, sizeof(INTERNAL_VEC));
        BD = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
        Bur = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
        Bui = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
        Bvr = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
        Bvi = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
        Bwr = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));
        Bwi = (INTERNAL_VEC *)checked_aligned_malloc((size_t)norder, sizeof(INTERNAL_VEC));

        if (!BUr || !BUi || !BD || !Bur || !Bui || !Bvr || !Bvi || !Bwr || !Bwi) {
            free(Rr);
            free(Ri);
            free(Yr);
            free(Yi);
            free(Xr);
            free(Xi);
            free(Ehr);
            free(Ehi);
            free(Ehpr);
            free(Ehpi);
            free(BUr);
            free(BUi);
            free(BD);
            free(Bur);
            free(Bui);
            free(Bvr);
            free(Bvi);
            free(Bwr);
            free(Bwi);
            return CHI2PER_ERR_ALLOC;
        }
    }

    int nan_count = 0;

    for (int base = 0; base < N; base += INTERNAL_VEC_LEN) {
        for (int k = 0; k < norder; ++k) {
            int h = k - degree;
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) idx = N - 1;

                Rr[k][lane] = Cw[(size_t)k * N + idx];
                Ri[k][lane] = NEG(Sw[(size_t)k * N + idx]);

                if (h == 0) {
                    Yr[k][lane] = Cyw[idx];
                    Yi[k][lane] = FCAST(0.0);
                } else if (h > 0) {
                    Yr[k][lane] = Cyw[(size_t)h * N + idx];
                    Yi[k][lane] = NEG(Syw[(size_t)h * N + idx]);
                } else {
                    int ah = -h;
                    Yr[k][lane] = Cyw[(size_t)ah * N + idx];
                    Yi[k][lane] = Syw[(size_t)ah * N + idx];
                }
            }
        }

        for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
            Rr[norder][lane] = FCAST(0.0);
            Ri[norder][lane] = FCAST(0.0);
        }

        int singular_lane[INTERNAL_VEC_LEN];
        for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) singular_lane[lane] = 0;
        INTERNAL_VEC condition_bound;

        if (solver == CHI2PER_SOLVER_LEVINSON) {
            condition_bound = SOLVE_LEVINSON((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ehr, Ehi, Ehpr, Ehpi);
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) continue;
                if (cond) cond[idx] = condition_bound[lane];
                if (!cond && condition_bound_is_singular(condition_bound[lane], solver)) {
                    singular_lane[lane] = 1;
                    power[idx] = FCAST(NAN);
                    ++nan_count;
                }
            }
        } else if (solver == CHI2PER_SOLVER_ZOHAR) {
            condition_bound = SOLVE_ZOHAR((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ehr, Ehi, Ehpr, Ehpi);
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) continue;
                if (cond) cond[idx] = condition_bound[lane];
                if (!cond && condition_bound_is_singular(condition_bound[lane], solver)) {
                    singular_lane[lane] = 1;
                    power[idx] = FCAST(NAN);
                    ++nan_count;
                }
            }
        } else if (solver == CHI2PER_SOLVER_BAREISS) {
            condition_bound = SOLVE_BAREISS((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, BUr, BUi, BD, Bur, Bui, Bvr, Bvi, Bwr, Bwi);
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) continue;
                if (cond) cond[idx] = condition_bound[lane];
                if (!cond && condition_bound_is_singular(condition_bound[lane], solver)) {
                    singular_lane[lane] = 1;
                    power[idx] = FCAST(NAN);
                    ++nan_count;
                }
            }
        } else {
            free(Rr);
            free(Ri);
            free(Yr);
            free(Yi);
            free(Xr);
            free(Xi);
            free(Ehr);
            free(Ehi);
            free(Ehpr);
            free(Ehpi);
            free(BUr);
            free(BUi);
            free(BD);
            free(Bur);
            free(Bui);
            free(Bvr);
            free(Bvi);
            free(Bwr);
            free(Bwi);
            return CHI2PER_ERR_SOLVER;
        }

        for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
            int idx = base + lane;
            if (idx >= N) continue;
            if (singular_lane[lane]) continue;

            FLOAT dot = FCAST(0.0);
            for (int k = 0; k < norder; ++k) {
                FLOAT yr = Yr[k][lane];
                FLOAT yi = Yi[k][lane];
                FLOAT xr = Xr[k][lane];
                FLOAT xi = Xi[k][lane];
                dot = ADD(dot, ADD(MUL(yr, xr), MUL(yi, xi)));
            }
            power[idx] = normalize_power_entry(dot, chi2_ref, condition_bound[lane], normalization, degree, M, cyb, b);
        }
    }

    free(Rr);
    free(Ri);
    free(Yr);
    free(Yi);
    free(Xr);
    free(Xi);
    free(Ehr);
    free(Ehi);
    free(Ehpr);
    free(Ehpi);
    free(BUr);
    free(BUi);
    free(BD);
    free(Bur);
    free(Bui);
    free(Bvr);
    free(Bvi);
    free(Bwr);
    free(Bwi);
    return cond ? CHI2PER_OK : nan_count;
}
#endif

int CHI2_PREFIX(fastchi2)(const TIME_INPUT_T *t, const FLOAT *y, const FLOAT *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                          int normalization, FLOAT *power, FLOAT *cond) {
    if (!t || !y || !dy || !power || M <= 0 || N <= 0 || degree <= 0 || f0 < 0.0 || df <= 0.0) return CHI2PER_ERR_ARGUMENT;
    if (normalization < TLS_NORM_STANDARD || normalization > TLS_NORM_BAYES) return CHI2PER_ERR_ARGUMENT;
    if (backend != CHI2PER_BACKEND_PSWF43 && backend != CHI2PER_BACKEND_PSWF21 && backend != CHI2PER_BACKEND_LRA) return CHI2PER_ERR_BACKEND;
    if (degree != 1 && solver != CHI2PER_SOLVER_LEVINSON && solver != CHI2PER_SOLVER_ZOHAR && solver != CHI2PER_SOLVER_BAREISS &&
        solver != CHI2PER_SOLVER_LDLT && solver != CHI2PER_SOLVER_SVD)
        return CHI2PER_ERR_SOLVER;
    if (normalization == TLS_NORM_BAYES && degree > 1 && (solver == CHI2PER_SOLVER_LDLT || solver == CHI2PER_SOLVER_SVD))
        return CHI2PER_ERR_SOLVER;
    if (M < 2 * degree + 2) return CHI2PER_ERR_DEGENERATE;

    bool use_pswf = (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21);
    bool use_pswf43 = (backend == CHI2PER_BACKEND_PSWF43);
    double beta = CHI2_LRA_BETA;
    double gamma = CHI2_LRA_GAMMA;
    if (use_pswf43) {
        beta = CHI2_PSWF43_BETA;
        gamma = CHI2_PSWF43_GAMMA;
    } else if (use_pswf) {
        beta = CHI2_PSWF21_BETA;
        gamma = CHI2_PSWF21_GAMMA;
    }
    int plan_block = tls_optimize_plan_size(N, M, degree, CHI2_ALPHA, beta, gamma, backend);
    if (use_pswf43) plan_block = tls_pswf43_plan_len_from_base(plan_block);
    int block = use_pswf43 ? tls_pswf43_output_len_for_plan(plan_block) : plan_block;
    int ladder_levels = tls_twiddle_ladder_levels(N, block);
    int max_factor = 2 * degree;

    FLOAT *w = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *yw = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    NUFFT_INPUT_T *x = (NUFFT_INPUT_T *)checked_aligned_malloc((size_t)M, sizeof(NUFFT_INPUT_T));
    FLOAT *Sw = (FLOAT *)checked_malloc((size_t)(max_factor + 1) * N, sizeof(FLOAT));
    FLOAT *Cw = (FLOAT *)checked_malloc((size_t)(max_factor + 1) * N, sizeof(FLOAT));
    FLOAT *Syw = (FLOAT *)checked_malloc((size_t)(degree + 1) * N, sizeof(FLOAT));
    FLOAT *Cyw = (FLOAT *)checked_malloc((size_t)(degree + 1) * N, sizeof(FLOAT));
    FLOAT *src_r = (FLOAT *)checked_aligned_malloc((size_t)ladder_levels * (size_t)M, sizeof(FLOAT));
    FLOAT *src_i = (FLOAT *)checked_aligned_malloc((size_t)ladder_levels * (size_t)M, sizeof(FLOAT));
    FLOAT *delta_r = (FLOAT *)checked_aligned_malloc((size_t)ladder_levels * (size_t)M, sizeof(FLOAT));
    FLOAT *delta_i = (FLOAT *)checked_aligned_malloc((size_t)ladder_levels * (size_t)M, sizeof(FLOAT));
    FLOAT *out_r = (FLOAT *)checked_aligned_malloc((size_t)block, sizeof(FLOAT));
    FLOAT *out_i = (FLOAT *)checked_aligned_malloc((size_t)block, sizeof(FLOAT));
    void *plan = NULL;
    int status = CHI2PER_OK;

    if (!w || !yw || !x || !Sw || !Cw || !Syw || !Cyw || !src_r || !src_i || !delta_r || !delta_i || !out_r || !out_i) {
        status = CHI2PER_ERR_ALLOC;
        goto cleanup;
    }

    FLOAT ws = FCAST(0.0);
    FLOAT wy = FCAST(0.0);
    TIME_INPUT_T t_mid = MUL(FCONST(0.5), ADD(t[0], t[M - 1]));
    for (int m = 0; m < M; ++m) {
        if (TO_DOUBLE(dy[m]) <= 0.0) {
            status = CHI2PER_ERR_ARGUMENT;
            goto cleanup;
        }
        TIME_INPUT_T centered_t = SUB(t[m], t_mid);
        FLOAT inv_dy = DIV(FCAST(1.0), dy[m]);
        w[m] = MUL(inv_dy, inv_dy);
        ws = ADD(ws, w[m]);
        wy = ADD(wy, MUL(w[m], y[m]));
        x[m] = time_to_nufft_input(centered_t);
    }

    if (TO_DOUBLE(ws) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT ymean = DIV(wy, ws);
    FLOAT chi2_ref = FCAST(0.0);
    FLOAT yws = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        FLOAT yc = SUB(y[m], ymean);
        yw[m] = MUL(yc, w[m]);
        yws = ADD(yws, yw[m]);
        chi2_ref = ADD(chi2_ref, MUL(yc, yw[m]));
    }

    if (TO_DOUBLE(chi2_ref) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    for (int k = 0; k < N; ++k) {
        Cw[k] = ws;
        Sw[k] = FCAST(0.0);
        Cyw[k] = yws;
        Syw[k] = FCAST(0.0);
    }

    if (backend == CHI2PER_BACKEND_PSWF43) {
        plan = NUFFT_PSWF_INIT(M, plan_block, NUFFT_W43, df, max_factor, "43");
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_PSWF_PRE((PSWF_PLAN_T *)plan, x);
    } else if (backend == CHI2PER_BACKEND_PSWF21) {
        plan = NUFFT_PSWF_INIT(M, plan_block, NUFFT_W21, df, max_factor, "21");
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_PSWF_PRE((PSWF_PLAN_T *)plan, x);
    } else {
        plan = NUFFT_LRA_INIT(M, block, NUFFT_RANK, df, max_factor);
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_LRA_PRE((LRA_PLAN_T *)plan, x, M, block, NUFFT_RANK);
    }

    status = compute_trig_sums(x, w, M, f0, df, N, max_factor, block, backend, plan, Sw, Cw, src_r, src_i, delta_r, delta_i, ladder_levels, out_r, out_i);
    if (status == CHI2PER_OK)
        status = compute_trig_sums(x, yw, M, f0, df, N, degree, block, backend, plan, Syw, Cyw, src_r, src_i, delta_r, delta_i, ladder_levels, out_r, out_i);

    cybenko_model_t cyb = cybenko_model_init(degree, M);
#if defined(DOUBLE_DOUBLE)
    FLOAT b = dd_mul(dd_make(0.5, 0.0), dd_make((double)(M - 1 - 2 * degree), 0.0));
#elif defined(DOUBLE)
    FLOAT b = 0.5 * (double)(M - 1 - 2 * degree);
#else
    FLOAT b = 0.5f * (float)(M - 1 - 2 * degree);
#endif

    if (status == CHI2PER_OK && degree == 1) {
        status = gls_impl(Sw, Cw, Syw, Cyw, N, ws, yws, chi2_ref, normalization, M, &cyb, b, power, cond);
    } else if (status == CHI2PER_OK) {
#if defined(DOUBLE_DOUBLE)
        status = solve_periodogram_dd(Sw, Cw, Syw, Cyw, N, degree, solver, chi2_ref, normalization, M, &cyb, b, power, cond);
#else
        status = solve_periodogram_vec(Sw, Cw, Syw, Cyw, N, degree, solver, chi2_ref, normalization, M, &cyb, b, power, cond);
#endif
    }

cleanup:
    if (plan && use_pswf)
        NUFFT_PSWF_FREE((PSWF_PLAN_T *)plan);
    else if (plan)
        NUFFT_LRA_FREE((LRA_PLAN_T *)plan);

    free(w);
    free(yw);
    free(x);
    free(Sw);
    free(Cw);
    free(Syw);
    free(Cyw);
    free(src_r);
    free(src_i);
    free(delta_r);
    free(delta_i);
    free(out_r);
    free(out_i);
    return status;
}
