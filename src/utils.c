// utils.c  —  Precision-generic peak-detection + precision-agnostic helpers
//
// Compiled 3× with different -D flags:
//   no -D         → FLOAT=float          → tlsf_* symbols
//   -D DOUBLE     → FLOAT=double         → tls_* symbols
//   -D DOUBLE_DOUBLE → FLOAT=dd_t        → tlsdd_* symbols
//
// Precision-agnostic functions (cost model, twiddle ladder) are emitted
// only once, when neither DOUBLE nor DOUBLE_DOUBLE is defined.

#include <nanofft.h>
#include <stdint.h>
#include <stdlib.h>
#include <tls.h>
#include <utils.h>

#ifndef MAX_TWIDDLE_REUSE
#    define MAX_TWIDDLE_REUSE 16
#endif

#if MAX_TWIDDLE_REUSE < 2 || (MAX_TWIDDLE_REUSE & (MAX_TWIDDLE_REUSE - 1)) != 0
#    error "MAX_TWIDDLE_REUSE must be a power of two greater than or equal to 2"
#endif

// Precision macro layer
#include <nanofft_precision.h>

// Symbol prefix per precision pass
#if defined(DOUBLE_DOUBLE)
#    define TLS(name) tlsdd_##name
#elif defined(DOUBLE)
#    define TLS(name) tls_##name
#else
#    define TLS(name) tlsf_##name
#endif

// --------------------------------------------------------------------------
// Precision-agnostic helpers — compiled once (float pass only)
// --------------------------------------------------------------------------
#if !defined(DOUBLE) && !defined(DOUBLE_DOUBLE)

static inline int bitceil(double x) { return x <= 1.0 ? 1 : 1 << (1 + (int)(log2(x))); }

double tls_approximate_cost(int N, int M, int block, int degree, double alpha, double beta, double gamma, int backend) {
    int block_eff = block;
    if (backend == 1) block_eff += block_eff >> 1;

    double N_eff = block_eff * ceil((double)N / block_eff);
    double gamma_eff = gamma;
    if (degree > 0) gamma_eff *= ((double)((2 * degree) + 1)) / (double)((3 * degree) + 1);

    double cost = N_eff * pow((double)block, alpha);
    cost += beta * (N_eff - (double)block_eff) * (double)M / (double)block_eff;
    cost += gamma_eff * block_eff;
    return cost;
}

int tls_optimize_plan_size(int N, int M, int degree, double alpha, double beta, double gamma, int backend) {
    const int min_block = 128;
    double start = pow((beta * (double)M / alpha), 1.0 / (alpha + 1.0));
    int block = bitceil(start);
    int n_cap = bitceil((double)N);

    if (block > n_cap) block = n_cap;
    if (block < min_block) block = min_block;

    double best = tls_approximate_cost(N, M, block, degree, alpha, beta, gamma, backend);
    while (block > min_block) {
        int next = block >> 1;
        if (next < min_block) next = min_block;
        double next_cost = tls_approximate_cost(N, M, next, degree, alpha, beta, gamma, backend);
        if (next_cost >= best) break;
        block = next;
        best = next_cost;
    }
    return block;
}

double tls_approximate_pswf_cost(int N, int M, int block, int degree, int is_ihs, int backend, double alpha, double beta, double gamma, double pre_m,
                                 double const_cost) {
    int block_eff = block;
    if (backend == TLS_BACKEND_PSWF43) block_eff += block_eff >> 1;
    if (block_eff <= 0) block_eff = 1;

    int d = (degree > 0) ? degree : 1;
    double n_blocks = ceil((double)N / (double)block_eff);
    if (n_blocks < 1.0) n_blocks = 1.0;
    double n_factors = is_ihs ? (double)d : (double)(2 * d);
    double n_execs = is_ihs ? (double)d : (double)(3 * d);

    double fft_cost = n_execs * n_blocks * (double)block_eff * pow((double)block, alpha);
    double deconv_cost = gamma * (double)block_eff;
    double pre_cost = pre_m * n_factors * (double)M;
    double exec_cost = beta * n_execs * n_blocks * (double)M;
    return fft_cost + deconv_cost + const_cost + pre_cost + exec_cost;
}

static tls_pswf_plan_choice tls_optimize_pswf_single_mode(int N, int M, int degree, int is_ihs, int backend, double alpha, double beta, double gamma,
                                                          double pre_m, double const_cost, int spread_mode) {
    const int min_block = 128;
    int n_cap = bitceil((double)N);
    if (n_cap < min_block) n_cap = min_block;
    if (n_cap > (1 << 29)) n_cap = (1 << 29);

    tls_pswf_plan_choice best;
    best.block = min_block;
    best.spread_mode = spread_mode;
    best.cost = tls_approximate_pswf_cost(N, M, min_block, degree, is_ihs, backend, alpha, beta, gamma, pre_m, const_cost);

    for (int block = min_block << 1; block <= n_cap; block <<= 1) {
        double c = tls_approximate_pswf_cost(N, M, block, degree, is_ihs, backend, alpha, beta, gamma, pre_m, const_cost);
        if (c < best.cost) {
            best.block = block;
            best.cost = c;
        }
        if (block == (1 << 29)) break;
    }
    return best;
}

tls_pswf_plan_choice tls_optimize_pswf_plan(int N, int M, int degree, int is_ihs, int backend, double alpha, double gamma, double beta_mm, double pre_mm,
                                            double beta_pp, double pre_pp, double ppoly_const, int forced_spread_mode) {
    if (forced_spread_mode == TLS_SPREAD_MINMAX) {
        return tls_optimize_pswf_single_mode(N, M, degree, is_ihs, backend, alpha, beta_mm, gamma, pre_mm, 0.0, TLS_SPREAD_MINMAX);
    }
    if (forced_spread_mode == TLS_SPREAD_PIECEWISE) {
        return tls_optimize_pswf_single_mode(N, M, degree, is_ihs, backend, alpha, beta_pp, gamma, pre_pp, ppoly_const, TLS_SPREAD_PIECEWISE);
    }
    tls_pswf_plan_choice mm = tls_optimize_pswf_single_mode(N, M, degree, is_ihs, backend, alpha, beta_mm, gamma, pre_mm, 0.0, TLS_SPREAD_MINMAX);
    tls_pswf_plan_choice pp = tls_optimize_pswf_single_mode(N, M, degree, is_ihs, backend, alpha, beta_pp, gamma, pre_pp, ppoly_const, TLS_SPREAD_PIECEWISE);
    return (pp.cost < mm.cost) ? pp : mm;
}

double tls_approximate_breadth_cost(int N, int M, int block, int degree, int is_ihs, int backend, int vec_len, double alpha, double b_alpha, double b_fft_ratio,
                                    double b_beta, double gamma, double b_pre_m, double const_cost) {
    int block_eff = block;
    if (backend == TLS_BACKEND_PSWF43) block_eff += block_eff >> 1;
    if (block_eff <= 0) block_eff = 1;
    if (vec_len < 1) vec_len = 1;

    int d = (degree > 0) ? degree : 1;
    double batch_eff = (double)vec_len * (double)block_eff;
    double n_batches = ceil((double)N / batch_eff);
    if (n_batches < 1.0) n_batches = 1.0;
    double n_factors = is_ihs ? (double)d : (double)(2 * d);
    double n_execs = is_ihs ? (double)d : (double)(3 * d);

    /* PSWF FFTs have Nfft = 2*block while unit_pswf is normalized by 2^(1+alpha),
     * so converting unit_bfft*(2*block)^(1+b_alpha) into unit_pswf units carries
     * the factor 2^(b_alpha - alpha).  LRA has Nfft = block (no factor of 2). */
    double b_fft_scale = b_fft_ratio;
    if (backend == TLS_BACKEND_PSWF43 || backend == TLS_BACKEND_PSWF21) b_fft_scale *= pow(2.0, b_alpha - alpha);

    double fft_cost = n_execs * n_batches * b_fft_scale * (double)block_eff * pow((double)block, b_alpha);
    double deconv_cost = gamma * (double)block_eff;
    double pre_cost = b_pre_m * n_factors * (double)M;
    double exec_cost = b_beta * n_execs * n_batches * (double)M;
    return fft_cost + deconv_cost + const_cost + pre_cost + exec_cost;
}

static tls_plan_choice tls_sweep_breadth_mode(int N, int M, int degree, int is_ihs, int backend, int vec_len, int min_block_breadth, double alpha,
                                              double b_alpha, double b_fft_ratio, double b_beta, double gamma, double b_pre_m, double const_cost,
                                              int spread_mode) {
    int min_b = (min_block_breadth >= 4) ? min_block_breadth : 4;
    double target = ceil((double)N / (double)((vec_len > 0) ? vec_len : 1));
    int n_cap = bitceil(target);
    if (n_cap < min_b) n_cap = min_b;
    if (n_cap > (1 << 29)) n_cap = (1 << 29);

    tls_plan_choice best;
    best.block = min_b;
    best.spread_mode = spread_mode;
    best.vec_mode = TLS_VEC_BREADTH;
    best.cost = tls_approximate_breadth_cost(N, M, min_b, degree, is_ihs, backend, vec_len, alpha, b_alpha, b_fft_ratio, b_beta, gamma, b_pre_m, const_cost);

    for (int block = min_b << 1; block <= n_cap; block <<= 1) {
        double c = tls_approximate_breadth_cost(N, M, block, degree, is_ihs, backend, vec_len, alpha, b_alpha, b_fft_ratio, b_beta, gamma, b_pre_m, const_cost);
        if (c < best.cost) {
            best.block = block;
            best.cost = c;
        }
        if (block == (1 << 29)) break;
    }
    return best;
}

static void tls_consider_choice(tls_plan_choice *best, int *have_best, tls_plan_choice cand) {
    if (!(*have_best) || cand.cost < best->cost) {
        *best = cand;
        *have_best = 1;
    }
}

tls_plan_choice tls_optimize_plan(int N, int M, int degree, int is_ihs, int backend, int vec_len, int min_block_breadth, int is_dd, double alpha, double gamma,
                                  double beta_mm, double pre_mm, double beta_pp, double pre_pp, double ppoly_const, double b_alpha, double b_fft_ratio,
                                  double b_beta_mm, double b_pre_mm, double b_beta_pp, double b_pre_pp, int forced_spread_mode, int forced_vec_mode) {
    const int is_lra = (backend == TLS_BACKEND_LRA);
    tls_plan_choice best = {128, TLS_SPREAD_MINMAX, TLS_VEC_DEPTH, 0.0};
    int have_best = 0;

    if (!is_dd && forced_vec_mode != TLS_VEC_BREADTH) {
        if (is_lra) {
            tls_pswf_plan_choice d = tls_optimize_pswf_single_mode(N, M, degree, is_ihs, backend, alpha, beta_mm, gamma, 0.0, 0.0, TLS_SPREAD_MINMAX);
            tls_consider_choice(&best, &have_best, (tls_plan_choice){d.block, TLS_SPREAD_MINMAX, TLS_VEC_DEPTH, d.cost});
        } else {
            if (forced_spread_mode != TLS_SPREAD_PIECEWISE) {
                tls_pswf_plan_choice d = tls_optimize_pswf_single_mode(N, M, degree, is_ihs, backend, alpha, beta_mm, gamma, pre_mm, 0.0, TLS_SPREAD_MINMAX);
                tls_consider_choice(&best, &have_best, (tls_plan_choice){d.block, TLS_SPREAD_MINMAX, TLS_VEC_DEPTH, d.cost});
            }
            if (forced_spread_mode != TLS_SPREAD_MINMAX) {
                tls_pswf_plan_choice d =
                    tls_optimize_pswf_single_mode(N, M, degree, is_ihs, backend, alpha, beta_pp, gamma, pre_pp, ppoly_const, TLS_SPREAD_PIECEWISE);
                tls_consider_choice(&best, &have_best, (tls_plan_choice){d.block, TLS_SPREAD_PIECEWISE, TLS_VEC_DEPTH, d.cost});
            }
        }
    }

    if (forced_vec_mode != TLS_VEC_DEPTH) {
        int min_b = min_block_breadth;
        if (is_dd && N < min_b * vec_len) min_b = 16;
        if (is_lra) {
            tls_plan_choice b = tls_sweep_breadth_mode(N, M, degree, is_ihs, backend, vec_len, min_b, alpha, b_alpha, b_fft_ratio, b_beta_mm, gamma, 0.0, 0.0,
                                                       TLS_SPREAD_MINMAX);
            tls_consider_choice(&best, &have_best, b);
        } else {
            if (is_dd || forced_spread_mode != TLS_SPREAD_PIECEWISE) {
                tls_plan_choice b = tls_sweep_breadth_mode(N, M, degree, is_ihs, backend, vec_len, min_b, alpha, b_alpha, b_fft_ratio, b_beta_mm, gamma,
                                                           b_pre_mm, 0.0, TLS_SPREAD_MINMAX);
                tls_consider_choice(&best, &have_best, b);
            }
            if (!is_dd && forced_spread_mode != TLS_SPREAD_MINMAX) {
                tls_plan_choice b = tls_sweep_breadth_mode(N, M, degree, is_ihs, backend, vec_len, min_b, alpha, b_alpha, b_fft_ratio, b_beta_pp, gamma,
                                                           b_pre_pp, ppoly_const, TLS_SPREAD_PIECEWISE);
                tls_consider_choice(&best, &have_best, b);
            }
        }
    }

    return best;
}

int tls_pswf43_plan_len_from_base(int base_len) {
    int plan_len = base_len;
    if (plan_len < 4) plan_len = 4;
    return (plan_len + 3) & ~3;
}

int tls_pswf43_output_len_for_plan(int plan_len) { return plan_len + (plan_len >> 1); }

int tls_twiddle_ladder_levels(int N, int block) {
    if (N <= 0 || block <= 0) return 1;

    size_t num_blocks = ((size_t)N + (size_t)block - 1) / (size_t)block;
    if (num_blocks <= 1) return 1;

    size_t max_advance = num_blocks - 1;
    size_t stride = (size_t)MAX_TWIDDLE_REUSE;
    int levels = 1;
    while (max_advance >= stride) {
        ++levels;
        if (stride > SIZE_MAX / (size_t)MAX_TWIDDLE_REUSE) break;
        stride *= (size_t)MAX_TWIDDLE_REUSE;
    }
    return levels;
}

int tls_twiddle_ladder_carry_level(size_t next_block, int levels) {
    int level = 0;
    size_t stride = (size_t)MAX_TWIDDLE_REUSE;
    while (level + 1 < levels && next_block % stride == 0) {
        ++level;
        if (stride > SIZE_MAX / (size_t)MAX_TWIDDLE_REUSE) break;
        stride *= (size_t)MAX_TWIDDLE_REUSE;
    }
    return level;
}

double tls_twiddle_ladder_advance(int block, int level) {
    double advance = (double)block;
    for (int i = 0; i < level; ++i) advance *= (double)MAX_TWIDDLE_REUSE;
    return advance;
}

#endif /* agnostic block */

// --------------------------------------------------------------------------
// Local divergence helpers — defined in every precision pass
// --------------------------------------------------------------------------

#if defined(DOUBLE_DOUBLE)

static inline int FLOAT_IS_FINITE(FLOAT v) {
    union {
        double f;
        uint64_t u;
    } hi_bits = {v.hi};
    union {
        double f;
        uint64_t u;
    } lo_bits = {v.lo};
    return (hi_bits.u & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000) &&
           (lo_bits.u & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
static inline int FLOAT_IS_ZERO(FLOAT v) { return v.hi == 0.0 && v.lo == 0.0; }
static inline int FLOAT_CMP(FLOAT a, FLOAT b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return 1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return 1;
    return 0;
}
#    define FLOAT_NAN dd_make(NAN, NAN)

#elif defined(DOUBLE)

static inline int FLOAT_IS_FINITE(FLOAT v) {
    union {
        double f;
        uint64_t u;
    } bits = {v};
    return (bits.u & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
#    define FLOAT_IS_ZERO(v) ((v) == FCONST(0))
#    define FLOAT_CMP(a, b) (((a) > (b)) - ((a) < (b)))
#    define FLOAT_NAN NAN

#else /* float (default) */

static inline int FLOAT_IS_FINITE(FLOAT v) {
    union {
        float f;
        uint32_t u;
    } bits = {v};
    return (bits.u & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}
#    define FLOAT_IS_ZERO(v) ((v) == FCONST(0))
#    define FLOAT_CMP(a, b) (((a) > (b)) - ((a) < (b)))
#    define FLOAT_NAN NAN

#endif

// --------------------------------------------------------------------------
// Precision-generic peak detection
// --------------------------------------------------------------------------

typedef struct {
    FLOAT freq;
    FLOAT power;
    FLOAT cond;
} TLS(peak);

static inline void TLS(quadratic_vertex)(FLOAT x0, FLOAT y0, FLOAT x1, FLOAT y1, FLOAT x2, FLOAT y2, FLOAT *vx, FLOAT *vy) {
    FLOAT slope01 = DIV(SUB(y1, y0), SUB(x1, x0));
    FLOAT slope12 = DIV(SUB(y2, y1), SUB(x2, x1));
    FLOAT curvature = DIV(SUB(slope12, slope01), SUB(x2, x0));
    if (FLOAT_IS_ZERO(curvature)) {
        *vx = x1;
        *vy = y1;
        return;
    }
    FLOAT linear = SUB(slope01, MUL(curvature, ADD(x0, x1)));
    *vx = DIV(NEG(linear), MUL(FCONST(2.0), curvature));
    *vy = ADD(y0, ADD(MUL(slope01, SUB(*vx, x0)), MUL(MUL(curvature, SUB(*vx, x0)), SUB(*vx, x1))));
}

static inline FLOAT TLS(quadratic_value)(FLOAT x, FLOAT x0, FLOAT y0, FLOAT x1, FLOAT y1, FLOAT x2, FLOAT y2) {
    FLOAT slope01 = DIV(SUB(y1, y0), SUB(x1, x0));
    FLOAT slope12 = DIV(SUB(y2, y1), SUB(x2, x1));
    FLOAT curvature = DIV(SUB(slope12, slope01), SUB(x2, x0));
    return ADD(y0, ADD(MUL(slope01, SUB(x, x0)), MUL(MUL(curvature, SUB(x, x0)), SUB(x, x1))));
}

static inline FLOAT TLS(linear_interp)(FLOAT x, FLOAT x0, FLOAT y0, FLOAT x1, FLOAT y1) {
    FLOAT t = DIV(SUB(x, x0), SUB(x1, x0));
    return ADD(y0, MUL(t, SUB(y1, y0)));
}

static void TLS(insert_peak)(TLS(peak) peak, FLOAT *out_freq, FLOAT *out_power, FLOAT *out_cond, int has_cond, int *count, int max_peaks) {
    if (max_peaks <= 0) return;
    if (*count == max_peaks && FLOAT_CMP(peak.power, out_power[max_peaks - 1]) <= 0) return;

    int pos = *count;
    if (pos == max_peaks) {
        pos = max_peaks - 1;
    } else {
        ++(*count);
    }
    while (pos > 0 && FLOAT_CMP(peak.power, out_power[pos - 1]) > 0) {
        out_freq[pos] = out_freq[pos - 1];
        out_power[pos] = out_power[pos - 1];
        if (has_cond) out_cond[pos] = out_cond[pos - 1];
        --pos;
    }
    out_freq[pos] = peak.freq;
    out_power[pos] = peak.power;
    if (has_cond) out_cond[pos] = peak.cond;
}

int TLS(get_peaks)(const FLOAT *freq, const FLOAT *power, const FLOAT *cond, int n, int max_peaks, FLOAT threshold, FLOAT *out_freq, FLOAT *out_power,
                   FLOAT *out_cond, int *out_count) {
    if (!out_count || n < 0 || max_peaks < 0 || !FLOAT_IS_FINITE(threshold)) return TLS_UTIL_ERR_ARGUMENT;
    *out_count = 0;
    if (n > 0 && (!freq || !power)) return TLS_UTIL_ERR_ARGUMENT;
    if (max_peaks > 0 && (!out_freq || !out_power || (cond && !out_cond))) return TLS_UTIL_ERR_ARGUMENT;

    for (int i = 0; i < n; ++i) {
        if (!FLOAT_IS_FINITE(freq[i])) return TLS_UTIL_ERR_ARGUMENT;
        if (i > 0 && FLOAT_CMP(freq[i], freq[i - 1]) <= 0) return TLS_UTIL_ERR_ARGUMENT;
    }
    if (n < 3 || max_peaks == 0) return TLS_UTIL_OK;

    int count = 0;
    for (int idx = 1; idx < n - 1; ++idx) {
        FLOAT p0 = power[idx - 1];
        FLOAT p1 = power[idx];
        FLOAT p2 = power[idx + 1];
        if (!FLOAT_IS_FINITE(p0) || !FLOAT_IS_FINITE(p1) || !FLOAT_IS_FINITE(p2)) continue;
        if (!(FLOAT_CMP(p1, p0) > 0 && FLOAT_CMP(p1, threshold) > 0)) continue;

        int cmp_right = FLOAT_CMP(p1, p2);
        if (cmp_right > 0) {
            FLOAT vertex_freq;
            FLOAT vertex_power;
            TLS(quadratic_vertex)(freq[idx - 1], p0, freq[idx], p1, freq[idx + 1], p2, &vertex_freq, &vertex_power);
            TLS(peak) peak;
            peak.freq = vertex_freq;
            peak.power = vertex_power;
            NANOFFT_DIAG_PUSH_NAN
            peak.cond = FLOAT_NAN;
            NANOFFT_DIAG_POP
            if (cond) {
                FLOAT c0 = cond[idx - 1];
                FLOAT c1 = cond[idx];
                FLOAT c2 = cond[idx + 1];
                if (FLOAT_IS_FINITE(c0) && FLOAT_IS_FINITE(c1) && FLOAT_IS_FINITE(c2)) {
                    peak.cond = TLS(quadratic_value)(vertex_freq, freq[idx - 1], c0, freq[idx], c1, freq[idx + 1], c2);
                }
            }
            TLS(insert_peak)(peak, out_freq, out_power, out_cond, cond != NULL, &count, max_peaks);
        } else if (cmp_right == 0) {
            int end = idx + 1;
            while (end + 1 < n && FLOAT_IS_FINITE(power[end + 1]) && FLOAT_CMP(power[end + 1], p1) == 0) {
                ++end;
            }
            if (end + 1 < n && FLOAT_IS_FINITE(power[end + 1]) && FLOAT_CMP(p1, power[end + 1]) > 0) {
                FLOAT mid_freq = MUL(FCONST(0.5), ADD(freq[idx], freq[end]));
                TLS(peak) peak;
                peak.freq = mid_freq;
                peak.power = p1;
                NANOFFT_DIAG_PUSH_NAN
                peak.cond = FLOAT_NAN;
                NANOFFT_DIAG_POP
                if (cond) {
                    int all_finite = 1;
                    for (int k = idx; k <= end; ++k) {
                        if (!FLOAT_IS_FINITE(cond[k])) {
                            all_finite = 0;
                            break;
                        }
                    }
                    if (all_finite) {
                        int lo = idx;
                        while (lo + 1 < end && FLOAT_CMP(freq[lo + 1], mid_freq) < 0) {
                            ++lo;
                        }
                        if (FLOAT_CMP(freq[lo + 1], mid_freq) == 0) {
                            peak.cond = cond[lo + 1];
                        } else {
                            peak.cond = TLS(linear_interp)(mid_freq, freq[lo], cond[lo], freq[lo + 1], cond[lo + 1]);
                        }
                    }
                }
                TLS(insert_peak)(peak, out_freq, out_power, out_cond, cond != NULL, &count, max_peaks);
            }
            idx = end;
        }
    }

    *out_count = count;
    return TLS_UTIL_OK;
}