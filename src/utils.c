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
        if (!(FLOAT_CMP(p1, p0) > 0 && FLOAT_CMP(p1, p2) > 0 && FLOAT_CMP(p1, threshold) > 0)) continue;

        FLOAT vertex_freq;
        FLOAT vertex_power;
        TLS(quadratic_vertex)(freq[idx - 1], p0, freq[idx], p1, freq[idx + 1], p2, &vertex_freq, &vertex_power);
        TLS(peak) peak;
        peak.freq = vertex_freq;
        peak.power = vertex_power;
        peak.cond = FLOAT_NAN;
        if (cond) {
            FLOAT c0 = cond[idx - 1];
            FLOAT c1 = cond[idx];
            FLOAT c2 = cond[idx + 1];
            if (FLOAT_IS_FINITE(c0) && FLOAT_IS_FINITE(c1) && FLOAT_IS_FINITE(c2)) {
                peak.cond = TLS(quadratic_value)(vertex_freq, freq[idx - 1], c0, freq[idx], c1, freq[idx + 1], c2);
            }
        }
        TLS(insert_peak)(peak, out_freq, out_power, out_cond, cond != NULL, &count, max_peaks);
    }

    *out_count = count;
    return TLS_UTIL_OK;
}