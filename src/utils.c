#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <utils.h>

#ifndef MAX_TWIDDLE_REUSE
#    define MAX_TWIDDLE_REUSE 16
#endif

#if MAX_TWIDDLE_REUSE < 2 || (MAX_TWIDDLE_REUSE & (MAX_TWIDDLE_REUSE - 1)) != 0
#    error "MAX_TWIDDLE_REUSE must be a power of two greater than or equal to 2"
#endif

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
    float freq;
    float power;
    float cond;
} tlsf_peak;

typedef struct {
    double freq;
    double power;
    double cond;
} tls_peak;

typedef struct {
    dd_t freq;
    dd_t power;
    dd_t cond;
} tlsdd_peak;

static int cmp_tlsf_peak_desc(const void *left, const void *right) {
    const tlsf_peak *a = (const tlsf_peak *)left;
    const tlsf_peak *b = (const tlsf_peak *)right;
    return (a->power < b->power) - (a->power > b->power);
}

static int cmp_tls_peak_desc(const void *left, const void *right) {
    const tls_peak *a = (const tls_peak *)left;
    const tls_peak *b = (const tls_peak *)right;
    return (a->power < b->power) - (a->power > b->power);
}

static inline void quadratic_vertex_float(float x0, float y0, float x1, float y1, float x2, float y2, float *vx, float *vy) {
    float slope01 = (y1 - y0) / (x1 - x0);
    float slope12 = (y2 - y1) / (x2 - x1);
    float curvature = (slope12 - slope01) / (x2 - x0);
    if (curvature == 0.0f) {
        *vx = x1;
        *vy = y1;
        return;
    }
    float linear = slope01 - curvature * (x0 + x1);
    *vx = -linear / (2.0f * curvature);
    *vy = y0 + slope01 * (*vx - x0) + curvature * (*vx - x0) * (*vx - x1);
}

static inline float quadratic_value_float(float x, float x0, float y0, float x1, float y1, float x2, float y2) {
    float slope01 = (y1 - y0) / (x1 - x0);
    float slope12 = (y2 - y1) / (x2 - x1);
    float curvature = (slope12 - slope01) / (x2 - x0);
    return y0 + slope01 * (x - x0) + curvature * (x - x0) * (x - x1);
}

static inline void quadratic_vertex_double(double x0, double y0, double x1, double y1, double x2, double y2, double *vx, double *vy) {
    double slope01 = (y1 - y0) / (x1 - x0);
    double slope12 = (y2 - y1) / (x2 - x1);
    double curvature = (slope12 - slope01) / (x2 - x0);
    if (curvature == 0.0) {
        *vx = x1;
        *vy = y1;
        return;
    }
    double linear = slope01 - curvature * (x0 + x1);
    *vx = -linear / (2.0 * curvature);
    *vy = y0 + slope01 * (*vx - x0) + curvature * (*vx - x0) * (*vx - x1);
}

static inline double quadratic_value_double(double x, double x0, double y0, double x1, double y1, double x2, double y2) {
    double slope01 = (y1 - y0) / (x1 - x0);
    double slope12 = (y2 - y1) / (x2 - x1);
    double curvature = (slope12 - slope01) / (x2 - x0);
    return y0 + slope01 * (x - x0) + curvature * (x - x0) * (x - x1);
}

int tlsf_get_peaks(const float *freq, const float *power, const float *cond, int n, float threshold, float *out_freq, float *out_power, float *out_cond,
                   int *out_count) {
    if (!out_count || n < 0 || !float_is_finite_bits(threshold)) return TLS_UTIL_ERR_ARGUMENT;
    *out_count = 0;
    if (n > 0 && (!freq || !power)) return TLS_UTIL_ERR_ARGUMENT;
    if (n >= 3 && (!out_freq || !out_power || (cond && !out_cond))) return TLS_UTIL_ERR_ARGUMENT;

    for (int i = 0; i < n; ++i) {
        if (!float_is_finite_bits(freq[i])) return TLS_UTIL_ERR_ARGUMENT;
        if (i > 0 && freq[i] <= freq[i - 1]) return TLS_UTIL_ERR_ARGUMENT;
    }
    if (n < 3) return TLS_UTIL_OK;

    tlsf_peak *peaks = (tlsf_peak *)malloc((size_t)(n - 2) * sizeof(*peaks));
    if (!peaks) return TLS_UTIL_ERR_ALLOC;

    int count = 0;
    for (int idx = 1; idx < n - 1; ++idx) {
        float p0 = power[idx - 1];
        float p1 = power[idx];
        float p2 = power[idx + 1];
        if (!float_is_finite_bits(p0) || !float_is_finite_bits(p1) || !float_is_finite_bits(p2)) continue;
        if (!(p1 > p0 && p1 > p2 && p1 > threshold)) continue;

        float vertex_freq;
        float vertex_power;
        quadratic_vertex_float(freq[idx - 1], p0, freq[idx], p1, freq[idx + 1], p2, &vertex_freq, &vertex_power);
        peaks[count].freq = vertex_freq;
        peaks[count].power = vertex_power;
        peaks[count].cond = NAN;
        if (cond) {
            float c0 = cond[idx - 1];
            float c1 = cond[idx];
            float c2 = cond[idx + 1];
            if (float_is_finite_bits(c0) && float_is_finite_bits(c1) && float_is_finite_bits(c2)) {
                peaks[count].cond = quadratic_value_float(vertex_freq, freq[idx - 1], c0, freq[idx], c1, freq[idx + 1], c2);
            }
        }
        ++count;
    }

    qsort(peaks, (size_t)count, sizeof(*peaks), cmp_tlsf_peak_desc);
    for (int i = 0; i < count; ++i) {
        out_freq[i] = peaks[i].freq;
        out_power[i] = peaks[i].power;
        if (cond) out_cond[i] = peaks[i].cond;
    }
    free(peaks);
    *out_count = count;
    return TLS_UTIL_OK;
}

int tls_get_peaks(const double *freq, const double *power, const double *cond, int n, double threshold, double *out_freq, double *out_power, double *out_cond,
                  int *out_count) {
    if (!out_count || n < 0 || !double_is_finite_bits(threshold)) return TLS_UTIL_ERR_ARGUMENT;
    *out_count = 0;
    if (n > 0 && (!freq || !power)) return TLS_UTIL_ERR_ARGUMENT;
    if (n >= 3 && (!out_freq || !out_power || (cond && !out_cond))) return TLS_UTIL_ERR_ARGUMENT;

    for (int i = 0; i < n; ++i) {
        if (!double_is_finite_bits(freq[i])) return TLS_UTIL_ERR_ARGUMENT;
        if (i > 0 && freq[i] <= freq[i - 1]) return TLS_UTIL_ERR_ARGUMENT;
    }
    if (n < 3) return TLS_UTIL_OK;

    tls_peak *peaks = (tls_peak *)malloc((size_t)(n - 2) * sizeof(*peaks));
    if (!peaks) return TLS_UTIL_ERR_ALLOC;

    int count = 0;
    for (int idx = 1; idx < n - 1; ++idx) {
        double p0 = power[idx - 1];
        double p1 = power[idx];
        double p2 = power[idx + 1];
        if (!double_is_finite_bits(p0) || !double_is_finite_bits(p1) || !double_is_finite_bits(p2)) continue;
        if (!(p1 > p0 && p1 > p2 && p1 > threshold)) continue;

        double vertex_freq;
        double vertex_power;
        quadratic_vertex_double(freq[idx - 1], p0, freq[idx], p1, freq[idx + 1], p2, &vertex_freq, &vertex_power);
        peaks[count].freq = vertex_freq;
        peaks[count].power = vertex_power;
        peaks[count].cond = NAN;
        if (cond) {
            double c0 = cond[idx - 1];
            double c1 = cond[idx];
            double c2 = cond[idx + 1];
            if (double_is_finite_bits(c0) && double_is_finite_bits(c1) && double_is_finite_bits(c2)) {
                peaks[count].cond = quadratic_value_double(vertex_freq, freq[idx - 1], c0, freq[idx], c1, freq[idx + 1], c2);
            }
        }
        ++count;
    }

    qsort(peaks, (size_t)count, sizeof(*peaks), cmp_tls_peak_desc);
    for (int i = 0; i < count; ++i) {
        out_freq[i] = peaks[i].freq;
        out_power[i] = peaks[i].power;
        if (cond) out_cond[i] = peaks[i].cond;
    }
    free(peaks);
    *out_count = count;
    return TLS_UTIL_OK;
}

static inline int dd_is_finite_bits(dd_t value) { return double_is_finite_bits(value.hi) && double_is_finite_bits(value.lo); }

static inline int dd_cmp(dd_t left, dd_t right) {
    if (left.hi < right.hi) return -1;
    if (left.hi > right.hi) return 1;
    if (left.lo < right.lo) return -1;
    if (left.lo > right.lo) return 1;
    return 0;
}

static inline int dd_is_zero(dd_t value) { return value.hi == 0.0 && value.lo == 0.0; }

static inline dd_t dd_neg(dd_t value) { return dd_make(-value.hi, -value.lo); }

static inline void quadratic_vertex_dd(dd_t x0, dd_t y0, dd_t x1, dd_t y1, dd_t x2, dd_t y2, dd_t *vx, dd_t *vy) {
    dd_t slope01 = dd_div(dd_sub(y1, y0), dd_sub(x1, x0));
    dd_t slope12 = dd_div(dd_sub(y2, y1), dd_sub(x2, x1));
    dd_t curvature = dd_div(dd_sub(slope12, slope01), dd_sub(x2, x0));
    if (dd_is_zero(curvature)) {
        *vx = x1;
        *vy = y1;
        return;
    }
    dd_t linear = dd_sub(slope01, dd_mul(curvature, dd_add(x0, x1)));
    *vx = dd_div(dd_neg(linear), dd_mul(dd_make(2.0, 0.0), curvature));
    *vy = dd_add(y0, dd_add(dd_mul(slope01, dd_sub(*vx, x0)), dd_mul(dd_mul(curvature, dd_sub(*vx, x0)), dd_sub(*vx, x1))));
}

static inline dd_t quadratic_value_dd(dd_t x, dd_t x0, dd_t y0, dd_t x1, dd_t y1, dd_t x2, dd_t y2) {
    dd_t slope01 = dd_div(dd_sub(y1, y0), dd_sub(x1, x0));
    dd_t slope12 = dd_div(dd_sub(y2, y1), dd_sub(x2, x1));
    dd_t curvature = dd_div(dd_sub(slope12, slope01), dd_sub(x2, x0));
    return dd_add(y0, dd_add(dd_mul(slope01, dd_sub(x, x0)), dd_mul(dd_mul(curvature, dd_sub(x, x0)), dd_sub(x, x1))));
}

static int cmp_tlsdd_peak_desc(const void *left, const void *right) {
    const tlsdd_peak *a = (const tlsdd_peak *)left;
    const tlsdd_peak *b = (const tlsdd_peak *)right;
    return -dd_cmp(a->power, b->power);
}

int tlsdd_get_peaks(const dd_t *freq, const dd_t *power, const dd_t *cond, int n, dd_t threshold, dd_t *out_freq, dd_t *out_power, dd_t *out_cond,
                    int *out_count) {
    if (!out_count || n < 0 || !dd_is_finite_bits(threshold)) return TLS_UTIL_ERR_ARGUMENT;
    *out_count = 0;
    if (n > 0 && (!freq || !power)) return TLS_UTIL_ERR_ARGUMENT;
    if (n >= 3 && (!out_freq || !out_power || (cond && !out_cond))) return TLS_UTIL_ERR_ARGUMENT;

    for (int i = 0; i < n; ++i) {
        if (!dd_is_finite_bits(freq[i])) return TLS_UTIL_ERR_ARGUMENT;
        if (i > 0 && dd_cmp(freq[i], freq[i - 1]) <= 0) return TLS_UTIL_ERR_ARGUMENT;
    }
    if (n < 3) return TLS_UTIL_OK;

    tlsdd_peak *peaks = (tlsdd_peak *)malloc((size_t)(n - 2) * sizeof(*peaks));
    if (!peaks) return TLS_UTIL_ERR_ALLOC;

    int count = 0;
    for (int idx = 1; idx < n - 1; ++idx) {
        dd_t p0 = power[idx - 1];
        dd_t p1 = power[idx];
        dd_t p2 = power[idx + 1];
        if (!dd_is_finite_bits(p0) || !dd_is_finite_bits(p1) || !dd_is_finite_bits(p2)) continue;
        if (!(dd_cmp(p1, p0) > 0 && dd_cmp(p1, p2) > 0 && dd_cmp(p1, threshold) > 0)) continue;

        dd_t vertex_freq;
        dd_t vertex_power;
        quadratic_vertex_dd(freq[idx - 1], p0, freq[idx], p1, freq[idx + 1], p2, &vertex_freq, &vertex_power);
        peaks[count].freq = vertex_freq;
        peaks[count].power = vertex_power;
        peaks[count].cond = dd_make(NAN, NAN);
        if (cond) {
            dd_t c0 = cond[idx - 1];
            dd_t c1 = cond[idx];
            dd_t c2 = cond[idx + 1];
            if (dd_is_finite_bits(c0) && dd_is_finite_bits(c1) && dd_is_finite_bits(c2)) {
                peaks[count].cond = quadratic_value_dd(vertex_freq, freq[idx - 1], c0, freq[idx], c1, freq[idx + 1], c2);
            }
        }
        ++count;
    }

    qsort(peaks, (size_t)count, sizeof(*peaks), cmp_tlsdd_peak_desc);
    for (int i = 0; i < count; ++i) {
        out_freq[i] = peaks[i].freq;
        out_power[i] = peaks[i].power;
        if (cond) out_cond[i] = peaks[i].cond;
    }
    free(peaks);
    *out_count = count;
    return TLS_UTIL_OK;
}
