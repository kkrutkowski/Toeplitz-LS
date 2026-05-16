#include <linalg.h>
#define NANOFFT_NEEDS_INTERNAL_VEC
#include <math.h>
#include <nanofft_precision.h>
#include <nufft1.h>
#include <scaling.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* Return codes from the public fastchi2 entry points. */
enum { CHI2PER_OK = 0, CHI2PER_ERR_ARGUMENT = -1, CHI2PER_ERR_BACKEND = -2, CHI2PER_ERR_ALLOC = -3, CHI2PER_ERR_DEGENERATE = -4, CHI2PER_ERR_SOLVER = -5 };

enum { CHI2PER_SOLVER_LEVINSON = 1, CHI2PER_SOLVER_ZOHAR = 2, CHI2PER_SOLVER_BAREISS = 3, CHI2PER_SOLVER_LDLT = 4 };

static inline int bitceil(double x) { return x <= 1.0 ? 1 : 1 << (1 + (int)(log2(x))); }

static double approximate_cost(int N, int M, int block, int degree, double alpha, double beta, double gamma) {
    // Include cost of zero-padding frequencies to the transform length
    double N_eff = block * ceil((double)N / block);
    // Reduction in the cost of precomputation caused by reusage of pre-generated
    // plans
    double gamma_eff = gamma * ((double)((2 * degree) + 1)) / (double)((3 * degree) + 1);
    // FFT execution cost
    double cost = N_eff * pow((double)block, alpha);
    // Frequency shift cost
    cost += beta * (N_eff - (double)(block)) * (double)(M) / (double)(block);
    // Precomputation cost per block size
    cost += gamma_eff * block;
    return cost;
}

// start at block = bitceil(pow((beta * M / alpha), (1.0 / (alpha + 1.0))))
// then bitshift downwards as long, as cost decreases with each bitshift
static int optimize_plan_size(int N, int M, int degree, double alpha, double beta, double gamma) {
    double start = pow((beta * (double)M / alpha), 1.0 / (alpha + 1.0));
    int block = bitceil(start);
    int n_cap = bitceil((double)N);

    if (block > n_cap) block = n_cap;
    if (block < 1) block = 1;

    double best = approximate_cost(N, M, block, degree, alpha, beta, gamma);
    while (block > 1) {
        int next = block >> 1;
        double next_cost = approximate_cost(N, M, next, degree, alpha, beta, gamma);
        if (next_cost >= best) break;
        block = next;
        best = next_cost;
    }
    return block;
}

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
#    define NUFFT_RANK 27
#    define NUFFT_W 32
#    define CHI2_ALPHA DD_ALPHA
#    define CHI2_LRA_BETA DD_LRA_BETA
#    define CHI2_LRA_GAMMA DD_LRA_GAMMA
#    define CHI2_PSWF_BETA DD_PSWF_BETA
#    define CHI2_PSWF_GAMMA DD_PSWF_GAMMA
#    define COND_SINGULARITY_THRESHOLD_LEVINSON 1e24
#    define COND_SINGULARITY_THRESHOLD_ZOHAR 1e24
#    define COND_SINGULARITY_THRESHOLD_BAREISS 1e24
#    define COND_SINGULARITY_THRESHOLD_LDLT 1e10
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
#        define NUFFT_RANK 16
#        define NUFFT_W 16
#        define CHI2_ALPHA D_ALPHA
#        define CHI2_LRA_BETA D_LRA_BETA
#        define CHI2_LRA_GAMMA D_LRA_GAMMA
#        define CHI2_PSWF_BETA D_PSWF_BETA
#        define CHI2_PSWF_GAMMA D_PSWF_GAMMA
#        define COND_SINGULARITY_THRESHOLD_LEVINSON 1e12
#        define COND_SINGULARITY_THRESHOLD_ZOHAR 1e12
#        define COND_SINGULARITY_THRESHOLD_BAREISS 1e12
#        define COND_SINGULARITY_THRESHOLD_LDLT 1e5
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
#        define NUFFT_RANK 9
#        define NUFFT_W 8
#        define CHI2_ALPHA F_ALPHA
#        define CHI2_LRA_BETA F_LRA_BETA
#        define CHI2_LRA_GAMMA F_LRA_GAMMA
#        define CHI2_PSWF_BETA F_PSWF_BETA
#        define CHI2_PSWF_GAMMA F_PSWF_GAMMA
#        define COND_SINGULARITY_THRESHOLD_LEVINSON 2e5
#        define COND_SINGULARITY_THRESHOLD_ZOHAR 2e5
#        define COND_SINGULARITY_THRESHOLD_BAREISS 2e5
#        define COND_SINGULARITY_THRESHOLD_LDLT 1e2
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
    0.0, COND_SINGULARITY_THRESHOLD_LEVINSON, COND_SINGULARITY_THRESHOLD_ZOHAR, COND_SINGULARITY_THRESHOLD_BAREISS, COND_SINGULARITY_THRESHOLD_LDLT,
};

static inline double condition_singularity_threshold(int solver) {
    if (solver < CHI2PER_SOLVER_LEVINSON || solver > CHI2PER_SOLVER_LDLT) return 0.0;
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
    if (backend == 0) {
        NUFFT_PSWF_EXEC((PSWF_PLAN_T *)plan, src_r, src_i, out_r, out_i, freq_factor);
    } else if (backend == 1) {
        NUFFT_LRA_EXEC((const LRA_PLAN_T *)plan, src_r, src_i, out_r, out_i, freq_factor);
    } else {
        return CHI2PER_ERR_BACKEND;
    }
    return CHI2PER_OK;
}

static int compute_trig_sums(const FLOAT *tc, const FLOAT *h, int M, double f0, double df, int N, int max_factor, int block, int backend, void *plan, FLOAT *S,
                             FLOAT *C, FLOAT *src_r, FLOAT *src_i, FLOAT *delta_r, FLOAT *delta_i, FLOAT *out_r, FLOAT *out_i) {
    for (int q = 1; q <= max_factor; ++q) {
        FLOAT qf0 = FCAST((double)q * f0);
        FLOAT q_delta = FCAST((double)q * df * (double)block);

        for (int m = 0; m < M; ++m) {
            FLOAT tm = tc[m];
            FLOAT phase0 = MUL(qf0, tm);
            FLOAT c0 = M_COS2PI(phase0);
            FLOAT s0 = M_SIN2PI(phase0);

            src_r[m] = MUL(h[m], c0);
            src_i[m] = MUL(h[m], s0);

            FLOAT phase_delta = MUL(q_delta, tm);
            delta_r[m] = M_COS2PI(phase_delta);
            delta_i[m] = M_SIN2PI(phase_delta);
        }

        for (int base = 0; base < N; base += block) {
            int status = execute_nufft_block(plan, backend, src_r, src_i, out_r, out_i, q);
            if (status != CHI2PER_OK) {
                return status;
            }

            int count = (base + block <= N) ? block : (N - base);
            for (int k = 0; k < count; ++k) {
                C[(size_t)q * N + base + k] = out_r[k];
                S[(size_t)q * N + base + k] = out_i[k];
            }

            if (base + block < N) {
                for (int m = 0; m < M; ++m) {
                    FLOAT yr = src_r[m];
                    FLOAT yi = src_i[m];
                    FLOAT dr = delta_r[m];
                    FLOAT di = delta_i[m];
                    src_r[m] = SUB(MUL(yr, dr), MUL(yi, di));
                    src_i[m] = ADD(MUL(yr, di), MUL(yi, dr));
                }
            }
        }
    }

    return CHI2PER_OK;
}

// Implementation of Generalized Scargle Periodogram using 3d NuFFTs instead of
// 4 used for generic AoV Solution by: Zechmeister and M. Kurster, A&A 496,
// 577-584 (2009) Trig identity: Press W.H. and Rybicki, G.B, "Fast algorithm
// for spectral analysis of unevenly sampled data". ApJ 1:338, p277, 1989
static int gls_impl(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, FLOAT ws, FLOAT yws, FLOAT chi2_ref, FLOAT *power) {
    FLOAT half = FCAST(0.5);
    FLOAT sqrt_half = M_SQRT(half);
    FLOAT inv_ws = DIV(FCAST(1.0), ws);
    FLOAT inv_chi2_ref = DIV(FCAST(1.0), chi2_ref);

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
        power[idx] = MUL(dot, inv_chi2_ref);
    }

    return CHI2PER_OK;
}

#if defined(DOUBLE_DOUBLE)
static int solve_periodogram_ldlt_dd(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, FLOAT chi2_ref, FLOAT *power) {
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

    FLOAT inv_chi2_ref = DIV(FCAST(1.0), chi2_ref);

    for (int idx = 0; idx < N; ++idx) {
        for (int row = 0; row < norder; ++row) {
            XTy[row] = real_rhs_value(Syw, Cyw, N, idx, row);
            for (int col = 0; col < norder; ++col) {
                XTX[(size_t)row * norder + col] = real_gram_value(Sw, Cw, N, idx, row, col);
            }
        }

        FLOAT condition_bound = SOLVE_LDLT((size_t)norder, XTX, XTy, X, L, D, Y, Z);
        if (condition_bound_is_singular(condition_bound, CHI2PER_SOLVER_LDLT)) {
            power[idx] = FCAST(NAN);
            continue;
        }

        FLOAT dot = FCAST(0.0);
        for (int k = 0; k < norder; ++k) dot = ADD(dot, MUL(XTy[k], X[k]));
        power[idx] = MUL(dot, inv_chi2_ref);
    }

    free(XTX);
    free(XTy);
    free(X);
    free(L);
    free(D);
    free(Y);
    free(Z);
    return CHI2PER_OK;
}

static int solve_periodogram_dd(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, int solver, FLOAT chi2_ref,
                                FLOAT *power) {
    if (solver == CHI2PER_SOLVER_LDLT) return solve_periodogram_ldlt_dd(Sw, Cw, Syw, Cyw, N, degree, chi2_ref, power);

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

    FLOAT inv_chi2_ref = DIV(FCAST(1.0), chi2_ref);

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

        if (solver == CHI2PER_SOLVER_LEVINSON) {
            FLOAT condition_bound = SOLVE_LEVINSON((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ar, Ai, Apr, Api);
            if (condition_bound_is_singular(condition_bound, solver)) {
                power[idx] = FCAST(NAN);
                continue;
            }
        } else if (solver == CHI2PER_SOLVER_ZOHAR) {
            FLOAT condition_bound = SOLVE_ZOHAR((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ar, Ai, Apr, Api);
            if (condition_bound_is_singular(condition_bound, solver)) {
                power[idx] = FCAST(NAN);
                continue;
            }
        } else if (solver == CHI2PER_SOLVER_BAREISS) {
            FLOAT condition_bound = SOLVE_BAREISS((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, BUr, BUi, BD, Bur, Bui, Bvr, Bvi, Bwr, Bwi);
            if (condition_bound_is_singular(condition_bound, solver)) {
                power[idx] = FCAST(NAN);
                continue;
            }
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

        FLOAT dot = FCAST(0.0);
        for (int k = 0; k < norder; ++k) dot = ADD(dot, ADD(MUL(Yr[k], Xr[k]), MUL(Yi[k], Xi[k])));
        power[idx] = MUL(dot, inv_chi2_ref);
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
    return CHI2PER_OK;
}
#else
static int solve_periodogram_ldlt_vec(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, FLOAT chi2_ref, FLOAT *power) {
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

    FLOAT inv_chi2_ref = DIV(FCAST(1.0), chi2_ref);

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
            if (condition_bound_is_singular(condition_bound[lane], CHI2PER_SOLVER_LDLT)) {
                power[idx] = FCAST(NAN);
                continue;
            }

            FLOAT dot = FCAST(0.0);
            for (int k = 0; k < norder; ++k) {
                dot = ADD(dot, MUL(XTy[k][lane], X[k][lane]));
            }
            power[idx] = MUL(dot, inv_chi2_ref);
        }
    }

    free(XTX);
    free(XTy);
    free(X);
    free(L);
    free(D);
    free(Y);
    free(Z);
    return CHI2PER_OK;
}

static int solve_periodogram_vec(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, int solver, FLOAT chi2_ref,
                                 FLOAT *power) {
    if (solver == CHI2PER_SOLVER_LDLT) return solve_periodogram_ldlt_vec(Sw, Cw, Syw, Cyw, N, degree, chi2_ref, power);

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

    FLOAT inv_chi2_ref = DIV(FCAST(1.0), chi2_ref);

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

        if (solver == CHI2PER_SOLVER_LEVINSON) {
            INTERNAL_VEC condition_bound = SOLVE_LEVINSON((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ehr, Ehi, Ehpr, Ehpi);
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx < N && condition_bound_is_singular(condition_bound[lane], solver)) {
                    singular_lane[lane] = 1;
                    power[idx] = FCAST(NAN);
                }
            }
        } else if (solver == CHI2PER_SOLVER_ZOHAR) {
            INTERNAL_VEC condition_bound = SOLVE_ZOHAR((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ehr, Ehi, Ehpr, Ehpi);
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx < N && condition_bound_is_singular(condition_bound[lane], solver)) {
                    singular_lane[lane] = 1;
                    power[idx] = FCAST(NAN);
                }
            }
        } else if (solver == CHI2PER_SOLVER_BAREISS) {
            INTERNAL_VEC condition_bound = SOLVE_BAREISS((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, BUr, BUi, BD, Bur, Bui, Bvr, Bvi, Bwr, Bwi);
            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx < N && condition_bound_is_singular(condition_bound[lane], solver)) {
                    singular_lane[lane] = 1;
                    power[idx] = FCAST(NAN);
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
            power[idx] = MUL(dot, inv_chi2_ref);
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
    return CHI2PER_OK;
}
#endif

int CHI2_PREFIX(fastchi2)(const TIME_INPUT_T *t, const FLOAT *y, const FLOAT *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                          FLOAT *power) {
    if (!t || !y || !dy || !power || M <= 0 || N <= 0 || degree <= 0 || f0 < 0.0 || df <= 0.0) return CHI2PER_ERR_ARGUMENT;
    if (backend < 0 || backend >= 2) return CHI2PER_ERR_BACKEND;
    if (solver != CHI2PER_SOLVER_LEVINSON && solver != CHI2PER_SOLVER_ZOHAR && solver != CHI2PER_SOLVER_BAREISS && solver != CHI2PER_SOLVER_LDLT)
        return CHI2PER_ERR_SOLVER;
    if (M < 2 * degree + 2) return CHI2PER_ERR_DEGENERATE;

    double beta = (backend == 0) ? CHI2_PSWF_BETA : CHI2_LRA_BETA;
    double gamma = (backend == 0) ? CHI2_PSWF_GAMMA : CHI2_LRA_GAMMA;
    int block = optimize_plan_size(N, M, degree, CHI2_ALPHA, beta, gamma);
    int max_factor = 2 * degree;

    FLOAT *w = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *yw = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *tc = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    NUFFT_INPUT_T *x = (NUFFT_INPUT_T *)checked_aligned_malloc((size_t)M, sizeof(NUFFT_INPUT_T));
    FLOAT *Sw = (FLOAT *)checked_malloc((size_t)(max_factor + 1) * N, sizeof(FLOAT));
    FLOAT *Cw = (FLOAT *)checked_malloc((size_t)(max_factor + 1) * N, sizeof(FLOAT));
    FLOAT *Syw = (FLOAT *)checked_malloc((size_t)(degree + 1) * N, sizeof(FLOAT));
    FLOAT *Cyw = (FLOAT *)checked_malloc((size_t)(degree + 1) * N, sizeof(FLOAT));
    FLOAT *src_r = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *src_i = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *delta_r = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *delta_i = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *out_r = (FLOAT *)checked_aligned_malloc((size_t)block, sizeof(FLOAT));
    FLOAT *out_i = (FLOAT *)checked_aligned_malloc((size_t)block, sizeof(FLOAT));
    void *plan = NULL;
    int status = CHI2PER_OK;

    if (!w || !yw || !tc || !x || !Sw || !Cw || !Syw || !Cyw || !src_r || !src_i || !delta_r || !delta_i || !out_r || !out_i) {
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
        tc[m] = time_to_float(centered_t);
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

    if (backend == 0) {
        plan = NUFFT_PSWF_INIT(M, block, NUFFT_W, df, max_factor);
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

    status = compute_trig_sums(tc, w, M, f0, df, N, max_factor, block, backend, plan, Sw, Cw, src_r, src_i, delta_r, delta_i, out_r, out_i);
    if (status == CHI2PER_OK)
        status = compute_trig_sums(tc, yw, M, f0, df, N, degree, block, backend, plan, Syw, Cyw, src_r, src_i, delta_r, delta_i, out_r, out_i);

    if (status == CHI2PER_OK && degree == 1) {
        status = gls_impl(Sw, Cw, Syw, Cyw, N, ws, yws, chi2_ref, power);
    } else if (status == CHI2PER_OK) {
#if defined(DOUBLE_DOUBLE)
        status = solve_periodogram_dd(Sw, Cw, Syw, Cyw, N, degree, solver, chi2_ref, power);
#else
        status = solve_periodogram_vec(Sw, Cw, Syw, Cyw, N, degree, solver, chi2_ref, power);
#endif
    }

cleanup:
    if (plan && backend == 0)
        NUFFT_PSWF_FREE((PSWF_PLAN_T *)plan);
    else if (plan)
        NUFFT_LRA_FREE((LRA_PLAN_T *)plan);

    free(w);
    free(yw);
    free(tc);
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
