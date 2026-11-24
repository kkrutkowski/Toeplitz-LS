#include <math.h>
#include <nanofft.h>
#include <nanofft_precision.h>
#include <nufft1.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tls.h>
#include <tls_internal.h>

/* ---------------------------------------------------------------------------
 * Intercept aligned_alloc to enforce safe SIMD memory padding.
 * ------------------------------------------------------------------------- */
static inline void *safe_aligned_alloc(size_t align, size_t size) {
    size_t pad = 128; /* minimum 128 bytes to prevent SIMD overruns */
    if (align > pad) pad = align;
    size_t padded_size = (size + pad - 1) & ~(pad - 1);
    return aligned_alloc(align, padded_size);
}
#define aligned_alloc safe_aligned_alloc

/* High-resolution monotonic timer */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ===========================================================================
 * Precision-dependent NuFFT API dispatch macros.
 *
 * Ranks and widths match those used in finufft_bench.cpp:
 *   single:        rank_f  =  9,  w_f = 8
 *   double:        rank_d  = 16,  w_d = 16
 *   double-double: rank_dd = 27,  w_dd = 32
 * ========================================================================= */
#if defined(DOUBLE_DOUBLE)
#    define NUFFT_LRA_INIT tlsdd_nufft_lra_initialize
#    define NUFFT_LRA_PRE tlsdd_nufft_lra_precompute
#    define NUFFT_LRA_EXEC tlsdd_nufft_lra_execute
#    define NUFFT_LRA_FREE tlsdd_nufft_free_lra_plan
#    define NUFFT_PSWF_INIT tlsdd_nufft_pswf_initialize
#    define NUFFT_PSWF_PRE tlsdd_nufft_pswf_precompute
#    define NUFFT_PSWF_EXEC tlsdd_nufft_pswf_execute
#    define NUFFT_PSWF_FREE tlsdd_nufft_free_pswf_plan
#    define NUFFT_LRA_INIT_OPTS tlsdd_nufft_lra_initialize_opts
#    define NUFFT_LRA_EXEC_BREADTH tlsdd_nufft_lra_execute_breadth
#    define NUFFT_PSWF_INIT_OPTS tlsdd_nufft_pswf_initialize_opts
#    define NUFFT_PSWF_EXEC_BREADTH tlsdd_nufft_pswf_execute_breadth
#    define LRA_PLAN_T tlsdd_nufft_lra_plan
#    define PSWF_PLAN_T tlsdd_nufft_pswf_plan
#    define NUFFT_RANK 27
#    define NUFFT_W21 32
#    define NUFFT_W43 36
#elif defined(DOUBLE)
#    define NUFFT_LRA_INIT tls_nufft_lra_initialize
#    define NUFFT_LRA_PRE tls_nufft_lra_precompute
#    define NUFFT_LRA_EXEC tls_nufft_lra_execute
#    define NUFFT_LRA_FREE tls_nufft_free_lra_plan
#    define NUFFT_PSWF_INIT tls_nufft_pswf_initialize
#    define NUFFT_PSWF_INIT_MODE tls_nufft_pswf_initialize_mode
#    define NUFFT_PSWF_BENCH_PPOLY tls_nufft_pswf_bench_ppoly_gen
#    define NUFFT_PSWF_PRE tls_nufft_pswf_precompute
#    define NUFFT_PSWF_EXEC tls_nufft_pswf_execute
#    define NUFFT_PSWF_FREE tls_nufft_free_pswf_plan
#    define NUFFT_LRA_INIT_OPTS tls_nufft_lra_initialize_opts
#    define NUFFT_LRA_EXEC_BREADTH tls_nufft_lra_execute_breadth
#    define NUFFT_PSWF_INIT_OPTS tls_nufft_pswf_initialize_opts
#    define NUFFT_PSWF_EXEC_BREADTH tls_nufft_pswf_execute_breadth
#    define LRA_PLAN_T tls_nufft_lra_plan
#    define PSWF_PLAN_T tls_nufft_pswf_plan
#    define NUFFT_RANK 16
#    define NUFFT_W21 16
#    define NUFFT_W43 18
#elif defined(HALF)
#    define NUFFT_LRA_INIT tlsh_nufft_lra_initialize
#    define NUFFT_LRA_PRE tlsh_nufft_lra_precompute
#    define NUFFT_LRA_EXEC tlsh_nufft_lra_execute
#    define NUFFT_LRA_FREE tlsh_nufft_free_lra_plan
#    define NUFFT_PSWF_INIT tlsh_nufft_pswf_initialize
#    define NUFFT_PSWF_INIT_MODE tlsh_nufft_pswf_initialize_mode
#    define NUFFT_PSWF_BENCH_PPOLY tlsh_nufft_pswf_bench_ppoly_gen
#    define NUFFT_PSWF_PRE tlsh_nufft_pswf_precompute
#    define NUFFT_PSWF_EXEC tlsh_nufft_pswf_execute
#    define NUFFT_PSWF_FREE tlsh_nufft_free_pswf_plan
#    define NUFFT_LRA_INIT_OPTS tlsh_nufft_lra_initialize_opts
#    define NUFFT_LRA_EXEC_BREADTH tlsh_nufft_lra_execute_breadth
#    define NUFFT_PSWF_INIT_OPTS tlsh_nufft_pswf_initialize_opts
#    define NUFFT_PSWF_EXEC_BREADTH tlsh_nufft_pswf_execute_breadth
#    define LRA_PLAN_T tlsh_nufft_lra_plan
#    define PSWF_PLAN_T tlsh_nufft_pswf_plan
#    define NUFFT_RANK 5
#    define NUFFT_W21 4
#    define NUFFT_W43 5
#else /* single (default) */
#    define NUFFT_LRA_INIT tlsf_nufft_lra_initialize
#    define NUFFT_LRA_PRE tlsf_nufft_lra_precompute
#    define NUFFT_LRA_EXEC tlsf_nufft_lra_execute
#    define NUFFT_LRA_FREE tlsf_nufft_free_lra_plan
#    define NUFFT_PSWF_INIT tlsf_nufft_pswf_initialize
#    define NUFFT_PSWF_INIT_MODE tlsf_nufft_pswf_initialize_mode
#    define NUFFT_PSWF_BENCH_PPOLY tlsf_nufft_pswf_bench_ppoly_gen
#    define NUFFT_PSWF_PRE tlsf_nufft_pswf_precompute
#    define NUFFT_PSWF_EXEC tlsf_nufft_pswf_execute
#    define NUFFT_PSWF_FREE tlsf_nufft_free_pswf_plan
#    define NUFFT_LRA_INIT_OPTS tlsf_nufft_lra_initialize_opts
#    define NUFFT_LRA_EXEC_BREADTH tlsf_nufft_lra_execute_breadth
#    define NUFFT_PSWF_INIT_OPTS tlsf_nufft_pswf_initialize_opts
#    define NUFFT_PSWF_EXEC_BREADTH tlsf_nufft_pswf_execute_breadth
#    define LRA_PLAN_T tlsf_nufft_lra_plan
#    define PSWF_PLAN_T tlsf_nufft_pswf_plan
#    define NUFFT_RANK 9
#    define NUFFT_W21 8
#    define NUFFT_W43 9
#endif

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

/* ---- Precision-dependent FFT sweep parameters ---- */
#if defined(DOUBLE_DOUBLE)
    int k_start = 8, k_end = 20;
    const char *precision_name = "Double-Double";
#elif defined(DOUBLE)
    int k_start = 8, k_end = 20;
    const char *precision_name = "Double";
#elif defined(HALF)
    int k_start = 9, k_end = 22;
    const char *precision_name = "Half";
#else
    int k_start = 9, k_end = 22;
    const char *precision_name = "Single";
#endif

    /* NuFFT constants — must match finufft_bench.cpp */
    const double df = 1.0;
    const int freq_factor = 1;
    const int rank = NUFFT_RANK;
    const int w21 = NUFFT_W21;
    const int w43 = NUFFT_W43;

    /* =========================================================================
     * Phase 1: FFT backend sweep → alpha, unit_fft
     *
     * Model: T/N = unit_fft * N^alpha  =>  T = unit_fft * N^(1 + alpha)
     * ======================================================================= */
    printf("--- Running nanoFFT Benchmark (%s Precision) ---\n", precision_name);

    int warmup_runs = 1024;
    int runs = 1 << 15; /* 32768 */

    double log2_N[30];
    double log2_T_per_N[30];
    int pt_idx = 0;

    for (int k = k_start; k <= k_end; k++) {
        uint32_t N = 1u << k;

        FLOAT *re = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));
        FLOAT *im = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));

        for (uint32_t i = 0; i < N; i++) {
            double r1 = (double)rand() / RAND_MAX;
            double r2 = (double)rand() / RAND_MAX;
#if defined(DOUBLE_DOUBLE)
            re[i] = dd_make(r1, 0.0);
            im[i] = dd_make(r2, 0.0);
#else
            re[i] = FCAST(r1);
            im[i] = FCAST(r2);
#endif
        }

        PREFIX(plan) *plan = PREFIX(make_plan)(N);

        for (int i = 0; i < warmup_runs; i++) PREFIX(execute)(plan, re, im);

        double t0 = get_time_sec();
        for (int i = 0; i < runs; i++) PREFIX(execute)(plan, re, im);
        double total = get_time_sec() - t0;
        double per_run = total / runs;

        log2_N[pt_idx] = (double)k;
        log2_T_per_N[pt_idx] = log2(per_run / N);
        pt_idx++;

        printf(
            "k=%2d | N=%8u | runs=%6d | warmups=%4d | "
            "total_time: %6.2fs | time/run: %6.2e s\n",
            k, N, runs, warmup_runs, total, per_run);

        if (total > 0.5) {
            runs /= 4;
            warmup_runs /= 4;
        } else if (total > 0.1) {
            runs /= 2;
            warmup_runs /= 2;
        }
        if (runs < 2) runs = 2;
        if (warmup_runs < 1) warmup_runs = 1;

        PREFIX(destroy_plan)(plan);
        free(re);
        free(im);
    }

    /* Log-log least-squares regression for alpha */
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < pt_idx; i++) {
        sx += log2_N[i];
        sy += log2_T_per_N[i];
        sxx += log2_N[i] * log2_N[i];
        sxy += log2_N[i] * log2_T_per_N[i];
    }
    double mx = sx / pt_idx, my = sy / pt_idx;
    double num_r = 0.0, den_r = 0.0;
    for (int i = 0; i < pt_idx; i++) {
        num_r += (log2_N[i] - mx) * (log2_T_per_N[i] - my);
        den_r += (log2_N[i] - mx) * (log2_N[i] - mx);
    }
    double alpha = num_r / den_r;
    double unit_fft = pow(2.0, my - alpha * mx);

    printf("\n--- FFT Complexity Fit (%s) ---\n", precision_name);
    printf("Model    : T = unit_fft * N^(1 + alpha)\n");
    printf("unit_fft : %e s\n", unit_fft);
    printf("alpha    : %.6f\n\n", alpha);

    /* =========================================================================
     * Phase 2: Derived NuFFT constant factors
     *
     * LRA:  rank independent size-N FFTs  → unit_lra  = rank * unit_fft
     * PSWF21: one size-2N FFT (2x oversampling)
     * PSWF43: one size-4N/3 FFT, represented as 2/3 of the PSWF21 unit
     * ======================================================================= */
    double unit_lra = (double)rank * unit_fft;
    double unit_pswf21 = unit_fft * pow(2.0, 1.0 + alpha);
    double unit_pswf43 = unit_pswf21 * (2.0 / 3.0);

    /* =========================================================================
     * Phase 3: gamma — precompute scaling coefficient  (N >> M regime)
     *
     * Fixed point: M_gamma = 4, N_gamma = 2^15
     * gamma_X = time_precompute / (N_gamma * unit_X)
     *
     * Warmup is used to avoid cold-cache artefacts, matching the alpha
     * measurement methodology.  Run counts are adaptive.
     * ======================================================================= */
    const int N_gamma = 1 << 15; /* 32768 — N >> M */
#if defined(DOUBLE_DOUBLE)
    const int N_gamma_pswf = N_gamma / 16;
#else
    const int N_gamma_pswf = N_gamma;
#endif
    const int M_gamma = 4;

    double *x_gamma = (double *)malloc(M_gamma * sizeof(double));
#if defined(DOUBLE_DOUBLE)
    dd_t *x_gamma_dd = (dd_t *)malloc(M_gamma * sizeof(dd_t));
#endif
    for (int i = 0; i < M_gamma; i++) {
        x_gamma[i] = (double)rand() / RAND_MAX;
#if defined(DOUBLE_DOUBLE)
        x_gamma_dd[i] = dd_make(x_gamma[i], 0.0);
#endif
    }

    double gamma_lra, gamma_pswf21, gamma_pswf43;

    /* --- LRA gamma --- */
    {
        LRA_PLAN_T *plan = NUFFT_LRA_INIT(M_gamma, N_gamma, rank, df, freq_factor);

        int wups = 4, reps = 32;

#if defined(DOUBLE_DOUBLE)
        for (int i = 0; i < wups; i++) NUFFT_LRA_PRE(plan, x_gamma_dd, M_gamma, N_gamma, rank);
#else
        for (int i = 0; i < wups; i++) NUFFT_LRA_PRE(plan, x_gamma, M_gamma, N_gamma, rank);
#endif

        double t0 = get_time_sec();
#if defined(DOUBLE_DOUBLE)
        for (int i = 0; i < reps; i++) NUFFT_LRA_PRE(plan, x_gamma_dd, M_gamma, N_gamma, rank);
#else
        for (int i = 0; i < reps; i++) NUFFT_LRA_PRE(plan, x_gamma, M_gamma, N_gamma, rank);
#endif
        double total = get_time_sec() - t0;
        double t_pre = total / reps;

        NUFFT_LRA_FREE(plan);
        gamma_lra = t_pre / ((double)N_gamma * unit_lra);
        printf("[gamma LRA ] precompute/run = %e s | gamma = %.6f\n", t_pre, gamma_lra);
    }

    /* --- PSWF21 gamma --- */
    {
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT(M_gamma, N_gamma_pswf, w21, df, freq_factor, "21");

        int wups = 4, reps = 32;

#if defined(DOUBLE_DOUBLE)
        for (int i = 0; i < wups; i++) NUFFT_PSWF_PRE(plan, x_gamma_dd);
#else
        for (int i = 0; i < wups; i++) NUFFT_PSWF_PRE(plan, x_gamma);
#endif

        double t0 = get_time_sec();
#if defined(DOUBLE_DOUBLE)
        for (int i = 0; i < reps; i++) NUFFT_PSWF_PRE(plan, x_gamma_dd);
#else
        for (int i = 0; i < reps; i++) NUFFT_PSWF_PRE(plan, x_gamma);
#endif
        double total = get_time_sec() - t0;
        double t_pre = total / reps;

        NUFFT_PSWF_FREE(plan);
        gamma_pswf21 = t_pre / ((double)N_gamma_pswf * unit_pswf21);
        printf("[gamma PSWF21] precompute/run = %e s | gamma = %.6f\n", t_pre, gamma_pswf21);
    }

    /* --- PSWF43 gamma --- */
    {
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT(M_gamma, N_gamma_pswf, w43, df, freq_factor, "43");
        int N_gamma_out43 = N_gamma_pswf + (N_gamma_pswf >> 1);

        int wups = 4, reps = 32;

#if defined(DOUBLE_DOUBLE)
        for (int i = 0; i < wups; i++) NUFFT_PSWF_PRE(plan, x_gamma_dd);
#else
        for (int i = 0; i < wups; i++) NUFFT_PSWF_PRE(plan, x_gamma);
#endif

        double t0 = get_time_sec();
#if defined(DOUBLE_DOUBLE)
        for (int i = 0; i < reps; i++) NUFFT_PSWF_PRE(plan, x_gamma_dd);
#else
        for (int i = 0; i < reps; i++) NUFFT_PSWF_PRE(plan, x_gamma);
#endif
        double total = get_time_sec() - t0;
        double t_pre = total / reps;

        NUFFT_PSWF_FREE(plan);
        gamma_pswf43 = t_pre / ((double)N_gamma_out43 * unit_pswf43);
        printf("[gamma PSWF43] precompute/run = %e s | gamma = %.6f\n", t_pre, gamma_pswf43);
    }

    free(x_gamma);
#if defined(DOUBLE_DOUBLE)
    free(x_gamma_dd);
#endif

    /* =========================================================================
     * Phase 4: beta — execute scaling coefficient  (M >> N regime)
     *
     * Fixed point: N_beta = 64, M_beta = 2^15
     *
     * Each timed iteration (plan pre-generated, not included):
     *   1. y_real[i] *= dy_real[i] - y_imag[i] * dy_imag[i]   (complex rotation)
     *   2. y_imag[i]  = y_real_old[i] * dy_imag[i] + y_imag[i] * dy_real[i]
     *   3. nufft_execute(plan, y_real, y_imag, out_real, out_imag, freq_factor)
     *
     * dy vectors are precomputed once outside the timed section:
     *   dy_real[i] = M_SIN2PI(x[i] * df * N_beta)
     *   dy_imag[i] = M_COS2PI(x[i] * df * N_beta)
     *
     *
     * beta_X = t_exec / (M_beta * unit_X)
     * ======================================================================= */
    const int N_beta = 64;
    const int N_beta_out43 = N_beta + (N_beta >> 1);
    const int M_beta = 1 << 15; /* 32768 — M >> N */

    double *x_beta = (double *)malloc(M_beta * sizeof(double));
#if defined(DOUBLE_DOUBLE)
    dd_t *x_beta_dd = (dd_t *)malloc(M_beta * sizeof(dd_t));
#endif
    FLOAT *y_re_src = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *y_im_src = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *y_re_run = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *y_im_run = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *out_re = (FLOAT *)aligned_alloc(64, N_beta_out43 * sizeof(FLOAT));
    FLOAT *out_im = (FLOAT *)aligned_alloc(64, N_beta_out43 * sizeof(FLOAT));
    FLOAT *dy_re = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *dy_im = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));

    /* Generate x positions and source y values */
    for (int i = 0; i < M_beta; i++) {
        double r1 = (double)rand() / RAND_MAX;
        double r2 = (double)rand() / RAND_MAX;
        x_beta[i] = (double)rand() / RAND_MAX;
#if defined(DOUBLE_DOUBLE)
        x_beta_dd[i] = dd_make(x_beta[i], 0.0);
#endif
#if defined(DOUBLE_DOUBLE)
        y_re_src[i] = dd_make(r1, 0.0);
        y_im_src[i] = dd_make(r2, 0.0);
#else
        y_re_src[i] = FCAST(r1);
        y_im_src[i] = FCAST(r2);
#endif
    }

    /* Precompute PSWF21/LRA phase-twist vectors (not included in beta timing) */
    for (int i = 0; i < M_beta; i++) {
        FLOAT arg = MUL(MUL(FCAST(x_beta[i]), FCAST(df)), FCAST((double)N_beta));
        dy_re[i] = M_SIN2PI(arg);
        dy_im[i] = M_COS2PI(arg);
    }

    double beta_lra, beta_pswf21, beta_pswf43;
    int wups_b = 4, reps_b = 32;

    /* --- LRA beta --- */
    {
        LRA_PLAN_T *plan = NUFFT_LRA_INIT(M_beta, N_beta, rank, df, freq_factor);
#if defined(DOUBLE_DOUBLE)
        NUFFT_LRA_PRE(plan, x_beta_dd, M_beta, N_beta, rank);
#else
        NUFFT_LRA_PRE(plan, x_beta, M_beta, N_beta, rank);
#endif

        /* Warmup: rotate, execute */
        for (int i = 0; i < wups_b; i++) {
            double t0 = get_time_sec();
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_LRA_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
            (void)(get_time_sec() - t0); /* discard warmup timing */
        }

        /* Timed runs: (rotate + execute) */
        double t_total = 0.0;
        for (int i = 0; i < reps_b; i++) {
            double t0 = get_time_sec();
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_LRA_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
            t_total += get_time_sec() - t0;
        }
        double t_exec = t_total / reps_b;

        NUFFT_LRA_FREE(plan);
        beta_lra = t_exec / ((double)M_beta * unit_lra);
        printf("[beta  LRA ] exec/run = %e s | beta  = %.6f\n", t_exec, beta_lra);
    }

    /* --- PSWF21 beta --- */
    {
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT(M_beta, N_beta, w21, df, freq_factor, "21");
#if defined(DOUBLE_DOUBLE)
        NUFFT_PSWF_PRE(plan, x_beta_dd);
#else
        NUFFT_PSWF_PRE(plan, x_beta);
#endif

        /* Warmup */
        for (int i = 0; i < wups_b; i++) {
            memcpy(y_re_run, y_re_src, M_beta * sizeof(FLOAT));
            memcpy(y_im_run, y_im_src, M_beta * sizeof(FLOAT));
            double t0 = get_time_sec();
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_PSWF_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
            (void)(get_time_sec() - t0);
        }

        /* Timed runs */
        double t_total = 0.0;
        for (int i = 0; i < reps_b; i++) {
            memcpy(y_re_run, y_re_src, M_beta * sizeof(FLOAT));
            memcpy(y_im_run, y_im_src, M_beta * sizeof(FLOAT));
            double t0 = get_time_sec();
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_PSWF_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
            t_total += get_time_sec() - t0;
        }
        double t_exec = t_total / reps_b;

        NUFFT_PSWF_FREE(plan);
        beta_pswf21 = t_exec / ((double)M_beta * unit_pswf21);
        printf("[beta  PSWF21] exec/run = %e s | beta  = %.6f\n", t_exec, beta_pswf21);
    }

    /* Precompute PSWF43 phase-twist vectors (not included in beta timing). */
    for (int i = 0; i < M_beta; i++) {
        FLOAT arg = MUL(MUL(FCAST(x_beta[i]), FCAST(df)), FCAST((double)N_beta_out43));
        dy_re[i] = M_SIN2PI(arg);
        dy_im[i] = M_COS2PI(arg);
    }

    /* --- PSWF43 beta --- */
    {
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT(M_beta, N_beta, w43, df, freq_factor, "43");
#if defined(DOUBLE_DOUBLE)
        NUFFT_PSWF_PRE(plan, x_beta_dd);
#else
        NUFFT_PSWF_PRE(plan, x_beta);
#endif

        /* Warmup */
        for (int i = 0; i < wups_b; i++) {
            memcpy(y_re_run, y_re_src, M_beta * sizeof(FLOAT));
            memcpy(y_im_run, y_im_src, M_beta * sizeof(FLOAT));
            double t0 = get_time_sec();
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_PSWF_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
            (void)(get_time_sec() - t0);
        }

        /* Timed runs */
        double t_total = 0.0;
        for (int i = 0; i < reps_b; i++) {
            memcpy(y_re_run, y_re_src, M_beta * sizeof(FLOAT));
            memcpy(y_im_run, y_im_src, M_beta * sizeof(FLOAT));
            double t0 = get_time_sec();
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_PSWF_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
            t_total += get_time_sec() - t0;
        }
        double t_exec = t_total / reps_b;

        NUFFT_PSWF_FREE(plan);
        beta_pswf43 = t_exec / ((double)M_beta * unit_pswf43);
        printf("[beta  PSWF43] exec/run = %e s | beta  = %.6f\n", t_exec, beta_pswf43);
    }

#if !defined(DOUBLE_DOUBLE)
    /* =========================================================================
     * Phase 5: PSWF spreading comparison (minmax vs on-the-fly piecewise)
     *
     * Measures:
     *   - ppoly_const: constant runtime piecewise-polynomial table generation
     *   - pre_mm / pre_pp: O(M) per-factor precompute cost (excluding deconv)
     *   - beta_pp: O(M) per-execute spreading + rotation cost for piecewise
     * ======================================================================= */
    double pre_pswf21 = 0.0, pre_pswf43 = 0.0;
    double beta_pp_pswf21 = 0.0, beta_pp_pswf43 = 0.0;
    double pre_pp_pswf21 = 0.0, pre_pp_pswf43 = 0.0;
    double ppoly_const_pswf21 = 0.0, ppoly_const_pswf43 = 0.0;

    /* --- Piecewise table generation constant cost --- */
    {
        const int gen_wups = 16;
        const int gen_reps = 256;
        NUFFT_PSWF_BENCH_PPOLY(w21, "21", gen_wups);
        double t0 = get_time_sec();
        NUFFT_PSWF_BENCH_PPOLY(w21, "21", gen_reps);
        double t_gen21 = (get_time_sec() - t0) / (double)gen_reps;
        ppoly_const_pswf21 = t_gen21 / unit_pswf21;
        printf("[ppoly PSWF21] gen/run = %e s | const = %.6f\n", t_gen21, ppoly_const_pswf21);

        NUFFT_PSWF_BENCH_PPOLY(w43, "43", gen_wups);
        t0 = get_time_sec();
        NUFFT_PSWF_BENCH_PPOLY(w43, "43", gen_reps);
        double t_gen43 = (get_time_sec() - t0) / (double)gen_reps;
        ppoly_const_pswf43 = t_gen43 / unit_pswf43;
        printf("[ppoly PSWF43] gen/run = %e s | const = %.6f\n", t_gen43, ppoly_const_pswf43);
    }

    /* --- O(M) precompute cost: minmax vs piecewise --- */
    {
        const int pre_wups = 4;
        const int pre_reps = 16;

        /* PSWF21 minmax pre */
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT_MODE(M_beta, N_beta, w21, df, freq_factor, "21", TLS_SPREAD_MINMAX);
        for (int i = 0; i < pre_wups; i++) NUFFT_PSWF_PRE(plan, x_beta);
        double t0 = get_time_sec();
        for (int i = 0; i < pre_reps; i++) NUFFT_PSWF_PRE(plan, x_beta);
        double t_pre_mm21 = (get_time_sec() - t0) / (double)pre_reps;
        NUFFT_PSWF_FREE(plan);
        double t_dec21 = gamma_pswf21 * (double)N_beta * unit_pswf21;
        if (t_pre_mm21 > t_dec21) t_pre_mm21 -= t_dec21;
        pre_pswf21 = t_pre_mm21 / ((double)M_beta * unit_pswf21);

        /* PSWF21 piecewise pre */
        plan = NUFFT_PSWF_INIT_MODE(M_beta, N_beta, w21, df, freq_factor, "21", TLS_SPREAD_PIECEWISE);
        for (int i = 0; i < pre_wups; i++) NUFFT_PSWF_PRE(plan, x_beta);
        t0 = get_time_sec();
        for (int i = 0; i < pre_reps; i++) NUFFT_PSWF_PRE(plan, x_beta);
        double t_pre_pp21 = (get_time_sec() - t0) / (double)pre_reps;
        NUFFT_PSWF_FREE(plan);
        if (t_pre_pp21 > t_dec21) t_pre_pp21 -= t_dec21;
        pre_pp_pswf21 = t_pre_pp21 / ((double)M_beta * unit_pswf21);
        printf("[pre   PSWF21] minmax = %e s (%.6f) | ppoly = %e s (%.6f)\n", t_pre_mm21, pre_pswf21, t_pre_pp21, pre_pp_pswf21);

        /* PSWF43 minmax pre */
        plan = NUFFT_PSWF_INIT_MODE(M_beta, N_beta, w43, df, freq_factor, "43", TLS_SPREAD_MINMAX);
        for (int i = 0; i < pre_wups; i++) NUFFT_PSWF_PRE(plan, x_beta);
        t0 = get_time_sec();
        for (int i = 0; i < pre_reps; i++) NUFFT_PSWF_PRE(plan, x_beta);
        double t_pre_mm43 = (get_time_sec() - t0) / (double)pre_reps;
        NUFFT_PSWF_FREE(plan);
        double t_dec43 = gamma_pswf43 * (double)N_beta_out43 * unit_pswf43;
        if (t_pre_mm43 > t_dec43) t_pre_mm43 -= t_dec43;
        pre_pswf43 = t_pre_mm43 / ((double)M_beta * unit_pswf43);

        /* PSWF43 piecewise pre */
        plan = NUFFT_PSWF_INIT_MODE(M_beta, N_beta, w43, df, freq_factor, "43", TLS_SPREAD_PIECEWISE);
        for (int i = 0; i < pre_wups; i++) NUFFT_PSWF_PRE(plan, x_beta);
        t0 = get_time_sec();
        for (int i = 0; i < pre_reps; i++) NUFFT_PSWF_PRE(plan, x_beta);
        double t_pre_pp43 = (get_time_sec() - t0) / (double)pre_reps;
        NUFFT_PSWF_FREE(plan);
        if (t_pre_pp43 > t_dec43) t_pre_pp43 -= t_dec43;
        pre_pp_pswf43 = t_pre_pp43 / ((double)M_beta * unit_pswf43);
        printf("[pre   PSWF43] minmax = %e s (%.6f) | ppoly = %e s (%.6f)\n", t_pre_mm43, pre_pswf43, t_pre_pp43, pre_pp_pswf43);
    }

    /* --- PSWF21 piecewise beta --- */
    {
        for (int i = 0; i < M_beta; i++) {
            FLOAT arg = MUL(MUL(FCAST(x_beta[i]), FCAST(df)), FCAST((double)N_beta));
            dy_re[i] = M_SIN2PI(arg);
            dy_im[i] = M_COS2PI(arg);
        }
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT_MODE(M_beta, N_beta, w21, df, freq_factor, "21", TLS_SPREAD_PIECEWISE);
        NUFFT_PSWF_PRE(plan, x_beta);

        for (int i = 0; i < wups_b; i++) {
            memcpy(y_re_run, y_re_src, M_beta * sizeof(FLOAT));
            memcpy(y_im_run, y_im_src, M_beta * sizeof(FLOAT));
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_PSWF_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
        }

        double t_total = 0.0;
        for (int i = 0; i < reps_b; i++) {
            memcpy(y_re_run, y_re_src, M_beta * sizeof(FLOAT));
            memcpy(y_im_run, y_im_src, M_beta * sizeof(FLOAT));
            double t0 = get_time_sec();
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_PSWF_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
            t_total += get_time_sec() - t0;
        }
        double t_exec = t_total / reps_b;
        NUFFT_PSWF_FREE(plan);
        beta_pp_pswf21 = t_exec / ((double)M_beta * unit_pswf21);
        printf("[beta  PSWF21 ppoly] exec/run = %e s | beta  = %.6f\n", t_exec, beta_pp_pswf21);
    }

    /* --- PSWF43 piecewise beta --- */
    {
        for (int i = 0; i < M_beta; i++) {
            FLOAT arg = MUL(MUL(FCAST(x_beta[i]), FCAST(df)), FCAST((double)N_beta_out43));
            dy_re[i] = M_SIN2PI(arg);
            dy_im[i] = M_COS2PI(arg);
        }
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT_MODE(M_beta, N_beta, w43, df, freq_factor, "43", TLS_SPREAD_PIECEWISE);
        NUFFT_PSWF_PRE(plan, x_beta);

        for (int i = 0; i < wups_b; i++) {
            memcpy(y_re_run, y_re_src, M_beta * sizeof(FLOAT));
            memcpy(y_im_run, y_im_src, M_beta * sizeof(FLOAT));
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_PSWF_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
        }

        double t_total = 0.0;
        for (int i = 0; i < reps_b; i++) {
            memcpy(y_re_run, y_re_src, M_beta * sizeof(FLOAT));
            memcpy(y_im_run, y_im_src, M_beta * sizeof(FLOAT));
            double t0 = get_time_sec();
            for (int j = 0; j < M_beta; j++) {
                FLOAT yr = y_re_run[j];
                FLOAT yi = y_im_run[j];
                y_re_run[j] = SUB(MUL(yr, dy_re[j]), MUL(yi, dy_im[j]));
                y_im_run[j] = ADD(MUL(yr, dy_im[j]), MUL(yi, dy_re[j]));
            }
            NUFFT_PSWF_EXEC(plan, y_re_run, y_im_run, out_re, out_im, freq_factor);
            t_total += get_time_sec() - t0;
        }
        double t_exec = t_total / reps_b;
        NUFFT_PSWF_FREE(plan);
        beta_pp_pswf43 = t_exec / ((double)M_beta * unit_pswf43);
        printf("[beta  PSWF43 ppoly] exec/run = %e s | beta  = %.6f\n", t_exec, beta_pp_pswf43);
    }
#endif

    /* =========================================================================
     * Phase 6: Breadth-wise SIMD calibration
     *
     * Measures:
     *   - b_alpha, b_fft_ratio = unit_bfft / unit_fft from execute_breadth
     *   - b_beta for LRA, PSWF21 (minmax [+ ppoly]), PSWF43 (minmax [+ ppoly])
     *   - b_pre for PSWF21/PSWF43 (minmax [+ ppoly])
     * ======================================================================= */
    double b_alpha = alpha;
    double b_fft_ratio = (double)BREADTH_VEC_LEN;
    double b_beta_lra = 0.0, b_beta_pswf21 = 0.0, b_beta_pswf43 = 0.0;
    double b_pre_pswf21 = 0.0, b_pre_pswf43 = 0.0;
#if !defined(DOUBLE_DOUBLE)
    double b_beta_pp_pswf21 = 0.0, b_beta_pp_pswf43 = 0.0;
    double b_pre_pp_pswf21 = 0.0, b_pre_pp_pswf43 = 0.0;
#endif

    /* --- Breadth FFT sweep --- */
    {
#if defined(DOUBLE_DOUBLE)
        const int bk_start = 5, bk_end = 13;
        int bwups = 64, bruns = 1024;
#else
        const int bk_start = 5, bk_end = 15;
        int bwups = 256, bruns = 8192;
#endif
        double blog2_N[30], blog2_T_per_N[30];
        int bpt = 0;
        for (int k = bk_start; k <= bk_end; ++k) {
            uint32_t N = 1u << k;
            BREADTH_VEC *bre = (BREADTH_VEC *)aligned_alloc(64, (size_t)N * sizeof(BREADTH_VEC));
            BREADTH_VEC *bim = (BREADTH_VEC *)aligned_alloc(64, (size_t)N * sizeof(BREADTH_VEC));
            for (uint32_t i = 0; i < N; ++i) {
                BREADTH_COMPUTE_VEC vr = BREADTH_ZERO, vi = BREADTH_ZERO;
                for (int l = 0; l < BREADTH_VEC_LEN; ++l) {
                    BREADTH_SET_LANE(vr, l, FCAST((double)rand() / RAND_MAX));
                    BREADTH_SET_LANE(vi, l, FCAST((double)rand() / RAND_MAX));
                }
                STORE_BVEC(&bre[i], vr);
                STORE_BVEC(&bim[i], vi);
            }
            PREFIX(plan) *bplan = PREFIX(make_plan_breadth)(N);
            for (int i = 0; i < bwups; ++i) PREFIX(execute_breadth)(bplan, bre, bim);
            double t0 = get_time_sec();
            for (int i = 0; i < bruns; ++i) PREFIX(execute_breadth)(bplan, bre, bim);
            double total = get_time_sec() - t0;
            double per_run = total / (double)bruns;
            blog2_N[bpt] = (double)k;
            blog2_T_per_N[bpt] = log2(per_run / (double)N);
            bpt++;
            printf("[bfft] k=%2d | N=%6u | runs=%5d | time/batch: %6.2e s\n", k, N, bruns, per_run);
            if (total > 0.3) {
                bruns /= 4;
                bwups /= 4;
            } else if (total > 0.08) {
                bruns /= 2;
                bwups /= 2;
            }
            if (bruns < 4) bruns = 4;
            if (bwups < 1) bwups = 1;
            PREFIX(destroy_plan)(bplan);
            free(bre);
            free(bim);
        }
        double bsx = 0.0, bsy = 0.0;
        for (int i = 0; i < bpt; ++i) {
            bsx += blog2_N[i];
            bsy += blog2_T_per_N[i];
        }
        double bmx = bsx / bpt, bmy = bsy / bpt;
        double bnum = 0.0, bden = 0.0;
        for (int i = 0; i < bpt; ++i) {
            bnum += (blog2_N[i] - bmx) * (blog2_T_per_N[i] - bmy);
            bden += (blog2_N[i] - bmx) * (blog2_N[i] - bmx);
        }
        b_alpha = bnum / bden;
        double unit_bfft = pow(2.0, bmy - b_alpha * bmx);
        b_fft_ratio = unit_bfft / unit_fft;
        printf("[bfft  fit] b_alpha = %.6f | unit_bfft = %e s | b_fft_ratio = %.6f\n", b_alpha, unit_bfft, b_fft_ratio);
    }

    /* --- Breadth beta & precompute --- */
    {
        BREADTH_VEC *bout_re = (BREADTH_VEC *)aligned_alloc(64, (size_t)N_beta_out43 * sizeof(BREADTH_VEC));
        BREADTH_VEC *bout_im = (BREADTH_VEC *)aligned_alloc(64, (size_t)N_beta_out43 * sizeof(BREADTH_VEC));
#if defined(DOUBLE_DOUBLE)
        const int bwups_b = 2, breps_b = 8;
        const int bpre_wups = 2, bpre_reps = 6;
        const dd_t *xb_in = x_beta_dd;
#else
        const int bwups_b = 4, breps_b = 32;
        const int bpre_wups = 4, bpre_reps = 16;
        const double *xb_in = x_beta;
#endif
        const double f0_b = 0.125;

        /* LRA breadth beta */
        {
            LRA_PLAN_T *plan = NUFFT_LRA_INIT_OPTS(M_beta, N_beta, rank, df, freq_factor, TLS_VEC_BREADTH);
            NUFFT_LRA_PRE(plan, xb_in, M_beta, N_beta, rank);
            for (int i = 0; i < bwups_b; ++i) NUFFT_LRA_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta, 0, freq_factor, bout_re, bout_im);
            double t0 = get_time_sec();
            for (int i = 0; i < breps_b; ++i) NUFFT_LRA_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta, 0, freq_factor, bout_re, bout_im);
            double t_exec = (get_time_sec() - t0) / (double)breps_b;
            NUFFT_LRA_FREE(plan);
            b_beta_lra = t_exec / ((double)M_beta * unit_lra);
            printf("[bbeta LRA   ] exec/batch = %e s | b_beta = %.6f\n", t_exec, b_beta_lra);
        }

        /* PSWF21 breadth minmax pre + beta */
        {
            PSWF_PLAN_T *plan = NUFFT_PSWF_INIT_OPTS(M_beta, N_beta, w21, df, freq_factor, "21", TLS_SPREAD_MINMAX, TLS_VEC_BREADTH);
            for (int i = 0; i < bpre_wups; ++i) NUFFT_PSWF_PRE(plan, xb_in);
            double t0 = get_time_sec();
            for (int i = 0; i < bpre_reps; ++i) NUFFT_PSWF_PRE(plan, xb_in);
            double t_pre = (get_time_sec() - t0) / (double)bpre_reps;
            double t_dec21 = gamma_pswf21 * (double)N_beta * unit_pswf21;
            if (t_pre > t_dec21) t_pre -= t_dec21;
            b_pre_pswf21 = t_pre / ((double)M_beta * unit_pswf21);

            for (int i = 0; i < bwups_b; ++i)
                NUFFT_PSWF_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta, 0, freq_factor, TLS_NUFFT_OUT_NATURAL, bout_re, bout_im);
            t0 = get_time_sec();
            for (int i = 0; i < breps_b; ++i)
                NUFFT_PSWF_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta, 0, freq_factor, TLS_NUFFT_OUT_NATURAL, bout_re, bout_im);
            double t_exec = (get_time_sec() - t0) / (double)breps_b;
            NUFFT_PSWF_FREE(plan);
            b_beta_pswf21 = t_exec / ((double)M_beta * unit_pswf21);
            printf("[bbeta PSWF21] exec/batch = %e s (%.6f) | b_pre = %.6f\n", t_exec, b_beta_pswf21, b_pre_pswf21);
        }

        /* PSWF43 breadth minmax pre + beta */
        {
            PSWF_PLAN_T *plan = NUFFT_PSWF_INIT_OPTS(M_beta, N_beta, w43, df, freq_factor, "43", TLS_SPREAD_MINMAX, TLS_VEC_BREADTH);
            for (int i = 0; i < bpre_wups; ++i) NUFFT_PSWF_PRE(plan, xb_in);
            double t0 = get_time_sec();
            for (int i = 0; i < bpre_reps; ++i) NUFFT_PSWF_PRE(plan, xb_in);
            double t_pre = (get_time_sec() - t0) / (double)bpre_reps;
            double t_dec43 = gamma_pswf43 * (double)N_beta_out43 * unit_pswf43;
            if (t_pre > t_dec43) t_pre -= t_dec43;
            b_pre_pswf43 = t_pre / ((double)M_beta * unit_pswf43);

            for (int i = 0; i < bwups_b; ++i)
                NUFFT_PSWF_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta_out43, 0, freq_factor, TLS_NUFFT_OUT_NATURAL, bout_re, bout_im);
            t0 = get_time_sec();
            for (int i = 0; i < breps_b; ++i)
                NUFFT_PSWF_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta_out43, 0, freq_factor, TLS_NUFFT_OUT_NATURAL, bout_re, bout_im);
            double t_exec = (get_time_sec() - t0) / (double)breps_b;
            NUFFT_PSWF_FREE(plan);
            b_beta_pswf43 = t_exec / ((double)M_beta * unit_pswf43);
            printf("[bbeta PSWF43] exec/batch = %e s (%.6f) | b_pre = %.6f\n", t_exec, b_beta_pswf43, b_pre_pswf43);
        }

#if !defined(DOUBLE_DOUBLE)
        /* PSWF21 breadth ppoly pre + beta */
        {
            PSWF_PLAN_T *plan = NUFFT_PSWF_INIT_OPTS(M_beta, N_beta, w21, df, freq_factor, "21", TLS_SPREAD_PIECEWISE, TLS_VEC_BREADTH);
            for (int i = 0; i < bpre_wups; ++i) NUFFT_PSWF_PRE(plan, xb_in);
            double t0 = get_time_sec();
            for (int i = 0; i < bpre_reps; ++i) NUFFT_PSWF_PRE(plan, xb_in);
            double t_pre = (get_time_sec() - t0) / (double)bpre_reps;
            double t_dec21 = gamma_pswf21 * (double)N_beta * unit_pswf21;
            if (t_pre > t_dec21) t_pre -= t_dec21;
            b_pre_pp_pswf21 = t_pre / ((double)M_beta * unit_pswf21);

            for (int i = 0; i < bwups_b; ++i)
                NUFFT_PSWF_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta, 0, freq_factor, TLS_NUFFT_OUT_NATURAL, bout_re, bout_im);
            t0 = get_time_sec();
            for (int i = 0; i < breps_b; ++i)
                NUFFT_PSWF_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta, 0, freq_factor, TLS_NUFFT_OUT_NATURAL, bout_re, bout_im);
            double t_exec = (get_time_sec() - t0) / (double)breps_b;
            NUFFT_PSWF_FREE(plan);
            b_beta_pp_pswf21 = t_exec / ((double)M_beta * unit_pswf21);
            printf("[bbeta PSWF21 ppoly] exec/batch = %e s (%.6f) | b_pre = %.6f\n", t_exec, b_beta_pp_pswf21, b_pre_pp_pswf21);
        }

        /* PSWF43 breadth ppoly pre + beta */
        {
            PSWF_PLAN_T *plan = NUFFT_PSWF_INIT_OPTS(M_beta, N_beta, w43, df, freq_factor, "43", TLS_SPREAD_PIECEWISE, TLS_VEC_BREADTH);
            for (int i = 0; i < bpre_wups; ++i) NUFFT_PSWF_PRE(plan, xb_in);
            double t0 = get_time_sec();
            for (int i = 0; i < bpre_reps; ++i) NUFFT_PSWF_PRE(plan, xb_in);
            double t_pre = (get_time_sec() - t0) / (double)bpre_reps;
            double t_dec43 = gamma_pswf43 * (double)N_beta_out43 * unit_pswf43;
            if (t_pre > t_dec43) t_pre -= t_dec43;
            b_pre_pp_pswf43 = t_pre / ((double)M_beta * unit_pswf43);

            for (int i = 0; i < bwups_b; ++i)
                NUFFT_PSWF_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta_out43, 0, freq_factor, TLS_NUFFT_OUT_NATURAL, bout_re, bout_im);
            t0 = get_time_sec();
            for (int i = 0; i < breps_b; ++i)
                NUFFT_PSWF_EXEC_BREADTH(plan, xb_in, y_re_src, f0_b, N_beta_out43, 0, freq_factor, TLS_NUFFT_OUT_NATURAL, bout_re, bout_im);
            double t_exec = (get_time_sec() - t0) / (double)breps_b;
            NUFFT_PSWF_FREE(plan);
            b_beta_pp_pswf43 = t_exec / ((double)M_beta * unit_pswf43);
            printf("[bbeta PSWF43 ppoly] exec/batch = %e s (%.6f) | b_pre = %.6f\n", t_exec, b_beta_pp_pswf43, b_pre_pp_pswf43);
        }
#endif
        free(bout_re);
        free(bout_im);
    }

    free(x_beta);
#if defined(DOUBLE_DOUBLE)
    free(x_beta_dd);
#endif
    free(y_re_src);
    free(y_im_src);
    free(y_re_run);
    free(y_im_run);
    free(out_re);
    free(out_im);
    free(dy_re);
    free(dy_im);

    /* =========================================================================
     * Summary
     * ======================================================================= */
    printf("\n");
    printf("================================================================\n");
    printf("  NuFFT Scaling Parameters — %s Precision\n", precision_name);
    printf("================================================================\n");
    printf("  FFT backend\n");
    printf("    unit_fft = %e s   alpha = %.6f\n\n", unit_fft, alpha);
    printf("  LRA  (rank = %2d)\n", rank);
    printf("    unit     = %e s   alpha = %.6f\n", unit_lra, alpha);
    printf("    beta     = %.6f   gamma = %.6f\n\n", beta_lra, gamma_lra);
    printf("  PSWF21 (w  = %2d)\n", w21);
    printf("    unit     = %e s   alpha = %.6f\n", unit_pswf21, alpha);
    printf("    beta     = %.6f   gamma = %.6f\n\n", beta_pswf21, gamma_pswf21);
    printf("  PSWF43 (w  = %2d)\n", w43);
    printf("    unit     = %e s   alpha = %.6f\n", unit_pswf43, alpha);
    printf("    beta     = %.6f   gamma = %.6f\n", beta_pswf43, gamma_pswf43);
    printf("================================================================\n");

#ifdef SAVE
    {
        const char *fname = "scaling.h";
        FILE *f = fopen(fname, "a"); /* always append; Makefile clears the file first */
        if (!f) {
            perror("fopen scaling.h");
            return 1;
        }

#    if defined(DOUBLE_DOUBLE)
#        define PFX "DD"
#    elif defined(DOUBLE)
#        define PFX "D"
#    elif defined(HALF)
#        define PFX "H"
#    else
#        define PFX "F"
#    endif

        fprintf(f,
                "#define " PFX
                "_ALPHA %.10g\n"
                "#define " PFX
                "_LRA_BETA %.10g\n"
                "#define " PFX
                "_LRA_GAMMA %.10g\n"
                "#define " PFX
                "_PSWF21_BETA %.10g\n"
                "#define " PFX
                "_PSWF21_GAMMA %.10g\n"
                "#define " PFX
                "_PSWF43_BETA %.10g\n"
                "#define " PFX "_PSWF43_GAMMA %.10g\n",
                alpha, beta_lra, gamma_lra, beta_pswf21, gamma_pswf21, beta_pswf43, gamma_pswf43);
#    if !defined(DOUBLE_DOUBLE)
        fprintf(f,
                "#define " PFX
                "_PSWF21_PRE %.10g\n"
                "#define " PFX
                "_PSWF21_PPOLY_BETA %.10g\n"
                "#define " PFX
                "_PSWF21_PPOLY_PRE %.10g\n"
                "#define " PFX
                "_PSWF21_PPOLY_CONST %.10g\n"
                "#define " PFX
                "_PSWF43_PRE %.10g\n"
                "#define " PFX
                "_PSWF43_PPOLY_BETA %.10g\n"
                "#define " PFX
                "_PSWF43_PPOLY_PRE %.10g\n"
                "#define " PFX "_PSWF43_PPOLY_CONST %.10g\n",
                pre_pswf21, beta_pp_pswf21, pre_pp_pswf21, ppoly_const_pswf21, pre_pswf43, beta_pp_pswf43, pre_pp_pswf43, ppoly_const_pswf43);
#    endif
        fprintf(f,
                "#define " PFX
                "_B_ALPHA %.10g\n"
                "#define " PFX
                "_B_FFT_RATIO %.10g\n"
                "#define " PFX
                "_LRA_B_BETA %.10g\n"
                "#define " PFX
                "_PSWF21_B_BETA %.10g\n"
                "#define " PFX
                "_PSWF21_B_PRE %.10g\n"
                "#define " PFX
                "_PSWF43_B_BETA %.10g\n"
                "#define " PFX "_PSWF43_B_PRE %.10g\n",
                b_alpha, b_fft_ratio, b_beta_lra, b_beta_pswf21, b_pre_pswf21, b_beta_pswf43, b_pre_pswf43);
#    if !defined(DOUBLE_DOUBLE)
        fprintf(f,
                "#define " PFX
                "_PSWF21_B_PPOLY_BETA %.10g\n"
                "#define " PFX
                "_PSWF21_B_PPOLY_PRE %.10g\n"
                "#define " PFX
                "_PSWF43_B_PPOLY_BETA %.10g\n"
                "#define " PFX "_PSWF43_B_PPOLY_PRE %.10g\n",
                b_beta_pp_pswf21, b_pre_pp_pswf21, b_beta_pp_pswf43, b_pre_pp_pswf43);
#    endif
        fprintf(f, "\n");
        fclose(f);

#    undef PFX
        printf("[SAVE] Written %s params to %s\n", precision_name, fname);
    }
#endif

    return 0;
}
