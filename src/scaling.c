#include <math.h>
#include <nanofft.h>
#include <nanofft_precision.h>
#include <nufft1.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
#    define LRA_PLAN_T tlsdd_nufft_lra_plan
#    define PSWF_PLAN_T tlsdd_nufft_pswf_plan
#    define NUFFT_RANK 27
#    define NUFFT_W 32
#elif defined(DOUBLE)
#    define NUFFT_LRA_INIT tls_nufft_lra_initialize
#    define NUFFT_LRA_PRE tls_nufft_lra_precompute
#    define NUFFT_LRA_EXEC tls_nufft_lra_execute
#    define NUFFT_LRA_FREE tls_nufft_free_lra_plan
#    define NUFFT_PSWF_INIT tls_nufft_pswf_initialize
#    define NUFFT_PSWF_PRE tls_nufft_pswf_precompute
#    define NUFFT_PSWF_EXEC tls_nufft_pswf_execute
#    define NUFFT_PSWF_FREE tls_nufft_free_pswf_plan
#    define LRA_PLAN_T tls_nufft_lra_plan
#    define PSWF_PLAN_T tls_nufft_pswf_plan
#    define NUFFT_RANK 16
#    define NUFFT_W 16
#else /* single (default) */
#    define NUFFT_LRA_INIT tlsf_nufft_lra_initialize
#    define NUFFT_LRA_PRE tlsf_nufft_lra_precompute
#    define NUFFT_LRA_EXEC tlsf_nufft_lra_execute
#    define NUFFT_LRA_FREE tlsf_nufft_free_lra_plan
#    define NUFFT_PSWF_INIT tlsf_nufft_pswf_initialize
#    define NUFFT_PSWF_PRE tlsf_nufft_pswf_precompute
#    define NUFFT_PSWF_EXEC tlsf_nufft_pswf_execute
#    define NUFFT_PSWF_FREE tlsf_nufft_free_pswf_plan
#    define LRA_PLAN_T tlsf_nufft_lra_plan
#    define PSWF_PLAN_T tlsf_nufft_pswf_plan
#    define NUFFT_RANK 9
#    define NUFFT_W 8
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
#else
    int k_start = 9, k_end = 22;
    const char *precision_name = "Single";
#endif

    /* NuFFT constants — must match finufft_bench.cpp */
    const double df = 1.0;
    const int freq_factor = 1;
    const int rank = NUFFT_RANK;
    const int w = NUFFT_W;

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
            re[i] = (FLOAT)r1;
            im[i] = (FLOAT)r2;
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
     * PSWF: one size-2N FFT (2x oversampling) → unit_pswf = unit_fft *
     * 2^(1+alpha)
     * ======================================================================= */
    double unit_lra = (double)rank * unit_fft;
    double unit_pswf = unit_fft * pow(2.0, 1.0 + alpha);

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

    double gamma_lra, gamma_pswf;

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

    /* --- PSWF gamma --- */
    {
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT(M_gamma, N_gamma, w, df, freq_factor);

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
        gamma_pswf = t_pre / ((double)N_gamma * unit_pswf);
        printf("[gamma PSWF] precompute/run = %e s | gamma = %.6f\n", t_pre, gamma_pswf);
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
    const int M_beta = 1 << 15; /* 32768 — M >> N */

    double *x_beta = (double *)malloc(M_beta * sizeof(double));
#if defined(DOUBLE_DOUBLE)
    dd_t *x_beta_dd = (dd_t *)malloc(M_beta * sizeof(dd_t));
#endif
    FLOAT *y_re_src = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *y_im_src = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *y_re_run = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *y_im_run = (FLOAT *)aligned_alloc(64, M_beta * sizeof(FLOAT));
    FLOAT *out_re = (FLOAT *)aligned_alloc(64, N_beta * sizeof(FLOAT));
    FLOAT *out_im = (FLOAT *)aligned_alloc(64, N_beta * sizeof(FLOAT));
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
        y_re_src[i] = (FLOAT)r1;
        y_im_src[i] = (FLOAT)r2;
#endif
    }

    /* Precompute dy phase-twist vectors (not included in beta timing) */
    for (int i = 0; i < M_beta; i++) {
        FLOAT arg = MUL(MUL(FCAST(x_beta[i]), FCAST(df)), FCAST((double)N_beta));
        dy_re[i] = M_SIN2PI(arg);
        dy_im[i] = M_COS2PI(arg);
    }

    double beta_lra, beta_pswf;
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

    /* --- PSWF beta --- */
    {
        PSWF_PLAN_T *plan = NUFFT_PSWF_INIT(M_beta, N_beta, w, df, freq_factor);
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
        beta_pswf = t_exec / ((double)M_beta * unit_pswf);
        printf("[beta  PSWF] exec/run = %e s | beta  = %.6f\n", t_exec, beta_pswf);
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
    printf("  PSWF (w    = %2d)\n", w);
    printf("    unit     = %e s   alpha = %.6f\n", unit_pswf, alpha);
    printf("    beta     = %.6f   gamma = %.6f\n", beta_pswf, gamma_pswf);
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
                "_PSWF_BETA %.10g\n"
                "#define " PFX "_PSWF_GAMMA %.10g\n\n",
                alpha, beta_lra, gamma_lra, beta_pswf, gamma_pswf);
        fclose(f);

#    undef PFX
        printf("[SAVE] Written %s params to %s\n", precision_name, fname);
    }
#endif

    return 0;
}
