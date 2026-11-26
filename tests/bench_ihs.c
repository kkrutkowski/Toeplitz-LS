/*
 * bench_ihs.c - interleaved comparative benchmark of deconvolution placement
 * across BOTH IHS and standard TLS (fastchi2) periodograms, covering all four
 * precisions (HALF, FLOAT, DOUBLE, DOUBLE_DOUBLE), degrees (d=1, 3, 8),
 * normalizations (STANDARD, IHS_DEFAULT, NLL), backends (PSWF43, PSWF21),
 * and vectorization modes (DEPTH, BREADTH).
 *
 * Variants compared per workload:
 *   natural   : defer=0 (FFT permutes Re & Im, deconvolves Re & Im pre-op)
 *   pre_dec   : defer=1, TLS_IHS_DECONV_PRE (slot-order deconv in emit;
 *               unmerged row_scale for HALF fastchi2)
 *   post_dec  : defer=1, TLS_IHS_DECONV_POST (default: raw slots + post-sum/
 *               post-permute deconv for IHS; raw slots + q=0/post-solve deconv
 *               for FP32+ fastchi2; merged row_scale for HALF fastchi2)
 *
 *   ./bench_ihs [rounds] [reps]
 */

#include <math.h>
#include <nanofft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tls.h>
#include <tls_tuning.h>

enum { PREC_HALF = 0, PREC_FLOAT = 1, PREC_DOUBLE = 2, PREC_DD = 3 };

typedef struct {
    const char *label;
    int prec;
    int is_ihs;
    int M;
    int N;
    int degree;
    int backend;
    int norm;
    int vec_mode;
} bench_case;

static const bench_case cases[] = {
    /* --- IHS: d=1 across all precisions & normalizations (depth & breadth) --- */
    {"ihs_h_d1_std_dep", PREC_HALF, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_h_d1_std_brd", PREC_HALF, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"ihs_h_d1_nll_dep", PREC_HALF, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_NLL, TLS_VEC_DEPTH},
    {"ihs_f_d1_std_dep", PREC_FLOAT, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_f_d1_std_brd", PREC_FLOAT, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"ihs_f_d1_def_dep", PREC_FLOAT, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_IHS_DEFAULT, TLS_VEC_DEPTH},
    {"ihs_f_d1_nll_dep", PREC_FLOAT, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_NLL, TLS_VEC_DEPTH},
    {"ihs_d_d1_std_dep", PREC_DOUBLE, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_d_d1_std_brd", PREC_DOUBLE, 1, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"ihs_dd_d1_std_brd", PREC_DD, 1, 300, 2048, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},

    /* --- IHS: d=3 & d=8 across precisions, normalizations, backends --- */
    {"ihs_h_d3_std_dep", PREC_HALF, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_h_d3_std_brd", PREC_HALF, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"ihs_h_d3_def_dep", PREC_HALF, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_IHS_DEFAULT, TLS_VEC_DEPTH},
    {"ihs_h_d3_nll_dep", PREC_HALF, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_NLL, TLS_VEC_DEPTH},
    {"ihs_f_d3_std_dep", PREC_FLOAT, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_f_d3_std_brd", PREC_FLOAT, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"ihs_f_d3_def_dep", PREC_FLOAT, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_IHS_DEFAULT, TLS_VEC_DEPTH},
    {"ihs_f_d3_nll_dep", PREC_FLOAT, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_NLL, TLS_VEC_DEPTH},
    {"ihs_f_d8_std_dep", PREC_FLOAT, 1, 600, 8192, 8, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_f_d3_p21_dep", PREC_FLOAT, 1, 600, 8192, 3, TLS_BACKEND_PSWF21, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_d_d3_std_dep", PREC_DOUBLE, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_d_d3_std_brd", PREC_DOUBLE, 1, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"ihs_d_d8_def_dep", PREC_DOUBLE, 1, 600, 8192, 8, TLS_BACKEND_PSWF43, TLS_NORM_IHS_DEFAULT, TLS_VEC_DEPTH},
    {"ihs_dd_d3_std_brd", PREC_DD, 1, 300, 2048, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},

    /* --- IHS: multi-block N=65536 across precisions & degrees --- */
    {"ihs_h_d1_b16_dep", PREC_HALF, 1, 600, 65536, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_h_d3_b16_dep", PREC_HALF, 1, 600, 65536, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_h_d8_b16_dep", PREC_HALF, 1, 600, 65536, 8, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_f_d1_b16_dep", PREC_FLOAT, 1, 600, 65536, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_f_d3_b16_dep", PREC_FLOAT, 1, 600, 65536, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_f_d8_b16_dep", PREC_FLOAT, 1, 600, 65536, 8, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_d_d1_b16_dep", PREC_DOUBLE, 1, 600, 65536, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_d_d3_b16_dep", PREC_DOUBLE, 1, 600, 65536, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"ihs_d_d8_b16_dep", PREC_DOUBLE, 1, 600, 65536, 8, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},

    /* --- Standard TLS (fastchi2): d=1, d=3, d=8 across all precisions --- */
    {"tls_h_d1_std_dep", PREC_HALF, 0, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_h_d1_std_brd", PREC_HALF, 0, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"tls_h_d3_std_dep", PREC_HALF, 0, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_h_d3_std_brd", PREC_HALF, 0, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"tls_h_d8_std_dep", PREC_HALF, 0, 600, 8192, 8, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_f_d1_std_dep", PREC_FLOAT, 0, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_f_d3_std_dep", PREC_FLOAT, 0, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_f_d3_std_brd", PREC_FLOAT, 0, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"tls_f_d3_p21_dep", PREC_FLOAT, 0, 600, 8192, 3, TLS_BACKEND_PSWF21, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_f_d8_std_dep", PREC_FLOAT, 0, 600, 8192, 8, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_h_d3_b16_dep", PREC_HALF, 0, 600, 65536, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_h_d3_b16_brd", PREC_HALF, 0, 600, 65536, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"tls_f_d3_b16_dep", PREC_FLOAT, 0, 600, 65536, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_d_d3_b16_dep", PREC_DOUBLE, 0, 600, 65536, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_d_d1_std_dep", PREC_DOUBLE, 0, 600, 8192, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_d_d3_std_dep", PREC_DOUBLE, 0, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_d_d3_std_brd", PREC_DOUBLE, 0, 600, 8192, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
    {"tls_d_d8_std_dep", PREC_DOUBLE, 0, 600, 8192, 8, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_DEPTH},
    {"tls_dd_d3_std_brd", PREC_DD, 0, 300, 2048, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_VEC_BREADTH},
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

static uint64_t rng_state = 88172645463325252ull;

static double urand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
}

static void *test_alloc(size_t bytes) {
    const size_t padded = (bytes + 63u) & ~(size_t)63u;
    void *ptr = aligned_alloc(64, padded ? padded : 64u);
    if (!ptr) {
        fprintf(stderr, "allocation failure\n");
        exit(1);
    }
    memset(ptr, 0, padded ? padded : 64u);
    return ptr;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void make_curve(int M, double *t, double *y, double *dy) {
    const double span = 1000.0;
    for (int m = 0; m < M; ++m) {
        t[m] = span * (double)m / (double)M + 0.4 * urand() * (span / (double)M) * 2.0;
        y[m] = 1.0 + 0.7 * sin(2.0 * M_PI * 3.1 * t[m] / span) + 0.4 * sin(2.0 * M_PI * 7.3 * t[m] / span) + 0.05 * (urand() - 0.5);
        dy[m] = 0.1 + 0.02 * urand();
    }
    for (int m = 1; m < M; ++m) {
        if (t[m] < t[m - 1]) {
            const double tmp = t[m];
            t[m] = t[m - 1];
            t[m - 1] = tmp;
        }
    }
}

static int run_one(const bench_case *c, const double *t, const double *y, const double *dy, const float *fy, const float *fdy, const dd_t *dt,
                   const dd_t *dy_dd, const dd_t *ddy_dd, float *pf, double *pd, dd_t *pdd) {
    const double f0 = 0.001;
    const double df = 0.0005;
    const int sp = (c->prec == PREC_DD) ? TLS_SPREAD_MINMAX : TLS_SPREAD_AUTO;
    if (c->is_ihs) {
        if (c->prec == PREC_HALF) return tlsh_ihs_opts(t, fy, fdy, c->M, f0, df, c->N, c->degree, c->backend, c->norm, sp, c->vec_mode, pf);
        if (c->prec == PREC_FLOAT) return tlsf_ihs_opts(t, fy, fdy, c->M, f0, df, c->N, c->degree, c->backend, c->norm, sp, c->vec_mode, pf);
        if (c->prec == PREC_DOUBLE) return tls_ihs_opts(t, y, dy, c->M, f0, df, c->N, c->degree, c->backend, c->norm, sp, c->vec_mode, pd);
        return tlsdd_ihs_opts(dt, dy_dd, ddy_dd, c->M, f0, df, c->N, c->degree, c->backend, c->norm, sp, c->vec_mode, pdd);
    } else {
        if (c->prec == PREC_HALF)
            return tlsh_fastchi2_opts(t, fy, fdy, c->M, f0, df, c->N, c->degree, c->backend, TLS_SOLVER_LEVINSON, c->norm, sp, c->vec_mode, pf, NULL);
        if (c->prec == PREC_FLOAT)
            return tlsf_fastchi2_opts(t, fy, fdy, c->M, f0, df, c->N, c->degree, c->backend, TLS_SOLVER_LEVINSON, c->norm, sp, c->vec_mode, pf, NULL);
        if (c->prec == PREC_DOUBLE)
            return tls_fastchi2_opts(t, y, dy, c->M, f0, df, c->N, c->degree, c->backend, TLS_SOLVER_LEVINSON, c->norm, sp, c->vec_mode, pd, NULL);
        return tlsdd_fastchi2_opts(dt, dy_dd, ddy_dd, c->M, f0, df, c->N, c->degree, c->backend, TLS_SOLVER_LEVINSON, c->norm, sp, c->vec_mode, pdd, NULL);
    }
}

static double max_rel_diff(int prec, int N, const float *pf0, const float *pf1, const double *pd0, const double *pd1, const dd_t *pdd0, const dd_t *pdd1) {
    double scale = 0.0, err = 0.0;
    for (int i = 0; i < N; ++i) {
        double a, b;
        if (prec == PREC_HALF || prec == PREC_FLOAT) {
            a = pf0[i];
            b = pf1[i];
        } else if (prec == PREC_DOUBLE) {
            a = pd0[i];
            b = pd1[i];
        } else {
            a = dd_to_double(pdd0[i]);
            b = dd_to_double(pdd1[i]);
        }
        if (isnan(a) || isnan(b)) continue;
        if (fabs(a) > scale) scale = fabs(a);
        double d = fabs(a - b);
        if (d > err) err = d;
    }
    return (scale > 0.0) ? (err / scale) : err;
}

int main(int argc, char **argv) {
    const int rounds = (argc > 1) ? atoi(argv[1]) : 5;
    const int reps = (argc > 2) ? atoi(argv[2]) : 5;

    const struct {
        int defer;
        int ihs_deconv;
        const char *name;
    } variants[3] = {
        {0, TLS_IHS_DECONV_PRE, "natural"},
        {1, TLS_IHS_DECONV_PRE, "pre_dec"},
        {1, TLS_IHS_DECONV_POST, "post_dec"},
    };

    printf("bench_ihs: %d rounds, %d reps (native_fp16=%d)\n", rounds, reps, tlsh_using_native_fp16());
    printf("%-20s  %10s  %10s  %10s  %8s  %8s  %9s\n", "workload", "natural us", "pre_dec us", "post_dec us", "post/nat", "post/pre", "rel_diff");
    printf("-----------------------------------------------------------------------------------------------\n");

    for (int ci = 0; ci < NCASES; ++ci) {
        const bench_case *c = &cases[ci];
        const int r_eff = (c->prec == PREC_DD) ? 2 : rounds;
        const int p_eff = (c->prec == PREC_DD) ? 2 : reps;

        double *t = (double *)test_alloc((size_t)c->M * sizeof(double));
        double *y = (double *)test_alloc((size_t)c->M * sizeof(double));
        double *dy = (double *)test_alloc((size_t)c->M * sizeof(double));
        float *fy = (float *)test_alloc((size_t)c->M * sizeof(float));
        float *fdy = (float *)test_alloc((size_t)c->M * sizeof(float));
        dd_t *dt = (dd_t *)test_alloc((size_t)c->M * sizeof(dd_t));
        dd_t *dy_dd = (dd_t *)test_alloc((size_t)c->M * sizeof(dd_t));
        dd_t *ddy_dd = (dd_t *)test_alloc((size_t)c->M * sizeof(dd_t));

        float *pf[3] = {NULL, NULL, NULL};
        double *pd[3] = {NULL, NULL, NULL};
        dd_t *pdd[3] = {NULL, NULL, NULL};
        for (int v = 0; v < 3; ++v) {
            pf[v] = (float *)test_alloc((size_t)c->N * sizeof(float));
            pd[v] = (double *)test_alloc((size_t)c->N * sizeof(double));
            pdd[v] = (dd_t *)test_alloc((size_t)c->N * sizeof(dd_t));
        }

        make_curve(c->M, t, y, dy);
        for (int m = 0; m < c->M; ++m) {
            fy[m] = (float)y[m];
            fdy[m] = (float)dy[m];
            dt[m] = dd_make(t[m], 0.0);
            dy_dd[m] = dd_make(y[m], 0.0);
            ddy_dd[m] = dd_make(dy[m], 0.0);
        }

        double best[3] = {1e30, 1e30, 1e30};
        for (int r = 0; r < r_eff; ++r) {
            for (int step = 0; step < 3; ++step) {
                const int v = (step + r) % 3;
                tls_tuning_set_defer(variants[v].defer);
                tls_tuning_set_ihs_deconv(variants[v].ihs_deconv);
                for (int rep = 0; rep < p_eff; ++rep) {
                    const double t0 = now_seconds();
                    int st = run_one(c, t, y, dy, fy, fdy, dt, dy_dd, ddy_dd, pf[v], pd[v], pdd[v]);
                    const double dt_s = now_seconds() - t0;
                    if (st < 0) {
                        fprintf(stderr, "ERROR: %s variant %s failed with %d\n", c->label, variants[v].name, st);
                        return 1;
                    }
                    if (dt_s < best[v]) best[v] = dt_s;
                }
            }
        }

        double diff = max_rel_diff(c->prec, c->N, pf[0], pf[2], pd[0], pd[2], pdd[0], pdd[2]);
        printf("%-20s  %10.1f  %10.1f  %10.1f  %7.3fx  %7.3fx  %9.2e\n", c->label, best[0] * 1e6, best[1] * 1e6, best[2] * 1e6, best[2] / best[0],
               best[2] / best[1], diff);
        fflush(stdout);

        for (int v = 0; v < 3; ++v) {
            free(pf[v]);
            free(pd[v]);
            free(pdd[v]);
        }
        free(t);
        free(y);
        free(dy);
        free(fy);
        free(fdy);
        free(dt);
        free(dy_dd);
        free(ddy_dd);
    }

    tls_tuning_set_defer(TLS_TUNING_DEFER_DEFAULT);
    tls_tuning_set_ihs_deconv(TLS_IHS_DECONV_POST);
    return 0;
}
