/*
 * bench_vectorize.c - interleaved benchmark of vectorization modes
 * (TLS_VEC_DEPTH vs TLS_VEC_BREADTH vs TLS_VEC_AUTO) across regimes,
 * backends, and precisions (float, double, double-double).
 */

#include <math.h>
#include <nanofft.h>
#include <scaling.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tls.h>
#include <tls_tuning.h>
#include <utils.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;

static double urand(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (double)(rng >> 11) * (1.0 / 9007199254740992.0);
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static void make_data(int M, double *t, double *y, double *dy, float *fy, float *fdy) {
    const double span = 1000.0;
    for (int m = 0; m < M; ++m) t[m] = span * urand();
    qsort(t, (size_t)M, sizeof(double), cmp_double);
    for (int m = 0; m < M; ++m) {
        y[m] = 1.0 + 0.7 * sin(2.0 * M_PI * 0.23 * t[m]) + 0.3 * sin(2.0 * M_PI * 0.69 * t[m]) + 0.1 * (urand() - 0.5);
        dy[m] = 0.1 + 0.02 * urand();
        fy[m] = (float)y[m];
        fdy[m] = (float)dy[m];
    }
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static const char *backend_name(int b) {
    if (b == TLS_BACKEND_PSWF43) return "pswf43";
    if (b == TLS_BACKEND_PSWF21) return "pswf21";
    return "lra";
}

static const char *vec_name(int m) {
    if (m == TLS_VEC_DEPTH) return "depth";
    if (m == TLS_VEC_BREADTH) return "breadth";
    return "auto";
}

static const char *spread_name(int m) { return (m == TLS_SPREAD_PIECEWISE) ? "ppoly" : "minmax"; }

static void bench_case(const char *label, int M, int N, int degree, int backend, int is_double, int is_ihs, int reps) {
    double *t = (double *)malloc((size_t)M * sizeof(double));
    double *y = (double *)malloc((size_t)M * sizeof(double));
    double *dy = (double *)malloc((size_t)M * sizeof(double));
    float *fy = (float *)malloc((size_t)M * sizeof(float));
    float *fdy = (float *)malloc((size_t)M * sizeof(float));
    double *pd = (double *)malloc((size_t)N * sizeof(double));
    float *pf = (float *)malloc((size_t)N * sizeof(float));
    double *times[3];
    for (int v = 0; v < 3; ++v) times[v] = (double *)malloc((size_t)reps * sizeof(double));

    make_data(M, t, y, dy, fy, fdy);
    const double f0 = 0.001;
    const double df = 0.0002;
    const int modes[3] = {TLS_VEC_DEPTH, TLS_VEC_BREADTH, TLS_VEC_AUTO};

    for (int v = 0; v < 3; ++v) {
        if (is_double) {
            if (is_ihs)
                tls_ihs_opts(t, y, dy, M, f0, df, N, degree, backend, TLS_NORM_IHS_DEFAULT, TLS_SPREAD_AUTO, modes[v], pd);
            else
                tls_fastchi2_opts(t, y, dy, M, f0, df, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_AUTO, modes[v], pd, NULL);
        } else {
            if (is_ihs)
                tlsf_ihs_opts(t, fy, fdy, M, f0, df, N, degree, backend, TLS_NORM_IHS_DEFAULT, TLS_SPREAD_AUTO, modes[v], pf);
            else
                tlsf_fastchi2_opts(t, fy, fdy, M, f0, df, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_AUTO, modes[v], pf, NULL);
        }
    }

    for (int r = 0; r < reps; ++r) {
        for (int v = 0; v < 3; ++v) {
            int idx = (r + v) % 3;
            double t0 = now_s();
            if (is_double) {
                if (is_ihs)
                    tls_ihs_opts(t, y, dy, M, f0, df, N, degree, backend, TLS_NORM_IHS_DEFAULT, TLS_SPREAD_AUTO, modes[idx], pd);
                else
                    tls_fastchi2_opts(t, y, dy, M, f0, df, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_AUTO, modes[idx], pd, NULL);
            } else {
                if (is_ihs)
                    tlsf_ihs_opts(t, fy, fdy, M, f0, df, N, degree, backend, TLS_NORM_IHS_DEFAULT, TLS_SPREAD_AUTO, modes[idx], pf);
                else
                    tlsf_fastchi2_opts(t, fy, fdy, M, f0, df, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_AUTO, modes[idx], pf,
                                       NULL);
            }
            times[idx][r] = (now_s() - t0) * 1e3;
        }
    }

    int is43 = (backend == TLS_BACKEND_PSWF43);
    int is_lra = (backend == TLS_BACKEND_LRA);
    tls_plan_choice ch;
    if (is_double) {
        ch = tls_optimize_plan(N, M, degree, is_ihs, backend, VEC_LEN, 32, 0, D_ALPHA, is_lra ? D_LRA_GAMMA : (is43 ? D_PSWF43_GAMMA : D_PSWF21_GAMMA),
                               is_lra ? D_LRA_BETA : (is43 ? D_PSWF43_BETA : D_PSWF21_BETA), is_lra ? 0.0 : (is43 ? D_PSWF43_PRE : D_PSWF21_PRE),
                               is_lra ? 0.0 : (is43 ? D_PSWF43_PPOLY_BETA : D_PSWF21_PPOLY_BETA),
                               is_lra ? 0.0 : (is43 ? D_PSWF43_PPOLY_PRE : D_PSWF21_PPOLY_PRE),
                               is_lra ? 0.0 : (is43 ? D_PSWF43_PPOLY_CONST : D_PSWF21_PPOLY_CONST), D_B_ALPHA, D_B_FFT_RATIO,
                               is_lra ? D_LRA_B_BETA : (is43 ? D_PSWF43_B_BETA : D_PSWF21_B_BETA), is_lra ? 0.0 : (is43 ? D_PSWF43_B_PRE : D_PSWF21_B_PRE),
                               is_lra ? 0.0 : (is43 ? D_PSWF43_B_PPOLY_BETA : D_PSWF21_B_PPOLY_BETA),
                               is_lra ? 0.0 : (is43 ? D_PSWF43_B_PPOLY_PRE : D_PSWF21_B_PPOLY_PRE), TLS_SPREAD_AUTO, TLS_VEC_AUTO);
    } else {
        ch = tls_optimize_plan(N, M, degree, is_ihs, backend, VECF_LEN, 16, 0, F_ALPHA, is_lra ? F_LRA_GAMMA : (is43 ? F_PSWF43_GAMMA : F_PSWF21_GAMMA),
                               is_lra ? F_LRA_BETA : (is43 ? F_PSWF43_BETA : F_PSWF21_BETA), is_lra ? 0.0 : (is43 ? F_PSWF43_PRE : F_PSWF21_PRE),
                               is_lra ? 0.0 : (is43 ? F_PSWF43_PPOLY_BETA : F_PSWF21_PPOLY_BETA),
                               is_lra ? 0.0 : (is43 ? F_PSWF43_PPOLY_PRE : F_PSWF21_PPOLY_PRE),
                               is_lra ? 0.0 : (is43 ? F_PSWF43_PPOLY_CONST : F_PSWF21_PPOLY_CONST), F_B_ALPHA, F_B_FFT_RATIO,
                               is_lra ? F_LRA_B_BETA : (is43 ? F_PSWF43_B_BETA : F_PSWF21_B_BETA), is_lra ? 0.0 : (is43 ? F_PSWF43_B_PRE : F_PSWF21_B_PRE),
                               is_lra ? 0.0 : (is43 ? F_PSWF43_B_PPOLY_BETA : F_PSWF21_B_PPOLY_BETA),
                               is_lra ? 0.0 : (is43 ? F_PSWF43_B_PPOLY_PRE : F_PSWF21_B_PPOLY_PRE), TLS_SPREAD_AUTO, TLS_VEC_AUTO);
    }

    double m_d = median(times[0], reps);
    double m_b = median(times[1], reps);
    double m_a = median(times[2], reps);
    printf("%-18s M=%5d N=%6d d=%d %-6s | depth=%7.2f ms  breadth=%7.2f ms  auto=%7.2f ms (%-7s/%-6s B=%5d) | d/b=%5.2fx  d/a=%5.2fx\n", label, M, N, degree,
           backend_name(backend), m_d, m_b, m_a, vec_name(ch.vec_mode), spread_name(ch.spread_mode), ch.block, m_d / m_b, m_d / m_a);

    for (int v = 0; v < 3; ++v) free(times[v]);
    free(t);
    free(y);
    free(dy);
    free(fy);
    free(fdy);
    free(pd);
    free(pf);
}

static void bench_dd_case(const char *label, int M, int N, int degree, int backend, int reps) {
    double *td = (double *)malloc((size_t)M * sizeof(double));
    double *yd = (double *)malloc((size_t)M * sizeof(double));
    double *dyd = (double *)malloc((size_t)M * sizeof(double));
    float *fy = (float *)malloc((size_t)M * sizeof(float));
    float *fdy = (float *)malloc((size_t)M * sizeof(float));
    dd_t *t = (dd_t *)malloc((size_t)M * sizeof(dd_t));
    dd_t *y = (dd_t *)malloc((size_t)M * sizeof(dd_t));
    dd_t *dy = (dd_t *)malloc((size_t)M * sizeof(dd_t));
    dd_t *p = (dd_t *)malloc((size_t)N * sizeof(dd_t));
    double *times = (double *)malloc((size_t)reps * sizeof(double));

    make_data(M, td, yd, dyd, fy, fdy);
    for (int m = 0; m < M; ++m) {
        t[m] = dd_make(td[m], 0.0);
        y[m] = dd_make(yd[m], 0.0);
        dy[m] = dd_make(dyd[m], 0.0);
    }

    tlsdd_fastchi2_opts(t, y, dy, M, 0.001, 0.0002, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX, TLS_VEC_BREADTH, p, NULL);
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s();
        tlsdd_fastchi2_opts(t, y, dy, M, 0.001, 0.0002, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX, TLS_VEC_BREADTH, p,
                            NULL);
        times[r] = (now_s() - t0) * 1e3;
    }
    printf("%-18s M=%5d N=%6d d=%d %-6s | dd-breadth (VECDD_LEN=%d) = %7.2f ms\n", label, M, N, degree, backend_name(backend), VECDD_LEN, median(times, reps));

    free(times);
    free(td);
    free(yd);
    free(dyd);
    free(fy);
    free(fdy);
    free(t);
    free(y);
    free(dy);
    free(p);
}

int main(int argc, char **argv) {
    int sweep = (argc > 1 && strcmp(argv[1], "--sweep") == 0);
    printf("=== bench_vectorize (VEC_BYTES=%d, VEC_LEN=%d, VECF_LEN=%d, VECDD_LEN=%d) ===\n", VEC_BYTES, VEC_LEN, VECF_LEN, VECDD_LEN);

    printf("\n-- Multi-block regimes (PSWF43 / PSWF21 / LRA) --\n");
    bench_case("tls_fc2", 20000, 16384, 3, TLS_BACKEND_PSWF43, 1, 0, 5);
    bench_case("tls_fc2", 20000, 16384, 3, TLS_BACKEND_PSWF21, 1, 0, 5);
    bench_case("tls_fc2", 20000, 16384, 3, TLS_BACKEND_LRA, 1, 0, 5);
    bench_case("tlsf_fc2", 20000, 16384, 3, TLS_BACKEND_PSWF43, 0, 0, 5);
    bench_case("tlsf_fc2", 20000, 16384, 3, TLS_BACKEND_LRA, 0, 0, 5);
    bench_case("tls_ihs", 20000, 16384, 3, TLS_BACKEND_PSWF43, 1, 1, 5);
    bench_case("tlsf_ihs", 20000, 16384, 3, TLS_BACKEND_PSWF43, 0, 1, 5);

    printf("\n-- Small-N / large-N regimes --\n");
    bench_case("smallN_d", 10000, 256, 3, TLS_BACKEND_PSWF43, 1, 0, 7);
    bench_case("smallN_f", 10000, 256, 3, TLS_BACKEND_PSWF43, 0, 0, 7);
    bench_case("smallM_d", 128, 16384, 3, TLS_BACKEND_PSWF43, 1, 0, 7);
    bench_case("largeN_d", 2000, 131072, 3, TLS_BACKEND_PSWF43, 1, 0, 5);
    bench_case("largeN_f", 2000, 131072, 3, TLS_BACKEND_PSWF43, 0, 0, 5);

    printf("\n-- Double-Double (VECDD) --\n");
    bench_dd_case("tlsdd_fc2", 512, 2048, 2, TLS_BACKEND_PSWF43, 3);
    bench_dd_case("tlsdd_fc2", 512, 2048, 2, TLS_BACKEND_PSWF21, 3);
    bench_dd_case("tlsdd_fc2", 512, 2048, 2, TLS_BACKEND_LRA, 3);

    if (sweep) {
        printf("\n-- Crossover sweep (N at M=16384, d=3, pswf43) --\n");
        const int ns[] = {256, 1024, 4096, 16384, 65536};
        for (int i = 0; i < 5; ++i) {
            bench_case("sweep_d", 16384, ns[i], 3, TLS_BACKEND_PSWF43, 1, 0, 5);
            bench_case("sweep_f", 16384, ns[i], 3, TLS_BACKEND_PSWF43, 0, 0, 5);
        }
    }
    return 0;
}
