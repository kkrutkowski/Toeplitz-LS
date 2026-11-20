/*
 * bench_spread.c - interleaved benchmark of PSWF spreading modes
 * (TLS_SPREAD_MINMAX vs TLS_SPREAD_PIECEWISE vs TLS_SPREAD_AUTO).
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
    for (int m = 0; m < M; ++m) {
        t[m] = span * urand();
    }
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

static const char *mode_name(int m) {
    if (m == TLS_SPREAD_MINMAX) return "minmax";
    if (m == TLS_SPREAD_PIECEWISE) return "ppoly";
    return "auto";
}

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
    const int modes[3] = {TLS_SPREAD_MINMAX, TLS_SPREAD_PIECEWISE, TLS_SPREAD_AUTO};

    /* Warmup */
    for (int v = 0; v < 3; ++v) {
        if (is_double) {
            if (is_ihs)
                tls_ihs_spread(t, y, dy, M, f0, df, N, degree, backend, TLS_NORM_IHS_DEFAULT, modes[v], pd);
            else
                tls_fastchi2_spread(t, y, dy, M, f0, df, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, modes[v], pd, NULL);
        } else {
            if (is_ihs)
                tlsf_ihs_spread(t, fy, fdy, M, f0, df, N, degree, backend, TLS_NORM_IHS_DEFAULT, modes[v], pf);
            else
                tlsf_fastchi2_spread(t, fy, fdy, M, f0, df, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, modes[v], pf, NULL);
        }
    }

    /* Interleaved timed runs */
    for (int r = 0; r < reps; ++r) {
        for (int v = 0; v < 3; ++v) {
            int idx = (r + v) % 3;
            double t0 = now_s();
            if (is_double) {
                if (is_ihs)
                    tls_ihs_spread(t, y, dy, M, f0, df, N, degree, backend, TLS_NORM_IHS_DEFAULT, modes[idx], pd);
                else
                    tls_fastchi2_spread(t, y, dy, M, f0, df, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, modes[idx], pd, NULL);
            } else {
                if (is_ihs)
                    tlsf_ihs_spread(t, fy, fdy, M, f0, df, N, degree, backend, TLS_NORM_IHS_DEFAULT, modes[idx], pf);
                else
                    tlsf_fastchi2_spread(t, fy, fdy, M, f0, df, N, degree, backend, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, modes[idx], pf, NULL);
            }
            times[idx][r] = (now_s() - t0) * 1e3;
        }
    }

    int is43 = (backend == TLS_BACKEND_PSWF43);
    tls_pswf_plan_choice ch;
    if (is_double) {
        ch = tls_optimize_pswf_plan(N, M, degree, is_ihs, backend, D_ALPHA, is43 ? D_PSWF43_GAMMA : D_PSWF21_GAMMA, is43 ? D_PSWF43_BETA : D_PSWF21_BETA,
                                    is43 ? D_PSWF43_PRE : D_PSWF21_PRE, is43 ? D_PSWF43_PPOLY_BETA : D_PSWF21_PPOLY_BETA,
                                    is43 ? D_PSWF43_PPOLY_PRE : D_PSWF21_PPOLY_PRE, is43 ? D_PSWF43_PPOLY_CONST : D_PSWF21_PPOLY_CONST, TLS_SPREAD_AUTO);
    } else {
        ch = tls_optimize_pswf_plan(N, M, degree, is_ihs, backend, F_ALPHA, is43 ? F_PSWF43_GAMMA : F_PSWF21_GAMMA, is43 ? F_PSWF43_BETA : F_PSWF21_BETA,
                                    is43 ? F_PSWF43_PRE : F_PSWF21_PRE, is43 ? F_PSWF43_PPOLY_BETA : F_PSWF21_PPOLY_BETA,
                                    is43 ? F_PSWF43_PPOLY_PRE : F_PSWF21_PPOLY_PRE, is43 ? F_PSWF43_PPOLY_CONST : F_PSWF21_PPOLY_CONST, TLS_SPREAD_AUTO);
    }

    double m_mm = median(times[0], reps);
    double m_pp = median(times[1], reps);
    double m_au = median(times[2], reps);
    printf("%-22s M=%5d N=%6d d=%d %-6s | mm=%7.2f ms  pp=%7.2f ms  auto=%7.2f ms (%-6s B=%5d) | mm/pp=%5.2fx  mm/auto=%5.2fx\n", label, M, N, degree,
           is43 ? "pswf43" : "pswf21", m_mm, m_pp, m_au, mode_name(ch.spread_mode), ch.block, m_mm / m_pp, m_mm / m_au);

    for (int v = 0; v < 3; ++v) free(times[v]);
    free(t);
    free(y);
    free(dy);
    free(fy);
    free(fdy);
    free(pd);
    free(pf);
}

int main(int argc, char **argv) {
    int sweep = (argc > 1 && strcmp(argv[1], "--sweep") == 0);
    printf("=== bench_spread (VEC_BYTES=%d) ===\n", VEC_BYTES);

    printf("\n-- Target regime: M=60000, N=4k..32k --\n");
    const int ns[] = {4096, 8192, 16384, 32768};
    for (int i = 0; i < 4; ++i) {
        bench_case("tls_fc2", 60000, ns[i], 3, TLS_BACKEND_PSWF43, 1, 0, 7);
        bench_case("tls_fc2", 60000, ns[i], 3, TLS_BACKEND_PSWF21, 1, 0, 7);
        bench_case("tlsf_fc2", 60000, ns[i], 3, TLS_BACKEND_PSWF43, 0, 0, 7);
        bench_case("tlsf_fc2", 60000, ns[i], 3, TLS_BACKEND_PSWF21, 0, 0, 7);
    }
    bench_case("tls_fc2_d1", 60000, 8192, 1, TLS_BACKEND_PSWF43, 1, 0, 7);
    bench_case("tlsf_fc2_d1", 60000, 8192, 1, TLS_BACKEND_PSWF43, 0, 0, 7);
    bench_case("tls_ihs_d3", 60000, 8192, 3, TLS_BACKEND_PSWF43, 1, 1, 7);
    bench_case("tlsf_ihs_d3", 60000, 8192, 3, TLS_BACKEND_PSWF43, 0, 1, 7);

    printf("\n-- Small-M / large-N regimes --\n");
    bench_case("smallM_d", 64, 8192, 1, TLS_BACKEND_PSWF43, 1, 0, 11);
    bench_case("smallM_f", 64, 8192, 1, TLS_BACKEND_PSWF43, 0, 0, 11);
    bench_case("medM_d", 580, 8192, 3, TLS_BACKEND_PSWF43, 1, 0, 9);
    bench_case("medM_f", 580, 8192, 3, TLS_BACKEND_PSWF43, 0, 0, 9);
    bench_case("largeN_d", 1800, 131072, 3, TLS_BACKEND_PSWF43, 1, 0, 7);
    bench_case("largeN_f", 1800, 131072, 3, TLS_BACKEND_PSWF43, 0, 0, 7);

    if (sweep) {
        printf("\n-- Crossover sweep (M at N=8192, d=3, pswf43) --\n");
        const int ms[] = {64, 128, 256, 512, 1024, 2048, 8192, 32768, 60000};
        for (int i = 0; i < 9; ++i) {
            bench_case("sweep_d", ms[i], 8192, 3, TLS_BACKEND_PSWF43, 1, 0, 9);
            bench_case("sweep_f", ms[i], 8192, 3, TLS_BACKEND_PSWF43, 0, 0, 9);
        }
    }
    return 0;
}
