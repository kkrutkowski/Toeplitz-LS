/*
 * check_periodogram.c - end-to-end checks for the block-outer periodogram.
 *
 * The library's semantic contract is exercised here at floating-point
 * resolution:
 *   1. deferred bit reversal (slot order) must produce exactly the same
 *      per-bin values as the frequency-ordered path - it only reorders them;
 *   2. the fused FFT depths must agree with depth 1 to float roundoff;
 *   3. the IHS path must be unchanged, with the d = 1 case staying in
 *      frequency order and d >= 2 taking the deferred path;
 *   4. a `--dump` mode prints one machine-readable line per case so the same
 *      source can be built against the pristine baseline tree and compared
 *      numerically (that is what tests/bench.sh does).
 */

#include <math.h>
#include <nanofft.h>
#include <scaling.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tls.h>
#include <tls_tuning.h>
#include <utils.h>

static int failures = 0;
static int dump_mode = 0;

#define CHECK(cond, ...)                                \
    do {                                                \
        if (!(cond)) {                                  \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                        \
            printf("\n");                               \
            ++failures;                                 \
        }                                               \
    } while (0)

static uint64_t rng_state = 88172645463325252ull;

static double urand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0); /* [0,1) */
}

static void *test_alloc(size_t bytes) {
    const size_t padded = (bytes + 63u) & ~(size_t)63u;
    void *ptr = aligned_alloc(64, padded);
    if (!ptr) {
        fprintf(stderr, "allocation failure\n");
        exit(1);
    }
    memset(ptr, 0, padded);
    return ptr;
}

/* Irregularly sampled multiharmonic signal with heterogeneous error bars. */
static void make_curve(int M, double *t, double *y) {
    const double span = 1000.0;
    for (int m = 0; m < M; ++m) {
        const double base = span * (double)m / (double)M;
        t[m] = base + 0.4 * urand() * (span / (double)M) * 2.0;
        y[m] = 1.0 + 0.7 * sin(2.0 * M_PI * 3.1 * t[m] / span) + 0.4 * sin(2.0 * M_PI * 7.3 * t[m] / span) + 0.05 * (urand() - 0.5);
    }
    for (int m = 1; m < M; ++m) { /* keep t sorted: the library assumes it */
        if (t[m] < t[m - 1]) {
            const double tmp = t[m];
            t[m] = t[m - 1];
            t[m - 1] = tmp;
        }
    }
}

static int same_values(const float *a, const float *b, int n, double rtol, double atol) {
    double scale = 0.0, err = 0.0;
    for (int i = 0; i < n; ++i) {
        const int na = isnan(a[i]) != 0, nb = isnan(b[i]) != 0;
        if (na || nb) {
            if (na != nb) return 0;
            continue;
        }
        const double da = a[i], db = b[i];
        if (fabs(da) > scale) scale = fabs(da);
        const double e = fabs(da - db);
        if (e > err) err = e;
    }
    return err <= atol + rtol * scale;
}

static int same_values_d(const double *a, const double *b, int n, double rtol, double atol) {
    double scale = 0.0, err = 0.0;
    for (int i = 0; i < n; ++i) {
        const int na = isnan(a[i]) != 0, nb = isnan(b[i]) != 0;
        if (na || nb) {
            if (na != nb) return 0;
            continue;
        }
        const double da = a[i], db = b[i];
        if (fabs(da) > scale) scale = fabs(da);
        const double e = fabs(da - db);
        if (e > err) err = e;
    }
    return err <= atol + rtol * scale;
}

static void dump_line(const char *tag, const char *name, int N, int degree, int backend, int solver, int norm, const float *power, int n, int nan_count,
                      unsigned long long bytes) {
    double checksum = 0.0, checksum2 = 0.0, peak = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = power[i];
        if (isnan(v)) continue;
        checksum += (double)v;
        checksum2 += (double)v * (double)(i % 97);
        if ((double)v > peak) peak = (double)v;
    }
    printf("DUMP %-6s %s N=%d d=%d be=%d so=%d nm=%d sum=%.9g chk=%.9g peak=%.9g nan=%d work=%llu\n", tag, name, N, degree, backend, solver, norm, checksum,
           checksum2, peak, nan_count, bytes);
}

/* ------------------------------------------------------------------ */

static void run_fastchi2_case(const char *name, int M, int N, int degree, int backend, int solver, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    float *fy = (float *)test_alloc((size_t)M * sizeof(float));
    float *dy = (float *)test_alloc((size_t)M * sizeof(float));
    float *p0 = (float *)test_alloc((size_t)N * sizeof(float));
    float *p1 = (float *)test_alloc((size_t)N * sizeof(float));
    float *pf = (float *)test_alloc((size_t)N * sizeof(float));
    float *c0 = (float *)test_alloc((size_t)N * sizeof(float));
    float *c1 = (float *)test_alloc((size_t)N * sizeof(float));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) {
        fy[m] = (float)y[m];
        dy[m] = (float)(0.1 + 0.02 * urand());
    }

    const double f0 = 0.001;
    const double df = 0.0005;

    tls_tuning_set_cobravo(1);
    tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);
    tls_tuning_set_defer(0);
    int s0 = tlsf_fastchi2(t, fy, dy, M, f0, df, N, degree, backend, solver, norm, p0, c0);
    unsigned long long bytes0 = tls_tuning_work_bytes();
    tls_tuning_set_defer(1);
    int s1 = tlsf_fastchi2(t, fy, dy, M, f0, df, N, degree, backend, solver, norm, p1, c1);
    unsigned long long bytes1 = tls_tuning_work_bytes();

    CHECK(s0 == s1, "%s: status %d vs %d (defer off/on)", name, s0, s1);
    if (s0 >= 0) {
        /* deferred is a pure permutation: allow only ULP-level drift */
        CHECK(same_values(p0, p1, N, 1e-6, 1e-9), "%s: power changed when deferring the bit reversal", name);
        CHECK(same_values(c0, c1, N, 1e-6, 1e-9), "%s: condition bound changed when deferring", name);
        if (norm == TLS_NORM_STANDARD) {
            for (int i = 0; i < N; ++i) {
                const float v = p1[i];
                if (isnan(v)) continue; /* singular solves report NaN */
                CHECK(v >= 0.0f && v < 1.0f, "%s: standard power[%d]=%.9g outside [0,1)", name, i, (double)v);
            }
        }
    }

    /* fused depths must agree with the plain transform */
    tls_tuning_set_defer(1);
    tls_tuning_set_fft_fuse(1);
    int sf = tlsf_fastchi2(t, fy, dy, M, f0, df, N, degree, backend, solver, norm, pf, c1);
    CHECK(sf == s0, "%s: status %d with fuse=1", name, sf);
    if (s0 >= 0) CHECK(same_values(p1, pf, N, 2e-4, 1e-6), "%s: fuse depth changed the periodogram beyond roundoff", name);
    tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);

    if (dump_mode) {
        dump_line("tlsf", name, N, degree, backend, solver, norm, p1, N, s0 < 0 ? s0 : 0, bytes1);
        printf("CHECKSUM %s work_off=%llu work_on=%llu\n", name, bytes0, bytes1);
    } else {
        printf("  %-28s d=%d be=%d so=%d nm=%d  work %llu -> %llu bytes  ok\n", name, degree, backend, solver, norm, bytes0, bytes1);
    }

    free(t);
    free(y);
    free(fy);
    free(dy);
    free(p0);
    free(p1);
    free(pf);
    free(c0);
    free(c1);
}

static void run_double_case(const char *name, int M, int N, int degree, int backend, int solver, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    double *dy = (double *)test_alloc((size_t)M * sizeof(double));
    double *p0 = (double *)test_alloc((size_t)N * sizeof(double));
    double *p1 = (double *)test_alloc((size_t)N * sizeof(double));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) dy[m] = 0.1 + 0.02 * urand();

    tls_tuning_set_defer(0);
    int s0 = tls_fastchi2(t, y, dy, M, 0.001, 0.0005, N, degree, backend, solver, norm, p0, NULL);
    tls_tuning_set_defer(1);
    int s1 = tls_fastchi2(t, y, dy, M, 0.001, 0.0005, N, degree, backend, solver, norm, p1, NULL);

    CHECK(s0 == s1, "%s: status %d vs %d (double, defer off/on)", name, s0, s1);
    if (s0 >= 0) {
        CHECK(same_values_d(p0, p1, N, 1e-12, 1e-15), "%s: double power changed when deferring", name);
        if (norm == TLS_NORM_STANDARD) {
            for (int i = 0; i < N; ++i) {
                const double v = p1[i];
                if (isnan(v)) continue;
                CHECK(v >= 0.0 && v < 1.0, "%s: double standard power[%d]=%.15g outside [0,1)", name, i, v);
            }
        }
    }
    if (dump_mode) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
            if (!isnan(p1[i])) sum += p1[i];
        printf("DUMPD  %-6s %s sum=%.15g nan=%d\n", "tls", name, sum, s0 < 0 ? s0 : 0);
    }

    free(t);
    free(y);
    free(dy);
    free(p0);
    free(p1);
}

static void run_ihs_case(const char *name, int M, int N, int degree, int backend, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    float *fy = (float *)test_alloc((size_t)M * sizeof(float));
    float *dy = (float *)test_alloc((size_t)M * sizeof(float));
    float *p0 = (float *)test_alloc((size_t)N * sizeof(float));
    float *p1 = (float *)test_alloc((size_t)N * sizeof(float));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) {
        fy[m] = (float)y[m];
        dy[m] = (float)(0.1 + 0.02 * urand());
    }

    tls_tuning_set_defer(0);
    int s0 = tlsf_ihs(t, fy, dy, M, 0.001, 0.0005, N, degree, backend, norm, p0);
    tls_tuning_set_defer(1);
    int s1 = tlsf_ihs(t, fy, dy, M, 0.001, 0.0005, N, degree, backend, norm, p1);

    CHECK(s0 == s1, "%s: IHS status %d vs %d", name, s0, s1);
    if (s0 >= 0) {
        CHECK(same_values(p0, p1, N, 1e-6, 1e-9), "%s: IHS power changed when deferring", name);
        for (int i = 0; i < N; ++i) {
            const float v = p1[i];
            CHECK(isfinite(v) && v >= 0.0f, "%s: IHS power[%d]=%.9g is not finite/non-negative", name, i, (double)v);
            if (norm == TLS_NORM_STANDARD && degree == 1) CHECK(v < 1.0f, "%s: IHS d=1 standard power[%d]=%.9g is not below 1", name, i, (double)v);
        }
    }
    if (dump_mode) dump_line("ihs", name, N, degree, backend, 0, norm, p1, N, s0 < 0 ? s0 : 0, 0);

    free(t);
    free(y);
    free(fy);
    free(dy);
    free(p0);
    free(p1);
}

static void run_dd_case(const char *name, int M, int N, int degree, int backend, int solver, int norm) {
    dd_t *t = (dd_t *)test_alloc((size_t)M * sizeof(dd_t));
    dd_t *y = (dd_t *)test_alloc((size_t)M * sizeof(dd_t));
    dd_t *dy = (dd_t *)test_alloc((size_t)M * sizeof(dd_t));
    dd_t *p0 = (dd_t *)test_alloc((size_t)N * sizeof(dd_t));
    dd_t *p1 = (dd_t *)test_alloc((size_t)N * sizeof(dd_t));
    double *td = (double *)test_alloc((size_t)M * sizeof(double));
    double *yd = (double *)test_alloc((size_t)M * sizeof(double));

    make_curve(M, td, yd);
    for (int m = 0; m < M; ++m) {
        t[m] = dd_make(td[m], 0.0);
        y[m] = dd_make(yd[m], 0.0);
        dy[m] = dd_make(0.1 + 0.02 * urand(), 0.0);
    }

    tls_tuning_set_defer(0);
    int s0 = tlsdd_fastchi2(t, y, dy, M, 0.001, 0.0005, N, degree, backend, solver, norm, p0, NULL);
    tls_tuning_set_defer(1);
    int s1 = tlsdd_fastchi2(t, y, dy, M, 0.001, 0.0005, N, degree, backend, solver, norm, p1, NULL);

    CHECK(s0 == s1, "%s: status %d vs %d (dd, defer off/on)", name, s0, s1);
    if (s0 >= 0) {
        double scale = 0.0, err = 0.0;
        for (int i = 0; i < N; ++i) {
            const double a = dd_to_double(p0[i]), b = dd_to_double(p1[i]);
            if (isnan(a) || isnan(b)) {
                CHECK(isnan(a) == isnan(b), "%s: NaN mask differs", name);
                continue;
            }
            if (fabs(a) > scale) scale = fabs(a);
            if (fabs(a - b) > err) err = fabs(a - b);
        }
        CHECK(err <= 1e-18 + 1e-12 * scale, "%s: double-double power changed when deferring (%.3g)", name, err);
        if (norm == TLS_NORM_STANDARD) {
            for (int i = 0; i < N; ++i) {
                const double v = dd_to_double(p1[i]);
                if (isnan(v)) continue;
                CHECK(v >= 0.0 && v < 1.0, "%s: double-double standard power[%d]=%.15g outside [0,1)", name, i, v);
            }
        }
    }

    free(t);
    free(y);
    free(dy);
    free(p0);
    free(p1);
    free(td);
    free(yd);
}

static void run_spread_fastchi2_f(const char *name, int M, int N, int degree, int backend, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    float *fy = (float *)test_alloc((size_t)M * sizeof(float));
    float *dy = (float *)test_alloc((size_t)M * sizeof(float));
    float *pmm = (float *)test_alloc((size_t)N * sizeof(float));
    float *ppp = (float *)test_alloc((size_t)N * sizeof(float));
    float *pauto = (float *)test_alloc((size_t)N * sizeof(float));
    float *cmm = (float *)test_alloc((size_t)N * sizeof(float));
    float *cpp = (float *)test_alloc((size_t)N * sizeof(float));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) {
        fy[m] = (float)y[m];
        dy[m] = (float)(0.1 + 0.02 * urand());
    }

    for (int defer = 0; defer <= 1; ++defer) {
        tls_tuning_set_defer(defer);
        int smm = tlsf_fastchi2_spread(t, fy, dy, M, 0.001, 0.0005, N, degree, backend, TLS_SOLVER_LEVINSON, norm, TLS_SPREAD_MINMAX, pmm, cmm);
        int spp = tlsf_fastchi2_spread(t, fy, dy, M, 0.001, 0.0005, N, degree, backend, TLS_SOLVER_LEVINSON, norm, TLS_SPREAD_PIECEWISE, ppp, cpp);
        int sauto = tlsf_fastchi2_spread(t, fy, dy, M, 0.001, 0.0005, N, degree, backend, TLS_SOLVER_LEVINSON, norm, TLS_SPREAD_AUTO, pauto, NULL);
        CHECK(smm == 0 && spp == 0 && sauto >= 0, "%s (defer=%d): status mm=%d pp=%d auto=%d", name, defer, smm, spp, sauto);
        CHECK(same_values(pmm, ppp, N, 2e-4, 2e-5), "%s (defer=%d): float minmax vs piecewise power mismatch", name, defer);
        CHECK(same_values(cmm, cpp, N, 2e-3, 1e-4), "%s (defer=%d): float minmax vs piecewise cond mismatch", name, defer);
        CHECK(same_values(pmm, pauto, N, 2e-4, 2e-5), "%s (defer=%d): float minmax vs auto power mismatch", name, defer);
    }

    free(t);
    free(y);
    free(fy);
    free(dy);
    free(pmm);
    free(ppp);
    free(pauto);
    free(cmm);
    free(cpp);
}

static void run_spread_fastchi2_d(const char *name, int M, int N, int degree, int backend, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    double *dy = (double *)test_alloc((size_t)M * sizeof(double));
    double *pmm = (double *)test_alloc((size_t)N * sizeof(double));
    double *ppp = (double *)test_alloc((size_t)N * sizeof(double));
    double *pauto = (double *)test_alloc((size_t)N * sizeof(double));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) dy[m] = 0.1 + 0.02 * urand();

    for (int defer = 0; defer <= 1; ++defer) {
        tls_tuning_set_defer(defer);
        int smm = tls_fastchi2_spread(t, y, dy, M, 0.001, 0.0005, N, degree, backend, TLS_SOLVER_LEVINSON, norm, TLS_SPREAD_MINMAX, pmm, NULL);
        int spp = tls_fastchi2_spread(t, y, dy, M, 0.001, 0.0005, N, degree, backend, TLS_SOLVER_LEVINSON, norm, TLS_SPREAD_PIECEWISE, ppp, NULL);
        int sauto = tls_fastchi2_spread(t, y, dy, M, 0.001, 0.0005, N, degree, backend, TLS_SOLVER_LEVINSON, norm, TLS_SPREAD_AUTO, pauto, NULL);
        CHECK(smm >= 0 && spp >= 0 && sauto >= 0, "%s (defer=%d): status mm=%d pp=%d auto=%d", name, defer, smm, spp, sauto);
        CHECK(same_values_d(pmm, ppp, N, 1e-11, 1e-13), "%s (defer=%d): double minmax vs piecewise power mismatch", name, defer);
        CHECK(same_values_d(pmm, pauto, N, 1e-11, 1e-13), "%s (defer=%d): double minmax vs auto power mismatch", name, defer);
    }

    free(t);
    free(y);
    free(dy);
    free(pmm);
    free(ppp);
    free(pauto);
}

static void run_spread_ihs(const char *name, int M, int N, int degree, int backend, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    double *dy = (double *)test_alloc((size_t)M * sizeof(double));
    float *fy = (float *)test_alloc((size_t)M * sizeof(float));
    float *fdy = (float *)test_alloc((size_t)M * sizeof(float));
    float *fpmm = (float *)test_alloc((size_t)N * sizeof(float));
    float *fppp = (float *)test_alloc((size_t)N * sizeof(float));
    double *dpmm = (double *)test_alloc((size_t)N * sizeof(double));
    double *dppp = (double *)test_alloc((size_t)N * sizeof(double));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) {
        dy[m] = 0.1 + 0.02 * urand();
        fy[m] = (float)y[m];
        fdy[m] = (float)dy[m];
    }

    tls_tuning_set_defer(1);
    int sf0 = tlsf_ihs_spread(t, fy, fdy, M, 0.001, 0.0005, N, degree, backend, norm, TLS_SPREAD_MINMAX, fpmm);
    int sf1 = tlsf_ihs_spread(t, fy, fdy, M, 0.001, 0.0005, N, degree, backend, norm, TLS_SPREAD_PIECEWISE, fppp);
    int sd0 = tls_ihs_spread(t, y, dy, M, 0.001, 0.0005, N, degree, backend, norm, TLS_SPREAD_MINMAX, dpmm);
    int sd1 = tls_ihs_spread(t, y, dy, M, 0.001, 0.0005, N, degree, backend, norm, TLS_SPREAD_PIECEWISE, dppp);
    CHECK(sf0 == 0 && sf1 == 0 && sd0 == 0 && sd1 == 0, "%s: ihs status f=(%d,%d) d=(%d,%d)", name, sf0, sf1, sd0, sd1);
    CHECK(same_values(fpmm, fppp, N, 2e-4, 2e-5), "%s: float IHS minmax vs piecewise mismatch", name);
    CHECK(same_values_d(dpmm, dppp, N, 1e-11, 1e-13), "%s: double IHS minmax vs piecewise mismatch", name);

    free(t);
    free(y);
    free(dy);
    free(fy);
    free(fdy);
    free(fpmm);
    free(fppp);
    free(dpmm);
    free(dppp);
}

static void run_spread_policy_and_errors(void) {
    /* Optimizer dynamic switch: small M -> minmax, large M + moderate N -> piecewise. */
    tls_pswf_plan_choice c_small_d = tls_optimize_pswf_plan(8192, 64, 1, 0, TLS_BACKEND_PSWF43, D_ALPHA, D_PSWF43_GAMMA, D_PSWF43_BETA, D_PSWF43_PRE,
                                                            D_PSWF43_PPOLY_BETA, D_PSWF43_PPOLY_PRE, D_PSWF43_PPOLY_CONST, TLS_SPREAD_AUTO);
    tls_pswf_plan_choice c_large_d = tls_optimize_pswf_plan(8192, 60000, 3, 0, TLS_BACKEND_PSWF43, D_ALPHA, D_PSWF43_GAMMA, D_PSWF43_BETA, D_PSWF43_PRE,
                                                            D_PSWF43_PPOLY_BETA, D_PSWF43_PPOLY_PRE, D_PSWF43_PPOLY_CONST, TLS_SPREAD_AUTO);
    tls_pswf_plan_choice c_large_f = tls_optimize_pswf_plan(8192, 60000, 3, 0, TLS_BACKEND_PSWF43, F_ALPHA, F_PSWF43_GAMMA, F_PSWF43_BETA, F_PSWF43_PRE,
                                                            F_PSWF43_PPOLY_BETA, F_PSWF43_PPOLY_PRE, F_PSWF43_PPOLY_CONST, TLS_SPREAD_AUTO);
    tls_pswf_plan_choice c_forced_mm = tls_optimize_pswf_plan(8192, 60000, 3, 0, TLS_BACKEND_PSWF43, D_ALPHA, D_PSWF43_GAMMA, D_PSWF43_BETA, D_PSWF43_PRE,
                                                              D_PSWF43_PPOLY_BETA, D_PSWF43_PPOLY_PRE, D_PSWF43_PPOLY_CONST, TLS_SPREAD_MINMAX);
    tls_pswf_plan_choice c_forced_pp = tls_optimize_pswf_plan(8192, 64, 1, 0, TLS_BACKEND_PSWF43, D_ALPHA, D_PSWF43_GAMMA, D_PSWF43_BETA, D_PSWF43_PRE,
                                                              D_PSWF43_PPOLY_BETA, D_PSWF43_PPOLY_PRE, D_PSWF43_PPOLY_CONST, TLS_SPREAD_PIECEWISE);

    CHECK(c_small_d.spread_mode == TLS_SPREAD_MINMAX, "optimizer should pick MINMAX for M=64, N=8192, d=1 (got %d)", c_small_d.spread_mode);
    CHECK(c_large_d.spread_mode == TLS_SPREAD_PIECEWISE, "optimizer should pick PIECEWISE for double M=60000, N=8192, d=3 (got %d)", c_large_d.spread_mode);
    CHECK(c_large_f.spread_mode == TLS_SPREAD_PIECEWISE, "optimizer should pick PIECEWISE for float M=60000, N=8192, d=3 (got %d)", c_large_f.spread_mode);
    CHECK(c_forced_mm.spread_mode == TLS_SPREAD_MINMAX, "optimizer should honour forced MINMAX");
    CHECK(c_forced_pp.spread_mode == TLS_SPREAD_PIECEWISE, "optimizer should honour forced PIECEWISE");

    /* Invalid spreading argument and double-double rejection of PIECEWISE. */
    double t[16], y[16], dy[16], p[16];
    float fy[16], fdy[16], fp[16];
    dd_t dt[16], ddy_y[16], ddy[16], dp[16];
    for (int i = 0; i < 16; ++i) {
        t[i] = (double)i;
        y[i] = sin((double)i);
        dy[i] = 0.1;
        fy[i] = (float)y[i];
        fdy[i] = 0.1f;
        dt[i] = dd_make(t[i], 0.0);
        ddy_y[i] = dd_make(y[i], 0.0);
        ddy[i] = dd_make(0.1, 0.0);
    }
    CHECK(tlsf_fastchi2_spread(t, fy, fdy, 16, 0.01, 0.01, 16, 1, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, 99, fp, NULL) == TLS_ERR_ARGUMENT,
          "tlsf_fastchi2_spread should reject invalid spreading");
    CHECK(tls_ihs_spread(t, y, dy, 16, 0.01, 0.01, 16, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, -1, p) == TLS_ERR_ARGUMENT,
          "tls_ihs_spread should reject invalid spreading");
    CHECK(tlsdd_fastchi2_spread(dt, ddy_y, ddy, 16, 0.01, 0.01, 16, 1, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_PIECEWISE, dp,
                                NULL) == TLS_ERR_ARGUMENT,
          "tlsdd_fastchi2_spread should reject PIECEWISE");
    CHECK(tlsdd_ihs_spread(dt, ddy_y, ddy, 16, 0.01, 0.01, 16, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_SPREAD_PIECEWISE, dp) == TLS_ERR_ARGUMENT,
          "tlsdd_ihs_spread should reject PIECEWISE");
}

static void run_vectorize_fastchi2_case(const char *name, int M, int N, int degree, int backend, int solver, int norm, int spread) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    double *dy = (double *)test_alloc((size_t)M * sizeof(double));
    float *fy = (float *)test_alloc((size_t)M * sizeof(float));
    float *fdy = (float *)test_alloc((size_t)M * sizeof(float));
    float *fpd = (float *)test_alloc((size_t)N * sizeof(float));
    float *fpb = (float *)test_alloc((size_t)N * sizeof(float));
    float *fpa = (float *)test_alloc((size_t)N * sizeof(float));
    float *fcd = (float *)test_alloc((size_t)N * sizeof(float));
    float *fcb = (float *)test_alloc((size_t)N * sizeof(float));
    double *dpd = (double *)test_alloc((size_t)N * sizeof(double));
    double *dpb = (double *)test_alloc((size_t)N * sizeof(double));
    double *dpa = (double *)test_alloc((size_t)N * sizeof(double));
    double *dcd = (double *)test_alloc((size_t)N * sizeof(double));
    double *dcb = (double *)test_alloc((size_t)N * sizeof(double));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) {
        dy[m] = 0.1 + 0.02 * urand();
        fy[m] = (float)y[m];
        fdy[m] = (float)dy[m];
    }

    for (int defer = 0; defer <= 1; ++defer) {
        tls_tuning_set_defer(defer);
        int sfd = tlsf_fastchi2_opts(t, fy, fdy, M, 0.001, 0.0005, N, degree, backend, solver, norm, spread, TLS_VEC_DEPTH, fpd, fcd);
        int sfb = tlsf_fastchi2_opts(t, fy, fdy, M, 0.001, 0.0005, N, degree, backend, solver, norm, spread, TLS_VEC_BREADTH, fpb, fcb);
        int sfa = tlsf_fastchi2_opts(t, fy, fdy, M, 0.001, 0.0005, N, degree, backend, solver, norm, spread, TLS_VEC_AUTO, fpa, NULL);
        CHECK(sfd == 0 && sfb == 0 && sfa >= 0, "%s float (defer=%d): status d=%d b=%d a=%d", name, defer, sfd, sfb, sfa);
        CHECK(same_values(fpd, fpb, N, 3e-4, 3e-5), "%s float (defer=%d): depth vs breadth power mismatch", name, defer);
        CHECK(same_values(fcd, fcb, N, 3e-3, 3e-4), "%s float (defer=%d): depth vs breadth cond mismatch", name, defer);
        CHECK(same_values(fpd, fpa, N, 3e-4, 3e-5), "%s float (defer=%d): depth vs auto power mismatch", name, defer);

        int sdd = tls_fastchi2_opts(t, y, dy, M, 0.001, 0.0005, N, degree, backend, solver, norm, spread, TLS_VEC_DEPTH, dpd, dcd);
        int sdb = tls_fastchi2_opts(t, y, dy, M, 0.001, 0.0005, N, degree, backend, solver, norm, spread, TLS_VEC_BREADTH, dpb, dcb);
        int sda = tls_fastchi2_opts(t, y, dy, M, 0.001, 0.0005, N, degree, backend, solver, norm, spread, TLS_VEC_AUTO, dpa, NULL);
        CHECK(sdd == 0 && sdb == 0 && sda >= 0, "%s double (defer=%d): status d=%d b=%d a=%d", name, defer, sdd, sdb, sda);
        CHECK(same_values_d(dpd, dpb, N, 1e-10, 1e-11), "%s double (defer=%d): depth vs breadth power mismatch", name, defer);
        CHECK(same_values_d(dcd, dcb, N, 1e-8, 1e-9), "%s double (defer=%d): depth vs breadth cond mismatch", name, defer);
        CHECK(same_values_d(dpd, dpa, N, 1e-10, 1e-11), "%s double (defer=%d): depth vs auto power mismatch", name, defer);
    }

    free(t);
    free(y);
    free(dy);
    free(fy);
    free(fdy);
    free(fpd);
    free(fpb);
    free(fpa);
    free(fcd);
    free(fcb);
    free(dpd);
    free(dpb);
    free(dpa);
    free(dcd);
    free(dcb);
}

static void run_vectorize_ihs_case(const char *name, int M, int N, int degree, int backend, int norm, int spread) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    double *dy = (double *)test_alloc((size_t)M * sizeof(double));
    float *fy = (float *)test_alloc((size_t)M * sizeof(float));
    float *fdy = (float *)test_alloc((size_t)M * sizeof(float));
    float *fpd = (float *)test_alloc((size_t)N * sizeof(float));
    float *fpb = (float *)test_alloc((size_t)N * sizeof(float));
    double *dpd = (double *)test_alloc((size_t)N * sizeof(double));
    double *dpb = (double *)test_alloc((size_t)N * sizeof(double));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) {
        dy[m] = 0.1 + 0.02 * urand();
        fy[m] = (float)y[m];
        fdy[m] = (float)dy[m];
    }

    for (int defer = 0; defer <= 1; ++defer) {
        tls_tuning_set_defer(defer);
        int sf0 = tlsf_ihs_opts(t, fy, fdy, M, 0.001, 0.0005, N, degree, backend, norm, spread, TLS_VEC_DEPTH, fpd);
        int sf1 = tlsf_ihs_opts(t, fy, fdy, M, 0.001, 0.0005, N, degree, backend, norm, spread, TLS_VEC_BREADTH, fpb);
        int sd0 = tls_ihs_opts(t, y, dy, M, 0.001, 0.0005, N, degree, backend, norm, spread, TLS_VEC_DEPTH, dpd);
        int sd1 = tls_ihs_opts(t, y, dy, M, 0.001, 0.0005, N, degree, backend, norm, spread, TLS_VEC_BREADTH, dpb);
        CHECK(sf0 == 0 && sf1 == 0 && sd0 == 0 && sd1 == 0, "%s (defer=%d): ihs status", name, defer);
        CHECK(same_values(fpd, fpb, N, 3e-4, 3e-5), "%s (defer=%d): float IHS depth vs breadth mismatch", name, defer);
        CHECK(same_values_d(dpd, dpb, N, 1e-10, 1e-11), "%s (defer=%d): double IHS depth vs breadth mismatch", name, defer);
    }

    free(t);
    free(y);
    free(dy);
    free(fy);
    free(fdy);
    free(fpd);
    free(fpb);
    free(dpd);
    free(dpb);
}

static void run_vectorize_dd_and_errors(void) {
    const int M = 180;
    const int N = 333;
    double *td = (double *)test_alloc((size_t)M * sizeof(double));
    double *yd = (double *)test_alloc((size_t)M * sizeof(double));
    double *dyd = (double *)test_alloc((size_t)M * sizeof(double));
    dd_t *t = (dd_t *)test_alloc((size_t)M * sizeof(dd_t));
    dd_t *y = (dd_t *)test_alloc((size_t)M * sizeof(dd_t));
    dd_t *dy = (dd_t *)test_alloc((size_t)M * sizeof(dd_t));
    dd_t *pdd = (dd_t *)test_alloc((size_t)N * sizeof(dd_t));
    double *pref = (double *)test_alloc((size_t)N * sizeof(double));
    double *pdd_as_d = (double *)test_alloc((size_t)N * sizeof(double));

    make_curve(M, td, yd);
    for (int m = 0; m < M; ++m) {
        dyd[m] = 0.1 + 0.02 * urand();
        t[m] = dd_make(td[m], 0.0);
        y[m] = dd_make(yd[m], 0.0);
        dy[m] = dd_make(dyd[m], 0.0);
    }

    const int backends[3] = {TLS_BACKEND_PSWF43, TLS_BACKEND_PSWF21, TLS_BACKEND_LRA};
    for (int bi = 0; bi < 3; ++bi) {
        int s_dd = tlsdd_fastchi2_opts(t, y, dy, M, 0.001, 0.0005, N, 2, backends[bi], TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX,
                                       TLS_VEC_BREADTH, pdd, NULL);
        int s_ref = tls_fastchi2_opts(td, yd, dyd, M, 0.001, 0.0005, N, 2, backends[bi], TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX,
                                      TLS_VEC_DEPTH, pref, NULL);
        CHECK(s_dd >= 0 && s_ref >= 0, "dd breadth fastchi2 be=%d status dd=%d ref=%d", backends[bi], s_dd, s_ref);
        for (int i = 0; i < N; ++i) pdd_as_d[i] = dd_to_double(pdd[i]);
        CHECK(same_values_d(pref, pdd_as_d, N, 1e-10, 1e-11), "dd breadth vs double depth fastchi2 mismatch be=%d", backends[bi]);

        int si_dd = tlsdd_ihs_opts(t, y, dy, M, 0.001, 0.0005, N, 2, backends[bi], TLS_NORM_IHS_DEFAULT, TLS_SPREAD_MINMAX, TLS_VEC_BREADTH, pdd);
        int si_ref = tls_ihs_opts(td, yd, dyd, M, 0.001, 0.0005, N, 2, backends[bi], TLS_NORM_IHS_DEFAULT, TLS_SPREAD_MINMAX, TLS_VEC_DEPTH, pref);
        CHECK(si_dd == 0 && si_ref == 0, "dd breadth ihs be=%d status dd=%d ref=%d", backends[bi], si_dd, si_ref);
        for (int i = 0; i < N; ++i) pdd_as_d[i] = dd_to_double(pdd[i]);
        CHECK(same_values_d(pref, pdd_as_d, N, 1e-10, 1e-11), "dd breadth vs double depth ihs mismatch be=%d", backends[bi]);
    }

    /* Validation: invalid vectorize & DD rejection of TLS_VEC_DEPTH. */
    CHECK(tlsdd_fastchi2_opts(t, y, dy, M, 0.001, 0.0005, 32, 1, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX, TLS_VEC_DEPTH,
                              pdd, NULL) == TLS_ERR_ARGUMENT,
          "tlsdd_fastchi2_opts must reject TLS_VEC_DEPTH");
    CHECK(tlsdd_ihs_opts(t, y, dy, M, 0.001, 0.0005, 32, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX, TLS_VEC_DEPTH, pdd) == TLS_ERR_ARGUMENT,
          "tlsdd_ihs_opts must reject TLS_VEC_DEPTH");
    CHECK(tls_fastchi2_opts(td, yd, dyd, M, 0.001, 0.0005, 32, 1, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_AUTO, 99, pref,
                            NULL) == TLS_ERR_ARGUMENT,
          "tls_fastchi2_opts must reject invalid vectorize");
    CHECK(tls_ihs_opts(td, yd, dyd, M, 0.001, 0.0005, 32, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_SPREAD_AUTO, -1, pref) == TLS_ERR_ARGUMENT,
          "tls_ihs_opts must reject invalid vectorize");

    free(td);
    free(yd);
    free(dyd);
    free(t);
    free(y);
    free(dy);
    free(pdd);
    free(pref);
    free(pdd_as_d);
}

static int f32_is_nan_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    return (u & 0x7fffffffu) > 0x7f800000u;
}
static int f64_is_nan_bits(double v) {
    uint64_t u;
    memcpy(&u, &v, sizeof u);
    return (u & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000);
}

static void run_half_precision_suite(void) {
    const int M = 400;
    const int N = 513;
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *yd = (double *)test_alloc((size_t)M * sizeof(double));
    double *dyd = (double *)test_alloc((size_t)M * sizeof(double));
    float *yf = (float *)test_alloc((size_t)M * sizeof(float));
    float *dyf = (float *)test_alloc((size_t)M * sizeof(float));
    double *pref = (double *)test_alloc((size_t)N * sizeof(double));
    float *ph0 = (float *)test_alloc((size_t)N * sizeof(float));
    float *ph1 = (float *)test_alloc((size_t)N * sizeof(float));
    float *phd = (float *)test_alloc((size_t)N * sizeof(float));
    float *ch = (float *)test_alloc((size_t)N * sizeof(float));

    make_curve(M, t, yd);
    for (int m = 0; m < M; ++m) {
        dyd[m] = 0.1 + 0.02 * urand();
        yf[m] = (float)yd[m];
        dyf[m] = (float)dyd[m];
    }

    const int native_fp16 = tlsh_using_native_fp16();
    CHECK(native_fp16 == 0 || native_fp16 == 1, "tlsh_using_native_fp16 invalid return %d", native_fp16);
    printf("  tlsh_using_native_fp16 = %d\n", native_fp16);

    /* Backends x solvers x spreads x vectorize x defer x normalizations */
    struct {
        const char *name;
        int degree;
        int backend;
        int solver;
        int norm;
    } cases[] = {
        {"h_gls_pswf43", 1, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
        {"h_gls_pswf21", 1, TLS_BACKEND_PSWF21, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
        {"h_d2_levinson", 2, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
        {"h_d3_levinson_nll", 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_NLL},
        {"h_d2_zohar_anova", 2, TLS_BACKEND_PSWF21, TLS_SOLVER_ZOHAR, TLS_NORM_ANOVA},
        {"h_d2_bareiss_reg", 2, TLS_BACKEND_PSWF43, TLS_SOLVER_BAREISS, TLS_NORM_REGULARIZED},
        {"h_d2_ldlt", 2, TLS_BACKEND_PSWF43, TLS_SOLVER_LDLT, TLS_NORM_STANDARD},
        {"h_d2_svd", 2, TLS_BACKEND_PSWF43, TLS_SOLVER_SVD, TLS_NORM_STANDARD},
        {"h_d2_lra", 2, TLS_BACKEND_LRA, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    };
    const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int ci = 0; ci < ncases; ++ci) {
        const int spread_max = (cases[ci].backend == TLS_BACKEND_LRA) ? 1 : 2;
        const int spreads[2] = {TLS_SPREAD_MINMAX, TLS_SPREAD_PIECEWISE};
        int s_ref = tls_fastchi2_opts(t, yd, dyd, M, 0.001, 0.0005, N, cases[ci].degree, cases[ci].backend, cases[ci].solver, cases[ci].norm, TLS_SPREAD_MINMAX,
                                      TLS_VEC_DEPTH, pref, NULL);
        CHECK(s_ref >= 0, "%s: double reference failed (%d)", cases[ci].name, s_ref);

        for (int si = 0; si < spread_max; ++si) {
            for (int defer = 0; defer <= 1; ++defer) {
                tls_tuning_set_defer(defer);
                int s0 = tlsh_fastchi2_opts(t, yf, dyf, M, 0.001, 0.0005, N, cases[ci].degree, cases[ci].backend, cases[ci].solver, cases[ci].norm, spreads[si],
                                            TLS_VEC_DEPTH, ph0, NULL);
                int s1 = tlsh_fastchi2_opts(t, yf, dyf, M, 0.001, 0.0005, N, cases[ci].degree, cases[ci].backend, cases[ci].solver, cases[ci].norm, spreads[si],
                                            TLS_VEC_BREADTH, ph1, NULL);
                int sd = tlsh_fastchi2_opts_d(t, yd, dyd, M, 0.001, 0.0005, N, cases[ci].degree, cases[ci].backend, cases[ci].solver, cases[ci].norm,
                                              spreads[si], TLS_VEC_DEPTH, phd, NULL);
                CHECK(s0 >= 0 && s1 >= 0 && sd >= 0, "%s (sp=%d defer=%d): status %d %d %d", cases[ci].name, spreads[si], defer, s0, s1, sd);
                /* Depth vs breadth and float vs double inputs must match within FP16 rounding */
                CHECK(same_values(ph0, ph1, N, 2e-2, 2e-2), "%s (sp=%d defer=%d): depth vs breadth", cases[ci].name, spreads[si], defer);
                CHECK(same_values(ph0, phd, N, 1e-2, 1e-2), "%s (sp=%d defer=%d): float vs double input", cases[ci].name, spreads[si], defer);
                if (cases[ci].norm == TLS_NORM_STANDARD && defer == 1 && si == 0) {
                    double max_diff = 0.0;
                    for (int k = 0; k < N; ++k) {
                        if (!f32_is_nan_bits(ph0[k]) && !f64_is_nan_bits(pref[k])) {
                            double d = fabs((double)ph0[k] - pref[k]);
                            if (d > max_diff) max_diff = d;
                        }
                    }
                    CHECK(max_diff < 2.5e-2, "%s: FP16 vs FP64 max |dR2|=%.3e", cases[ci].name, max_diff);
                }
            }
        }
    }

    /* IHS across backends, normalizations, depth/breadth, float/double inputs */
    const int ihs_norms[3] = {TLS_NORM_IHS_RAW, TLS_NORM_IHS_DEFAULT, TLS_NORM_NLL};
    const int ihs_backends[3] = {TLS_BACKEND_PSWF43, TLS_BACKEND_PSWF21, TLS_BACKEND_LRA};
    for (int bi = 0; bi < 3; ++bi) {
        for (int ni = 0; ni < 3; ++ni) {
            for (int defer = 0; defer <= 1; ++defer) {
                tls_tuning_set_defer(defer);
                int s0 = tlsh_ihs_opts(t, yf, dyf, M, 0.001, 0.0005, N, 3, ihs_backends[bi], ihs_norms[ni], TLS_SPREAD_MINMAX, TLS_VEC_DEPTH, ph0);
                int s1 = tlsh_ihs_opts(t, yf, dyf, M, 0.001, 0.0005, N, 3, ihs_backends[bi], ihs_norms[ni], TLS_SPREAD_MINMAX, TLS_VEC_BREADTH, ph1);
                int sd = tlsh_ihs_opts_d(t, yd, dyd, M, 0.001, 0.0005, N, 3, ihs_backends[bi], ihs_norms[ni], TLS_SPREAD_MINMAX, TLS_VEC_DEPTH, phd);
                CHECK(s0 == 0 && s1 == 0 && sd == 0, "ihs half be=%d nm=%d defer=%d status %d %d %d", ihs_backends[bi], ihs_norms[ni], defer, s0, s1, sd);
                CHECK(same_values(ph0, ph1, N, 2e-2, 2e-2), "ihs half depth vs breadth be=%d nm=%d defer=%d", ihs_backends[bi], ihs_norms[ni], defer);
                CHECK(same_values(ph0, phd, N, 1e-2, 1e-2), "ihs half float vs double be=%d nm=%d defer=%d", ihs_backends[bi], ihs_norms[ni], defer);
            }
        }
    }

    /* Ill-conditioned clustered sampling: verify condition threshold flags
     * inaccurate bins as NaN when cond==NULL and reports finite/inf cond when cond!=NULL. */
    {
        const int M_ill = 120;
        const int N_ill = 256;
        double *ti = (double *)test_alloc((size_t)M_ill * sizeof(double));
        double *yi = (double *)test_alloc((size_t)M_ill * sizeof(double));
        double *dyi = (double *)test_alloc((size_t)M_ill * sizeof(double));
        float *pi_nan = (float *)test_alloc((size_t)N_ill * sizeof(float));
        float *pi_cond = (float *)test_alloc((size_t)N_ill * sizeof(float));
        float *ci_arr = (float *)test_alloc((size_t)N_ill * sizeof(float));
        double *pi_ref = (double *)test_alloc((size_t)N_ill * sizeof(double));
        for (int m = 0; m < M_ill; ++m) {
            /* Clustered around integers -> strong aliasing at f = 0.5, 1.0, ... */
            ti[m] = (double)(m % 20) + 0.02 * ((double)(m / 20) - 2.5);
            yi[m] = sin(2.0 * 3.141592653589793 * 0.23 * ti[m]) + 0.1 * urand();
            dyi[m] = 0.1;
        }
        tls_tuning_set_defer(1);
        int snan = tlsh_fastchi2_opts_d(ti, yi, dyi, M_ill, 0.05, 0.01, N_ill, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX,
                                        TLS_VEC_DEPTH, pi_nan, NULL);
        int scond = tlsh_fastchi2_opts_d(ti, yi, dyi, M_ill, 0.05, 0.01, N_ill, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD,
                                         TLS_SPREAD_MINMAX, TLS_VEC_DEPTH, pi_cond, ci_arr);
        int sref = tls_fastchi2_opts(ti, yi, dyi, M_ill, 0.05, 0.01, N_ill, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX,
                                     TLS_VEC_DEPTH, pi_ref, NULL);
        CHECK(snan > 0 && scond == 0 && sref >= 0, "half ill-cond: expected NaNs (snan=%d scond=%d sref=%d)", snan, scond, sref);
        int high_cond_flagged = 0;
        double max_kept_err = 0.0;
        for (int k = 0; k < N_ill; ++k) {
            if (ci_arr[k] > 500.0f) {
                CHECK(f32_is_nan_bits(pi_nan[k]), "half ill-cond bin %d cond=%.1f not NaN", k, (double)ci_arr[k]);
                ++high_cond_flagged;
            } else if (!f32_is_nan_bits(pi_nan[k]) && !f64_is_nan_bits(pi_ref[k])) {
                double e = fabs((double)pi_nan[k] - pi_ref[k]);
                if (e > max_kept_err) max_kept_err = e;
            }
        }
        CHECK(high_cond_flagged > 0, "half ill-cond: expected some cond > 500 bins");
        printf("  half ill-cond: %d/%d bins flagged NaN (max kept |dR2| = %.3e)\n", snan, N_ill, max_kept_err);
        free(ti);
        free(yi);
        free(dyi);
        free(pi_nan);
        free(pi_cond);
        free(ci_arr);
        free(pi_ref);
    }

    free(t);
    free(yd);
    free(dyd);
    free(yf);
    free(dyf);
    free(pref);
    free(ph0);
    free(ph1);
    free(phd);
    free(ch);
}

int main(int argc, char **argv) {
    dump_mode = (argc > 1 && strcmp(argv[1], "--dump") == 0);

    printf("check_periodogram: fastchi2, defer off vs on\n");
    /* d = 1 (GLS) also takes the deferred path, as requested. */
    run_fastchi2_case("gls_d1", 400, 512, 1, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_fastchi2_case("gls_d1_pswf21", 400, 512, 1, 2, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_fastchi2_case("d2_levinson", 500, 700, 2, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_fastchi2_case("d3_levinson_nll", 600, 1024, 3, 1, TLS_SOLVER_LEVINSON, TLS_NORM_NLL);
    run_fastchi2_case("d3_zohar_anova", 600, 900, 3, 1, TLS_SOLVER_ZOHAR, TLS_NORM_ANOVA);
    run_fastchi2_case("d3_bareiss_reg", 600, 1024, 3, 1, TLS_SOLVER_BAREISS, TLS_NORM_REGULARIZED);
    run_fastchi2_case("d2_ldlt", 400, 512, 2, 1, TLS_SOLVER_LDLT, TLS_NORM_STANDARD);
    run_fastchi2_case("d2_svd", 400, 512, 2, 1, TLS_SOLVER_SVD, TLS_NORM_STANDARD);
    run_fastchi2_case("d8_levinson", 800, 1024, 8, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_fastchi2_case("d3_lra", 500, 900, 3, 3, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    /* a grid that is not a multiple of the plan block size */
    run_fastchi2_case("d2_odd_grid", 500, 1237, 2, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);

    printf("check_periodogram: double precision\n");
    run_double_case("gls_d1", 400, 512, 1, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_double_case("d3", 600, 900, 3, 1, TLS_SOLVER_LEVINSON, TLS_NORM_NLL);

    printf("check_periodogram: IHS (d = 1 stays natural, d >= 2 defers)\n");
    run_ihs_case("ihs_d1", 400, 512, 1, 1, TLS_NORM_STANDARD);
    run_ihs_case("ihs_d3_raw", 400, 512, 3, 1, TLS_NORM_IHS_RAW);
    run_ihs_case("ihs_d3_default", 400, 512, 3, 1, TLS_NORM_IHS_DEFAULT);
    run_ihs_case("ihs_d3_nll", 400, 512, 3, 1, TLS_NORM_NLL);
    run_ihs_case("ihs_d3_odd", 400, 777, 3, 1, TLS_NORM_NLL);

    printf("check_periodogram: double-double\n");
    run_dd_case("gls_d1", 200, 256, 1, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_dd_case("d2", 220, 256, 2, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);

    printf("check_periodogram: piecewise vs minmax PSWF spreading\n");
    run_spread_fastchi2_f("f_pswf43_d1", 500, 512, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD);
    run_spread_fastchi2_f("f_pswf21_d1", 500, 512, 1, TLS_BACKEND_PSWF21, TLS_NORM_STANDARD);
    run_spread_fastchi2_f("f_pswf43_d3", 600, 1024, 3, TLS_BACKEND_PSWF43, TLS_NORM_NLL);
    run_spread_fastchi2_f("f_pswf21_d3", 600, 1024, 3, TLS_BACKEND_PSWF21, TLS_NORM_STANDARD);
    run_spread_fastchi2_f("f_pswf43_largeM", 16384, 4096, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD);
    run_spread_fastchi2_d("d_pswf43_d1", 500, 512, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD);
    run_spread_fastchi2_d("d_pswf21_d1", 500, 512, 1, TLS_BACKEND_PSWF21, TLS_NORM_STANDARD);
    run_spread_fastchi2_d("d_pswf43_d3", 600, 1024, 3, TLS_BACKEND_PSWF43, TLS_NORM_NLL);
    run_spread_fastchi2_d("d_pswf21_d3", 600, 1024, 3, TLS_BACKEND_PSWF21, TLS_NORM_STANDARD);
    run_spread_fastchi2_d("d_pswf43_largeM", 16384, 4096, 3, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD);
    run_spread_ihs("ihs_pswf43_d1", 500, 512, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD);
    run_spread_ihs("ihs_pswf21_d3", 600, 1024, 3, TLS_BACKEND_PSWF21, TLS_NORM_NLL);
    run_spread_ihs("ihs_pswf43_d3", 600, 1024, 3, TLS_BACKEND_PSWF43, TLS_NORM_IHS_DEFAULT);
    run_spread_policy_and_errors();
    tls_tuning_set_spread(TLS_SPREAD_AUTO);

    printf("check_periodogram: breadth vs depth vs auto vectorization\n");
    run_vectorize_fastchi2_case("v_gls_pswf43_mm", 500, 777, 1, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX);
    run_vectorize_fastchi2_case("v_gls_pswf43_pp", 500, 777, 1, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_PIECEWISE);
    run_vectorize_fastchi2_case("v_d3_pswf43_mm", 600, 1237, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_NLL, TLS_SPREAD_MINMAX);
    run_vectorize_fastchi2_case("v_d3_pswf43_pp", 600, 1237, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_NLL, TLS_SPREAD_PIECEWISE);
    run_vectorize_fastchi2_case("v_d2_pswf21_pp", 500, 777, 2, TLS_BACKEND_PSWF21, TLS_SOLVER_ZOHAR, TLS_NORM_ANOVA, TLS_SPREAD_PIECEWISE);
    run_vectorize_fastchi2_case("v_d2_bareiss_reg", 500, 513, 2, TLS_BACKEND_PSWF43, TLS_SOLVER_BAREISS, TLS_NORM_REGULARIZED, TLS_SPREAD_MINMAX);
    run_vectorize_fastchi2_case("v_d2_ldlt", 400, 513, 2, TLS_BACKEND_PSWF43, TLS_SOLVER_LDLT, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX);
    run_vectorize_fastchi2_case("v_d2_svd", 400, 513, 2, TLS_BACKEND_PSWF43, TLS_SOLVER_SVD, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX);
    run_vectorize_fastchi2_case("v_d3_lra", 500, 777, 3, TLS_BACKEND_LRA, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX);
    run_vectorize_ihs_case("v_ihs_d1_pswf43", 500, 777, 1, TLS_BACKEND_PSWF43, TLS_NORM_STANDARD, TLS_SPREAD_MINMAX);
    run_vectorize_ihs_case("v_ihs_d3_pswf43_pp", 600, 1237, 3, TLS_BACKEND_PSWF43, TLS_NORM_NLL, TLS_SPREAD_PIECEWISE);
    run_vectorize_ihs_case("v_ihs_d3_pswf21_mm", 600, 777, 3, TLS_BACKEND_PSWF21, TLS_NORM_IHS_DEFAULT, TLS_SPREAD_MINMAX);
    run_vectorize_ihs_case("v_ihs_d3_lra", 500, 777, 3, TLS_BACKEND_LRA, TLS_NORM_NLL, TLS_SPREAD_MINMAX);
    run_vectorize_dd_and_errors();
    tls_tuning_set_vectorize(TLS_VEC_AUTO);

    printf("check_periodogram: half precision (FP16)\n");
    run_half_precision_suite();

    tls_tuning_set_defer(TLS_TUNING_DEFER_DEFAULT);
    tls_tuning_set_cobravo(TLS_TUNING_COBRAVO_DEFAULT);
    tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);

    if (failures) {
        printf("check_periodogram: %d failure(s)\n", failures);
        return 1;
    }
    printf("check_periodogram: OK\n");
    return 0;
}
