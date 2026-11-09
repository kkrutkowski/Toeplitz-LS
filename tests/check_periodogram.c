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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tls.h>
#include <tls_tuning.h>

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
