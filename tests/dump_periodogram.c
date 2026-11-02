/*
 * dump_periodogram.c - public-API-only summary of a fixed set of periodograms.
 *
 * It includes nothing but the installed headers, so the very same file can be
 * compiled against a pristine baseline checkout and against the optimized
 * branch; tests/bench.sh diffs the two outputs as the "did the restructuring
 * change the numbers" gate.
 *
 *   ./dump_periodogram            # human-readable
 *   ./dump_periodogram --dump     # machine-readable DUMP lines
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tls.h>

static uint64_t rng_state = 88172645463325252ull;

static double urand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
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

static void make_curve(int M, double *t, double *y) {
    const double span = 1000.0;
    for (int m = 0; m < M; ++m) {
        t[m] = span * (double)m / (double)M + 0.4 * urand() * (span / (double)M) * 2.0;
        y[m] = 1.0 + 0.7 * sin(2.0 * M_PI * 3.1 * t[m] / span) + 0.4 * sin(2.0 * M_PI * 7.3 * t[m] / span) + 0.05 * (urand() - 0.5);
    }
    for (int m = 1; m < M; ++m) {
        if (t[m] < t[m - 1]) {
            const double tmp = t[m];
            t[m] = t[m - 1];
            t[m - 1] = tmp;
        }
    }
}

static void report(const char *name, int N, int degree, int backend, int solver, int norm, const double *power, int n, int status) {
    double sum = 0.0, chk = 0.0, peak = 0.0;
    int nan_count = 0;
    for (int i = 0; i < n; ++i) {
        const double v = power[i];
        if (isnan(v)) {
            ++nan_count;
            continue;
        }
        sum += v;
        chk += v * (double)(i % 97);
        if (v > peak) peak = v;
    }
    printf("DUMP %-16s N=%d d=%d be=%d so=%d nm=%d st=%d sum=%.10g chk=%.10g peak=%.10g nan=%d\n", name, N, degree, backend, solver, norm, status, sum, chk,
           peak, nan_count);
}

static void run_case(const char *name, int M, int N, int degree, int backend, int solver, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    float *fy = (float *)test_alloc((size_t)M * sizeof(float));
    float *dy = (float *)test_alloc((size_t)M * sizeof(float));
    float *p = (float *)test_alloc((size_t)N * sizeof(float));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) {
        fy[m] = (float)y[m];
        dy[m] = (float)(0.1 + 0.02 * urand());
    }

    const int status = tlsf_fastchi2(t, fy, dy, M, 0.001, 0.0005, N, degree, backend, solver, norm, p, NULL);
    /* float array -> double so both builds report through the same path */
    {
        double *pd = (double *)test_alloc((size_t)N * sizeof(double));
        for (int i = 0; i < N; ++i) pd[i] = (double)p[i];
        report(name, N, degree, backend, solver, norm, pd, N, status);
        free(pd);
    }

    free(t);
    free(y);
    free(fy);
    free(dy);
    free(p);
}

static void run_gls(const char *name, int M, int N, int backend, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    double *dy = (double *)test_alloc((size_t)M * sizeof(double));
    double *p = (double *)test_alloc((size_t)N * sizeof(double));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) dy[m] = 0.1 + 0.02 * urand();
    const int status = tls_fastchi2(t, y, dy, M, 0.001, 0.0005, N, 1, backend, TLS_SOLVER_LEVINSON, norm, p, NULL);
    report(name, N, 1, backend, TLS_SOLVER_LEVINSON, norm, p, N, status);

    free(t);
    free(y);
    free(dy);
    free(p);
}

static void run_ihs(const char *name, int M, int N, int degree, int norm) {
    double *t = (double *)test_alloc((size_t)M * sizeof(double));
    double *y = (double *)test_alloc((size_t)M * sizeof(double));
    float *fy = (float *)test_alloc((size_t)M * sizeof(float));
    float *dy = (float *)test_alloc((size_t)M * sizeof(float));
    float *p = (float *)test_alloc((size_t)N * sizeof(float));
    double *pd = (double *)test_alloc((size_t)N * sizeof(double));

    make_curve(M, t, y);
    for (int m = 0; m < M; ++m) {
        fy[m] = (float)y[m];
        dy[m] = (float)(0.1 + 0.02 * urand());
    }
    const int status = tlsf_ihs(t, fy, dy, M, 0.001, 0.0005, N, degree, 1, norm, p);
    for (int i = 0; i < N; ++i) pd[i] = (double)p[i];
    report(name, N, degree, 1, 0, norm, pd, N, status);

    free(t);
    free(y);
    free(fy);
    free(dy);
    free(p);
    free(pd);
}

int main(void) {
    run_case("gls_d1", 400, 512, 1, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_case("d2_levinson", 500, 700, 2, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_case("d3_nll", 600, 1024, 3, 1, TLS_SOLVER_LEVINSON, TLS_NORM_NLL);
    run_case("d2_odd_grid", 500, 1237, 2, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_case("d3_lra", 500, 900, 3, 3, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_case("d8_levinson", 800, 1024, 8, 1, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD);
    run_gls("gls_d1_double", 400, 512, 1, TLS_NORM_STANDARD);
    run_ihs("ihs_d1", 400, 512, 1, TLS_NORM_STANDARD);
    run_ihs("ihs_d3", 400, 512, 3, TLS_NORM_STANDARD);
    run_ihs("ihs_d3_nll", 400, 512, 3, TLS_NORM_NLL);
    return 0;
}
