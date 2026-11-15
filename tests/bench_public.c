/*
 * bench_public.c - public-API-only timing harness.
 *
 * Uses nothing but <tls.h>, so the identical file compiles against a pristine
 * baseline checkout; tests/bench.sh interleaves the two binaries to compare
 * the branch against main without any cross-binary noise.
 *
 *   ./bench_public [rounds] [reps]     # best-of-rounds(best-of-reps)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    const char *label;
    int M;
    int N;
    int degree;
    int backend;
    int solver;
    int norm;
} workload;

static const workload workloads[] = {
    {"blap_b12_d3", 580, 4096, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"blap_b13_d3", 580, 8192, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"blap_b14_d3", 580, 12288, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"blap_b13_d1", 580, 8192, 1, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"blap_b13_d8", 580, 8192, 8, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"blap_b13_d3_nll", 580, 8192, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_NLL},
    {"blap_b13_d3_pswf21", 580, 8192, 3, TLS_BACKEND_PSWF21, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"lmc_b13_d3", 1800, 8192, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"rrlyr_b12_d3", 950, 4096, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"blap_b16_d3", 580, 65536, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"blap_b17_d3", 580, 131072, 3, TLS_BACKEND_PSWF43, TLS_SOLVER_LEVINSON, TLS_NORM_STANDARD},
    {"ihs_b13_d3", 580, 8192, 3, TLS_BACKEND_PSWF43, 0, TLS_NORM_STANDARD},
    {"ihs_b13_d1", 580, 8192, 1, TLS_BACKEND_PSWF43, 0, TLS_NORM_STANDARD},
};

#define NWORKLOADS ((int)(sizeof(workloads) / sizeof(workloads[0])))

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

int main(int argc, char **argv) {
    const int rounds = (argc > 1) ? atoi(argv[1]) : 5;
    const int reps = (argc > 2) ? atoi(argv[2]) : 5;
    const double f0 = 0.001;
    const double df = 0.0005;

    for (int w = 0; w < NWORKLOADS; ++w) {
        const workload *wl = &workloads[w];
        double *t = (double *)test_alloc((size_t)wl->M * sizeof(double));
        double *y = (double *)test_alloc((size_t)wl->M * sizeof(double));
        double *dy = (double *)test_alloc((size_t)wl->M * sizeof(double));
        float *fy = (float *)test_alloc((size_t)wl->M * sizeof(float));
        float *fdy = (float *)test_alloc((size_t)wl->M * sizeof(float));
        float *power = (float *)test_alloc((size_t)wl->N * sizeof(float));

        make_curve(wl->M, t, y, dy);
        for (int m = 0; m < wl->M; ++m) {
            fy[m] = (float)y[m];
            fdy[m] = (float)dy[m];
        }

        const int is_ihs = (wl->solver == 0);
        int status = 0;
        double best = 1e30;
        for (int round = 0; round < rounds; ++round) {
            for (int rep = 0; rep < reps; ++rep) {
                const double t0 = now_seconds();
                if (is_ihs) {
                    status = tlsf_ihs(t, fy, fdy, wl->M, f0, df, wl->N, wl->degree, wl->backend, wl->norm, power);
                } else {
                    status = tlsf_fastchi2(t, fy, fdy, wl->M, f0, df, wl->N, wl->degree, wl->backend, wl->solver, wl->norm, power, NULL);
                }
                const double dt = now_seconds() - t0;
                if (dt < best) best = dt;
            }
        }
        printf("RESULT %-20s %10.1f ns  st=%d\n", wl->label, best * 1e9, status);
        fflush(stdout);

        free(t);
        free(y);
        free(dy);
        free(fy);
        free(fdy);
        free(power);
    }
    return 0;
}
