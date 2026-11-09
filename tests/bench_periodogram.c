/*
 * bench_periodogram.c - interleaved end-to-end benchmark of the algorithm
 * variants, all inside one process (so binary layout and startup cost cannot
 * bias one variant against another).
 *
 * Variants:
 *   fuse=1..N  cobravo=vec  defer=on     (fused-twiddle depth sweep)
 *   fuse=best  cobravo=vec  defer=off    (deferred bit reversal)
 *   fuse=best  cobravo=scal defer=on     (vectorized COBRAVO)
 *   fuse=1     cobravo=scal defer=off    (the old code's behaviour)
 *
 * Every round runs every variant once on the same workload, rotating the order
 * so no variant is always measured on a cold/boosted clock; the reported time
 * is the best of R rounds of the best of REPS internal repetitions.
 *
 *   ./bench_periodogram [rounds] [reps]
 */

#include <math.h>
#include <nanofft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tls.h>
#include <tls_tuning.h>

#define MAX_VARIANTS 8

typedef struct {
    int fuse;
    int cobravo;
    int defer;
    int order;
    const char *name;
} variant;

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
    const int fuse_build = NANOFFT_FFT_FUSE_FLOAT;
    const double f0 = 0.001;
    const double df = 0.0005;

    variant variants[MAX_VARIANTS];
    int nvariants = 0;
    char names[MAX_VARIANTS][48];
    /* default order first, then the same depth in the other order, so the
     * fusion-order effect is visible end to end, not just in the FFT bench */
    for (int fuse = 1; fuse <= fuse_build; ++fuse) {
        const int vi = nvariants;
        snprintf(names[vi], sizeof(names[vi]), "fuse=%d vec end", fuse);
        variants[vi] = (variant){fuse, 1, 1, TLS_FFT_ORDER_END, names[vi]};
        ++nvariants;
        if (fuse == 3) {
            const int vf = nvariants;
            snprintf(names[vf], sizeof(names[vf]), "fuse=%d vec front", fuse);
            variants[vf] = (variant){fuse, 1, 1, TLS_FFT_ORDER_FRONT, names[vf]};
            ++nvariants;
        }
    }
    int vi;
    vi = nvariants;
    snprintf(names[vi], sizeof(names[vi]), "fuse=1 vec natural");
    variants[vi] = (variant){1, 1, 0, TLS_FFT_ORDER_FRONT, names[vi]};
    ++nvariants;

    vi = nvariants;
    snprintf(names[vi], sizeof(names[vi]), "fuse=auto vec natural");
    variants[vi] = (variant){TLS_TUNING_FFT_FUSE_AUTO, 1, 0, TLS_FFT_ORDER_END, names[vi]};
    ++nvariants;

    vi = nvariants;
    snprintf(names[vi], sizeof(names[vi]), "fuse=auto scal defer");
    variants[vi] = (variant){TLS_TUNING_FFT_FUSE_AUTO, 0, 1, TLS_FFT_ORDER_END, names[vi]};
    ++nvariants;

    vi = nvariants;
    snprintf(names[vi], sizeof(names[vi]), "fuse=1 scal natural");
    variants[vi] = (variant){1, 0, 0, TLS_FFT_ORDER_FRONT, names[vi]};
    ++nvariants;

    printf("bench_periodogram: %d rounds, %d reps, build fuse depth %d\n", rounds, reps, fuse_build);

    for (int w = 0; w < NWORKLOADS; ++w) {
        const workload *wl = &workloads[w];
        double *t = (double *)test_alloc((size_t)wl->M * sizeof(double));
        double *y = (double *)test_alloc((size_t)wl->M * sizeof(double));
        double *dy = (double *)test_alloc((size_t)wl->M * sizeof(double));
        float *fy = (float *)test_alloc((size_t)wl->M * sizeof(float));
        float *fdy = (float *)test_alloc((size_t)wl->M * sizeof(float));
        float *ref = (float *)test_alloc((size_t)wl->N * sizeof(float));
        float *power = (float *)test_alloc((size_t)wl->N * sizeof(float));

        make_curve(wl->M, t, y, dy);
        for (int m = 0; m < wl->M; ++m) {
            fy[m] = (float)y[m];
            fdy[m] = (float)dy[m];
        }

        double best[MAX_VARIANTS];
        unsigned long long bytes[MAX_VARIANTS];
        int status[MAX_VARIANTS];
        for (int v = 0; v < nvariants; ++v) {
            best[v] = 1e30;
            bytes[v] = 0;
            status[v] = -999;
        }

        const int is_ihs = (wl->solver == 0);
        for (int round = 0; round < rounds; ++round) {
            for (int step = 0; step < nvariants; ++step) {
                const int v = (step + round) % nvariants;
                tls_tuning_set_fft_fuse(variants[v].fuse);
                tls_tuning_set_cobravo(variants[v].cobravo);
                tls_tuning_set_defer(variants[v].defer);
                tls_tuning_set_fft_order(variants[v].order);
                for (int rep = 0; rep < reps; ++rep) {
                    const double t0 = now_seconds();
                    int st;
                    if (is_ihs) {
                        st = tlsf_ihs(t, fy, fdy, wl->M, f0, df, wl->N, wl->degree, wl->backend, wl->norm, power);
                    } else {
                        st = tlsf_fastchi2(t, fy, fdy, wl->M, f0, df, wl->N, wl->degree, wl->backend, wl->solver, wl->norm, power, NULL);
                    }
                    const double dt = now_seconds() - t0;
                    if (dt < best[v]) best[v] = dt;
                    status[v] = st;
                }
                bytes[v] = tls_tuning_work_bytes();
            }
        }

        /* One more run per variant: the periodograms must agree up to the FFT
         * roundoff between fused depths (deferral is a pure permutation). */
        float *sums = (float *)test_alloc((size_t)nvariants * sizeof(float));
        for (int v = 0; v < nvariants; ++v) {
            tls_tuning_set_fft_fuse(variants[v].fuse);
            tls_tuning_set_cobravo(variants[v].cobravo);
            tls_tuning_set_defer(variants[v].defer);
            tls_tuning_set_fft_order(variants[v].order);
            if (is_ihs) {
                status[v] = tlsf_ihs(t, fy, fdy, wl->M, f0, df, wl->N, wl->degree, wl->backend, wl->norm, power);
            } else {
                status[v] = tlsf_fastchi2(t, fy, fdy, wl->M, f0, df, wl->N, wl->degree, wl->backend, wl->solver, wl->norm, power, NULL);
            }
            double sum = 0.0, chk = 0.0;
            for (int i = 0; i < wl->N; ++i) {
                if (isnan(power[i])) continue;
                sum += (double)power[i];
                chk += (double)power[i] * (double)(i % 97);
            }
            sums[v] = (float)(sum + 1e-3 * chk);
            if (v == 0) memcpy(ref, power, (size_t)wl->N * sizeof(float));
        }
        float worst_dev = 0.0f;
        for (int v = 1; v < nvariants; ++v) {
            const float dev = (sums[0] != 0.0f) ? fabsf(sums[v] - sums[0]) / fabsf(sums[0]) : 0.0f;
            if (dev > worst_dev) worst_dev = dev;
        }
        free(sums);

        double base_ns = best[nvariants - 1] * 1e9;
        printf("\n%-20s M=%-5d N=%-6d d=%-2d %s\n", wl->label, wl->M, wl->N, wl->degree, is_ihs ? "ihs" : "fastchi2");
        printf("  %-22s %10s %8s %6s %12s\n", "variant", "ns", "vs old", "st", "work bytes");
        for (int v = 0; v < nvariants; ++v) {
            printf("  %-22s %10.1f %8.3f %6d %12llu\n", variants[v].name, best[v] * 1e9, best[v] * 1e9 / base_ns, status[v], bytes[v]);
        }
        printf("  cross-variant deviation: %.3g\n", (double)worst_dev);
        fflush(stdout);

        free(t);
        free(y);
        free(dy);
        free(fy);
        free(fdy);
        free(ref);
        free(power);
    }

    tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);
    tls_tuning_set_cobravo(TLS_TUNING_COBRAVO_DEFAULT);
    tls_tuning_set_defer(TLS_TUNING_DEFER_DEFAULT);
    return 0;
}
