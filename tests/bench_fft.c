/*
 * bench_fft.c - interleaved microbenchmark of the nanofft kernels.
 *
 * Every round runs every variant once, rotating the order so no variant is
 * always measured on a cold/boosted clock, and reports the best of R rounds
 * (each round itself the best of REPS internal repetitions).
 *
 * Two independent tables:
 *   1. butterflies only (execute_raw): fus(1..NANOFFT_FFT_FUSE_MAX)
 *   2. full transform (execute): the same depths with the vectorized COBRAVO
 *      permutation, plus the scalar in-tile permutation for its speedup.
 *
 *   ./bench_fft [rounds] [reps] [min_log2] [max_log2]
 */

#include <math.h>
#include <nanofft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tls_tuning.h>

#define MAX_SIZES 12
#define MAX_VARIANTS 8

typedef struct {
    int fuse;
    int cobravo;
} variant_t;

static const uint32_t bench_sizes[MAX_SIZES] = {1u << 8,  1u << 9,  1u << 10, 1u << 11, 1u << 12, 1u << 13,
                                                1u << 14, 1u << 15, 1u << 16, 1u << 17, 1u << 18, 1u << 19};

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *aligned_bytes(size_t bytes) {
    const size_t padded = (bytes + 63u) & ~(size_t)63u;
    void *ptr = aligned_alloc(64, padded);
    if (!ptr) {
        fprintf(stderr, "allocation failure\n");
        exit(1);
    }
    memset(ptr, 0, padded);
    return ptr;
}

static double max_rel_error(const float *ar, const float *ai, const float *br, const float *bi, uint32_t n) {
    double scale = 0.0, err = 0.0;
    for (uint32_t k = 0; k < n; ++k) {
        const double m = sqrt((double)ar[k] * ar[k] + (double)ai[k] * ai[k]);
        if (m > scale) scale = m;
        const double dr = (double)ar[k] - br[k], di = (double)ai[k] - bi[k];
        const double e = sqrt(dr * dr + di * di);
        if (e > err) err = e;
    }
    return scale > 0.0 ? err / scale : err;
}

int main(int argc, char **argv) {
    const int rounds = (argc > 1) ? atoi(argv[1]) : 5;
    const int reps = (argc > 2) ? atoi(argv[2]) : 5;
    const int fuse_max = NANOFFT_FFT_FUSE_MAX;
    const int min_log = (argc > 3) ? atoi(argv[3]) : 8;
    const int max_log = (argc > 4) ? atoi(argv[4]) : 19;

    variant_t variants[MAX_VARIANTS];
    int nvariants = 0;
    for (int fuse = 1; fuse <= fuse_max; ++fuse) {
        variants[nvariants].fuse = fuse;
        variants[nvariants].cobravo = 1;
        ++nvariants;
    }
    variants[nvariants].fuse = TLS_TUNING_FFT_FUSE_DEFAULT;
    variants[nvariants].cobravo = 0;
    ++nvariants;

    uint32_t sizes[MAX_SIZES];
    int nsizes = 0;
    for (int s = 0; s < MAX_SIZES; ++s) {
        const int log2_n = (int)log2((double)bench_sizes[s]);
        if (log2_n < min_log || log2_n > max_log) continue;
        sizes[nsizes++] = bench_sizes[s];
    }
    if (nsizes == 0) {
        fprintf(stderr, "no sizes selected\n");
        return 1;
    }

    printf("bench_fft: %d rounds, %d reps, sizes 2^%d..2^%d\n", rounds, reps, (int)log2((double)sizes[0]), (int)log2((double)sizes[nsizes - 1]));

    for (int si = 0; si < nsizes; ++si) {
        const uint32_t n = sizes[si];
        const int logn = (int)(log2((double)n) + 0.5);

        nanofftf_plan *plans[MAX_VARIANTS];
        for (int v = 0; v < nvariants; ++v) {
            tls_tuning_set_fft_fuse(variants[v].fuse);
            tls_tuning_set_cobravo(variants[v].cobravo);
            plans[v] = nanofftf_make_plan(n);
        }
        tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);
        tls_tuning_set_cobravo(TLS_TUNING_COBRAVO_DEFAULT);

        float *tmpl_r = (float *)aligned_bytes(n * sizeof(float));
        float *tmpl_i = (float *)aligned_bytes(n * sizeof(float));
        float *work_r[MAX_VARIANTS];
        float *work_i[MAX_VARIANTS];
        double best_raw[MAX_VARIANTS];
        double best_full[MAX_VARIANTS];
        for (int v = 0; v < nvariants; ++v) {
            work_r[v] = (float *)aligned_bytes(n * sizeof(float));
            work_i[v] = (float *)aligned_bytes(n * sizeof(float));
            best_raw[v] = 1e9;
            best_full[v] = 1e9;
        }

        uint32_t rng = 2463534242u;
        for (uint32_t k = 0; k < n; ++k) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            tmpl_r[k] = (float)((double)(rng & 0xFFFFFFu) / 8388608.0 - 1.0);
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            tmpl_i[k] = (float)((double)(rng & 0xFFFFFFu) / 8388608.0 - 1.0);
        }

        for (int round = 0; round < rounds; ++round) {
            for (int step = 0; step < nvariants; ++step) {
                const int v = (step + round) % nvariants;
                double t_raw = 1e9, t_full = 1e9;
                for (int rep = 0; rep < reps; ++rep) {
                    memcpy(work_r[v], tmpl_r, n * sizeof(float));
                    memcpy(work_i[v], tmpl_i, n * sizeof(float));
                    double t0 = now_seconds();
                    nanofftf_execute_raw(plans[v], work_r[v], work_i[v]);
                    double t1 = now_seconds();
                    nanofftf_permute(plans[v], work_r[v], work_i[v]);
                    double t2 = now_seconds();
                    if (t1 - t0 < t_raw) t_raw = t1 - t0;
                    if (t2 - t1 < t_full) t_full = t2 - t1;
                }
                if (t_raw < best_raw[v]) best_raw[v] = t_raw;
                if (t_full < best_full[v]) best_full[v] = t_full;
            }
        }

        /* correctness: every variant must agree with the first one */
        double worst = 0.0;
        for (int v = 1; v < nvariants; ++v) {
            const double rel = max_rel_error(work_r[0], work_i[0], work_r[v], work_i[v], n);
            if (rel > worst) worst = rel;
        }

        printf("\n2^%-2d  n=%-7u  max relative disagreement between variants: %.3g\n", logn, n, worst);
        printf("  %-22s %10s %10s %10s %10s %8s\n", "variant", "butterfly", "permute", "full ns", "ns/elem", "vs best");
        double best_full_ns = 1e9;
        for (int v = 0; v < nvariants; ++v) {
            const double full_ns = (best_raw[v] + best_full[v]) * 1e9;
            if (full_ns < best_full_ns) best_full_ns = full_ns;
        }
        for (int v = 0; v < nvariants; ++v) {
            char name[32];
            const char *fuse_name = variants[v].fuse == TLS_TUNING_FFT_FUSE_AUTO ? "auto" : "";
            if (variants[v].fuse == TLS_TUNING_FFT_FUSE_AUTO)
                snprintf(name, sizeof(name), "fuse=%s cobravo=%s", fuse_name, variants[v].cobravo ? "vec" : "scalar");
            else
                snprintf(name, sizeof(name), "fuse=%d cobravo=%s", variants[v].fuse, variants[v].cobravo ? "vec" : "scalar");
            const double full_ns = (best_raw[v] + best_full[v]) * 1e9;
            printf("  %-22s %10.1f %10.1f %10.1f %10.3f %8.3f\n", name, best_raw[v] * 1e9, best_full[v] * 1e9, full_ns, full_ns / (double)n,
                   full_ns / best_full_ns);
        }

        for (int v = 0; v < nvariants; ++v) {
            nanofftf_destroy_plan(plans[v]);
            free(work_r[v]);
            free(work_i[v]);
        }
        free(tmpl_r);
        free(tmpl_i);
    }
    return 0;
}
