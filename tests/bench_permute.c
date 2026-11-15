/* bench_permute.c - bit-reversal strategy comparison, both precisions.
 *
 * The FFT butterflies are excluded: this table answers one question, which bit
 * reversal is cheapest for a given transform length, and where the crossovers
 * sit.  Five strategies are interleaved in one process (the order rotates every
 * round) so clock ramp and binary layout cannot favour one of them:
 *
 *   auto      what the shipped build picks for this length
 *   table     scalar table shuffle (reverse_bits per element, branch on j > i)
 *   pairs     precomputed swap-pair walk (branch-free, N/2 - #fixed points)
 *   cobravo   tiled reversal, vectorized gather emission
 *   cobravo-s tiled reversal, scalar in-tile swap
 *
 * The reported numbers are ns per element, best of `reps` inside `rounds`
 * rotations.  Buffer contents must agree between strategies, which is checked
 * from a fresh copy of the input after the timing loop.
 *
 * Usage: ./bench_permute [rounds] [reps] [min_log2] [max_log2]
 */

#include <nanofft.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tls_tuning.h>

#define MAX_STRATEGIES 5

typedef struct {
    const char *name;
    int mode;
    int cobravo;
} strategy;

/* One entry per precision: the plan and buffer types differ, so the benchmark
 * drives them through void pointers and per-precision callbacks. */
typedef struct {
    const char *name;
    size_t elem;
    void *(*make_plan)(uint32_t);
    void (*destroy_plan)(void *);
    void (*permute)(void *, void *, void *);
    double (*max_rel_error)(const void *, const void *, const void *, const void *, uint32_t);
} precision;

static const strategy STRATEGIES[MAX_STRATEGIES] = {
    {"auto", TLS_PERMUTE_AUTO, 1},       {"table", TLS_PERMUTE_TABLE, 1},       {"pairs", TLS_PERMUTE_PAIRS, 1},
    {"cobravo", TLS_PERMUTE_COBRAVO, 1}, {"cobravo-s", TLS_PERMUTE_COBRAVO, 0},
};

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void *aligned_bytes(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) {
        fprintf(stderr, "out of memory\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

static double float_max_rel_error(const void *ar, const void *ai, const void *br, const void *bi, uint32_t n) {
    const float *fr_a = (const float *)ar, *fi_a = (const float *)ai;
    const float *fr_b = (const float *)br, *fi_b = (const float *)bi;
    double worst = 0.0;
    for (uint32_t k = 0; k < n; ++k) {
        const double dr = (double)fr_a[k] - (double)fr_b[k];
        const double di = (double)fi_a[k] - (double)fi_b[k];
        const double scale = 1.0 + (double)fr_a[k] * (double)fr_a[k] + (double)fi_a[k] * (double)fi_a[k];
        const double rel = (dr * dr + di * di) / scale;
        if (rel > worst) worst = rel;
    }
    return worst;
}

static double double_max_rel_error(const void *ar, const void *ai, const void *br, const void *bi, uint32_t n) {
    const double *dr_a = (const double *)ar, *di_a = (const double *)ai;
    const double *dr_b = (const double *)br, *di_b = (const double *)bi;
    double worst = 0.0;
    for (uint32_t k = 0; k < n; ++k) {
        const double dr = dr_a[k] - dr_b[k];
        const double di = di_a[k] - di_b[k];
        const double scale = 1.0 + dr_a[k] * dr_a[k] + di_a[k] * di_a[k];
        const double rel = (dr * dr + di * di) / scale;
        if (rel > worst) worst = rel;
    }
    return worst;
}

static void fill_templates(uint32_t n, void *tmpl_r, void *tmpl_i, size_t elem) {
    uint32_t rng = 2463534242u;
    for (int part = 0; part < 2; ++part) {
        void *dst = part ? tmpl_i : tmpl_r;
        for (uint32_t k = 0; k < n; ++k) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            const double value = (double)(rng & 0xFFFFFFu) / 8388608.0 - 1.0;
            if (elem == sizeof(float))
                ((float *)dst)[k] = (float)value;
            else
                ((double *)dst)[k] = value;
        }
    }
}

static void run_precision(const precision *api, int rounds, int reps, int min_log, int max_log) {
    printf("\n%s: permute-only cost, ns per element\n", api->name);
    printf("%-4s %-8s", "log2", "n");
    for (int v = 0; v < MAX_STRATEGIES; ++v) printf(" %11s", STRATEGIES[v].name);
    printf(" %10s %10s\n", "best", "auto/best");

    for (int logn = min_log; logn <= max_log; ++logn) {
        const uint32_t n = 1u << logn;
        void *plans[MAX_STRATEGIES];
        for (int v = 0; v < MAX_STRATEGIES; ++v) {
            tls_tuning_set_permute(STRATEGIES[v].mode);
            tls_tuning_set_cobravo(STRATEGIES[v].cobravo);
            plans[v] = api->make_plan(n);
        }
        tls_tuning_set_permute(TLS_PERMUTE_AUTO);
        tls_tuning_set_cobravo(TLS_TUNING_COBRAVO_DEFAULT);

        void *tmpl_r = aligned_bytes((size_t)n * api->elem);
        void *tmpl_i = aligned_bytes((size_t)n * api->elem);
        void *work_r[MAX_STRATEGIES];
        void *work_i[MAX_STRATEGIES];
        double best[MAX_STRATEGIES];
        for (int v = 0; v < MAX_STRATEGIES; ++v) {
            work_r[v] = aligned_bytes((size_t)n * api->elem);
            work_i[v] = aligned_bytes((size_t)n * api->elem);
            best[v] = 1e18;
        }
        fill_templates(n, tmpl_r, tmpl_i, api->elem);

        for (int round = 0; round < rounds; ++round) {
            for (int step = 0; step < MAX_STRATEGIES; ++step) {
                const int v = (step + round) % MAX_STRATEGIES;
                double fastest = 1e18;
                for (int rep = 0; rep < reps; ++rep) {
                    memcpy(work_r[v], tmpl_r, (size_t)n * api->elem);
                    memcpy(work_i[v], tmpl_i, (size_t)n * api->elem);
                    const double t0 = now_seconds();
                    api->permute(plans[v], work_r[v], work_i[v]);
                    const double t1 = now_seconds();
                    if (t1 - t0 < fastest) fastest = t1 - t0;
                }
                if (fastest < best[v]) best[v] = fastest;
            }
        }

        /* Every strategy must produce the same permutation: apply one from a
         * fresh copy of the input and compare against the automatic one. */
        for (int v = 0; v < MAX_STRATEGIES; ++v) {
            memcpy(work_r[v], tmpl_r, (size_t)n * api->elem);
            memcpy(work_i[v], tmpl_i, (size_t)n * api->elem);
            api->permute(plans[v], work_r[v], work_i[v]);
        }
        double worst = 0.0;
        int worst_v = 0;
        for (int v = 1; v < MAX_STRATEGIES; ++v) {
            const double err = api->max_rel_error(work_r[0], work_i[0], work_r[v], work_i[v], n);
            if (err > worst) {
                worst = err;
                worst_v = v;
            }
        }

        double best_ns = 1e18;
        for (int v = 0; v < MAX_STRATEGIES; ++v) {
            const double ns = best[v] * 1e9 / (double)n;
            if (ns < best_ns) best_ns = ns;
        }
        printf("%-4d %-8u", logn, n);
        for (int v = 0; v < MAX_STRATEGIES; ++v) printf(" %11.4f", best[v] * 1e9 / (double)n);
        printf(" %10.4f %10.3f", best_ns, (best[0] * 1e9 / (double)n) / best_ns);
        if (worst > 1e-12) printf("  MISMATCH(vs %s) %.3g", STRATEGIES[worst_v].name, worst);
        printf("\n");
        fflush(stdout);

        for (int v = 0; v < MAX_STRATEGIES; ++v) {
            api->destroy_plan(plans[v]);
            free(work_r[v]);
            free(work_i[v]);
        }
        free(tmpl_r);
        free(tmpl_i);
    }
}

int main(int argc, char **argv) {
    const int rounds = (argc > 1) ? atoi(argv[1]) : 7;
    const int reps = (argc > 2) ? atoi(argv[2]) : 9;
    const int min_log = (argc > 3) ? atoi(argv[3]) : 4;
    const int max_log = (argc > 4) ? atoi(argv[4]) : 16;

    const precision float_api = {
        "float",
        sizeof(float),
        (void *(*)(uint32_t))nanofftf_make_plan,
        (void (*)(void *))nanofftf_destroy_plan,
        (void (*)(void *, void *, void *))nanofftf_permute,
        float_max_rel_error,
    };
    const precision double_api = {
        "double",
        sizeof(double),
        (void *(*)(uint32_t))nanofft_make_plan,
        (void (*)(void *))nanofft_destroy_plan,
        (void (*)(void *, void *, void *))nanofft_permute,
        double_max_rel_error,
    };

    run_precision(&float_api, rounds, reps, min_log, max_log);
    run_precision(&double_api, rounds, reps, min_log, max_log);
    return 0;
}
