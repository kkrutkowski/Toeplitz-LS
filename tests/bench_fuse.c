/*
 * bench_fuse.c - interleaved benchmark of the fused-pass schedule.
 *
 * Variants are (fusion depth x fusion order): the leftover stages of an odd
 * stage count go either first (TLS_FFT_ORDER_END, the shipped default) or last
 * (TLS_FFT_ORDER_FRONT).  Both vectorized precisions and every power-of-two
 * grid in the supported range are covered.
 *
 * Two things make the comparison trustworthy, both learned the hard way:
 *
 *   - one plan at a time, created and destroyed around each timed block, so
 *     every variant's twiddle buffers are recycled into the same heap block.
 *     Holding one plan per variant at once lets heap placement alone move a
 *     variant by up to 1.8x, which is larger than anything being measured.
 *   - each round shifts the input by one more 64-byte line and nudges the heap
 *     by a different 64-byte multiple before building the plan, so a full sweep
 *     visits all 64 relative 4 KiB offsets between the signal and the twiddle
 *     table.  The reported number is the median over those layouts: the
 *     store/load aliasing penalty depends on that offset and can reach tens of
 *     percent, so a single offset measures the layout, not the kernel.  The
 *     rotation of the variant order keeps nothing pinned to a cold or boosted
 *     clock.
 *
 * Variant 0 is (fuse=1), the un-fused reference the others are checked against.
 * The reported number is ns/element of execute_raw (the butterflies).  The
 * `default` variant and `d3-end` are the same schedule on this build, so the
 * summary prints their ratio as a measurement self-check.
 *
 *   ./bench_fuse [rounds] [reps] [minlog] [maxlog]
 *
 * 256-bit profile (no AVX-512, 16 vector registers):
 *   make -C tests BUILD=build-v3 \
 *        CFLAGS="-std=gnu11 -O3 -ffast-math -march=x86-64-v3 -mtune=native -Wall -Wextra" \
 *        build-v3/bench_fuse && taskset -c 2 ./tests/build-v3/bench_fuse
 *
 * Double-double is not vectorized (INTERNAL_VEC_LEN = 1) and is left out of the
 * timing sweep; check_nanofft covers its correctness.
 */

#include <math.h>
#include <nanofft.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tls_tuning.h>

#define MAX_VARIANTS 16
#define MAX_SIZES 24
#define NLAYOUTS 64 /* every 64-byte offset of a 4 KiB page */

typedef struct {
    int fuse; /* 0 = the per-precision build default */
    int order;
    char name[32];
} variant;

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

static int exact_log2(uint32_t n) {
    int k = 0;
    while ((1u << k) < n) ++k;
    return k;
}

#define DEFINE_BENCH(TAG, PLAN_T_, MAKE, DESTROY, EXEC, TYPE, LANES, DEFDEPTH)                                                                   \
    static void bench_##TAG(const variant *vars, int nvars, const uint32_t *sizes, int nsizes, int rounds, int reps, double tol) {               \
        printf("\n== %s (VEC_LEN=%d) ==\n", #TAG, (int)(LANES));                                                                                 \
        printf("  %-6s", "grid");                                                                                                                \
        for (int v = 0; v < nvars; ++v) printf(" %10s", vars[v].name);                                                                           \
        printf("   best   vs(ref)\n");                                                                                                           \
        double geo_default = 0.0, geo_ref = 0.0, selfcheck = 0.0;                                                                                \
        int ngeo = 0, nself = 0;                                                                                                                 \
        for (int si = 0; si < nsizes; ++si) {                                                                                                    \
            const uint32_t n = sizes[si];                                                                                                        \
            const int logn = exact_log2(n);                                                                                                      \
            const size_t shift = 64 / sizeof(TYPE); /* one cache line, keeps alignment */                                                        \
            const size_t padded = (size_t)n + (size_t)NLAYOUTS * shift;                                                                          \
            TYPE *buf_r = (TYPE *)aligned_bytes(padded * sizeof(TYPE));                                                                          \
            TYPE *buf_i = (TYPE *)aligned_bytes(padded * sizeof(TYPE));                                                                          \
            TYPE *tmpl_r = (TYPE *)aligned_bytes(n * sizeof(TYPE));                                                                              \
            TYPE *tmpl_i = (TYPE *)aligned_bytes(n * sizeof(TYPE));                                                                              \
            TYPE *ref_r = (TYPE *)aligned_bytes(n * sizeof(TYPE));                                                                               \
            TYPE *ref_i = (TYPE *)aligned_bytes(n * sizeof(TYPE));                                                                               \
            const int layouts = (rounds < NLAYOUTS) ? rounds : NLAYOUTS;                                                                         \
            double per_round[NLAYOUTS][MAX_VARIANTS];                                                                                            \
            for (int r = 0; r < NLAYOUTS; ++r)                                                                                                   \
                for (int v = 0; v < nvars; ++v) per_round[r][v] = 1e30;                                                                          \
            uint32_t rng = 2463534242u;                                                                                                          \
            for (uint32_t k = 0; k < n; ++k) {                                                                                                   \
                rng ^= rng << 13;                                                                                                                \
                rng ^= rng >> 17;                                                                                                                \
                rng ^= rng << 5;                                                                                                                 \
                tmpl_r[k] = (TYPE)((double)(rng & 0xFFFFFFu) / 8388608.0 - 1.0);                                                                 \
                rng ^= rng << 13;                                                                                                                \
                rng ^= rng >> 17;                                                                                                                \
                rng ^= rng << 5;                                                                                                                 \
                tmpl_i[k] = (TYPE)((double)(rng & 0xFFFFFFu) / 8388608.0 - 1.0);                                                                 \
            }                                                                                                                                    \
            for (int round = 0; round < rounds; ++round) {                                                                                       \
                TYPE *r = buf_r + (size_t)(round % NLAYOUTS) * shift;                                                                            \
                TYPE *i = buf_i + (size_t)(round % NLAYOUTS) * shift;                                                                            \
                const size_t pad_bytes = (size_t)(((round * 17) % NLAYOUTS) * 64);                                                               \
                for (int step = 0; step < nvars; ++step) {                                                                                       \
                    const int v = (step + round) % nvars;                                                                                        \
                    tls_tuning_set_fft_fuse(vars[v].fuse);                                                                                       \
                    tls_tuning_set_fft_order(vars[v].order);                                                                                     \
                    /* Nudge the heap before the plan so the twiddle page offset changes                                                         \
                     * with the round too; the same pad is used by every variant of a                                                            \
                     * round, so the comparison stays fair. */                                                                                   \
                    void *pad = pad_bytes ? aligned_bytes(pad_bytes) : NULL;                                                                     \
                    PLAN_T_ *plan = MAKE(n);                                                                                                     \
                    for (int rep = 0; rep < reps; ++rep) {                                                                                       \
                        memcpy(r, tmpl_r, n * sizeof(TYPE));                                                                                     \
                        memcpy(i, tmpl_i, n * sizeof(TYPE));                                                                                     \
                        const double t0 = now_seconds();                                                                                         \
                        EXEC(plan, r, i);                                                                                                        \
                        const double dt = now_seconds() - t0;                                                                                    \
                        if (dt < per_round[round % NLAYOUTS][v]) per_round[round % NLAYOUTS][v] = dt;                                            \
                    }                                                                                                                            \
                    DESTROY(plan);                                                                                                               \
                    free(pad);                                                                                                                   \
                }                                                                                                                                \
            }                                                                                                                                    \
            /* median over the layouts: typical behaviour, not the luckiest page */                                                              \
            double best[MAX_VARIANTS];                                                                                                           \
            for (int v = 0; v < nvars; ++v) {                                                                                                    \
                double col[NLAYOUTS];                                                                                                            \
                for (int r = 0; r < layouts; ++r) col[r] = per_round[r][v];                                                                      \
                for (int a = 1; a < layouts; ++a) {                                                                                              \
                    const double key = col[a];                                                                                                   \
                    int b = a - 1;                                                                                                               \
                    while (b >= 0 && col[b] > key) {                                                                                             \
                        col[b + 1] = col[b];                                                                                                     \
                        --b;                                                                                                                     \
                    }                                                                                                                            \
                    col[b + 1] = key;                                                                                                            \
                }                                                                                                                                \
                best[v] = col[layouts / 2];                                                                                                      \
            }                                                                                                                                    \
            /* every variant must reproduce the un-fused reference */                                                                            \
            double err = 0.0, scale = 0.0;                                                                                                       \
            for (int v = 0; v < nvars; ++v) {                                                                                                    \
                TYPE *r = buf_r, *i = buf_i;                                                                                                     \
                tls_tuning_set_fft_fuse(vars[v].fuse);                                                                                           \
                tls_tuning_set_fft_order(vars[v].order);                                                                                         \
                PLAN_T_ *plan = MAKE(n);                                                                                                         \
                memcpy(r, tmpl_r, n * sizeof(TYPE));                                                                                             \
                memcpy(i, tmpl_i, n * sizeof(TYPE));                                                                                             \
                EXEC(plan, r, i);                                                                                                                \
                DESTROY(plan);                                                                                                                   \
                if (v == 0) {                                                                                                                    \
                    memcpy(ref_r, r, n * sizeof(TYPE));                                                                                          \
                    memcpy(ref_i, i, n * sizeof(TYPE));                                                                                          \
                    continue;                                                                                                                    \
                }                                                                                                                                \
                for (uint32_t k = 0; k < n; ++k) {                                                                                               \
                    const double dr = (double)r[k] - (double)ref_r[k];                                                                           \
                    const double di = (double)i[k] - (double)ref_i[k];                                                                           \
                    const double e = dr * dr + di * di;                                                                                          \
                    if (e > err) err = e;                                                                                                        \
                }                                                                                                                                \
            }                                                                                                                                    \
            /* same metric as the correctness checks: the largest per-bin deviation                                                              \
             * relative to the largest bin */                                                                                                    \
            for (uint32_t k = 0; k < n; ++k) {                                                                                                   \
                const double m = (double)ref_r[k] * (double)ref_r[k] + (double)ref_i[k] * (double)ref_i[k];                                      \
                if (m > scale) scale = m;                                                                                                        \
            }                                                                                                                                    \
            const double worst = (scale > 0.0) ? sqrt(err / scale) : 0.0;                                                                        \
            int best_v = 0;                                                                                                                      \
            for (int v = 1; v < nvars; ++v)                                                                                                      \
                if (best[v] < best[best_v]) best_v = v;                                                                                          \
            printf("  2^%-3d", logn);                                                                                                            \
            for (int v = 0; v < nvars; ++v) printf(" %10.3f", best[v] * 1e9 / (double)n);                                                        \
            printf("  %5s %7.3f %s (err %.1e)\n", vars[best_v].name, best[best_v] / best[0], worst > tol ? "MISMATCH" : "ok", worst);            \
            int iauto = -1, idefault = -1;                                                                                                       \
            for (int v = 0; v < nvars; ++v) {                                                                                                    \
                if (vars[v].fuse == 0 && iauto < 0) iauto = v;                                                                                   \
                if (vars[v].fuse == (DEFDEPTH) && vars[v].order == TLS_FFT_ORDER_END && idefault < 0) idefault = v;                              \
            }                                                                                                                                    \
            if (iauto >= 0 && idefault >= 0 && n >= 1024 && n <= 4096) {                                                                         \
                selfcheck += log(best[iauto] / best[idefault]);                                                                                  \
                ++nself;                                                                                                                         \
            }                                                                                                                                    \
            if (iauto >= 0 && n >= 1024 && n <= 4096) {                                                                                          \
                int iref = -1;                                                                                                                   \
                for (int v = 0; v < nvars; ++v)                                                                                                  \
                    if (vars[v].fuse == 3 && vars[v].order == TLS_FFT_ORDER_FRONT) iref = v;                                                     \
                if (iref >= 0) {                                                                                                                 \
                    geo_default += log(best[iauto] / best[iref]);                                                                                \
                    geo_ref += log(best[0] / best[iref]);                                                                                        \
                    ++ngeo;                                                                                                                      \
                }                                                                                                                                \
            }                                                                                                                                    \
            free(buf_r);                                                                                                                         \
            free(buf_i);                                                                                                                         \
            free(tmpl_r);                                                                                                                        \
            free(tmpl_i);                                                                                                                        \
            free(ref_r);                                                                                                                         \
            free(ref_i);                                                                                                                         \
        }                                                                                                                                        \
        if (ngeo) printf("  geomean 2^10..2^12 vs fuse=3 front: default %.3fx, un-fused %.3fx\n", exp(geo_default / ngeo), exp(geo_ref / ngeo)); \
        if (nself)                                                                                                                               \
            printf("  self-check: default vs the same schedule spelled out = %.3fx (should be 1.00; a couple of percent is run noise)\n",        \
                   exp(selfcheck / nself));                                                                                                      \
    }

DEFINE_BENCH(float, nanofftf_plan, nanofftf_make_plan, nanofftf_destroy_plan, nanofftf_execute_raw, float, VECF_LEN, NANOFFT_FFT_FUSE_FLOAT)
DEFINE_BENCH(double, nanofft_plan, nanofft_make_plan, nanofft_destroy_plan, nanofft_execute_raw, double, VEC_LEN, NANOFFT_FFT_FUSE_DOUBLE)

/* =========================================================================
 * --tune: per-grid fusion-depth table
 *
 * Same measurement discipline as the benchmark above (one plan at a time, a
 * fresh heap nudge per layout, every layout visited, median over layouts) but
 * with a build-friendly budget: it sweeps the depths that are compiled in,
 * picks the median-fastest one per grid, and writes the header that
 * nanofft.c includes when NANOFFT_DEPTH_TABLE is defined.
 *
 * The table is host- and variant-specific, exactly like the generated
 * scaling.h; the release build regenerates it per variant (or copies it from
 * the runnable variant with the same vector width when cross-compiling).
 * ========================================================================= */
#define DEFINE_TUNE(TAG, PLAN_T_, MAKE, DESTROY, EXEC, TYPE, MAXDEPTH)                                       \
    static void tune_##TAG(unsigned char *table, const uint32_t *sizes, int nsizes, int layouts, int reps) { \
        printf("  %-6s %-6s", "grid", "ns/e");                                                               \
        for (int d = 1; d <= (MAXDEPTH); ++d) printf(" %9s", "depth");                                       \
        printf("   chosen\n");                                                                               \
        const size_t shift = 64 / sizeof(TYPE);                                                              \
        for (int si = 0; si < nsizes; ++si) {                                                                \
            const uint32_t n = sizes[si];                                                                    \
            const int logn = exact_log2(n);                                                                  \
            const size_t padded = (size_t)n + (size_t)NLAYOUTS * shift;                                      \
            TYPE *buf_r = (TYPE *)aligned_bytes(padded * sizeof(TYPE));                                      \
            TYPE *buf_i = (TYPE *)aligned_bytes(padded * sizeof(TYPE));                                      \
            TYPE *tmpl_r = (TYPE *)aligned_bytes(n * sizeof(TYPE));                                          \
            TYPE *tmpl_i = (TYPE *)aligned_bytes(n * sizeof(TYPE));                                          \
            uint32_t rng = 2463534242u;                                                                      \
            for (uint32_t k = 0; k < n; ++k) {                                                               \
                rng ^= rng << 13;                                                                            \
                rng ^= rng >> 17;                                                                            \
                rng ^= rng << 5;                                                                             \
                tmpl_r[k] = (TYPE)((double)(rng & 0xFFFFFFu) / 8388608.0 - 1.0);                             \
                rng ^= rng << 13;                                                                            \
                rng ^= rng >> 17;                                                                            \
                rng ^= rng << 5;                                                                             \
                tmpl_i[k] = (TYPE)((double)(rng & 0xFFFFFFu) / 8388608.0 - 1.0);                             \
            }                                                                                                \
            double best[NANOFFT_FFT_FUSE_MAX + 1];                                                           \
            for (int d = 1; d <= (MAXDEPTH); ++d) {                                                          \
                double per[NLAYOUTS];                                                                        \
                for (int r = 0; r < layouts; ++r) per[r] = 1e30;                                             \
                for (int r = 0; r < layouts; ++r) {                                                          \
                    TYPE *rp = buf_r + (size_t)r * shift;                                                    \
                    TYPE *ip = buf_i + (size_t)r * shift;                                                    \
                    const size_t pad_bytes = (size_t)(((r * 17) % NLAYOUTS) * 64);                           \
                    void *pad = pad_bytes ? aligned_bytes(pad_bytes) : NULL;                                 \
                    tls_tuning_set_fft_fuse(d);                                                              \
                    tls_tuning_set_fft_order(TLS_FFT_ORDER_END);                                             \
                    PLAN_T_ *plan = MAKE(n);                                                                 \
                    for (int rep = 0; rep < reps; ++rep) {                                                   \
                        memcpy(rp, tmpl_r, n * sizeof(TYPE));                                                \
                        memcpy(ip, tmpl_i, n * sizeof(TYPE));                                                \
                        const double t0 = now_seconds();                                                     \
                        EXEC(plan, rp, ip);                                                                  \
                        const double dt = now_seconds() - t0;                                                \
                        if (dt < per[r]) per[r] = dt;                                                        \
                    }                                                                                        \
                    DESTROY(plan);                                                                           \
                    free(pad);                                                                               \
                }                                                                                            \
                for (int a = 1; a < layouts; ++a) {                                                          \
                    const double key = per[a];                                                               \
                    int b = a - 1;                                                                           \
                    while (b >= 0 && per[b] > key) {                                                         \
                        per[b + 1] = per[b];                                                                 \
                        --b;                                                                                 \
                    }                                                                                        \
                    per[b + 1] = key;                                                                        \
                }                                                                                            \
                best[d] = per[layouts / 2];                                                                  \
            }                                                                                                \
            int winner = 1;                                                                                  \
            for (int d = 2; d <= (MAXDEPTH); ++d)                                                            \
                if (best[d] < best[winner]) winner = d;                                                      \
            /* Prefer the shallowest depth within 2% of the fastest: the measured */                         \
            /* differences around the optimum are at the noise floor, and a        */                        \
            /* shallower pass never costs registers.                              */                         \
            for (int d = 1; d < winner; ++d)                                                                 \
                if (best[d] <= best[winner] * 1.02) {                                                        \
                    winner = d;                                                                              \
                    break;                                                                                   \
                }                                                                                            \
            table[logn] = (unsigned char)winner;                                                             \
            printf("  2^%-3d %6.3f", logn, best[1] * 1e9 / (double)n);                                       \
            for (int d = 1; d <= (MAXDEPTH); ++d) printf(" %9.3f", best[d] * 1e9 / (double)n);               \
            printf("   d%d\n", winner);                                                                      \
            free(buf_r);                                                                                     \
            free(buf_i);                                                                                     \
            free(tmpl_r);                                                                                    \
            free(tmpl_i);                                                                                    \
        }                                                                                                    \
    }

DEFINE_TUNE(float, nanofftf_plan, nanofftf_make_plan, nanofftf_destroy_plan, nanofftf_execute_raw, float, NANOFFT_FFT_FUSE_FLOAT)
DEFINE_TUNE(double, nanofft_plan, nanofft_make_plan, nanofft_destroy_plan, nanofft_execute_raw, double, NANOFFT_FFT_FUSE_DOUBLE)

static void emit_depth_header(const char *path, const unsigned char *tf, const unsigned char *td, const char *note) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fprintf(f,
            "/* Generated by tests/bench_fuse --tune -- do not edit.\n"
            " *\n"
            " * Per-grid fusion depth: entry [log2 N] is the depth whose median\n"
            " * execute_raw time was lowest for that grid, 0 meaning \"no opinion, use\n"
            " * the per-precision default\".  This is a build artifact like scaling.h:\n"
            " * it is measured per microarchitecture variant on the build host.\n"
            " *\n"
            " * source: %s\n"
            " *\n"
            " * Regenerate with `make tune` (native) or as part of `make wheels`.\n"
            " */\n"
            "#ifndef NANOFFT_DEPTH_TABLE_H\n"
            "#define NANOFFT_DEPTH_TABLE_H\n\n",
            note);
    const char *names[3] = {"nanofft_depth_float", "nanofft_depth_double", "nanofft_depth_dd"};
    const unsigned char *tabs[3] = {tf, td, NULL};
    for (int t = 0; t < 2; ++t) {
        fprintf(f, "static const unsigned char %s[32] = {", names[t]);
        for (int i = 0; i < 32; ++i) fprintf(f, "%s%d", (i % 16) ? ", " : (i ? ",\n    " : ""), (int)tabs[t][i]);
        fprintf(f, "};\n");
    }
    fprintf(f, "/* Double-double is scalar (one lane) and keeps its 2-stage default. */\n");
    fprintf(f, "static const unsigned char %s[32] = {", names[2]);
    for (int i = 0; i < 32; ++i) fprintf(f, "%s0", (i % 16) ? ", " : (i ? ",\n    " : ""));
    fprintf(f, "};\n\n#endif /* NANOFFT_DEPTH_TABLE_H */\n");
    fclose(f);
    printf("[emit] wrote %s\n", path);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--tune") == 0) {
        const char *emit = NULL;
        const char *note = "unrecorded build";
        int layouts = 16, reps = 8, min_log = 0, max_log = 18;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--emit") == 0 && i + 1 < argc)
                emit = argv[++i];
            else if (strcmp(argv[i], "--note") == 0 && i + 1 < argc)
                note = argv[++i];
            else if (strcmp(argv[i], "--layouts") == 0 && i + 1 < argc)
                layouts = atoi(argv[++i]);
            else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
                reps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--minlog") == 0 && i + 1 < argc)
                min_log = atoi(argv[++i]);
            else if (strcmp(argv[i], "--maxlog") == 0 && i + 1 < argc)
                max_log = atoi(argv[++i]);
            else {
                fprintf(stderr, "unknown option %s\n", argv[i]);
                return 2;
            }
        }
        if (layouts < 1) layouts = 1;
        if (layouts > NLAYOUTS) layouts = NLAYOUTS;
        if (reps < 1) reps = 1;

        const int floor_log = exact_log2(2u * ((VECF_LEN > VEC_LEN) ? VECF_LEN : VEC_LEN));
        if (min_log < floor_log) min_log = floor_log;
        if (max_log > 20) max_log = 20;
        if (max_log < min_log) max_log = min_log;

        uint32_t sizes[MAX_SIZES];
        int nsizes = 0;
        for (int k = min_log; k <= max_log && nsizes < MAX_SIZES; ++k) sizes[nsizes++] = 1u << k;

        static unsigned char table_float[32];
        static unsigned char table_double[32];
        printf("tune: grids 2^%d..2^%d, %d layouts, %d reps, depths 1..%d/%d\n", min_log, max_log, layouts, reps, NANOFFT_FFT_FUSE_FLOAT,
               NANOFFT_FFT_FUSE_DOUBLE);
        printf("== double ==\n");
        tune_double(table_double, sizes, nsizes, layouts, reps);
        printf("== float ==\n");
        tune_float(table_float, sizes, nsizes, layouts, reps);
        if (emit) emit_depth_header(emit, table_float, table_double, note);
        tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);
        tls_tuning_set_fft_order(TLS_FFT_ORDER_END);
        return 0;
    }
    /* one round per layout by default: 64 rounds x 8 reps per variant */
    const int rounds = (argc > 1) ? atoi(argv[1]) : 64;
    const int reps = (argc > 2) ? atoi(argv[2]) : 8;
    int min_log = (argc > 3) ? atoi(argv[3]) : 0;
    int max_log = (argc > 4) ? atoi(argv[4]) : 18;

    /* The vectorized tail loads two vectors per step, so 2 * VEC_LEN is the
     * smallest grid make_plan accepts. */
    const int floor_log = exact_log2(2u * ((VECF_LEN > VEC_LEN) ? VECF_LEN : VEC_LEN));
    if (min_log < floor_log) min_log = floor_log;
    if (max_log > 20) max_log = 20;

    variant vars[MAX_VARIANTS];
    int nvars = 0;
    variant v0;
    v0.fuse = 1;
    v0.order = TLS_FFT_ORDER_FRONT;
    snprintf(v0.name, sizeof(v0.name), "fuse1");
    vars[nvars++] = v0;
    for (int fuse = 2; fuse <= NANOFFT_FFT_FUSE_MAX; ++fuse) {
        variant ve, vf;
        ve.fuse = fuse;
        ve.order = TLS_FFT_ORDER_END;
        snprintf(ve.name, sizeof(ve.name), "d%d-end", fuse);
        vars[nvars++] = ve;
        vf.fuse = fuse;
        vf.order = TLS_FFT_ORDER_FRONT;
        snprintf(vf.name, sizeof(vf.name), "d%d-front", fuse);
        vars[nvars++] = vf;
    }
    variant va;
    va.fuse = 0;
    va.order = TLS_FFT_ORDER_END;
    snprintf(va.name, sizeof(va.name), "default");
    vars[nvars++] = va;

    uint32_t sizes[MAX_SIZES];
    int nsizes = 0;
    for (int k = min_log; k <= max_log && nsizes < MAX_SIZES; ++k) sizes[nsizes++] = 1u << k;

    printf("bench_fuse: %d rounds, %d reps, grids 2^%d..2^%d, %d variants\n", rounds, reps, min_log, max_log, nvars);
    printf("build defaults: float depth %d, double %d, double-double %d (NANOFFT_FFT_FUSE_MAX = %d)\n", NANOFFT_FFT_FUSE_FLOAT, NANOFFT_FFT_FUSE_DOUBLE,
           NANOFFT_FFT_FUSE_DD, NANOFFT_FFT_FUSE_MAX);

    bench_double(vars, nvars, sizes, nsizes, rounds, reps, 1e-10);
    bench_float(vars, nvars, sizes, nsizes, rounds, reps, 2e-5);

    tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);
    tls_tuning_set_fft_order(TLS_FFT_ORDER_END);
    return 0;
}
