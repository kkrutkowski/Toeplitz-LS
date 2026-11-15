/*
 * check_nanofft.c - self-checks for the FFT kernels and the bit-reversal
 * permutation.  Covers what the high-level periodogram tests cannot isolate:
 *   - the permutation matches reverse_bits for every supported size,
 *   - execute_raw + permute == execute, and the 1-D permutation agrees with
 *     the real half of the 2-D one,
 *   - the vectorized (COBRAVO) and scalar tile paths agree bit for bit,
 *   - every fused pass depth (1..4) reproduces a double-precision naive DFT,
 *   - the fused depths agree with each other to float roundoff,
 *   - both fusion orders reproduce the un-fused transform on every supported
 *     grid, and are bit-identical when the stage count divides by the depth,
 *   - make_plan() rejects grids below 2 * VEC_LEN instead of reading past the
 *     signal, and the smallest accepted grid still matches the naive DFT.
 */

#include <math.h>
#include <nanofft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <tls_tuning.h>
#include <unistd.h>

#ifndef NANOFFT_FFT_FUSE_MAX
#    define NANOFFT_FFT_FUSE_MAX 4
#endif

static int failures = 0;

#define CHECK(cond, ...)                                \
    do {                                                \
        if (!(cond)) {                                  \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                        \
            printf("\n");                               \
            ++failures;                                 \
        }                                               \
    } while (0)

/* nanofft expects 64-byte aligned signals (the vector loads are aligned). */
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

static uint32_t rng_state = 123456789u;

static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((double)(rng_state >> 8) / 8388608.0 - 1.0); /* [-1, 1) */
}

static uint32_t reverse_bits_ref(uint32_t value, uint32_t bits) {
    uint32_t out = 0;
    for (uint32_t b = 0; b < bits; ++b) out |= ((value >> b) & 1u) << (bits - 1u - b);
    return out;
}

static void naive_dft(const float *r, const float *i, double *out_r, double *out_i, int n) {
    const double two_pi = 6.283185307179586476925286766559;
    for (int k = 0; k < n; ++k) {
        double acc_r = 0.0, acc_i = 0.0;
        for (int m = 0; m < n; ++m) {
            const double th = two_pi * (double)k * (double)m / (double)n;
            const double c = cos(th), s = sin(th);
            /* nanofft's DIF uses the NuFFT sign convention: e^{+2 pi i k m / N}. */
            acc_r += (double)r[m] * c - (double)i[m] * s;
            acc_i += (double)i[m] * c + (double)r[m] * s;
        }
        out_r[k] = acc_r;
        out_i[k] = acc_i;
    }
}

static double max_rel_error(const float *ar, const float *ai, const float *br, const float *bi, int n) {
    double scale = 0.0, err = 0.0;
    for (int k = 0; k < n; ++k) {
        const double arv = ar[k], aiv = ai[k], brv = br[k], biv = bi[k];
        const double mag = sqrt(arv * arv + aiv * aiv);
        if (mag > scale) scale = mag;
        const double dr = arv - brv, di = aiv - biv;
        const double e = sqrt(dr * dr + di * di);
        if (e > err) err = e;
    }
    return scale > 0.0 ? err / scale : err;
}

/* ------------------------------------------------------------------ */

static void check_permutation(void) {
    for (uint32_t n = 64; n <= (1u << 14); n <<= 1) {
        const uint32_t log_n = (uint32_t)(log2((double)n) + 0.5);
        nanofftf_plan *plan_vec = NULL, *plan_scalar = NULL;
        tls_tuning_set_cobravo(1);
        plan_vec = nanofftf_make_plan(n);
        tls_tuning_set_cobravo(0);
        plan_scalar = nanofftf_make_plan(n);
        tls_tuning_set_cobravo(1);

        float *v = (float *)test_alloc(n * sizeof(float));
        float *z = (float *)test_alloc(n * sizeof(float));
        float *w = (float *)test_alloc(n * sizeof(float));
        float *v2 = (float *)test_alloc(n * sizeof(float));

        for (uint32_t i = 0; i < n; ++i) v[i] = (float)i;
        memcpy(v2, v, n * sizeof(float));
        nanofftf_permute_1d(plan_vec, v);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t want = reverse_bits_ref(i, log_n);
            CHECK(v[i] == (float)want, "permute_1d n=%u i=%u got %g want %u", n, i, (double)v[i], want);
            if (v[i] != (float)want) break;
        }

        /* the scalar tile path must emit the identical permutation */
        nanofftf_permute_1d(plan_scalar, v2);
        CHECK(memcmp(v, v2, n * sizeof(float)) == 0, "scalar/vector permutation mismatch at n=%u", n);

        /* permuting twice is the identity (bit reversal is an involution) */
        nanofftf_permute_1d(plan_vec, v);
        int identity = 1;
        for (uint32_t i = 0; i < n; ++i)
            if (v[i] != (float)i) identity = 0;
        CHECK(identity, "bit reversal is not an involution at n=%u", n);

        /* 1-D permutation == real half of the 2-D permutation */
        for (uint32_t i = 0; i < n; ++i) {
            v[i] = frand();
            z[i] = 0.0f;
            w[i] = v[i];
        }
        nanofftf_permute(plan_vec, v, z);
        nanofftf_permute_1d(plan_vec, w);
        CHECK(memcmp(v, w, n * sizeof(float)) == 0, "1-D/2-D permutation mismatch at n=%u", n);

        /* execute == execute_raw + permute, bit for bit */
        float *r0 = (float *)test_alloc(n * sizeof(float));
        float *i0 = (float *)test_alloc(n * sizeof(float));
        float *r1 = (float *)test_alloc(n * sizeof(float));
        float *i1 = (float *)test_alloc(n * sizeof(float));
        for (uint32_t i = 0; i < n; ++i) {
            r0[i] = frand();
            i0[i] = frand();
            r1[i] = r0[i];
            i1[i] = i0[i];
        }
        nanofftf_execute(plan_vec, r0, i0);
        nanofftf_execute_raw(plan_vec, r1, i1);
        nanofftf_permute(plan_vec, r1, i1);
        CHECK(memcmp(r0, r1, n * sizeof(float)) == 0 && memcmp(i0, i1, n * sizeof(float)) == 0, "execute != raw+permute at n=%u", n);

        free(r0);
        free(i0);
        free(r1);
        free(i1);
        free(v);
        free(z);
        free(w);
        free(v2);
        nanofftf_destroy_plan(plan_vec);
        nanofftf_destroy_plan(plan_scalar);
    }
}

static void check_fft_float(void) {
    for (uint32_t n = 256; n <= 2048; n <<= 1) {
        float *r = (float *)test_alloc(n * sizeof(float));
        float *i = (float *)test_alloc(n * sizeof(float));
        float *or = (float *)test_alloc(n * sizeof(float));
        float *oi = (float *)test_alloc(n * sizeof(float));
        float *ref_r = (float *)test_alloc(n * sizeof(float));
        float *ref_i = (float *)test_alloc(n * sizeof(float));
        double *dr = (double *)test_alloc(n * sizeof(double));
        double *di = (double *)test_alloc(n * sizeof(double));

        for (uint32_t k = 0; k < n; ++k) {
            r[k] = frand();
            i[k] = frand();
            ref_r[k] = r[k];
            ref_i[k] = i[k];
        }
        naive_dft(r, i, dr, di, (int)n);

        double worst = 0.0;
        for (int fuse = 1; fuse <= NANOFFT_FFT_FUSE_MAX; ++fuse) {
            tls_tuning_set_fft_fuse(fuse);
            nanofftf_plan *plan = nanofftf_make_plan(n);
            memcpy(or, r, n * sizeof(float));
            memcpy(oi, i, n * sizeof(float));
            nanofftf_execute(plan, or, oi);

            double err = 0.0, scale = 0.0;
            for (uint32_t k = 0; k < n; ++k) {
                const double mag = sqrt(dr[k] * dr[k] + di[k] * di[k]);
                if (mag > scale) scale = mag;
                const double er = (double) or [k] - dr[k], ei = (double)oi[k] - di[k];
                const double e = sqrt(er * er + ei * ei);
                if (e > err) err = e;
            }
            const double rel = scale > 0.0 ? err / scale : err;
            if (rel > worst) worst = rel;
            CHECK(rel < 5e-5, "float FFT n=%u fuse=%d relative error %.3g", n, fuse, rel);

            nanofftf_destroy_plan(plan);
        }
        printf("  float n=%-6u naive-DFT relative error %.3g\n", n, worst);

        free(r);
        free(i);
        free(or);
        free(oi);
        free(ref_r);
        free(ref_i);
        free(dr);
        free(di);
    }
}

static void check_fft_fuse_agreement(void) {
    for (uint32_t n = 256; n <= (1u << 15); n <<= 1) {
        float *r = (float *)test_alloc(n * sizeof(float));
        float *i = (float *)test_alloc(n * sizeof(float));
        float *base_r = (float *)test_alloc(n * sizeof(float));
        float *base_i = (float *)test_alloc(n * sizeof(float));
        float *or = (float *)test_alloc(n * sizeof(float));
        float *oi = (float *)test_alloc(n * sizeof(float));

        for (uint32_t k = 0; k < n; ++k) {
            r[k] = frand();
            i[k] = frand();
        }

        tls_tuning_set_fft_fuse(1);
        nanofftf_plan *base = nanofftf_make_plan(n);
        memcpy(base_r, r, n * sizeof(float));
        memcpy(base_i, i, n * sizeof(float));
        nanofftf_execute(base, base_r, base_i);
        nanofftf_destroy_plan(base);

        for (int fuse = 2; fuse <= NANOFFT_FFT_FUSE_MAX; ++fuse) {
            tls_tuning_set_fft_fuse(fuse);
            nanofftf_plan *plan = nanofftf_make_plan(n);
            memcpy(or, r, n * sizeof(float));
            memcpy(oi, i, n * sizeof(float));
            nanofftf_execute(plan, or, oi);
            const double rel = max_rel_error(base_r, base_i, or, oi, (int)n);
            CHECK(rel < 2e-5, "fuse=%d disagrees with fuse=1 at n=%u (%.3g)", fuse, n, rel);
            nanofftf_destroy_plan(plan);
        }
        tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);

        free(r);
        free(i);
        free(base_r);
        free(base_i);
        free(or);
        free(oi);
    }
}

static void check_fft_double(void) {
    const uint32_t n = 1024;
    double *r = (double *)test_alloc(n * sizeof(double));
    double *i = (double *)test_alloc(n * sizeof(double));
    double *or = (double *)test_alloc(n * sizeof(double));
    double *oi = (double *)test_alloc(n * sizeof(double));
    float *fr = (float *)test_alloc(n * sizeof(float));
    float *fi = (float *)test_alloc(n * sizeof(float));
    double *dr = (double *)test_alloc(n * sizeof(double));
    double *di = (double *)test_alloc(n * sizeof(double));

    for (uint32_t k = 0; k < n; ++k) {
        r[k] = (double)frand();
        i[k] = (double)frand();
        fr[k] = (float)r[k];
        fi[k] = (float)i[k];
    }
    naive_dft(fr, fi, dr, di, (int)n);

    for (int fuse = 1; fuse <= NANOFFT_FFT_FUSE_MAX; ++fuse) {
        for (int ord = 0; ord < 2; ++ord) {
            tls_tuning_set_fft_fuse(fuse);
            tls_tuning_set_fft_order(ord ? TLS_FFT_ORDER_FRONT : TLS_FFT_ORDER_END);
            nanofft_plan *plan = nanofft_make_plan(n);
            memcpy(or, r, n * sizeof(double));
            memcpy(oi, i, n * sizeof(double));
            nanofft_execute(plan, or, oi);
            double err = 0.0, scale = 0.0;
            for (uint32_t k = 0; k < n; ++k) {
                const double mag = sqrt(dr[k] * dr[k] + di[k] * di[k]);
                if (mag > scale) scale = mag;
                const double er = or [k] - dr[k], ei = oi[k] - di[k];
                const double e = sqrt(er * er + ei * ei);
                if (e > err) err = e;
            }
            const double rel = scale > 0.0 ? err / scale : err;
            CHECK(rel < 1e-4, "double FFT fuse=%d order=%d relative error %.3g", fuse, ord, rel);
            nanofft_destroy_plan(plan);
        }
    }
    tls_tuning_set_fft_order(TLS_FFT_ORDER_END);
    tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);

    free(r);
    free(i);
    free(or);
    free(oi);
    free(fr);
    free(fi);
    free(dr);
    free(di);
}

static void check_dd_permutation(void) {
    const uint32_t n = 1024;
    const uint32_t log_n = 10;
    nanofftdd_plan *plan = nanofftdd_make_plan(n);
    dd_t *v = (dd_t *)test_alloc(n * sizeof(dd_t));
    dd_t *w = (dd_t *)test_alloc(n * sizeof(dd_t));
    for (uint32_t k = 0; k < n; ++k) {
        v[k] = dd_make((double)k, 0.0);
        w[k] = dd_make((double)k, 0.0);
    }
    nanofftdd_permute_1d(plan, v);
    int ok = 1;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t want = reverse_bits_ref(k, log_n);
        if (v[k].hi != (double)want) ok = 0;
    }
    CHECK(ok, "double-double permutation mismatch at n=%u", n);
    (void)w;
    free(v);
    free(w);
    nanofftdd_destroy_plan(plan);
}

static void check_fft_dd(void) {
#ifndef NANOFFT_FFT_FUSE_DD
#    define NANOFFT_FFT_FUSE_DD 3
#endif
    const uint32_t n = 256;
    dd_t *r = (dd_t *)test_alloc(n * sizeof(dd_t));
    dd_t *i = (dd_t *)test_alloc(n * sizeof(dd_t));
    dd_t *or_ = (dd_t *)test_alloc(n * sizeof(dd_t));
    dd_t *oi = (dd_t *)test_alloc(n * sizeof(dd_t));
    float *fr = (float *)test_alloc(n * sizeof(float));
    float *fi = (float *)test_alloc(n * sizeof(float));
    double *dr = (double *)test_alloc(n * sizeof(double));
    double *di = (double *)test_alloc(n * sizeof(double));

    for (uint32_t k = 0; k < n; ++k) {
        const double re = (double)frand(), im = (double)frand();
        r[k] = dd_make(re, 0.0);
        i[k] = dd_make(im, 0.0);
        fr[k] = (float)re;
        fi[k] = (float)im;
    }
    naive_dft(fr, fi, dr, di, (int)n);

    for (int fuse = 1; fuse <= NANOFFT_FFT_FUSE_DD; ++fuse) {
        for (int ord = 0; ord < 2; ++ord) {
            tls_tuning_set_fft_fuse(fuse);
            tls_tuning_set_fft_order(ord ? TLS_FFT_ORDER_FRONT : TLS_FFT_ORDER_END);
            nanofftdd_plan *plan = nanofftdd_make_plan(n);
            memcpy(or_, r, n * sizeof(dd_t));
            memcpy(oi, i, n * sizeof(dd_t));
            nanofftdd_execute(plan, or_, oi);
            double err = 0.0, scale = 0.0;
            for (uint32_t k = 0; k < n; ++k) {
                const double mag = sqrt(dr[k] * dr[k] + di[k] * di[k]);
                if (mag > scale) scale = mag;
                const double er = dd_to_double(or_[k]) - dr[k], ei = dd_to_double(oi[k]) - di[k];
                const double e = sqrt(er * er + ei * ei);
                if (e > err) err = e;
            }
            const double rel = scale > 0.0 ? err / scale : err;
            CHECK(rel < 1e-12, "double-double FFT fuse=%d order=%d relative error %.3g", fuse, ord, rel);
            nanofftdd_destroy_plan(plan);
        }
    }
    tls_tuning_set_fft_order(TLS_FFT_ORDER_END);
    tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);

    free(r);
    free(i);
    free(or_);
    free(oi);
    free(fr);
    free(fi);
    free(dr);
    free(di);
}

/* ------------------------------------------------------------------ */

/* Both fusion orders must reproduce the un-fused transform for every depth on
 * every supported grid.  Where the stage count is a multiple of the depth the
 * two orders describe the same schedule, so they must agree bit for bit. */
static void check_fft_order(void) {
    const uint32_t min_n = 2u * (uint32_t)VECF_LEN;
    const uint32_t log_v = (uint32_t)(log2((double)VECF_LEN) + 0.5);
    const uint32_t cap = 1u << 16;
    float *r = (float *)test_alloc(cap * sizeof(float));
    float *i = (float *)test_alloc(cap * sizeof(float));
    float *ref_r = (float *)test_alloc(cap * sizeof(float));
    float *ref_i = (float *)test_alloc(cap * sizeof(float));
    float *a_r = (float *)test_alloc(cap * sizeof(float));
    float *a_i = (float *)test_alloc(cap * sizeof(float));
    float *b_r = (float *)test_alloc(cap * sizeof(float));
    float *b_i = (float *)test_alloc(cap * sizeof(float));

    for (uint32_t n = min_n; n <= cap; n <<= 1) {
        const uint32_t log_n = (uint32_t)(log2((double)n) + 0.5);
        const uint32_t stages = (log_n > log_v) ? log_n - log_v : 0;
        for (uint32_t k = 0; k < n; ++k) {
            r[k] = frand();
            i[k] = frand();
            ref_r[k] = r[k];
            ref_i[k] = i[k];
        }

        tls_tuning_set_fft_fuse(1);
        nanofftf_plan *ref = nanofftf_make_plan(n);
        nanofftf_execute(ref, ref_r, ref_i);
        nanofftf_destroy_plan(ref);

        for (int fuse = 1; fuse <= NANOFFT_FFT_FUSE_MAX; ++fuse) {
            for (int ord = 0; ord < 2; ++ord) {
                const int order = ord ? TLS_FFT_ORDER_FRONT : TLS_FFT_ORDER_END;
                tls_tuning_set_fft_fuse(fuse);
                tls_tuning_set_fft_order(order);
                nanofftf_plan *plan = nanofftf_make_plan(n);
                memcpy(a_r, r, n * sizeof(float));
                memcpy(a_i, i, n * sizeof(float));
                nanofftf_execute(plan, a_r, a_i);
                nanofftf_destroy_plan(plan);
                const double rel = max_rel_error(ref_r, ref_i, a_r, a_i, (int)n);
                CHECK(rel < 2e-5, "order=%d fuse=%d disagrees with fuse=1 at n=%u (%.3g)", order, fuse, n, rel);
                if (ord == 0) {
                    memcpy(b_r, a_r, n * sizeof(float));
                    memcpy(b_i, a_i, n * sizeof(float));
                }
            }
            /* the plan clamps the request to the depth this build compiled */
            const uint32_t eff = (fuse > NANOFFT_FFT_FUSE_FLOAT) ? (uint32_t)NANOFFT_FFT_FUSE_FLOAT : (uint32_t)fuse;
            if (eff > 1 && (stages % eff) == 0)
                CHECK(memcmp(b_r, a_r, n * sizeof(float)) == 0 && memcmp(b_i, a_i, n * sizeof(float)) == 0,
                      "fuse=%d n=%u: the two orders should describe the same schedule", fuse, n);
        }
    }

    tls_tuning_set_fft_fuse(TLS_TUNING_FFT_FUSE_DEFAULT);
    tls_tuning_set_fft_order(TLS_FFT_ORDER_END);
    free(r);
    free(i);
    free(ref_r);
    free(ref_i);
    free(a_r);
    free(a_i);
    free(b_r);
    free(b_i);
}

/* make_plan() must reject a grid shorter than two vectors (the intra-vector
 * tail loads two per step) instead of reading past the signal.  The expected
 * exit() runs in a child so the checks survive it. */
static int child_rejects(uint32_t n) {
    fflush(stdout);
    const pid_t pid = fork();
    CHECK(pid >= 0, "fork failed");
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(2);
        (void)nanofftf_make_plan(n);
        _exit(0); /* only reached if the unsupported grid was accepted */
    }
    int status = 0;
    if (pid > 0) waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE;
}

static void check_min_grid(void) {
    const uint32_t min_f = 2u * (uint32_t)VECF_LEN;
    const uint32_t min_d = 2u * (uint32_t)VEC_LEN;
    CHECK(child_rejects(min_f / 2), "float make_plan accepted n=%u (< %u)", min_f / 2, min_f);
    CHECK(child_rejects(min_d / 2), "double make_plan accepted n=%u (< %u)", min_d / 2, min_d);

    /* the smallest accepted float grid must still match the naive DFT */
    const uint32_t n = min_f;
    float *r = (float *)test_alloc(n * sizeof(float));
    float *i = (float *)test_alloc(n * sizeof(float));
    float *ref_r = (float *)test_alloc(n * sizeof(float));
    float *ref_i = (float *)test_alloc(n * sizeof(float));
    double *dr = (double *)test_alloc(n * sizeof(double));
    double *di = (double *)test_alloc(n * sizeof(double));
    for (uint32_t k = 0; k < n; ++k) {
        r[k] = frand();
        i[k] = frand();
    }
    naive_dft(r, i, dr, di, (int)n);
    for (int ord = 0; ord < 2; ++ord) {
        tls_tuning_set_fft_order(ord ? TLS_FFT_ORDER_FRONT : TLS_FFT_ORDER_END);
        nanofftf_plan *plan = nanofftf_make_plan(n);
        memcpy(ref_r, r, n * sizeof(float));
        memcpy(ref_i, i, n * sizeof(float));
        nanofftf_execute(plan, ref_r, ref_i);
        double err = 0.0, scale = 0.0;
        for (uint32_t k = 0; k < n; ++k) {
            const double mag = sqrt(dr[k] * dr[k] + di[k] * di[k]);
            if (mag > scale) scale = mag;
            const double er = (double)ref_r[k] - dr[k], ei = (double)ref_i[k] - di[k];
            const double e = sqrt(er * er + ei * ei);
            if (e > err) err = e;
        }
        const double rel = scale > 0.0 ? err / scale : err;
        CHECK(rel < 5e-5, "smallest grid n=%u order=%d relative error %.3g", n, ord, rel);
        nanofftf_destroy_plan(plan);
    }
    tls_tuning_set_fft_order(TLS_FFT_ORDER_END);
    free(r);
    free(i);
    free(ref_r);
    free(ref_i);
    free(dr);
    free(di);
    printf("  smallest grid n=%u accepted, rejected below\n", n);
}

int main(void) {
    printf("check_nanofft: permutation\n");
    check_permutation();
    printf("check_nanofft: float FFT vs naive DFT\n");
    check_fft_float();
    printf("check_nanofft: fused depth agreement\n");
    check_fft_fuse_agreement();
    printf("check_nanofft: fused order and grid range\n");
    check_fft_order();
    check_min_grid();
    printf("check_nanofft: double FFT\n");
    check_fft_double();
    printf("check_nanofft: double-double permutation\n");
    check_dd_permutation();
    printf("check_nanofft: double-double FFT\n");
    check_fft_dd();

    if (failures) {
        printf("check_nanofft: %d failure(s)\n", failures);
        return 1;
    }
    printf("check_nanofft: OK\n");
    return 0;
}
