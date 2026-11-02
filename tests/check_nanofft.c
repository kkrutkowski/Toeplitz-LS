/*
 * check_nanofft.c - self-checks for the FFT kernels and the bit-reversal
 * permutation.  Covers what the high-level periodogram tests cannot isolate:
 *   - the permutation matches reverse_bits for every supported size,
 *   - execute_raw + permute == execute, and the 1-D permutation agrees with
 *     the real half of the 2-D one,
 *   - the vectorized (COBRAVO) and scalar tile paths agree bit for bit,
 *   - every fused pass depth (1..4) reproduces a double-precision naive DFT,
 *   - the fused depths agree with each other to float roundoff.
 */

#include <math.h>
#include <nanofft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tls_tuning.h>

#ifndef NANOFFT_FFT_FUSE_MAX
#    define NANOFFT_FFT_FUSE_MAX 4
#endif

static int failures = 0;

#define CHECK(cond, ...)                                                                                                                            \
    do {                                                                                                                                            \
        if (!(cond)) {                                                                                                                              \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                                                             \
            printf(__VA_ARGS__);                                                                                                                    \
            printf("\n");                                                                                                                           \
            ++failures;                                                                                                                             \
        }                                                                                                                                           \
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
                const double er = (double)or[k] - dr[k], ei = (double)oi[k] - di[k];
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
        tls_tuning_set_fft_fuse(fuse);
        nanofft_plan *plan = nanofft_make_plan(n);
        memcpy(or, r, n * sizeof(double));
        memcpy(oi, i, n * sizeof(double));
        nanofft_execute(plan, or, oi);
        double err = 0.0, scale = 0.0;
        for (uint32_t k = 0; k < n; ++k) {
            const double mag = sqrt(dr[k] * dr[k] + di[k] * di[k]);
            if (mag > scale) scale = mag;
            const double er = or[k] - dr[k], ei = oi[k] - di[k];
            const double e = sqrt(er * er + ei * ei);
            if (e > err) err = e;
        }
        const double rel = scale > 0.0 ? err / scale : err;
        CHECK(rel < 1e-4, "double FFT fuse=%d relative error %.3g", fuse, rel);
        nanofft_destroy_plan(plan);
    }
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
        tls_tuning_set_fft_fuse(fuse);
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
        CHECK(rel < 1e-12, "double-double FFT fuse=%d relative error %.3g", fuse, rel);
        nanofftdd_destroy_plan(plan);
    }
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

int main(void) {
    printf("check_nanofft: permutation\n");
    check_permutation();
    printf("check_nanofft: float FFT vs naive DFT\n");
    check_fft_float();
    printf("check_nanofft: fused depth agreement\n");
    check_fft_fuse_agreement();
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
