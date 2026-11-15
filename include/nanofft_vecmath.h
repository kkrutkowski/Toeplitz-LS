#ifndef NANOFFT_VECMATH_H
#define NANOFFT_VECMATH_H

/*
 * Vector sqrt / exp helpers for the float and double precisions.
 *
 * These are compiled with the caller's fast-math settings on purpose:
 * reassociating the Horner chain and contracting it to FMA is wanted, and it
 * costs a few ulp (tests/check_vecmath.c pins the bound).  What fast-math must
 * not break is handled explicitly below: the ln2 split is exact by
 * construction (k * hi is representable and lo is the true remainder), and the
 * infinity / NaN lanes are rebuilt from their bit pattern after the reduction.
 *
 * Why the library carries its own exponential: a lane loop of expf()/exp() is
 * recognised by GCC as a libmvec call (_ZGVdN4v_expf) when compiling against
 * glibc, and libmvec exists in neither musl, macOS libSystem nor the MSVC CRT,
 * so the same sources would link differently per platform.  The kernel is
 * therefore self-contained:
 *
 *     k = rint(x / ln2)                    exponent of the result
 *     r = x - k*ln2_hi - k*ln2_lo          |r| <= ln2/2
 *     p = P(r)                             Remez minimax fit of exp on that range
 *     exp(x) = p * 2^k1 * 2^k2             k split so each factor is normal
 *
 * The coefficients come from src/miscellaneous/remez_exp_lut.cpp (Chebyshev
 * regression, IRLS at degree 4/8/16, then Remez exchange); see
 * include/nanofft_exp_coeffs.h.  The two clamped powers keep the way
 * branch-free while still saturating to 0 / inf exactly like exp, denormals
 * included (hardware gradual underflow does the work).
 *
 * Bit patterns travel as whole vectors, never through the address of a single
 * lane: clang in MSVC mode rejects taking a vector element's address.
 *
 * Requires VECF/VEC, VECF_INT/VEC_INT and VECF_LEN/VEC_LEN from nanofft.h.
 */

#include <nanofft_exp_coeffs.h>
#include <string.h>
#include <tls_platform.h>

static inline VECF vecf_sqrt(VECF v) {
    VECF r;
    for (int i = 0; i < VECF_LEN; i++) r[i] = sqrtf(v[i]);
    return r;
}

static inline VEC vec_sqrt(VEC v) {
    VEC r;
    for (int i = 0; i < VEC_LEN; i++) r[i] = sqrt(v[i]);
    return r;
}

static inline VECF vecf_exp(VECF v) {
    /* k = nearest integer of x/ln2. */
    VECF k;
    for (int i = 0; i < VECF_LEN; i++) k[i] = rintf(v[i] * (float)NANOFFT_EXP_INV_LN2_F);

    /* r = x - k*ln2 with the two-term split; k*hi is exact and the first
     * subtraction is exact by Sterbenz, so r keeps full precision. */
    VECF r = v - k * (float)NANOFFT_EXP_LN2_HI_F;
    r = r - k * (float)NANOFFT_EXP_LN2_LO_F;

    /* Remez polynomial, Horner (contracts to FMA where the target has one). */
    VECF p = (VECF){} + nanofft_expf_coeffs[NANOFFT_EXPF_DEGREE];
    for (int j = NANOFFT_EXPF_DEGREE - 1; j >= 0; --j) p = p * r + ((VECF){} + nanofft_expf_coeffs[j]);

    /* 2^k as two clamped powers of two, so the product underflows to zero and
     * overflows to infinity without any branch or special case. */
    VECF_INT bits1, bits2;
    for (int i = 0; i < VECF_LEN; i++) {
        float kh = k[i] < -126.0f ? -126.0f : (k[i] > 126.0f ? 126.0f : k[i]);
        float kl = k[i] - kh;
        kl = kl < -126.0f ? -126.0f : (kl > 126.0f ? 126.0f : kl);
        bits1[i] = ((uint32_t)((int)kh + 127)) << 23;
        bits2[i] = ((uint32_t)((int)kl + 127)) << 23;
    }
    VECF scale1, scale2;
    memcpy(&scale1, &bits1, sizeof scale1);
    memcpy(&scale2, &bits2, sizeof scale2);
    VECF result = p * scale1 * scale2;

    /* The reduction is undefined for infinities (inf - inf), so those lanes are
     * rebuilt from their bit pattern.  One cheap test keeps this out of the
     * common path; exp(-inf) = 0, exp(+inf) = +inf and NaN propagates. */
    uint32_t raw[VECF_LEN];
    memcpy(raw, &v, sizeof raw);
    unsigned special = 0;
    for (int i = 0; i < VECF_LEN; i++) special |= ((raw[i] & 0x7f800000u) == 0x7f800000u);
    if (special) {
        /* Rebuilt through integers: -ffinite-math-only makes an infinite or NaN
         * float constant undefined, but the bit pattern is always exact. */
        uint32_t out[VECF_LEN];
        memcpy(out, &result, sizeof out);
        for (int i = 0; i < VECF_LEN; i++) {
            const uint32_t b = raw[i];
            if ((b & 0x7f800000u) != 0x7f800000u) continue;
            out[i] = (b & 0x007fffffu) ? b : ((b & 0x80000000u) ? 0u : 0x7f800000u);
        }
        memcpy(&result, out, sizeof result);
    }
    return result;
}

static inline VEC vec_exp(VEC v) {
    VEC k;
    for (int i = 0; i < VEC_LEN; i++) k[i] = rint(v[i] * NANOFFT_EXP_INV_LN2);

    VEC r = v - k * NANOFFT_EXP_LN2_HI;
    r = r - k * NANOFFT_EXP_LN2_LO;

    VEC p = (VEC){} + nanofft_exp_coeffs[NANOFFT_EXP_DEGREE];
    for (int j = NANOFFT_EXP_DEGREE - 1; j >= 0; --j) p = p * r + ((VEC){} + nanofft_exp_coeffs[j]);

    VEC_INT bits1, bits2;
    for (int i = 0; i < VEC_LEN; i++) {
        double kh = k[i] < -1022.0 ? -1022.0 : (k[i] > 1023.0 ? 1023.0 : k[i]);
        double kl = k[i] - kh;
        kl = kl < -1022.0 ? -1022.0 : (kl > 1023.0 ? 1023.0 : kl);
        bits1[i] = ((uint64_t)((int64_t)kh + 1023)) << 52;
        bits2[i] = ((uint64_t)((int64_t)kl + 1023)) << 52;
    }
    VEC scale1, scale2;
    memcpy(&scale1, &bits1, sizeof scale1);
    memcpy(&scale2, &bits2, sizeof scale2);
    VEC result = p * scale1 * scale2;

    uint64_t raw[VEC_LEN];
    memcpy(raw, &v, sizeof raw);
    unsigned special = 0;
    for (int i = 0; i < VEC_LEN; i++) special |= ((raw[i] & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL);
    if (special) {
        uint64_t out[VEC_LEN];
        memcpy(out, &result, sizeof out);
        for (int i = 0; i < VEC_LEN; i++) {
            const uint64_t b = raw[i];
            if ((b & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL) continue;
            out[i] = (b & 0x000fffffffffffffULL) ? b : ((b & 0x8000000000000000ULL) ? 0ull : 0x7ff0000000000000ULL);
        }
        memcpy(&result, out, sizeof result);
    }
    return result;
}

#endif /* NANOFFT_VECMATH_H */
