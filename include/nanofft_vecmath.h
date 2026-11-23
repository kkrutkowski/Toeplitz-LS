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
#if defined(__SSE2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && \
    ((VEC_BYTES == 64 && defined(__AVX512F__)) || (VEC_BYTES == 32 && defined(__AVX__)) || VEC_BYTES == 16)
#    if VEC_BYTES == 64
    return (VECF)_mm512_sqrt_ps((__m512)v);
#    elif VEC_BYTES == 32
    return (VECF)_mm256_sqrt_ps((__m256)v);
#    else
    return (VECF)_mm_sqrt_ps((__m128)v);
#    endif
#elif defined(__aarch64__) && VEC_BYTES == 16
    return (VECF)vsqrtq_f32((float32x4_t)v);
#else
    VECF r = {};
    for (int i = 0; i < VECF_LEN; i++) r[i] = sqrtf(v[i]);
    return r;
#endif
}

static inline VEC vec_sqrt(VEC v) {
#if defined(__SSE2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && \
    ((VEC_BYTES == 64 && defined(__AVX512F__)) || (VEC_BYTES == 32 && defined(__AVX__)) || VEC_BYTES == 16)
#    if VEC_BYTES == 64
    return (VEC)_mm512_sqrt_pd((__m512d)v);
#    elif VEC_BYTES == 32
    return (VEC)_mm256_sqrt_pd((__m256d)v);
#    else
    return (VEC)_mm_sqrt_pd((__m128d)v);
#    endif
#elif defined(__aarch64__) && VEC_BYTES == 16
    return (VEC)vsqrtq_f64((float64x2_t)v);
#else
    VEC r = {};
    for (int i = 0; i < VEC_LEN; i++) r[i] = sqrt(v[i]);
    return r;
#endif
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

/* -------------------------------------------------------------------------
 * Vectorized 2*pi-periodic sincos helpers (double, float-from-double, dd)
 *
 * Evaluates cos(2*pi*x) and sin(2*pi*x) simultaneously using the exact minimax
 * polynomials from nanofft.h with branch-free quadrant folding and exact
 * floor()-based mod-1 range reduction (no int32 overflow for large x).
 * ------------------------------------------------------------------------- */

static inline VEC vec_floor_d(VEC x) {
#if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX512F__) || defined(__aarch64__)
    VEC k = (VEC){};
    for (int i = 0; i < VEC_LEN; i++) k[i] = floor(x[i]);
    return k;
#else
    /* Baseline SSE2 (no roundsd): int64_t truncation is safe under -ffast-math
     * and lowers to cvttsd2si + cvtsi2sd with no libm call for |x| < 2^52. */
    const VEC one = (VEC){} + 1.0;
    const VEC lim = (VEC){} + 4503599627370496.0; /* 2^52 */
    VEC k;
    for (int i = 0; i < VEC_LEN; i++) k[i] = (double)(int64_t)x[i];
    VEC_INT gt = (VEC_INT)(k > x);
    VEC_INT one_b;
    memcpy(&one_b, &one, sizeof one_b);
    one_b &= gt;
    VEC adj;
    memcpy(&adj, &one_b, sizeof adj);
    k = k - adj;
    VEC_INT big = (VEC_INT)(x >= lim) | (VEC_INT)(x <= -lim);
    uint64_t any = 0;
    for (int i = 0; i < VEC_LEN; i++) any |= big[i];
    if (__builtin_expect(any != 0, 0)) {
        for (int i = 0; i < VEC_LEN; i++)
            if (big[i]) k[i] = floor(x[i]);
    }
    return k;
#endif
}

static inline VEC vec_select_d(VEC_INT mask, VEC a, VEC b) {
    VEC_INT ab, bb, rb;
    memcpy(&ab, &a, sizeof ab);
    memcpy(&bb, &b, sizeof bb);
    rb = (ab & mask) | (bb & ~mask);
    VEC r;
    memcpy(&r, &rb, sizeof r);
    return r;
}

static inline VECF vecf_select_f(VECF_INT mask, VECF a, VECF b) {
    VECF_INT ab, bb, rb;
    memcpy(&ab, &a, sizeof ab);
    memcpy(&bb, &b, sizeof bb);
    rb = (ab & mask) | (bb & ~mask);
    VECF r;
    memcpy(&r, &rb, sizeof r);
    return r;
}

static inline void vec_sincos2pi(VEC x, VEC *c_out, VEC *s_out) {
#if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX512F__) || defined(__aarch64__)
    VEC k = (VEC){};
    for (int i = 0; i < VEC_LEN; i++) k[i] = rint(x[i] * 4.0);
    const VEC r = x - k * ((VEC){} + 0.25);
    VEC_INT q = (VEC_INT){};
    for (int i = 0; i < VEC_LEN; i++) q[i] = (uint64_t)(int64_t)k[i];
#else
    const VEC f = x - vec_floor_d(x);
    VEC_INT q = (VEC_INT){};
    VEC k = (VEC){};
    for (int i = 0; i < VEC_LEN; i++) {
        const int64_t qi = (int64_t)(f[i] * 4.0 + 0.5);
        q[i] = (uint64_t)qi;
        k[i] = (double)qi;
    }
    const VEC r = f - k * ((VEC){} + 0.25);
#endif

    const VEC f2 = r * r;

    VEC ps = f2 * ((VEC){} + NANOFFT_SIN2PI_S6) + ((VEC){} + NANOFFT_SIN2PI_S5);
    ps = ps * f2 + ((VEC){} + NANOFFT_SIN2PI_S4);
    ps = ps * f2 + ((VEC){} + NANOFFT_SIN2PI_S3);
    ps = ps * f2 + ((VEC){} + NANOFFT_SIN2PI_S2);
    ps = ps * f2 + ((VEC){} + NANOFFT_SIN2PI_S1);
    ps = ps * f2 + ((VEC){} + NANOFFT_SIN2PI_S0);
    const VEC s_oct = ps * r;

    VEC pc = f2 * ((VEC){} + NANOFFT_COS2PI_C6) + ((VEC){} + NANOFFT_COS2PI_C5);
    pc = pc * f2 + ((VEC){} + NANOFFT_COS2PI_C4);
    pc = pc * f2 + ((VEC){} + NANOFFT_COS2PI_C3);
    pc = pc * f2 + ((VEC){} + NANOFFT_COS2PI_C2);
    pc = pc * f2 + ((VEC){} + NANOFFT_COS2PI_C1);
    const VEC c_oct = pc * f2 + ((VEC){} + NANOFFT_COS2PI_C0);

    const VEC_INT one = (VEC_INT){} + UINT64_C(1);
    const VEC_INT two = (VEC_INT){} + UINT64_C(2);
    const VEC_INT swap = (VEC_INT)((q & one) != (VEC_INT){});
    const VEC cm = vec_select_d(swap, s_oct, c_oct);
    const VEC sm = vec_select_d(swap, c_oct, s_oct);

    VEC_INT sb, cb;
    memcpy(&sb, &sm, sizeof sb);
    memcpy(&cb, &cm, sizeof cb);
    sb ^= (q & two) << 62;
    cb ^= ((q + one) & two) << 62;
    memcpy(s_out, &sb, sizeof sb);
    memcpy(c_out, &cb, sizeof cb);
}

static inline void vecf_sincos2pi_d(VEC x_lo, VEC x_hi, VECF *c_out, VECF *s_out) {
    const VEC r_lo = x_lo - vec_floor_d(x_lo);
    const VEC r_hi = x_hi - vec_floor_d(x_hi);
    VECF f = (VECF){};
    for (int i = 0; i < VEC_LEN; i++) {
        f[i] = (float)r_lo[i];
        f[VEC_LEN + i] = (float)r_hi[i];
    }

#if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX512F__) || defined(__aarch64__)
    VECF k = (VECF){};
    for (int i = 0; i < VECF_LEN; i++) k[i] = rintf(f[i] * 4.0f);
    const VECF r = f - k * ((VECF){} + 0.25f);
    VECF_INT q = (VECF_INT){};
    for (int i = 0; i < VECF_LEN; i++) q[i] = (uint32_t)(int32_t)k[i];
#else
    VECF_INT q = (VECF_INT){};
    VECF k = (VECF){};
    for (int i = 0; i < VECF_LEN; i++) {
        const int32_t qi = (int32_t)(f[i] * 4.0f + 0.5f);
        q[i] = (uint32_t)qi;
        k[i] = (float)qi;
    }
    const VECF r = f - k * ((VECF){} + 0.25f);
#endif

    const VECF f2 = r * r;

    VECF ps = f2 * ((VECF){} + NANOFFT_SIN2PIF_S3) + ((VECF){} + NANOFFT_SIN2PIF_S2);
    ps = ps * f2 + ((VECF){} + NANOFFT_SIN2PIF_S1);
    ps = ps * f2 + ((VECF){} + NANOFFT_SIN2PIF_S0);
    const VECF s_oct = ps * r;

    VECF pc = f2 * ((VECF){} + NANOFFT_COS2PIF_C3) + ((VECF){} + NANOFFT_COS2PIF_C2);
    pc = pc * f2 + ((VECF){} + NANOFFT_COS2PIF_C1);
    const VECF c_oct = pc * f2 + ((VECF){} + NANOFFT_COS2PIF_C0);

    const VECF_INT one = (VECF_INT){} + 1u;
    const VECF_INT two = (VECF_INT){} + 2u;
    const VECF_INT swap = (VECF_INT)((q & one) != (VECF_INT){});
    const VECF cm = vecf_select_f(swap, s_oct, c_oct);
    const VECF sm = vecf_select_f(swap, c_oct, s_oct);

    VECF_INT sb, cb;
    memcpy(&sb, &sm, sizeof sb);
    memcpy(&cb, &cm, sizeof cb);
    sb ^= (q & two) << 30;
    cb ^= ((q + one) & two) << 30;
    memcpy(s_out, &sb, sizeof sb);
    memcpy(c_out, &cb, sizeof cb);
}

static inline void vecdd_sincos2pi(VECDD x, VECDD *c_out, VECDD *s_out) {
    const VEC zero = (VEC){};
    const VECDD f = vecdd_sub(x, vecdd_make(vec_floor_d(x.hi), zero));

    VEC k = (VEC){};
    VEC_INT q = (VEC_INT){};
#if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX512F__) || defined(__aarch64__)
    for (int i = 0; i < VEC_LEN; i++) k[i] = rint(f.hi[i] * 4.0);
    for (int i = 0; i < VEC_LEN; i++) q[i] = (uint64_t)(int64_t)k[i];
#else
    for (int i = 0; i < VEC_LEN; i++) {
        const int64_t qi = (int64_t)(f.hi[i] * 4.0 + (f.hi[i] >= 0.0 ? 0.5 : -0.5));
        q[i] = (uint64_t)qi;
        k[i] = (double)qi;
    }
#endif
    const VECDD r = vecdd_sub(f, vecdd_make(k * ((VEC){} + 0.25), zero));
    const VECDD f2 = vecdd_mul(r, r);

    const VEC tc = (f2.hi * ((VEC){} + NANOFFT_COS2PIDD_C11) + ((VEC){} + NANOFFT_COS2PIDD_C10)) * f2.hi + ((VEC){} + NANOFFT_COS2PIDD_C9);
    VECDD pc = vecdd_make(tc, zero);
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C8));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C7));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C6));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C5));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C4));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C3));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C2));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C1));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat(NANOFFT_COS2PIDD_C0));

    VEC ts = f2.hi * ((VEC){} + NANOFFT_SIN2PIDD_S11) + ((VEC){} + NANOFFT_SIN2PIDD_S10);
    ts = ts * f2.hi + ((VEC){} + NANOFFT_SIN2PIDD_S9);
    ts = ts * f2.hi + ((VEC){} + NANOFFT_SIN2PIDD_S8);
    VECDD ps = vecdd_make(ts, zero);
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat(NANOFFT_SIN2PIDD_S7));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat(NANOFFT_SIN2PIDD_S6));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat(NANOFFT_SIN2PIDD_S5));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat(NANOFFT_SIN2PIDD_S4));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat(NANOFFT_SIN2PIDD_S3));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat(NANOFFT_SIN2PIDD_S2));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat(NANOFFT_SIN2PIDD_S1));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat(NANOFFT_SIN2PIDD_S0));
    ps = vecdd_mul(ps, r);

    const VEC_INT one = (VEC_INT){} + UINT64_C(1);
    const VEC_INT two = (VEC_INT){} + UINT64_C(2);
    const VEC_INT swap = (VEC_INT)((q & one) != (VEC_INT){});
    const VEC cm_hi = vec_select_d(swap, ps.hi, pc.hi);
    const VEC cm_lo = vec_select_d(swap, ps.lo, pc.lo);
    const VEC sm_hi = vec_select_d(swap, pc.hi, ps.hi);
    const VEC sm_lo = vec_select_d(swap, pc.lo, ps.lo);

    const VEC_INT s_flip = (q & two) << 62;
    const VEC_INT c_flip = ((q + one) & two) << 62;
    VEC_INT ch, cl, sh, sl;
    memcpy(&ch, &cm_hi, sizeof ch);
    memcpy(&cl, &cm_lo, sizeof cl);
    memcpy(&sh, &sm_hi, sizeof sh);
    memcpy(&sl, &sm_lo, sizeof sl);
    ch ^= c_flip;
    cl ^= c_flip;
    sh ^= s_flip;
    sl ^= s_flip;
    VEC c_hi, c_lo, s_hi, s_lo;
    memcpy(&c_hi, &ch, sizeof c_hi);
    memcpy(&c_lo, &cl, sizeof c_lo);
    memcpy(&s_hi, &sh, sizeof s_hi);
    memcpy(&s_lo, &sl, sizeof s_lo);
    *c_out = vecdd_make(c_hi, c_lo);
    *s_out = vecdd_make(s_hi, s_lo);
}

#endif /* NANOFFT_VECMATH_H */
