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
    VEC k;
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
    const VEC zero = (VEC){};
    const VEC half = (VEC){} + 0.5;
    const VEC qtr = (VEC){} + 0.25;
    const VEC_INT sign_bit = (VEC_INT){} + UINT64_C(0x8000000000000000);

    VEC f = x - vec_floor_d(x);
    const VEC_INT ge_half = (VEC_INT)(f >= half);
    f = f - vec_select_d(ge_half, half, zero);
    const VEC_INT gt_qtr = (VEC_INT)(f > qtr);
    f = vec_select_d(gt_qtr, half - f, f);

    const VEC f2 = f * f;

    VEC ps = f2 * ((VEC){} + (-0.69093588239655204752473229179535744712019258849149)) + ((VEC){} + 3.816997428325180431886496831784263511664844308953);
    ps = ps * f2 + ((VEC){} + (-15.094471616631987517150792141240583632857536104374));
    ps = ps * f2 + ((VEC){} + 42.058688305389948652978636940879319419236423555816);
    ps = ps * f2 + ((VEC){} + (-76.705859647469529848428028740175591232140822775877));
    ps = ps * f2 + ((VEC){} + 81.605249275026290026468387665436788455350446194769);
    ps = ps * f2 + ((VEC){} + (-41.341702240395082626913754855518082730586045930014));
    ps = ps * f2 + ((VEC){} + 6.2831853071795803915351819993098437430793561832783);
    VEC s_res = ps * f;

    VEC pc = f2 * ((VEC){} + 0.2719476416639800139555005900627203625997526419121) + ((VEC){} + (-1.7132188587562783227021022500030456711383616906867));
    pc = pc * f2 + ((VEC){} + 7.9034625375924737942470808816341151858745136793854);
    pc = pc * f2 + ((VEC){} + (-26.426254069097002482358197039994768718882874895547));
    pc = pc * f2 + ((VEC){} + 60.244641313230924425507669180781096172477629015917);
    pc = pc * f2 + ((VEC){} + (-85.456817205981735856122645737764931252180966287231));
    pc = pc * f2 + ((VEC){} + 64.939394022663960757319625985316082107868087096872);
    pc = pc * f2 + ((VEC){} + (-19.739208802178707095137280041056835739350734071865));
    pc = pc * f2 + ((VEC){} + 0.99999999999999999608981951072301324039762563006958);

    VEC_INT sb, cb;
    memcpy(&sb, &s_res, sizeof sb);
    memcpy(&cb, &pc, sizeof cb);
    sb ^= (ge_half & sign_bit);
    cb ^= ((ge_half ^ gt_qtr) & sign_bit);
    memcpy(s_out, &sb, sizeof sb);
    memcpy(c_out, &cb, sizeof cb);
}

static inline void vecf_sincos2pi_d(VEC x_lo, VEC x_hi, VECF *c_out, VECF *s_out) {
    const VEC r_lo = x_lo - vec_floor_d(x_lo);
    const VEC r_hi = x_hi - vec_floor_d(x_hi);
    VECF f;
    for (int i = 0; i < VEC_LEN; i++) {
        f[i] = (float)r_lo[i];
        f[VEC_LEN + i] = (float)r_hi[i];
    }

    const VECF zero = (VECF){};
    const VECF half = (VECF){} + 0.5f;
    const VECF qtr = (VECF){} + 0.25f;
    const VECF_INT sign_bit = (VECF_INT){} + 0x80000000u;

    const VECF_INT ge_half = (VECF_INT)(f >= half);
    f = f - vecf_select_f(ge_half, half, zero);
    const VECF_INT gt_qtr = (VECF_INT)(f > qtr);
    f = vecf_select_f(gt_qtr, half - f, f);

    const VECF f2 = f * f;

    VECF ps = f2 * ((VECF){} + 39.536706065730207835108712734262f) + ((VECF){} + (-76.549782293595742666226937116116f));
    ps = ps * f2 + ((VECF){} + 81.601004073261773523492199897936f);
    ps = ps * f2 + ((VECF){} + (-41.341655031416278077153126232486f));
    ps = ps * f2 + ((VECF){} + 6.2831851600894774430188071795666f);
    VECF s_res = ps * f;

    VECF pc = f2 * ((VECF){} + 56.242380464873243259663276802701f) + ((VECF){} + (-85.240330322699427859509454517828f));
    pc = pc * f2 + ((VECF){} + 64.934590626780991246193352727536f);
    pc = pc * f2 + ((VECF){} + (-19.739171434702393618770795066531f));
    pc = pc * f2 + ((VECF){} + 0.99999995346667013630639784578184f);

    VECF_INT sb, cb;
    memcpy(&sb, &s_res, sizeof sb);
    memcpy(&cb, &pc, sizeof cb);
    sb ^= (ge_half & sign_bit);
    cb ^= ((ge_half ^ gt_qtr) & sign_bit);
    memcpy(s_out, &sb, sizeof sb);
    memcpy(c_out, &cb, sizeof cb);
}

static inline void vecdd_sincos2pi(VECDD x, VECDD *c_out, VECDD *s_out) {
    const VEC zero = (VEC){};
    const VEC one = (VEC){} + 1.0;
    const VEC neg_one = (VEC){} + (-1.0);
    const VEC half = (VEC){} + 0.5;
    const VEC qtr = (VEC){} + 0.25;
    const VEC_INT sign_bit = (VEC_INT){} + UINT64_C(0x8000000000000000);

    VECDD f = vecdd_sub(x, vecdd_make(vec_floor_d(x.hi), zero));
    const VEC_INT lt_zero = (VEC_INT)(f.hi < zero);
    const VEC_INT ge_one = (VEC_INT)(f.hi >= one);
    VEC adj = vec_select_d(lt_zero, one, vec_select_d(ge_one, neg_one, zero));
    f = vecdd_add(f, vecdd_make(adj, zero));

    const VEC_INT ge_half = (VEC_INT)(f.hi > half) | ((VEC_INT)(f.hi == half) & (VEC_INT)(f.lo >= zero));
    f = vecdd_sub(f, vecdd_make(vec_select_d(ge_half, half, zero), zero));

    const VEC_INT gt_qtr = (VEC_INT)(f.hi > qtr) | ((VEC_INT)(f.hi == qtr) & (VEC_INT)(f.lo > zero));
    const VECDD f_alt = vecdd_sub(vecdd_splat_d(0.5), f);
    f = vecdd_make(vec_select_d(gt_qtr, f_alt.hi, f.hi), vec_select_d(gt_qtr, f_alt.lo, f.lo));

    const VECDD f2 = vecdd_mul(f, f);

    VECDD pc = vecdd_splat((dd_t){-1.3712976302871458e-06, -1.4907449409481538e-23});
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){2.3093788063462385e-05, 4.5435375960601516e-22}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){-0.00032299035911159625, -8.9810661988144363e-21}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){0.0037798341474950792, -1.9249220349568919e-19}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){-0.036382841139801829, 2.9056792650806832e-18}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){0.28200596845569187, 1.5852734583076034e-17}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){-1.7143907110886696, 5.0670981029021671e-17}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){7.9035363713184692, -4.4106464905449524e-16}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){-26.426256783374399, 1.1988182697337608e-15}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){60.244641371876661, -3.2213071685554461e-16}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){-85.456817206693728, 2.0361754233578647e-16}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){64.939394022668296, -4.2563201319361759e-15}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){-19.739208802178716, -1.2530591017478957e-15}));
    pc = vecdd_add(vecdd_mul(pc, f2), vecdd_splat((dd_t){1.0, -7.4159205569012457e-33}));

    VECDD ps = vecdd_splat((dd_t){-3.1937621498346393e-07, -4.2494386355335374e-24});
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){5.8042120014329398e-06, -3.9949772624199443e-22}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){-8.8235163179284102e-05, 9.9742533594686292e-22}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){0.0011309237346037141, 9.0689538417711371e-20}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){-0.012031585941374802, 3.5320565199955664e-19}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){0.10422916220811097, 3.2446843930090187e-18}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){-0.71812230177849967, -4.3916880697967638e-17}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){3.819952584848282, 7.214503754465747e-17}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){-15.09464257682299, -6.8142151350129174e-16}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){42.058693944897655, -1.9698158197627135e-15}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){-76.705859753061389, 3.6416334350664191e-15}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){81.605249276075057, -2.5379220305392744e-15}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){-41.341702240399762, 1.8132498260632083e-15}));
    ps = vecdd_add(vecdd_mul(ps, f2), vecdd_splat((dd_t){6.2831853071795862, 2.4492935982947059e-16}));
    ps = vecdd_mul(ps, f);

    const VEC_INT s_flip = ge_half & sign_bit;
    const VEC_INT c_flip = (ge_half ^ gt_qtr) & sign_bit;
    VEC_INT ch, cl, sh, sl;
    memcpy(&ch, &pc.hi, sizeof ch);
    memcpy(&cl, &pc.lo, sizeof cl);
    memcpy(&sh, &ps.hi, sizeof sh);
    memcpy(&sl, &ps.lo, sizeof sl);
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
