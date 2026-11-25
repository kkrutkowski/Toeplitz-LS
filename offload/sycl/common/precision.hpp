// Re-includable precision macro engine (intentionally no #pragma once),
// mirroring include/nanofft_precision.h.
//
// Select precision before including:
//   (default)       -> FP32:        f suffix   (Complexf,  DifFftPlanf,  ...)
//   SYCL_FFT_HALF   -> FP16 (2x):   h suffix   (Complexh,  DifFftPlanh,  ...)
//   SYCL_FFT_DOUBLE -> FP64:        no suffix  (Complex,   DifFftPlan,   ...)
//   SYCL_FFT_DF     -> DoubleFloat: df suffix  (Complexdf, DifFftPlandf, ...)
#undef SYCL_FFT_SUFFIX
#undef SYCL_FFT_REAL
#undef SYCL_FFT_COMPLEX
#undef SYCL_FFT_SMEM_T
#undef SYCL_FFT_ANGLE_T
#undef SYCL_FFT_WIDE
#undef SYCL_FFT_PREC_TAG
#undef SYCL_FFT_IS_DOUBLE
#undef SYCL_FFT_IS_DF
#undef SYCL_FFT_IS_HALF
#undef SYCL_FFT_PACKED_BATCH
#undef SYCL_FFT_ZERO
#undef SYCL_FFT_ONE
#undef SYCL_FFT_SQRT1_2
#undef SYCL_FFT_ADD
#undef SYCL_FFT_SUB
#undef SYCL_FFT_NEG
#undef SYCL_FFT_MUL
#undef SYCL_FFT_SCALE_HALF
#undef SYCL_FFT_CMUL
#undef SYCL_FFT_CSQR
#undef SYCL_FFT_CMUL_SQRT1_2
#undef SYCL_FFT_LOAD_COMPLEX
#undef SYCL_FFT_STORE_COMPLEX
#undef SYCL_FFT_TO_SMEM
#undef SYCL_FFT_FROM_SMEM
#undef SYCL_FFT_SHUFFLE_XOR
#undef SYCL_FFT_COND_NEG
#undef SYCL_FFT_TWIDDLE32_RE
#undef SYCL_FFT_TWIDDLE256_RE
#undef SYCL_FFT_TWIDDLE256_IM
#undef SYCL_FFT_TWIDDLE32_IM
#undef SYCL_FFT_SINCOS2PI_OCT1
#undef SYCL_FFT_SINCOS2PI_HALF

#if defined(SYCL_FFT_DF)

#define SYCL_FFT_SUFFIX(name) name##df
#define SYCL_FFT_REAL df_t
#define SYCL_FFT_COMPLEX Complexdf
#define SYCL_FFT_SMEM_T df_t
#define SYCL_FFT_ANGLE_T float
#define SYCL_FFT_WIDE 1
#define SYCL_FFT_PREC_TAG "df"
#define SYCL_FFT_IS_DOUBLE 0
#define SYCL_FFT_IS_DF 1
#define SYCL_FFT_IS_HALF 0
#define SYCL_FFT_PACKED_BATCH 1

#define SYCL_FFT_ZERO df_make(0.0f, 0.0f)
#define SYCL_FFT_ONE df_make(1.0f, 0.0f)
#define SYCL_FFT_SQRT1_2 kSqrt1_2df

#define SYCL_FFT_ADD(a, b) df_add((a), (b))
#define SYCL_FFT_SUB(a, b) df_sub((a), (b))
#define SYCL_FFT_NEG(a) df_neg((a))
#define SYCL_FFT_MUL(a, b) df_mul((a), (b))
#define SYCL_FFT_SCALE_HALF(x) df_scale_pow2((x), 0.5f)
#define SYCL_FFT_CMUL(ar, ai, br, bi, or_, oi) df_cmul_fused((ar), (ai), (br), (bi), (or_), (oi))
#define SYCL_FFT_CSQR(ar, ai, or_, oi) df_csqr_fused((ar), (ai), (or_), (oi))
#define SYCL_FFT_CMUL_SQRT1_2(NegRe, pos_sign, dr, di, or_, oi) \
    df_cmul_sqrt1_2_rt<(NegRe)>((dr), (di), (pos_sign), (or_), (oi))

#define SYCL_FFT_LOAD_COMPLEX(ptr, idx, re_out, im_out)     \
    do {                                                    \
        const Complexdf _v = (ptr)[(idx)];                  \
        (re_out) = _v.re;                                   \
        (im_out) = _v.im;                                   \
    } while (0)

#define SYCL_FFT_STORE_COMPLEX(ptr, idx, re_in, im_in)      \
    do {                                                    \
        (ptr)[(idx)] = Complexdf{(re_in), (im_in)};         \
    } while (0)

#define SYCL_FFT_TO_SMEM(r) (r)
#define SYCL_FFT_FROM_SMEM(s) (s)

#define SYCL_FFT_SHUFFLE_XOR(sg, val, mask)                                             \
    df_t{sycl::permute_group_by_xor((sg), (val).hi, static_cast<uint32_t>(mask)),       \
         sycl::permute_group_by_xor((sg), (val).lo, static_cast<uint32_t>(mask))}

#define SYCL_FFT_COND_NEG(val, cond) df_xor_sign((val), static_cast<uint32_t>(cond) << 31)

#define SYCL_FFT_TWIDDLE32_RE(Sign, idx) twiddle32_redf<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE32_IM(Sign, idx) twiddle32_imdf<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_RE(Sign, idx) twiddle256_redf<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_IM(Sign, idx) twiddle256_imdf<(Sign)>((idx))
#define SYCL_FFT_SINCOS2PI_OCT1(Sign, f_exact, c, s) sincos2pidf_oct1_f<(Sign)>((f_exact), (c), (s))
#define SYCL_FFT_SINCOS2PI_HALF(Sign, f_exact, c, s) sincos2pidf_half_f<(Sign)>((f_exact), (c), (s))

#elif defined(SYCL_FFT_DOUBLE)

#define SYCL_FFT_SUFFIX(name) name
#define SYCL_FFT_REAL double
#define SYCL_FFT_COMPLEX Complex
#define SYCL_FFT_SMEM_T double
#define SYCL_FFT_ANGLE_T double
#define SYCL_FFT_WIDE 1
#define SYCL_FFT_PREC_TAG "d"
#define SYCL_FFT_IS_DOUBLE 1
#define SYCL_FFT_IS_DF 0
#define SYCL_FFT_IS_HALF 0
#define SYCL_FFT_PACKED_BATCH 1

#define SYCL_FFT_ZERO 0.0
#define SYCL_FFT_ONE 1.0
#define SYCL_FFT_SQRT1_2 kSqrt1_2

#define SYCL_FFT_ADD(a, b) ((a) + (b))
#define SYCL_FFT_SUB(a, b) ((a) - (b))
#define SYCL_FFT_NEG(a) (-(a))
#define SYCL_FFT_MUL(a, b) ((a) * (b))
#define SYCL_FFT_SCALE_HALF(x) ((x) * 0.5)
#define SYCL_FFT_CMUL(ar, ai, br, bi, or_, oi)              \
    do {                                                    \
        const double _ar = (ar), _ai = (ai);                \
        const double _br = (br), _bi = (bi);                \
        (or_) = sycl::fma(_ar, _br, -(_ai * _bi));          \
        (oi)  = sycl::fma(_ar, _bi,   _ai * _br);           \
    } while (0)

#define SYCL_FFT_CSQR(ar, ai, or_, oi)                      \
    do {                                                    \
        const double _ar = (ar), _ai = (ai);                \
        (or_) = sycl::fma(_ar, _ar, -(_ai * _ai));          \
        (oi)  = (2.0 * _ar) * _ai;                          \
    } while (0)

#define SYCL_FFT_CMUL_SQRT1_2(NegRe, pos_sign, dr, di, or_, oi) \
    do {                                                        \
        const double _dr = (dr), _di = (di);                    \
        const uint64_t _sb = static_cast<uint64_t>(pos_sign) << 63; \
        const double _sdr = SYCL_FFT_BIT_CAST(double, SYCL_FFT_BIT_CAST(uint64_t, _dr) ^ _sb); \
        const double _sdi = SYCL_FFT_BIT_CAST(double, SYCL_FFT_BIT_CAST(uint64_t, _di) ^ _sb); \
        if constexpr (!(NegRe)) {                               \
            (or_) = (_dr + _sdi) * kSqrt1_2;                    \
            (oi)  = (_di - _sdr) * kSqrt1_2;                    \
        } else {                                                \
            (or_) = (_sdi - _dr) * kSqrt1_2;                    \
            (oi)  = -(_sdr + _di) * kSqrt1_2;                   \
        }                                                       \
    } while (0)

#define SYCL_FFT_LOAD_COMPLEX(ptr, idx, re_out, im_out)     \
    do {                                                    \
        const Complex _v = (ptr)[(idx)];                    \
        (re_out) = _v.x();                                  \
        (im_out) = _v.y();                                  \
    } while (0)

#define SYCL_FFT_STORE_COMPLEX(ptr, idx, re_in, im_in)      \
    do {                                                    \
        (ptr)[(idx)] = Complex{(re_in), (im_in)};           \
    } while (0)

#define SYCL_FFT_TO_SMEM(r) (r)
#define SYCL_FFT_FROM_SMEM(s) (s)

#define SYCL_FFT_SHUFFLE_XOR(sg, val, mask)                 \
    sycl::permute_group_by_xor((sg), (val), static_cast<uint32_t>(mask))

#define SYCL_FFT_COND_NEG(val, cond)                                                               \
    SYCL_FFT_BIT_CAST(double, SYCL_FFT_BIT_CAST(uint64_t, static_cast<double>(val)) ^              \
                              (static_cast<uint64_t>(cond) << 63))

#define SYCL_FFT_TWIDDLE32_RE(Sign, idx) twiddle32_re<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE32_IM(Sign, idx) twiddle32_im<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_RE(Sign, idx) twiddle256_re<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_IM(Sign, idx) twiddle256_im<(Sign)>((idx))
#define SYCL_FFT_SINCOS2PI_OCT1(Sign, f_exact, c, s) sincos2pi_oct1<(Sign)>((f_exact), (c), (s))
#define SYCL_FFT_SINCOS2PI_HALF(Sign, f_exact, c, s) sincos2pi_half<(Sign)>((f_exact), (c), (s))

#elif defined(SYCL_FFT_HALF)

#define SYCL_FFT_SUFFIX(name) name##h
#define SYCL_FFT_REAL f16x2_t
#define SYCL_FFT_COMPLEX Complexh
#define SYCL_FFT_SMEM_T uint32_t
#define SYCL_FFT_ANGLE_T float
#define SYCL_FFT_WIDE 0
#define SYCL_FFT_PREC_TAG "h"
#define SYCL_FFT_IS_DOUBLE 0
#define SYCL_FFT_IS_DF 0
#define SYCL_FFT_IS_HALF 1
#define SYCL_FFT_PACKED_BATCH 2

#define SYCL_FFT_ZERO kZero_f16x2
#define SYCL_FFT_ONE kOne_f16x2
#define SYCL_FFT_SQRT1_2 kSqrt1_2_f16x2

#define SYCL_FFT_ADD(a, b) ((a) + (b))
#define SYCL_FFT_SUB(a, b) ((a) - (b))
#define SYCL_FFT_NEG(a) (-(a))
#define SYCL_FFT_MUL(a, b) ((a) * (b))
#define SYCL_FFT_SCALE_HALF(x) ((x) * kHalf_f16x2)
#define SYCL_FFT_CMUL(ar, ai, br, bi, or_, oi)              \
    do {                                                    \
        const f16x2_t _ar = (ar), _ai = (ai);               \
        const f16x2_t _br = (br), _bi = (bi);               \
        (or_) = SYCL_FFT_F16X2_FMA(_ar, _br, -(_ai * _bi)); \
        (oi)  = SYCL_FFT_F16X2_FMA(_ar, _bi,   _ai * _br);  \
    } while (0)
#define SYCL_FFT_CSQR(ar, ai, or_, oi)                      \
    do {                                                    \
        const f16x2_t _ar = (ar), _ai = (ai);               \
        (or_) = SYCL_FFT_F16X2_FMA(_ar, _ar, -(_ai * _ai)); \
        (oi)  = (kTwo_f16x2 * _ar) * _ai;                   \
    } while (0)
#define SYCL_FFT_CMUL_SQRT1_2(NegRe, pos_sign, dr, di, or_, oi) \
    do {                                                        \
        const f16x2_t _dr = (dr), _di = (di);                   \
        const uint32_t _sb = (pos_sign) ? 0x80008000u : 0u;     \
        const f16x2_t _sdr = SYCL_FFT_BIT_CAST(f16x2_t, SYCL_FFT_BIT_CAST(uint32_t, _dr) ^ _sb); \
        const f16x2_t _sdi = SYCL_FFT_BIT_CAST(f16x2_t, SYCL_FFT_BIT_CAST(uint32_t, _di) ^ _sb); \
        if constexpr (!(NegRe)) {                               \
            (or_) = (_dr + _sdi) * kSqrt1_2_f16x2;              \
            (oi)  = (_di - _sdr) * kSqrt1_2_f16x2;              \
        } else {                                                \
            (or_) = (_sdi - _dr) * kSqrt1_2_f16x2;              \
            (oi)  = -(_sdr + _di) * kSqrt1_2_f16x2;             \
        }                                                       \
    } while (0)

#define SYCL_FFT_LOAD_COMPLEX(ptr, idx, re_out, im_out)                             \
    f16x2_load_pair((ptr), (ptr), (idx), true, false, (re_out), (im_out))

#define SYCL_FFT_STORE_COMPLEX(ptr, idx, re_in, im_in)                              \
    f16x2_store_pair((ptr), (ptr), (idx), true, false, (re_in), (im_in))

#define SYCL_FFT_TO_SMEM(r) SYCL_FFT_BIT_CAST(uint32_t, r)
#define SYCL_FFT_FROM_SMEM(s) SYCL_FFT_BIT_CAST(f16x2_t, s)

#define SYCL_FFT_SHUFFLE_XOR(sg, val, mask)                                         \
    SYCL_FFT_BIT_CAST(f16x2_t, sycl::permute_group_by_xor(                          \
        (sg), SYCL_FFT_BIT_CAST(uint32_t, val), static_cast<uint32_t>(mask)))

#define SYCL_FFT_COND_NEG(val, cond)                                                \
    SYCL_FFT_BIT_CAST(f16x2_t, SYCL_FFT_BIT_CAST(uint32_t, val) ^ ((cond) ? 0x80008000u : 0u))

#define SYCL_FFT_TWIDDLE32_RE(Sign, idx) kCos32h[(idx)]
#define SYCL_FFT_TWIDDLE32_IM(Sign, idx) ((Sign) < 0 ? kSin32Negh[(idx)] : -kSin32Negh[(idx)])
#define SYCL_FFT_TWIDDLE256_RE(Sign, idx) SYCL_FFT_F16X2_SPLAT_F(twiddle256_ref<(Sign)>((idx)))
#define SYCL_FFT_TWIDDLE256_IM(Sign, idx) SYCL_FFT_F16X2_SPLAT_F(twiddle256_imf<(Sign)>((idx)))
#define SYCL_FFT_SINCOS2PI_OCT1(Sign, f_exact, c, s)        \
    do {                                                    \
        float _cf = 0.0f, _sf = 0.0f;                       \
        sincos2pif_oct1<(Sign)>((f_exact), _cf, _sf);       \
        const f16_scalar_t _ch = static_cast<f16_scalar_t>(_cf); \
        const f16_scalar_t _sh = static_cast<f16_scalar_t>(_sf); \
        (c) = SYCL_FFT_F16X2_SPLAT_H(_ch);                  \
        (s) = SYCL_FFT_F16X2_SPLAT_H(_sh);                  \
    } while (0)
#define SYCL_FFT_SINCOS2PI_HALF(Sign, f_exact, c, s)        \
    do {                                                    \
        float _cf = 0.0f, _sf = 0.0f;                       \
        sincos2pif_half<(Sign)>((f_exact), _cf, _sf);       \
        const f16_scalar_t _ch = static_cast<f16_scalar_t>(_cf); \
        const f16_scalar_t _sh = static_cast<f16_scalar_t>(_sf); \
        (c) = SYCL_FFT_F16X2_SPLAT_H(_ch);                  \
        (s) = SYCL_FFT_F16X2_SPLAT_H(_sh);                  \
    } while (0)

#else // FP32 (default)

#define SYCL_FFT_SUFFIX(name) name##f
#define SYCL_FFT_REAL float
#define SYCL_FFT_COMPLEX Complexf
#define SYCL_FFT_SMEM_T float
#define SYCL_FFT_ANGLE_T float
#define SYCL_FFT_WIDE 0
#define SYCL_FFT_PREC_TAG "f"
#define SYCL_FFT_IS_DOUBLE 0
#define SYCL_FFT_IS_DF 0
#define SYCL_FFT_IS_HALF 0
#define SYCL_FFT_PACKED_BATCH 1

#define SYCL_FFT_ZERO 0.0f
#define SYCL_FFT_ONE 1.0f
#define SYCL_FFT_SQRT1_2 kSqrt1_2f

#define SYCL_FFT_ADD(a, b) ((a) + (b))
#define SYCL_FFT_SUB(a, b) ((a) - (b))
#define SYCL_FFT_NEG(a) (-(a))
#define SYCL_FFT_MUL(a, b) ((a) * (b))
#define SYCL_FFT_SCALE_HALF(x) ((x) * 0.5f)
#define SYCL_FFT_CMUL(ar, ai, br, bi, or_, oi)              \
    do {                                                    \
        const float _ar = (ar), _ai = (ai);                 \
        const float _br = (br), _bi = (bi);                 \
        (or_) = sycl::fma(_ar, _br, -(_ai * _bi));          \
        (oi)  = sycl::fma(_ar, _bi,   _ai * _br);           \
    } while (0)

#define SYCL_FFT_CSQR(ar, ai, or_, oi)                      \
    do {                                                    \
        const float _ar = (ar), _ai = (ai);                 \
        (or_) = sycl::fma(_ar, _ar, -(_ai * _ai));          \
        (oi)  = (2.0f * _ar) * _ai;                         \
    } while (0)

#define SYCL_FFT_CMUL_SQRT1_2(NegRe, pos_sign, dr, di, or_, oi) \
    do {                                                        \
        const float _dr = (dr), _di = (di);                     \
        const uint32_t _sb = static_cast<uint32_t>(pos_sign) << 31; \
        const float _sdr = SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, _dr) ^ _sb); \
        const float _sdi = SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, _di) ^ _sb); \
        if constexpr (!(NegRe)) {                               \
            (or_) = (_dr + _sdi) * kSqrt1_2f;                   \
            (oi)  = (_di - _sdr) * kSqrt1_2f;                   \
        } else {                                                \
            (or_) = (_sdi - _dr) * kSqrt1_2f;                   \
            (oi)  = -(_sdr + _di) * kSqrt1_2f;                  \
        }                                                       \
    } while (0)

#define SYCL_FFT_LOAD_COMPLEX(ptr, idx, re_out, im_out)     \
    do {                                                    \
        const Complexf _v = (ptr)[(idx)];                   \
        (re_out) = _v.x();                                  \
        (im_out) = _v.y();                                  \
    } while (0)

#define SYCL_FFT_STORE_COMPLEX(ptr, idx, re_in, im_in)      \
    do {                                                    \
        (ptr)[(idx)] = Complexf{(re_in), (im_in)};          \
    } while (0)

#define SYCL_FFT_TO_SMEM(r) (r)
#define SYCL_FFT_FROM_SMEM(s) (s)

#define SYCL_FFT_SHUFFLE_XOR(sg, val, mask)                 \
    sycl::permute_group_by_xor((sg), (val), static_cast<uint32_t>(mask))

#define SYCL_FFT_COND_NEG(val, cond)                                                               \
    SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, static_cast<float>(val)) ^                \
                             (static_cast<uint32_t>(cond) << 31))

#define SYCL_FFT_TWIDDLE32_RE(Sign, idx) twiddle32_ref<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE32_IM(Sign, idx) twiddle32_imf<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_RE(Sign, idx) twiddle256_ref<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_IM(Sign, idx) twiddle256_imf<(Sign)>((idx))
#define SYCL_FFT_SINCOS2PI_OCT1(Sign, f_exact, c, s) sincos2pif_oct1<(Sign)>((f_exact), (c), (s))
#define SYCL_FFT_SINCOS2PI_HALF(Sign, f_exact, c, s) sincos2pif_half<(Sign)>((f_exact), (c), (s))

#endif
