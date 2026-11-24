// Re-includable precision macro engine (intentionally no #pragma once),
// mirroring include/nanofft_precision.h.
//
// Select precision before including:
//   (default)       -> FP32:        f suffix   (Complexf, DifFftPlanf, ...)
//   SYCL_FFT_DOUBLE -> FP64:        no suffix  (Complex,  DifFftPlan,  ...)
//   SYCL_FFT_DF     -> DoubleFloat: df suffix  (Complexdf, DifFftPlandf, ...)

#undef SYCL_FFT_SUFFIX
#undef SYCL_FFT_REAL
#undef SYCL_FFT_COMPLEX
#undef SYCL_FFT_SMEM_T
#undef SYCL_FFT_ANGLE_T
#undef SYCL_FFT_WIDE
#undef SYCL_FFT_STASH_8192
#undef SYCL_FFT_COMPUTE_HEAVY
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
#define SYCL_FFT_SMEM_T sycl::float2
#define SYCL_FFT_ANGLE_T float
#define SYCL_FFT_WIDE 1
#define SYCL_FFT_STASH_8192 1
#define SYCL_FFT_COMPUTE_HEAVY 0

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
#define SYCL_FFT_CMUL_SQRT1_2(NegRe, NegIm, dr, di, or_, oi) \
    df_cmul_sqrt1_2<(NegRe), (NegIm)>((dr), (di), (or_), (oi))

#define SYCL_FFT_LOAD_COMPLEX(ptr, idx, re_out, im_out)                             \
    do {                                                                            \
        const sycl::float4 _v4 = reinterpret_cast<const sycl::float4*>(ptr)[(idx)]; \
        (re_out) = df_t{_v4.x(), _v4.y()};                                          \
        (im_out) = df_t{_v4.z(), _v4.w()};                                          \
    } while (0)

#define SYCL_FFT_STORE_COMPLEX(ptr, idx, re_in, im_in)                                            \
    do {                                                                                          \
        reinterpret_cast<sycl::float4*>(ptr)[(idx)] =                                             \
            sycl::float4{(re_in).hi, (re_in).lo, (im_in).hi, (im_in).lo};                         \
    } while (0)

#define SYCL_FFT_TO_SMEM(r) sycl::float2{(r).hi, (r).lo}
#define SYCL_FFT_FROM_SMEM(s) df_t{(s).x(), (s).y()}

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
#define SYCL_FFT_STASH_8192 0
#define SYCL_FFT_COMPUTE_HEAVY 1

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

#define SYCL_FFT_CMUL_SQRT1_2(NegRe, NegIm, dr, di, or_, oi)    \
    do {                                                        \
        const double _dr = (dr), _di = (di);                    \
        if constexpr (!(NegRe) && (NegIm)) {                    \
            (or_) = (_dr + _di) * kSqrt1_2;                     \
            (oi)  = (_di - _dr) * kSqrt1_2;                     \
        } else if constexpr ((NegRe) && (NegIm)) {              \
            (or_) = (_di - _dr) * kSqrt1_2;                     \
            (oi)  = -(_dr + _di) * kSqrt1_2;                    \
        } else if constexpr (!(NegRe) && !(NegIm)) {            \
            (or_) = (_dr - _di) * kSqrt1_2;                     \
            (oi)  = (_dr + _di) * kSqrt1_2;                     \
        } else {                                                \
            (or_) = -(_dr + _di) * kSqrt1_2;                    \
            (oi)  = (_dr - _di) * kSqrt1_2;                     \
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
    sycl::bit_cast<double>(sycl::bit_cast<uint64_t>(static_cast<double>(val)) ^                    \
                           (static_cast<uint64_t>(cond) << 63))

#define SYCL_FFT_TWIDDLE32_RE(Sign, idx) twiddle32_re<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE32_IM(Sign, idx) twiddle32_im<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_RE(Sign, idx) twiddle256_re<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_IM(Sign, idx) twiddle256_im<(Sign)>((idx))
#define SYCL_FFT_SINCOS2PI_OCT1(Sign, f_exact, c, s) sincos2pi_oct1<(Sign)>((f_exact), (c), (s))
#define SYCL_FFT_SINCOS2PI_HALF(Sign, f_exact, c, s) sincos2pi_half<(Sign)>((f_exact), (c), (s))

#else // FP32 (default)

#define SYCL_FFT_SUFFIX(name) name##f
#define SYCL_FFT_REAL float
#define SYCL_FFT_COMPLEX Complexf
#define SYCL_FFT_SMEM_T float
#define SYCL_FFT_ANGLE_T float
#define SYCL_FFT_WIDE 0
#define SYCL_FFT_STASH_8192 0
#define SYCL_FFT_COMPUTE_HEAVY 0

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

#define SYCL_FFT_CMUL_SQRT1_2(NegRe, NegIm, dr, di, or_, oi)    \
    do {                                                        \
        const float _dr = (dr), _di = (di);                     \
        if constexpr (!(NegRe) && (NegIm)) {                    \
            (or_) = (_dr + _di) * kSqrt1_2f;                    \
            (oi)  = (_di - _dr) * kSqrt1_2f;                    \
        } else if constexpr ((NegRe) && (NegIm)) {              \
            (or_) = (_di - _dr) * kSqrt1_2f;                    \
            (oi)  = -(_dr + _di) * kSqrt1_2f;                   \
        } else if constexpr (!(NegRe) && !(NegIm)) {            \
            (or_) = (_dr - _di) * kSqrt1_2f;                    \
            (oi)  = (_dr + _di) * kSqrt1_2f;                    \
        } else {                                                \
            (or_) = -(_dr + _di) * kSqrt1_2f;                   \
            (oi)  = (_dr - _di) * kSqrt1_2f;                    \
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
    sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(static_cast<float>(val)) ^                      \
                          (static_cast<uint32_t>(cond) << 31))

#define SYCL_FFT_TWIDDLE32_RE(Sign, idx) twiddle32_ref<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE32_IM(Sign, idx) twiddle32_imf<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_RE(Sign, idx) twiddle256_ref<(Sign)>((idx))
#define SYCL_FFT_TWIDDLE256_IM(Sign, idx) twiddle256_imf<(Sign)>((idx))
#define SYCL_FFT_SINCOS2PI_OCT1(Sign, f_exact, c, s) sincos2pif_oct1<(Sign)>((f_exact), (c), (s))
#define SYCL_FFT_SINCOS2PI_HALF(Sign, f_exact, c, s) sincos2pif_half<(Sign)>((f_exact), (c), (s))

#endif
