/* Precision-selection macros shared by nanoFFT and NuFFT1 implementation
 * passes. This header is intentionally re-includable after changing
 * DOUBLE/DOUBLE_DOUBLE/HALF.
 */

#undef FLOAT
#undef PREFIX
#undef FALLBACK_PREFIX
#undef FALLBACK_FLOAT
#undef INTERNAL_VEC
#undef INTERNAL_VEC_INT
#undef INTERNAL_VEC_LEN
#undef IVEC_SCALAR
#undef IVEC_SPLAT
#undef M_COS2PI
#undef M_SIN2PI
#undef M_SINCOS2PI
#undef M_FMA
#undef M_EXP
#undef M_FABS
#undef ADD
#undef SUB
#undef MUL
#undef DIV
#undef NEG
#undef M_SQRT
#undef FCAST
#undef FCONST
#undef NANOFFT_FCONST_1
#undef NANOFFT_FCONST_2
#undef NANOFFT_FCONST_SELECT
#undef TO_DOUBLE
#undef FROM_FALLBACK
#undef TO_FALLBACK
#undef BREADTH_VEC
#undef BREADTH_VEC_LEN
#undef BREADTH_ADD
#undef BREADTH_SUB
#undef BREADTH_MUL
#undef BREADTH_MUL_S
#undef BREADTH_NEG
#undef BREADTH_SPLAT
#undef BREADTH_ZERO
#undef BREADTH_GET_LANE
#undef BREADTH_SET_LANE
#undef BREADTH_COMPUTE_VEC
#undef LOAD_VEC
#undef STORE_VEC
#undef LOAD_BVEC
#undef STORE_BVEC

#if defined(DOUBLE_DOUBLE)

#    define FLOAT dd_t
#    define PREFIX(name) nanofftdd_##name
#    define FALLBACK_PREFIX(name) nanofft_##name
#    define FALLBACK_FLOAT double

#    ifdef NANOFFT_NEEDS_INTERNAL_VEC
typedef dd_t INTERNAL_VEC;
typedef uint32_t INTERNAL_VEC_INT;
#    endif
#    define INTERNAL_VEC_LEN 1
#    define IVEC_SCALAR dd_t
#    define IVEC_SPLAT(s) (s)
#    define M_COS2PI(x) cos2pidd((x))
#    define M_SIN2PI(x) sin2pidd((x))
#    define M_SINCOS2PI(x, c, s) sincos2pidd((x), (c), (s))
#    define M_FMA(a, b, c) dd_add(dd_mul((a), (b)), (c))
#    define M_EXP(a) dd_exp((a))
#    define M_FABS(a) fabs(dd_to_double(a))

#    define ADD(a, b) dd_add((a), (b))
#    define SUB(a, b) dd_sub((a), (b))
#    define MUL(a, b) dd_mul((a), (b))
#    define DIV(a, b) dd_div((a), (b))
#    define M_SQRT(a) dd_sqrt((a))
#    define NEG(a) dd_make(-(a).hi, -(a).lo)
#    define FCAST(a) dd_make((double)(a), 0.0)
#    define NANOFFT_FCONST_1(a) dd_make((double)(a), 0.0)
#    define NANOFFT_FCONST_2(hi, lo) ((dd_t){(double)(hi), (double)(lo)})
#    define NANOFFT_FCONST_SELECT(_1, _2, NAME, ...) NAME
#    define FCONST(...) NANOFFT_FCONST_SELECT(__VA_ARGS__, NANOFFT_FCONST_2, NANOFFT_FCONST_1)(__VA_ARGS__)
#    define TO_DOUBLE(a) dd_to_double(a)
#    define FROM_FALLBACK(a) dd_make((double)(a), 0.0)
#    define TO_FALLBACK(a) ((a).hi)
#    define BREADTH_VEC VECDD
#    define BREADTH_COMPUTE_VEC VECDD
#    define BREADTH_VEC_LEN VECDD_LEN
#    define BREADTH_ADD(a, b) vecdd_add((a), (b))
#    define BREADTH_SUB(a, b) vecdd_sub((a), (b))
#    define BREADTH_MUL(a, b) vecdd_mul((a), (b))
#    define BREADTH_MUL_S(v, s) vecdd_mul_scalar((v), (s))
#    define BREADTH_NEG(a) vecdd_neg((a))
#    define BREADTH_SPLAT(s) vecdd_splat((s))
#    define BREADTH_ZERO ((VECDD){(VEC){}, (VEC){}})
#    define BREADTH_GET_LANE(v, lane) dd_make((v).hi[(lane)], (v).lo[(lane)])
#    define BREADTH_SET_LANE(v, lane, val) \
        do {                               \
            dd_t _bl_ = (val);             \
            (v).hi[(lane)] = _bl_.hi;      \
            (v).lo[(lane)] = _bl_.lo;      \
        } while (0)

#elif defined(DOUBLE)

#    define FLOAT double
#    define PREFIX(name) nanofft_##name
#    define FALLBACK_PREFIX(name) nanofftf_##name
#    define FALLBACK_FLOAT float

#    ifdef NANOFFT_NEEDS_INTERNAL_VEC
typedef VEC INTERNAL_VEC;
typedef VEC_INT INTERNAL_VEC_INT;
#    endif
#    define INTERNAL_VEC_LEN VEC_LEN
#    define IVEC_SCALAR double
#    define IVEC_SPLAT(s) ((VEC){} + (double)(s))
#    define M_COS2PI(x) cos2pi((double)(x))
#    define M_SIN2PI(x) sin2pi((double)(x))
#    define M_SINCOS2PI(x, c, s) sincos2pi((double)(x), (c), (s))
#    define M_FMA(a, b, c) fma((a), (b), (c))
#    define M_FABS(a) fabs(a)
#    define ADD(a, b) ((a) + (b))
#    define SUB(a, b) ((a) - (b))
#    define MUL(a, b) ((a) * (b))
#    define DIV(a, b) ((a) / (b))
#    define M_SQRT(a)                                                                                                                     \
        __extension__({                                                                                                                   \
            __typeof__(a) _msqrt_ = (a);                                                                                                  \
            _Generic((__typeof__(_msqrt_) *){0}, VEC *: vec_sqrt(*(VEC *)(void *)&_msqrt_), double *: sqrt(*(double *)(void *)&_msqrt_)); \
        })
#    define M_EXP(a)                                                                                                                 \
        __extension__({                                                                                                              \
            __typeof__(a) _mexp_ = (a);                                                                                              \
            _Generic((__typeof__(_mexp_) *){0}, VEC *: vec_exp(*(VEC *)(void *)&_mexp_), double *: exp(*(double *)(void *)&_mexp_)); \
        })
#    define NEG(a) (-(a))
#    define FCAST(a) ((double)(a))
#    define FCONST(hi, ...) ((double)(hi))
#    define TO_DOUBLE(a) ((double)(a))
#    define FROM_FALLBACK(a) ((double)(a))
#    define TO_FALLBACK(a) ((float)(a))
#    define BREADTH_VEC VEC
#    define BREADTH_COMPUTE_VEC VEC
#    define BREADTH_VEC_LEN VEC_LEN
#    define BREADTH_ADD(a, b) ((a) + (b))
#    define BREADTH_SUB(a, b) ((a) - (b))
#    define BREADTH_MUL(a, b) ((a) * (b))
#    define BREADTH_MUL_S(v, s) ((v) * ((VEC){} + (double)(s)))
#    define BREADTH_NEG(a) (-(a))
#    define BREADTH_SPLAT(s) ((VEC){} + (double)(s))
#    define BREADTH_ZERO ((VEC){})
#    define BREADTH_GET_LANE(v, lane) ((v)[(lane)])
#    define BREADTH_SET_LANE(v, lane, val) \
        do {                               \
            (v)[(lane)] = (val);           \
        } while (0)

#elif defined(HALF)

#    define FLOAT tls_f16_t
#    ifdef HALF_NATIVE_EMBED
#        define PREFIX(name) nanoffth_nat_##name
#    else
#        define PREFIX(name) nanoffth_##name
#    endif
#    define FALLBACK_PREFIX(name) nanofftf_##name
#    define FALLBACK_FLOAT float

#    define M_COS2PI(x) tls_f16_cos2pi((x))
#    define M_SIN2PI(x) tls_f16_sin2pi((x))
#    define M_SINCOS2PI(x, c, s) tls_f16_sincos2pi((x), (c), (s))
#    define M_FMA(a, b, c) tls_f16_fma((a), (b), (c))
#    define M_FABS(a) tls_f16_fabs((a))
#    define FCAST(a) tls_f32_to_f16((float)(a))
#    define FCONST(hi, ...) tls_f32_to_f16((float)(hi))
#    define TO_DOUBLE(a) ((double)tls_f16_to_f32((a)))
#    define FROM_FALLBACK(a) tls_f32_to_f16((a))
#    define TO_FALLBACK(a) tls_f16_to_f32((a))

#    define ADD(a, b)                                                                                                                   \
        __extension__({                                                                                                                 \
            __typeof__(a) _add_a_ = (a);                                                                                                \
            __typeof__(b) _add_b_ = (b);                                                                                                \
            _Generic((_add_a_), tls_half_ivec_t: tls_half_ivec_add(&_add_a_, &_add_b_), tls_f16_t: tls_half_s_add(&_add_a_, &_add_b_)); \
        })
#    define SUB(a, b)                                                                                                                   \
        __extension__({                                                                                                                 \
            __typeof__(a) _sub_a_ = (a);                                                                                                \
            __typeof__(b) _sub_b_ = (b);                                                                                                \
            _Generic((_sub_a_), tls_half_ivec_t: tls_half_ivec_sub(&_sub_a_, &_sub_b_), tls_f16_t: tls_half_s_sub(&_sub_a_, &_sub_b_)); \
        })
#    define MUL(a, b)                                                                                                                   \
        __extension__({                                                                                                                 \
            __typeof__(a) _mul_a_ = (a);                                                                                                \
            __typeof__(b) _mul_b_ = (b);                                                                                                \
            _Generic((_mul_a_), tls_half_ivec_t: tls_half_ivec_mul(&_mul_a_, &_mul_b_), tls_f16_t: tls_half_s_mul(&_mul_a_, &_mul_b_)); \
        })
#    define DIV(a, b)                                                                                                                   \
        __extension__({                                                                                                                 \
            __typeof__(a) _div_a_ = (a);                                                                                                \
            __typeof__(b) _div_b_ = (b);                                                                                                \
            _Generic((_div_a_), tls_half_ivec_t: tls_half_ivec_div(&_div_a_, &_div_b_), tls_f16_t: tls_half_s_div(&_div_a_, &_div_b_)); \
        })
#    define NEG(a)                                                                                                  \
        __extension__({                                                                                             \
            __typeof__(a) _neg_a_ = (a);                                                                            \
            _Generic((_neg_a_), tls_half_ivec_t: tls_half_ivec_neg(&_neg_a_), tls_f16_t: tls_half_s_neg(&_neg_a_)); \
        })
#    define M_SQRT(a)                                                                                                       \
        __extension__({                                                                                                     \
            __typeof__(a) _msqrt_a_ = (a);                                                                                  \
            _Generic((_msqrt_a_), tls_half_ivec_t: tls_half_ivec_sqrt(&_msqrt_a_), tls_f16_t: tls_half_s_sqrt(&_msqrt_a_)); \
        })
#    define M_EXP(a)                                                                                                   \
        __extension__({                                                                                                \
            __typeof__(a) _mexp_a_ = (a);                                                                              \
            _Generic((_mexp_a_), tls_half_ivec_t: tls_half_ivec_exp(&_mexp_a_), tls_f16_t: tls_half_s_exp(&_mexp_a_)); \
        })

#    if NANOFFT_NATIVE_FP16
#        ifdef NANOFFT_NEEDS_INTERNAL_VEC
typedef VECH INTERNAL_VEC;
typedef VECH_INT INTERNAL_VEC_INT;
#        endif
#        define INTERNAL_VEC_LEN VECH_LEN
#        define IVEC_SCALAR _Float16
#        define IVEC_SPLAT(s) ((VECH){} + tls_f16_to_fp16((s)))
#        define LOAD_VEC(ptr) (*(const VECH *)(const void *)(ptr))
#        define STORE_VEC(ptr, val) (*(VECH *)(void *)(ptr) = (val))
#        define BREADTH_VEC VECH
#        define BREADTH_COMPUTE_VEC VECH
#        define BREADTH_VEC_LEN VECH_LEN
#        define LOAD_BVEC(ptr) (*(const VECH *)(const void *)(ptr))
#        define STORE_BVEC(ptr, val) (*(VECH *)(void *)(ptr) = (val))
#        define BREADTH_ADD(a, b) ((a) + (b))
#        define BREADTH_SUB(a, b) ((a) - (b))
#        define BREADTH_MUL(a, b) ((a) * (b))
#        define BREADTH_MUL_S(v, s) ((v) * IVEC_SPLAT((s)))
#        define BREADTH_NEG(a) (-(a))
#        define BREADTH_SPLAT(s) IVEC_SPLAT((s))
#        define BREADTH_ZERO ((VECH){})
#        define BREADTH_GET_LANE(v, lane) tls_fp16_to_f16((v)[(lane)])
#        define BREADTH_SET_LANE(v, lane, val)        \
            do {                                      \
                (v)[(lane)] = tls_f16_to_fp16((val)); \
            } while (0)
#    else
#        ifdef NANOFFT_NEEDS_INTERNAL_VEC
typedef VECF INTERNAL_VEC;
typedef VECF_INT INTERNAL_VEC_INT;
#        endif
#        define INTERNAL_VEC_LEN VECF_LEN
#        define IVEC_SCALAR float
#        define IVEC_SPLAT(s) ((VECF){} + tls_f16_to_f32((s)))
#        define LOAD_VEC(ptr) vech_load_f32((const tls_f16_t *)(ptr))
#        define STORE_VEC(ptr, val) vech_store_f32((tls_f16_t *)(ptr), (val))
#        define BREADTH_VEC VECH
#        define BREADTH_COMPUTE_VEC VECF
#        define BREADTH_VEC_LEN VECH_LEN
#        define LOAD_BVEC(ptr) vech_to_vecf(*(const VECH *)(const void *)(ptr))
#        define STORE_BVEC(ptr, val) (*(VECH *)(void *)(ptr) = vecf_to_vech((val)))
#        define BREADTH_ADD(a, b) ((a) + (b))
#        define BREADTH_SUB(a, b) ((a) - (b))
#        define BREADTH_MUL(a, b) ((a) * (b))
#        define BREADTH_MUL_S(v, s) ((v) * IVEC_SPLAT((s)))
#        define BREADTH_NEG(a) (-(a))
#        define BREADTH_SPLAT(s) IVEC_SPLAT((s))
#        define BREADTH_ZERO ((VECF){})
#        define BREADTH_GET_LANE(v, lane) tls_f32_to_f16((v)[(lane)])
#        define BREADTH_SET_LANE(v, lane, val)       \
            do {                                     \
                (v)[(lane)] = tls_f16_to_f32((val)); \
            } while (0)
#    endif
#else

#    define FLOAT float
#    define PREFIX(name) nanofftf_##name
#    define FALLBACK_PREFIX(name) nanofft_##name
#    define FALLBACK_FLOAT double

#    ifdef NANOFFT_NEEDS_INTERNAL_VEC
typedef VECF INTERNAL_VEC;
typedef VECF_INT INTERNAL_VEC_INT;
#    endif
#    define INTERNAL_VEC_LEN VECF_LEN
#    define IVEC_SCALAR float
#    define IVEC_SPLAT(s) ((VECF){} + (float)(s))
#    define M_COS2PI(x) cos2pif((float)(x))
#    define M_SIN2PI(x) sin2pif((float)(x))
#    define M_SINCOS2PI(x, c, s) sincos2pif((float)(x), (c), (s))
#    define M_FMA(a, b, c) fmaf((a), (b), (c))
#    define M_FABS(a) fabsf(a)
#    define ADD(a, b) ((a) + (b))
#    define SUB(a, b) ((a) - (b))
#    define MUL(a, b) ((a) * (b))
#    define DIV(a, b) ((a) / (b))
#    define M_SQRT(a)                                                                                                                       \
        __extension__({                                                                                                                     \
            __typeof__(a) _msqrt_ = (a);                                                                                                    \
            _Generic((__typeof__(_msqrt_) *){0}, VECF *: vecf_sqrt(*(VECF *)(void *)&_msqrt_), float *: sqrtf(*(float *)(void *)&_msqrt_)); \
        })
#    define M_EXP(a)                                                                                                                   \
        __extension__({                                                                                                                \
            __typeof__(a) _mexp_ = (a);                                                                                                \
            _Generic((__typeof__(_mexp_) *){0}, VECF *: vecf_exp(*(VECF *)(void *)&_mexp_), float *: expf(*(float *)(void *)&_mexp_)); \
        })
#    define NEG(a) (-(a))
#    define FCAST(a) ((float)(a))
#    define FCONST(hi, ...) ((float)(hi))
#    define TO_DOUBLE(a) ((double)(a))
#    define FROM_FALLBACK(a) ((float)(a))
#    define TO_FALLBACK(a) ((double)(a))
#    define BREADTH_VEC VECF
#    define BREADTH_COMPUTE_VEC VECF
#    define BREADTH_VEC_LEN VECF_LEN
#    define BREADTH_ADD(a, b) ((a) + (b))
#    define BREADTH_SUB(a, b) ((a) - (b))
#    define BREADTH_MUL(a, b) ((a) * (b))
#    define BREADTH_MUL_S(v, s) ((v) * ((VECF){} + (float)(s)))
#    define BREADTH_NEG(a) (-(a))
#    define BREADTH_SPLAT(s) ((VECF){} + (float)(s))
#    define BREADTH_ZERO ((VECF){})
#    define BREADTH_GET_LANE(v, lane) ((v)[(lane)])
#    define BREADTH_SET_LANE(v, lane, val) \
        do {                               \
            (v)[(lane)] = (val);           \
        } while (0)

#endif

/* =========================================================================
 * Vector load / store helpers
 * ========================================================================= */

#ifndef LOAD_VEC
#    define LOAD_VEC(ptr) (*(const INTERNAL_VEC *)(ptr))
#    define STORE_VEC(ptr, val) (*(INTERNAL_VEC *)(ptr) = (val))
#endif
#ifndef LOAD_BVEC
#    define LOAD_BVEC(ptr) (*(const BREADTH_VEC *)(ptr))
#    define STORE_BVEC(ptr, val) (*(BREADTH_VEC *)(ptr) = (val))
#endif
