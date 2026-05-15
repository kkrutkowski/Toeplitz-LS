/* Precision-selection macros shared by nanoFFT and NuFFT1 implementation
 * passes. This header is intentionally re-includable after changing
 * DOUBLE/DOUBLE_DOUBLE.
 */

#undef FLOAT
#undef PREFIX
#undef FALLBACK_PREFIX
#undef FALLBACK_FLOAT
#undef INTERNAL_VEC
#undef INTERNAL_VEC_INT
#undef INTERNAL_VEC_LEN
#undef M_COS2PI
#undef M_SIN2PI
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
#    define M_COS2PI(x) cos2pidd((x))
#    define M_SIN2PI(x) sin2pidd((x))
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
#    define M_COS2PI(x) cos2pi((double)(x))
#    define M_SIN2PI(x) sin2pi((double)(x))
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
#    define M_COS2PI(x) cos2pif((float)(x))
#    define M_SIN2PI(x) sin2pif((float)(x))
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

#endif

/* =========================================================================
 * Vector load / store helpers
 * ========================================================================= */

#define LOAD_VEC(ptr) (*(const INTERNAL_VEC *)(ptr))
#define STORE_VEC(ptr, val) (*(INTERNAL_VEC *)(ptr) = (val))
