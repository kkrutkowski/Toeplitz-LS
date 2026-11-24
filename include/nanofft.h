#ifndef NANOFFT_H
#define NANOFFT_H

#if (defined(__FMA__) || defined(__SSE2__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
#    include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_FEATURE_FMA)
#    include <arm_neon.h>
#endif
#include <math.h>
#include <stdint.h>
#include <tls_export.h>

/* -------------------------------------------------------------------------
 * SIMD vector width selection
 * ------------------------------------------------------------------------- */
#ifndef VEC_BYTES
#    ifdef __AVX512F__
#        define VEC_BYTES 64
#    elif defined(__AVX__)
#        define VEC_BYTES 32
#    elif defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
/* SVE: the vector length has to be a compile-time constant for the GNU vector
 * extensions below, which is what -msve-vector-bits=N fixes (128/256/512).
 * Without that flag __ARM_FEATURE_SVE_BITS is 0 and the NEON width is used. */
#        define VEC_BYTES (__ARM_FEATURE_SVE_BITS / 8)
#    elif defined(__ARM_NEON) || defined(__aarch64__)
#        define VEC_BYTES 16
#    else
#        define VEC_BYTES 16
#    endif
#endif

#if defined(__AVX512FP16__) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#    define NANOFFT_NATIVE_FP16 1
#else
#    define NANOFFT_NATIVE_FP16 0
#endif

/* Alignment the plan and the transform buffers ask the allocator for: exactly
 * one vector, which is all an aligned vector load needs. */
#define NANOFFT_ALIGN VEC_BYTES

/* Portable allocation helpers, the shuffle macros and the math constants. */
#include <tls_platform.h>

/* -------------------------------------------------------------------------
 * Fused-twiddle FFT depth
 * ------------------------------------------------------------------------- */
/*
 * Fused-twiddle FFT depth, chosen at build time and configurable separately
 * for each precision:
 *
 *   NANOFFT_FFT_FUSE_FLOAT   float pass      (default floor(log2(VECF_LEN)) - 1)
 *   NANOFFT_FFT_FUSE_DOUBLE  double pass     (default floor(log2(VEC_LEN)))
 *   NANOFFT_FFT_FUSE_DD      double-double   (default 2, not vectorized)
 *
 * Each value is the number of radix-2 DIF stages one pass fuses (1 = plain
 * stage-at-a-time transform, 4 = radix-16 pass).  It is also the maximum the
 * runtime switch (tls_tuning_set_fft_fuse) can select for that precision, and
 * only kernels up to that depth are compiled.
 *
 * Single precision is deliberately one step shallower than its own lane count
 * suggests: it takes the fusion depth that a double-precision vector of the
 * same width would use.  One fused pass keeps 2^(L+1) data vectors plus ~6
 * twiddle/temporary registers live, so the full log2(VECF_LEN) depth fills the
 * whole register file, while the double-equivalent depth needs half of it.
 * The preliminary benchmarks (tests/bench_fuse.c, both -march=native and
 * -march=x86-64-v3, median over 64 relative cache-line offsets) show the
 * full-depth pass ending up slightly slower or at best equal to its shallower
 * neighbour, never faster, so there is no reason to pay for the extra pressure.
 * Depth 4 stays compiled and selectable at runtime; it is just not the default.
 *
 *   make FFT_FUSE_FLOAT=2 FFT_FUSE_DOUBLE=4
 */
#if VEC_BYTES >= 64
#    define NANOFFT_FFT_FUSE_FLOAT_DEFAULT 3  /* max(log2(VECF_LEN) - 1, 1), VECF_LEN = 16 */
#    define NANOFFT_FFT_FUSE_DOUBLE_DEFAULT 3 /* log2(VEC_LEN),  VEC_LEN  =  8 */
#elif VEC_BYTES >= 32
#    define NANOFFT_FFT_FUSE_FLOAT_DEFAULT 2  /* VECF_LEN = 8 */
#    define NANOFFT_FFT_FUSE_DOUBLE_DEFAULT 2 /* VEC_LEN  = 4 */
#else
#    define NANOFFT_FFT_FUSE_FLOAT_DEFAULT 1  /* VECF_LEN = 4 */
#    define NANOFFT_FFT_FUSE_DOUBLE_DEFAULT 1 /* VEC_LEN  = 2 */
#endif
#define NANOFFT_FFT_FUSE_DD_DEFAULT 2
#if VEC_BYTES >= 64
#    define NANOFFT_FFT_FUSE_HALF_DEFAULT 3
#elif VEC_BYTES >= 32
#    define NANOFFT_FFT_FUSE_HALF_DEFAULT 2
#else
#    define NANOFFT_FFT_FUSE_HALF_DEFAULT 1
#endif

#ifndef NANOFFT_FFT_FUSE_FLOAT
#    define NANOFFT_FFT_FUSE_FLOAT NANOFFT_FFT_FUSE_FLOAT_DEFAULT
#endif
#ifndef NANOFFT_FFT_FUSE_DOUBLE
#    define NANOFFT_FFT_FUSE_DOUBLE NANOFFT_FFT_FUSE_DOUBLE_DEFAULT
#endif
#ifndef NANOFFT_FFT_FUSE_DD
#    define NANOFFT_FFT_FUSE_DD NANOFFT_FFT_FUSE_DD_DEFAULT
#endif
#ifndef NANOFFT_FFT_FUSE_HALF
#    define NANOFFT_FFT_FUSE_HALF NANOFFT_FFT_FUSE_HALF_DEFAULT
#endif

/* Breadth-wise fused-twiddle FFT depth defaults (stages fused per pass). */
#if VEC_BYTES >= 64
#    define NANOFFT_FFT_BFUSE_FLOAT_DEFAULT 2
#    define NANOFFT_FFT_BFUSE_DOUBLE_DEFAULT 2
#    define NANOFFT_FFT_BFUSE_DD_DEFAULT 2
#elif VEC_BYTES >= 32
#    define NANOFFT_FFT_BFUSE_FLOAT_DEFAULT 2
#    define NANOFFT_FFT_BFUSE_DOUBLE_DEFAULT 2
#    define NANOFFT_FFT_BFUSE_DD_DEFAULT 1
#else
#    define NANOFFT_FFT_BFUSE_FLOAT_DEFAULT 2
#    define NANOFFT_FFT_BFUSE_DOUBLE_DEFAULT 2
#    define NANOFFT_FFT_BFUSE_DD_DEFAULT 1
#endif
#define NANOFFT_FFT_BFUSE_HALF_DEFAULT 2

#ifndef NANOFFT_FFT_BFUSE_FLOAT
#    define NANOFFT_FFT_BFUSE_FLOAT NANOFFT_FFT_BFUSE_FLOAT_DEFAULT
#endif
#ifndef NANOFFT_FFT_BFUSE_DOUBLE
#    define NANOFFT_FFT_BFUSE_DOUBLE NANOFFT_FFT_BFUSE_DOUBLE_DEFAULT
#endif
#ifndef NANOFFT_FFT_BFUSE_DD
#    define NANOFFT_FFT_BFUSE_DD NANOFFT_FFT_BFUSE_DD_DEFAULT
#endif
#ifndef NANOFFT_FFT_BFUSE_HALF
#    define NANOFFT_FFT_BFUSE_HALF NANOFFT_FFT_BFUSE_HALF_DEFAULT
#endif

/*
 * Fusion depth a plan would use for a grid of `N` points when the caller asks
 * for the automatic choice (TLS_TUNING_FFT_FUSE_AUTO): the per-grid value from
 * the tuned table when the library was built with one (-DNANOFFT_DEPTH_TABLE,
 * produced by `make tune` and by the release build), otherwise the
 * per-precision default above.  An explicit tls_tuning_set_fft_fuse() value
 * always wins over both.
 */
TLS_API int nanofftf_depth_for(uint32_t N);
TLS_API int nanofft_depth_for(uint32_t N);
TLS_API int nanofftdd_depth_for(uint32_t N);
TLS_API int nanofftf_breadth_depth_for(uint32_t N);
TLS_API int nanofft_breadth_depth_for(uint32_t N);
TLS_API int nanofftdd_breadth_depth_for(uint32_t N);
TLS_API int nanoffth_depth_for(uint32_t N);
TLS_API int nanoffth_breadth_depth_for(uint32_t N);

/* Largest depth any precision may be built with. */
#ifndef NANOFFT_FFT_FUSE_MAX
#    define NANOFFT_FFT_FUSE_MAX 4
#endif

#if NANOFFT_FFT_FUSE_FLOAT < 1 || NANOFFT_FFT_FUSE_FLOAT > NANOFFT_FFT_FUSE_MAX
#    error "NANOFFT_FFT_FUSE_FLOAT must be between 1 and NANOFFT_FFT_FUSE_MAX"
#endif
#if NANOFFT_FFT_FUSE_DOUBLE < 1 || NANOFFT_FFT_FUSE_DOUBLE > NANOFFT_FFT_FUSE_MAX
#    error "NANOFFT_FFT_FUSE_DOUBLE must be between 1 and NANOFFT_FFT_FUSE_MAX"
#endif
#if NANOFFT_FFT_FUSE_DD < 1 || NANOFFT_FFT_FUSE_DD > NANOFFT_FFT_FUSE_MAX
#    error "NANOFFT_FFT_FUSE_DD must be between 1 and NANOFFT_FFT_FUSE_MAX"
#endif
#if NANOFFT_FFT_BFUSE_FLOAT < 1 || NANOFFT_FFT_BFUSE_FLOAT > NANOFFT_FFT_FUSE_MAX
#    error "NANOFFT_FFT_BFUSE_FLOAT must be between 1 and NANOFFT_FFT_FUSE_MAX"
#endif
#if NANOFFT_FFT_BFUSE_DOUBLE < 1 || NANOFFT_FFT_BFUSE_DOUBLE > NANOFFT_FFT_FUSE_MAX
#    error "NANOFFT_FFT_BFUSE_DOUBLE must be between 1 and NANOFFT_FFT_FUSE_MAX"
#endif
#if NANOFFT_FFT_BFUSE_DD < 1 || NANOFFT_FFT_BFUSE_DD > NANOFFT_FFT_FUSE_MAX
#    error "NANOFFT_FFT_BFUSE_DD must be between 1 and NANOFFT_FFT_FUSE_MAX"
#endif
#if NANOFFT_FFT_FUSE_HALF < 1 || NANOFFT_FFT_FUSE_HALF > NANOFFT_FFT_FUSE_MAX
#    error "NANOFFT_FFT_FUSE_HALF must be between 1 and NANOFFT_FFT_FUSE_MAX"
#endif
#if NANOFFT_FFT_BFUSE_HALF < 1 || NANOFFT_FFT_BFUSE_HALF > NANOFFT_FFT_FUSE_MAX
#    error "NANOFFT_FFT_BFUSE_HALF must be between 1 and NANOFFT_FFT_FUSE_MAX"
#endif

/* -------------------------------------------------------------------------
 * Fast 2π-periodic trig approximations (Single & Double)
 * ------------------------------------------------------------------------- */

/* Reverse the low `bits` bits of `value`: the FFT's slot -> bin map.  Only
 * used while plans are built, never in a transform loop. */
static inline uint32_t nanofft_reverse_bits(uint32_t value, uint32_t bits) {
    uint32_t out = 0;
    for (uint32_t b = 0; b < bits; ++b) out = (out << 1) | ((value >> b) & 1u);
    return out;
}

/* Octant [0, 0.125] Chebyshev minimax coefficients for cos(2*pi*f) and
 * sin(2*pi*f)/f, shared between scalar (nanofft.h) and SIMD (nanofft_vecmath.h)
 * evaluators.  Reducing to [0, 0.125] (f^2 <= 1/64) cuts polynomial degree
 * across all three precisions while hitting < 0.5 ULP and exact cos(0) = 1. */
#define NANOFFT_COS2PIF_C3 (-83.5918198f)
#define NANOFFT_COS2PIF_C2 64.9212341f
#define NANOFFT_COS2PIF_C1 (-19.7391529f)
#define NANOFFT_COS2PIF_C0 1.0f

#define NANOFFT_SIN2PIF_S3 (-75.4016113f)
#define NANOFFT_SIN2PIF_S2 81.5925446f
#define NANOFFT_SIN2PIF_S1 (-41.3416634f)
#define NANOFFT_SIN2PIF_S0 6.28318548f

#define NANOFFT_COS2PI_C6 7.8102909681441908
#define NANOFFT_COS2PI_C5 (-26.42425703488269)
#define NANOFFT_COS2PI_C4 60.244620085930293
#define NANOFFT_COS2PI_C3 (-85.456817090341147)
#define NANOFFT_COS2PI_C2 64.939394022365377
#define NANOFFT_COS2PI_C1 (-19.739208802178421)
#define NANOFFT_COS2PI_C0 1.0

#define NANOFFT_SIN2PI_S6 3.7808689593022957
#define NANOFFT_SIN2PI_S5 (-15.093804209987114)
#define NANOFFT_SIN2PI_S4 42.058685020160894
#define NANOFFT_SIN2PI_S3 (-76.705859704274545)
#define NANOFFT_SIN2PI_S2 81.60524927594804
#define NANOFFT_SIN2PI_S1 (-41.341702240399634)
#define NANOFFT_SIN2PI_S0 6.2831853071795862

static inline float nanofft_cos2pif_oct1(float f2) {
    float p = f2 * NANOFFT_COS2PIF_C3 + NANOFFT_COS2PIF_C2;
    p = p * f2 + NANOFFT_COS2PIF_C1;
    return p * f2 + NANOFFT_COS2PIF_C0;
}

static inline float nanofft_sin2pif_oct1(float f, float f2) {
    float p = f2 * NANOFFT_SIN2PIF_S3 + NANOFFT_SIN2PIF_S2;
    p = p * f2 + NANOFFT_SIN2PIF_S1;
    p = p * f2 + NANOFFT_SIN2PIF_S0;
    return p * f;
}

static inline void sincos2pif(float x, float *c_out, float *s_out) {
#if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX512F__) || defined(__aarch64__)
    const float k = rintf(x * 4.0f);
    const float r = x - k * 0.25f;
    const uint32_t q = (uint32_t)(int32_t)k;
#else
    float f = x - (float)((int)x);
    if (f < 0.0f) f += 1.0f;
    const uint32_t q = (uint32_t)(int)(f * 4.0f + 0.5f);
    const float r = f - (float)q * 0.25f;
#endif
    const float f2 = r * r;
    const float pc = nanofft_cos2pif_oct1(f2);
    const float ps = nanofft_sin2pif_oct1(r, f2);
    const float cm = (q & 1u) ? ps : pc;
    const float sm = (q & 1u) ? pc : ps;
    *c_out = ((q + 1u) & 2u) ? -cm : cm;
    *s_out = (q & 2u) ? -sm : sm;
}

static inline float sin2pif(float x) {
    float c, s;
    sincos2pif(x, &c, &s);
    return s;
}

static inline float cos2pif(float x) {
    float c, s;
    sincos2pif(x, &c, &s);
    return c;
}

static inline double nanofft_cos2pi_oct1(double f2) {
    double p = f2 * NANOFFT_COS2PI_C6 + NANOFFT_COS2PI_C5;
    p = p * f2 + NANOFFT_COS2PI_C4;
    p = p * f2 + NANOFFT_COS2PI_C3;
    p = p * f2 + NANOFFT_COS2PI_C2;
    p = p * f2 + NANOFFT_COS2PI_C1;
    return p * f2 + NANOFFT_COS2PI_C0;
}

static inline double nanofft_sin2pi_oct1(double f, double f2) {
    double p = f2 * NANOFFT_SIN2PI_S6 + NANOFFT_SIN2PI_S5;
    p = p * f2 + NANOFFT_SIN2PI_S4;
    p = p * f2 + NANOFFT_SIN2PI_S3;
    p = p * f2 + NANOFFT_SIN2PI_S2;
    p = p * f2 + NANOFFT_SIN2PI_S1;
    p = p * f2 + NANOFFT_SIN2PI_S0;
    return p * f;
}

static inline void sincos2pi(double x, double *c_out, double *s_out) {
#if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX512F__) || defined(__aarch64__)
    const double k = rint(x * 4.0);
    const double r = x - k * 0.25;
    const uint64_t q = (uint64_t)(int64_t)k;
#else
    double f = x - (double)((int64_t)x);
    if (f < 0.0) f += 1.0;
    const uint64_t q = (uint64_t)(int)(f * 4.0 + 0.5);
    const double r = f - (double)q * 0.25;
#endif
    const double f2 = r * r;
    const double pc = nanofft_cos2pi_oct1(f2);
    const double ps = nanofft_sin2pi_oct1(r, f2);
    const double cm = (q & 1u) ? ps : pc;
    const double sm = (q & 1u) ? pc : ps;
    *c_out = ((q + 1u) & 2u) ? -cm : cm;
    *s_out = (q & 2u) ? -sm : sm;
}

static inline double sin2pi(double x) {
    double c, s;
    sincos2pi(x, &c, &s);
    return s;
}

static inline double cos2pi(double x) {
    double c, s;
    sincos2pi(x, &c, &s);
    return c;
}

/* -------------------------------------------------------------------------
 * Double-Double Precision Core Logic
 * ------------------------------------------------------------------------- */
typedef struct {
    double hi;
    double lo;
} dd_t;

NANOFFT_DIAG_PUSH_NAN
static inline dd_t dd_make(double hi, double lo) {
    dd_t res = {hi, lo};
    return res;
}

static inline void dd_two_sum(double a, double b, double *r, double *e) {
    *r = a + b;
    double v = *r - a;
    *e = (a - (*r - v)) + (b - v);
}

static inline void dd_two_sum_quick(double x, double y, double *r, double *e) {
    *r = x + y;
    *e = y - (*r - x);
}

static inline void dd_two_difference(double x, double y, double *r, double *e) {
    *r = x - y;
    double t = *r - x;
    *e = (x - (*r - t)) - (y + t);
}

static inline void dd_two_product(double x, double y, double *r, double *e) {
    *r = x * y;
    *e = fma(x, y, -*r);
}

static inline dd_t dd_add(dd_t a, dd_t b) {
    double r, e, r_f, e_f;
    dd_two_sum(a.hi, b.hi, &r, &e);
    e = e + a.lo + b.lo;
    dd_two_sum_quick(r, e, &r_f, &e_f);
    return dd_make(r_f, e_f);
}

static inline dd_t dd_sub(dd_t a, dd_t b) {
    double r, e, r_f, e_f;
    dd_two_difference(a.hi, b.hi, &r, &e);
    e = e + a.lo - b.lo;
    dd_two_sum_quick(r, e, &r_f, &e_f);
    return dd_make(r_f, e_f);
}

static inline dd_t dd_mul(dd_t a, dd_t b) {
    double r, e, r_f, e_f;
    dd_two_product(a.hi, b.hi, &r, &e);
    e = e + a.hi * b.lo + a.lo * b.hi;
    dd_two_sum_quick(r, e, &r_f, &e_f);
    return dd_make(r_f, e_f);
}

static inline double dd_to_double(dd_t a) { return a.hi + a.lo; }

static inline dd_t dd_div(dd_t a, dd_t b) {
    double q1 = a.hi / b.hi;
    dd_t q1_dd = dd_make(q1, 0.0);
    dd_t r = dd_sub(a, dd_mul(b, q1_dd));
    double q2 = r.hi / b.hi;
    return dd_add(q1_dd, dd_make(q2, 0.0));
}

static inline dd_t dd_sqrt(dd_t a) {
    if (a.hi > 0.0) {
        double c = sqrt(a.hi);
        double u, uu;

        // Exact product of c * c (equivalent to mul12 in your snippet)
        dd_two_product(c, c, &u, &uu);

        double cc = (a.hi - u - uu + a.lo) * 0.5 / c;
        double y = c + cc;
        double yy = c - y + cc;

        return dd_make(y, yy);
    }

    // Fallback if (x, xx) is not positive
    return dd_make(0.0, 0.0);
}

static inline dd_t exp2pidd_frac(dd_t x) {
    dd_t p = ((dd_t){3.01823422295644e-19, 1.4384934072545501e-35});
    p = dd_add(dd_mul(p, x), ((dd_t){5.1215627889193201e-18, -1.5467327444786251e-35}));
    p = dd_add(dd_mul(p, x), ((dd_t){1.3683397725627715e-16, -4.2604127187320126e-33}));
    p = dd_add(dd_mul(p, x), ((dd_t){3.1303369987019389e-15, -1.4969449397011624e-31}));
    p = dd_add(dd_mul(p, x), ((dd_t){6.7790064178603746e-14, 6.0883955717752676e-30}));
    p = dd_add(dd_mul(p, x), ((dd_t){1.3691461048536834e-12, -6.5395093947142517e-29}));
    p = dd_add(dd_mul(p, x), ((dd_t){2.5678438089317052e-11, 8.6900370612387885e-28}));
    p = dd_add(dd_mul(p, x), ((dd_t){4.4455382597571941e-10, -1.2760677676510325e-26}));
    p = dd_add(dd_mul(p, x), ((dd_t){7.0549116213394674e-09, 3.0104875606943678e-25}));
    p = dd_add(dd_mul(p, x), ((dd_t){1.017808600922137e-07, -5.796087189860976e-24}));
    p = dd_add(dd_mul(p, x), ((dd_t){1.3215486790144784e-06, -7.4623801416282544e-23}));
    p = dd_add(dd_mul(p, x), ((dd_t){1.5252733804059831e-05, 2.2898070277551725e-22}));
    p = dd_add(dd_mul(p, x), ((dd_t){0.00015403530393381609, 1.3065306543085402e-20}));
    p = dd_add(dd_mul(p, x), ((dd_t){0.0013333558146428443, 1.3801899555202445e-20}));
    p = dd_add(dd_mul(p, x), ((dd_t){0.0096181291076284769, 2.8325431266368226e-19}));
    p = dd_add(dd_mul(p, x), ((dd_t){0.055504108664821583, -3.1658226189498479e-18}));
    p = dd_add(dd_mul(p, x), ((dd_t){0.24022650695910072, -9.4939312462674605e-18}));
    p = dd_add(dd_mul(p, x), ((dd_t){0.69314718055994529, 2.3190468138405264e-17}));
    return dd_add(dd_mul(p, x), ((dd_t){1, 8.0108558802676677e-32}));
}

static inline dd_t dd_ldexp(dd_t a, int e) { return dd_make(ldexp(a.hi, e), ldexp(a.lo, e)); }

static inline dd_t dd_exp2(dd_t x) {
    double xd = dd_to_double(x);
    int e = (int)floor(xd);
    dd_t frac = dd_sub(x, dd_make((double)e, 0.0));
    return dd_ldexp(exp2pidd_frac(frac), e);
}

static inline dd_t dd_exp(dd_t x) {
    static const dd_t inv_ln2 = {1.44269504088896340736, 2.03552737409310331136e-17};
    return dd_exp2(dd_mul(x, inv_ln2));
}

static inline dd_t dd_log(dd_t x) {
    if (x.hi <= 0.0) return dd_make(-INFINITY, 0.0);
    double d = log(x.hi);
    dd_t y = dd_make(d, 0.0);
    dd_t ey = dd_exp(y);
    dd_t dy = dd_div(dd_sub(x, ey), ey);
    y = dd_add(y, dy);
    ey = dd_exp(y);
    dy = dd_div(dd_sub(x, ey), ey);
    return dd_add(y, dy);
}

/* Double-double octant [0, 0.125] Chebyshev minimax coefficients (deg 22 cos,
 * deg 23 sin, <= 1.6e-32 max error = 106 bits).  The highest terms satisfy
 * |c_k| * (1/64)^k * 2^-53 < 1e-34, so they are evaluated in plain double
 * before entering the compensated dd_t Horner tail (cutting DD Horner steps
 * from 13 to 9 for cos and 8 for sin). */
#define NANOFFT_COS2PIDD_C11 (-0.00032083236829691917)
#define NANOFFT_COS2PIDD_C10 0.0037797457541697286
#define NANOFFT_COS2PIDD_C9 (-0.036382839059704221)
#define NANOFFT_COS2PIDD_C8 ((dd_t){0.28200596842468023, -5.8346137740404667e-19})
#define NANOFFT_COS2PIDD_C7 ((dd_t){-1.7143907110883652, 4.9665774462462369e-17})
#define NANOFFT_COS2PIDD_C6 ((dd_t){7.9035363713184665, 2.4813881349981867e-16})
#define NANOFFT_COS2PIDD_C5 ((dd_t){-26.426256783374399, 1.2070288783354739e-15})
#define NANOFFT_COS2PIDD_C4 ((dd_t){60.244641371876661, -3.2215051856825278e-16})
#define NANOFFT_COS2PIDD_C3 ((dd_t){-85.456817206693728, 2.0361756146129074e-16})
#define NANOFFT_COS2PIDD_C2 ((dd_t){64.939394022668296, -4.2563201319203861e-15})
#define NANOFFT_COS2PIDD_C1 ((dd_t){-19.739208802178716, -1.2530591017479316e-15})
#define NANOFFT_COS2PIDD_C0 ((dd_t){1.0, 0.0})

#define NANOFFT_SIN2PIDD_S11 (-8.7692667141639106e-05)
#define NANOFFT_SIN2PIDD_S10 0.0011309015121999477
#define NANOFFT_SIN2PIDD_S9 (-0.01203158541845983)
#define NANOFFT_SIN2PIDD_S8 0.10422916220031785
#define NANOFFT_SIN2PIDD_S7 ((dd_t){-0.71812230177842329, -4.7054679255853891e-17})
#define NANOFFT_SIN2PIDD_S6 ((dd_t){3.8199525848482816, 2.3811552486513893e-17})
#define NANOFFT_SIN2PIDD_S5 ((dd_t){-15.09464257682299, -6.7943115622588666e-16})
#define NANOFFT_SIN2PIDD_S4 ((dd_t){42.058693944897655, -1.9698199606797813e-15})
#define NANOFFT_SIN2PIDD_S3 ((dd_t){-76.705859753061389, 3.6416334340474009e-15})
#define NANOFFT_SIN2PIDD_S2 ((dd_t){81.605249276075057, -2.5379220305126954e-15})
#define NANOFFT_SIN2PIDD_S1 ((dd_t){-41.341702240399762, 1.8132498260631586e-15})
#define NANOFFT_SIN2PIDD_S0 ((dd_t){6.2831853071795862, 2.4492935982947064e-16})

static inline dd_t nanofft_cos2pidd_oct1(dd_t f2) {
    const double t = (NANOFFT_COS2PIDD_C11 * f2.hi + NANOFFT_COS2PIDD_C10) * f2.hi + NANOFFT_COS2PIDD_C9;
    dd_t p = dd_make(t, 0.0);
    p = dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C8);
    p = dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C7);
    p = dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C6);
    p = dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C5);
    p = dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C4);
    p = dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C3);
    p = dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C2);
    p = dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C1);
    return dd_add(dd_mul(p, f2), NANOFFT_COS2PIDD_C0);
}

static inline dd_t nanofft_sin2pidd_oct1(dd_t f, dd_t f2) {
    double t = NANOFFT_SIN2PIDD_S11 * f2.hi + NANOFFT_SIN2PIDD_S10;
    t = t * f2.hi + NANOFFT_SIN2PIDD_S9;
    t = t * f2.hi + NANOFFT_SIN2PIDD_S8;
    dd_t p = dd_make(t, 0.0);
    p = dd_add(dd_mul(p, f2), NANOFFT_SIN2PIDD_S7);
    p = dd_add(dd_mul(p, f2), NANOFFT_SIN2PIDD_S6);
    p = dd_add(dd_mul(p, f2), NANOFFT_SIN2PIDD_S5);
    p = dd_add(dd_mul(p, f2), NANOFFT_SIN2PIDD_S4);
    p = dd_add(dd_mul(p, f2), NANOFFT_SIN2PIDD_S3);
    p = dd_add(dd_mul(p, f2), NANOFFT_SIN2PIDD_S2);
    p = dd_add(dd_mul(p, f2), NANOFFT_SIN2PIDD_S1);
    p = dd_add(dd_mul(p, f2), NANOFFT_SIN2PIDD_S0);
    return dd_mul(p, f);
}

static inline dd_t nanofft_reduce_oct1_dd(dd_t x, uint32_t *q_out) {
    const dd_t f = dd_sub(x, dd_make(floor(x.hi), 0.0));
#if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX512F__) || defined(__aarch64__)
    const double k = rint(f.hi * 4.0);
    *q_out = (uint32_t)(int32_t)k;
    return dd_sub(f, dd_make(k * 0.25, 0.0));
#else
    const int32_t k = (int32_t)(f.hi * 4.0 + (f.hi >= 0.0 ? 0.5 : -0.5));
    *q_out = (uint32_t)k;
    return dd_sub(f, dd_make((double)k * 0.25, 0.0));
#endif
}

static inline void sincos2pidd(dd_t x, dd_t *c_out, dd_t *s_out) {
    uint32_t q;
    const dd_t r = nanofft_reduce_oct1_dd(x, &q);
    const dd_t f2 = dd_mul(r, r);
    const dd_t pc = nanofft_cos2pidd_oct1(f2);
    const dd_t ps = nanofft_sin2pidd_oct1(r, f2);
    const dd_t cm = (q & 1u) ? ps : pc;
    const dd_t sm = (q & 1u) ? pc : ps;
    *c_out = ((q + 1u) & 2u) ? dd_make(-cm.hi, -cm.lo) : cm;
    *s_out = (q & 2u) ? dd_make(-sm.hi, -sm.lo) : sm;
}

static inline dd_t cos2pidd(dd_t x) {
    uint32_t q;
    const dd_t r = nanofft_reduce_oct1_dd(x, &q);
    const dd_t f2 = dd_mul(r, r);
    const dd_t p = (q & 1u) ? nanofft_sin2pidd_oct1(r, f2) : nanofft_cos2pidd_oct1(f2);
    return ((q + 1u) & 2u) ? dd_make(-p.hi, -p.lo) : p;
}

static inline dd_t sin2pidd(dd_t x) {
    uint32_t q;
    const dd_t r = nanofft_reduce_oct1_dd(x, &q);
    const dd_t f2 = dd_mul(r, r);
    const dd_t p = (q & 1u) ? nanofft_cos2pidd_oct1(f2) : nanofft_sin2pidd_oct1(r, f2);
    return (q & 2u) ? dd_make(-p.hi, -p.lo) : p;
}
NANOFFT_DIAG_POP

/* -------------------------------------------------------------------------
 * Single Precision API (nanofftf)
 *
 * Plans accept any power of two from 2 * INTERNAL_VEC_LEN upwards (32 for
 * float on AVX-512, 16 for float on AVX2 or double on AVX-512, 8 for double on
 * AVX2).  The intra-vector tail loads two vectors per step, so shorter grids
 * would read past the signal; make_plan() rejects them instead.  Every fusion
 * depth and both fusion orders are valid on every accepted grid.
 *
 * Signals handed to execute/permute must be aligned to NANOFFT_ALIGN bytes.
 * ------------------------------------------------------------------------- */
#define VECF_LEN (VEC_BYTES / 4)
typedef float VECF __attribute__((vector_size(VEC_BYTES)));
typedef uint32_t VECF_INT __attribute__((vector_size(VEC_BYTES)));

/* -------------------------------------------------------------------------
 * Half Precision Types & Converters (IEEE 754 binary16)
 * ------------------------------------------------------------------------- */
typedef struct {
    uint16_t bits;
} tls_f16_t;

#if NANOFFT_NATIVE_FP16
static inline _Float16 tls_f16_to_fp16(tls_f16_t h) {
    _Float16 f;
    memcpy(&f, &h.bits, sizeof f);
    return f;
}

static inline tls_f16_t tls_fp16_to_f16(_Float16 f) {
    tls_f16_t h;
    memcpy(&h.bits, &f, sizeof h.bits);
    return h;
}

static inline float tls_f16_to_f32(tls_f16_t h) { return (float)tls_f16_to_fp16(h); }
static inline tls_f16_t tls_f32_to_f16(float f) { return tls_fp16_to_f16((_Float16)f); }
#elif (defined(__F16C__) || defined(__AVX512F__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
static inline float tls_f16_to_f32(tls_f16_t h) { return _cvtsh_ss(h.bits); }
static inline tls_f16_t tls_f32_to_f16(float f) {
    tls_f16_t h;
    h.bits = _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return h;
}
#elif defined(__aarch64__)
static inline float tls_f16_to_f32(tls_f16_t h) {
    uint16x4_t u = vdup_n_u16(h.bits);
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(u)), 0);
}
static inline tls_f16_t tls_f32_to_f16(float f) {
    float32x4_t v = vdupq_n_f32(f);
    tls_f16_t h;
    h.bits = vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(v)), 0);
    return h;
}
#else
static inline float tls_f16_to_f32(tls_f16_t h) {
    uint32_t sign = ((uint32_t)h.bits & 0x8000u) << 16;
    uint32_t exp = (h.bits >> 10) & 0x1fu;
    uint32_t frac = h.bits & 0x03ffu;
    uint32_t out;
    if (exp == 0u) {
        if (frac == 0u) {
            out = sign;
        } else {
            int e = -14;
            while ((frac & 0x0400u) == 0u) {
                frac <<= 1;
                e--;
            }
            frac &= 0x03ffu;
            out = sign | ((uint32_t)(e + 127) << 23) | (frac << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        out = sign | ((exp - 15u + 127u) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &out, sizeof f);
    return f;
}

static inline tls_f16_t tls_f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t exp = (u >> 23) & 0xffu;
    uint32_t frac = u & 0x7fffffu;
    tls_f16_t h;
    if (exp == 0xffu) {
        h.bits = (uint16_t)(sign | 0x7c00u | (frac ? (0x0200u | (frac >> 13)) : 0u));
        return h;
    }
    int e = (int)exp - 127 + 15;
    if (e >= 31) {
        h.bits = (uint16_t)(sign | 0x7c00u);
        return h;
    }
    if (e <= 0) {
        if (e < -10) {
            h.bits = (uint16_t)sign;
            return h;
        }
        uint32_t m = frac | 0x800000u;
        int shift = 14 - e;
        uint32_t round_bit = 1u << (shift - 1);
        uint32_t sticky = m & (round_bit - 1u);
        uint32_t out = m >> shift;
        if ((m & round_bit) && (sticky || (out & 1u))) out++;
        h.bits = (uint16_t)(sign | out);
        return h;
    }
    uint32_t round_bit = 0x1000u;
    uint32_t sticky = frac & 0x0fffu;
    uint32_t out = frac >> 13;
    if ((frac & round_bit) && (sticky || (out & 1u))) {
        out++;
        if (out == 0x0400u) {
            out = 0u;
            e++;
            if (e >= 31) {
                h.bits = (uint16_t)(sign | 0x7c00u);
                return h;
            }
        }
    }
    h.bits = (uint16_t)(sign | ((uint32_t)e << 10) | out);
    return h;
}
#endif

#if NANOFFT_NATIVE_FP16
#    define VECH_LEN (VEC_BYTES / 2)
typedef _Float16 VECH __attribute__((vector_size(VEC_BYTES)));
typedef uint16_t VECH_INT __attribute__((vector_size(VEC_BYTES)));

static inline tls_f16_t tls_f16_add(tls_f16_t a, tls_f16_t b) { return tls_fp16_to_f16(tls_f16_to_fp16(a) + tls_f16_to_fp16(b)); }
static inline tls_f16_t tls_f16_sub(tls_f16_t a, tls_f16_t b) { return tls_fp16_to_f16(tls_f16_to_fp16(a) - tls_f16_to_fp16(b)); }
static inline tls_f16_t tls_f16_mul(tls_f16_t a, tls_f16_t b) { return tls_fp16_to_f16(tls_f16_to_fp16(a) * tls_f16_to_fp16(b)); }
static inline tls_f16_t tls_f16_div(tls_f16_t a, tls_f16_t b) { return tls_fp16_to_f16(tls_f16_to_fp16(a) / tls_f16_to_fp16(b)); }
static inline tls_f16_t tls_f16_fma(tls_f16_t a, tls_f16_t b, tls_f16_t c) {
    return tls_fp16_to_f16(tls_f16_to_fp16(a) * tls_f16_to_fp16(b) + tls_f16_to_fp16(c));
}
#else
#    define VECH_LEN VECF_LEN
typedef struct {
    tls_f16_t elems[VECH_LEN];
} VECH __attribute__((aligned(VEC_BYTES / 2)));

static inline VECF vech_load_f32(const tls_f16_t *p) {
#    if VEC_BYTES == 64 && defined(__AVX512F__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    return (VECF)_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(const void *)p));
#    elif VEC_BYTES == 32 && defined(__F16C__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    return (VECF)_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)p));
#    elif VEC_BYTES == 16 && defined(__F16C__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    return (VECF)_mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(const void *)p));
#    elif defined(__aarch64__) && VEC_BYTES == 16
    return (VECF)vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t *)(const void *)p)));
#    else
    VECF v = (VECF){};
    for (int i = 0; i < VECH_LEN; i++) v[i] = tls_f16_to_f32(p[i]);
    return v;
#    endif
}

static inline void vech_store_f32(tls_f16_t *p, VECF v) {
#    if VEC_BYTES == 64 && defined(__AVX512F__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    _mm256_storeu_si256((__m256i *)(void *)p, _mm512_cvtps_ph((__m512)v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#    elif VEC_BYTES == 32 && defined(__F16C__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    _mm_storeu_si128((__m128i *)(void *)p, _mm256_cvtps_ph((__m256)v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#    elif VEC_BYTES == 16 && defined(__F16C__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    _mm_storel_epi64((__m128i *)(void *)p, _mm_cvtps_ph((__m128)v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#    elif defined(__aarch64__) && VEC_BYTES == 16
    vst1_u16((uint16_t *)(void *)p, vreinterpret_u16_f16(vcvt_f16_f32((float32x4_t)v)));
#    else
    for (int i = 0; i < VECH_LEN; i++) p[i] = tls_f32_to_f16(v[i]);
#    endif
}

static inline VECF vech_to_vecf(VECH h) { return vech_load_f32(h.elems); }
static inline VECH vecf_to_vech(VECF v) {
    VECH h;
    vech_store_f32(h.elems, v);
    return h;
}

static inline tls_f16_t tls_f16_add(tls_f16_t a, tls_f16_t b) { return tls_f32_to_f16(tls_f16_to_f32(a) + tls_f16_to_f32(b)); }
static inline tls_f16_t tls_f16_sub(tls_f16_t a, tls_f16_t b) { return tls_f32_to_f16(tls_f16_to_f32(a) - tls_f16_to_f32(b)); }
static inline tls_f16_t tls_f16_mul(tls_f16_t a, tls_f16_t b) { return tls_f32_to_f16(tls_f16_to_f32(a) * tls_f16_to_f32(b)); }
static inline tls_f16_t tls_f16_div(tls_f16_t a, tls_f16_t b) { return tls_f32_to_f16(tls_f16_to_f32(a) / tls_f16_to_f32(b)); }
static inline tls_f16_t tls_f16_fma(tls_f16_t a, tls_f16_t b, tls_f16_t c) {
    return tls_f32_to_f16(fmaf(tls_f16_to_f32(a), tls_f16_to_f32(b), tls_f16_to_f32(c)));
}
#endif

static inline tls_f16_t tls_f16_neg(tls_f16_t a) {
    tls_f16_t r;
    r.bits = (uint16_t)(a.bits ^ 0x8000u);
    return r;
}
static inline tls_f16_t tls_f16_sqrt(tls_f16_t a) { return tls_f32_to_f16(sqrtf(tls_f16_to_f32(a))); }
static inline tls_f16_t tls_f16_exp(tls_f16_t a) { return tls_f32_to_f16(expf(tls_f16_to_f32(a))); }
static inline float tls_f16_fabs(tls_f16_t a) { return fabsf(tls_f16_to_f32(a)); }
static inline void tls_f16_sincos2pi(tls_f16_t x, tls_f16_t *c_out, tls_f16_t *s_out) {
    float cf, sf;
    sincos2pif(tls_f16_to_f32(x), &cf, &sf);
    *c_out = tls_f32_to_f16(cf);
    *s_out = tls_f32_to_f16(sf);
}
static inline tls_f16_t tls_f16_cos2pi(tls_f16_t x) { return tls_f32_to_f16(cos2pif(tls_f16_to_f32(x))); }
static inline tls_f16_t tls_f16_sin2pi(tls_f16_t x) { return tls_f32_to_f16(sin2pif(tls_f16_to_f32(x))); }

typedef struct nanoffth_plan nanoffth_plan;
TLS_API nanoffth_plan *nanoffth_make_plan(uint32_t N);
TLS_API void nanoffth_destroy_plan(nanoffth_plan *plan);
TLS_API void nanoffth_execute(nanoffth_plan *plan, tls_f16_t *real_signal, tls_f16_t *imag_signal);
TLS_API void nanoffth_execute_raw(nanoffth_plan *plan, tls_f16_t *real_signal, tls_f16_t *imag_signal);
TLS_API void nanoffth_permute(nanoffth_plan *plan, tls_f16_t *real_signal, tls_f16_t *imag_signal);
TLS_API void nanoffth_permute_1d(nanoffth_plan *plan, tls_f16_t *values);
TLS_API nanoffth_plan *nanoffth_make_plan_breadth(uint32_t N);
TLS_API void nanoffth_execute_breadth(nanoffth_plan *plan, VECH *real_signal, VECH *imag_signal);
TLS_API void nanoffth_execute_raw_breadth(nanoffth_plan *plan, VECH *real_signal, VECH *imag_signal);
TLS_API void nanoffth_permute_breadth(nanoffth_plan *plan, VECH *real_signal, VECH *imag_signal);
TLS_API void nanoffth_permute_1d_breadth(nanoffth_plan *plan, VECH *values);

#ifdef HALF_NATIVE_EMBED
typedef struct nanoffth_nat_plan nanoffth_nat_plan;
int nanoffth_nat_depth_for(uint32_t N);
int nanoffth_nat_breadth_depth_for(uint32_t N);
nanoffth_nat_plan *nanoffth_nat_make_plan(uint32_t N);
void nanoffth_nat_destroy_plan(nanoffth_nat_plan *plan);
void nanoffth_nat_execute(nanoffth_nat_plan *plan, tls_f16_t *real_signal, tls_f16_t *imag_signal);
void nanoffth_nat_execute_raw(nanoffth_nat_plan *plan, tls_f16_t *real_signal, tls_f16_t *imag_signal);
void nanoffth_nat_permute(nanoffth_nat_plan *plan, tls_f16_t *real_signal, tls_f16_t *imag_signal);
void nanoffth_nat_permute_1d(nanoffth_nat_plan *plan, tls_f16_t *values);
nanoffth_nat_plan *nanoffth_nat_make_plan_breadth(uint32_t N);
void nanoffth_nat_execute_breadth(nanoffth_nat_plan *plan, VECH *real_signal, VECH *imag_signal);
void nanoffth_nat_execute_raw_breadth(nanoffth_nat_plan *plan, VECH *real_signal, VECH *imag_signal);
void nanoffth_nat_permute_breadth(nanoffth_nat_plan *plan, VECH *real_signal, VECH *imag_signal);
void nanoffth_nat_permute_1d_breadth(nanoffth_nat_plan *plan, VECH *values);
#endif
typedef struct nanofftf_plan nanofftf_plan;
TLS_API nanofftf_plan *nanofftf_make_plan(uint32_t N);
TLS_API void nanofftf_destroy_plan(nanofftf_plan *plan);
/* Full transform: Sande-Tukey passes plus bit reversal (frequency order). */
TLS_API void nanofftf_execute(nanofftf_plan *plan, float *real_signal, float *imag_signal);
/* Sande-Tukey passes only: the spectrum stays in the FFT's bit-reversed slot
 * order, so one permutation can be deferred to a single scalar result array. */
TLS_API void nanofftf_execute_raw(nanofftf_plan *plan, float *real_signal, float *imag_signal);
/* In-place bit reversal of a slot-ordered pair / single array. */
TLS_API void nanofftf_permute(nanofftf_plan *plan, float *real_signal, float *imag_signal);
TLS_API void nanofftf_permute_1d(nanofftf_plan *plan, float *values);
/* Breadth-wise transform: each VECF holds VECF_LEN independent blocks at the same bin. */
TLS_API nanofftf_plan *nanofftf_make_plan_breadth(uint32_t N);
TLS_API void nanofftf_execute_breadth(nanofftf_plan *plan, VECF *real_signal, VECF *imag_signal);
TLS_API void nanofftf_execute_raw_breadth(nanofftf_plan *plan, VECF *real_signal, VECF *imag_signal);
TLS_API void nanofftf_permute_breadth(nanofftf_plan *plan, VECF *real_signal, VECF *imag_signal);
TLS_API void nanofftf_permute_1d_breadth(nanofftf_plan *plan, VECF *values);

/* -------------------------------------------------------------------------
 * Double Precision API (nanofft)
 * ------------------------------------------------------------------------- */
#define VEC_LEN (VEC_BYTES / 8)
typedef double VEC __attribute__((vector_size(VEC_BYTES)));
typedef uint64_t VEC_INT __attribute__((vector_size(VEC_BYTES)));

/* -------------------------------------------------------------------------
 * Double-Double SIMD Vector Type (VECDD)
 *
 * Shares double's lane count (VECDD_LEN == VEC_LEN).  Every lane holds one
 * compensated hi/lo pair; arithmetic uses hardware FMA when available and
 * portable Veltkamp splitting otherwise (never scalar libm fma()).
 * ------------------------------------------------------------------------- */
#define VECDD_LEN VEC_LEN
typedef struct {
    VEC hi;
    VEC lo;
} VECDD;

static inline VECDD vecdd_make(VEC hi, VEC lo) {
    VECDD res = {hi, lo};
    return res;
}

static inline VECDD vecdd_splat(dd_t a) {
    VECDD res;
    res.hi = (VEC){} + a.hi;
    res.lo = (VEC){} + a.lo;
    return res;
}

static inline VECDD vecdd_splat_d(double a) {
    VECDD res;
    res.hi = (VEC){} + a;
    res.lo = (VEC){};
    return res;
}

static inline VECDD vecdd_neg(VECDD a) { return vecdd_make(-a.hi, -a.lo); }

static inline void vecdd_two_sum(VEC a, VEC b, VEC *r, VEC *e) {
    *r = a + b;
    VEC v = *r - a;
    *e = (a - (*r - v)) + (b - v);
}

static inline void vecdd_two_sum_quick(VEC x, VEC y, VEC *r, VEC *e) {
    *r = x + y;
    *e = y - (*r - x);
}

static inline void vecdd_two_difference(VEC x, VEC y, VEC *r, VEC *e) {
    *r = x - y;
    VEC t = *r - x;
    *e = (x - (*r - t)) - (y + t);
}

static inline void vecdd_two_product(VEC x, VEC y, VEC *r, VEC *e) {
    *r = x * y;
#if defined(__FMA__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && \
    ((VEC_BYTES == 64 && defined(__AVX512F__)) || (VEC_BYTES == 32 && defined(__AVX__)) || (VEC_BYTES == 16 && defined(__SSE2__)))
#    if VEC_BYTES == 64
    *e = (VEC)_mm512_fmsub_pd((__m512d)x, (__m512d)y, (__m512d)(*r));
#    elif VEC_BYTES == 32
    *e = (VEC)_mm256_fmsub_pd((__m256d)x, (__m256d)y, (__m256d)(*r));
#    else
    *e = (VEC)_mm_fmsub_pd((__m128d)x, (__m128d)y, (__m128d)(*r));
#    endif
#elif (defined(__aarch64__) || defined(__ARM_FEATURE_FMA)) && VEC_BYTES == 16
    *e = (VEC)vfmaq_f64((float64x2_t)(-(*r)), (float64x2_t)x, (float64x2_t)y);
#else
    /* Portable IEEE-754 Veltkamp splitting (exact under -fno-fast-math). */
    const VEC split = (VEC){} + 134217729.0; /* 2^27 + 1 */
    VEC cx = split * x;
    VEC x_hi = cx - (cx - x);
    VEC x_lo = x - x_hi;
    VEC cy = split * y;
    VEC y_hi = cy - (cy - y);
    VEC y_lo = y - y_hi;
    *e = ((x_hi * y_hi - *r) + x_hi * y_lo + x_lo * y_hi) + x_lo * y_lo;
#endif
}

static inline VECDD vecdd_add(VECDD a, VECDD b) {
    VEC r, e, r_f, e_f;
    vecdd_two_sum(a.hi, b.hi, &r, &e);
    e = e + a.lo + b.lo;
    vecdd_two_sum_quick(r, e, &r_f, &e_f);
    return vecdd_make(r_f, e_f);
}

static inline VECDD vecdd_sub(VECDD a, VECDD b) {
    VEC r, e, r_f, e_f;
    vecdd_two_difference(a.hi, b.hi, &r, &e);
    e = e + a.lo - b.lo;
    vecdd_two_sum_quick(r, e, &r_f, &e_f);
    return vecdd_make(r_f, e_f);
}

static inline VECDD vecdd_mul(VECDD a, VECDD b) {
    VEC r, e, r_f, e_f;
    vecdd_two_product(a.hi, b.hi, &r, &e);
    e = e + a.hi * b.lo + a.lo * b.hi;
    vecdd_two_sum_quick(r, e, &r_f, &e_f);
    return vecdd_make(r_f, e_f);
}

static inline VECDD vecdd_mul_scalar(VECDD a, dd_t s) { return vecdd_mul(a, vecdd_splat(s)); }

static inline VECDD vecdd_div(VECDD a, VECDD b) {
    VEC q1 = a.hi / b.hi;
    VECDD q1_dd = vecdd_make(q1, (VEC){});
    VECDD r = vecdd_sub(a, vecdd_mul(b, q1_dd));
    VEC q2 = r.hi / b.hi;
    return vecdd_add(q1_dd, vecdd_make(q2, (VEC){}));
}

/* -------------------------------------------------------------------------
 * Vectorized sqrt / exp helpers  (float and double only, not dd_t)
 *
 * sqrt lowers to a single vsqrtps / vsqrtpd.  exp is the library's own Remez
 * polynomial instead of the platform's vector exponential: glibc would provide
 * libmvec, which musl, macOS and the MSVC CRT do not, and without it the same
 * lane loop silently degrades to scalar calls.  See nanofft_vecmath.h and
 * nanofft_exp_coeffs.h; both precisions are accurate to 1 ulp over the whole
 * finite range, denormals and saturation included.
 * ------------------------------------------------------------------------- */
#include <nanofft_vecmath.h>

static inline VECDD vecdd_sqrt(VECDD a) {
    VEC c = vec_sqrt(a.hi);
    VEC u, uu;
    vecdd_two_product(c, c, &u, &uu);
    VEC half = (VEC){} + 0.5;
    VEC cc = (a.hi - u - uu + a.lo) * half / c;
    VEC y = c + cc;
    VEC yy = c - y + cc;
    for (int lane = 0; lane < VECDD_LEN; lane++) {
        if (!(a.hi[lane] > 0.0)) {
            y[lane] = 0.0;
            yy[lane] = 0.0;
        }
    }
    return vecdd_make(y, yy);
}

typedef struct nanofft_plan nanofft_plan;
TLS_API nanofft_plan *nanofft_make_plan(uint32_t N);
TLS_API void nanofft_destroy_plan(nanofft_plan *plan);
TLS_API void nanofft_execute(nanofft_plan *plan, double *real_signal, double *imag_signal);
TLS_API void nanofft_execute_raw(nanofft_plan *plan, double *real_signal, double *imag_signal);
TLS_API void nanofft_permute(nanofft_plan *plan, double *real_signal, double *imag_signal);
TLS_API void nanofft_permute_1d(nanofft_plan *plan, double *values);
TLS_API nanofft_plan *nanofft_make_plan_breadth(uint32_t N);
TLS_API void nanofft_execute_breadth(nanofft_plan *plan, VEC *real_signal, VEC *imag_signal);
TLS_API void nanofft_execute_raw_breadth(nanofft_plan *plan, VEC *real_signal, VEC *imag_signal);
TLS_API void nanofft_permute_breadth(nanofft_plan *plan, VEC *real_signal, VEC *imag_signal);
TLS_API void nanofft_permute_1d_breadth(nanofft_plan *plan, VEC *values);

/* -------------------------------------------------------------------------
 * Double-Double Precision API (nanofftdd)
 * ------------------------------------------------------------------------- */
typedef struct nanofftdd_plan nanofftdd_plan;
TLS_API nanofftdd_plan *nanofftdd_make_plan(uint32_t N);
TLS_API void nanofftdd_destroy_plan(nanofftdd_plan *plan);
TLS_API void nanofftdd_execute(nanofftdd_plan *plan, dd_t *real_signal, dd_t *imag_signal);
TLS_API void nanofftdd_execute_raw(nanofftdd_plan *plan, dd_t *real_signal, dd_t *imag_signal);
TLS_API void nanofftdd_permute(nanofftdd_plan *plan, dd_t *real_signal, dd_t *imag_signal);
TLS_API void nanofftdd_permute_1d(nanofftdd_plan *plan, dd_t *values);
TLS_API nanofftdd_plan *nanofftdd_make_plan_breadth(uint32_t N);
TLS_API void nanofftdd_execute_breadth(nanofftdd_plan *plan, VECDD *real_signal, VECDD *imag_signal);
TLS_API void nanofftdd_execute_raw_breadth(nanofftdd_plan *plan, VECDD *real_signal, VECDD *imag_signal);
TLS_API void nanofftdd_permute_breadth(nanofftdd_plan *plan, VECDD *real_signal, VECDD *imag_signal);
TLS_API void nanofftdd_permute_1d_breadth(nanofftdd_plan *plan, VECDD *values);

#endif /* NANOFFT_H */
