#ifndef NANOFFT_H
#define NANOFFT_H

#include <math.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * SIMD vector width selection
 * ------------------------------------------------------------------------- */
#ifndef VEC_BYTES
#    ifdef __AVX512F__
#        define VEC_BYTES 64
#    elif defined(__AVX__)
#        define VEC_BYTES 32
#    else
#        define VEC_BYTES 16
#    endif
#endif

/* -------------------------------------------------------------------------
 * Fast 2π-periodic trig approximations (Single & Double)
 * ------------------------------------------------------------------------- */

static inline float sin2pif(float x) {
    float f = x - (float)((int)x);
    if (f < 0.0f) f += 1.0f;
    float sign = 1.0f;
    if (f >= 0.5f) {
        sign = -1.0f;
        f -= 0.5f;
    }
    if (f > 0.25f) f = 0.5f - f;
    float f2 = f * f;
    float p = f2 * 39.536706065730207835108712734262f - 76.549782293595742666226937116116f;
    p = p * f2 + 81.601004073261773523492199897936f;
    p = p * f2 - 41.341655031416278077153126232486f;
    p = p * f2 + 6.2831851600894774430188071795666f;
    p = p * f;
    return p * sign;
}

static inline float cos2pif(float x) {
    float f = x - (float)((int)x);
    if (f < 0.0f) f += 1.0f;
    if (f > 0.5f) f = 1.0f - f;
    float sign = 1.0f;
    if (f > 0.25f) {
        sign = -1.0f;
        f = 0.5f - f;
    }
    float f2 = f * f;
    float p = f2 * 56.242380464873243259663276802701f - 85.240330322699427859509454517828f;
    p = p * f2 + 64.934590626780991246193352727536f;
    p = p * f2 - 19.739171434702393618770795066531f;
    p = p * f2 + 0.99999995346667013630639784578184f;
    return p * sign;
}

static inline double sin2pi(double x) {
    double f = x - (double)((int)x);
    if (f < 0.0) f += 1.0;
    double sign = 1.0;
    if (f >= 0.5) {
        sign = -1.0;
        f -= 0.5;
    }
    if (f > 0.25) f = 0.5 - f;
    double f2 = f * f;
    double p = f2 * -0.69093588239655204752473229179535744712019258849149 + 3.816997428325180431886496831784263511664844308953;
    p = p * f2 - 15.094471616631987517150792141240583632857536104374;
    p = p * f2 + 42.058688305389948652978636940879319419236423555816;
    p = p * f2 - 76.705859647469529848428028740175591232140822775877;
    p = p * f2 + 81.605249275026290026468387665436788455350446194769;
    p = p * f2 - 41.341702240395082626913754855518082730586045930014;
    p = p * f2 + 6.2831853071795803915351819993098437430793561832783;
    p = p * f;
    return p * sign;
}

static inline double cos2pi(double x) {
    double f = x - (double)((int)x);
    if (f < 0.0) f += 1.0;
    if (f > 0.5) f = 1.0 - f;
    double sign = 1.0;
    if (f > 0.25) {
        sign = -1.0;
        f = 0.5 - f;
    }
    double f2 = f * f;
    double p = f2 * 0.2719476416639800139555005900627203625997526419121 - 1.7132188587562783227021022500030456711383616906867;
    p = p * f2 + 7.9034625375924737942470808816341151858745136793854;
    p = p * f2 - 26.426254069097002482358197039994768718882874895547;
    p = p * f2 + 60.244641313230924425507669180781096172477629015917;
    p = p * f2 - 85.456817205981735856122645737764931252180966287231;
    p = p * f2 + 64.939394022663960757319625985316082107868087096872;
    p = p * f2 - 19.739208802178707095137280041056835739350734071865;
    p = p * f2 + 0.99999999999999999608981951072301324039762563006958;
    return p * sign;
}

/* -------------------------------------------------------------------------
 * Double-Double Precision Core Logic
 * ------------------------------------------------------------------------- */
typedef struct {
    double hi;
    double lo;
} dd_t;

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline dd_t dd_make(double hi,
                                                                                                                                                  double lo) {
    dd_t res = {hi, lo};
    return res;
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline void dd_two_sum(
    double a, double b, double *r, double *e) {
    *r = a + b;
    double v = *r - a;
    *e = (a - (*r - v)) + (b - v);
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline void dd_two_sum_quick(
    double x, double y, double *r, double *e) {
    *r = x + y;
    *e = y - (*r - x);
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline void dd_two_difference(
    double x, double y, double *r, double *e) {
    *r = x - y;
    double t = *r - x;
    *e = (x - (*r - t)) - (y + t);
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline void dd_two_product(
    double x, double y, double *r, double *e) {
    *r = x * y;
    *e = fma(x, y, -*r);
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline dd_t dd_add(dd_t a,
                                                                                                                                                 dd_t b) {
    double r, e, r_f, e_f;
    dd_two_sum(a.hi, b.hi, &r, &e);
    e = e + a.lo + b.lo;
    dd_two_sum_quick(r, e, &r_f, &e_f);
    return dd_make(r_f, e_f);
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline dd_t dd_sub(dd_t a,
                                                                                                                                                 dd_t b) {
    double r, e, r_f, e_f;
    dd_two_difference(a.hi, b.hi, &r, &e);
    e = e + a.lo - b.lo;
    dd_two_sum_quick(r, e, &r_f, &e_f);
    return dd_make(r_f, e_f);
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline dd_t dd_mul(dd_t a,
                                                                                                                                                 dd_t b) {
    double r, e, r_f, e_f;
    dd_two_product(a.hi, b.hi, &r, &e);
    e = e + a.hi * b.lo + a.lo * b.hi;
    dd_two_sum_quick(r, e, &r_f, &e_f);
    return dd_make(r_f, e_f);
}

static inline double dd_to_double(dd_t a) { return a.hi + a.lo; }

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline dd_t dd_div(dd_t a,
                                                                                                                                                 dd_t b) {
    double q1 = a.hi / b.hi;
    dd_t q1_dd = dd_make(q1, 0.0);
    dd_t r = dd_sub(a, dd_mul(b, q1_dd));
    double q2 = r.hi / b.hi;
    return dd_add(q1_dd, dd_make(q2, 0.0));
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline dd_t dd_sqrt(dd_t a) {
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

static inline dd_t cos2pidd(dd_t x) {
    // 1. Truncate argument modulo 1 using dd_t math
    double int_part = (double)((int64_t)x.hi);
    dd_t f = dd_sub(x, dd_make(int_part, 0.0));

    if (f.hi < 0.0) f = dd_add(f, dd_make(1.0, 0.0));
    if (f.hi > 0.5) f = dd_sub(dd_make(1.0, 0.0), f);

    double sign = 1.0;
    if (f.hi > 0.25) {
        sign = -1.0;
        f = dd_sub(dd_make(0.5, 0.0), f);
    }

    dd_t f2 = dd_mul(f, f);
    dd_t p;

    p = ((dd_t){-1.3712976302871458e-06, -1.4907449409481538e-23});
    p = dd_add(dd_mul(p, f2), ((dd_t){2.3093788063462385e-05, 4.5435375960601516e-22}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-0.00032299035911159625, -8.9810661988144363e-21}));
    p = dd_add(dd_mul(p, f2), ((dd_t){0.0037798341474950792, -1.9249220349568919e-19}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-0.036382841139801829, 2.9056792650806832e-18}));
    p = dd_add(dd_mul(p, f2), ((dd_t){0.28200596845569187, 1.5852734583076034e-17}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-1.7143907110886696, 5.0670981029021671e-17}));
    p = dd_add(dd_mul(p, f2), ((dd_t){7.9035363713184692, -4.4106464905449524e-16}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-26.426256783374399, 1.1988182697337608e-15}));
    p = dd_add(dd_mul(p, f2), ((dd_t){60.244641371876661, -3.2213071685554461e-16}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-85.456817206693728, 2.0361754233578647e-16}));
    p = dd_add(dd_mul(p, f2), ((dd_t){64.939394022668296, -4.2563201319361759e-15}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-19.739208802178716, -1.2530591017478957e-15}));
    p = dd_add(dd_mul(p, f2), ((dd_t){1, -7.4159205569012457e-33}));

    return (sign < 0.0) ? dd_make(-p.hi, -p.lo) : p;
}

static inline dd_t sin2pidd(dd_t x) {
    // 1. Truncate argument modulo 1 using dd_t math
    double int_part = (double)((int64_t)x.hi);
    dd_t f = dd_sub(x, dd_make(int_part, 0.0));

    if (f.hi < 0.0) f = dd_add(f, dd_make(1.0, 0.0));

    double sign = 1.0;
    if (f.hi >= 0.5) {
        sign = -1.0;
        f = dd_sub(f, dd_make(0.5, 0.0));
    }
    if (f.hi > 0.25) {
        f = dd_sub(dd_make(0.5, 0.0), f);
    }

    if (sign < 0.0) {
        f = dd_make(-f.hi, -f.lo);
    }
    sign = 1.0;

    dd_t f2 = dd_mul(f, f);
    dd_t p;

    p = ((dd_t){-3.1937621498346393e-07, -4.2494386355335374e-24});
    p = dd_add(dd_mul(p, f2), ((dd_t){5.8042120014329398e-06, -3.9949772624199443e-22}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-8.8235163179284102e-05, 9.9742533594686292e-22}));
    p = dd_add(dd_mul(p, f2), ((dd_t){0.0011309237346037141, 9.0689538417711371e-20}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-0.012031585941374802, 3.5320565199955664e-19}));
    p = dd_add(dd_mul(p, f2), ((dd_t){0.10422916220811097, 3.2446843930090187e-18}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-0.71812230177849967, -4.3916880697967638e-17}));
    p = dd_add(dd_mul(p, f2), ((dd_t){3.819952584848282, 7.214503754465747e-17}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-15.09464257682299, -6.8142151350129174e-16}));
    p = dd_add(dd_mul(p, f2), ((dd_t){42.058693944897655, -1.9698158197627135e-15}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-76.705859753061389, 3.6416334350664191e-15}));
    p = dd_add(dd_mul(p, f2), ((dd_t){81.605249276075057, -2.5379220305392744e-15}));
    p = dd_add(dd_mul(p, f2), ((dd_t){-41.341702240399762, 1.8132498260632083e-15}));
    p = dd_add(dd_mul(p, f2), ((dd_t){6.2831853071795862, 2.4492935982947059e-16}));

    p = dd_mul(p, f);
    return dd_make(p.hi * sign, p.lo * sign);
}

/* -------------------------------------------------------------------------
 * Single Precision API (nanofftf)
 * ------------------------------------------------------------------------- */
#define VECF_LEN (VEC_BYTES / 4)
typedef float VECF __attribute__((vector_size(VEC_BYTES)));
typedef uint32_t VECF_INT __attribute__((vector_size(VEC_BYTES)));

typedef struct nanofftf_plan nanofftf_plan;
nanofftf_plan *nanofftf_make_plan(uint32_t N);
void nanofftf_destroy_plan(nanofftf_plan *plan);
void nanofftf_execute(nanofftf_plan *plan, float *real_signal, float *imag_signal);

/* -------------------------------------------------------------------------
 * Double Precision API (nanofft)
 * ------------------------------------------------------------------------- */
#define VEC_LEN (VEC_BYTES / 8)
typedef double VEC __attribute__((vector_size(VEC_BYTES)));
typedef uint64_t VEC_INT __attribute__((vector_size(VEC_BYTES)));

/* -------------------------------------------------------------------------
 * Vectorized exp / sqrt helpers  (float and double only, not dd_t)
 *
 * Each function loops over a compile-time-constant lane count so the
 * auto-vectorizer can lower the loop to:
 *   sqrt  -> vsqrtps / vsqrtpd  (IEEE-exact; always vectorized at -O2)
 *   exp   -> _ZGVdN*v_exp / _ZGVsN*v_expf  (libmvec, glibc >= 2.22)
 *            or SVML equivalents when using icc/icx.
 *
 * GCC vector-extension types support subscript access v[i] directly, so
 * no union tricks or pointer casts are needed.
 * ------------------------------------------------------------------------- */

static inline VECF vecf_sqrt(VECF v) {
    VECF r;
    for (int i = 0; i < VECF_LEN; i++) r[i] = sqrtf(v[i]);
    return r;
}

static inline VECF vecf_exp(VECF v) {
    VECF r;
    for (int i = 0; i < VECF_LEN; i++) r[i] = expf(v[i]);
    return r;
}

static inline VEC vec_sqrt(VEC v) {
    VEC r;
    for (int i = 0; i < VEC_LEN; i++) r[i] = sqrt(v[i]);
    return r;
}

static inline VEC vec_exp(VEC v) {
    VEC r;
    for (int i = 0; i < VEC_LEN; i++) r[i] = exp(v[i]);
    return r;
}

typedef struct nanofft_plan nanofft_plan;
nanofft_plan *nanofft_make_plan(uint32_t N);
void nanofft_destroy_plan(nanofft_plan *plan);
void nanofft_execute(nanofft_plan *plan, double *real_signal, double *imag_signal);

/* -------------------------------------------------------------------------
 * Double-Double Precision API (nanofftdd)
 * ------------------------------------------------------------------------- */
typedef struct nanofftdd_plan nanofftdd_plan;
nanofftdd_plan *nanofftdd_make_plan(uint32_t N);
void nanofftdd_destroy_plan(nanofftdd_plan *plan);
void nanofftdd_execute(nanofftdd_plan *plan, dd_t *real_signal, dd_t *imag_signal);

#endif /* NANOFFT_H */
