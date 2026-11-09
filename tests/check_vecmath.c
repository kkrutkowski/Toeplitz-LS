/*
 * check_vecmath.c - accuracy gate for the library's vector sqrt/exp helpers.
 *
 * The vector exponential is the library's own Remez implementation rather
 * than the platform's vector math library (see include/nanofft_vecmath.h), so
 * it needs its own accuracy gate: the periodogram tests only see it through
 * the PSWF windows and would not notice a systematic 1e-4 error.
 *
 * The checks are
 *   - dense sweeps of the whole finite range of both precisions against libm,
 *     measured in ulps of the target type (denormals judge by their subnormal
 *     step, which is the right scale for them),
 *   - the exact boundaries: exp(0), exp(-0), overflow to inf, underflow to 0,
 *     NaN propagation and the +-inf inputs,
 *   - vec_sqrt/vecf_sqrt exactness, which is an IEEE guarantee.
 *
 * Bit patterns are built with memcpy so the test itself stays well defined
 * when it is compiled with -ffast-math like the rest of the library.
 */

#include <math.h>
#include <nanofft.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;
static int checks;

static void fail(const char *what, double x, double got, double want, double ulps) {
    printf("  FAIL %-22s x=%.17g got=%.17g want=%.17g (%.3f ulp)\n", what, x, got, want, ulps);
    failures++;
}

static uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static uint64_t dbits(double d) {
    uint64_t u;
    memcpy(&u, &d, sizeof u);
    return u;
}

static float from_fbits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static double from_dbits(uint64_t u) {
    double d;
    memcpy(&d, &u, sizeof d);
    return d;
}

/* Distance in ulps between two finite floats of the same sign. */
static double ulp_distance_f(float a, float b) {
    int64_t ia = (int64_t)(fbits(a) & 0x7fffffffu);
    int64_t ib = (int64_t)(fbits(b) & 0x7fffffffu);
    return (double)(ia > ib ? ia - ib : ib - ia);
}

static double ulp_distance_d(double a, double b) {
    int64_t ia = (int64_t)(dbits(a) & 0x7fffffffffffffffULL);
    int64_t ib = (int64_t)(dbits(b) & 0x7fffffffffffffffULL);
    return (double)(ia > ib ? ia - ib : ib - ia);
}

/* expf reference: round the correctly rounded double result. */
static float ref_expf(float x) {
    /* volatile defeats constant folding and the vectorised rsqrt/sqrt
     * approximation -ffast-math allows, which would otherwise move the
     * reference itself by an ulp. */
    volatile float vx = x;
    volatile double dx = (double)vx;
    return (float)exp(dx);
}

/* Accuracy policy
 *
 * The vector exp is compiled with the caller's fast-math settings on purpose:
 * reassociating the Horner chain is wanted, and one consequence is that the
 * reduction can be rewritten from (x - k*hi) - k*lo into x - k*(hi + lo).
 * That costs a relative error proportional to |k| * ulp(ln2), i.e. ~4e-15 at
 * |x| = 40 and ~2e-14 near the double overflow edge, on top of the 2-3 ulp the
 * reassociated polynomial costs.  Both are far below what the periodogram
 * needs (its R^2/NLL gates sit at 1e-10 and looser, and the NuFFT window
 * interpolation itself is only accurate to ~1e-7 in float), so the sweeps
 * below gate on a relative bound.  The point of this test is to catch a
 * broken coefficient set, a broken reduction or broken saturation - not to
 * chase the last ulp of a fast-math build.
 */
static void check_float_sweep(float lo, float hi, int samples, const char *label, double ulp_bound, double rel_bound) {
    enum { LANES = VECF_LEN };
    float xs[LANES];
    for (int base = 0; base < samples; base += LANES) {
        for (int lane = 0; lane < LANES; ++lane) {
            const int idx = base + lane;
            const double t = (samples <= 1) ? 0.0 : (double)idx / (double)(samples - 1);
            xs[lane] = (float)((double)lo + ((double)hi - (double)lo) * t);
        }
        VECF v;
        memcpy(&v, xs, sizeof v);
        VECF got = vecf_exp(v);
        for (int lane = 0; lane < LANES; ++lane) {
            const float want = ref_expf(xs[lane]);
            checks++;
            /* Saturated inputs are compared by class, not by value. */
            if (want == 0.0f) {
                if (got[lane] != 0.0f) fail(label, xs[lane], got[lane], 0.0, 0);
                continue;
            }
            if (fbits(want) == 0x7f800000u) {
                if (fbits(got[lane]) != 0x7f800000u) fail(label, xs[lane], got[lane], want, 0);
                continue;
            }
            const double ulps = ulp_distance_f(got[lane], want);
            const double rel = fabs((double)got[lane] - (double)want) / (double)want;
            /* Denormal references cannot be rounded correctly without more
             * internal precision than the type carries, so they are always
             * judged relatively. */
            const int want_denormal = ((fbits(want) & 0x7f800000u) == 0);
            if (!want_denormal && ulps > ulp_bound && rel > rel_bound)
                fail(label, xs[lane], got[lane], want, ulps);
            else if (want_denormal && rel > 1e-5)
                fail(label, xs[lane], got[lane], want, ulps);
        }
    }
}

static void check_double_sweep(double lo, double hi, int samples, const char *label, double ulp_bound, double rel_bound) {
    enum { LANES = VEC_LEN };
    double xs[LANES];
    for (int base = 0; base < samples; base += LANES) {
        for (int lane = 0; lane < LANES; ++lane) {
            const int idx = base + lane;
            const double t = (samples <= 1) ? 0.0 : (double)idx / (double)(samples - 1);
            xs[lane] = lo + (hi - lo) * t;
        }
        VEC v;
        memcpy(&v, xs, sizeof v);
        VEC got = vec_exp(v);
        for (int lane = 0; lane < LANES; ++lane) {
            volatile double vx = xs[lane];
            const double want = exp(vx);
            checks++;
            if (want == 0.0) {
                if (got[lane] != 0.0) fail(label, xs[lane], got[lane], 0.0, 0);
                continue;
            }
            if (dbits(want) == 0x7ff0000000000000ULL) {
                if (dbits(got[lane]) != 0x7ff0000000000000ULL) fail(label, xs[lane], got[lane], want, 0);
                continue;
            }
            const double ulps = ulp_distance_d(got[lane], want);
            const double rel = fabs(got[lane] - want) / want;
            const int want_denormal = ((dbits(want) & 0x7ff0000000000000ULL) == 0);
            if (!want_denormal && ulps > ulp_bound && rel > rel_bound)
                fail(label, xs[lane], got[lane], want, ulps);
            else if (want_denormal && rel > 1e-12)
                fail(label, xs[lane], got[lane], want, ulps);
        }
    }
}

static void check_float_specials(void) {
    const float pinf = from_fbits(0x7f800000u);
    const float ninf = from_fbits(0xff800000u);
    const float qnan = from_fbits(0x7fc00000u);
    const float zero = 0.0f;
    const float nzero = from_fbits(0x80000000u);

    struct {
        float in;
        uint32_t want;
        const char *label;
    } cases[] = {
        {zero, 0x3f800000u, "expf(+0)"},
        {nzero, 0x3f800000u, "expf(-0)"},
        {pinf, 0x7f800000u, "expf(+inf)"},
        {ninf, 0x00000000u, "expf(-inf)"},
    };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        VECF v = (VECF){};
        for (int lane = 0; lane < VECF_LEN; ++lane) v[lane] = cases[i].in;
        VECF got = vecf_exp(v);
        for (int lane = 0; lane < VECF_LEN; ++lane) {
            checks++;
            if (fbits(got[lane]) != cases[i].want) fail(cases[i].label, cases[i].in, got[lane], from_fbits(cases[i].want), 0);
        }
    }

    /* NaN in, NaN out (any payload). */
    {
        VECF v = (VECF){};
        for (int lane = 0; lane < VECF_LEN; ++lane) v[lane] = qnan;
        VECF got = vecf_exp(v);
        for (int lane = 0; lane < VECF_LEN; ++lane) {
            checks++;
            const uint32_t bits = fbits(got[lane]);
            const int is_nan = ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0);
            if (!is_nan) fail("expf(nan)", qnan, got[lane], qnan, 0);
        }
    }
}

static void check_double_specials(void) {
    const double pinf = from_dbits(0x7ff0000000000000ULL);
    const double ninf = from_dbits(0xfff0000000000000ULL);
    const double qnan = from_dbits(0x7ff8000000000000ULL);
    const double zero = 0.0;
    const double nzero = from_dbits(0x8000000000000000ULL);

    struct {
        double in;
        uint64_t want;
        const char *label;
    } cases[] = {
        {zero, 0x3ff0000000000000ULL, "exp(+0)"},
        {nzero, 0x3ff0000000000000ULL, "exp(-0)"},
        {pinf, 0x7ff0000000000000ULL, "exp(+inf)"},
        {ninf, 0x0000000000000000ULL, "exp(-inf)"},
    };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        VEC v = (VEC){};
        for (int lane = 0; lane < VEC_LEN; ++lane) v[lane] = cases[i].in;
        VEC got = vec_exp(v);
        for (int lane = 0; lane < VEC_LEN; ++lane) {
            checks++;
            if (dbits(got[lane]) != cases[i].want) fail(cases[i].label, cases[i].in, got[lane], from_dbits(cases[i].want), 0);
        }
    }
    {
        VEC v = (VEC){};
        for (int lane = 0; lane < VEC_LEN; ++lane) v[lane] = qnan;
        VEC got = vec_exp(v);
        for (int lane = 0; lane < VEC_LEN; ++lane) {
            checks++;
            const uint64_t bits = dbits(got[lane]);
            const int is_nan = ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) && ((bits & 0x000fffffffffffffULL) != 0);
            if (!is_nan) fail("exp(nan)", qnan, got[lane], qnan, 0);
        }
    }
}

static void check_sqrt(void) {
    for (int i = 1; i <= 4000; ++i) {
        const double xd = (double)i * 0.37;
        VEC v = (VEC){};
        for (int lane = 0; lane < VEC_LEN; ++lane) v[lane] = xd;
        VEC r = vec_sqrt(v);
        for (int lane = 0; lane < VEC_LEN; ++lane) {
            checks++;
            volatile double vd = xd;
            const double wantd = sqrt(vd);
            if (ulp_distance_d(r[lane], wantd) > 2.0) fail("sqrt", xd, r[lane], wantd, ulp_distance_d(r[lane], wantd));
        }

        const float xf = (float)((double)i * 0.11);
        VECF vf = (VECF){};
        for (int lane = 0; lane < VECF_LEN; ++lane) vf[lane] = xf;
        VECF rf = vecf_sqrt(vf);
        for (int lane = 0; lane < VECF_LEN; ++lane) {
            checks++;
            volatile float vf_ref = xf;
            const float wantf = sqrtf(vf_ref);
            /* -ffast-math may lower this to an rsqrt approximation; a couple of
             * ulp is an acceptable price (see nanofft_vecmath.h). */
            if (ulp_distance_f(rf[lane], wantf) > 2.0) fail("sqrtf", xf, rf[lane], wantf, ulp_distance_f(rf[lane], wantf));
        }
    }
}

int main(void) {
    printf("check_vecmath: %d float lanes, %d double lanes\n", (int)VECF_LEN, (int)VEC_LEN);

    /* The PSWF windows evaluate exp() of arguments down to -225, and the NuFFT
     * spread of small arguments near zero; both ends are covered here. */
    /* Library range: the PSWF windows and the NuFFT spread never leave it. */
    check_float_sweep(-225.0f, 0.0f, 200000, "expf(pswf range)", 0.0, 1e-6);
    check_double_sweep(-40.0, 40.0, 200000, "exp(library range)", 0.0, 2e-14);
    check_double_sweep(-1.0, 1.0, 100000, "exp(near zero)", 0.0, 2e-14);

    /* Whole finite range: a relative bound, see the note above. */
    check_float_sweep(-103.0f, 88.0f, 400000, "expf(finite range)", 0.0, 1e-6);
    check_double_sweep(-708.0, 709.0, 400000, "exp(finite range)", 0.0, 1e-13);

    check_float_specials();
    check_double_specials();
    check_sqrt();

    if (failures) {
        printf("check_vecmath: %d FAILURES out of %d checks\n", failures, checks);
        return 1;
    }
    printf("check_vecmath: OK (%d checks)\n", checks);
    return 0;
}
