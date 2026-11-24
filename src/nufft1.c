/* nufft1.c – Non-Uniform FFT Type-1 implementation */

#include <float.h>
#include <math.h>
#include <nufft1.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tls.h>
#include <tls_internal.h>
#include <tls_tuning.h>

/* =========================================================================
 * Common Helpers
 * ========================================================================= */

#if !defined(HALF)
static const int pswf_degrees[19] = {0, 0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 14, 14, 16, 18, 18, 20, 20, 23};
#endif
enum { PSWF_MAX_W = 36 };

static inline size_t sz_max(size_t a, size_t b) { return (a > b) ? a : b; }

static inline int bitceil(unsigned int N) {
    if (N <= 1) return 1;
    unsigned int v = N - 1U;
    int shift = 0;
    while (v) {
        v >>= 1;
        shift++;
    }
    return 1 << shift;
}

static int next_fast_len(int N) { return bitceil((unsigned int)N); }

#ifdef DOUBLE_DOUBLE
typedef dd_t nufft_input_t;
#else
typedef double nufft_input_t;
#endif

static void get_lra_params(const nufft_input_t *x, int Mpoints, int Nfft, int *Tidx) {
    for (int i = 0; i < Mpoints; ++i) {
#ifdef DOUBLE_DOUBLE
        dd_t z = dd_mul(dd_make((double)Nfft, 0.0), x[i]);
#else
        dd_t z = dd_mul(dd_make((double)Nfft, 0.0), dd_make(x[i], 0.0));
#endif
        long long nearest = (long long)floor(dd_to_double(z) + 0.5);
        int idx = (int)(nearest % (long long)Nfft);
        if (idx < 0) idx += Nfft;
        Tidx[i] = idx;
    }
}

/* =========================================================================
 * Precision-Dependent Implementation
 * ========================================================================= */

#define NANOFFT_NEEDS_INTERNAL_VEC
#include "nanofft_precision.h"
#include "utils.h"

#ifdef DOUBLE_DOUBLE
#    define PFX(name) tlsdd_nufft_##name
#    define NANOFFT_PLAN nanofftdd_plan
#    define NANOFFT_MAKE_PLAN nanofftdd_make_plan
#    define NANOFFT_EXECUTE nanofftdd_execute
#    define NANOFFT_EXECUTE_RAW nanofftdd_execute_raw
#    define NANOFFT_PERMUTE_1D nanofftdd_permute_1d
#    define NANOFFT_MAKE_PLAN_BREADTH nanofftdd_make_plan_breadth
#    define NANOFFT_EXECUTE_BREADTH nanofftdd_execute_breadth
#    define NANOFFT_EXECUTE_RAW_BREADTH nanofftdd_execute_raw_breadth
#    define NANOFFT_PERMUTE_1D_BREADTH nanofftdd_permute_1d_breadth
#    define NANOFFT_DESTROY_PLAN nanofftdd_destroy_plan
#    define LRA_BESSEL_TERMS 14
#    define GL_NEWTON_TOL FCONST(5e-32, 0)
#    define PSWF_KERNEL_MAX_EXP 26 /* (p_sig + 2) / 4; future FP16 tlsh (w <= 5) uses 3 */
#elif defined(DOUBLE)
#    define PFX(name) tls_nufft_##name
#    define NANOFFT_PLAN nanofft_plan
#    define NANOFFT_MAKE_PLAN nanofft_make_plan
#    define NANOFFT_EXECUTE nanofft_execute
#    define NANOFFT_EXECUTE_RAW nanofft_execute_raw
#    define NANOFFT_PERMUTE_1D nanofft_permute_1d
#    define NANOFFT_MAKE_PLAN_BREADTH nanofft_make_plan_breadth
#    define NANOFFT_EXECUTE_BREADTH nanofft_execute_breadth
#    define NANOFFT_EXECUTE_RAW_BREADTH nanofft_execute_raw_breadth
#    define NANOFFT_PERMUTE_1D_BREADTH nanofft_permute_1d_breadth
#    define NANOFFT_DESTROY_PLAN nanofft_destroy_plan
#    define LRA_BESSEL_TERMS 8
#    define GL_NEWTON_TOL FCONST(2.0e-16)
#    define PSWF_KERNEL_MAX_EXP 13
#elif defined(HALF)
#    ifdef HALF_NATIVE_EMBED
#        define PFX(name) tlsh_nat_nufft_##name
#        define NANOFFT_PLAN nanoffth_nat_plan
#        define NANOFFT_MAKE_PLAN nanoffth_nat_make_plan
#        define NANOFFT_EXECUTE nanoffth_nat_execute
#        define NANOFFT_EXECUTE_RAW nanoffth_nat_execute_raw
#        define NANOFFT_PERMUTE_1D nanoffth_nat_permute_1d
#        define NANOFFT_MAKE_PLAN_BREADTH nanoffth_nat_make_plan_breadth
#        define NANOFFT_EXECUTE_BREADTH nanoffth_nat_execute_breadth
#        define NANOFFT_EXECUTE_RAW_BREADTH nanoffth_nat_execute_raw_breadth
#        define NANOFFT_PERMUTE_1D_BREADTH nanoffth_nat_permute_1d_breadth
#        define NANOFFT_DESTROY_PLAN nanoffth_nat_destroy_plan
#    else
#        define PFX(name) tlsh_nufft_##name
#        define NANOFFT_PLAN nanoffth_plan
#        define NANOFFT_MAKE_PLAN nanoffth_make_plan
#        define NANOFFT_EXECUTE nanoffth_execute
#        define NANOFFT_EXECUTE_RAW nanoffth_execute_raw
#        define NANOFFT_PERMUTE_1D nanoffth_permute_1d
#        define NANOFFT_MAKE_PLAN_BREADTH nanoffth_make_plan_breadth
#        define NANOFFT_EXECUTE_BREADTH nanoffth_execute_breadth
#        define NANOFFT_EXECUTE_RAW_BREADTH nanoffth_execute_raw_breadth
#        define NANOFFT_PERMUTE_1D_BREADTH nanoffth_permute_1d_breadth
#        define NANOFFT_DESTROY_PLAN nanoffth_destroy_plan
#    endif
#    define LRA_BESSEL_TERMS 4
#    define GL_NEWTON_TOL FCONST(1.0e-4)
#    define PSWF_KERNEL_MAX_EXP 3
#else
#    define PFX(name) tlsf_nufft_##name
#    define NANOFFT_PLAN nanofftf_plan
#    define NANOFFT_MAKE_PLAN nanofftf_make_plan
#    define NANOFFT_EXECUTE nanofftf_execute
#    define NANOFFT_EXECUTE_RAW nanofftf_execute_raw
#    define NANOFFT_PERMUTE_1D nanofftf_permute_1d
#    define NANOFFT_MAKE_PLAN_BREADTH nanofftf_make_plan_breadth
#    define NANOFFT_EXECUTE_BREADTH nanofftf_execute_breadth
#    define NANOFFT_EXECUTE_RAW_BREADTH nanofftf_execute_raw_breadth
#    define NANOFFT_PERMUTE_1D_BREADTH nanofftf_permute_1d_breadth
#    define NANOFFT_DESTROY_PLAN nanofftf_destroy_plan
#    define LRA_BESSEL_TERMS 6
#    define GL_NEWTON_TOL FCONST(1.0e-7)
#    define PSWF_KERNEL_MAX_EXP 6
#endif

/* =========================================================================
 * Gauss-Legendre Quadrature
 * ========================================================================= */

static void PFX(get_lege_roots)(int n, FLOAT *roots, FLOAT *weights) {
    int half_n = (n + 1) / 2;
#if defined(HALF)
    for (int i = 1; i <= half_n; i++) {
        double x = cos2pi(0.5 * ((double)i - 0.25) / ((double)n + 0.5));
        double p1 = 0.0, p2 = 0.0, dp = 0.0, step = 0.0;
        int iter = 0;
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 1; j <= n; j++) {
                double p0 = ((double)(2 * j - 1) * x * p1 - (double)(j - 1) * p2) / (double)j;
                p2 = p1;
                p1 = p0;
            }
            dp = (double)n * (x * p1 - p2) / (x * x - 1.0);
            step = p1 / dp;
            x -= step;
        } while (fabs(step) > 1.0e-12 && ++iter < 64);

        FLOAT rx = FCAST(x);
        FLOAT rw = FCAST(2.0 / ((1.0 - x * x) * dp * dp));
        roots[i - 1] = rx;
        weights[i - 1] = rw;
        roots[n - i] = NEG(rx);
        weights[n - i] = rw;
    }
#else
    for (int i = 1; i <= half_n; i++) {
        FLOAT x = M_COS2PI(DIV(MUL(FCAST(0.5), FCAST(i - 0.25)), FCAST(n + 0.5)));
        FLOAT p1 = FCAST(0.0), p2 = FCAST(0.0), dp = FCAST(0.0);
        FLOAT step;
        int iter = 0;
        do {
            p1 = FCAST(1.0);
            p2 = FCAST(0.0);
            for (int j = 1; j <= n; j++) {
                FLOAT p0 = DIV(SUB(MUL(MUL(FCAST(2 * j - 1), x), p1), MUL(FCAST(j - 1), p2)), FCAST(j));
                p2 = p1;
                p1 = p0;
            }
            dp = DIV(MUL(FCAST(n), SUB(MUL(x, p1), p2)), SUB(MUL(x, x), FCAST(1.0)));
            step = DIV(p1, dp);
            x = SUB(x, step);
        } while (M_FABS(step) > TO_DOUBLE(GL_NEWTON_TOL) && ++iter < 64);

        roots[i - 1] = x;
        weights[i - 1] = DIV(FCAST(2.0), MUL(MUL(SUB(FCAST(1.0), MUL(x, x)), dp), dp));

        roots[n - i] = NEG(x);
        weights[n - i] = weights[i - 1];
    }
#endif
}

/* =========================================================================
 * Lookup Tables
 * ========================================================================= */

#if defined(HALF)
static const double PFX(fact_d)[28] = {1.0,
                                       1.0,
                                       2.0,
                                       6.0,
                                       24.0,
                                       120.0,
                                       720.0,
                                       5040.0,
                                       40320.0,
                                       362880.0,
                                       3628800.0,
                                       39916800.0,
                                       479001600.0,
                                       6227020800.0,
                                       87178291200.0,
                                       1307674368000.0,
                                       20922789888000.0,
                                       355687428096000.0,
                                       6402373705728000.0,
                                       1.21645100408832e+17,
                                       2.43290200817664e+18,
                                       5.109094217170944e+19,
                                       1.1240007277776077e+21,
                                       2.5852016738884978e+22,
                                       6.2044840173323941e+23,
                                       1.5511210043330986e+25,
                                       4.0329146112660565e+26,
                                       1.0888869450418352e+28};

/* Hardcoded FP16 PSWF Remez degree-5 LUTs (w = 4 and w = 5 ONLY, generated by
 * src/miscellaneous/pswf_remez_lut.cpp with IRLS L2->L4->L8->L12->L16->Remez). */
static const tls_f16_t PFX(pswf21_h_c)[2] = {
    {0x48b0u}, /* w=4: c=9.37477796 */
    {0x49deu}, /* w=5: c=11.73097245 */
};
static const tls_f16_t PFX(pswf21_h_coeffs)[2][6] = {
    /* w=4: {1.000004, 0.3867161, -0.9216739, -0.7881353, -0.2697224, 0.6903307} */
    {{0x3c00u}, {0x3630u}, {0xbb60u}, {0xba4eu}, {0xb451u}, {0x3986u}},
    /* w=5: {1.000018, 0.3816939, -1.172602, -1.331939, 0.5295519, 0.6331383} */
    {{0x3c00u}, {0x361bu}, {0xbcb1u}, {0xbd54u}, {0x383du}, {0x3911u}},
};
static const tls_f16_t PFX(pswf43_h_c)[2] = {
    {0x47ceu}, /* w=4: c=7.80398163 */
    {0x48e2u}, /* w=5: c=9.76747704 */
};
static const tls_f16_t PFX(pswf43_h_coeffs)[2][6] = {
    /* w=4: {0.9999901, 0.3908943, -0.7396056, -0.5325859, -0.4544860, 0.5283378} */
    {{0x3c00u}, {0x3641u}, {0xb9ebu}, {0xb843u}, {0xb746u}, {0x383au}},
    /* w=5: {1.000006, 0.3856690, -0.9628945, -0.8759618, -0.1640284, 0.6995837} */
    {{0x3c00u}, {0x362cu}, {0xbbb4u}, {0xbb02u}, {0xb140u}, {0x3999u}},
};
#else
static const FLOAT PFX(fact)[28] = {FCONST(1, 0),
                                    FCONST(1, 0),
                                    FCONST(2, 0),
                                    FCONST(6, 0),
                                    FCONST(24, 0),
                                    FCONST(120, 0),
                                    FCONST(720, 0),
                                    FCONST(5040.0, 0.0),
                                    FCONST(40320.0, 0.0),
                                    FCONST(362880.0, 0.0),
                                    FCONST(3628800.0, 0.0),
                                    FCONST(39916800.0, 0.0),
                                    FCONST(479001600.0, 0.0),
                                    FCONST(6227020800.0, 0.0),
                                    FCONST(87178291200.0, 0.0),
                                    FCONST(1307674368000.0, 0.0),
                                    FCONST(20922789888000.0, 0.0),
                                    FCONST(355687428096000.0, 0.0),
                                    FCONST(6402373705728000.0, 0.0),
                                    FCONST(1.21645100408832e+17, 0.0),
                                    FCONST(2.43290200817664e+18, 0.0),
                                    FCONST(5.109094217170944e+19, 0.0),
                                    FCONST(1.1240007277776077e+21, 0.0),
                                    FCONST(2.5852016738884978e+22, -1572864.0),
                                    FCONST(6.2044840173323941e+23, 29360128.0),
                                    FCONST(1.5511210043330986e+25, -71303168.0),
                                    FCONST(4.0329146112660565e+26, -14738784256.0),
                                    FCONST(1.0888869450418352e+28, 220528115712.0)};

static const FLOAT PFX(pswf_coeffs)[19][24] = {
    {FCONST(1, 0)},
    {FCONST(1, 0)},
    {FCONST(1.03, -2.6645352591003756e-17), FCONST(-0.02197, -5.4400928206632669e-20)},
    {FCONST(0.99787999999999999, 1.0764722446765518e-17), FCONST(0.49042999999999998, 2.2932766796657233e-17), FCONST(-1.1474, -2.4513724383723456e-17)},
    {FCONST(0.99919899999999995, 5.1670667744474484e-17), FCONST(0.44010899999999997, 2.7547741865419083e-17),
     FCONST(-1.3426199999999999, -7.5388584264146627e-17), FCONST(-0.086526000000000006, 5.6852300645005012e-18)},
    {FCONST(1.000073, 1.0260237104375846e-17), FCONST(0.3748977, -3.8653524825349451e-19), FCONST(-1.0590809999999999, -6.0879301599925378e-17),
     FCONST(-1.910318, -3.9463543544115964e-17), FCONST(1.602468, -3.5242919693700971e-18)},
    {FCONST(0.99999590000000005, -4.8635229177307336e-17), FCONST(0.38321049000000001, -1.4720882290930604e-17),
     FCONST(-1.5374471999999999, -9.721361493575387e-17), FCONST(-1.0897965999999999, -1.0682867923605954e-16),
     FCONST(-0.83140645000000002, 1.9073809198744129e-17), FCONST(3.5757615, -3.7147174225538038e-17), FCONST(-1.4842199, 3.6399683267518414e-17)},
    {FCONST(0.99999949200000005, -4.5344165755523134e-17), FCONST(0.38132025200000003, -2.6182391366091906e-17),
     FCONST(-1.8112185000000001, 6.4531491261732297e-17), FCONST(-1.56723755, -2.0722268345707562e-17), FCONST(0.445511616, 8.1746520663728006e-19),
     FCONST(2.2872740999999999, 1.3432668310997542e-16), FCONST(0.43926544699999998, 2.4436388912363326e-17), FCONST(-1.17300092, 2.4743940230109728e-18)},
    {FCONST(1.0000001350000001, -6.7719838625635017e-17), FCONST(0.38030810120000003, -2.6054226509586442e-17), FCONST(-2.099454572, -3.6661276681115849e-17),
     FCONST(-1.930650019, -4.9825530368252656e-18), FCONST(1.5911306080000001, -5.7344891501998059e-17), FCONST(0.41027796129999999, 6.3615345879952659e-18),
     FCONST(5.6755215970000004, -3.6224309951649047e-16), FCONST(-7.1979701699999996, -3.8771304389229043e-16),
     FCONST(2.1704596899999999, 5.8465730035095481e-17)},
    {FCONST(0.99999999470000001, -5.5652435548836368e-18), FCONST(0.37972244149000001, -6.680816113657784e-18),
     FCONST(-2.3975013074999998, -2.1025925889261999e-16), FCONST(-2.1330693984, -1.7015190678648649e-17), FCONST(1.5897245363000001, -1.0171841040573782e-16),
     FCONST(3.8108029991999999, 1.2576419976539911e-16), FCONST(-1.6718386273000001, 1.0188291525992099e-16),
     FCONST(4.5085214979000003, -3.4101922210538758e-16), FCONST(-10.932012939, -1.8182072381023319e-16), FCONST(7.1570552787999997, 3.0201936169760302e-16),
     FCONST(-1.3107861639, 2.311921889486257e-17)},
    {FCONST(0.99999999983900001, -8.8832607616495803e-18), FCONST(0.37921831127599998, 2.4759419602560228e-17), FCONST(-2.69234679819, 1.227417669724673e-18),
     FCONST(-2.3974019054300002, 2.0184649656584951e-16), FCONST(2.27827635493, -2.4978653527796267e-17), FCONST(4.1020222531400004, -3.5183426007279195e-16),
     FCONST(1.2214946871100001, -8.8026140474539711e-17), FCONST(-4.9919652597799997, -2.8043066777172499e-16),
     FCONST(3.4089870503899999, 6.0118863984826018e-17), FCONST(-9.3614316377200009, 8.6453317635459821e-16), FCONST(10.4341893843, 2.4119053705362605e-16),
     FCONST(-3.3809296660100001, 1.1195310435141437e-16)},
    {FCONST(0.99999999999669997, 2.6918395815300756e-17), FCONST(0.37881342242100002, -2.3376791432383469e-17),
     FCONST(-2.987312464565, -8.0216886999551202e-18), FCONST(-2.6550782841539999, -6.5397447542636654e-17), FCONST(2.9604461492029999, 1.3406522884906734e-16),
     FCONST(5.2526899189630001, -8.1930658780038356e-17), FCONST(0.42381785619939999, 1.0674628492779448e-17),
     FCONST(-3.9013965922180001, 5.8949201047653329e-17), FCONST(-4.3407684118959997, -2.878421546483878e-16), FCONST(3.397724997868, 2.5767410988919438e-17),
     FCONST(-1.8385776443149999, -7.4032586780958814e-17), FCONST(4.2468839870839998, 1.6594429689575918e-16), FCONST(0.1018023977991, 2.6891101242654257e-18),
     FCONST(-3.3575996868060001, 7.218117752927355e-17), FCONST(1.3185988264670001, -1.0688789461710258e-16)},
    {FCONST(0.99999999999475997, 3.0631625813839491e-17), FCONST(0.37847950466805003, -2.6339187752455472e-17),
     FCONST(-3.2821927063989, 3.9488353286287748e-17), FCONST(-2.9130653113245, -2.6988629542756827e-17), FCONST(3.7367776537493, -4.327325150370598e-17),
     FCONST(6.4439952292656999, 1.0221954944427125e-16), FCONST(-0.028647532639073998, -1.560539333468114e-18),
     FCONST(-5.1014644696004998, -1.8188619651482441e-16), FCONST(-8.0026009042134003, 3.1241470569511876e-16), FCONST(12.40269540836, 3.8185804442036898e-16),
     FCONST(-16.866425128766998, -1.6019222966860981e-15), FCONST(26.760296782967, 1.5590005205012858e-17), FCONST(-18.807606085854001, 1.0656270169420169e-15),
     FCONST(3.8893610645163998, 1.5373077658296097e-16), FCONST(0.39041741028215998, 1.7673466945780092e-17)},
    {FCONST(1.0000000000001099, 8.7920562109502504e-17), FCONST(0.378199407354322, -3.6270895434427074e-18), FCONST(-3.5770139986825802, 1.869940733013209e-16),
     FCONST(-3.17106171609238, -3.2664453901816161e-17), FCONST(4.6015041846580296, 4.2742562189232558e-16),
     FCONST(7.7291034720738301, -8.3388656930765137e-17), FCONST(-0.40639058167962999, -8.0930786953103963e-18),
     FCONST(-8.3642280826021498, -2.1695752220693976e-16), FCONST(-4.7369278522136904, 3.930572771059815e-16),
     FCONST(1.1558161772377999, 5.3446690617420245e-17), FCONST(10.284975780218501, -6.7386822047410537e-16),
     FCONST(-5.4847902338839596, -4.1295322531368585e-16), FCONST(4.3017673654658397, 2.9022946531767954e-16),
     FCONST(-1.2273245543718501, 6.2750581855652856e-17), FCONST(-9.3996972384128608, 7.5582956982543689e-16),
     FCONST(9.5269856761381408, -8.4609202836873005e-16), FCONST(-2.61091383596797, -1.8913712847279385e-18)},
    {FCONST(0.999999999999999, -7.9927783735911361e-19), FCONST(0.3779610866286775, -6.0954108448640911e-19),
     FCONST(-3.8717912205127738, -1.9876775752345565e-16), FCONST(-3.42897991770863, -2.3597210092702879e-17),
     FCONST(5.5519015145967314, -4.2397832634742372e-16), FCONST(9.1508428870126082, -1.7333604791201652e-16),
     FCONST(-1.155990217095924, -2.2447270588600076e-17), FCONST(-11.027460720694281, 7.3803010192932556e-16),
     FCONST(-6.2691197517624264, 4.1794033261248844e-16), FCONST(4.502127102513481, -2.7301622816594316e-17), FCONST(11.05756937060568, -3.478077199542895e-16),
     FCONST(-7.6489629590119899, -1.4055197811103425e-16), FCONST(22.306029568286739, 8.9350259280763559e-16),
     FCONST(-55.451579376534703, 2.7033385529648514e-15), FCONST(75.380137520845466, 4.1849996475502848e-15),
     FCONST(-78.617232429159216, -3.8328639627434316e-15), FCONST(58.643434952945782, -1.9016897701658308e-15),
     FCONST(-24.835000118745501, 1.3962335215182975e-15), FCONST(4.3361141446053626, 4.030374499852769e-16)},
    {FCONST(0.99999999999999656, 1.6913763379852753e-18), FCONST(0.37775583437447724, 4.432889165764209e-19),
     FCONST(-4.1665330297646515, -4.7646481165429575e-17), FCONST(-3.6868697152225005, 9.8102883493993438e-18),
     FCONST(6.5891093488609851, 6.3345234026201069e-18), FCONST(10.691136548136187, -3.0088364827679469e-16),
     FCONST(-2.1252745511089199, 4.0389796465751715e-17), FCONST(-14.282139810836357, 8.5577219550032172e-17),
     FCONST(-7.653101449542981, -5.9325796028133483e-18), FCONST(8.7137050826624076, -1.9409350445494054e-17),
     FCONST(9.1243814552467288, -3.3685665449593218e-17), FCONST(8.2919286693115239, -4.7271780343726278e-17),
     FCONST(-16.306902421855611, -1.0372700775042176e-16), FCONST(0.28886026125708886, 2.7437078418734018e-18),
     FCONST(11.33824869824528, -3.6588162649422882e-16), FCONST(-30.628067660806806, 3.1603172514587642e-16),
     FCONST(41.663072254002898, -3.7580203311517833e-16), FCONST(-24.4083546881213, -3.973011451307684e-16),
     FCONST(5.179045787526336, -4.2105867818463591e-17)},
    {FCONST(1, 0),
     FCONST(0.37757721004221101, 8.8705188303720204e-20),
     FCONST(-4.4612463940955109, -3.5050251148641108e-18),
     FCONST(-3.9447358998858486, -3.2282123278127983e-18),
     FCONST(7.7130911363447661, -3.3596569823566822e-18),
     FCONST(12.350475445257159, 3.1220767844934018e-17),
     FCONST(-3.3438000845841924, 2.4544072785647587e-18),
     FCONST(-18.167653295915915, -2.0944850030355155e-17),
     FCONST(-8.693898554005635, -1.2752017471939325e-18),
     FCONST(12.22916996882079, -4.9113214889075604e-17),
     FCONST(14.336309511959138, 3.6793737893458454e-17),
     FCONST(8.1540456802670853, 3.5272695170715455e-18),
     FCONST(-34.907179241661794, -3.2975756190717222e-17),
     FCONST(59.036374386226896, -4.9310050532221795e-17),
     FCONST(-141.11290053249562, 3.7995759863406417e-16),
     FCONST(235.15750453799728, -4.7803466022014618e-16),
     FCONST(-276.55667697949536, 3.213375471532345e-16),
     FCONST(246.37035090147938, 2.349641090258956e-16),
     FCONST(-151.01678036859818, -3.3518168888986109e-16),
     FCONST(53.613450542726589, 2.3701090319082141e-17),
     FCONST(-8.133477833185907, 2.1891966462135315e-18)},
    {FCONST(1, 0),
     FCONST(0.3774203445803887, 3.8136267478694207e-20),
     FCONST(-4.7559365577734267, -4.6990335476584729e-19),
     FCONST(-4.2025825873642164, 4.375677753705531e-19),
     FCONST(8.9238302194469057, -1.6396228410303593e-19),
     FCONST(14.129180754869632, -2.1921099186874925e-18),
     FCONST(-4.8422240685286253, -4.2347093415446578e-19),
     FCONST(-22.678825070632282, 4.9214453017339108e-18),
     FCONST(-9.7111679761838037, -3.8152619730681182e-20),
     FCONST(16.998854578127613, -4.2076760623604057e-18),
     FCONST(21.722575872272856, -4.2578572407364844e-19),
     FCONST(-2.9000442599504099, -4.1817278857342899e-19),
     FCONST(-10.575002355438688, -1.1630810750648379e-18),
     FCONST(-15.304429872528518, -1.570335286669433e-19),
     FCONST(-12.593066319413357, 4.137270753271878e-18),
     FCONST(75.997327488183629, 3.8113754242658613e-19),
     FCONST(-130.40820325656375, -7.1730144321918494e-18),
     FCONST(165.47161060055438, 2.6706946082413195e-17),
     FCONST(-134.8251428939067, 4.6988780237734319e-17),
     FCONST(58.368985502137242, 6.3692233525216584e-19),
     FCONST(-10.193160153326607, -3.1588982068933548e-18)},
    {FCONST(1.0, 0.0),
     FCONST(0.37728148845174042, -6.7790776549372823e-22),
     FCONST(-5.050607546808652, -3.7842848571017383e-20),
     FCONST(-4.4604132333947311, -3.2358439872041343e-20),
     FCONST(10.221326336437485, -1.3777612522244453e-19),
     FCONST(16.027197832098171, 2.885869247838855e-19),
     FCONST(-6.645530584802577, 1.7868683394044636e-20),
     FCONST(-27.865564308170583, 2.0454588718712329e-19),
     FCONST(-10.673847394182808, -1.7664137668907642e-19),
     FCONST(23.750024628278606, 3.8945180643349888e-19),
     FCONST(27.351439471708623, 4.0335739031434057e-20),
     FCONST(-4.9246135660911623, -3.3540767338126898e-20),
     FCONST(-14.319666646375101, -4.9329369049519297e-19),
     FCONST(-50.976578316769071, -2.1417332813143731e-20),
     FCONST(114.33110650398193, -2.8963434137403966e-18),
     FCONST(-233.76579082946395, 4.7831168398261071e-18),
     FCONST(450.57762648574271, 2.1160989999771118e-18),
     FCONST(-620.27013781391054, 1.1084021627902984e-18),
     FCONST(635.60005348396851, -1.7953138053417204e-18),
     FCONST(-482.98277423690405, 3.120700642466545e-18),
     FCONST(241.83364551815194, 1.6412841901183129e-18),
     FCONST(-63.912327799432276, 1.7909811809659003e-19),
     FCONST(3.4936129332440493, 3.5321480431593956e-20),
     FCONST(1.2845375935583203, 2.9466888110619039e-20)},
};
#endif /* !defined(HALF) */

#ifdef DOUBLE
static const int PFX(pswf43_degrees)[19] = {
    [8] = 8,
    [9] = 10,
    [16] = 16,
    [18] = 18,
};

static const FLOAT PFX(pswf43_c)[19] = {
    [8] = FCONST(15.658),
    [9] = FCONST(17.6215),
    [16] = FCONST(31.3659),
    [18] = FCONST(35.2929),
};

static const FLOAT PFX(pswf43_coeffs)[19][24] = {
    /* 4/3 specialized double-precision PSWF LUT, w=8, c=15.658 */
    [8] = {FCONST(1.000000159e+00), FCONST(3.814843807e-01), FCONST(-1.705772933e+00), FCONST(-1.574809829e+00), FCONST(8.972535502e-01),
           FCONST(3.081707523e-01), FCONST(3.170650492e+00), FCONST(-3.077828580e+00), FCONST(6.059434412e-01)},

    /* 4/3 specialized double-precision PSWF LUT, w=9, c=17.6215 */
    [9] = {FCONST(9.999999963e-01), FCONST(3.807464487e-01), FCONST(-1.954514108e+00), FCONST(-1.748689243e+00), FCONST(8.940126727e-01),
           FCONST(2.249806234e+00), FCONST(-2.729149505e-02), FCONST(9.313050829e-01), FCONST(-2.756708707e+00), FCONST(8.265223870e-01),
           FCONST(2.070220289e-01)},

    /* 4/3 specialized double-precision PSWF LUT, w=16, c=31.3659 */
    [16] = {FCONST(1.0000000000001339e+00), FCONST(3.7811581234051489e-01), FCONST(-3.6752769217932180e+00), FCONST(-3.2570386355695962e+00),
            FCONST(4.9086940673661745e+00), FCONST(8.1890620790437438e+00), FCONST(-6.2358453493817967e-01), FCONST(-9.2710446448475210e+00),
            FCONST(-4.8016686948535714e+00), FCONST(4.3204916239116353e-01), FCONST(1.5243689642512440e+01), FCONST(-1.4536218990429497e+01),
            FCONST(1.7764805508774899e+01), FCONST(-1.5898691650854410e+01), FCONST(-5.3046632079871514e-01), FCONST(7.1215110334374296e+00),
            FCONST(-2.4439347353446412e+00)},

    /* 4/3 specialized double-precision PSWF LUT, w=18, c=35.2929 */
    [18] = {FCONST(9.9999999999999656e-01), FCONST(3.7775583480425901e-01), FCONST(-4.1665323666296548e+00), FCONST(-3.6868691349989500e+00),
            FCONST(6.5891069175294641e+00), FCONST(1.0691132948304883e+01), FCONST(-2.1252720951508564e+00), FCONST(-1.4282131845918899e+01),
            FCONST(-7.6530982991550429e+00), FCONST(8.7136941477823111e+00), FCONST(9.1243872186275823e+00), FCONST(8.2918839729832232e+00),
            FCONST(-1.6306787738758960e+01), FCONST(2.8868238021460646e-01), FCONST(1.1338466753315100e+01), FCONST(-3.0628247597577907e+01),
            FCONST(4.1663152766264226e+01), FCONST(-2.4408368708189325e+01), FCONST(5.1790454589192194e+00)},
};
#endif

/* =========================================================================
 * Aligned Allocation
 * ========================================================================= */

static FLOAT *PFX(allocate_aligned)(size_t N) {
    size_t alloc_size = (sz_max(N * sizeof(FLOAT), 64) + 63) & ~63;
    FLOAT *ptr = (FLOAT *)nanofft_aligned_alloc(64, alloc_size);
    if (!ptr) exit(EXIT_FAILURE);
    memset(ptr, 0, alloc_size);
    return ptr;
}

static BREADTH_VEC *PFX(allocate_aligned_bvec)(size_t N) { return (BREADTH_VEC *)nanofft_calloc(N, sizeof(BREADTH_VEC)); }

/* =========================================================================
 * Kahan Summation (SOA layout)
 * ========================================================================= */

NANOFFT_EXACT_FP __attribute__((noinline)) static void PFX(kahan_add_soa)(FLOAT *sum_r, FLOAT *sum_i, FLOAT *comp_r, FLOAT *comp_i, FLOAT input_r,
                                                                          FLOAT input_i) {
    NANOFFT_EXACT_FP_PRAGMA
    FLOAT yr = SUB(input_r, *comp_r);
    FLOAT yi = SUB(input_i, *comp_i);
    FLOAT tr = ADD(*sum_r, yr);
    FLOAT ti = ADD(*sum_i, yi);
    *comp_r = SUB(SUB(tr, *sum_r), yr);
    *comp_i = SUB(SUB(ti, *sum_i), yi);
    *sum_r = tr;
    *sum_i = ti;
}

NANOFFT_EXACT_FP __attribute__((noinline)) static void PFX(kahan_add_soa_breadth)(BREADTH_VEC *sum_r, BREADTH_VEC *sum_i, BREADTH_VEC *comp_r,
                                                                                  BREADTH_VEC *comp_i, BREADTH_COMPUTE_VEC input_r,
                                                                                  BREADTH_COMPUTE_VEC input_i) {
    NANOFFT_EXACT_FP_PRAGMA
    const BREADTH_COMPUTE_VEC sr = LOAD_BVEC(sum_r);
    const BREADTH_COMPUTE_VEC si = LOAD_BVEC(sum_i);
    const BREADTH_COMPUTE_VEC cr = LOAD_BVEC(comp_r);
    const BREADTH_COMPUTE_VEC ci = LOAD_BVEC(comp_i);
    const BREADTH_COMPUTE_VEC yr = BREADTH_SUB(input_r, cr);
    const BREADTH_COMPUTE_VEC yi = BREADTH_SUB(input_i, ci);
    const BREADTH_COMPUTE_VEC tr = BREADTH_ADD(sr, yr);
    const BREADTH_COMPUTE_VEC ti = BREADTH_ADD(si, yi);
    STORE_BVEC(comp_r, BREADTH_SUB(BREADTH_SUB(tr, sr), yr));
    STORE_BVEC(comp_i, BREADTH_SUB(BREADTH_SUB(ti, si), yi));
    STORE_BVEC(sum_r, tr);
    STORE_BVEC(sum_i, ti);
}

/* =========================================================================
 * Double-Word Arithmetic (error-free transforms)
 * ========================================================================= */

typedef struct {
    FLOAT hi;
    FLOAT lo;
} PFX(ff_t);

NANOFFT_EXACT_FP static inline PFX(ff_t) PFX(ff_make)(FLOAT hi, FLOAT lo) {
    NANOFFT_EXACT_FP_PRAGMA
    PFX(ff_t) res = {hi, lo};
    return res;
}

NANOFFT_EXACT_FP static inline void PFX(ff_two_sum_quick)(FLOAT x, FLOAT y, FLOAT *r, FLOAT *e) {
    NANOFFT_EXACT_FP_PRAGMA
    *r = ADD(x, y);
    *e = SUB(y, SUB(*r, x));
}

NANOFFT_EXACT_FP static inline void PFX(ff_two_difference)(FLOAT x, FLOAT y, FLOAT *r, FLOAT *e) {
    NANOFFT_EXACT_FP_PRAGMA
    *r = SUB(x, y);
    FLOAT t = SUB(*r, x);
    *e = SUB(SUB(x, SUB(*r, t)), ADD(y, t));
}

NANOFFT_EXACT_FP static inline void PFX(ff_two_product)(FLOAT x, FLOAT y, FLOAT *r, FLOAT *e) {
    NANOFFT_EXACT_FP_PRAGMA
    *r = MUL(x, y);
    *e = M_FMA(x, y, NEG(*r));
}

NANOFFT_EXACT_FP NANOFFT_UNUSED static inline PFX(ff_t) PFX(ff_sub)(PFX(ff_t) a, PFX(ff_t) b) {
    NANOFFT_EXACT_FP_PRAGMA
    FLOAT r, e, r_f, e_f;
    PFX(ff_two_difference)(a.hi, b.hi, &r, &e);
    e = SUB(ADD(e, a.lo), b.lo);
    PFX(ff_two_sum_quick)(r, e, &r_f, &e_f);
    return PFX(ff_make)(r_f, e_f);
}

NANOFFT_EXACT_FP NANOFFT_UNUSED static inline PFX(ff_t) PFX(ff_mul)(PFX(ff_t) a, PFX(ff_t) b) {
    NANOFFT_EXACT_FP_PRAGMA
    FLOAT r, e, r_f, e_f;
    PFX(ff_two_product)(a.hi, b.hi, &r, &e);
    e = ADD(ADD(e, MUL(a.hi, b.lo)), MUL(a.lo, b.hi));
    PFX(ff_two_sum_quick)(r, e, &r_f, &e_f);
    return PFX(ff_make)(r_f, e_f);
}

/* =========================================================================
 * Matrix Helpers
 * ========================================================================= */

typedef struct {
    int rows, cols;
    FLOAT *d;
} PFX(MatrixF);

typedef struct {
    int rows, cols;
    FLOAT *r;
    FLOAT *i;
} PFX(MatrixC);

static PFX(MatrixF) * PFX(mat_alloc)(int r, int c) {
    PFX(MatrixF) *M = (PFX(MatrixF) *)nanofft_malloc(sizeof(PFX(MatrixF)));
    M->rows = r;
    M->cols = c;
    M->d = (FLOAT *)nanofft_calloc((size_t)r * c, sizeof(FLOAT));
    return M;
}

static PFX(MatrixC) * PFX(matc_alloc)(int r, int c) {
    PFX(MatrixC) *M = (PFX(MatrixC) *)nanofft_malloc(sizeof(PFX(MatrixC)));
    M->rows = r;
    M->cols = c;
    M->r = (FLOAT *)nanofft_calloc((size_t)r * c, sizeof(FLOAT));
    M->i = (FLOAT *)nanofft_calloc((size_t)r * c, sizeof(FLOAT));
    return M;
}

static void PFX(mat_free)(PFX(MatrixF) * M) {
    nanofft_aligned_free(M->d);
    nanofft_aligned_free(M);
}

static void PFX(matc_free)(PFX(MatrixC) * M) {
    nanofft_aligned_free(M->r);
    nanofft_aligned_free(M->i);
    nanofft_aligned_free(M);
}

static inline FLOAT PFX(mat_get)(const PFX(MatrixF) * M, int i, int j) { return M->d[i * M->cols + j]; }
static inline void PFX(mat_set)(PFX(MatrixF) * M, int i, int j, FLOAT v) { M->d[i * M->cols + j] = v; }
static inline FLOAT PFX(matc_get_r)(const PFX(MatrixC) * M, int i, int j) { return M->r[i * M->cols + j]; }
static inline FLOAT PFX(matc_get_i)(const PFX(MatrixC) * M, int i, int j) { return M->i[i * M->cols + j]; }
static inline void PFX(matc_set_r)(PFX(MatrixC) * M, int i, int j, FLOAT v) { M->r[i * M->cols + j] = v; }
static inline void PFX(matc_set_i)(PFX(MatrixC) * M, int i, int j, FLOAT v) { M->i[i * M->cols + j] = v; }

static void PFX(ipow)(int q, FLOAT *r, FLOAT *i) {
    int qm = ((q % 4) + 4) % 4;
    *r = FCAST(0.0);
    *i = FCAST(0.0);
    if (qm == 0)
        *r = FCAST(1.0);
    else if (qm == 1)
        *i = FCAST(1.0);
    else if (qm == 2)
        *r = FCAST(-1.0);
    else
        *i = FCAST(-1.0);
}

static FLOAT PFX(jn_approx)(int n, FLOAT x, int num_terms) {
    if (num_terms <= 0) return FCAST(0.0);
    int n_abs = (n < 0) ? -n : n;
    int max_terms = 28 - n_abs;
    if (max_terms < 0) max_terms = 0;
    int terms = (num_terms < max_terms) ? num_terms : max_terms;
#if defined(HALF)
    double half_x = 0.5 * TO_DOUBLE(x);
    double half_x2 = half_x * half_x;
    double term_numer = 1.0;
    for (int i = 0; i < n_abs; i++) term_numer *= half_x;
    double result = 0.0;
    for (int k = 0; k < terms; ++k) {
        double denom = PFX(fact_d)[k] * PFX(fact_d)[n_abs + k];
        result += term_numer / denom;
        term_numer *= -half_x2;
    }
    if (n < 0 && (n & 1)) result = -result;
    return FCAST(result);
#else
    FLOAT half_x = MUL(FCAST(0.5), x);
    FLOAT half_x2 = MUL(half_x, half_x);
    FLOAT term_numer = FCAST(1.0);
    for (int i = 0; i < n_abs; i++) term_numer = MUL(term_numer, half_x);
    FLOAT result = FCAST(0.0);
    for (int k = 0; k < terms; ++k) {
        FLOAT denom = MUL(PFX(fact)[k], PFX(fact)[n_abs + k]);
        result = ADD(result, DIV(term_numer, denom));
        term_numer = MUL(term_numer, NEG(half_x2));
    }
    if (n < 0 && (n & 1)) result = NEG(result);
    return result;
#endif
}

/* =========================================================================
 * LRA NUFFT Implementation
 * ========================================================================= */

struct PFX(lra_plan) {
    int Mpoints, N, Nfft, K;
    int max_Mpoints, max_N, max_K, Nfft_allocated;
    int num_factors;
    double df;
    int is_breadth;
    int *Tidx;
    FLOAT *conjU_r, *conjU_i;
    FLOAT *conjV;
    NANOFFT_PLAN *nanofft_p;
    FLOAT *fft_real, *fft_imag;
    FLOAT *temp_r, *temp_i, *temp_comp_r, *temp_comp_i;
    BREADTH_VEC *bfft_real, *bfft_imag;
    BREADTH_VEC *btemp_r, *btemp_i, *btemp_comp_r, *btemp_comp_i;
};

static PFX(MatrixF) * PFX(chebyshev_mat)(int d, const FLOAT *x, int M) {
    int cols = d + 1;
    PFX(MatrixF) *T = PFX(mat_alloc)(M, cols);
    for (int i = 0; i < M; ++i) PFX(mat_set)(T, i, 0, FCAST(1.0));
    if (d > 0)
        for (int i = 0; i < M; ++i) PFX(mat_set)(T, i, 1, x[i]);
    for (int k = 1; k < d; ++k)
        for (int i = 0; i < M; ++i) PFX(mat_set)(T, i, k + 1, SUB(MUL(MUL(FCAST(2.0), x[i]), PFX(mat_get)(T, i, k)), PFX(mat_get)(T, i, k - 1)));
    return T;
}

static PFX(MatrixC) * PFX(bessel_mat)(int K) {
    PFX(MatrixC) *C = PFX(matc_alloc)(K, K);
    FLOAT arg = MUL(FCONST(3.1415926535897931, 1.2246467991473532e-16), FCAST(-0.25));
    for (int p = 0; p < K; ++p) {
        for (int q = 0; q < K; ++q) {
            if (((q - p) & 1) != 0) continue;
            FLOAT j1 = PFX(jn_approx)((p + q) / 2, arg, LRA_BESSEL_TERMS);
            FLOAT j2 = PFX(jn_approx)((q - p) / 2, arg, LRA_BESSEL_TERMS);
            FLOAT ip_r, ip_i;
            PFX(ipow)(q, &ip_r, &ip_i);
            FLOAT scalar = MUL(MUL(FCAST(4.0), j1), j2);
            PFX(matc_set_r)(C, p, q, MUL(ip_r, scalar));
            PFX(matc_set_i)(C, p, q, MUL(ip_i, scalar));
        }
    }
    for (int q = 0; q < K; ++q) {
        PFX(matc_set_r)(C, 0, q, MUL(PFX(matc_get_r)(C, 0, q), FCAST(0.5)));
        PFX(matc_set_i)(C, 0, q, MUL(PFX(matc_get_i)(C, 0, q), FCAST(0.5)));
    }
    for (int p = 0; p < K; ++p) {
        PFX(matc_set_r)(C, p, 0, MUL(PFX(matc_get_r)(C, p, 0), FCAST(0.5)));
        PFX(matc_set_i)(C, p, 0, MUL(PFX(matc_get_i)(C, p, 0), FCAST(0.5)));
    }
    return C;
}

struct PFX(lra_plan) * PFX(lra_initialize_opts)(int max_Mpoints, int max_N, int max_rank, double df, int freq_factor, int vec_mode) {
    struct PFX(lra_plan) *plan = (struct PFX(lra_plan) *)nanofft_calloc(1, sizeof(struct PFX(lra_plan)));
    if (!plan) return NULL;
    plan->max_Mpoints = max_Mpoints;
    plan->max_N = max_N;
    plan->max_K = max_rank;
    plan->num_factors = freq_factor;
    plan->df = df;
    plan->is_breadth = (vec_mode == TLS_VEC_BREADTH) ? 1 : 0;
    plan->Nfft_allocated = next_fast_len(max_N);
    plan->Nfft = plan->Nfft_allocated;

    plan->Tidx = (int *)nanofft_malloc((size_t)max_Mpoints * freq_factor * sizeof(int));
    plan->conjU_r = (FLOAT *)nanofft_calloc((size_t)max_Mpoints * max_rank * freq_factor, sizeof(FLOAT));
    plan->conjU_i = (FLOAT *)nanofft_calloc((size_t)max_Mpoints * max_rank * freq_factor, sizeof(FLOAT));
    plan->conjV = (FLOAT *)nanofft_calloc((size_t)plan->Nfft_allocated * max_rank, sizeof(FLOAT));

    size_t V_size = (size_t)plan->Nfft_allocated * max_rank;
    if (plan->is_breadth) {
        plan->bfft_real = PFX(allocate_aligned_bvec)((size_t)plan->Nfft_allocated);
        plan->bfft_imag = PFX(allocate_aligned_bvec)((size_t)plan->Nfft_allocated);
        plan->nanofft_p = NANOFFT_MAKE_PLAN_BREADTH((uint32_t)plan->Nfft_allocated);
        plan->btemp_r = PFX(allocate_aligned_bvec)(V_size);
        plan->btemp_i = PFX(allocate_aligned_bvec)(V_size);
        plan->btemp_comp_r = PFX(allocate_aligned_bvec)(V_size);
        plan->btemp_comp_i = PFX(allocate_aligned_bvec)(V_size);
    } else {
        plan->fft_real = PFX(allocate_aligned)(plan->Nfft_allocated);
        plan->fft_imag = PFX(allocate_aligned)(plan->Nfft_allocated);
        plan->nanofft_p = NANOFFT_MAKE_PLAN((uint32_t)plan->Nfft_allocated);
        plan->temp_r = (FLOAT *)nanofft_calloc(V_size, sizeof(FLOAT));
        plan->temp_i = (FLOAT *)nanofft_calloc(V_size, sizeof(FLOAT));
        plan->temp_comp_r = (FLOAT *)nanofft_calloc(V_size, sizeof(FLOAT));
        plan->temp_comp_i = (FLOAT *)nanofft_calloc(V_size, sizeof(FLOAT));
    }
    return plan;
}

struct PFX(lra_plan) * PFX(lra_initialize)(int max_Mpoints, int max_N, int max_rank, double df, int freq_factor) {
    return PFX(lra_initialize_opts)(max_Mpoints, max_N, max_rank, df, freq_factor, TLS_VEC_DEPTH);
}

void PFX(lra_precompute)(struct PFX(lra_plan) * plan, const nufft_input_t *x, int Mpoints, int N, int rank) {
    plan->Mpoints = Mpoints;
    plan->N = N;
    plan->K = rank;
    plan->Nfft = plan->Nfft_allocated;
    int Nfft = plan->Nfft;

    for (int f_idx = 0; f_idx < plan->num_factors; ++f_idx) {
        int k_factor = f_idx + 1;

        nufft_input_t *x_scaled = (nufft_input_t *)nanofft_malloc(Mpoints * sizeof(nufft_input_t));
        for (int i = 0; i < Mpoints; ++i) {
#ifdef DOUBLE_DOUBLE
            x_scaled[i] = dd_mul(x[i], dd_make(plan->df * k_factor, 0.0));
#else
            x_scaled[i] = x[i] * plan->df * k_factor;
#endif
        }

        size_t offset_T = (size_t)f_idx * plan->max_Mpoints;
        int *current_Tidx = plan->Tidx + offset_T;
        get_lra_params(x_scaled, Mpoints, Nfft, current_Tidx);

        FLOAT *er = (FLOAT *)nanofft_malloc(Mpoints * sizeof(FLOAT));
        FLOAT *scaled = (FLOAT *)nanofft_malloc(Mpoints * sizeof(FLOAT));
        for (int i = 0; i < Mpoints; ++i) {
#ifdef DOUBLE_DOUBLE
            dd_t z = dd_mul(x_scaled[i], dd_make((double)Nfft, 0.0));
            double nearest = floor(dd_to_double(z) + 0.5);
            er[i] = dd_sub(z, dd_make(nearest, 0.0));
#else
            double z = (double)Nfft * x_scaled[i];
            er[i] = FCAST(z - floor(z + 0.5));
#endif
            scaled[i] = MUL(er[i], FCAST(2.0));
        }

        PFX(MatrixF) *Tcheb_U = PFX(chebyshev_mat)(rank - 1, scaled, Mpoints);
        nanofft_aligned_free(scaled);
        PFX(MatrixC) *B = PFX(bessel_mat)(rank);

        size_t offset_U = (size_t)f_idx * plan->max_Mpoints * plan->max_K;
        for (int i = 0; i < Mpoints; ++i) {
            FLOAT theta = MUL(FCAST(-0.5), er[i]);
            FLOAT phase_r = M_COS2PI(theta);
            FLOAT phase_i = M_SIN2PI(theta);
            for (int q = 0; q < rank; ++q) {
                FLOAT acc_r = FCAST(0.0), acc_i = FCAST(0.0), comp_r = FCAST(0.0), comp_i = FCAST(0.0);
                for (int p = 0; p < rank; ++p) {
                    FLOAT t_val = PFX(mat_get)(Tcheb_U, i, p);
                    FLOAT val_r = MUL(t_val, PFX(matc_get_r)(B, p, q));
                    FLOAT val_i = MUL(t_val, PFX(matc_get_i)(B, p, q));
                    PFX(kahan_add_soa)(&acc_r, &acc_i, &comp_r, &comp_i, val_r, val_i);
                }
                size_t idx = offset_U + (size_t)i * rank + q;
                plan->conjU_r[idx] = SUB(MUL(phase_r, acc_r), MUL(phase_i, acc_i));
                plan->conjU_i[idx] = NEG(ADD(MUL(phase_r, acc_i), MUL(phase_i, acc_r)));
            }
        }

        PFX(mat_free)(Tcheb_U);
        PFX(matc_free)(B);
        nanofft_aligned_free(er);
        nanofft_aligned_free(x_scaled);
    }

    FLOAT *X = (FLOAT *)nanofft_malloc((size_t)Nfft * sizeof(FLOAT));
    for (int w = 0; w < Nfft; ++w) {
#if defined(HALF)
        X[w] = FCAST((2.0 * (double)w) / (double)Nfft - 1.0);
#else
        X[w] = SUB(DIV(FCAST(2.0 * w), FCAST(Nfft)), FCAST(1.0));
#endif
    }
    PFX(MatrixF) *Tcheb_V = PFX(chebyshev_mat)(rank - 1, X, Nfft);
    nanofft_aligned_free(X);
    for (int w = 0; w < Nfft; ++w) {
        for (int k = 0; k < rank; ++k) plan->conjV[(size_t)w * rank + k] = PFX(mat_get)(Tcheb_V, w, k);
    }
    PFX(mat_free)(Tcheb_V);
}

void PFX(lra_execute)(const struct PFX(lra_plan) * plan, const FLOAT *y_real, const FLOAT *y_imag, FLOAT *out_real, FLOAT *out_imag, int freq_factor) {
    int f_idx = freq_factor - 1;
    if (f_idx < 0 || f_idx >= plan->num_factors) return;

    size_t per_matrix = (size_t)plan->Nfft * plan->K;
    memset(plan->temp_r, 0, per_matrix * sizeof(FLOAT));
    memset(plan->temp_i, 0, per_matrix * sizeof(FLOAT));
    memset(plan->temp_comp_r, 0, per_matrix * sizeof(FLOAT));
    memset(plan->temp_comp_i, 0, per_matrix * sizeof(FLOAT));

    size_t offset_T = (size_t)f_idx * plan->max_Mpoints;
    int *current_Tidx = plan->Tidx + offset_T;

    size_t offset_U = (size_t)f_idx * plan->max_Mpoints * plan->max_K;
    FLOAT *current_conjU_r = plan->conjU_r + offset_U;
    FLOAT *current_conjU_i = plan->conjU_i + offset_U;

    for (int m = 0; m < plan->Mpoints; ++m) {
        int t = current_Tidx[m];
        if (t < 0 || t >= plan->Nfft) continue;
        FLOAT yr = y_real[m], yi = y_imag[m];
        for (int k = 0; k < plan->K; ++k) {
            size_t cu_idx = (size_t)m * plan->K + k;
            FLOAT term_r = SUB(MUL(current_conjU_r[cu_idx], yr), MUL(current_conjU_i[cu_idx], yi));
            FLOAT term_i = ADD(MUL(current_conjU_r[cu_idx], yi), MUL(current_conjU_i[cu_idx], yr));
            size_t t_idx = (size_t)t * plan->K + k;
            PFX(kahan_add_soa)(&plan->temp_r[t_idx], &plan->temp_i[t_idx], &plan->temp_comp_r[t_idx], &plan->temp_comp_i[t_idx], term_r, term_i);
        }
    }
    for (int k = 0; k < plan->K; ++k) {
        for (int n = 0; n < plan->Nfft; ++n) {
            plan->fft_real[n] = plan->temp_r[(size_t)n * plan->K + k];
            plan->fft_imag[n] = plan->temp_i[(size_t)n * plan->K + k];
        }
        NANOFFT_EXECUTE(plan->nanofft_p, plan->fft_real, plan->fft_imag);
        for (int n = 0; n < plan->Nfft; ++n) {
            plan->temp_r[(size_t)n * plan->K + k] = plan->fft_real[n];
            plan->temp_i[(size_t)n * plan->K + k] = plan->fft_imag[n];
        }
    }
    for (int n = 0; n < plan->Nfft; ++n) {
        FLOAT acc_r = FCAST(0.0), acc_i = FCAST(0.0);
        for (int k = 0; k < plan->K; ++k) {
            size_t idx = (size_t)n * plan->K + k;
            FLOAT cvr = plan->conjV[idx];
            acc_r = ADD(acc_r, MUL(cvr, plan->temp_r[idx]));
            acc_i = ADD(acc_i, MUL(cvr, plan->temp_i[idx]));
        }
        if (n < plan->N) {
            out_real[n] = acc_r;
            out_imag[n] = acc_i;
        }
    }
}

void PFX(lra_execute_breadth)(const struct PFX(lra_plan) * plan, const nufft_input_t *x, const FLOAT *val, double f0, int block, int block_base,
                              int freq_factor, BREADTH_VEC *out_real, BREADTH_VEC *out_imag) {
    int f_idx = freq_factor - 1;
    if (!plan || !plan->is_breadth || f_idx < 0 || f_idx >= plan->num_factors) return;

    const size_t per_matrix = (size_t)plan->Nfft * (size_t)plan->K;
    memset(plan->btemp_r, 0, per_matrix * sizeof(BREADTH_VEC));
    memset(plan->btemp_i, 0, per_matrix * sizeof(BREADTH_VEC));
    memset(plan->btemp_comp_r, 0, per_matrix * sizeof(BREADTH_VEC));
    memset(plan->btemp_comp_i, 0, per_matrix * sizeof(BREADTH_VEC));

    const size_t offset_T = (size_t)f_idx * (size_t)plan->max_Mpoints;
    const int *current_Tidx = plan->Tidx + offset_T;

    const size_t offset_U = (size_t)f_idx * (size_t)plan->max_Mpoints * (size_t)plan->max_K;
    const FLOAT *current_conjU_r = plan->conjU_r + offset_U;
    const FLOAT *current_conjU_i = plan->conjU_i + offset_U;

#if defined(DOUBLE_DOUBLE)
    VEC f_hi, f_lo;
    const dd_t d_f0 = dd_make(f0, 0.0);
    const dd_t d_df = dd_make(plan->df, 0.0);
    const dd_t d_q = dd_make((double)freq_factor, 0.0);
    for (int v = 0; v < VECDD_LEN; ++v) {
        dd_t k_idx = dd_mul(dd_make((double)(block_base + v), 0.0), dd_make((double)block, 0.0));
        dd_t fv = dd_mul(d_q, dd_add(d_f0, dd_mul(k_idx, d_df)));
        f_hi[v] = fv.hi;
        f_lo[v] = fv.lo;
    }
    const VECDD v_freq = vecdd_make(f_hi, f_lo);
#elif defined(DOUBLE)
    VEC v_freq;
    for (int v = 0; v < VEC_LEN; ++v) v_freq[v] = (double)freq_factor * (f0 + (double)(block_base + v) * (double)block * plan->df);
#elif defined(HALF) && NANOFFT_NATIVE_FP16
    VEC v_freq[VECH_LEN / VEC_LEN];
    for (int ch = 0; ch < (VECH_LEN / VEC_LEN); ++ch) {
        for (int v = 0; v < VEC_LEN; ++v) {
            v_freq[ch][v] = (double)freq_factor * (f0 + (double)(block_base + ch * VEC_LEN + v) * (double)block * plan->df);
        }
    }
#else
    VEC v_freq_lo, v_freq_hi;
    for (int v = 0; v < VEC_LEN; ++v) {
        v_freq_lo[v] = (double)freq_factor * (f0 + (double)(block_base + v) * (double)block * plan->df);
        v_freq_hi[v] = (double)freq_factor * (f0 + (double)(block_base + VEC_LEN + v) * (double)block * plan->df);
    }
#endif

    for (int m = 0; m < plan->Mpoints; ++m) {
        const int t = current_Tidx[m];
        if (t < 0 || t >= plan->Nfft) continue;
        BREADTH_COMPUTE_VEC c, s;
#if defined(DOUBLE_DOUBLE)
        vecdd_sincos2pi(vecdd_mul(vecdd_splat(x[m]), v_freq), &c, &s);
#elif defined(DOUBLE)
        vec_sincos2pi(((VEC){} + x[m]) * v_freq, &c, &s);
#elif defined(HALF) && NANOFFT_NATIVE_FP16
        const VEC xm_v = (VEC){} + x[m];
        VEC ang[VECH_LEN / VEC_LEN];
        for (int ch = 0; ch < (VECH_LEN / VEC_LEN); ++ch) ang[ch] = xm_v * v_freq[ch];
        vech_sincos2pi_d(ang, &c, &s);
#else
        const VEC xm_v = (VEC){} + x[m];
        vecf_sincos2pi_d(xm_v * v_freq_lo, xm_v * v_freq_hi, &c, &s);
#endif
        const BREADTH_COMPUTE_VEC yr = BREADTH_MUL_S(c, val[m]);
        const BREADTH_COMPUTE_VEC yi = BREADTH_MUL_S(s, val[m]);
        for (int k = 0; k < plan->K; ++k) {
            const size_t cu_idx = (size_t)m * (size_t)plan->K + (size_t)k;
            const BREADTH_COMPUTE_VEC cur = BREADTH_SPLAT(current_conjU_r[cu_idx]);
            const BREADTH_COMPUTE_VEC cui = BREADTH_SPLAT(current_conjU_i[cu_idx]);
            const BREADTH_COMPUTE_VEC term_r = BREADTH_SUB(BREADTH_MUL(cur, yr), BREADTH_MUL(cui, yi));
            const BREADTH_COMPUTE_VEC term_i = BREADTH_ADD(BREADTH_MUL(cur, yi), BREADTH_MUL(cui, yr));
            const size_t t_idx = (size_t)t * (size_t)plan->K + (size_t)k;
            PFX(kahan_add_soa_breadth)(&plan->btemp_r[t_idx], &plan->btemp_i[t_idx], &plan->btemp_comp_r[t_idx], &plan->btemp_comp_i[t_idx], term_r, term_i);
        }
    }

    for (int k = 0; k < plan->K; ++k) {
        for (int n = 0; n < plan->Nfft; ++n) {
            plan->bfft_real[n] = plan->btemp_r[(size_t)n * (size_t)plan->K + (size_t)k];
            plan->bfft_imag[n] = plan->btemp_i[(size_t)n * (size_t)plan->K + (size_t)k];
        }
        NANOFFT_EXECUTE_BREADTH(plan->nanofft_p, plan->bfft_real, plan->bfft_imag);
        for (int n = 0; n < plan->Nfft; ++n) {
            plan->btemp_r[(size_t)n * (size_t)plan->K + (size_t)k] = plan->bfft_real[n];
            plan->btemp_i[(size_t)n * (size_t)plan->K + (size_t)k] = plan->bfft_imag[n];
        }
    }

    for (int n = 0; n < plan->N; ++n) {
        BREADTH_COMPUTE_VEC acc_r = BREADTH_ZERO;
        BREADTH_COMPUTE_VEC acc_i = BREADTH_ZERO;
        for (int k = 0; k < plan->K; ++k) {
            const size_t idx = (size_t)n * (size_t)plan->K + (size_t)k;
            const FLOAT cvr = plan->conjV[idx];
            acc_r = BREADTH_ADD(acc_r, BREADTH_MUL_S(LOAD_BVEC(&plan->btemp_r[idx]), cvr));
            acc_i = BREADTH_ADD(acc_i, BREADTH_MUL_S(LOAD_BVEC(&plan->btemp_i[idx]), cvr));
        }
        STORE_BVEC(&out_real[n], acc_r);
        STORE_BVEC(&out_imag[n], acc_i);
    }
}

void PFX(free_lra_plan)(struct PFX(lra_plan) * plan) {
    if (!plan) return;
    if (plan->nanofft_p) NANOFFT_DESTROY_PLAN(plan->nanofft_p);
    nanofft_aligned_free(plan->fft_real);
    nanofft_aligned_free(plan->fft_imag);
    nanofft_aligned_free(plan->conjU_r);
    nanofft_aligned_free(plan->conjU_i);
    nanofft_aligned_free(plan->conjV);
    nanofft_aligned_free(plan->temp_r);
    nanofft_aligned_free(plan->temp_i);
    nanofft_aligned_free(plan->temp_comp_r);
    nanofft_aligned_free(plan->temp_comp_i);
    nanofft_aligned_free(plan->bfft_real);
    nanofft_aligned_free(plan->bfft_imag);
    nanofft_aligned_free(plan->btemp_r);
    nanofft_aligned_free(plan->btemp_i);
    nanofft_aligned_free(plan->btemp_comp_r);
    nanofft_aligned_free(plan->btemp_comp_i);
    nanofft_aligned_free(plan->Tidx);
    nanofft_aligned_free(plan);
}

/* =========================================================================
 * PSWF NUFFT Implementation
 * ========================================================================= */

struct PFX(pswf_plan) {
    int Mpoints, N, Nout, Nfft, w;
    int spread_stride, spread_pad, fft_alloc_len;
    int output_shift;
    char upsamp[2];
    FLOAT alpha;
    int num_factors;
    double df;
    int use_ppoly;
    int is_breadth;
    int ppoly_nc;
    int ppoly_stride;
    FLOAT *ppoly_coeffs;
    int *spread_first_idx;
    FLOAT *spread_z;
    int *spread_base_idx;
    FLOAT *spread_weight;
    FLOAT *deconv, *shift_r, *shift_i;
    NANOFFT_PLAN *nanofft_p;
    FLOAT *fft_real, *fft_imag;
    /* Slot-order (deferred bit reversal) support.  `slot_deconv` is the
     * deconvolution table indexed by FFT position, with the NuFFT oversampling
     * padding zeroed; `period`/`nvalid`/`slot_offsets` describe the periodic
     * pattern of kept positions so only those are ever written out. */
    int slots, valid_slots, period, nvalid;
    int slot_tables_built;
    unsigned char slot_offsets[64];
    FLOAT *slot_deconv;
    FLOAT *perm_scratch;
    BREADTH_VEC *bfft_real, *bfft_imag;
    BREADTH_VEC *bperm_scratch;
};

static inline int PFX(positive_mod_int)(int value, int modulus) {
    int result = value % modulus;
    return result < 0 ? result + modulus : result;
}

static inline int PFX(round_up_int)(int value, int multiple) { return ((value + multiple - 1) / multiple) * multiple; }

static inline bool PFX(pswf_lut_lookup)(int w, bool mode43, const FLOAT **coeffs, int *deg, FLOAT *c) {
#if defined(HALF)
    if (w == 4 || w == 5) {
        int idx = w - 4;
        *coeffs = mode43 ? PFX(pswf43_h_coeffs)[idx] : PFX(pswf21_h_coeffs)[idx];
        *deg = 5;
        *c = mode43 ? PFX(pswf43_h_c)[idx] : PFX(pswf21_h_c)[idx];
        return true;
    }
    return false;
#else
#    ifdef DOUBLE
    if (mode43 && (w == 8 || w == 9 || w == 16 || w == 18)) {
        *coeffs = PFX(pswf43_coeffs)[w];
        *deg = PFX(pswf43_degrees)[w];
        *c = PFX(pswf43_c)[w];
        return true;
    }
#    endif

    if (!mode43 && w < 19) {
        *coeffs = PFX(pswf_coeffs)[w];
        *deg = pswf_degrees[w];
        *c = FCAST(NANOFFT_PI * w * 0.75 - 0.05);
        return true;
    }

    return false;
#endif
}

static inline int PFX(pswf_kernel_scale_exp)(int w, bool mode43, bool is_ppoly) {
    if (w <= 1) return 0;
    int safe_w = (w > PSWF_MAX_W) ? PSWF_MAX_W : w;
    const FLOAT *coeffs = NULL;
    int deg = 0;
    FLOAT c_val = FCAST(0.0);
    const bool uses_pswf = is_ppoly || PFX(pswf_lut_lookup)(safe_w, mode43, &coeffs, &deg, &c_val);
    const double inv_4ln2 = 0.36067376022224085; /* 1 / (4 * ln(2)) */
    double raw_e;
    if (uses_pswf) {
        double c = is_ppoly ? (NANOFFT_PI * (double)safe_w * (mode43 ? 0.625 : 0.75) - 0.05) : TO_DOUBLE(c_val);
        if (c <= 1.0) return 0;
        raw_e = (c - 0.5 * log(4.0 * NANOFFT_PI * c)) * inv_4ln2;
    } else {
        double beta = (mode43 ? 1.90 : 2.30) * (double)safe_w;
        raw_e = beta * inv_4ln2;
    }
    int e = (int)floor(raw_e + 0.5);
    if (e < 0) e = 0;
    if (e > PSWF_KERNEL_MAX_EXP) e = PSWF_KERNEL_MAX_EXP;
    return e;
}

static inline FLOAT PFX(pswf_log_scale)(int e) { return MUL(FCAST(e), FCONST(0.69314718055994529, 2.3190468138462996e-17)); }

static inline FLOAT PFX(pswf0)(FLOAT z, int w, bool mode43) {
    int safe_w = w;
    if (safe_w < 1) safe_w = 1;
    if (safe_w > PSWF_MAX_W) safe_w = PSWF_MAX_W;

    FLOAT Z = MUL(z, z);
    const FLOAT *coeffs = NULL;
    int deg = 0;
    FLOAT c = FCAST(0.0);
    const FLOAT log_scale = PFX(pswf_log_scale)(PFX(pswf_kernel_scale_exp)(safe_w, mode43, false));

    if (PFX(pswf_lut_lookup)(safe_w, mode43, &coeffs, &deg, &c)) {
        FLOAT poly = coeffs[deg];
        for (int i = deg - 1; i >= 0; --i) poly = M_FMA(poly, Z, coeffs[i]);

        return MUL(M_EXP(ADD(log_scale, MUL(MUL(NEG(c), Z), FCAST(0.5)))), poly);
    } else {
        FLOAT beta = FCAST((mode43 ? 1.90 : 2.30) * safe_w);
        FLOAT inner = SUB(FCAST(1.0), Z);
        if (TO_DOUBLE(inner) < 0.0) inner = FCAST(0.0);

        return M_EXP(ADD(log_scale, MUL(beta, SUB(M_SQRT(inner), FCAST(1.0)))));
    }
}

static void PFX(pswf0_batch)(const FLOAT *__restrict__ z_arr, FLOAT *__restrict__ out_arr, int n, int w, bool mode43) {
    int safe_w = w;
    if (safe_w < 1) safe_w = 1;
    if (safe_w > PSWF_MAX_W) safe_w = PSWF_MAX_W;

#if INTERNAL_VEC_LEN > 1
    const FLOAT *coeffs = NULL;
    int deg = 0;
    FLOAT c = FCAST(0.0);
    bool has_lut = PFX(pswf_lut_lookup)(safe_w, mode43, &coeffs, &deg, &c);
    const FLOAT log_scale = PFX(pswf_log_scale)(PFX(pswf_kernel_scale_exp)(safe_w, mode43, false));
    const INTERNAL_VEC v_log_scale = IVEC_SPLAT(log_scale);

    int n_vec = n - (n % INTERNAL_VEC_LEN);

    if (has_lut) {
        const INTERNAL_VEC v_neg_c_half = IVEC_SPLAT(NEG(MUL(c, FCAST(0.5))));

        for (int i = 0; i < n_vec; i += INTERNAL_VEC_LEN) {
            INTERNAL_VEC v_z = LOAD_VEC(&z_arr[i]);
            INTERNAL_VEC v_Z = MUL(v_z, v_z);

            INTERNAL_VEC v_poly = IVEC_SPLAT(coeffs[deg]);
            for (int j = deg - 1; j >= 0; --j) v_poly = ADD(MUL(v_poly, v_Z), IVEC_SPLAT(coeffs[j]));

            INTERNAL_VEC v_result = MUL(M_EXP(ADD(v_log_scale, MUL(v_Z, v_neg_c_half))), v_poly);
            STORE_VEC(&out_arr[i], v_result);
        }
    } else {
        FLOAT beta = FCAST((mode43 ? 1.90 : 2.30) * safe_w);
        INTERNAL_VEC v_one = IVEC_SPLAT(FCAST(1.0));
        INTERNAL_VEC v_beta = IVEC_SPLAT(beta);

        for (int i = 0; i < n_vec; i += INTERNAL_VEC_LEN) {
            INTERNAL_VEC v_z = LOAD_VEC(&z_arr[i]);
            INTERNAL_VEC v_Z = MUL(v_z, v_z);
            INTERNAL_VEC v_inner = SUB(v_one, v_Z);

            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                if (v_inner[lane] < (IVEC_SCALAR)0.0) v_inner[lane] = (IVEC_SCALAR)0.0;
            }

            INTERNAL_VEC v_result = M_EXP(ADD(v_log_scale, MUL(v_beta, SUB(M_SQRT(v_inner), v_one))));
            STORE_VEC(&out_arr[i], v_result);
        }
    }

    for (int i = n_vec; i < n; ++i) out_arr[i] = PFX(pswf0)(z_arr[i], w, mode43);
#else
    for (int i = 0; i < n; ++i) out_arr[i] = PFX(pswf0)(z_arr[i], w, mode43);
#endif
}

#if !defined(DOUBLE_DOUBLE)

enum { PSWF_PROL_MAX_M = 160, PSWF_PPOLY_MAX_NC = 16 };

static inline double PFX(hypot2_safe)(double a, double b) {
    double aa = fabs(a);
    double bb = fabs(b);
    if (aa < bb) {
        double tmp = aa;
        aa = bb;
        bb = tmp;
    }
    if (aa == 0.0) return 0.0;
    double r = bb / aa;
    return aa * sqrt(1.0 + r * r);
}

/* Implicit-shift QL for eigenvalues of a symmetric tridiagonal matrix
 * (diag d[0..m-1], subdiag e[0..m-2] with e[m-1] = 0). */
static bool PFX(prol_tql1)(int m, double *d, double *e) {
    e[m - 1] = 0.0;
    for (int l = 0; l < m; ++l) {
        int iter = 0;
        for (;;) {
            int k = l;
            while (k < m - 1) {
                double dd = fabs(d[k]) + fabs(d[k + 1]);
                if (fabs(e[k]) <= DBL_EPSILON * dd) break;
                ++k;
            }
            if (k == l) break;
            if (++iter > 40) return false;

            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = PFX(hypot2_safe)(g, 1.0);
            double shift = (g >= 0.0) ? (g + r) : (g - r);
            g = d[k] - d[l] + e[l] / shift;

            double s = 1.0;
            double c = 1.0;
            double p = 0.0;
            int broken = 0;
            for (int i = k - 1; i >= l; --i) {
                double f = s * e[i];
                double b = c * e[i];
                r = PFX(hypot2_safe)(f, g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[k] = 0.0;
                    broken = 1;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if (!broken) {
                d[l] -= p;
                e[l] = g;
                e[k] = 0.0;
            }
        }
    }
    return true;
}

typedef struct {
    int n_terms;
    double leg_c[PSWF_PROL_MAX_M];
    double rec_a[PSWF_PROL_MAX_M];
    double rec_b[PSWF_PROL_MAX_M];
    double rec_c[PSWF_PROL_MAX_M];
    double inv_val0;
} PFX(pswf_leg_data);

/* Exact Legendre-basis eigensolver for the zeroth prolate spheroidal wave
 * function psi_0^c(x) on [-1, 1], normalized so psi_0^c(0) = 1. */
static bool PFX(pswf_leg_init)(PFX(pswf_leg_data) * out, double c) {
    static const int ns[20] = {48, 64, 80, 92, 106, 120, 130, 144, 156, 168, 178, 190, 202, 214, 224, 236, 248, 258, 268, 280};
    int n_full;
    if (c < 200.0) {
        int idx = (int)(c / 10.0);
        if (idx < 0) idx = 0;
        if (idx > 19) idx = 19;
        n_full = ns[idx];
    } else {
        n_full = (int)(c * 1.5);
    }
    int m = (n_full + 1) / 2;
    if (m < 24) m = 24;
    if (m > PSWF_PROL_MAX_M) m = PSWF_PROL_MAX_M;

    double diag[PSWF_PROL_MAX_M];
    double off[PSWF_PROL_MAX_M];
    double d_work[PSWF_PROL_MAX_M];
    double e_work[PSWF_PROL_MAX_M];

    double c2 = c * c;
    double max_entry = 0.0;
    for (int i = 0; i < m; ++i) {
        double k = 2.0 * (double)i;
        double beta0 = (i == 0) ? (1.0 / 3.0) : (((k + 1.0) * (k + 1.0) / (2.0 * k + 3.0) + k * k / (2.0 * k - 1.0)) / (2.0 * k + 1.0));
        diag[i] = k * (k + 1.0) + c2 * beta0;
        if (fabs(diag[i]) > max_entry) max_entry = fabs(diag[i]);
        if (i < m - 1) {
            double gamma0 = (k + 1.0) * (k + 2.0) / ((2.0 * k + 1.0) * (2.0 * k + 3.0));
            double scale = sqrt((k + 0.5) / (k + 2.5));
            off[i] = c2 * gamma0 * scale;
            if (fabs(off[i]) > max_entry) max_entry = fabs(off[i]);
        } else {
            off[i] = 0.0;
        }
        d_work[i] = diag[i];
        e_work[i] = off[i];
    }

    if (!PFX(prol_tql1)(m, d_work, e_work)) return false;

    double lambda = d_work[0];
    for (int i = 1; i < m; ++i) {
        if (d_work[i] < lambda) lambda = d_work[i];
    }

    /* Tridiagonal LU with partial pivoting on (T - lambda * I) for inverse iteration. */
    double u0[PSWF_PROL_MAX_M];
    double u1[PSWF_PROL_MAX_M];
    double u2[PSWF_PROL_MAX_M];
    double lfac[PSWF_PROL_MAX_M];
    unsigned char swap_row[PSWF_PROL_MAX_M];

    double sub_next = (m > 1) ? off[0] : 0.0;
    double cur0 = diag[0] - lambda;
    double cur1 = (m > 1) ? off[0] : 0.0;
    double piv_floor = (max_entry > 0.0 ? max_entry : 1.0) * 1e-30;

    for (int i = 0; i < m - 1; ++i) {
        double next0 = diag[i + 1] - lambda;
        double next1 = (i + 2 < m) ? off[i + 1] : 0.0;
        if (fabs(sub_next) > fabs(cur0)) {
            swap_row[i] = 1;
            u0[i] = sub_next;
            u1[i] = next0;
            u2[i] = next1;
            double fac = cur0 / u0[i];
            lfac[i] = fac;
            cur0 = cur1 - fac * u1[i];
            cur1 = -fac * u2[i];
        } else {
            swap_row[i] = 0;
            if (fabs(cur0) < piv_floor) cur0 = (cur0 >= 0.0) ? piv_floor : -piv_floor;
            u0[i] = cur0;
            u1[i] = cur1;
            u2[i] = 0.0;
            double fac = sub_next / u0[i];
            lfac[i] = fac;
            cur0 = next0 - fac * u1[i];
            cur1 = next1;
        }
        sub_next = (i + 2 < m) ? off[i + 1] : 0.0;
    }
    if (fabs(cur0) < piv_floor) cur0 = (cur0 >= 0.0) ? piv_floor : -piv_floor;
    u0[m - 1] = cur0;
    u1[m - 1] = 0.0;
    u2[m - 1] = 0.0;
    swap_row[m - 1] = 0;
    lfac[m - 1] = 0.0;

    double vec[PSWF_PROL_MAX_M];
    for (int i = 0; i < m; ++i) vec[i] = 1.0 / (double)(i + 1);

    for (int it = 0; it < 4; ++it) {
        for (int i = 0; i < m - 1; ++i) {
            if (swap_row[i]) {
                double tmp = vec[i];
                vec[i] = vec[i + 1];
                vec[i + 1] = tmp;
            }
            vec[i + 1] -= lfac[i] * vec[i];
        }
        vec[m - 1] /= u0[m - 1];
        if (m >= 2) vec[m - 2] = (vec[m - 2] - u1[m - 2] * vec[m - 1]) / u0[m - 2];
        for (int i = m - 3; i >= 0; --i) vec[i] = (vec[i] - u1[i] * vec[i + 1] - u2[i] * vec[i + 2]) / u0[i];

        double norm2 = 0.0;
        for (int i = 0; i < m; ++i) norm2 += vec[i] * vec[i];
        if (!(norm2 > 0.0)) return false;
        double inv_norm = 1.0 / sqrt(norm2);
        for (int i = 0; i < m; ++i) vec[i] *= inv_norm;
    }

    double max_c = 0.0;
    for (int i = 0; i < m; ++i) {
        out->leg_c[i] = vec[i] * sqrt(2.0 * (double)i + 0.5);
        double ac = fabs(out->leg_c[i]);
        if (ac > max_c) max_c = ac;
    }
    if (!(max_c > 0.0)) return false;

    int n_terms = m;
    double trim_tol = max_c * 1e-16;
    while (n_terms > 1 && fabs(out->leg_c[n_terms - 1]) <= trim_tol) --n_terms;
    out->n_terms = n_terms;

    /* 2-step even-Legendre recurrence coefficients for P_{2(i+1)}(x) from P_{2i}, P_{2(i-1)}. */
    for (int i = 1; i < n_terms - 1; ++i) {
        double n = 2.0 * (double)i;
        double denom_a = (n + 2.0) * (n + 1.0);
        double denom_bc = denom_a * (2.0 * n - 1.0);
        out->rec_a[i] = (2.0 * n + 3.0) * (2.0 * n + 1.0) / denom_a;
        out->rec_b[i] = -(2.0 * n + 1.0) * (2.0 * n * n + 2.0 * n - 1.0) / denom_bc;
        out->rec_c[i] = -n * (n - 1.0) * (2.0 * n + 3.0) / denom_bc;
    }

    out->inv_val0 = 1.0;
    double val0 = out->leg_c[0];
    if (n_terms > 1) {
        double p_prev = 1.0;
        double p_curr = -0.5;
        val0 += out->leg_c[1] * p_curr;
        for (int i = 1; i < n_terms - 1; ++i) {
            double p_next = out->rec_b[i] * p_curr + out->rec_c[i] * p_prev;
            val0 += out->leg_c[i + 1] * p_next;
            p_prev = p_curr;
            p_curr = p_next;
        }
    }
    if (fabs(val0) < 1e-30) return false;
    out->inv_val0 = 1.0 / val0;
    return true;
}

static inline double PFX(pswf_leg_eval)(const PFX(pswf_leg_data) * data, double x) {
    double x2 = x * x;
    double sum = data->leg_c[0];
    if (data->n_terms > 1) {
        double p_prev = 1.0;
        double p_curr = 1.5 * x2 - 0.5;
        sum += data->leg_c[1] * p_curr;
        for (int i = 1; i < data->n_terms - 1; ++i) {
            double p_next = (data->rec_a[i] * x2 + data->rec_b[i]) * p_curr + data->rec_c[i] * p_prev;
            sum += data->leg_c[i + 1] * p_next;
            p_prev = p_curr;
            p_curr = p_next;
        }
    }
    return sum * data->inv_val0;
}

/* Build transposed piecewise-polynomial coefficients on [-1, 1] for w equal
 * panels via Type-1 Chebyshev interpolation of exact Legendre psi_0^c.
 * Output layout: coeffs[k * stride + j] with k=0 highest power, k=nc-1 const,
 * nc always even in [4, PSWF_PPOLY_MAX_NC], tail lanes [w..stride-1] zeroed. */
static bool PFX(pswf_build_ppoly)(int w, bool mode43, int *out_nc, int *out_stride, FLOAT **out_coeffs) {
    if (w < 1 || w > PSWF_MAX_W) return false;

    double c = NANOFFT_PI * (double)w * (mode43 ? 0.625 : 0.75) - 0.05;
    PFX(pswf_leg_data) leg;
    if (!PFX(pswf_leg_init)(&leg, c)) return false;

    int nc_fit = w + 3;
    if (nc_fit > PSWF_PPOLY_MAX_NC) nc_fit = PSWF_PPOLY_MAX_NC;
    if (nc_fit < 4) nc_fit = 4;

    double nodes[PSWF_PPOLY_MAX_NC];
    for (int k = 0; k < nc_fit; ++k) {
        nodes[k] = cos(NANOFFT_PI * (double)(2 * k + 1) / (double)(2 * nc_fit));
    }

    int half_w = (w + 1) / 2;
    double panel_mono[PSWF_MAX_W][PSWF_PPOLY_MAX_NC];
#    if defined(DOUBLE)
    const double tol = 1e-15;
#    elif defined(HALF)
    const double tol = 2e-4;
#    else
    const double tol = 1e-7;
#    endif
    const double cutoff = tol * 0.05;
    int nc_needed = 4;

    double inv_w = 1.0 / (double)w;
    for (int j = 0; j < half_w; ++j) {
        double shift = (double)(2 * j + 1 - w);
        double dd[PSWF_PPOLY_MAX_NC];
        for (int k = 0; k < nc_fit; ++k) {
            double z = (nodes[k] + shift) * inv_w;
            dd[k] = PFX(pswf_leg_eval)(&leg, z);
        }
        /* Newton divided differences in place. */
        for (int step = 1; step < nc_fit; ++step) {
            for (int k = nc_fit - 1; k >= step; --k) {
                dd[k] = (dd[k] - dd[k - 1]) / (nodes[k] - nodes[k - step]);
            }
        }
        /* Convert Newton form to ascending monomial coefficients mono[0..nc_fit-1]. */
        double mono[PSWF_PPOLY_MAX_NC];
        for (int k = 0; k < nc_fit; ++k) mono[k] = 0.0;
        mono[0] = dd[nc_fit - 1];
        for (int step = nc_fit - 2; step >= 0; --step) {
            double xk = nodes[step];
            for (int k = nc_fit - 1; k >= 1; --k) mono[k] = mono[k - 1] - xk * mono[k];
            mono[0] = dd[step] - xk * mono[0];
        }

        int deg_j = nc_fit - 1;
        while (deg_j >= 4 && fabs(mono[deg_j]) < cutoff) --deg_j;
        if (deg_j + 1 > nc_needed) nc_needed = deg_j + 1;

        for (int k = 0; k < nc_fit; ++k) panel_mono[j][k] = mono[k];
        int mirror = w - 1 - j;
        if (mirror != j) {
            for (int k = 0; k < nc_fit; ++k) panel_mono[mirror][k] = (k & 1) ? -mono[k] : mono[k];
        }
    }

    /* Keep nc even so the symmetric Horner split (k_odd / k_even) is branch-free. */
    int nc = nc_needed;
    if (nc & 1) ++nc;
    if (nc > PSWF_PPOLY_MAX_NC) nc = PSWF_PPOLY_MAX_NC;

    int stride = PFX(round_up_int)(w, INTERNAL_VEC_LEN);
    FLOAT *coeffs = NULL;
    if (out_coeffs) {
        coeffs = PFX(allocate_aligned)((size_t)nc * (size_t)stride);
        if (!coeffs) return false;
        memset(coeffs, 0, (size_t)nc * (size_t)stride * sizeof(FLOAT));
        const double kernel_scale = ldexp(1.0, PFX(pswf_kernel_scale_exp)(w, mode43, true));
        for (int k = 0; k < nc; ++k) {
            int power = (nc - 1) - k;
            for (int j = 0; j < w; ++j) {
                double val = (power < nc_fit) ? panel_mono[j][power] : 0.0;
                coeffs[(size_t)k * (size_t)stride + (size_t)j] = FCAST(val * kernel_scale);
            }
        }
        *out_coeffs = coeffs;
    }
    *out_nc = nc;
    *out_stride = stride;
    return true;
}

static inline FLOAT PFX(ppoly_eval_scalar)(FLOAT z, int w, int nc, int stride, const FLOAT *coeffs) {
    double zd = TO_DOUBLE(z);
    if (zd < -1.0) zd = -1.0;
    if (zd > 1.0) zd = 1.0;
    double s = 0.5 * (double)w * (zd + 1.0);
    int j = (int)floor(s);
    if (j < 0) j = 0;
    if (j >= w) j = w - 1;
    FLOAT u = FCAST((double)w * zd - (double)(2 * j + 1 - w));
    FLOAT val = coeffs[j];
    for (int k = 1; k < nc; ++k) val = M_FMA(val, u, coeffs[(size_t)k * (size_t)stride + (size_t)j]);
    return val;
}

int PFX(pswf_bench_ppoly_gen)(int w, const char upsamp[2], int reps) {
    if (!upsamp || reps <= 0) return -1;
    bool mode21 = (upsamp[0] == '2' && upsamp[1] == '1');
    bool mode43 = (upsamp[0] == '4' && upsamp[1] == '3');
    if (!mode21 && !mode43) return -1;
    int nc = 0;
    int stride = 0;
    for (int r = 0; r < reps; ++r) {
        FLOAT *coeffs = NULL;
        if (!PFX(pswf_build_ppoly)(w, mode43, &nc, &stride, &coeffs)) return -1;
        nanofft_aligned_free(coeffs);
    }
    return nc;
}

static inline INTERNAL_VEC PFX(load_vec_u)(const FLOAT *p) {
#    if defined(HALF) && !NANOFFT_NATIVE_FP16
    return LOAD_VEC(p);
#    else
    INTERNAL_VEC v;
    memcpy(&v, p, sizeof(v));
    return v;
#    endif
}

static inline void PFX(store_vec_u)(FLOAT *p, INTERNAL_VEC v) {
#    if defined(HALF) && !NANOFFT_NATIVE_FP16
    STORE_VEC(p, v);
#    else
    memcpy(p, &v, sizeof(v));
#    endif
}

static inline INTERNAL_VEC PFX(rev_vec)(INTERNAL_VEC v) {
#    if INTERNAL_VEC_LEN == 2
    return NANOFFT_SHUFFLE1(INTERNAL_VEC_INT, v, 1, 0);
#    elif INTERNAL_VEC_LEN == 4
    return NANOFFT_SHUFFLE1(INTERNAL_VEC_INT, v, 3, 2, 1, 0);
#    elif INTERNAL_VEC_LEN == 8
    return NANOFFT_SHUFFLE1(INTERNAL_VEC_INT, v, 7, 6, 5, 4, 3, 2, 1, 0);
#    elif INTERNAL_VEC_LEN == 16
    return NANOFFT_SHUFFLE1(INTERNAL_VEC_INT, v, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#    elif INTERNAL_VEC_LEN == 32
    return NANOFFT_SHUFFLE1(INTERNAL_VEC_INT, v, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                            1, 0);
#    else
    INTERNAL_VEC out;
    for (int i = 0; i < INTERNAL_VEC_LEN; ++i) out[i] = v[INTERNAL_VEC_LEN - 1 - i];
    return out;
#    endif
}

static inline void PFX(scatter_vec_u)(FLOAT *NANOFFT_RESTRICT fr, FLOAT *NANOFFT_RESTRICT fi, int idx, INTERNAL_VEC v_sy_r, INTERNAL_VEC v_sy_i,
                                      INTERNAL_VEC kval) {
    INTERNAL_VEC gr = PFX(load_vec_u)(&fr[idx]);
    INTERNAL_VEC gi = PFX(load_vec_u)(&fi[idx]);
    PFX(store_vec_u)(&fr[idx], ADD(gr, MUL(v_sy_r, kval)));
    PFX(store_vec_u)(&fi[idx], ADD(gi, MUL(v_sy_i, kval)));
}

/* Vectorized symmetric even/odd Horner evaluation fused directly with grid
 * scatter-add (no stack scratch buffer, no partial-overlap store-forwarding
 * stalls, no wasted padded-tail RMWs).  Requires nc even >= 2. */
static inline void PFX(ppoly_spread_point)(FLOAT *NANOFFT_RESTRICT fr, FLOAT *NANOFFT_RESTRICT fi, const FLOAT *NANOFFT_RESTRICT coeffs, int first, FLOAT u,
                                           FLOAT sy_r, FLOAT sy_i, int w, int nc, int stride) {
    const INTERNAL_VEC v_u = IVEC_SPLAT(u);
    const INTERNAL_VEC v_u2 = MUL(v_u, v_u);
    const INTERNAL_VEC v_sy_r = IVEC_SPLAT(sy_r);
    const INTERNAL_VEC v_sy_i = IVEC_SPLAT(sy_i);
    const int V = INTERNAL_VEC_LEN;

    if (w <= V) {
        INTERNAL_VEC k_odd = LOAD_VEC(&coeffs[0]);
        INTERNAL_VEC k_even = LOAD_VEC(&coeffs[stride]);
        for (int k = 2; k < nc; k += 2) {
            k_odd = ADD(MUL(k_odd, v_u2), LOAD_VEC(&coeffs[(size_t)k * (size_t)stride]));
            k_even = ADD(MUL(k_even, v_u2), LOAD_VEC(&coeffs[(size_t)(k + 1) * (size_t)stride]));
        }
        PFX(scatter_vec_u)(fr, fi, first, v_sy_r, v_sy_i, ADD(k_even, MUL(v_u, k_odd)));
        return;
    }

    const int n_sym = (w / 2) / V;
    const int mid = n_sym * V;
    const int rem = w - 2 * mid;

    for (int ii = 0; ii < mid; ii += V) {
        INTERNAL_VEC k_odd = LOAD_VEC(&coeffs[ii]);
        INTERNAL_VEC k_even = LOAD_VEC(&coeffs[(size_t)stride + (size_t)ii]);
        for (int k = 2; k < nc; k += 2) {
            k_odd = ADD(MUL(k_odd, v_u2), LOAD_VEC(&coeffs[(size_t)k * (size_t)stride + (size_t)ii]));
            k_even = ADD(MUL(k_even, v_u2), LOAD_VEC(&coeffs[(size_t)(k + 1) * (size_t)stride + (size_t)ii]));
        }
        INTERNAL_VEC u_odd = MUL(v_u, k_odd);
        PFX(scatter_vec_u)(fr, fi, first + ii, v_sy_r, v_sy_i, ADD(k_even, u_odd));
        PFX(scatter_vec_u)(fr, fi, first + (w - ii - V), v_sy_r, v_sy_i, PFX(rev_vec)(SUB(k_even, u_odd)));
    }

    if (rem == 0) return;

    if (rem == 1) {
        FLOAT u2 = MUL(u, u);
        FLOAT ke = coeffs[(size_t)stride + (size_t)mid];
        for (int k = 3; k < nc; k += 2) ke = ADD(MUL(ke, u2), coeffs[(size_t)k * (size_t)stride + (size_t)mid]);
        fr[first + mid] = ADD(fr[first + mid], MUL(sy_r, ke));
        fi[first + mid] = ADD(fi[first + mid], MUL(sy_i, ke));
        return;
    }

    if (rem == 2) {
        FLOAT u2 = MUL(u, u);
        FLOAT ko = coeffs[mid];
        FLOAT ke = coeffs[(size_t)stride + (size_t)mid];
        for (int k = 2; k < nc; k += 2) {
            ko = ADD(MUL(ko, u2), coeffs[(size_t)k * (size_t)stride + (size_t)mid]);
            ke = ADD(MUL(ke, u2), coeffs[(size_t)(k + 1) * (size_t)stride + (size_t)mid]);
        }
        FLOAT uo = MUL(u, ko);
        FLOAT kl = ADD(ke, uo);
        FLOAT kr = SUB(ke, uo);
        fr[first + mid] = ADD(fr[first + mid], MUL(sy_r, kl));
        fi[first + mid] = ADD(fi[first + mid], MUL(sy_i, kl));
        fr[first + mid + 1] = ADD(fr[first + mid + 1], MUL(sy_r, kr));
        fi[first + mid + 1] = ADD(fi[first + mid + 1], MUL(sy_i, kr));
        return;
    }

    if (rem >= V) {
        INTERNAL_VEC k_odd = LOAD_VEC(&coeffs[mid]);
        INTERNAL_VEC k_even = LOAD_VEC(&coeffs[(size_t)stride + (size_t)mid]);
        for (int k = 2; k < nc; k += 2) {
            k_odd = ADD(MUL(k_odd, v_u2), LOAD_VEC(&coeffs[(size_t)k * (size_t)stride + (size_t)mid]));
            k_even = ADD(MUL(k_even, v_u2), LOAD_VEC(&coeffs[(size_t)(k + 1) * (size_t)stride + (size_t)mid]));
        }
        INTERNAL_VEC u_odd = MUL(v_u, k_odd);
        PFX(scatter_vec_u)(fr, fi, first + mid, v_sy_r, v_sy_i, ADD(k_even, u_odd));
        if (rem > V) {
            INTERNAL_VEC v_sym = SUB(k_even, u_odd);
            int tail = rem - V;
            for (int t = 0; t < tail; ++t) {
                FLOAT kt = FCAST(v_sym[t]);
                int idx = first + w - mid - 1 - t;
                fr[idx] = ADD(fr[idx], MUL(sy_r, kt));
                fi[idx] = ADD(fi[idx], MUL(sy_i, kt));
            }
        }
        return;
    }

    /* Fallback for rare 2 < rem < V (e.g., non-standard odd/small w). */
    FLOAT u2 = MUL(u, u);
    for (int j = 0; j < rem / 2; ++j) {
        int col = mid + j;
        FLOAT ko = coeffs[col];
        FLOAT ke = coeffs[(size_t)stride + (size_t)col];
        for (int k = 2; k < nc; k += 2) {
            ko = ADD(MUL(ko, u2), coeffs[(size_t)k * (size_t)stride + (size_t)col]);
            ke = ADD(MUL(ke, u2), coeffs[(size_t)(k + 1) * (size_t)stride + (size_t)col]);
        }
        FLOAT uo = MUL(u, ko);
        FLOAT kl = ADD(ke, uo);
        FLOAT kr = SUB(ke, uo);
        int idx_l = first + col;
        int idx_r = first + w - 1 - col;
        fr[idx_l] = ADD(fr[idx_l], MUL(sy_r, kl));
        fi[idx_l] = ADD(fi[idx_l], MUL(sy_i, kl));
        fr[idx_r] = ADD(fr[idx_r], MUL(sy_r, kr));
        fi[idx_r] = ADD(fi[idx_r], MUL(sy_i, kr));
    }
    if (rem & 1) {
        int col = mid + rem / 2;
        FLOAT ke = coeffs[(size_t)stride + (size_t)col];
        for (int k = 3; k < nc; k += 2) ke = ADD(MUL(ke, u2), coeffs[(size_t)k * (size_t)stride + (size_t)col]);
        fr[first + col] = ADD(fr[first + col], MUL(sy_r, ke));
        fi[first + col] = ADD(fi[first + col], MUL(sy_i, ke));
    }
}

/* Vectorized symmetric even/odd Horner evaluation of all w kernel weights for
 * one point into aligned stack storage `kval[0..w-1]`, shared across all
 * BREADTH_VEC_LEN lanes in breadth spreading. */
static inline void PFX(ppoly_eval_weights)(FLOAT *NANOFFT_RESTRICT kval, const FLOAT *NANOFFT_RESTRICT coeffs, FLOAT u, int w, int nc, int stride) {
    const INTERNAL_VEC v_u = IVEC_SPLAT(u);
    const INTERNAL_VEC v_u2 = MUL(v_u, v_u);
    const int V = INTERNAL_VEC_LEN;

    if (w <= V) {
        INTERNAL_VEC k_odd = LOAD_VEC(&coeffs[0]);
        INTERNAL_VEC k_even = LOAD_VEC(&coeffs[stride]);
        for (int k = 2; k < nc; k += 2) {
            k_odd = ADD(MUL(k_odd, v_u2), LOAD_VEC(&coeffs[(size_t)k * (size_t)stride]));
            k_even = ADD(MUL(k_even, v_u2), LOAD_VEC(&coeffs[(size_t)(k + 1) * (size_t)stride]));
        }
        STORE_VEC(&kval[0], ADD(k_even, MUL(v_u, k_odd)));
        return;
    }

    const int n_sym = (w / 2) / V;
    const int mid = n_sym * V;
    const int rem = w - 2 * mid;

    for (int ii = 0; ii < mid; ii += V) {
        INTERNAL_VEC k_odd = LOAD_VEC(&coeffs[ii]);
        INTERNAL_VEC k_even = LOAD_VEC(&coeffs[(size_t)stride + (size_t)ii]);
        for (int k = 2; k < nc; k += 2) {
            k_odd = ADD(MUL(k_odd, v_u2), LOAD_VEC(&coeffs[(size_t)k * (size_t)stride + (size_t)ii]));
            k_even = ADD(MUL(k_even, v_u2), LOAD_VEC(&coeffs[(size_t)(k + 1) * (size_t)stride + (size_t)ii]));
        }
        const INTERNAL_VEC u_odd = MUL(v_u, k_odd);
        PFX(store_vec_u)(&kval[ii], ADD(k_even, u_odd));
        PFX(store_vec_u)(&kval[w - ii - V], PFX(rev_vec)(SUB(k_even, u_odd)));
    }

    if (rem == 0) return;

    if (rem == 1) {
        const FLOAT u2 = MUL(u, u);
        FLOAT ke = coeffs[(size_t)stride + (size_t)mid];
        for (int k = 3; k < nc; k += 2) ke = ADD(MUL(ke, u2), coeffs[(size_t)k * (size_t)stride + (size_t)mid]);
        kval[mid] = ke;
        return;
    }

    if (rem == 2) {
        const FLOAT u2 = MUL(u, u);
        FLOAT ko = coeffs[mid];
        FLOAT ke = coeffs[(size_t)stride + (size_t)mid];
        for (int k = 2; k < nc; k += 2) {
            ko = ADD(MUL(ko, u2), coeffs[(size_t)k * (size_t)stride + (size_t)mid]);
            ke = ADD(MUL(ke, u2), coeffs[(size_t)(k + 1) * (size_t)stride + (size_t)mid]);
        }
        const FLOAT uo = MUL(u, ko);
        kval[mid] = ADD(ke, uo);
        kval[mid + 1] = SUB(ke, uo);
        return;
    }

    if (rem >= V) {
        INTERNAL_VEC k_odd = LOAD_VEC(&coeffs[mid]);
        INTERNAL_VEC k_even = LOAD_VEC(&coeffs[(size_t)stride + (size_t)mid]);
        for (int k = 2; k < nc; k += 2) {
            k_odd = ADD(MUL(k_odd, v_u2), LOAD_VEC(&coeffs[(size_t)k * (size_t)stride + (size_t)mid]));
            k_even = ADD(MUL(k_even, v_u2), LOAD_VEC(&coeffs[(size_t)(k + 1) * (size_t)stride + (size_t)mid]));
        }
        const INTERNAL_VEC u_odd = MUL(v_u, k_odd);
        PFX(store_vec_u)(&kval[mid], ADD(k_even, u_odd));
        if (rem > V) {
            const INTERNAL_VEC v_sym = SUB(k_even, u_odd);
            const int tail = rem - V;
            for (int t = 0; t < tail; ++t) kval[w - mid - 1 - t] = FCAST(v_sym[t]);
        }
        return;
    }

    const FLOAT u2 = MUL(u, u);
    for (int j = 0; j < rem / 2; ++j) {
        const int col = mid + j;
        FLOAT ko = coeffs[col];
        FLOAT ke = coeffs[(size_t)stride + (size_t)col];
        for (int k = 2; k < nc; k += 2) {
            ko = ADD(MUL(ko, u2), coeffs[(size_t)k * (size_t)stride + (size_t)col]);
            ke = ADD(MUL(ke, u2), coeffs[(size_t)(k + 1) * (size_t)stride + (size_t)col]);
        }
        const FLOAT uo = MUL(u, ko);
        kval[col] = ADD(ke, uo);
        kval[w - 1 - col] = SUB(ke, uo);
    }
    if (rem & 1) {
        const int col = mid + rem / 2;
        FLOAT ke = coeffs[(size_t)stride + (size_t)col];
        for (int k = 3; k < nc; k += 2) ke = ADD(MUL(ke, u2), coeffs[(size_t)k * (size_t)stride + (size_t)col]);
        kval[col] = ke;
    }
}

#endif /* !defined(DOUBLE_DOUBLE) */

struct PFX(pswf_plan) * PFX(pswf_initialize_opts)(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2], int spread_mode, int vec_mode) {
    if (!upsamp) return NULL;

    bool mode21 = (upsamp[0] == '2' && upsamp[1] == '1');
    bool mode43 = (upsamp[0] == '4' && upsamp[1] == '3');
    if (!mode21 && !mode43) return NULL;
    if (mode43 && (N % 4) != 0) return NULL;
    if (w < 1 || w > PSWF_MAX_W) return NULL;

    struct PFX(pswf_plan) *plan = (struct PFX(pswf_plan) *)nanofft_calloc(1, sizeof(struct PFX(pswf_plan)));
    if (!plan) return NULL;
    plan->Mpoints = Mpoints;
    plan->N = N;
    plan->Nout = mode43 ? N + (N >> 1) : N;
    plan->output_shift = mode43 ? 3 * N / 4 : N / 2;
    plan->w = w;
    plan->upsamp[0] = upsamp[0];
    plan->upsamp[1] = upsamp[1];
    plan->num_factors = freq_factor;
    plan->df = df;
    plan->is_breadth = (vec_mode == TLS_VEC_BREADTH) ? 1 : 0;
    plan->Nfft = next_fast_len(2 * N);
#if defined(HALF)
    plan->alpha = FCAST((double)plan->w * 0.5 / (double)plan->Nfft);
#else
    plan->alpha = DIV(MUL(FCAST(plan->w), FCONST(0.5)), FCAST(plan->Nfft));
#endif

#if !defined(DOUBLE_DOUBLE)
    if (spread_mode == TLS_SPREAD_PIECEWISE) {
        if (PFX(pswf_build_ppoly)(w, mode43, &plan->ppoly_nc, &plan->ppoly_stride, &plan->ppoly_coeffs)) {
            plan->use_ppoly = 1;
        }
    }
#else
    (void)spread_mode;
#endif

    if (plan->is_breadth) {
        if (plan->use_ppoly) {
            plan->spread_stride = plan->ppoly_stride;
            plan->spread_pad = plan->w;
            plan->fft_alloc_len = plan->Nfft + plan->spread_pad;
            plan->spread_first_idx = (int *)nanofft_malloc((size_t)Mpoints * (size_t)freq_factor * sizeof(int));
            plan->spread_z = (FLOAT *)nanofft_malloc((size_t)Mpoints * (size_t)freq_factor * sizeof(FLOAT));
        } else {
            plan->spread_stride = plan->w;
            plan->spread_pad = plan->w;
            plan->fft_alloc_len = plan->Nfft + plan->spread_pad;
            plan->spread_first_idx = (int *)nanofft_malloc((size_t)Mpoints * (size_t)freq_factor * sizeof(int));
            plan->spread_weight = PFX(allocate_aligned)((size_t)Mpoints * (size_t)plan->w * (size_t)freq_factor);
        }
        plan->deconv = (FLOAT *)nanofft_malloc((size_t)plan->Nout * sizeof(FLOAT));
        plan->bfft_real = PFX(allocate_aligned_bvec)((size_t)plan->fft_alloc_len);
        plan->bfft_imag = PFX(allocate_aligned_bvec)((size_t)plan->fft_alloc_len);
        plan->nanofft_p = NANOFFT_MAKE_PLAN_BREADTH((uint32_t)plan->Nfft);

        plan->slots = plan->Nfft;
        plan->valid_slots = plan->Nout;
        plan->period = 0;
        plan->nvalid = 0;
        plan->slot_deconv = PFX(allocate_aligned)((size_t)plan->slots);
        plan->bperm_scratch = PFX(allocate_aligned_bvec)((size_t)plan->slots);
        return plan;
    }

    if (plan->use_ppoly) {
        plan->spread_stride = plan->ppoly_stride;
        plan->spread_pad = plan->ppoly_stride;
        plan->fft_alloc_len = plan->Nfft + plan->spread_pad;
        plan->spread_first_idx = (int *)nanofft_malloc((size_t)Mpoints * (size_t)freq_factor * sizeof(int));
        plan->spread_z = (FLOAT *)nanofft_malloc((size_t)Mpoints * (size_t)freq_factor * sizeof(FLOAT));
    } else {
        plan->spread_stride = PFX(round_up_int)(plan->w + INTERNAL_VEC_LEN - 1, INTERNAL_VEC_LEN);
        plan->spread_pad = plan->spread_stride;
        plan->fft_alloc_len = plan->Nfft + plan->spread_pad;
        plan->spread_base_idx = (int *)nanofft_malloc((size_t)Mpoints * (size_t)freq_factor * sizeof(int));
        plan->spread_weight = PFX(allocate_aligned)((size_t)Mpoints * (size_t)plan->spread_stride * (size_t)freq_factor);
    }
    plan->deconv = (FLOAT *)nanofft_malloc((size_t)plan->Nout * sizeof(FLOAT));

    plan->shift_r = (FLOAT *)nanofft_malloc((size_t)Mpoints * (size_t)freq_factor * sizeof(FLOAT));
    plan->shift_i = (FLOAT *)nanofft_malloc((size_t)Mpoints * (size_t)freq_factor * sizeof(FLOAT));

    plan->fft_real = PFX(allocate_aligned)((size_t)plan->fft_alloc_len);
    plan->fft_imag = PFX(allocate_aligned)((size_t)plan->fft_alloc_len);
    plan->nanofft_p = NANOFFT_MAKE_PLAN((uint32_t)plan->Nfft);

    plan->slots = plan->Nfft;
    plan->valid_slots = plan->Nout;
    plan->period = 0;
    plan->nvalid = 0;
    plan->slot_deconv = PFX(allocate_aligned)((size_t)plan->slots);
    plan->perm_scratch = PFX(allocate_aligned)((size_t)plan->slots);
    return plan;
}

struct PFX(pswf_plan) * PFX(pswf_initialize_mode)(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2], int spread_mode) {
    return PFX(pswf_initialize_opts)(Mpoints, N, w, df, freq_factor, upsamp, spread_mode, TLS_VEC_DEPTH);
}

struct PFX(pswf_plan) * PFX(pswf_initialize)(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2]) {
    int mode = (tls_tuning_spread() == TLS_SPREAD_PIECEWISE) ? TLS_SPREAD_PIECEWISE : TLS_SPREAD_MINMAX;
    return PFX(pswf_initialize_mode)(Mpoints, N, w, df, freq_factor, upsamp, mode);
}

/* =========================================================================
 * Slot-order (deferred bit reversal) support
 * ========================================================================= */

static inline uint32_t PFX(nfft_log2)(uint32_t n) {
    uint32_t bits = 0;
    while ((1u << bits) < n) ++bits;
    return bits;
}

/*
 * Slot p holds FFT bin reverse_bits(p).  The kept bins are the two contiguous
 * runs at the ends of the spectrum, so their bit-reversed image repeats with a
 * small power-of-two period; only the kept offsets inside one period are ever
 * written out.  Bins the output window never maps (the oversampling padding)
 * get a zero deconvolution factor.
 */
static void PFX(pswf_build_slot_tables)(struct PFX(pswf_plan) * plan) {
    const int nfft = plan->Nfft;
    const uint32_t log_nfft = PFX(nfft_log2)((uint32_t)nfft);
    const int first_bin = nfft - plan->output_shift;
    const int tail = plan->Nout - plan->output_shift;
    FLOAT *slot_deconv = plan->slot_deconv;

    plan->period = 0;
    plan->nvalid = 0;
    plan->slot_tables_built = 1;
    if (!slot_deconv || !plan->deconv) return;

    for (int p = 0; p < nfft; ++p) {
        const int bin = (int)nanofft_reverse_bits((uint32_t)p, log_nfft);
        if (bin >= first_bin) {
            slot_deconv[p] = plan->deconv[bin - first_bin];
        } else if (bin < tail) {
            slot_deconv[p] = plan->deconv[bin + plan->output_shift];
        } else {
            slot_deconv[p] = FCAST(0.0); /* NuFFT padding: never mapped */
        }
    }

    int period = 0;
    for (int candidate = 1; candidate <= (int)sizeof(plan->slot_offsets) && candidate <= nfft; candidate <<= 1) {
        bool uniform = true;
        for (int p = candidate; p < nfft && uniform; ++p) {
            const bool here = TO_DOUBLE(slot_deconv[p]) != 0.0;
            const bool folded = TO_DOUBLE(slot_deconv[p % candidate]) != 0.0;
            uniform = (here == folded);
        }
        if (uniform) {
            period = candidate;
            break;
        }
    }
    if (period == 0) return; /* no small period: stay in frequency order */

    int nvalid = 0;
    for (int r = 0; r < period; ++r) {
        if (TO_DOUBLE(slot_deconv[r]) != 0.0) plan->slot_offsets[nvalid++] = (unsigned char)r;
    }
    if ((long long)nvalid * (long long)(nfft / period) != (long long)plan->Nout) return; /* defensive */

    plan->period = period;
    plan->nvalid = nvalid;
}

void PFX(pswf_precompute)(struct PFX(pswf_plan) * plan, const nufft_input_t *x) {
    bool mode43 = (plan->upsamp[0] == '4' && plan->upsamp[1] == '3');
    int p = mode43 ? (4 * plan->w + 16) : (int)(1.5 * plan->w + 2);
    int num_gl_nodes = 2 * p;

    FLOAT *gl_nodes = PFX(allocate_aligned)(num_gl_nodes);
    FLOAT *gl_weights = PFX(allocate_aligned)(num_gl_nodes);
    FLOAT *precomp_vals = PFX(allocate_aligned)(num_gl_nodes);

    PFX(get_lege_roots)(num_gl_nodes, gl_nodes, gl_weights);
#if !defined(DOUBLE_DOUBLE)
    if (plan->use_ppoly) {
        for (int j = 0; j < p; ++j) precomp_vals[j] = PFX(ppoly_eval_scalar)(gl_nodes[j], plan->w, plan->ppoly_nc, plan->ppoly_stride, plan->ppoly_coeffs);
    } else {
        PFX(pswf0_batch)(gl_nodes, precomp_vals, p, plan->w, mode43);
    }
#else
    PFX(pswf0_batch)(gl_nodes, precomp_vals, p, plan->w, mode43);
#endif

    const FLOAT fw = FCAST(plan->w);
    for (int j = 0; j < p; ++j) precomp_vals[j] = MUL(MUL(fw, gl_weights[j]), precomp_vals[j]);

#if defined(HALF)
    {
        const double alpha_d = (double)plan->w * 0.5 / (double)plan->Nfft;
        for (int k = 0; k < plan->Nout; ++k) {
            const double xi_k = alpha_d * (double)(k - plan->output_shift);
            double phi_hat_half = 0.0;
            for (int j = 0; j < p; ++j) {
                phi_hat_half += TO_DOUBLE(precomp_vals[j]) * cos2pi(xi_k * TO_DOUBLE(gl_nodes[j]));
            }
            plan->deconv[k] = FCAST(1.0 / phi_hat_half);
        }
    }
#else
    for (int k = 0; k < plan->Nout; ++k) {
        FLOAT xi_k = MUL(plan->alpha, FCAST(k - plan->output_shift));
        FLOAT phi_hat_half = FCAST(0.0);

        for (int j = 0; j < p; ++j) phi_hat_half = ADD(phi_hat_half, MUL(precomp_vals[j], M_COS2PI(MUL(xi_k, gl_nodes[j]))));

        plan->deconv[k] = DIV(FCAST(1.0), phi_hat_half);
    }
#endif

    nanofft_aligned_free(precomp_vals);
    nanofft_aligned_free(gl_nodes);
    nanofft_aligned_free(gl_weights);

    FLOAT z_buf[64] __attribute__((aligned(64)));
    FLOAT kval_buf[64] __attribute__((aligned(64)));

    for (int f_idx = 0; f_idx < plan->num_factors; ++f_idx) {
        int k_factor = f_idx + 1;
        size_t offset_shift = (size_t)f_idx * (size_t)plan->Mpoints;
        size_t offset_base = (size_t)f_idx * (size_t)plan->Mpoints;

#if defined(DOUBLE_DOUBLE)
        size_t offset_spread = offset_base * (size_t)plan->spread_stride;
        FLOAT *current_spread_weight = plan->spread_weight + offset_spread;
        PFX(ff_t) n_ff = PFX(ff_make)(FCAST(plan->Nfft), FCAST(0.0));
        PFX(ff_t) scale_ff = PFX(ff_make)(FCAST(plan->df * k_factor), FCAST(0.0));

        if (plan->is_breadth) {
            int *current_first_idx = plan->spread_first_idx + offset_base;
            for (int m = 0; m < plan->Mpoints; ++m) {
                PFX(ff_t) x_ff = PFX(ff_make)(FCAST(x[m].hi), FCAST(x[m].lo));
                x_ff = PFX(ff_mul)(x_ff, scale_ff);

                PFX(ff_t) x_n_ff = PFX(ff_mul)(x_ff, n_ff);
                FLOAT x_n_approx = ADD(x_n_ff.hi, x_n_ff.lo);
                int m_left = (int)ceil(TO_DOUBLE(x_n_approx) - (double)plan->w * 0.5);

                for (int l = 0; l < plan->w; ++l) {
                    int idx = m_left + l;
                    PFX(ff_t) idx_ff = PFX(ff_make)(FCAST(idx), FCAST(0.0));
                    PFX(ff_t) dist_ff = PFX(ff_sub)(idx_ff, x_n_ff);
                    FLOAT dist = ADD(dist_ff.hi, dist_ff.lo);
                    z_buf[l] = DIV(MUL(FCAST(2.0), dist), FCAST(plan->w));
                }

                PFX(pswf0_batch)(z_buf, kval_buf, plan->w, plan->w, mode43);

                current_first_idx[m] = PFX(positive_mod_int)(m_left, plan->Nfft);
                size_t w_base = (size_t)m * (size_t)plan->w;
                for (int l = 0; l < plan->w; ++l) current_spread_weight[w_base + (size_t)l] = kval_buf[l];
            }
        } else {
            FLOAT *current_shift_r = plan->shift_r + offset_shift;
            FLOAT *current_shift_i = plan->shift_i + offset_shift;
            int *current_spread_base_idx = plan->spread_base_idx + offset_base;
            int phase_shift = mode43 ? plan->output_shift / 3 : plan->output_shift;
            PFX(ff_t) out_shift_ff = PFX(ff_make)(FCAST(phase_shift), FCAST(0.0));

            for (int m = 0; m < plan->Mpoints; ++m) {
                PFX(ff_t) x_ff = PFX(ff_make)(FCAST(x[m].hi), FCAST(x[m].lo));
                x_ff = PFX(ff_mul)(x_ff, scale_ff);

                PFX(ff_t) phase_ff = PFX(ff_mul)(x_ff, out_shift_ff);
                FLOAT p_int = FCAST((int)TO_DOUBLE(phase_ff.hi));
                PFX(ff_t) phase_frac_ff = PFX(ff_sub)(phase_ff, PFX(ff_make)(p_int, FCAST(0.0)));
                FLOAT phase_frac = ADD(phase_frac_ff.hi, phase_frac_ff.lo);

                FLOAT phase_c = M_COS2PI(phase_frac);
                FLOAT phase_s = M_SIN2PI(phase_frac);
                if (mode43) {
                    nanofft_triple_angle(phase_c, phase_s, &current_shift_r[m], &current_shift_i[m]);
                } else {
                    current_shift_r[m] = phase_c;
                    current_shift_i[m] = phase_s;
                }

                PFX(ff_t) x_n_ff = PFX(ff_mul)(x_ff, n_ff);
                FLOAT x_n_approx = ADD(x_n_ff.hi, x_n_ff.lo);
                int m_left = (int)ceil(TO_DOUBLE(x_n_approx) - (double)plan->w * 0.5);

                for (int l = 0; l < plan->w; ++l) {
                    int idx = m_left + l;
                    PFX(ff_t) idx_ff = PFX(ff_make)(FCAST(idx), FCAST(0.0));
                    PFX(ff_t) dist_ff = PFX(ff_sub)(idx_ff, x_n_ff);
                    FLOAT dist = ADD(dist_ff.hi, dist_ff.lo);
                    z_buf[l] = DIV(MUL(FCAST(2.0), dist), FCAST(plan->w));
                }

                PFX(pswf0_batch)(z_buf, kval_buf, plan->w, plan->w, mode43);

                int first = PFX(positive_mod_int)(m_left, plan->Nfft);
                int base = first - (first % INTERNAL_VEC_LEN);
                int lane_offset = first - base;
                size_t w_base = (size_t)m * (size_t)plan->spread_stride;
                current_spread_base_idx[m] = base;
                for (int l = 0; l < plan->spread_stride; ++l) current_spread_weight[w_base + (size_t)l] = FCAST(0.0);
                for (int l = 0; l < plan->w; ++l) current_spread_weight[w_base + (size_t)(lane_offset + l)] = kval_buf[l];
            }
        }

#else
        double out_shift = (double)plan->output_shift;
        double n_fft = (double)plan->Nfft;

        if (plan->is_breadth) {
            int *current_first_idx = plan->spread_first_idx + offset_base;
            if (plan->use_ppoly) {
                FLOAT *current_z = plan->spread_z + offset_base;
                double half_w = (double)plan->w * 0.5;
                double w_minus_1 = (double)(plan->w - 1);
                for (int m = 0; m < plan->Mpoints; ++m) {
                    double xm = x[m] * plan->df * k_factor;
                    double x_n = xm * n_fft;
                    int m_left = (int)ceil(x_n - half_w);
                    double u = 2.0 * ((double)m_left - x_n) + w_minus_1;
                    if (u < -1.0) u = -1.0;
                    if (u > 1.0) u = 1.0;
                    current_first_idx[m] = PFX(positive_mod_int)(m_left, plan->Nfft);
                    current_z[m] = FCAST(u);
                }
            } else {
                size_t offset_spread = offset_base * (size_t)plan->w;
                FLOAT *current_spread_weight = plan->spread_weight + offset_spread;
                for (int m = 0; m < plan->Mpoints; ++m) {
                    double xm = x[m] * plan->df * k_factor;
                    double x_n = xm * n_fft;
                    int m_left = (int)ceil(x_n - (double)plan->w * 0.5);
                    for (int l = 0; l < plan->w; ++l) {
                        int idx = m_left + l;
                        double dist = (double)idx - x_n;
                        z_buf[l] = FCAST((2.0 * dist) / (double)plan->w);
                    }
                    PFX(pswf0_batch)(z_buf, kval_buf, plan->w, plan->w, mode43);
                    current_first_idx[m] = PFX(positive_mod_int)(m_left, plan->Nfft);
                    size_t w_base = (size_t)m * (size_t)plan->w;
                    for (int l = 0; l < plan->w; ++l) current_spread_weight[w_base + (size_t)l] = kval_buf[l];
                }
            }
        } else if (plan->use_ppoly) {
            FLOAT *current_shift_r = plan->shift_r + offset_shift;
            FLOAT *current_shift_i = plan->shift_i + offset_shift;
            int *current_first_idx = plan->spread_first_idx + offset_base;
            FLOAT *current_z = plan->spread_z + offset_base;
            double half_w = (double)plan->w * 0.5;
            double w_minus_1 = (double)(plan->w - 1);

            for (int m = 0; m < plan->Mpoints; ++m) {
                double xm = x[m] * plan->df * k_factor;

#    if defined(DOUBLE)
                if (mode43) {
                    double phase = xm * (double)(plan->output_shift / 3);
                    double c = cos2pi(phase);
                    double s = sin2pi(phase);
                    nanofft_triple_angle(c, s, &current_shift_r[m], &current_shift_i[m]);
                } else {
#    endif
                    double phase = xm * out_shift;
                    double p_int = (double)((int)phase);
                    double phase_frac = phase - p_int;
                    current_shift_r[m] = M_COS2PI(FCAST(phase_frac));
                    current_shift_i[m] = M_SIN2PI(FCAST(phase_frac));
#    if defined(DOUBLE)
                }
#    endif

                double x_n = xm * n_fft;
                int m_left = (int)ceil(x_n - half_w);
                double u = 2.0 * ((double)m_left - x_n) + w_minus_1;
                if (u < -1.0) u = -1.0;
                if (u > 1.0) u = 1.0;
                current_first_idx[m] = PFX(positive_mod_int)(m_left, plan->Nfft);
                current_z[m] = FCAST(u);
            }
        } else {
            FLOAT *current_shift_r = plan->shift_r + offset_shift;
            FLOAT *current_shift_i = plan->shift_i + offset_shift;
            size_t offset_spread = offset_base * (size_t)plan->spread_stride;
            int *current_spread_base_idx = plan->spread_base_idx + offset_base;
            FLOAT *current_spread_weight = plan->spread_weight + offset_spread;
            for (int m = 0; m < plan->Mpoints; ++m) {
                double xm = x[m] * plan->df * k_factor;

#    if defined(DOUBLE)
                if (mode43) {
                    double phase = xm * (double)(plan->output_shift / 3);
                    double c = cos2pi(phase);
                    double s = sin2pi(phase);
                    nanofft_triple_angle(c, s, &current_shift_r[m], &current_shift_i[m]);
                } else {
#    endif
                    double phase = xm * out_shift;
                    double p_int = (double)((int)phase);
                    double phase_frac = phase - p_int;
                    current_shift_r[m] = M_COS2PI(FCAST(phase_frac));
                    current_shift_i[m] = M_SIN2PI(FCAST(phase_frac));
#    if defined(DOUBLE)
                }
#    endif

                double x_n = xm * n_fft;
                int m_left = (int)ceil(x_n - (double)plan->w * 0.5);

                for (int l = 0; l < plan->w; ++l) {
                    int idx = m_left + l;
                    double dist = (double)idx - x_n;
                    z_buf[l] = FCAST((2.0 * dist) / (double)plan->w);
                }

                PFX(pswf0_batch)(z_buf, kval_buf, plan->w, plan->w, mode43);

                int first = PFX(positive_mod_int)(m_left, plan->Nfft);
                int base = first - (first % INTERNAL_VEC_LEN);
                int lane_offset = first - base;
                size_t w_base = (size_t)m * (size_t)plan->spread_stride;
                current_spread_base_idx[m] = base;
                for (int l = 0; l < plan->spread_stride; ++l) current_spread_weight[w_base + (size_t)l] = FCAST(0.0);
                for (int l = 0; l < plan->w; ++l) current_spread_weight[w_base + (size_t)(lane_offset + l)] = kval_buf[l];
            }
        }
#endif
    }
}

/* Compact slot output: only the kept FFT positions, in slot order. */
static void PFX(pswf_emit_slots)(const struct PFX(pswf_plan) * plan, const FLOAT *fr, const FLOAT *fi, FLOAT *out_r, FLOAT *out_i) {
    const int period = plan->period;
    const int nvalid = plan->nvalid;
    FLOAT *dst_r = out_r, *dst_i = out_i;
    for (int base = 0; base < plan->slots; base += period) {
        const FLOAT *sd = plan->slot_deconv + base;
        const FLOAT *sr = fr + base, *si = fi + base;
        for (int k = 0; k < nvalid; ++k) {
            const int src = (int)plan->slot_offsets[k];
            *dst_r++ = MUL(sr[src], sd[src]);
            *dst_i++ = MUL(si[src], sd[src]);
        }
    }
}

/*
 * Turn one compact slot block (as written by TLS_NUFFT_OUT_SLOTS) back into
 * frequency order: pad the dropped positions with zeros so the permutation
 * stays a plain power-of-two bit reversal, reverse once, then rotate the two
 * contiguous bin runs into grid order.
 */
void PFX(pswf_finish_slots)(struct PFX(pswf_plan) * plan, const FLOAT *compact, FLOAT *out, int count) {
    if (!plan || !compact || !out || count <= 0 || plan->period <= 0 || !plan->perm_scratch) return;

    FLOAT *padded = plan->perm_scratch;
    memset(padded, 0, (size_t)plan->slots * sizeof(FLOAT));
    int s = 0;
    for (int base = 0; base < plan->slots; base += plan->period) {
        for (int k = 0; k < plan->nvalid; ++k) padded[base + (int)plan->slot_offsets[k]] = compact[s++];
    }

    NANOFFT_PERMUTE_1D(plan->nanofft_p, padded);

    const int split = plan->output_shift;
    const int first_bin = plan->Nfft - split;
    const int first = (split < count) ? split : count;
    for (int k = 0; k < first; ++k) out[k] = padded[first_bin + k];
    for (int k = first; k < count; ++k) out[k] = padded[k - split];
}

static void PFX(pswf_emit_slots_breadth)(const struct PFX(pswf_plan) * plan, const BREADTH_VEC *fr, const BREADTH_VEC *fi, BREADTH_VEC *out_r,
                                         BREADTH_VEC *out_i) {
    const int period = plan->period;
    const int nvalid = plan->nvalid;
    BREADTH_VEC *dst_r = out_r;
    BREADTH_VEC *dst_i = out_i;
    for (int base = 0; base < plan->slots; base += period) {
        const FLOAT *sd = plan->slot_deconv + base;
        const BREADTH_VEC *sr = fr + base;
        const BREADTH_VEC *si = fi + base;
        for (int k = 0; k < nvalid; ++k) {
            const int src = (int)plan->slot_offsets[k];
            const FLOAT d = sd[src];
            STORE_BVEC(dst_r++, BREADTH_MUL_S(LOAD_BVEC(&sr[src]), d));
            STORE_BVEC(dst_i++, BREADTH_MUL_S(LOAD_BVEC(&si[src]), d));
        }
    }
}

void PFX(pswf_finish_slots_breadth)(struct PFX(pswf_plan) * plan, const BREADTH_VEC *compact, FLOAT *out, int block, int active_lanes, int total_remaining) {
    if (!plan || !plan->is_breadth || !compact || !out || active_lanes <= 0 || total_remaining <= 0 || plan->period <= 0 || !plan->bperm_scratch) return;

    BREADTH_VEC *padded = plan->bperm_scratch;
    memset(padded, 0, (size_t)plan->slots * sizeof(BREADTH_VEC));
    int s = 0;
    for (int base = 0; base < plan->slots; base += plan->period) {
        for (int k = 0; k < plan->nvalid; ++k) padded[base + (int)plan->slot_offsets[k]] = compact[s++];
    }

    NANOFFT_PERMUTE_1D_BREADTH(plan->nanofft_p, padded);

    const int split = plan->output_shift;
    const int first_bin = plan->Nfft - split;
    for (int v = 0; v < active_lanes; ++v) {
        const int rem = total_remaining - v * block;
        if (rem <= 0) break;
        const int count_v = (rem < block) ? rem : block;
        const int first = (split < count_v) ? split : count_v;
        FLOAT *dst = out + (size_t)v * (size_t)block;
        for (int k = 0; k < first; ++k) dst[k] = BREADTH_GET_LANE(LOAD_BVEC(&padded[first_bin + k]), v);
        for (int k = first; k < count_v; ++k) dst[k] = BREADTH_GET_LANE(LOAD_BVEC(&padded[k - split]), v);
    }
}

#if defined(HALF)
static inline uint32_t PFX(log2_u32)(uint32_t n) {
    uint32_t bits = 0u;
    while (n > 1u) {
        n >>= 1u;
        ++bits;
    }
    return bits;
}

void PFX(pswf_finish_slots_f32)(struct PFX(pswf_plan) * plan, const float *compact, float *out, int count) {
    if (!plan || !compact || !out || count <= 0 || plan->period <= 0) return;
    const uint32_t log_nfft = PFX(log2_u32)((uint32_t)plan->Nfft);
    const int split = plan->output_shift;
    const int first_bin = plan->Nfft - split;
    int s = 0;
    for (int base = 0; base < plan->slots; base += plan->period) {
        for (int k = 0; k < plan->nvalid; ++k) {
            const uint32_t slot_pos = (uint32_t)(base + (int)plan->slot_offsets[k]);
            const int bin = (int)nanofft_reverse_bits(slot_pos, log_nfft);
            const int out_idx = (bin >= first_bin) ? (bin - first_bin) : (bin + split);
            if (out_idx >= 0 && out_idx < count) out[out_idx] = compact[s];
            ++s;
        }
    }
}

void PFX(pswf_finish_slots_breadth_f32)(struct PFX(pswf_plan) * plan, const float *compact, float *out, int block, int active_lanes, int total_remaining) {
    if (!plan || !plan->is_breadth || !compact || !out || active_lanes <= 0 || total_remaining <= 0 || plan->period <= 0) return;
    const uint32_t log_nfft = PFX(log2_u32)((uint32_t)plan->Nfft);
    const int split = plan->output_shift;
    const int first_bin = plan->Nfft - split;
    int s = 0;
    for (int base = 0; base < plan->slots; base += plan->period) {
        for (int k = 0; k < plan->nvalid; ++k) {
            const uint32_t slot_pos = (uint32_t)(base + (int)plan->slot_offsets[k]);
            const int bin = (int)nanofft_reverse_bits(slot_pos, log_nfft);
            const int out_idx = (bin >= first_bin) ? (bin - first_bin) : (bin + split);
            const float *src = compact + (size_t)s * (size_t)BREADTH_VEC_LEN;
            if (out_idx >= 0 && out_idx < block) {
                for (int v = 0; v < active_lanes; ++v) {
                    const int rem = total_remaining - v * block;
                    if (out_idx < rem) out[(size_t)v * (size_t)block + (size_t)out_idx] = src[v];
                }
            }
            ++s;
        }
    }
}
#endif

/* Slot support is built on demand: a plan that only ever produces frequency
 * order (the IHS d = 1 path, every natural-order call) never pays for it.
 * Callers must have run pswf_precompute() first, as chi2per always does. */
static void PFX(pswf_ensure_slot_tables)(struct PFX(pswf_plan) * plan) {
    if (!plan || plan->slot_tables_built) return;
    PFX(pswf_build_slot_tables)(plan);
}

int PFX(pswf_slots_available)(struct PFX(pswf_plan) * plan) {
    PFX(pswf_ensure_slot_tables)(plan);
    return (plan && plan->period > 0 && plan->nvalid > 0) ? 1 : 0;
}

void PFX(pswf_execute_mode)(struct PFX(pswf_plan) * plan, const FLOAT *y_real, const FLOAT *y_imag, FLOAT *out_real, FLOAT *out_imag, int freq_factor,
                            int mode) {
    int f_idx = freq_factor - 1;
    if (f_idx < 0 || f_idx >= plan->num_factors) return;

    size_t offset_shift = (size_t)f_idx * plan->Mpoints;
    size_t offset_base = (size_t)f_idx * plan->Mpoints;
    FLOAT *current_shift_r = plan->shift_r + offset_shift;
    FLOAT *current_shift_i = plan->shift_i + offset_shift;

    memset(plan->fft_real, 0, (size_t)plan->fft_alloc_len * sizeof(FLOAT));
    memset(plan->fft_imag, 0, (size_t)plan->fft_alloc_len * sizeof(FLOAT));

#if !defined(DOUBLE_DOUBLE)
    if (plan->use_ppoly) {
        const int *current_first_idx = plan->spread_first_idx + offset_base;
        const FLOAT *current_z = plan->spread_z + offset_base;
        const FLOAT *coeffs = plan->ppoly_coeffs;
        const int w = plan->w;
        const int nc = plan->ppoly_nc;
        const int stride = plan->ppoly_stride;
        FLOAT *NANOFFT_RESTRICT fr = plan->fft_real;
        FLOAT *NANOFFT_RESTRICT fi = plan->fft_imag;

        for (int m = 0; m < plan->Mpoints; ++m) {
            FLOAT yr = y_real[m], yi = y_imag[m];
            FLOAT sr = current_shift_r[m], si = current_shift_i[m];
            FLOAT sy_r = SUB(MUL(yr, sr), MUL(yi, si));
            FLOAT sy_i = ADD(MUL(yr, si), MUL(yi, sr));
            PFX(ppoly_spread_point)(fr, fi, coeffs, current_first_idx[m], current_z[m], sy_r, sy_i, w, nc, stride);
        }
    } else
#endif
    {
        size_t offset_spread = offset_base * plan->spread_stride;
        int *current_spread_base_idx = plan->spread_base_idx + offset_base;
        FLOAT *current_spread_weight = plan->spread_weight + offset_spread;

        for (int m = 0; m < plan->Mpoints; ++m) {
            FLOAT yr = y_real[m], yi = y_imag[m];
            FLOAT sr = current_shift_r[m], si = current_shift_i[m];
            FLOAT sy_r = SUB(MUL(yr, sr), MUL(yi, si));
            FLOAT sy_i = ADD(MUL(yr, si), MUL(yi, sr));
            int base = current_spread_base_idx[m];
            size_t w_base = (size_t)m * plan->spread_stride;

#if INTERNAL_VEC_LEN > 1
            INTERNAL_VEC v_sy_r = IVEC_SPLAT(sy_r);
            INTERNAL_VEC v_sy_i = IVEC_SPLAT(sy_i);

            for (int l = 0; l < plan->spread_stride; l += INTERNAL_VEC_LEN) {
                int idx = base + l;
                INTERNAL_VEC kval = LOAD_VEC(&current_spread_weight[w_base + l]);
                INTERNAL_VEC gr = LOAD_VEC(&plan->fft_real[idx]);
                INTERNAL_VEC gi = LOAD_VEC(&plan->fft_imag[idx]);
                STORE_VEC(&plan->fft_real[idx], ADD(gr, MUL(v_sy_r, kval)));
                STORE_VEC(&plan->fft_imag[idx], ADD(gi, MUL(v_sy_i, kval)));
            }
#else
            for (int l = 0; l < plan->spread_stride; ++l) {
                int idx = base + l;
                FLOAT kval = current_spread_weight[w_base + l];
                plan->fft_real[idx] = ADD(plan->fft_real[idx], MUL(sy_r, kval));
                plan->fft_imag[idx] = ADD(plan->fft_imag[idx], MUL(sy_i, kval));
            }
#endif
        }
    }

    for (int i = 0; i < plan->spread_pad; ++i) {
        int src = plan->Nfft + i;
        int dst = i % plan->Nfft;
        plan->fft_real[dst] = ADD(plan->fft_real[dst], plan->fft_real[src]);
        plan->fft_imag[dst] = ADD(plan->fft_imag[dst], plan->fft_imag[src]);
    }

    if (mode == TLS_NUFFT_OUT_SLOTS) PFX(pswf_ensure_slot_tables)(plan);
    if (mode == TLS_NUFFT_OUT_SLOTS && plan->period > 0) {
        NANOFFT_EXECUTE_RAW(plan->nanofft_p, plan->fft_real, plan->fft_imag);
        PFX(pswf_emit_slots)(plan, plan->fft_real, plan->fft_imag, out_real, out_imag);
        return;
    }

    NANOFFT_EXECUTE(plan->nanofft_p, plan->fft_real, plan->fft_imag);

    for (int k = 0; k < plan->Nout; ++k) {
        int k_prime = k - plan->output_shift;
        int fft_idx = k_prime >= 0 ? k_prime : plan->Nfft + k_prime;
        out_real[k] = MUL(plan->fft_real[fft_idx], plan->deconv[k]);
        out_imag[k] = MUL(plan->fft_imag[fft_idx], plan->deconv[k]);
    }
}

void PFX(pswf_execute_breadth)(struct PFX(pswf_plan) * plan, const nufft_input_t *x, const FLOAT *val, double f0, int block, int block_base, int freq_factor,
                               int mode, BREADTH_VEC *out_real, BREADTH_VEC *out_imag) {
    int f_idx = freq_factor - 1;
    if (!plan || !plan->is_breadth || f_idx < 0 || f_idx >= plan->num_factors) return;

    const size_t offset_base = (size_t)f_idx * (size_t)plan->Mpoints;
    BREADTH_VEC *NANOFFT_RESTRICT fr = plan->bfft_real;
    BREADTH_VEC *NANOFFT_RESTRICT fi = plan->bfft_imag;
    memset(fr, 0, (size_t)plan->fft_alloc_len * sizeof(BREADTH_VEC));
    memset(fi, 0, (size_t)plan->fft_alloc_len * sizeof(BREADTH_VEC));

    /* Precompute per-lane shifted frequencies (folding output_shift into the on-the-fly phase). */
#if defined(DOUBLE_DOUBLE)
    VEC f_hi, f_lo;
    const dd_t d_f0 = dd_make(f0, 0.0);
    const dd_t d_df = dd_make(plan->df, 0.0);
    const dd_t d_q = dd_make((double)freq_factor, 0.0);
    const dd_t d_shift = dd_make((double)plan->output_shift, 0.0);
    for (int v = 0; v < VECDD_LEN; ++v) {
        dd_t k_idx = dd_add(dd_mul(dd_make((double)(block_base + v), 0.0), dd_make((double)block, 0.0)), d_shift);
        dd_t fv = dd_mul(d_q, dd_add(d_f0, dd_mul(k_idx, d_df)));
        f_hi[v] = fv.hi;
        f_lo[v] = fv.lo;
    }
    const VECDD v_freq = vecdd_make(f_hi, f_lo);
#elif defined(DOUBLE)
    VEC v_freq;
    for (int v = 0; v < VEC_LEN; ++v) {
        v_freq[v] = (double)freq_factor * (f0 + ((double)(block_base + v) * (double)block + (double)plan->output_shift) * plan->df);
    }
#elif defined(HALF) && NANOFFT_NATIVE_FP16
    VEC v_freq[VECH_LEN / VEC_LEN];
    for (int ch = 0; ch < (VECH_LEN / VEC_LEN); ++ch) {
        for (int v = 0; v < VEC_LEN; ++v) {
            v_freq[ch][v] = (double)freq_factor * (f0 + ((double)(block_base + ch * VEC_LEN + v) * (double)block + (double)plan->output_shift) * plan->df);
        }
    }
#else
    VEC v_freq_lo, v_freq_hi;
    for (int v = 0; v < VEC_LEN; ++v) {
        v_freq_lo[v] = (double)freq_factor * (f0 + ((double)(block_base + v) * (double)block + (double)plan->output_shift) * plan->df);
        v_freq_hi[v] = (double)freq_factor * (f0 + ((double)(block_base + VEC_LEN + v) * (double)block + (double)plan->output_shift) * plan->df);
    }
#endif

    const int w = plan->w;
    const int *current_first_idx = plan->spread_first_idx + offset_base;

#if !defined(DOUBLE_DOUBLE)
    if (plan->use_ppoly) {
        const FLOAT *current_z = plan->spread_z + offset_base;
        const FLOAT *coeffs = plan->ppoly_coeffs;
        const int nc = plan->ppoly_nc;
        const int stride = plan->ppoly_stride;
        FLOAT kval[64] __attribute__((aligned(64)));

        for (int m = 0; m < plan->Mpoints; ++m) {
            BREADTH_COMPUTE_VEC c, s;
#    if defined(DOUBLE)
            vec_sincos2pi(((VEC){} + x[m]) * v_freq, &c, &s);
#    elif defined(HALF) && NANOFFT_NATIVE_FP16
            const VEC xm_v = (VEC){} + x[m];
            VEC ang[VECH_LEN / VEC_LEN];
            for (int ch = 0; ch < (VECH_LEN / VEC_LEN); ++ch) ang[ch] = xm_v * v_freq[ch];
            vech_sincos2pi_d(ang, &c, &s);
#    else
            const VEC xm_v = (VEC){} + x[m];
            vecf_sincos2pi_d(xm_v * v_freq_lo, xm_v * v_freq_hi, &c, &s);
#    endif
            const BREADTH_COMPUTE_VEC sy_r = BREADTH_MUL_S(c, val[m]);
            const BREADTH_COMPUTE_VEC sy_i = BREADTH_MUL_S(s, val[m]);
            const int first = current_first_idx[m];
            PFX(ppoly_eval_weights)(kval, coeffs, current_z[m], w, nc, stride);
            for (int l = 0; l < w; ++l) {
                const int idx = first + l;
                const FLOAT kv = kval[l];
                STORE_BVEC(&fr[idx], BREADTH_ADD(LOAD_BVEC(&fr[idx]), BREADTH_MUL_S(sy_r, kv)));
                STORE_BVEC(&fi[idx], BREADTH_ADD(LOAD_BVEC(&fi[idx]), BREADTH_MUL_S(sy_i, kv)));
            }
        }
    } else
#endif
    {
        const FLOAT *current_spread_weight = plan->spread_weight + offset_base * (size_t)w;
        for (int m = 0; m < plan->Mpoints; ++m) {
            BREADTH_COMPUTE_VEC c, s;
#if defined(DOUBLE_DOUBLE)
            vecdd_sincos2pi(vecdd_mul(vecdd_splat(x[m]), v_freq), &c, &s);
#elif defined(DOUBLE)
            vec_sincos2pi(((VEC){} + x[m]) * v_freq, &c, &s);
#elif defined(HALF) && NANOFFT_NATIVE_FP16
            const VEC xm_v = (VEC){} + x[m];
            VEC ang[VECH_LEN / VEC_LEN];
            for (int ch = 0; ch < (VECH_LEN / VEC_LEN); ++ch) ang[ch] = xm_v * v_freq[ch];
            vech_sincos2pi_d(ang, &c, &s);
#else
            const VEC xm_v = (VEC){} + x[m];
            vecf_sincos2pi_d(xm_v * v_freq_lo, xm_v * v_freq_hi, &c, &s);
#endif
            const BREADTH_COMPUTE_VEC sy_r = BREADTH_MUL_S(c, val[m]);
            const BREADTH_COMPUTE_VEC sy_i = BREADTH_MUL_S(s, val[m]);
            const int first = current_first_idx[m];
            const FLOAT *kw = current_spread_weight + (size_t)m * (size_t)w;
            for (int l = 0; l < w; ++l) {
                const int idx = first + l;
                const FLOAT kv = kw[l];
                STORE_BVEC(&fr[idx], BREADTH_ADD(LOAD_BVEC(&fr[idx]), BREADTH_MUL_S(sy_r, kv)));
                STORE_BVEC(&fi[idx], BREADTH_ADD(LOAD_BVEC(&fi[idx]), BREADTH_MUL_S(sy_i, kv)));
            }
        }
    }

    for (int i = 0; i < plan->spread_pad; ++i) {
        const int src = plan->Nfft + i;
        const int dst = i % plan->Nfft;
        STORE_BVEC(&fr[dst], BREADTH_ADD(LOAD_BVEC(&fr[dst]), LOAD_BVEC(&fr[src])));
        STORE_BVEC(&fi[dst], BREADTH_ADD(LOAD_BVEC(&fi[dst]), LOAD_BVEC(&fi[src])));
    }

    if (mode == TLS_NUFFT_OUT_SLOTS) PFX(pswf_ensure_slot_tables)(plan);
    if (mode == TLS_NUFFT_OUT_SLOTS && plan->period > 0) {
        NANOFFT_EXECUTE_RAW_BREADTH(plan->nanofft_p, fr, fi);
        PFX(pswf_emit_slots_breadth)(plan, fr, fi, out_real, out_imag);
        return;
    }

    NANOFFT_EXECUTE_BREADTH(plan->nanofft_p, fr, fi);

    for (int k = 0; k < plan->Nout; ++k) {
        const int k_prime = k - plan->output_shift;
        const int fft_idx = k_prime >= 0 ? k_prime : plan->Nfft + k_prime;
        const FLOAT d = plan->deconv[k];
        STORE_BVEC(&out_real[k], BREADTH_MUL_S(LOAD_BVEC(&fr[fft_idx]), d));
        STORE_BVEC(&out_imag[k], BREADTH_MUL_S(LOAD_BVEC(&fi[fft_idx]), d));
    }
}

void PFX(pswf_execute)(struct PFX(pswf_plan) * plan, const FLOAT *y_real, const FLOAT *y_imag, FLOAT *out_real, FLOAT *out_imag, int freq_factor) {
    PFX(pswf_execute_mode)(plan, y_real, y_imag, out_real, out_imag, freq_factor, TLS_NUFFT_OUT_NATURAL);
}

void PFX(free_pswf_plan)(struct PFX(pswf_plan) * plan) {
    if (!plan) return;
    nanofft_aligned_free(plan->ppoly_coeffs);
    nanofft_aligned_free(plan->spread_first_idx);
    nanofft_aligned_free(plan->spread_z);
    nanofft_aligned_free(plan->spread_base_idx);
    nanofft_aligned_free(plan->spread_weight);
    nanofft_aligned_free(plan->deconv);
    nanofft_aligned_free(plan->shift_r);
    nanofft_aligned_free(plan->shift_i);
    nanofft_aligned_free(plan->fft_real);
    nanofft_aligned_free(plan->fft_imag);
    nanofft_aligned_free(plan->slot_deconv);
    nanofft_aligned_free(plan->perm_scratch);
    nanofft_aligned_free(plan->bfft_real);
    nanofft_aligned_free(plan->bfft_imag);
    nanofft_aligned_free(plan->bperm_scratch);
    if (plan->nanofft_p) NANOFFT_DESTROY_PLAN(plan->nanofft_p);
    nanofft_aligned_free(plan);
}
