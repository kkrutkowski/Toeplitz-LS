/* nufft1.c – Non-Uniform FFT Type-1 implementation */

#include <float.h>
#include <math.h>
#include <nufft1.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Common Helpers
 * ========================================================================= */

static const int pswf_degrees[19] = {0, 0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 14, 14, 16, 18, 18, 20, 20, 23};
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

#ifdef DOUBLE_DOUBLE
#    define PFX(name) tlsdd_nufft_##name
#    define NANOFFT_PLAN nanofftdd_plan
#    define NANOFFT_MAKE_PLAN nanofftdd_make_plan
#    define NANOFFT_EXECUTE nanofftdd_execute
#    define NANOFFT_DESTROY_PLAN nanofftdd_destroy_plan
#    define LRA_BESSEL_TERMS 14
#    define GL_NEWTON_TOL FCONST(5e-32, 0)
#elif defined(DOUBLE)
#    define PFX(name) tls_nufft_##name
#    define NANOFFT_PLAN nanofft_plan
#    define NANOFFT_MAKE_PLAN nanofft_make_plan
#    define NANOFFT_EXECUTE nanofft_execute
#    define NANOFFT_DESTROY_PLAN nanofft_destroy_plan
#    define LRA_BESSEL_TERMS 8
#    define GL_NEWTON_TOL FCONST(2.0e-16)
#else
#    define PFX(name) tlsf_nufft_##name
#    define NANOFFT_PLAN nanofftf_plan
#    define NANOFFT_MAKE_PLAN nanofftf_make_plan
#    define NANOFFT_EXECUTE nanofftf_execute
#    define NANOFFT_DESTROY_PLAN nanofftf_destroy_plan
#    define LRA_BESSEL_TERMS 6
#    define GL_NEWTON_TOL FCONST(1.0e-7)
#endif

/* =========================================================================
 * Gauss-Legendre Quadrature
 * ========================================================================= */

static void PFX(get_lege_roots)(int n, FLOAT *roots, FLOAT *weights) {
    int half_n = (n + 1) / 2;
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
}

/* =========================================================================
 * Lookup Tables
 * ========================================================================= */

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

static FLOAT *PFX(allocate_aligned)(uint32_t N) {
    size_t alloc_size = (sz_max(N * sizeof(FLOAT), 64) + 63) & ~63;
    FLOAT *ptr = (FLOAT *)aligned_alloc(64, alloc_size);
    if (!ptr) exit(EXIT_FAILURE);
    memset(ptr, 0, alloc_size);
    return ptr;
}

/* =========================================================================
 * Kahan Summation (SOA layout)
 * ========================================================================= */

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) __attribute__((noinline)) static void
PFX(kahan_add_soa)(FLOAT *sum_r, FLOAT *sum_i, FLOAT *comp_r, FLOAT *comp_i, FLOAT input_r, FLOAT input_i) {
    FLOAT yr = SUB(input_r, *comp_r);
    FLOAT yi = SUB(input_i, *comp_i);
    FLOAT tr = ADD(*sum_r, yr);
    FLOAT ti = ADD(*sum_i, yi);
    *comp_r = SUB(SUB(tr, *sum_r), yr);
    *comp_i = SUB(SUB(ti, *sum_i), yi);
    *sum_r = tr;
    *sum_i = ti;
}

/* =========================================================================
 * Double-Word Arithmetic (error-free transforms)
 * ========================================================================= */

typedef struct {
    FLOAT hi;
    FLOAT lo;
} PFX(ff_t);

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline PFX(ff_t)
    PFX(ff_make)(FLOAT hi, FLOAT lo) {
    PFX(ff_t) res = {hi, lo};
    return res;
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline void PFX(ff_two_sum_quick)(
    FLOAT x, FLOAT y, FLOAT *r, FLOAT *e) {
    *r = ADD(x, y);
    *e = SUB(y, SUB(*r, x));
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline void PFX(
    ff_two_difference)(FLOAT x, FLOAT y, FLOAT *r, FLOAT *e) {
    *r = SUB(x, y);
    FLOAT t = SUB(*r, x);
    *e = SUB(SUB(x, SUB(*r, t)), ADD(y, t));
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline void PFX(ff_two_product)(
    FLOAT x, FLOAT y, FLOAT *r, FLOAT *e) {
    *r = MUL(x, y);
    *e = M_FMA(x, y, NEG(*r));
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline PFX(ff_t)
    PFX(ff_sub)(PFX(ff_t) a, PFX(ff_t) b) {
    FLOAT r, e, r_f, e_f;
    PFX(ff_two_difference)(a.hi, b.hi, &r, &e);
    e = SUB(ADD(e, a.lo), b.lo);
    PFX(ff_two_sum_quick)(r, e, &r_f, &e_f);
    return PFX(ff_make)(r_f, e_f);
}

__attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations"))) static inline PFX(ff_t)
    PFX(ff_mul)(PFX(ff_t) a, PFX(ff_t) b) {
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
    PFX(MatrixF) *M = (PFX(MatrixF) *)malloc(sizeof(PFX(MatrixF)));
    M->rows = r;
    M->cols = c;
    M->d = (FLOAT *)calloc((size_t)r * c, sizeof(FLOAT));
    return M;
}

static PFX(MatrixC) * PFX(matc_alloc)(int r, int c) {
    PFX(MatrixC) *M = (PFX(MatrixC) *)malloc(sizeof(PFX(MatrixC)));
    M->rows = r;
    M->cols = c;
    M->r = (FLOAT *)calloc((size_t)r * c, sizeof(FLOAT));
    M->i = (FLOAT *)calloc((size_t)r * c, sizeof(FLOAT));
    return M;
}

static void PFX(mat_free)(PFX(MatrixF) * M) {
    free(M->d);
    free(M);
}

static void PFX(matc_free)(PFX(MatrixC) * M) {
    free(M->r);
    free(M->i);
    free(M);
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
}

/* =========================================================================
 * LRA NUFFT Implementation
 * ========================================================================= */

struct PFX(lra_plan) {
    int Mpoints, N, Nfft, K;
    int max_Mpoints, max_N, max_K, Nfft_allocated;
    int num_factors;
    double df;
    int *Tidx;
    FLOAT *conjU_r, *conjU_i;
    FLOAT *conjV;
    NANOFFT_PLAN *nanofft_p;
    FLOAT *fft_real, *fft_imag;
    FLOAT *temp_r, *temp_i, *temp_comp_r, *temp_comp_i;
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

struct PFX(lra_plan) * PFX(lra_initialize)(int max_Mpoints, int max_N, int max_rank, double df, int freq_factor) {
    struct PFX(lra_plan) *plan = (struct PFX(lra_plan) *)calloc(1, sizeof(struct PFX(lra_plan)));
    plan->max_Mpoints = max_Mpoints;
    plan->max_N = max_N;
    plan->max_K = max_rank;
    plan->num_factors = freq_factor;
    plan->df = df;
    plan->Nfft_allocated = next_fast_len(max_N);
    plan->Nfft = plan->Nfft_allocated;

    plan->Tidx = (int *)malloc((size_t)max_Mpoints * freq_factor * sizeof(int));
    plan->conjU_r = (FLOAT *)calloc((size_t)max_Mpoints * max_rank * freq_factor, sizeof(FLOAT));
    plan->conjU_i = (FLOAT *)calloc((size_t)max_Mpoints * max_rank * freq_factor, sizeof(FLOAT));
    plan->conjV = (FLOAT *)calloc((size_t)plan->Nfft_allocated * max_rank, sizeof(FLOAT));

    plan->fft_real = PFX(allocate_aligned)(plan->Nfft_allocated);
    plan->fft_imag = PFX(allocate_aligned)(plan->Nfft_allocated);
    plan->nanofft_p = NANOFFT_MAKE_PLAN(plan->Nfft_allocated);

    size_t V_size = (size_t)plan->Nfft_allocated * max_rank;
    plan->temp_r = (FLOAT *)calloc(V_size, sizeof(FLOAT));
    plan->temp_i = (FLOAT *)calloc(V_size, sizeof(FLOAT));
    plan->temp_comp_r = (FLOAT *)calloc(V_size, sizeof(FLOAT));
    plan->temp_comp_i = (FLOAT *)calloc(V_size, sizeof(FLOAT));
    return plan;
}

void PFX(lra_precompute)(struct PFX(lra_plan) * plan, const nufft_input_t *x, int Mpoints, int N, int rank) {
    plan->Mpoints = Mpoints;
    plan->N = N;
    plan->K = rank;
    plan->Nfft = plan->Nfft_allocated;
    int Nfft = plan->Nfft;

    for (int f_idx = 0; f_idx < plan->num_factors; ++f_idx) {
        int k_factor = f_idx + 1;

        nufft_input_t *x_scaled = (nufft_input_t *)malloc(Mpoints * sizeof(nufft_input_t));
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

        FLOAT *er = (FLOAT *)malloc(Mpoints * sizeof(FLOAT));
        FLOAT *scaled = (FLOAT *)malloc(Mpoints * sizeof(FLOAT));
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
        free(scaled);
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
        free(er);
        free(x_scaled);
    }

    FLOAT *X = (FLOAT *)malloc((size_t)Nfft * sizeof(FLOAT));
    for (int w = 0; w < Nfft; ++w) {
        X[w] = SUB(DIV(FCAST(2.0 * w), FCAST(Nfft)), FCAST(1.0));
    }
    PFX(MatrixF) *Tcheb_V = PFX(chebyshev_mat)(rank - 1, X, Nfft);
    free(X);
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

void PFX(free_lra_plan)(struct PFX(lra_plan) * plan) {
    if (!plan) return;
    if (plan->nanofft_p) NANOFFT_DESTROY_PLAN(plan->nanofft_p);
    free(plan->fft_real);
    free(plan->fft_imag);
    free(plan->conjU_r);
    free(plan->conjU_i);
    free(plan->conjV);
    free(plan->temp_r);
    free(plan->temp_i);
    free(plan->temp_comp_r);
    free(plan->temp_comp_i);
    free(plan->Tidx);
    free(plan);
}

/* =========================================================================
 * PSWF NUFFT Implementation
 * ========================================================================= */

struct PFX(pswf_plan) {
    int Mpoints, N, Nout, Nfft, w;
    int output_shift;
    char upsamp[2];
    FLOAT alpha;
    int num_factors;
    double df;
    int *spread_idx;
    FLOAT *spread_weight;
    FLOAT *spread_comp_r, *spread_comp_i;
    FLOAT *deconv, *shift_r, *shift_i;
    NANOFFT_PLAN *nanofft_p;
    FLOAT *fft_real, *fft_imag;
};

static inline bool PFX(pswf_lut_lookup)(int w, bool mode43, const FLOAT **coeffs, int *deg, FLOAT *c) {
#ifdef DOUBLE
    if (mode43 && (w == 8 || w == 9 || w == 16 || w == 18)) {
        *coeffs = PFX(pswf43_coeffs)[w];
        *deg = PFX(pswf43_degrees)[w];
        *c = PFX(pswf43_c)[w];
        return true;
    }
#endif

    if (!mode43 && w < 19) {
        *coeffs = PFX(pswf_coeffs)[w];
        *deg = pswf_degrees[w];
        *c = FCAST(M_PI * w * 0.75 - 0.05);
        return true;
    }

    return false;
}

static inline FLOAT PFX(pswf0)(FLOAT z, int w, bool mode43) {
    int safe_w = w;
    if (safe_w < 1) safe_w = 1;
    if (safe_w > PSWF_MAX_W) safe_w = PSWF_MAX_W;

    FLOAT Z = MUL(z, z);
    const FLOAT *coeffs = NULL;
    int deg = 0;
    FLOAT c = FCAST(0.0);

    if (PFX(pswf_lut_lookup)(safe_w, mode43, &coeffs, &deg, &c)) {
        FLOAT poly = coeffs[deg];
        for (int i = deg - 1; i >= 0; --i) poly = M_FMA(poly, Z, coeffs[i]);

        return MUL(M_EXP(MUL(MUL(NEG(c), Z), FCAST(0.5))), poly);
    } else {
        FLOAT beta = FCAST((mode43 ? 1.90 : 2.30) * safe_w);
        FLOAT inner = SUB(FCAST(1.0), Z);
        if (TO_DOUBLE(inner) < 0.0) inner = FCAST(0.0);

        return M_EXP(MUL(beta, SUB(M_SQRT(inner), FCAST(1.0))));
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

    int n_vec = n - (n % INTERNAL_VEC_LEN);

    if (has_lut) {
        FLOAT neg_c_half = NEG(MUL(c, FCAST(0.5)));

        for (int i = 0; i < n_vec; i += INTERNAL_VEC_LEN) {
            INTERNAL_VEC v_z = LOAD_VEC(&z_arr[i]);
            INTERNAL_VEC v_Z = MUL(v_z, v_z);

            INTERNAL_VEC v_poly = (INTERNAL_VEC){} + coeffs[deg];
            for (int j = deg - 1; j >= 0; --j) v_poly = ADD(MUL(v_poly, v_Z), (INTERNAL_VEC){} + coeffs[j]);

            INTERNAL_VEC v_result = MUL(M_EXP(MUL(v_Z, neg_c_half)), v_poly);
            STORE_VEC(&out_arr[i], v_result);
        }
    } else {
        FLOAT beta = FCAST((mode43 ? 1.90 : 2.30) * safe_w);
        INTERNAL_VEC v_one = (INTERNAL_VEC){} + FCAST(1.0);
        INTERNAL_VEC v_beta = (INTERNAL_VEC){} + beta;

        for (int i = 0; i < n_vec; i += INTERNAL_VEC_LEN) {
            INTERNAL_VEC v_z = LOAD_VEC(&z_arr[i]);
            INTERNAL_VEC v_Z = MUL(v_z, v_z);
            INTERNAL_VEC v_inner = SUB(v_one, v_Z);

            for (int lane = 0; lane < INTERNAL_VEC_LEN; ++lane) {
                if (v_inner[lane] < FCAST(0.0)) v_inner[lane] = FCAST(0.0);
            }

            INTERNAL_VEC v_result = M_EXP(MUL(v_beta, SUB(M_SQRT(v_inner), v_one)));
            STORE_VEC(&out_arr[i], v_result);
        }
    }

    for (int i = n_vec; i < n; ++i) out_arr[i] = PFX(pswf0)(z_arr[i], w, mode43);
#else
    for (int i = 0; i < n; ++i) out_arr[i] = PFX(pswf0)(z_arr[i], w, mode43);
#endif
}

struct PFX(pswf_plan) * PFX(pswf_initialize)(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2]) {
    if (!upsamp) return NULL;

    bool mode21 = (upsamp[0] == '2' && upsamp[1] == '1');
    bool mode43 = (upsamp[0] == '4' && upsamp[1] == '3');
    if (!mode21 && !mode43) return NULL;
    if (mode43 && (N % 4) != 0) return NULL;
    if (w < 1 || w > PSWF_MAX_W) return NULL;

    struct PFX(pswf_plan) *plan = (struct PFX(pswf_plan) *)calloc(1, sizeof(struct PFX(pswf_plan)));
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
    plan->Nfft = next_fast_len(2 * N);
    plan->alpha = DIV(MUL(FCAST(plan->w), FCONST(0.5)), FCAST(plan->Nfft));

    plan->spread_idx = (int *)malloc((size_t)Mpoints * plan->w * freq_factor * sizeof(int));
    plan->spread_weight = (FLOAT *)malloc((size_t)Mpoints * plan->w * freq_factor * sizeof(FLOAT));
    plan->spread_comp_r = PFX(allocate_aligned)(plan->Nfft);
    plan->spread_comp_i = PFX(allocate_aligned)(plan->Nfft);
    plan->deconv = (FLOAT *)malloc((size_t)plan->Nout * sizeof(FLOAT));

    plan->shift_r = (FLOAT *)malloc((size_t)Mpoints * freq_factor * sizeof(FLOAT));
    plan->shift_i = (FLOAT *)malloc((size_t)Mpoints * freq_factor * sizeof(FLOAT));

    plan->fft_real = PFX(allocate_aligned)(plan->Nfft);
    plan->fft_imag = PFX(allocate_aligned)(plan->Nfft);
    plan->nanofft_p = NANOFFT_MAKE_PLAN(plan->Nfft);
    return plan;
}

void PFX(pswf_precompute)(struct PFX(pswf_plan) * plan, const nufft_input_t *x) {
    bool mode43 = (plan->upsamp[0] == '4' && plan->upsamp[1] == '3');
    int p = mode43 ? (4 * plan->w + 16) : (int)(1.5 * plan->w + 2);
    int num_gl_nodes = 2 * p;

    FLOAT *gl_nodes = PFX(allocate_aligned)(num_gl_nodes);
    FLOAT *gl_weights = PFX(allocate_aligned)(num_gl_nodes);
    FLOAT *precomp_vals = PFX(allocate_aligned)(num_gl_nodes);

    PFX(get_lege_roots)(num_gl_nodes, gl_nodes, gl_weights);
    PFX(pswf0_batch)(gl_nodes, precomp_vals, p, plan->w, mode43);

    for (int j = 0; j < p; ++j) precomp_vals[j] = MUL(gl_weights[j], precomp_vals[j]);

    for (int k = 0; k < plan->Nout; ++k) {
        FLOAT xi_k = MUL(plan->alpha, FCAST(k - plan->output_shift));
        FLOAT phi_hat_half = FCAST(0.0);

        for (int j = 0; j < p; ++j) phi_hat_half = ADD(phi_hat_half, MUL(precomp_vals[j], M_COS2PI(MUL(xi_k, gl_nodes[j]))));

        plan->deconv[k] = DIV(FCAST(1.0), MUL(FCAST(plan->w), phi_hat_half));
    }

    free(precomp_vals);
    free(gl_nodes);
    free(gl_weights);

    FLOAT z_buf[64] __attribute__((aligned(64)));
    FLOAT kval_buf[64] __attribute__((aligned(64)));

    for (int f_idx = 0; f_idx < plan->num_factors; ++f_idx) {
        int k_factor = f_idx + 1;
        size_t offset_shift = (size_t)f_idx * plan->Mpoints;
        size_t offset_spread = (size_t)f_idx * plan->Mpoints * plan->w;

        FLOAT *current_shift_r = plan->shift_r + offset_shift;
        FLOAT *current_shift_i = plan->shift_i + offset_shift;
        int *current_spread_idx = plan->spread_idx + offset_spread;
        FLOAT *current_spread_weight = plan->spread_weight + offset_spread;

#if defined(DOUBLE_DOUBLE)
        PFX(ff_t) out_shift_ff = PFX(ff_make)(FCAST(plan->output_shift), FCAST(0.0));
        PFX(ff_t) n_ff = PFX(ff_make)(FCAST(plan->Nfft), FCAST(0.0));
        PFX(ff_t) scale_ff = PFX(ff_make)(FCAST(plan->df * k_factor), FCAST(0.0));

        for (int m = 0; m < plan->Mpoints; ++m) {
            PFX(ff_t) x_ff = PFX(ff_make)(FCAST(x[m].hi), FCAST(x[m].lo));
            x_ff = PFX(ff_mul)(x_ff, scale_ff);

            PFX(ff_t) phase_ff = PFX(ff_mul)(x_ff, out_shift_ff);
            FLOAT p_int = FCAST((int)TO_DOUBLE(phase_ff.hi));
            PFX(ff_t) phase_frac_ff = PFX(ff_sub)(phase_ff, PFX(ff_make)(p_int, FCAST(0.0)));
            FLOAT phase_frac = ADD(phase_frac_ff.hi, phase_frac_ff.lo);

            current_shift_r[m] = M_COS2PI(phase_frac);
            current_shift_i[m] = M_SIN2PI(phase_frac);

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

            for (int l = 0; l < plan->w; ++l) {
                int idx = m_left + l;
                int wrap_idx = (idx % plan->Nfft + plan->Nfft) % plan->Nfft;
                size_t w_idx = (size_t)m * plan->w + l;
                current_spread_idx[w_idx] = wrap_idx;
                current_spread_weight[w_idx] = kval_buf[l];
            }
        }

#else
        double out_shift = (double)plan->output_shift;
        double n_fft = (double)plan->Nfft;

        for (int m = 0; m < plan->Mpoints; ++m) {
            double xm = x[m] * plan->df * k_factor;

            double phase = xm * out_shift;
            double p_int = (double)((int)phase);
            double phase_frac = phase - p_int;

            current_shift_r[m] = M_COS2PI(FCAST(phase_frac));
            current_shift_i[m] = M_SIN2PI(FCAST(phase_frac));

            double x_n = xm * n_fft;
            int m_left = (int)ceil(x_n - (double)plan->w * 0.5);

            for (int l = 0; l < plan->w; ++l) {
                int idx = m_left + l;
                double dist = (double)idx - x_n;
                z_buf[l] = FCAST((2.0 * dist) / (double)plan->w);
            }

            PFX(pswf0_batch)(z_buf, kval_buf, plan->w, plan->w, mode43);

            for (int l = 0; l < plan->w; ++l) {
                int idx = m_left + l;
                int wrap_idx = (idx % plan->Nfft + plan->Nfft) % plan->Nfft;
                size_t w_idx = (size_t)m * plan->w + l;
                current_spread_idx[w_idx] = wrap_idx;
                current_spread_weight[w_idx] = kval_buf[l];
            }
        }
#endif
    }
}

void PFX(pswf_execute)(struct PFX(pswf_plan) * plan, const FLOAT *y_real, const FLOAT *y_imag, FLOAT *out_real, FLOAT *out_imag, int freq_factor) {
    int f_idx = freq_factor - 1;
    if (f_idx < 0 || f_idx >= plan->num_factors) return;

    size_t offset_shift = (size_t)f_idx * plan->Mpoints;
    size_t offset_spread = (size_t)f_idx * plan->Mpoints * plan->w;

    FLOAT *current_shift_r = plan->shift_r + offset_shift;
    FLOAT *current_shift_i = plan->shift_i + offset_shift;
    int *current_spread_idx = plan->spread_idx + offset_spread;
    FLOAT *current_spread_weight = plan->spread_weight + offset_spread;

    memset(plan->fft_real, 0, plan->Nfft * sizeof(FLOAT));
    memset(plan->fft_imag, 0, plan->Nfft * sizeof(FLOAT));
    memset(plan->spread_comp_r, 0, plan->Nfft * sizeof(FLOAT));
    memset(plan->spread_comp_i, 0, plan->Nfft * sizeof(FLOAT));

    for (int m = 0; m < plan->Mpoints; ++m) {
        FLOAT yr = y_real[m], yi = y_imag[m];
        FLOAT sr = current_shift_r[m], si = current_shift_i[m];
        FLOAT sy_r = SUB(MUL(yr, sr), MUL(yi, si));
        FLOAT sy_i = ADD(MUL(yr, si), MUL(yi, sr));
        for (int l = 0; l < plan->w; ++l) {
            size_t w_idx = (size_t)m * plan->w + l;
            int idx = current_spread_idx[w_idx];
            FLOAT kval = current_spread_weight[w_idx];
            PFX(kahan_add_soa)(&plan->fft_real[idx], &plan->fft_imag[idx], &plan->spread_comp_r[idx], &plan->spread_comp_i[idx], MUL(sy_r, kval),
                               MUL(sy_i, kval));
        }
    }

    NANOFFT_EXECUTE(plan->nanofft_p, plan->fft_real, plan->fft_imag);

    for (int k = 0; k < plan->Nout; ++k) {
        int k_prime = k - plan->output_shift;
        int fft_idx = k_prime >= 0 ? k_prime : plan->Nfft + k_prime;
        out_real[k] = MUL(plan->fft_real[fft_idx], plan->deconv[k]);
        out_imag[k] = MUL(plan->fft_imag[fft_idx], plan->deconv[k]);
    }
}

void PFX(free_pswf_plan)(struct PFX(pswf_plan) * plan) {
    if (!plan) return;
    free(plan->spread_idx);
    free(plan->spread_weight);
    free(plan->spread_comp_r);
    free(plan->spread_comp_i);
    free(plan->deconv);
    free(plan->shift_r);
    free(plan->shift_i);
    free(plan->fft_real);
    free(plan->fft_imag);
    if (plan->nanofft_p) NANOFFT_DESTROY_PLAN(plan->nanofft_p);
    free(plan);
}
