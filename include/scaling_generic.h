#if defined(__AVX512F__) || (defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS >= 512)
#    define F_ALPHA 0.172
#    define F_LRA_BETA 4.74
#    define F_LRA_GAMMA 1.77
#    define F_PSWF21_BETA 4.53
#    define F_PSWF21_GAMMA 19.2
#    define F_PSWF43_BETA 6.79
#    define F_PSWF43_GAMMA 37.3
#    define F_PSWF21_PRE 71.0
#    define F_PSWF21_PPOLY_BETA 6.85
#    define F_PSWF21_PPOLY_PRE 17.0
#    define F_PSWF21_PPOLY_CONST 23900.0
#    define F_PSWF43_PRE 107.0
#    define F_PSWF43_PPOLY_BETA 10.4
#    define F_PSWF43_PPOLY_PRE 23.9
#    define F_PSWF43_PPOLY_CONST 35900.0

#    define D_ALPHA 0.195
#    define D_LRA_BETA 3.05
#    define D_LRA_GAMMA 2.48
#    define D_PSWF21_BETA 4.21
#    define D_PSWF21_GAMMA 18.7
#    define D_PSWF43_BETA 7.55
#    define D_PSWF43_GAMMA 72.4
#    define D_PSWF21_PRE 48.0
#    define D_PSWF21_PPOLY_BETA 7.60
#    define D_PSWF21_PPOLY_PRE 12.7
#    define D_PSWF21_PPOLY_CONST 35300.0
#    define D_PSWF43_PRE 96.1
#    define D_PSWF43_PPOLY_BETA 15.1
#    define D_PSWF43_PPOLY_PRE 27.6
#    define D_PSWF43_PPOLY_CONST 52900.0
#elif defined(__AVX__) || (defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS >= 256)
#    define F_ALPHA 0.183
#    define F_LRA_BETA 4.04
#    define F_LRA_GAMMA 1.80
#    define F_PSWF21_BETA 2.91
#    define F_PSWF21_GAMMA 13.8
#    define F_PSWF43_BETA 4.07
#    define F_PSWF43_GAMMA 35.2
#    define F_PSWF21_PRE 43.6
#    define F_PSWF21_PPOLY_BETA 5.39
#    define F_PSWF21_PPOLY_PRE 18.1
#    define F_PSWF21_PPOLY_CONST 24000.0
#    define F_PSWF43_PRE 72.1
#    define F_PSWF43_PPOLY_BETA 10.5
#    define F_PSWF43_PPOLY_PRE 28.6
#    define F_PSWF43_PPOLY_CONST 36000.0

#    define D_ALPHA 0.200
#    define D_LRA_BETA 2.69
#    define D_LRA_GAMMA 2.27
#    define D_PSWF21_BETA 3.28
#    define D_PSWF21_GAMMA 20.0
#    define D_PSWF43_BETA 5.65
#    define D_PSWF43_GAMMA 81.7
#    define D_PSWF21_PRE 70.4
#    define D_PSWF21_PPOLY_BETA 7.68
#    define D_PSWF21_PPOLY_PRE 11.1
#    define D_PSWF21_PPOLY_CONST 31900.0
#    define D_PSWF43_PRE 123.0
#    define D_PSWF43_PPOLY_BETA 13.9
#    define D_PSWF43_PPOLY_PRE 24.9
#    define D_PSWF43_PPOLY_CONST 47600.0
#else
#    define F_ALPHA 0.156
#    define F_LRA_BETA 2.27
#    define F_LRA_GAMMA 0.802
#    define F_PSWF21_BETA 1.53
#    define F_PSWF21_GAMMA 11.1
#    define F_PSWF43_BETA 2.32
#    define F_PSWF43_GAMMA 44.3
#    define F_PSWF21_PRE 30.7
#    define F_PSWF21_PPOLY_BETA 3.11
#    define F_PSWF21_PPOLY_PRE 9.96
#    define F_PSWF21_PPOLY_CONST 11400.0
#    define F_PSWF43_PRE 53.9
#    define F_PSWF43_PPOLY_BETA 6.80
#    define F_PSWF43_PPOLY_PRE 15.4
#    define F_PSWF43_PPOLY_CONST 17100.0

#    define D_ALPHA 0.141
#    define D_LRA_BETA 1.08
#    define D_LRA_GAMMA 0.788
#    define D_PSWF21_BETA 1.50
#    define D_PSWF21_GAMMA 19.2
#    define D_PSWF43_BETA 2.48
#    define D_PSWF43_GAMMA 87.0
#    define D_PSWF21_PRE 53.4
#    define D_PSWF21_PPOLY_BETA 6.00
#    define D_PSWF21_PPOLY_PRE 5.28
#    define D_PSWF21_PPOLY_CONST 11700.0
#    define D_PSWF43_PRE 87.1
#    define D_PSWF43_PPOLY_BETA 10.4
#    define D_PSWF43_PPOLY_PRE 10.5
#    define D_PSWF43_PPOLY_CONST 17300.0
#endif

#define DD_ALPHA 0.11
#define DD_LRA_BETA 0.65
#define DD_LRA_GAMMA 0.135
#define DD_PSWF21_BETA 2.5
#define DD_PSWF21_GAMMA 100.0
#define DD_PSWF43_BETA 3.75
#define DD_PSWF43_GAMMA 450.0
