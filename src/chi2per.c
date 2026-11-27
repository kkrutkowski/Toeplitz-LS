#include <linalg.h>
#define NANOFFT_NEEDS_INTERNAL_VEC
#include <math.h>
#include <nanofft_precision.h>
#include <nufft1.h>
#include <scaling.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <tls.h>
#include <tls_cpu.h>
#include <tls_internal.h>
#include <tls_tuning.h>
#include <utils.h>

/* Return codes from the public fastchi2 entry points. */
enum { CHI2PER_OK = 0, CHI2PER_ERR_ARGUMENT = -1, CHI2PER_ERR_BACKEND = -2, CHI2PER_ERR_ALLOC = -3, CHI2PER_ERR_DEGENERATE = -4, CHI2PER_ERR_SOLVER = -5 };

enum { CHI2PER_BACKEND_PSWF43 = 1, CHI2PER_BACKEND_PSWF21 = 2, CHI2PER_BACKEND_LRA = 3 };

enum { CHI2PER_SOLVER_LEVINSON = 1, CHI2PER_SOLVER_ZOHAR = 2, CHI2PER_SOLVER_BAREISS = 3, CHI2PER_SOLVER_LDLT = 4, CHI2PER_SOLVER_SVD = 5 };

__attribute__((unused)) static void *checked_malloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return NULL;
    if (count > SIZE_MAX / size) return NULL;
    return nanofft_malloc(count * size);
}

/* NaN sentinel for degenerate input; kept in one place so the fast-math
 * warning about the NAN macro is silenced once instead of at every use. */
NANOFFT_DIAG_PUSH_NAN
static inline NANOFFT_UNUSED FLOAT chi2_nan(void) { return FCAST(NAN); }
#if defined(HALF)
static inline float chi2_out_nan(void) { return (float)NAN; }
#else
static inline FLOAT chi2_out_nan(void) { return FCAST(NAN); }
#endif
NANOFFT_DIAG_POP
static void *checked_aligned_malloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return NULL;
    if (count > SIZE_MAX / size) return NULL;

    size_t bytes = count * size;
    size_t align = 64;
    size_t padded = (bytes + align - 1) & ~(align - 1);
    return nanofft_aligned_alloc(align, padded);
}

static inline int chi2_bitceil(unsigned int n) {
    if (n <= 1u) return 1;
    unsigned int v = n - 1u;
    int shift = 0;
    while (v) {
        v >>= 1;
        ++shift;
    }
    return 1 << shift;
}

#if defined(DOUBLE_DOUBLE)
#    define CHI2_PREFIX(name) tlsdd_##name
#    define TIME_INPUT_T dd_t
#    define NUFFT_INPUT_T dd_t
#    define NUFFT_LRA_INIT tlsdd_nufft_lra_initialize
#    define NUFFT_LRA_PRE tlsdd_nufft_lra_precompute
#    define NUFFT_LRA_EXEC tlsdd_nufft_lra_execute
#    define NUFFT_LRA_FREE tlsdd_nufft_free_lra_plan
#    define NUFFT_PSWF_INIT tlsdd_nufft_pswf_initialize
#    define NUFFT_PSWF_PRE tlsdd_nufft_pswf_precompute
#    define NUFFT_PSWF_EXEC tlsdd_nufft_pswf_execute
#    define NUFFT_PSWF_FREE tlsdd_nufft_free_pswf_plan
#    define LRA_PLAN_T tlsdd_nufft_lra_plan
#    define PSWF_PLAN_T tlsdd_nufft_pswf_plan
#    define SOLVE_LDLT tlsdd_solve_ldlt
#    define SOLVE_LEVINSON tlsdd_solve_levinson
#    define SOLVE_BAREISS tlsdd_solve_bareiss
#    define SOLVE_ZOHAR tlsdd_solve_zohar
#    define SOLVE_SVD tlsdd_solve_svd
#    define NUFFT_RANK 27
#    define NUFFT_W21 32
#    define NUFFT_W43 36
#    define CHI2_ALPHA DD_ALPHA
#    define CHI2_LRA_BETA DD_LRA_BETA
#    define CHI2_LRA_GAMMA DD_LRA_GAMMA
#    define CHI2_PSWF21_BETA DD_PSWF21_BETA
#    define CHI2_PSWF21_GAMMA DD_PSWF21_GAMMA
#    define CHI2_PSWF43_BETA DD_PSWF43_BETA
#    define CHI2_PSWF43_GAMMA DD_PSWF43_GAMMA
#    define CHI2_B_ALPHA DD_B_ALPHA
#    define CHI2_B_FFT_RATIO DD_B_FFT_RATIO
#    define CHI2_LRA_B_BETA DD_LRA_B_BETA
#    define CHI2_PSWF21_B_BETA DD_PSWF21_B_BETA
#    define CHI2_PSWF21_B_PRE DD_PSWF21_B_PRE
#    define CHI2_PSWF43_B_BETA DD_PSWF43_B_BETA
#    define CHI2_PSWF43_B_PRE DD_PSWF43_B_PRE
#    define CHI2_MIN_BLOCK_BREADTH 64
#    define COND_SINGULARITY_THRESHOLD_LEVINSON 1e26
#    define COND_SINGULARITY_THRESHOLD_ZOHAR 1e26
#    define COND_SINGULARITY_THRESHOLD_BAREISS 1e26
#    define COND_SINGULARITY_THRESHOLD_LDLT 1e10
#    define COND_SINGULARITY_THRESHOLD_SVD 1e16
static inline NANOFFT_UNUSED FLOAT time_to_float(TIME_INPUT_T x) { return x; }
static inline NUFFT_INPUT_T time_to_nufft_input(TIME_INPUT_T x) { return x; }
#else
#    define TIME_INPUT_T double
#    define NUFFT_INPUT_T double
#    if defined(DOUBLE)
#        define CHI2_PREFIX(name) tls_##name
#        define NUFFT_LRA_INIT tls_nufft_lra_initialize
#        define NUFFT_LRA_PRE tls_nufft_lra_precompute
#        define NUFFT_LRA_EXEC tls_nufft_lra_execute
#        define NUFFT_LRA_FREE tls_nufft_free_lra_plan
#        define NUFFT_PSWF_INIT tls_nufft_pswf_initialize
#        define NUFFT_PSWF_INIT_MODE tls_nufft_pswf_initialize_mode
#        define NUFFT_PSWF_PRE tls_nufft_pswf_precompute
#        define NUFFT_PSWF_EXEC tls_nufft_pswf_execute
#        define NUFFT_PSWF_FREE tls_nufft_free_pswf_plan
#        define LRA_PLAN_T tls_nufft_lra_plan
#        define PSWF_PLAN_T tls_nufft_pswf_plan
#        define SOLVE_LDLT tls_solve_ldlt
#        define SOLVE_LEVINSON tls_solve_levinson
#        define SOLVE_BAREISS tls_solve_bareiss
#        define SOLVE_ZOHAR tls_solve_zohar
#        define SOLVE_SVD tls_solve_svd
#        define NUFFT_RANK 16
#        define NUFFT_W21 16
#        define NUFFT_W43 18
#        define CHI2_ALPHA D_ALPHA
#        define CHI2_LRA_BETA D_LRA_BETA
#        define CHI2_LRA_GAMMA D_LRA_GAMMA
#        define CHI2_PSWF21_BETA D_PSWF21_BETA
#        define CHI2_PSWF21_GAMMA D_PSWF21_GAMMA
#        define CHI2_PSWF43_BETA D_PSWF43_BETA
#        define CHI2_PSWF43_GAMMA D_PSWF43_GAMMA
#        define CHI2_PSWF21_PRE D_PSWF21_PRE
#        define CHI2_PSWF21_PPOLY_BETA D_PSWF21_PPOLY_BETA
#        define CHI2_PSWF21_PPOLY_PRE D_PSWF21_PPOLY_PRE
#        define CHI2_PSWF21_PPOLY_CONST D_PSWF21_PPOLY_CONST
#        define CHI2_PSWF43_PRE D_PSWF43_PRE
#        define CHI2_PSWF43_PPOLY_BETA D_PSWF43_PPOLY_BETA
#        define CHI2_PSWF43_PPOLY_PRE D_PSWF43_PPOLY_PRE
#        define CHI2_PSWF43_PPOLY_CONST D_PSWF43_PPOLY_CONST
#        define CHI2_B_ALPHA D_B_ALPHA
#        define CHI2_B_FFT_RATIO D_B_FFT_RATIO
#        define CHI2_LRA_B_BETA D_LRA_B_BETA
#        define CHI2_PSWF21_B_BETA D_PSWF21_B_BETA
#        define CHI2_PSWF21_B_PRE D_PSWF21_B_PRE
#        define CHI2_PSWF43_B_BETA D_PSWF43_B_BETA
#        define CHI2_PSWF43_B_PRE D_PSWF43_B_PRE
#        define CHI2_PSWF21_B_PPOLY_BETA D_PSWF21_B_PPOLY_BETA
#        define CHI2_PSWF21_B_PPOLY_PRE D_PSWF21_B_PPOLY_PRE
#        define CHI2_PSWF43_B_PPOLY_BETA D_PSWF43_B_PPOLY_BETA
#        define CHI2_PSWF43_B_PPOLY_PRE D_PSWF43_B_PPOLY_PRE
#        define CHI2_MIN_BLOCK_BREADTH 32
#        define COND_SINGULARITY_THRESHOLD_LEVINSON 1e13
#        define COND_SINGULARITY_THRESHOLD_ZOHAR 1e13
#        define COND_SINGULARITY_THRESHOLD_BAREISS 1e13
#        define COND_SINGULARITY_THRESHOLD_LDLT 1e5
#        define COND_SINGULARITY_THRESHOLD_SVD 1e8
#    elif defined(HALF)
#        ifdef HALF_NATIVE_EMBED
#            define CHI2_PREFIX(name) tlsh_nat_##name
#            define NUFFT_LRA_INIT tlsh_nat_nufft_lra_initialize
#            define NUFFT_LRA_PRE tlsh_nat_nufft_lra_precompute
#            define NUFFT_LRA_EXEC tlsh_nat_nufft_lra_execute
#            define NUFFT_LRA_FREE tlsh_nat_nufft_free_lra_plan
#            define NUFFT_PSWF_INIT tlsh_nat_nufft_pswf_initialize
#            define NUFFT_PSWF_INIT_MODE tlsh_nat_nufft_pswf_initialize_mode
#            define NUFFT_PSWF_PRE tlsh_nat_nufft_pswf_precompute
#            define NUFFT_PSWF_EXEC tlsh_nat_nufft_pswf_execute
#            define NUFFT_PSWF_FREE tlsh_nat_nufft_free_pswf_plan
#            define LRA_PLAN_T tlsh_nat_nufft_lra_plan
#            define PSWF_PLAN_T tlsh_nat_nufft_pswf_plan
#            define SOLVE_LDLT tlsh_nat_solve_ldlt
#            define SOLVE_LEVINSON tlsh_nat_solve_levinson
#            define SOLVE_BAREISS tlsh_nat_solve_bareiss
#            define SOLVE_ZOHAR tlsh_nat_solve_zohar
#            define SOLVE_SVD tlsh_nat_solve_svd
#        else
#            define CHI2_PREFIX(name) tlsh_##name
#            define NUFFT_LRA_INIT tlsh_nufft_lra_initialize
#            define NUFFT_LRA_PRE tlsh_nufft_lra_precompute
#            define NUFFT_LRA_EXEC tlsh_nufft_lra_execute
#            define NUFFT_LRA_FREE tlsh_nufft_free_lra_plan
#            define NUFFT_PSWF_INIT tlsh_nufft_pswf_initialize
#            define NUFFT_PSWF_INIT_MODE tlsh_nufft_pswf_initialize_mode
#            define NUFFT_PSWF_PRE tlsh_nufft_pswf_precompute
#            define NUFFT_PSWF_EXEC tlsh_nufft_pswf_execute
#            define NUFFT_PSWF_FREE tlsh_nufft_free_pswf_plan
#            define LRA_PLAN_T tlsh_nufft_lra_plan
#            define PSWF_PLAN_T tlsh_nufft_pswf_plan
#            define SOLVE_LDLT tlsh_solve_ldlt
#            define SOLVE_LEVINSON tlsh_solve_levinson
#            define SOLVE_BAREISS tlsh_solve_bareiss
#            define SOLVE_ZOHAR tlsh_solve_zohar
#            define SOLVE_SVD tlsh_solve_svd
#        endif
#        define NUFFT_RANK 5
#        define NUFFT_W21 4
#        define NUFFT_W43 5
#        define CHI2_ALPHA H_ALPHA
#        define CHI2_LRA_BETA H_LRA_BETA
#        define CHI2_LRA_GAMMA H_LRA_GAMMA
#        define CHI2_PSWF21_BETA H_PSWF21_BETA
#        define CHI2_PSWF21_GAMMA H_PSWF21_GAMMA
#        define CHI2_PSWF43_BETA H_PSWF43_BETA
#        define CHI2_PSWF43_GAMMA H_PSWF43_GAMMA
#        define CHI2_PSWF21_PRE H_PSWF21_PRE
#        define CHI2_PSWF21_PPOLY_BETA H_PSWF21_PPOLY_BETA
#        define CHI2_PSWF21_PPOLY_PRE H_PSWF21_PPOLY_PRE
#        define CHI2_PSWF21_PPOLY_CONST H_PSWF21_PPOLY_CONST
#        define CHI2_PSWF43_PRE H_PSWF43_PRE
#        define CHI2_PSWF43_PPOLY_BETA H_PSWF43_PPOLY_BETA
#        define CHI2_PSWF43_PPOLY_PRE H_PSWF43_PPOLY_PRE
#        define CHI2_PSWF43_PPOLY_CONST H_PSWF43_PPOLY_CONST
#        define CHI2_B_ALPHA H_B_ALPHA
#        define CHI2_B_FFT_RATIO H_B_FFT_RATIO
#        define CHI2_LRA_B_BETA H_LRA_B_BETA
#        define CHI2_PSWF21_B_BETA H_PSWF21_B_BETA
#        define CHI2_PSWF21_B_PRE H_PSWF21_B_PRE
#        define CHI2_PSWF43_B_BETA H_PSWF43_B_BETA
#        define CHI2_PSWF43_B_PRE H_PSWF43_B_PRE
#        define CHI2_PSWF21_B_PPOLY_BETA H_PSWF21_B_PPOLY_BETA
#        define CHI2_PSWF21_B_PPOLY_PRE H_PSWF21_B_PPOLY_PRE
#        define CHI2_PSWF43_B_PPOLY_BETA H_PSWF43_B_PPOLY_BETA
#        define CHI2_PSWF43_B_PPOLY_PRE H_PSWF43_B_PPOLY_PRE
#        define CHI2_MIN_BLOCK_BREADTH 16
#        define COND_SINGULARITY_THRESHOLD_LEVINSON 500.0
#        define COND_SINGULARITY_THRESHOLD_ZOHAR 500.0
#        define COND_SINGULARITY_THRESHOLD_BAREISS 500.0
#        define COND_SINGULARITY_THRESHOLD_LDLT 100.0
#        define COND_SINGULARITY_THRESHOLD_SVD 200.0
#    else
#        define CHI2_PREFIX(name) tlsf_##name
#        define NUFFT_LRA_INIT tlsf_nufft_lra_initialize
#        define NUFFT_LRA_PRE tlsf_nufft_lra_precompute
#        define NUFFT_LRA_EXEC tlsf_nufft_lra_execute
#        define NUFFT_LRA_FREE tlsf_nufft_free_lra_plan
#        define NUFFT_PSWF_INIT tlsf_nufft_pswf_initialize
#        define NUFFT_PSWF_INIT_MODE tlsf_nufft_pswf_initialize_mode
#        define NUFFT_PSWF_PRE tlsf_nufft_pswf_precompute
#        define NUFFT_PSWF_EXEC tlsf_nufft_pswf_execute
#        define NUFFT_PSWF_FREE tlsf_nufft_free_pswf_plan
#        define LRA_PLAN_T tlsf_nufft_lra_plan
#        define PSWF_PLAN_T tlsf_nufft_pswf_plan
#        define SOLVE_LDLT tlsf_solve_ldlt
#        define SOLVE_LEVINSON tlsf_solve_levinson
#        define SOLVE_BAREISS tlsf_solve_bareiss
#        define SOLVE_ZOHAR tlsf_solve_zohar
#        define SOLVE_SVD tlsf_solve_svd
#        define NUFFT_RANK 9
#        define NUFFT_W21 8
#        define NUFFT_W43 9
#        define CHI2_ALPHA F_ALPHA
#        define CHI2_LRA_BETA F_LRA_BETA
#        define CHI2_LRA_GAMMA F_LRA_GAMMA
#        define CHI2_PSWF21_BETA F_PSWF21_BETA
#        define CHI2_PSWF21_GAMMA F_PSWF21_GAMMA
#        define CHI2_PSWF43_BETA F_PSWF43_BETA
#        define CHI2_PSWF43_GAMMA F_PSWF43_GAMMA
#        define CHI2_PSWF21_PRE F_PSWF21_PRE
#        define CHI2_PSWF21_PPOLY_BETA F_PSWF21_PPOLY_BETA
#        define CHI2_PSWF21_PPOLY_PRE F_PSWF21_PPOLY_PRE
#        define CHI2_PSWF21_PPOLY_CONST F_PSWF21_PPOLY_CONST
#        define CHI2_PSWF43_PRE F_PSWF43_PRE
#        define CHI2_PSWF43_PPOLY_BETA F_PSWF43_PPOLY_BETA
#        define CHI2_PSWF43_PPOLY_PRE F_PSWF43_PPOLY_PRE
#        define CHI2_PSWF43_PPOLY_CONST F_PSWF43_PPOLY_CONST
#        define CHI2_B_ALPHA F_B_ALPHA
#        define CHI2_B_FFT_RATIO F_B_FFT_RATIO
#        define CHI2_LRA_B_BETA F_LRA_B_BETA
#        define CHI2_PSWF21_B_BETA F_PSWF21_B_BETA
#        define CHI2_PSWF21_B_PRE F_PSWF21_B_PRE
#        define CHI2_PSWF43_B_BETA F_PSWF43_B_BETA
#        define CHI2_PSWF43_B_PRE F_PSWF43_B_PRE
#        define CHI2_PSWF21_B_PPOLY_BETA F_PSWF21_B_PPOLY_BETA
#        define CHI2_PSWF21_B_PPOLY_PRE F_PSWF21_B_PPOLY_PRE
#        define CHI2_PSWF43_B_PPOLY_BETA F_PSWF43_B_PPOLY_BETA
#        define CHI2_PSWF43_B_PPOLY_PRE F_PSWF43_B_PPOLY_PRE
#        define CHI2_MIN_BLOCK_BREADTH 16
#        define COND_SINGULARITY_THRESHOLD_LEVINSON 5e5
#        define COND_SINGULARITY_THRESHOLD_ZOHAR 5e5
#        define COND_SINGULARITY_THRESHOLD_BAREISS 5e5
#        define COND_SINGULARITY_THRESHOLD_LDLT 1e2
#        define COND_SINGULARITY_THRESHOLD_SVD 1e3
#    endif
static inline NANOFFT_UNUSED FLOAT time_to_float(TIME_INPUT_T x) { return FCAST(x); }
static inline NUFFT_INPUT_T time_to_nufft_input(TIME_INPUT_T x) { return x; }
#endif

#if defined(HALF)
typedef float OUT_FLOAT;
typedef struct {
    float elems[BREADTH_VEC_LEN];
} OUT_BVEC;
#    define OUT_BVEC_ZERO ((OUT_BVEC){{0.0f}})
#    define OUT_BVEC_GET_LANE(v, lane) ((v).elems[(lane)])
#    define OUT_BVEC_SET_LANE(v, lane, val) \
        do {                                \
            (v).elems[(lane)] = (val);      \
        } while (0)
#    define OUT_BVEC_STORE(ptr, val) (*(ptr) = (val))
#    define OUT_FROM_FLOAT(v) tls_f16_to_f32((v))
#    define OUT_TO_DOUBLE(v) ((double)(v))
#    define OUT_CAST(x) ((float)(x))
#    define OUT_MUL(a, b) ((a) * (b))
#    define NUFFT_PSWF_FINISH_SLOTS_OUT CHI2_PREFIX(nufft_pswf_finish_slots_f32)
#    define NUFFT_PSWF_FINISH_SLOTS_BREADTH_OUT(plan, compact, out, blk, al, rem) \
        CHI2_PREFIX(nufft_pswf_finish_slots_breadth_f32)((plan), (const float *)(const void *)(compact), (out), (blk), (al), (rem))
#else
typedef FLOAT OUT_FLOAT;
typedef BREADTH_VEC OUT_BVEC;
#    define OUT_BVEC_ZERO BREADTH_ZERO
#    define OUT_BVEC_GET_LANE(v, lane) BREADTH_GET_LANE((v), (lane))
#    define OUT_BVEC_SET_LANE(v, lane, val) BREADTH_SET_LANE((v), (lane), (val))
#    define OUT_BVEC_STORE(ptr, val) STORE_BVEC((ptr), (val))
#    define OUT_FROM_FLOAT(v) (v)
#    define OUT_TO_DOUBLE(v) TO_DOUBLE((v))
#    define OUT_CAST(x) FCAST((x))
#    define OUT_MUL(a, b) MUL((a), (b))
#    define NUFFT_PSWF_FINISH_SLOTS_OUT NUFFT_PSWF_FINISH_SLOTS
#    define NUFFT_PSWF_FINISH_SLOTS_BREADTH_OUT(plan, compact, out, blk, al, rem) NUFFT_PSWF_FINISH_SLOTS_BREADTH((plan), (compact), (out), (blk), (al), (rem))
#endif

#if defined(HALF) && !NANOFFT_NATIVE_FP16
#    define BVEC_STORE_ZERO ((VECH){{{0}}})
#    define BVEC_STORE_GET_LANE(v, lane) ((v).elems[(lane)])
#    define BVEC_STORE_SET_LANE(v, lane, val) \
        do {                                  \
            (v).elems[(lane)] = (val);        \
        } while (0)
#else
#    define BVEC_STORE_ZERO BREADTH_ZERO
#    define BVEC_STORE_GET_LANE(v, lane) BREADTH_GET_LANE((v), (lane))
#    define BVEC_STORE_SET_LANE(v, lane, val) BREADTH_SET_LANE((v), (lane), (val))
#endif

/* Slot-order NuFFT output (deferred bit reversal) and the tuning switch,
 * both resolved through the precision prefix. */
#define NUFFT_PSWF_EXEC_MODE CHI2_PREFIX(nufft_pswf_execute_mode)
#define NUFFT_PSWF_EXEC_MODE_PAIR CHI2_PREFIX(nufft_pswf_execute_mode_pair)
#define NUFFT_PSWF_EXEC_BREADTH_PAIR CHI2_PREFIX(nufft_pswf_execute_breadth_pair)
#define NUFFT_PSWF_FINISH_SLOTS CHI2_PREFIX(nufft_pswf_finish_slots)
#define NUFFT_PSWF_SLOTS_AVAILABLE CHI2_PREFIX(nufft_pswf_slots_available)
#define NUFFT_PSWF_SLOT_PHI_HAT CHI2_PREFIX(nufft_pswf_slot_phi_hat)
#define NUFFT_PSWF_SLOT_DECONV CHI2_PREFIX(nufft_pswf_slot_deconv)
#define NUFFT_PSWF_DECONV CHI2_PREFIX(nufft_pswf_deconv)
#if defined(HALF)
#    define NUFFT_PSWF_SET_OUTPUT_SCALE CHI2_PREFIX(nufft_pswf_set_output_scale)
#    define NUFFT_PSWF_INIT_GEO_OPTS CHI2_PREFIX(nufft_pswf_initialize_geo_opts)
#    define NUFFT_PSWF_IHS_DECONV CHI2_PREFIX(nufft_pswf_deconv_f32)
#    define NUFFT_PSWF_IHS_SLOT_DECONV CHI2_PREFIX(nufft_pswf_slot_deconv_f32)
#else
#    define NUFFT_PSWF_IHS_DECONV NUFFT_PSWF_DECONV
#    define NUFFT_PSWF_IHS_SLOT_DECONV NUFFT_PSWF_SLOT_DECONV
#endif
#define NUFFT_PSWF_INIT_OPTS CHI2_PREFIX(nufft_pswf_initialize_opts)
#define NUFFT_PSWF_EXEC_BREADTH CHI2_PREFIX(nufft_pswf_execute_breadth)
#define NUFFT_PSWF_FINISH_SLOTS_BREADTH CHI2_PREFIX(nufft_pswf_finish_slots_breadth)
#define NUFFT_LRA_INIT_OPTS CHI2_PREFIX(nufft_lra_initialize_opts)
#define NUFFT_LRA_EXEC_BREADTH CHI2_PREFIX(nufft_lra_execute_breadth)
static tls_plan_choice select_periodogram_plan(int N, int M, int degree, int is_ihs, int backend, int spreading, int vectorize) {
    const bool use_pswf43 = (backend == CHI2PER_BACKEND_PSWF43);
    const bool is_lra = (backend == CHI2PER_BACKEND_LRA);
    const double gamma = is_lra ? CHI2_LRA_GAMMA : (use_pswf43 ? CHI2_PSWF43_GAMMA : CHI2_PSWF21_GAMMA);
    const double beta_mm = is_lra ? CHI2_LRA_BETA : (use_pswf43 ? CHI2_PSWF43_BETA : CHI2_PSWF21_BETA);
    const double b_beta_mm = is_lra ? CHI2_LRA_B_BETA : (use_pswf43 ? CHI2_PSWF43_B_BETA : CHI2_PSWF21_B_BETA);
    const double b_pre_mm = is_lra ? 0.0 : (use_pswf43 ? CHI2_PSWF43_B_PRE : CHI2_PSWF21_B_PRE);
#if defined(DOUBLE_DOUBLE)
    return tls_optimize_plan(N, M, degree, is_ihs, backend, BREADTH_VEC_LEN, CHI2_MIN_BLOCK_BREADTH, 1, CHI2_ALPHA, gamma, beta_mm, 0.0, 0.0, 0.0, 0.0,
                             CHI2_B_ALPHA, CHI2_B_FFT_RATIO, b_beta_mm, b_pre_mm, 0.0, 0.0, spreading, vectorize);
#else
    const double pre_mm = is_lra ? 0.0 : (use_pswf43 ? CHI2_PSWF43_PRE : CHI2_PSWF21_PRE);
    const double beta_pp = is_lra ? 0.0 : (use_pswf43 ? CHI2_PSWF43_PPOLY_BETA : CHI2_PSWF21_PPOLY_BETA);
    const double pre_pp = is_lra ? 0.0 : (use_pswf43 ? CHI2_PSWF43_PPOLY_PRE : CHI2_PSWF21_PPOLY_PRE);
    const double ppoly_const = is_lra ? 0.0 : (use_pswf43 ? CHI2_PSWF43_PPOLY_CONST : CHI2_PSWF21_PPOLY_CONST);
    const double b_beta_pp = is_lra ? 0.0 : (use_pswf43 ? CHI2_PSWF43_B_PPOLY_BETA : CHI2_PSWF21_B_PPOLY_BETA);
    const double b_pre_pp = is_lra ? 0.0 : (use_pswf43 ? CHI2_PSWF43_B_PPOLY_PRE : CHI2_PSWF21_B_PPOLY_PRE);
    return tls_optimize_plan(N, M, degree, is_ihs, backend, BREADTH_VEC_LEN, CHI2_MIN_BLOCK_BREADTH, 0, CHI2_ALPHA, gamma, beta_mm, pre_mm, beta_pp, pre_pp,
                             ppoly_const, CHI2_B_ALPHA, CHI2_B_FFT_RATIO, b_beta_mm, b_pre_mm, b_beta_pp, b_pre_pp, spreading, vectorize);
#endif
}

static inline NANOFFT_UNUSED int double_is_nan_bits(double value) {
    union {
        double f;
        uint64_t u;
    } bits = {value};
    return (bits.u & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000);
}

static inline NANOFFT_UNUSED int float_is_nan_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return (bits.u & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
}

static const double condition_singularity_thresholds[] = {
    0.0,
    COND_SINGULARITY_THRESHOLD_LEVINSON,
    COND_SINGULARITY_THRESHOLD_ZOHAR,
    COND_SINGULARITY_THRESHOLD_BAREISS,
    COND_SINGULARITY_THRESHOLD_LDLT,
    COND_SINGULARITY_THRESHOLD_SVD,
};

static inline double condition_singularity_threshold(int solver) {
    if (solver < CHI2PER_SOLVER_LEVINSON || solver > CHI2PER_SOLVER_SVD) return 0.0;
    return condition_singularity_thresholds[solver];
}

#if defined(HALF)
static inline NANOFFT_UNUSED int f16_is_nan_bits(tls_f16_t h) { return (h.bits & 0x7c00u) == 0x7c00u && (h.bits & 0x03ffu) != 0u; }
static inline NANOFFT_UNUSED int f16_is_finite_bits(tls_f16_t h) { return (h.bits & 0x7c00u) != 0x7c00u; }
#endif

static inline int condition_bound_is_singular(FLOAT bound, int solver) {
    double threshold = condition_singularity_threshold(solver);
#if defined(DOUBLE_DOUBLE)
    double value = TO_DOUBLE(bound);
    return double_is_nan_bits(bound.hi) || double_is_nan_bits(bound.lo) || value < 0.0 || value > threshold;
#elif defined(DOUBLE)
    return double_is_nan_bits(bound) || bound < 0.0 || bound > threshold;
#elif defined(HALF)
    double value = TO_DOUBLE(bound);
    return f16_is_nan_bits(bound) || !f16_is_finite_bits(bound) || value < 0.0 || value > threshold;
#else
    return float_is_nan_bits(bound) || bound < 0.0f || (double)bound > threshold;
#endif
}

static inline int condition_bound_is_invalid(FLOAT bound) {
#if defined(DOUBLE_DOUBLE)
    double value = TO_DOUBLE(bound);
    return double_is_nan_bits(bound.hi) || double_is_nan_bits(bound.lo) || value < 0.0;
#elif defined(DOUBLE)
    return double_is_nan_bits(bound) || bound < 0.0;
#elif defined(HALF)
    return f16_is_nan_bits(bound) || TO_DOUBLE(bound) < 0.0;
#else
    return float_is_nan_bits(bound) || bound < 0.0f;
#endif
}

static inline int power_is_nan(OUT_FLOAT value) {
#if defined(DOUBLE_DOUBLE)
    return double_is_nan_bits(value.hi) || double_is_nan_bits(value.lo);
#elif defined(DOUBLE)
    return double_is_nan_bits(value);
#else
    return float_is_nan_bits(value);
#endif
}

static inline int double_is_finite_bits(double value) {
    union {
        double f;
        uint64_t u;
    } bits = {value};
    return (bits.u & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

static inline NANOFFT_UNUSED int float_is_finite_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return (bits.u & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static inline int power_is_finite(OUT_FLOAT value) {
#if defined(DOUBLE_DOUBLE)
    return double_is_finite_bits(value.hi) && double_is_finite_bits(value.lo);
#elif defined(DOUBLE)
    return double_is_finite_bits(value);
#else
    return float_is_finite_bits(value);
#endif
}

typedef struct {
    double var_S;
    double mean_S;
    double k_eff;     /* shape parameter a */
    double theta_eff; /* scale parameter theta */
    double lgamma_k;  /* ln(Gamma(k_eff)) cached */
    double sum_N_eff; /* sum_{j=1}^{2d} (M-j)^2 / M */
    double inv_2d;    /* 1.0 / (2.0 * d) */
} cybenko_model_t;

static inline cybenko_model_t cybenko_model_init(int d, double M) {
    cybenko_model_t m;
    double d_dbl = (double)d;
    double M_dbl = M;
    double inv_M = 1.0 / M_dbl;
    double inv_M2 = inv_M * inv_M;

    /* Exact 3rd-order Edgeworth expansion for Var(S) */
    double term1 = 2.0 * d_dbl;
    double term2 = (8.0 * d_dbl * d_dbl - 14.0 * d_dbl) * inv_M;
    double term3 = ((184.0 / 3.0) * d_dbl * d_dbl * d_dbl - 98.0 * d_dbl * d_dbl + (158.0 / 3.0) * d_dbl) * inv_M2;
    m.var_S = term1 + term2 + term3;

    m.mean_S = 2.0 * d_dbl;
    m.k_eff = (m.mean_S * m.mean_S) / m.var_S;
    m.theta_eff = m.var_S / m.mean_S;
    m.lgamma_k = lgamma(m.k_eff);
    m.inv_2d = 1.0 / (2.0 * d_dbl);

    double n = 2.0 * d_dbl;
    m.sum_N_eff = n * M_dbl - n * (n + 1.0) + (n * (n + 1.0) * (2.0 * n + 1.0)) / (6.0 * M_dbl);

    return m;
}

static inline NANOFFT_UNUSED double cybenko_kappa_to_S(double kappa, const cybenko_model_t *m) {
    if (kappa <= 1.0 || !double_is_finite_bits(kappa)) return 0.0;
    if (kappa > 5000.0) kappa = 5000.0;
    double pow_k = pow(kappa, m->inv_2d);
    double k_val = (pow_k - 1.0) / (pow_k + 1.0);
    double k2 = k_val * k_val;
    return m->sum_N_eff * k2;
}

#if defined(DOUBLE_DOUBLE)
static inline dd_t cybenko_kappa_to_S_dd(dd_t kappa, const cybenko_model_t *m) {
    double kh = kappa.hi;
    if (kh <= 1.0 || !isfinite(kh)) return dd_make(0.0, 0.0);
    if (kh > 5000.0) kh = 5000.0;
    double pow_k = pow(kh, m->inv_2d);
    double k_val = (pow_k - 1.0) / (pow_k + 1.0);
    double k2 = k_val * k_val;
    return dd_mul(dd_make(m->sum_N_eff, 0.0), dd_make(k2, 0.0));
}

static inline dd_t nll_exact_pochhammer_dd(int d, dd_t r2, dd_t b) {
    if (r2.hi <= 0.0 || !isfinite(r2.hi)) return dd_make(0.0, 0.0);
    if (r2.hi > 0.999999999999999) r2 = dd_make(0.999999999999999, 0.0);

    dd_t one = dd_make(1.0, 0.0);
    dd_t one_minus_r2 = dd_sub(one, r2);
    dd_t ln_one_minus_r2 = dd_log(one_minus_r2);
    dd_t term0 = dd_mul(dd_make(-b.hi, -b.lo), ln_one_minus_r2);
    if (d <= 1) {
        return (term0.hi > 0.0 && isfinite(term0.hi)) ? term0 : dd_make(0.0, 0.0);
    }

    dd_t c = one;
    dd_t s_minus_1 = dd_make(0.0, 0.0);
    for (int k = 1; k < d; ++k) {
        dd_t num = dd_add(b, dd_make((double)(k - 1), 0.0));
        dd_t factor = dd_div(num, dd_make((double)k, 0.0));
        c = dd_mul(dd_mul(c, factor), r2);
        s_minus_1 = dd_add(s_minus_1, c);
    }

    dd_t s = dd_add(one, s_minus_1);
    dd_t ln_s = dd_log(s);
    dd_t nll = dd_sub(term0, ln_s);
    return (nll.hi > 0.0 && isfinite(nll.hi)) ? nll : dd_make(0.0, 0.0);
}

static inline dd_t nll_cybenko_dd(dd_t kappa, const cybenko_model_t *m, int d) {
    dd_t s = cybenko_kappa_to_S_dd(kappa, m);
    if (s.hi <= 0.0 || !isfinite(s.hi)) return dd_make(0.0, 0.0);

    double k_eff = m->k_eff;
    double th_eff = m->theta_eff;
    double lg_k = m->lgamma_k;

    if (d <= 2) {
        dd_t z = dd_div(s, dd_make(th_eff, 0.0));
        int order = 2 * d;
        double delta_val = k_eff - (double)order;

        dd_t c = dd_make(1.0, 0.0);
        dd_t poly = dd_make(1.0, 0.0);
        dd_t d_poly = dd_make(0.0, 0.0);

        for (int k = 1; k < order; ++k) {
            dd_t factor = dd_make(1.0 / (double)k, 0.0);
            c = dd_mul(dd_mul(c, z), factor);
            poly = dd_add(poly, c);
            if (fabs(delta_val) > 1.0e-5) {
                double H_k = 0.0;
                for (int i = 1; i <= k; ++i) H_k += 1.0 / (double)i;
                d_poly = dd_sub(d_poly, dd_mul(c, dd_make(H_k, 0.0)));
            }
        }

        dd_t ln_poly = dd_log(poly);
        dd_t nll = dd_sub(z, ln_poly);
        if (fabs(delta_val) > 1.0e-5) {
            dd_t ln_z = dd_log(z);
            dd_t term = dd_add(ln_z, dd_div(d_poly, poly));
            nll = dd_sub(nll, dd_mul(dd_make(delta_val, 0.0), term));
        }
        return (nll.hi > 0.0 && isfinite(nll.hi)) ? nll : dd_make(0.0, 0.0);
    } else {
        double s_d = s.hi;
        double a = k_eff;
        double z_d = s_d / th_eff;
        if (z_d < a + 3.0) {
            double c = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 60; ++k) {
                c *= z_d / (a + (double)(k - 1));
                sum += c;
                if (c < 1.0e-14 * sum) break;
            }
            double lg_inc = a * log(z_d) - z_d - log(a) + log(sum);
            double nll = lg_k - lg_inc;
            return dd_make((nll > 0.0 && isfinite(nll)) ? nll : 0.0, 0.0);
        } else {
            double inv_z = 1.0 / z_d;
            double t1 = (a - 1.0) * inv_z;
            double t2 = t1 * ((a - 2.0) * inv_z);
            double t3 = t2 * ((a - 3.0) * inv_z);
            double t4 = t3 * ((a - 4.0) * inv_z);
            double poly_val = 1.0 + t1 + t2 + t3 + t4;
            double nll = z_d - (a - 1.0) * log(z_d) + lg_k - log(poly_val);
            return dd_make((nll > 0.0 && isfinite(nll)) ? nll : 0.0, 0.0);
        }
    }
}

static inline dd_t compute_bayes_laplace_dd(dd_t delta) {
    static const dd_t ln2 = {0.693147180559945309417, 2.31904681384629955841e-17};
    static const dd_t half = {0.5, 0.0};
    static const dd_t one = {1.0, 0.0};

    if (delta.hi >= 0.0) {
        return dd_add(delta, ln2);
    } else {
        dd_t exp_delta = dd_exp(delta);
        dd_t inner = dd_sub(one, dd_mul(half, exp_delta));
        if (inner.hi <= 0.0) return dd_make(0.0, 0.0);
        dd_t log_inner = dd_log(inner);
        return dd_make(-log_inner.hi, -log_inner.lo);
    }
}

static inline FLOAT normalize_power_entry(FLOAT dot, FLOAT chi2_ref, FLOAT condition_bound, int normalization, int degree, int M, const cybenko_model_t *cyb,
                                          FLOAT b) {
    /* Non-finite explained chi^2 or R^2 >= 1 is a numerical artefact of an
     * unstable solve: report NaN so it cannot be mistaken for zero power.
     * Finite non-positive values (e.g. near-zero rounding) report 0. */
    if (!double_is_finite_bits(dot.hi) || !double_is_finite_bits(dot.lo)) return chi2_nan();
    if (dot.hi > chi2_ref.hi || (dot.hi == chi2_ref.hi && dot.lo >= chi2_ref.lo)) return chi2_nan();
    if (dot.hi < 0.0 || (dot.hi == 0.0 && dot.lo <= 0.0)) return FCAST(0.0);
    dd_t r2 = DIV(dot, chi2_ref);
    if (!double_is_finite_bits(r2.hi) || !double_is_finite_bits(r2.lo) || r2.hi >= 1.0) return chi2_nan();
    if (normalization == TLS_NORM_STANDARD) {
        return r2;
    } else if (normalization == TLS_NORM_ANOVA) {
        double d1 = 2.0 * (double)degree + 1.0;
        double d2 = (double)M - 1.0 - d1;
        dd_t residual = dd_sub(chi2_ref, dot);
        if (residual.hi <= 1e-32) residual = dd_make(1e-32, 0.0);
        return dd_div(dd_mul(dd_make(d2, 0.0), dot), dd_mul(dd_make(d1, 0.0), residual));
    } else if (normalization == TLS_NORM_NLL) {
        return nll_exact_pochhammer_dd(degree, r2, b);
    } else { /* TLS_NORM_REGULARIZED */
        dd_t nll_r2 = nll_exact_pochhammer_dd(degree, r2, b);
        /* condition_bound is (2 * degree + 1) * P; NLL uses the product P. */
        dd_t kappa = dd_div(condition_bound, dd_make((double)(2 * degree + 1), 0.0));
        dd_t nll_cond = nll_cybenko_dd(kappa, cyb, degree);
        dd_t delta = dd_sub(nll_r2, nll_cond);
        return compute_bayes_laplace_dd(delta);
    }
}
#elif defined(DOUBLE) || defined(HALF)
static inline double nll_exact_pochhammer_scalar_d(int d, double r2, double b) {
    if (!double_is_finite_bits(r2) || r2 <= 0.0) return 0.0;
    if (r2 > 0.9999999999999) r2 = 0.9999999999999;
    double term0 = -b * log1p(-r2);
    if (d <= 1) return (term0 > 0.0 && double_is_finite_bits(term0)) ? term0 : 0.0;
    double c = 1.0;
    double s_minus_1 = 0.0;
    for (int k = 1; k < d; ++k) {
        c *= (b + (double)(k - 1)) / (double)k * r2;
        s_minus_1 += c;
    }
    double ln_s = log1p(s_minus_1);
    double nll = term0 - ln_s;
    return (nll > 0.0 && double_is_finite_bits(nll)) ? nll : 0.0;
}

static inline double nll_cybenko_scalar_d(double kappa, const cybenko_model_t *m, int d) {
    double s = cybenko_kappa_to_S(kappa, m);
    if (s <= 0.0 || !double_is_finite_bits(s)) return 0.0;
    double k_eff = m->k_eff;
    double th_eff = m->theta_eff;
    double lg_k = m->lgamma_k;

    if (d <= 2) {
        double z = s / th_eff;
        int order = 2 * d;
        double delta = k_eff - (double)order;
        double c = 1.0;
        double poly = 1.0;
        double d_poly = 0.0;
        for (int k = 1; k < order; ++k) {
            c *= (z / (double)k);
            poly += c;
            if (fabs(delta) > 1.0e-5) {
                double H_k = 0.0;
                for (int i = 1; i <= k; ++i) H_k += 1.0 / (double)i;
                d_poly += c * (-H_k);
            }
        }
        double nll = z - log(poly);
        if (fabs(delta) > 1.0e-5) {
            nll -= delta * (log(z) + d_poly / poly);
        }
        return (nll > 0.0 && double_is_finite_bits(nll)) ? nll : 0.0;
    } else {
        double a = k_eff;
        double z = s / th_eff;
        if (z < a + 3.0) {
            double c = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 60; ++k) {
                c *= z / (a + (double)(k - 1));
                sum += c;
                if (c < 1.0e-14 * sum) break;
            }
            double lg_inc = a * log(z) - z - log(a) + log(sum);
            double nll = lg_k - lg_inc;
            return (nll > 0.0 && double_is_finite_bits(nll)) ? nll : 0.0;
        } else {
            double inv_z = 1.0 / z;
            double t1 = (a - 1.0) * inv_z;
            double t2 = t1 * ((a - 2.0) * inv_z);
            double t3 = t2 * ((a - 3.0) * inv_z);
            double t4 = t3 * ((a - 4.0) * inv_z);
            double poly = 1.0 + t1 + t2 + t3 + t4;
            double nll = z - (a - 1.0) * log(z) + lg_k - log(poly);
            return (nll > 0.0 && double_is_finite_bits(nll)) ? nll : 0.0;
        }
    }
}

static inline double compute_bayes_laplace_d(double delta) {
    if (delta >= 0.0) {
        return delta + 0.693147180559945309417;  // ln(2)
    } else {
        double exp_delta = exp(delta);
        double inner = 1.0 - 0.5 * exp_delta;
        if (inner <= 0.0) return 0.0;
        return -log(inner);
    }
}

static inline OUT_FLOAT normalize_power_entry(FLOAT dot, FLOAT chi2_ref, FLOAT condition_bound, int normalization, int degree, int M,
                                              const cybenko_model_t *cyb, OUT_FLOAT b) {
    /* Non-finite explained chi^2 or R^2 >= 1 is a numerical artefact of an
     * unstable solve: report NaN so it cannot be mistaken for zero power.
     * Finite non-positive values (e.g. near-zero rounding) report 0. */
    const double dot_d = TO_DOUBLE(dot);
    const double ref_d = TO_DOUBLE(chi2_ref);
#    if defined(HALF)
    if (!f16_is_finite_bits(dot) || dot_d >= ref_d) return chi2_out_nan();
#    else
    if (!double_is_finite_bits(dot_d) || dot_d >= ref_d) return chi2_out_nan();
#    endif
    if (dot_d <= 0.0) return OUT_CAST(0.0);
    double r2 = dot_d / ref_d;
    if (!double_is_finite_bits(r2) || r2 >= 1.0) return chi2_out_nan();
    if (normalization == TLS_NORM_STANDARD) {
        return OUT_CAST(r2);
    } else if (normalization == TLS_NORM_ANOVA) {
        double d1 = 2.0 * (double)degree + 1.0;
        double d2 = (double)M - 1.0 - d1;
        double residual = ref_d - dot_d;
        if (residual < 1e-32) residual = 1e-32;
        return OUT_CAST((d2 * dot_d) / (d1 * residual));
    } else if (normalization == TLS_NORM_NLL) {
        return OUT_CAST(nll_exact_pochhammer_scalar_d(degree, r2, OUT_TO_DOUBLE(b)));
    } else { /* TLS_NORM_REGULARIZED */
        double nll_r2 = nll_exact_pochhammer_scalar_d(degree, r2, OUT_TO_DOUBLE(b));
        /* condition_bound is (2 * degree + 1) * P; NLL uses the product P. */
        double kappa = TO_DOUBLE(condition_bound) / (double)(2 * degree + 1);
        double nll_cond = nll_cybenko_scalar_d(kappa, cyb, degree);
        double delta = nll_r2 - nll_cond;
        return OUT_CAST(compute_bayes_laplace_d(delta));
    }
}
#else
static inline float nll_exact_pochhammer_scalar_f(int d, float r2, float b) {
    if (!float_is_finite_bits(r2) || r2 <= 0.0f) return 0.0f;
    if (r2 > 0.9999999f) r2 = 0.9999999f;
    float term0 = -b * log1pf(-r2);
    if (d <= 1) return (term0 > 0.0f && float_is_finite_bits(term0)) ? term0 : 0.0f;
    double c = 1.0;
    double s_minus_1 = 0.0;
    double b_d = (double)b;
    double r2_d = (double)r2;
    for (int k = 1; k < d; ++k) {
        c *= (b_d + (double)(k - 1)) / (double)k * r2_d;
        s_minus_1 += c;
    }
    double ln_s = log1p(s_minus_1);
    double nll = (double)term0 - ln_s;
    return (nll > 0.0 && double_is_finite_bits(nll)) ? (float)nll : 0.0f;
}

static inline float nll_cybenko_scalar_f(float kappa, const cybenko_model_t *m, int d) {
    double s = cybenko_kappa_to_S((double)kappa, m);
    if (s <= 0.0 || !double_is_finite_bits(s)) return 0.0f;
    float k_eff_f = (float)m->k_eff;
    float th_eff_f = (float)m->theta_eff;
    float lg_k_f = (float)m->lgamma_k;
    float s_f = (float)s;

    if (d <= 2) {
        float z = s_f / th_eff_f;
        int order = 2 * d;
        float delta = k_eff_f - (float)order;
        float c = 1.0f;
        float poly = 1.0f;
        float d_poly = 0.0f;
        for (int k = 1; k < order; ++k) {
            c *= (z / (float)k);
            poly += c;
            if (fabsf(delta) > 1.0e-5f) {
                float H_k = 0.0f;
                for (int i = 1; i <= k; ++i) H_k += 1.0f / (float)i;
                d_poly += c * (-H_k);
            }
        }
        float nll = z - logf(poly);
        if (fabsf(delta) > 1.0e-5f) {
            nll -= delta * (logf(z) + d_poly / poly);
        }
        return (nll > 0.0f && float_is_finite_bits(nll)) ? nll : 0.0f;
    } else {
        float a = k_eff_f;
        float z = s_f / th_eff_f;
        if (z < a + 3.0f) {
            double c = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 60; ++k) {
                c *= (double)z / ((double)a + (double)(k - 1));
                sum += c;
                if (c < 1.0e-7 * sum) break;
            }
            double lg_inc = (double)a * log((double)z) - (double)z - log((double)a) + log(sum);
            double nll = (double)lg_k_f - lg_inc;
            return (nll > 0.0 && double_is_finite_bits(nll)) ? (float)nll : 0.0f;
        } else {
            float inv_z = 1.0f / z;
            float t1 = (a - 1.0f) * inv_z;
            float t2 = t1 * ((a - 2.0f) * inv_z);
            float t3 = t2 * ((a - 3.0f) * inv_z);
            float t4 = t3 * ((a - 4.0f) * inv_z);
            float poly = 1.0f + t1 + t2 + t3 + t4;
            float nll = z - (a - 1.0f) * logf(z) + lg_k_f - logf(poly);
            return (nll > 0.0f && float_is_finite_bits(nll)) ? nll : 0.0f;
        }
    }
}

static inline float compute_bayes_laplace_f(float delta) {
    if (delta >= 0.0f) {
        return delta + 0.6931471805599453f;  // ln(2)
    } else {
        float exp_delta = expf(delta);
        float inner = 1.0f - 0.5f * exp_delta;
        if (inner <= 0.0f) return 0.0f;
        return -logf(inner);
    }
}

static inline FLOAT normalize_power_entry(FLOAT dot, FLOAT chi2_ref, FLOAT condition_bound, int normalization, int degree, int M, const cybenko_model_t *cyb,
                                          FLOAT b) {
    /* Non-finite explained chi^2 or R^2 >= 1 is a numerical artefact of an
     * unstable solve: report NaN so it cannot be mistaken for zero power.
     * Finite non-positive values (e.g. near-zero rounding) report 0. */
    if (!float_is_finite_bits(dot) || dot >= chi2_ref) return chi2_nan();
    if (dot <= 0.0f) return FCAST(0.0);
    float r2 = dot / chi2_ref;
    if (!float_is_finite_bits(r2) || r2 >= 1.0f) return chi2_nan();
    if (normalization == TLS_NORM_STANDARD) {
        return r2;
    } else if (normalization == TLS_NORM_ANOVA) {
        float d1 = 2.0f * (float)degree + 1.0f;
        float d2 = (float)M - 1.0f - d1;
        float residual = chi2_ref - dot;
        if (residual < 1e-32f) residual = 1e-32f;
        return (d2 * dot) / (d1 * residual);
    } else if (normalization == TLS_NORM_NLL) {
        return nll_exact_pochhammer_scalar_f(degree, r2, b);
    } else { /* TLS_NORM_REGULARIZED */
        float nll_r2 = nll_exact_pochhammer_scalar_f(degree, r2, b);
        /* condition_bound is (2 * degree + 1) * P; NLL uses the product P. */
        float kappa = condition_bound / (float)(2 * degree + 1);
        float nll_cond = nll_cybenko_scalar_f(kappa, cyb, degree);
        float delta = nll_r2 - nll_cond;
        return compute_bayes_laplace_f(delta);
    }
}
#endif

static inline int real_term_harmonic(int term) { return term == 0 ? 0 : (term + 1) / 2; }

static inline int real_term_is_sin(int term) { return term > 0 && (term & 1); }

static inline FLOAT signed_float_value(FLOAT value, int sign) {
    if (sign < 0) return NEG(value);
    if (sign > 0) return value;
    return FCAST(0.0);
}

static inline FLOAT real_gram_value(const FLOAT *Sw, const FLOAT *Cw, int N, int idx, int row, int col) {
    int m = real_term_harmonic(col);
    int n = real_term_harmonic(row);
    int col_sin = real_term_is_sin(col);
    int row_sin = real_term_is_sin(row);
    int diff = abs(m - n);
    int sum = m + n;
    FLOAT half = FCAST(0.5);

    if (col_sin && row_sin) {
        return MUL(half, SUB(Cw[(size_t)diff * N + idx], Cw[(size_t)sum * N + idx]));
    }
    if (!col_sin && !row_sin) {
        return MUL(half, ADD(Cw[(size_t)diff * N + idx], Cw[(size_t)sum * N + idx]));
    }
    if (col_sin) {
        return MUL(half, ADD(signed_float_value(Sw[(size_t)diff * N + idx], m - n), Sw[(size_t)sum * N + idx]));
    }
    return MUL(half, ADD(signed_float_value(Sw[(size_t)diff * N + idx], n - m), Sw[(size_t)sum * N + idx]));
}

static inline FLOAT real_rhs_value(const FLOAT *Syw, const FLOAT *Cyw, int N, int idx, int term) {
    int h = real_term_harmonic(term);
    if (real_term_is_sin(term)) return Syw[(size_t)h * N + idx];
    return Cyw[(size_t)h * N + idx];
}

static int execute_nufft_block(void *plan, int backend, const FLOAT *src_r, const FLOAT *src_i, FLOAT *out_r, FLOAT *out_i, int freq_factor, int mode) {
    if (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21) {
        if (mode == TLS_NUFFT_OUT_SLOTS || mode == TLS_NUFFT_OUT_SLOTS_RAW)
            NUFFT_PSWF_EXEC_MODE((PSWF_PLAN_T *)plan, src_r, src_i, out_r, out_i, freq_factor, mode);
        else
            NUFFT_PSWF_EXEC((PSWF_PLAN_T *)plan, src_r, src_i, out_r, out_i, freq_factor);
    } else if (backend == CHI2PER_BACKEND_LRA) {
        (void)mode; /* LRA always returns frequency order */
        NUFFT_LRA_EXEC((const LRA_PLAN_T *)plan, src_r, src_i, out_r, out_i, freq_factor);
    } else {
        return CHI2PER_ERR_BACKEND;
    }
    return CHI2PER_OK;
}

static int execute_nufft_block_pair(void *plan, int backend, const FLOAT *src1_r, const FLOAT *src1_i, const FLOAT *src2_r, const FLOAT *src2_i, FLOAT *out1_r,
                                    FLOAT *out1_i, FLOAT *out2_r, FLOAT *out2_i, int freq_factor, int mode) {
    if (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21) {
        NUFFT_PSWF_EXEC_MODE_PAIR((PSWF_PLAN_T *)plan, src1_r, src1_i, src2_r, src2_i, out1_r, out1_i, out2_r, out2_i, freq_factor, mode);
    } else if (backend == CHI2PER_BACKEND_LRA) {
        (void)mode;
        NUFFT_LRA_EXEC((const LRA_PLAN_T *)plan, src1_r, src1_i, out1_r, out1_i, freq_factor);
        NUFFT_LRA_EXEC((const LRA_PLAN_T *)plan, src2_r, src2_i, out2_r, out2_i, freq_factor);
    } else {
        return CHI2PER_ERR_BACKEND;
    }
    return CHI2PER_OK;
}

static int execute_nufft_breadth(void *plan, int backend, const NUFFT_INPUT_T *x, const FLOAT *val, double f0, int block, int block_base, int freq_factor,
                                 int mode, BREADTH_VEC *out_r, BREADTH_VEC *out_i) {
    if (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21) {
        NUFFT_PSWF_EXEC_BREADTH((PSWF_PLAN_T *)plan, x, val, f0, block, block_base, freq_factor, mode, out_r, out_i);
    } else if (backend == CHI2PER_BACKEND_LRA) {
        (void)mode;
        NUFFT_LRA_EXEC_BREADTH((const LRA_PLAN_T *)plan, x, val, f0, block, block_base, freq_factor, out_r, out_i);
    } else {
        return CHI2PER_ERR_BACKEND;
    }
    return CHI2PER_OK;
}

static int execute_nufft_breadth_pair(void *plan, int backend, const NUFFT_INPUT_T *x, const FLOAT *val1, const FLOAT *val2, double f0, int block,
                                      int block_base, int freq_factor, int mode, BREADTH_VEC *out1_r, BREADTH_VEC *out1_i, BREADTH_VEC *out2_r,
                                      BREADTH_VEC *out2_i) {
    if (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21) {
        NUFFT_PSWF_EXEC_BREADTH_PAIR((PSWF_PLAN_T *)plan, x, val1, val2, f0, block, block_base, freq_factor, mode, out1_r, out1_i, out2_r, out2_i);
    } else if (backend == CHI2PER_BACKEND_LRA) {
        (void)mode;
        NUFFT_LRA_EXEC_BREADTH((const LRA_PLAN_T *)plan, x, val1, f0, block, block_base, freq_factor, out1_r, out1_i);
        NUFFT_LRA_EXEC_BREADTH((const LRA_PLAN_T *)plan, x, val2, f0, block, block_base, freq_factor, out2_r, out2_i);
    } else {
        return CHI2PER_ERR_BACKEND;
    }
    return CHI2PER_OK;
}

static void compute_source_phase(NUFFT_INPUT_T tm, double qf0, FLOAT *phase_r, FLOAT *phase_i) {
#if defined(DOUBLE_DOUBLE)
    FLOAT phase = MUL(tm, FCAST(qf0));
    *phase_r = M_COS2PI(phase);
    *phase_i = M_SIN2PI(phase);
#else
    double phase = (double)tm * qf0;
    *phase_r = FCAST(cos2pi(phase));
    *phase_i = FCAST(sin2pi(phase));
#endif
}

static void compute_twiddle_delta(int backend, NUFFT_INPUT_T tm, double qdf, double advance, FLOAT *delta_r, FLOAT *delta_i) {
    (void)backend;
#if defined(DOUBLE_DOUBLE)
    if (backend == CHI2PER_BACKEND_PSWF43) {
        FLOAT phase_delta = MUL(MUL(tm, FCAST(qdf)), FCAST(advance / 3.0));
        FLOAT c = M_COS2PI(phase_delta);
        FLOAT s = M_SIN2PI(phase_delta);
        nanofft_triple_angle(c, s, delta_r, delta_i);
        return;
    }
#elif defined(DOUBLE)
    if (backend == CHI2PER_BACKEND_PSWF43) {
        double phase_delta = tm * qdf * (advance / 3.0);
        double c = cos2pi(phase_delta);
        double s = sin2pi(phase_delta);
        nanofft_triple_angle(c, s, delta_r, delta_i);
        return;
    }
#endif

#if defined(DOUBLE_DOUBLE)
    FLOAT phase_delta = MUL(MUL(tm, FCAST(qdf)), FCAST(advance));
    *delta_r = M_COS2PI(phase_delta);
    *delta_i = M_SIN2PI(phase_delta);
#else
    double phase_delta = (double)tm * qdf * advance;
    *delta_r = FCAST(cos2pi(phase_delta));
    *delta_i = FCAST(sin2pi(phase_delta));
#endif
}

static void rotate_source_level(FLOAT *src_r, FLOAT *src_i, const FLOAT *delta_r, const FLOAT *delta_i, int M) {
    for (int m = 0; m < M; ++m) {
        FLOAT yr = src_r[m];
        FLOAT yi = src_i[m];
        FLOAT dr = delta_r[m];
        FLOAT di = delta_i[m];
        src_r[m] = SUB(MUL(yr, dr), MUL(yi, di));
        src_i[m] = ADD(MUL(yr, di), MUL(yi, dr));
    }
}

static void copy_twiddle_level_down(FLOAT *src_r_levels, FLOAT *src_i_levels, int M, int src_level) {
    for (int level = src_level; level > 0; --level) {
        size_t dst = (size_t)(level - 1) * (size_t)M;
        size_t src = (size_t)level * (size_t)M;
        memcpy(src_r_levels + dst, src_r_levels + src, (size_t)M * sizeof(FLOAT));
        memcpy(src_i_levels + dst, src_i_levels + src, (size_t)M * sizeof(FLOAT));
    }
}

/* =========================================================================
 * Block twiddle ladder and block-sized trig-sum rows
 *
 * One source state per harmonic and ladder level, advanced once per block, so
 * the frequency grid can be walked block-first: every NuFFT output of a block
 * lands in a block-sized row set and is solved immediately.  That is what lets
 * the work arrays scale with the NuFFT block instead of the whole grid.
 *
 * The ladder itself is the exact-phase one: level 0 moves by one block per
 * complex multiply with a precomputed delta, and level L is regenerated from an
 * exact double-precision phase every MAX_TWIDDLE_REUSE^L blocks
 * (tls_twiddle_ladder_*).  Only the ownership changed - the states are per
 * harmonic now - so the arithmetic is the same as before.
 * ========================================================================= */

typedef struct {
    int levels, M, block, max_factor, degree;
    FLOAT *srcw_r, *srcw_i; /* max_factor harmonics of the weighted source */
    FLOAT *srcy_r, *srcy_i; /* degree harmonics of the y-weighted source, or NULL */
    FLOAT *delta_r, *delta_i;
    int prepared; /* block index the state currently describes */
} block_ladder;

static size_t ladder_offset(const block_ladder *lad, int q, int level) { return (((size_t)(q - 1) * (size_t)lad->levels) + (size_t)level) * (size_t)lad->M; }

static void block_ladder_free(block_ladder *lad) {
    if (!lad) return;
    nanofft_aligned_free(lad->srcw_r);
    nanofft_aligned_free(lad->srcw_i);
    nanofft_aligned_free(lad->srcy_r);
    nanofft_aligned_free(lad->srcy_i);
    nanofft_aligned_free(lad->delta_r);
    nanofft_aligned_free(lad->delta_i);
    memset(lad, 0, sizeof(*lad));
}

static int block_ladder_init(block_ladder *lad, const NUFFT_INPUT_T *x, const FLOAT *weight, const FLOAT *weighted_y, int M, int max_factor, int degree,
                             int block, int levels, int backend, double f0, double df) {
    memset(lad, 0, sizeof(*lad));
    lad->levels = levels;
    lad->M = M;
    lad->block = block;
    lad->max_factor = max_factor;
    lad->degree = weighted_y ? degree : 0;

    const size_t count = (size_t)max_factor * (size_t)levels * (size_t)M;
    lad->srcw_r = (FLOAT *)checked_aligned_malloc(count, sizeof(FLOAT));
    lad->srcw_i = (FLOAT *)checked_aligned_malloc(count, sizeof(FLOAT));
    lad->delta_r = (FLOAT *)checked_aligned_malloc(count, sizeof(FLOAT));
    lad->delta_i = (FLOAT *)checked_aligned_malloc(count, sizeof(FLOAT));
    if (lad->degree > 0) {
        const size_t count_y = (size_t)lad->degree * (size_t)levels * (size_t)M;
        lad->srcy_r = (FLOAT *)checked_aligned_malloc(count_y, sizeof(FLOAT));
        lad->srcy_i = (FLOAT *)checked_aligned_malloc(count_y, sizeof(FLOAT));
    }
    if (!lad->srcw_r || !lad->srcw_i || !lad->delta_r || !lad->delta_i || (lad->degree > 0 && (!lad->srcy_r || !lad->srcy_i))) {
        block_ladder_free(lad);
        return CHI2PER_ERR_ALLOC;
    }

    for (int q = 1; q <= max_factor; ++q) {
        const FLOAT qf0 = FCAST((double)q * f0);
        const double qdf = (double)q * df;
        for (int m = 0; m < M; ++m) {
            const NUFFT_INPUT_T xm = x[m];
            FLOAT c0;
            FLOAT s0;
            compute_source_phase(xm, TO_DOUBLE(qf0), &c0, &s0);
            const FLOAT srcw_r0 = MUL(weight[m], c0);
            const FLOAT srcw_i0 = MUL(weight[m], s0);
            const bool has_y = (q <= lad->degree);
            const FLOAT srcy_r0 = has_y ? MUL(weighted_y[m], c0) : FCAST(0.0);
            const FLOAT srcy_i0 = has_y ? MUL(weighted_y[m], s0) : FCAST(0.0);

            for (int level = 0; level < levels; ++level) {
                const size_t idx = ladder_offset(lad, q, level) + (size_t)m;
                lad->srcw_r[idx] = srcw_r0;
                lad->srcw_i[idx] = srcw_i0;
                if (has_y) {
                    lad->srcy_r[idx] = srcy_r0;
                    lad->srcy_i[idx] = srcy_i0;
                }
                const double advance = tls_twiddle_ladder_advance(block, level);
                compute_twiddle_delta(backend, xm, qdf, advance, &lad->delta_r[idx], &lad->delta_i[idx]);
            }
        }
    }
    lad->prepared = 0;
    return CHI2PER_OK;
}

/* Advance every harmonic state to `block_idx` (one carry rotate plus the level
 * copy-down, exactly as the per-pass loop used to do). */
static void block_ladder_advance(block_ladder *lad, int block_idx) {
    if (block_idx <= lad->prepared) return;
    const int level = tls_twiddle_ladder_carry_level((size_t)block_idx, lad->levels);
    for (int q = 1; q <= lad->max_factor; ++q) {
        const size_t base = ladder_offset(lad, q, level);
        rotate_source_level(lad->srcw_r + base, lad->srcw_i + base, lad->delta_r + base, lad->delta_i + base, lad->M);
        copy_twiddle_level_down(lad->srcw_r + ladder_offset(lad, q, 0), lad->srcw_i + ladder_offset(lad, q, 0), lad->M, level);
        if (q <= lad->degree && lad->srcy_r) {
            rotate_source_level(lad->srcy_r + base, lad->srcy_i + base, lad->delta_r + base, lad->delta_i + base, lad->M);
            copy_twiddle_level_down(lad->srcy_r + ladder_offset(lad, q, 0), lad->srcy_i + ladder_offset(lad, q, 0), lad->M, level);
        }
    }
    lad->prepared = block_idx;
}

typedef struct {
    int stride, max_factor, degree;
    FLOAT *Sw, *Cw, *Syw, *Cyw;
} trig_rows;

static void trig_rows_free(trig_rows *rows) {
    if (!rows) return;
    nanofft_aligned_free(rows->Sw);
    nanofft_aligned_free(rows->Cw);
    nanofft_aligned_free(rows->Syw);
    nanofft_aligned_free(rows->Cyw);
    memset(rows, 0, sizeof(*rows));
}

static int trig_rows_init(trig_rows *rows, int block, int max_factor, int degree, int align) {
    memset(rows, 0, sizeof(*rows));
    int stride = block;
    if (align > 1) stride = ((block + align - 1) / align) * align;
    rows->stride = stride;
    rows->max_factor = max_factor;
    rows->degree = degree;
    rows->Sw = (FLOAT *)checked_aligned_malloc((size_t)(max_factor + 1) * (size_t)stride, sizeof(FLOAT));
    rows->Cw = (FLOAT *)checked_aligned_malloc((size_t)(max_factor + 1) * (size_t)stride, sizeof(FLOAT));
    rows->Syw = (FLOAT *)checked_aligned_malloc((size_t)(degree + 1) * (size_t)stride, sizeof(FLOAT));
    rows->Cyw = (FLOAT *)checked_aligned_malloc((size_t)(degree + 1) * (size_t)stride, sizeof(FLOAT));
    if (!rows->Sw || !rows->Cw || !rows->Syw || !rows->Cyw) {
        trig_rows_free(rows);
        return CHI2PER_ERR_ALLOC;
    }
    return CHI2PER_OK;
}

/* Shrink the rows to `count` columns for the trailing partial block, so the
 * solver can index them with a single stride.  Always copies downwards. */
static void trig_rows_compact(trig_rows *rows, int count) {
    if (count == rows->stride) return;
    const size_t bytes = (size_t)count * sizeof(FLOAT);
    for (int q = 0; q <= rows->max_factor; ++q) {
        memmove(rows->Sw + (size_t)q * (size_t)count, rows->Sw + (size_t)q * (size_t)rows->stride, bytes);
        memmove(rows->Cw + (size_t)q * (size_t)count, rows->Cw + (size_t)q * (size_t)rows->stride, bytes);
    }
    for (int q = 0; q <= rows->degree; ++q) {
        memmove(rows->Syw + (size_t)q * (size_t)count, rows->Syw + (size_t)q * (size_t)rows->stride, bytes);
        memmove(rows->Cyw + (size_t)q * (size_t)count, rows->Cyw + (size_t)q * (size_t)rows->stride, bytes);
    }
    rows->stride = count;
}

typedef struct {
    int block;
    int max_factor;
    int degree;
    BREADTH_VEC *Sw, *Cw, *Syw, *Cyw;
} trig_rows_breadth;

static void trig_rows_breadth_free(trig_rows_breadth *rows) {
    if (!rows) return;
    nanofft_aligned_free(rows->Sw);
    nanofft_aligned_free(rows->Cw);
    nanofft_aligned_free(rows->Syw);
    nanofft_aligned_free(rows->Cyw);
    memset(rows, 0, sizeof(*rows));
}

static int trig_rows_breadth_init(trig_rows_breadth *rows, int block, int max_factor, int degree) {
    memset(rows, 0, sizeof(*rows));
    rows->block = block;
    rows->max_factor = max_factor;
    rows->degree = degree;
    rows->Sw = (BREADTH_VEC *)checked_aligned_malloc((size_t)(max_factor + 1) * (size_t)block, sizeof(BREADTH_VEC));
    rows->Cw = (BREADTH_VEC *)checked_aligned_malloc((size_t)(max_factor + 1) * (size_t)block, sizeof(BREADTH_VEC));
    rows->Syw = (BREADTH_VEC *)checked_aligned_malloc((size_t)(degree + 1) * (size_t)block, sizeof(BREADTH_VEC));
    rows->Cyw = (BREADTH_VEC *)checked_aligned_malloc((size_t)(degree + 1) * (size_t)block, sizeof(BREADTH_VEC));
    if (!rows->Sw || !rows->Cw || !rows->Syw || !rows->Cyw) {
        trig_rows_breadth_free(rows);
        return CHI2PER_ERR_ALLOC;
    }
    return CHI2PER_OK;
}

/* Erlang logarithmic correction applied after the per-harmonic Wilkie terms. */
static inline OUT_FLOAT ihs_correct_power(OUT_FLOAT sum, int degree) {
#if defined(DOUBLE_DOUBLE)
    if (TO_DOUBLE(sum) <= 0.0) return dd_make(0.0, 0.0);

    dd_t log_sum = dd_make(1.0, 0.0);
    dd_t term = dd_make(1.0, 0.0);
    for (int i = 1; i < degree; ++i) {
        term = dd_mul(term, dd_div(sum, dd_make((double)i, 0.0)));
        log_sum = dd_add(log_sum, term);
    }
    if (!(log_sum.hi > 0.0) || !isfinite(log_sum.hi)) return dd_make(0.0, 0.0);

    dd_t nll = dd_sub(sum, dd_log(log_sum));
    return (nll.hi > 0.0 && isfinite(nll.hi)) ? nll : dd_make(0.0, 0.0);
#else
    double s = OUT_TO_DOUBLE(sum);
    if (!(s > 0.0) || !isfinite(s)) return OUT_CAST(0.0);

    double log_sum = 1.0;
    double term = 1.0;
    for (int i = 1; i < degree; ++i) {
        term *= s / (double)i;
        log_sum += term;
    }
    if (!(log_sum > 0.0) || !isfinite(log_sum)) return OUT_CAST(0.0);

    double nll = s - log(log_sum);
    return (nll > 0.0 && isfinite(nll)) ? OUT_CAST(nll) : OUT_CAST(0.0);
#endif
}

static inline OUT_FLOAT ihs_wilkie_correct_power(OUT_FLOAT p, double n_eff) {
#if defined(DOUBLE_DOUBLE)
    if (!(TO_DOUBLE(p) > 0.0) || !(n_eff > 0.0) || !isfinite(n_eff)) return p;

    dd_t z = p;
    dd_t n = dd_make(n_eff, 0.0);
    dd_t inv_n = dd_div(dd_make(1.0, 0.0), n);
    dd_t z2 = dd_mul(z, z);
    dd_t z3 = dd_mul(z2, z);
    dd_t z4 = dd_mul(z3, z);
    dd_t term1 = dd_mul(dd_sub(dd_mul(dd_make(2.0, 0.0), z), z2), dd_mul(dd_make(0.25, 0.0), inv_n));
    dd_t numerator =
        dd_add(dd_sub(dd_mul(dd_make(24.0, 0.0), z), dd_mul(dd_make(132.0, 0.0), z2)), dd_sub(dd_mul(dd_make(76.0, 0.0), z3), dd_mul(dd_make(9.0, 0.0), z4)));
    dd_t denominator = dd_mul(dd_make(288.0, 0.0), dd_mul(n, n));
    dd_t delta = dd_sub(term1, dd_div(numerator, denominator));
    dd_t one_plus = dd_add(dd_make(1.0, 0.0), delta);
    if (!(one_plus.hi > 0.0) || !isfinite(one_plus.hi)) return p;

    dd_t corrected = dd_sub(z, dd_log(one_plus));
    return (corrected.hi > 0.0 && isfinite(corrected.hi)) ? corrected : p;
#else
    if (!(n_eff > 0.0) || !isfinite(n_eff)) return p;
    double z = OUT_TO_DOUBLE(p);
    if (!(z > 0.0) || !isfinite(z)) return OUT_CAST(0.0);

    double inv_n = 1.0 / n_eff;
    double inv_n2 = inv_n * inv_n;
    double z2 = z * z;
    double z3 = z2 * z;
    double z4 = z3 * z;
    double delta = (2.0 * z - z2) * 0.25 * inv_n;
    delta -= (24.0 * z - 132.0 * z2 + 76.0 * z3 - 9.0 * z4) * inv_n2 / 288.0;
    if (!(delta > -1.0) || !isfinite(delta)) return p;

    double corrected = z - log1p(delta);
    return (corrected > 0.0 && isfinite(corrected)) ? OUT_CAST(corrected) : p;
#endif
}

/* Post-permutation / final per-block IHS normalization helper.
 * When `post_scale != NULL`, multiplies each bin by `post_scale[k]`
 * (pre-fused `base_scale * deconv[k]^2`) before Wilkie (d=1 NLL), Erlang
 * (IHS_DEFAULT / NLL), or the d=1 STANDARD [0, 1) sanity check. */
static inline void ihs_finalize_block(OUT_FLOAT *dst, int count, int degree, int normalization, OUT_FLOAT standard_scale, double n_eff,
                                      const OUT_FLOAT *post_scale) {
    if (normalization == TLS_NORM_STANDARD) {
        if (post_scale) {
            for (int k = 0; k < count; ++k) {
                OUT_FLOAT v = OUT_MUL(dst[k], post_scale[k]);
                if (degree == 1 && (!power_is_finite(v) || OUT_TO_DOUBLE(v) >= 1.0)) v = chi2_out_nan();
                dst[k] = v;
            }
        } else {
            for (int k = 0; k < count; ++k) {
                OUT_FLOAT v = OUT_MUL(dst[k], standard_scale);
                if (degree == 1 && (!power_is_finite(v) || OUT_TO_DOUBLE(v) >= 1.0)) v = chi2_out_nan();
                dst[k] = v;
            }
        }
    } else if (normalization == TLS_NORM_NLL && degree == 1) {
        for (int k = 0; k < count; ++k) {
            OUT_FLOAT z = post_scale ? OUT_MUL(dst[k], post_scale[k]) : dst[k];
            if (post_scale) z = ihs_wilkie_correct_power(z, n_eff);
            OUT_FLOAT v = ihs_correct_power(z, 1);
            dst[k] = (OUT_TO_DOUBLE(v) >= 0.0) ? v : OUT_CAST(0.0);
        }
    } else {
        for (int k = 0; k < count; ++k) {
            OUT_FLOAT s = post_scale ? OUT_MUL(dst[k], post_scale[k]) : dst[k];
            OUT_FLOAT v = ihs_correct_power(s, degree);
            dst[k] = (OUT_TO_DOUBLE(v) >= 0.0) ? v : OUT_CAST(0.0);
        }
    }
}

/*
 * Depth-wise IHS sweep for all degrees d >= 1.
 *
 * When `defer != 0` on a PSWF backend, full blocks run in slot order
 * (even for d = 1, where one real bit-reversal of |Z|^2 replaces two
 * Nfft complex-component permutations).  Under TLS_IHS_DECONV_POST (default):
 *   - STANDARD, IHS_DEFAULT, and d = 1 NLL skip NuFFT deconvolution entirely,
 *     sum raw |Z_q|^2 in slot order, permute once, then multiply by
 *     `scale * deconv[k]^2` in natural order before Wilkie / Erlang.
 *   - d >= 2 NLL skips NuFFT (re, im) deconvolution via TLS_NUFFT_OUT_SLOTS_RAW
 *     and multiplies |Z_q|^2 by precomputed `nll_scale * slot_deconv[k]^2`
 *     before the per-harmonic nonlinear Wilkie correction, then permutes once.
 */
static int ihs_power_sweep_depth(block_ladder *ladder, int N, int block, int degree, void *plan, int backend, OUT_FLOAT *power, FLOAT *out_r, FLOAT *out_i,
                                 int defer, int normalization, OUT_FLOAT standard_scale, OUT_FLOAT nll_scale, double n_eff) {
    const size_t num_blocks = ((size_t)N + (size_t)block - 1) / (size_t)block;
    const bool use_pswf = (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21);
    const bool want_post = (defer && use_pswf && tls_tuning_ihs_deconv() == TLS_IHS_DECONV_POST);
    const bool nll_multi_post = (want_post && normalization == TLS_NORM_NLL && degree > 1);
    const bool post_after_perm = (want_post && !nll_multi_post);

    OUT_FLOAT *acc_block = defer ? (OUT_FLOAT *)checked_aligned_malloc((size_t)block, sizeof(OUT_FLOAT)) : NULL;
    OUT_FLOAT *post_scale = post_after_perm ? (OUT_FLOAT *)checked_aligned_malloc((size_t)block, sizeof(OUT_FLOAT)) : NULL;
    OUT_FLOAT *slot_scale = nll_multi_post ? (OUT_FLOAT *)checked_aligned_malloc((size_t)block, sizeof(OUT_FLOAT)) : NULL;
    if ((defer && !acc_block) || (post_after_perm && !post_scale) || (nll_multi_post && !slot_scale)) {
        nanofft_aligned_free(acc_block);
        nanofft_aligned_free(post_scale);
        nanofft_aligned_free(slot_scale);
        return CHI2PER_ERR_ALLOC;
    }

    if (post_after_perm) {
        const OUT_FLOAT *deconv = NUFFT_PSWF_IHS_DECONV((const PSWF_PLAN_T *)plan);
        const OUT_FLOAT base_scale = (normalization == TLS_NORM_STANDARD) ? standard_scale : nll_scale;
        for (int k = 0; k < block; ++k) {
            const OUT_FLOAT d = deconv[k];
            post_scale[k] = OUT_MUL(base_scale, OUT_MUL(d, d));
        }
    } else if (nll_multi_post) {
        const OUT_FLOAT *sdec = NUFFT_PSWF_IHS_SLOT_DECONV((PSWF_PLAN_T *)plan);
        for (int k = 0; k < block; ++k) {
            const OUT_FLOAT d = sdec[k];
            slot_scale[k] = OUT_MUL(nll_scale, OUT_MUL(d, d));
        }
    }

    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int base = (int)(block_idx * (size_t)block);
        const int count = (base + block <= N) ? block : (N - base);
        const int slot_block = (defer && count == block) ? 1 : 0;
        const bool block_post_perm = (slot_block && post_after_perm);
        const bool block_nll_multi = (slot_block && nll_multi_post);
        const int mode = (block_post_perm || block_nll_multi) ? TLS_NUFFT_OUT_SLOTS_RAW : (slot_block ? TLS_NUFFT_OUT_SLOTS : TLS_NUFFT_OUT_NATURAL);

        block_ladder_advance(ladder, (int)block_idx);
        OUT_FLOAT *acc = slot_block ? acc_block : (power + base);
        for (int k = 0; k < count; ++k) acc[k] = OUT_CAST(0.0);

        for (int q = 1; q <= degree; ++q) {
            const size_t src_off = (size_t)(q - 1) * (size_t)ladder->levels * (size_t)ladder->M;
            const int status = execute_nufft_block(plan, backend, ladder->srcw_r + src_off, ladder->srcw_i + src_off, out_r, out_i, q, mode);
            if (status != CHI2PER_OK) {
                nanofft_aligned_free(acc_block);
                nanofft_aligned_free(post_scale);
                nanofft_aligned_free(slot_scale);
                return status;
            }
            if (block_post_perm || normalization == TLS_NORM_STANDARD) {
                for (int k = 0; k < count; ++k) {
#if defined(HALF)
                    float re = tls_f16_to_f32(out_r[k]);
                    float im = tls_f16_to_f32(out_i[k]);
                    acc[k] += re * re + im * im;
#else
                    FLOAT re = out_r[k];
                    FLOAT im = out_i[k];
                    acc[k] = ADD(acc[k], ADD(MUL(re, re), MUL(im, im)));
#endif
                }
            } else if (block_nll_multi) {
                for (int k = 0; k < count; ++k) {
#if defined(HALF)
                    float re = tls_f16_to_f32(out_r[k]);
                    float im = tls_f16_to_f32(out_i[k]);
                    float z = (re * re + im * im) * slot_scale[k];
                    acc[k] += ihs_wilkie_correct_power(z, n_eff);
#else
                    FLOAT re = out_r[k];
                    FLOAT im = out_i[k];
                    FLOAT z = MUL(ADD(MUL(re, re), MUL(im, im)), slot_scale[k]);
                    acc[k] = ADD(acc[k], ihs_wilkie_correct_power(z, n_eff));
#endif
                }
            } else {
                for (int k = 0; k < count; ++k) {
#if defined(HALF)
                    float re = tls_f16_to_f32(out_r[k]);
                    float im = tls_f16_to_f32(out_i[k]);
                    float z = (re * re + im * im) * nll_scale;
                    if (normalization == TLS_NORM_NLL) z = ihs_wilkie_correct_power(z, n_eff);
                    acc[k] += z;
#else
                    FLOAT re = out_r[k];
                    FLOAT im = out_i[k];
                    FLOAT z = MUL(ADD(MUL(re, re), MUL(im, im)), nll_scale);
                    if (normalization == TLS_NORM_NLL) z = ihs_wilkie_correct_power(z, n_eff);
                    acc[k] = ADD(acc[k], z);
#endif
                }
            }
        }

        if (slot_block) NUFFT_PSWF_FINISH_SLOTS_OUT((PSWF_PLAN_T *)plan, acc_block, power + base, count);
        ihs_finalize_block(power + base, count, degree, normalization, standard_scale, n_eff, block_post_perm ? post_scale : NULL);
    }

    nanofft_aligned_free(acc_block);
    nanofft_aligned_free(post_scale);
    nanofft_aligned_free(slot_scale);
    return CHI2PER_OK;
}

static int ihs_power_sweep_breadth(const NUFFT_INPUT_T *x, const FLOAT *wy, double f0, int N, int block, int degree, void *plan, int backend, OUT_FLOAT *power,
                                   int defer, int normalization, OUT_FLOAT standard_scale, OUT_FLOAT nll_scale, double n_eff) {
    const size_t num_blocks = ((size_t)N + (size_t)block - 1u) / (size_t)block;
    const bool use_pswf = (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21);
    const int slot_block = (defer && use_pswf) ? 1 : 0;
    const bool want_post = (slot_block && tls_tuning_ihs_deconv() == TLS_IHS_DECONV_POST);
    const bool nll_multi_post = (want_post && normalization == TLS_NORM_NLL && degree > 1);
    const bool post_after_perm = (want_post && !nll_multi_post);
    const int mode = want_post ? TLS_NUFFT_OUT_SLOTS_RAW : (slot_block ? TLS_NUFFT_OUT_SLOTS : TLS_NUFFT_OUT_NATURAL);

    BREADTH_VEC *out_r = (BREADTH_VEC *)checked_aligned_malloc((size_t)block, sizeof(BREADTH_VEC));
    BREADTH_VEC *out_i = (BREADTH_VEC *)checked_aligned_malloc((size_t)block, sizeof(BREADTH_VEC));
    OUT_BVEC *acc_b = (OUT_BVEC *)checked_aligned_malloc((size_t)block, sizeof(OUT_BVEC));
    OUT_FLOAT *post_scale = post_after_perm ? (OUT_FLOAT *)checked_aligned_malloc((size_t)block, sizeof(OUT_FLOAT)) : NULL;
    OUT_FLOAT *slot_scale = nll_multi_post ? (OUT_FLOAT *)checked_aligned_malloc((size_t)block, sizeof(OUT_FLOAT)) : NULL;
    if (!out_r || !out_i || !acc_b || (post_after_perm && !post_scale) || (nll_multi_post && !slot_scale)) {
        nanofft_aligned_free(out_r);
        nanofft_aligned_free(out_i);
        nanofft_aligned_free(acc_b);
        nanofft_aligned_free(post_scale);
        nanofft_aligned_free(slot_scale);
        return CHI2PER_ERR_ALLOC;
    }

    if (post_after_perm) {
        const OUT_FLOAT *deconv = NUFFT_PSWF_IHS_DECONV((const PSWF_PLAN_T *)plan);
        const OUT_FLOAT base_scale = (normalization == TLS_NORM_STANDARD) ? standard_scale : nll_scale;
        for (int k = 0; k < block; ++k) {
            const OUT_FLOAT d = deconv[k];
            post_scale[k] = OUT_MUL(base_scale, OUT_MUL(d, d));
        }
    } else if (nll_multi_post) {
        const OUT_FLOAT *sdec = NUFFT_PSWF_IHS_SLOT_DECONV((PSWF_PLAN_T *)plan);
        for (int k = 0; k < block; ++k) {
            const OUT_FLOAT d = sdec[k];
            slot_scale[k] = OUT_MUL(nll_scale, OUT_MUL(d, d));
        }
    }

#if !defined(HALF)
    const BREADTH_COMPUTE_VEC v_nll_scale = BREADTH_SPLAT(nll_scale);
#endif
    int status = CHI2PER_OK;

    for (size_t block_base = 0; block_base < num_blocks; block_base += (size_t)BREADTH_VEC_LEN) {
        const size_t rem_blocks = num_blocks - block_base;
        const int active_lanes = (rem_blocks < (size_t)BREADTH_VEC_LEN) ? (int)rem_blocks : BREADTH_VEC_LEN;
        const int base = (int)(block_base * (size_t)block);
        const int total_remaining = N - base;

        memset(acc_b, 0, (size_t)block * sizeof(OUT_BVEC));

        for (int q = 1; q <= degree; ++q) {
            status = execute_nufft_breadth(plan, backend, x, wy, f0, block, (int)block_base, q, mode, out_r, out_i);
            if (status != CHI2PER_OK) goto done;

#if defined(HALF)
            if (post_after_perm || normalization == TLS_NORM_STANDARD) {
                for (int k = 0; k < block; ++k) {
                    for (int l = 0; l < active_lanes; ++l) {
                        float re = tls_f16_to_f32(BVEC_STORE_GET_LANE(out_r[k], l));
                        float im = tls_f16_to_f32(BVEC_STORE_GET_LANE(out_i[k], l));
                        acc_b[k].elems[l] += re * re + im * im;
                    }
                }
            } else if (nll_multi_post) {
                for (int k = 0; k < block; ++k) {
                    const float sk = slot_scale[k];
                    for (int l = 0; l < active_lanes; ++l) {
                        float re = tls_f16_to_f32(BVEC_STORE_GET_LANE(out_r[k], l));
                        float im = tls_f16_to_f32(BVEC_STORE_GET_LANE(out_i[k], l));
                        float z = ihs_wilkie_correct_power((re * re + im * im) * sk, n_eff);
                        acc_b[k].elems[l] += z;
                    }
                }
            } else if (normalization == TLS_NORM_NLL) {
                for (int k = 0; k < block; ++k) {
                    for (int l = 0; l < active_lanes; ++l) {
                        float re = tls_f16_to_f32(BVEC_STORE_GET_LANE(out_r[k], l));
                        float im = tls_f16_to_f32(BVEC_STORE_GET_LANE(out_i[k], l));
                        float z = ihs_wilkie_correct_power((re * re + im * im) * nll_scale, n_eff);
                        acc_b[k].elems[l] += z;
                    }
                }
            } else {
                for (int k = 0; k < block; ++k) {
                    for (int l = 0; l < active_lanes; ++l) {
                        float re = tls_f16_to_f32(BVEC_STORE_GET_LANE(out_r[k], l));
                        float im = tls_f16_to_f32(BVEC_STORE_GET_LANE(out_i[k], l));
                        acc_b[k].elems[l] += (re * re + im * im) * nll_scale;
                    }
                }
            }
#else
            if (post_after_perm || normalization == TLS_NORM_STANDARD) {
                for (int k = 0; k < block; ++k) {
                    const BREADTH_COMPUTE_VEC re = LOAD_BVEC(&out_r[k]);
                    const BREADTH_COMPUTE_VEC im = LOAD_BVEC(&out_i[k]);
                    const BREADTH_COMPUTE_VEC p2 = BREADTH_ADD(BREADTH_MUL(re, re), BREADTH_MUL(im, im));
                    STORE_BVEC(&acc_b[k], BREADTH_ADD(LOAD_BVEC(&acc_b[k]), p2));
                }
            } else if (nll_multi_post) {
                for (int k = 0; k < block; ++k) {
                    const BREADTH_COMPUTE_VEC re = LOAD_BVEC(&out_r[k]);
                    const BREADTH_COMPUTE_VEC im = LOAD_BVEC(&out_i[k]);
                    const BREADTH_COMPUTE_VEC zv = BREADTH_MUL_S(BREADTH_ADD(BREADTH_MUL(re, re), BREADTH_MUL(im, im)), slot_scale[k]);
                    BREADTH_COMPUTE_VEC acc = LOAD_BVEC(&acc_b[k]);
                    for (int l = 0; l < active_lanes; ++l) {
                        FLOAT z = ihs_wilkie_correct_power(BREADTH_GET_LANE(zv, l), n_eff);
                        BREADTH_SET_LANE(acc, l, ADD(BREADTH_GET_LANE(acc, l), z));
                    }
                    STORE_BVEC(&acc_b[k], acc);
                }
            } else if (normalization == TLS_NORM_NLL) {
                for (int k = 0; k < block; ++k) {
                    const BREADTH_COMPUTE_VEC re = LOAD_BVEC(&out_r[k]);
                    const BREADTH_COMPUTE_VEC im = LOAD_BVEC(&out_i[k]);
                    BREADTH_COMPUTE_VEC zv = BREADTH_MUL(BREADTH_ADD(BREADTH_MUL(re, re), BREADTH_MUL(im, im)), v_nll_scale);
                    BREADTH_COMPUTE_VEC acc = LOAD_BVEC(&acc_b[k]);
                    for (int l = 0; l < active_lanes; ++l) {
                        FLOAT z = ihs_wilkie_correct_power(BREADTH_GET_LANE(zv, l), n_eff);
                        BREADTH_SET_LANE(acc, l, ADD(BREADTH_GET_LANE(acc, l), z));
                    }
                    STORE_BVEC(&acc_b[k], acc);
                }
            } else {
                for (int k = 0; k < block; ++k) {
                    const BREADTH_COMPUTE_VEC re = LOAD_BVEC(&out_r[k]);
                    const BREADTH_COMPUTE_VEC im = LOAD_BVEC(&out_i[k]);
                    const BREADTH_COMPUTE_VEC zv = BREADTH_MUL(BREADTH_ADD(BREADTH_MUL(re, re), BREADTH_MUL(im, im)), v_nll_scale);
                    STORE_BVEC(&acc_b[k], BREADTH_ADD(LOAD_BVEC(&acc_b[k]), zv));
                }
            }
#endif
        }

        if (slot_block) {
            NUFFT_PSWF_FINISH_SLOTS_BREADTH_OUT((PSWF_PLAN_T *)plan, acc_b, power + base, block, active_lanes, total_remaining);
        } else {
            for (int v = 0; v < active_lanes; ++v) {
                const int rem = total_remaining - v * block;
                if (rem <= 0) break;
                const int count_v = (rem < block) ? rem : block;
                OUT_FLOAT *dst = power + base + (size_t)v * (size_t)block;
                for (int k = 0; k < count_v; ++k) dst[k] = OUT_BVEC_GET_LANE(acc_b[k], v);
            }
        }

        for (int v = 0; v < active_lanes; ++v) {
            const int rem = total_remaining - v * block;
            if (rem <= 0) break;
            const int count_v = (rem < block) ? rem : block;
            ihs_finalize_block(power + base + (size_t)v * (size_t)block, count_v, degree, normalization, standard_scale, n_eff,
                               post_after_perm ? post_scale : NULL);
        }
    }

done:
    nanofft_aligned_free(out_r);
    nanofft_aligned_free(out_i);
    nanofft_aligned_free(acc_b);
    nanofft_aligned_free(post_scale);
    nanofft_aligned_free(slot_scale);
    return status;
}

/* One Cybenko reflection-coefficient factor (1 + |k|) / (1 - |k|). */
static inline FLOAT reflection_condition_step(FLOAT abs2) {
    FLOAT abs_refl = M_SQRT(abs2);
    return DIV(ADD(FCAST(1.0), abs_refl), SUB(FCAST(1.0), abs_refl));
}

/* Keep |k|^2 strictly below one; NaN also maps to the finite maximum. */
static inline FLOAT clamp_reflection_abs2(FLOAT abs2) {
    double value = TO_DOUBLE(abs2);
#if defined(DOUBLE_DOUBLE) || defined(DOUBLE)
    const double limit = 0.999999999999999;
#elif defined(HALF)
    const double limit = 0.999;
#else
    const double limit = 0.9999999;
#endif
    if (!(value >= 0.0) || value > limit) return FCAST(limit);
    return abs2;
}

// Implementation of Generalized Scargle Periodogram using 3d NuFFTs instead of
// 4 used for generic AoV Solution by: Zechmeister and M. Kurster, A&A 496,
// 577-584 (2009) Trig identity: Press W.H. and Rybicki, G.B, "Fast algorithm
// for spectral analysis of unevenly sampled data". ApJ 1:338, p277, 1989
static int gls_impl(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, FLOAT ws, FLOAT yws, FLOAT chi2_ref, int normalization, int M,
                    const cybenko_model_t *cyb, OUT_FLOAT b, const FLOAT *dot_scale, OUT_FLOAT *power, OUT_FLOAT *cond) {
    FLOAT half = FCAST(0.5);
    FLOAT sqrt_half = M_SQRT(half);
    FLOAT inv_ws_base = DIV(FCAST(1.0), ws);
    /*
     * The Cybenko product is consumed only by the regularized normalization
     * and by callers that explicitly ask for the condition array.  Skip it
     * for standard/anova/nll on the common autonan=True path.
     */
    const bool need_cond = (normalization == TLS_NORM_REGULARIZED) || (cond != NULL);
    int nan_count = 0;

    for (int idx = 0; idx < N; ++idx) {
        FLOAT ws_k = dot_scale ? Cw[idx] : ws;
        FLOAT yws_k = dot_scale ? Cyw[idx] : yws;
        FLOAT inv_ws = dot_scale ? MUL(inv_ws_base, dot_scale[idx]) : inv_ws_base;
        FLOAT S = Sw[(size_t)N + idx];
        FLOAT C = Cw[(size_t)N + idx];
        FLOAT S2 = Sw[(size_t)2 * N + idx];
        FLOAT C2 = Cw[(size_t)2 * N + idx];
        FLOAT Sh = Syw[(size_t)N + idx];
        FLOAT Ch = Cyw[(size_t)N + idx];

        FLOAT num = SUB(S2, MUL(MUL(FCAST(2.0), S), MUL(C, inv_ws)));
        FLOAT den = SUB(C2, MUL(SUB(MUL(C, C), MUL(S, S)), inv_ws));
        FLOAT radius = M_SQRT(ADD(MUL(num, num), MUL(den, den)));

        FLOAT S2w = FCAST(0.0);
        FLOAT C2w = FCAST(1.0);
        if (TO_DOUBLE(radius) != 0.0) {
            FLOAT den_abs = den;
            FLOAT den_sign = FCAST(1.0);
            if (TO_DOUBLE(den) < 0.0) {
                den_abs = NEG(den);
                den_sign = FCAST(-1.0);
            }
            S2w = MUL(den_sign, DIV(num, radius));
            C2w = DIV(den_abs, radius);
        }

        FLOAT one_plus_c2 = ADD(FCAST(1.0), C2w);
        FLOAT one_minus_c2 = SUB(FCAST(1.0), C2w);
        if (TO_DOUBLE(one_plus_c2) < 0.0) one_plus_c2 = FCAST(0.0);
        if (TO_DOUBLE(one_minus_c2) < 0.0) one_minus_c2 = FCAST(0.0);

        FLOAT Ctau = MUL(sqrt_half, M_SQRT(one_plus_c2));
        FLOAT Stau = FCAST(0.0);
        if (TO_DOUBLE(S2w) > 0.0)
            Stau = MUL(sqrt_half, M_SQRT(one_minus_c2));
        else if (TO_DOUBLE(S2w) < 0.0)
            Stau = NEG(MUL(sqrt_half, M_SQRT(one_minus_c2)));

        FLOAT YC = ADD(MUL(Ch, Ctau), MUL(Sh, Stau));
        FLOAT YS = SUB(MUL(Sh, Ctau), MUL(Ch, Stau));
        FLOAT XC = ADD(MUL(C, Ctau), MUL(S, Stau));
        FLOAT XS = SUB(MUL(S, Ctau), MUL(C, Stau));

        FLOAT CC = MUL(half, ADD(ws_k, ADD(MUL(C2, C2w), MUL(S2, S2w))));
        FLOAT SS = MUL(half, SUB(ws_k, ADD(MUL(C2, C2w), MUL(S2, S2w))));
        CC = SUB(CC, MUL(MUL(XC, XC), inv_ws));
        SS = SUB(SS, MUL(MUL(XS, XS), inv_ws));
        YC = SUB(YC, MUL(MUL(yws_k, XC), inv_ws));
        YS = SUB(YS, MUL(MUL(yws_k, XS), inv_ws));

        FLOAT dot = ADD(DIV(MUL(YC, YC), CC), DIV(MUL(YS, YS), SS));
        if (dot_scale) dot = MUL(dot, dot_scale[idx]);
        FLOAT condition_bound = FCAST(0.0);
        if (need_cond) {
            /*
             * Cybenko product P for the 3x3 Toeplitz matrix
             * R0 = ws, R1 = C - iS, R2 = C2 - iS2:
             *   |k1| = |R1| / R0
             *   |k2| = |R0 R2 - R1^2| / (R0^2 - |R1|^2)
             * The condition value is n * P with n = 3.
             */
            FLOAT c = MUL(C, inv_ws);
            FLOAT s = MUL(S, inv_ws);
            FLOAT c2 = MUL(C2, inv_ws);
            FLOAT s2 = MUL(S2, inv_ws);
            FLOAT c2_plus_s2 = ADD(MUL(c, c), MUL(s, s));
            FLOAT k1_abs2 = clamp_reflection_abs2(c2_plus_s2);
            FLOAT k2_num_r = SUB(ADD(c2, MUL(s, s)), MUL(c, c));
            FLOAT k2_num_i = SUB(s2, MUL(FCAST(2.0), MUL(c, s)));
            FLOAT k2_den = SUB(FCAST(1.0), c2_plus_s2);
            FLOAT max_abs2 = clamp_reflection_abs2(FCAST(10.0));
            FLOAT k2_abs2 = max_abs2;
            if (TO_DOUBLE(k2_den) > 0.0) {
                k2_abs2 = clamp_reflection_abs2(DIV(ADD(MUL(k2_num_r, k2_num_r), MUL(k2_num_i, k2_num_i)), MUL(k2_den, k2_den)));
            }
            condition_bound = MUL(FCAST(3.0), MUL(reflection_condition_step(k1_abs2), reflection_condition_step(k2_abs2)));
            if (cond) cond[idx] = OUT_FROM_FLOAT(condition_bound);
        }
        power[idx] = normalize_power_entry(dot, chi2_ref, condition_bound, normalization, 1, M, cyb, b);
        if (!cond && power_is_nan(power[idx])) ++nan_count;
    }

    return cond ? CHI2PER_OK : nan_count;
}

static int gls_impl_breadth(const BREADTH_VEC *Sw, const BREADTH_VEC *Cw, const BREADTH_VEC *Syw, const BREADTH_VEC *Cyw, int block, int active_lanes, FLOAT ws,
                            FLOAT yws, FLOAT chi2_ref, int normalization, int M, const cybenko_model_t *cyb, OUT_FLOAT b, const FLOAT *dot_scale,
                            OUT_BVEC *power_b, OUT_BVEC *cond_b) {
    FLOAT half = FCAST(0.5);
    FLOAT sqrt_half = M_SQRT(half);
    FLOAT inv_ws_base = DIV(FCAST(1.0), ws);
    const bool need_cond = (normalization == TLS_NORM_REGULARIZED) || (cond_b != NULL);

    for (int k = 0; k < block; ++k) {
        const FLOAT dscale_k = dot_scale ? dot_scale[k] : FCAST(1.0);
        const FLOAT ws_k = dot_scale ? BREADTH_GET_LANE(LOAD_BVEC(&Cw[k]), 0) : ws;
        const FLOAT yws_k = dot_scale ? BREADTH_GET_LANE(LOAD_BVEC(&Cyw[k]), 0) : yws;
        const FLOAT inv_ws = dot_scale ? MUL(inv_ws_base, dscale_k) : inv_ws_base;
        const BREADTH_COMPUTE_VEC vS = LOAD_BVEC(&Sw[(size_t)block + (size_t)k]);
        const BREADTH_COMPUTE_VEC vC = LOAD_BVEC(&Cw[(size_t)block + (size_t)k]);
        const BREADTH_COMPUTE_VEC vS2 = LOAD_BVEC(&Sw[(size_t)2 * (size_t)block + (size_t)k]);
        const BREADTH_COMPUTE_VEC vC2 = LOAD_BVEC(&Cw[(size_t)2 * (size_t)block + (size_t)k]);
        const BREADTH_COMPUTE_VEC vSh = LOAD_BVEC(&Syw[(size_t)block + (size_t)k]);
        const BREADTH_COMPUTE_VEC vCh = LOAD_BVEC(&Cyw[(size_t)block + (size_t)k]);
        OUT_BVEC p_out = OUT_BVEC_ZERO;
        OUT_BVEC c_out = OUT_BVEC_ZERO;

        for (int lane = 0; lane < active_lanes; ++lane) {
            FLOAT S = BREADTH_GET_LANE(vS, lane);
            FLOAT C = BREADTH_GET_LANE(vC, lane);
            FLOAT S2 = BREADTH_GET_LANE(vS2, lane);
            FLOAT C2 = BREADTH_GET_LANE(vC2, lane);
            FLOAT Sh = BREADTH_GET_LANE(vSh, lane);
            FLOAT Ch = BREADTH_GET_LANE(vCh, lane);
            FLOAT num = SUB(S2, MUL(MUL(FCAST(2.0), S), MUL(C, inv_ws)));
            FLOAT den = SUB(C2, MUL(SUB(MUL(C, C), MUL(S, S)), inv_ws));
            FLOAT radius = M_SQRT(ADD(MUL(num, num), MUL(den, den)));

            FLOAT S2w = FCAST(0.0);
            FLOAT C2w = FCAST(1.0);
            if (TO_DOUBLE(radius) != 0.0) {
                FLOAT den_abs = den;
                FLOAT den_sign = FCAST(1.0);
                if (TO_DOUBLE(den) < 0.0) {
                    den_abs = NEG(den);
                    den_sign = FCAST(-1.0);
                }
                S2w = MUL(den_sign, DIV(num, radius));
                C2w = DIV(den_abs, radius);
            }

            FLOAT one_plus_c2 = ADD(FCAST(1.0), C2w);
            FLOAT one_minus_c2 = SUB(FCAST(1.0), C2w);
            if (TO_DOUBLE(one_plus_c2) < 0.0) one_plus_c2 = FCAST(0.0);
            if (TO_DOUBLE(one_minus_c2) < 0.0) one_minus_c2 = FCAST(0.0);

            FLOAT Ctau = MUL(sqrt_half, M_SQRT(one_plus_c2));
            FLOAT Stau = FCAST(0.0);
            if (TO_DOUBLE(S2w) > 0.0)
                Stau = MUL(sqrt_half, M_SQRT(one_minus_c2));
            else if (TO_DOUBLE(S2w) < 0.0)
                Stau = NEG(MUL(sqrt_half, M_SQRT(one_minus_c2)));

            FLOAT YC = ADD(MUL(Ch, Ctau), MUL(Sh, Stau));
            FLOAT YS = SUB(MUL(Sh, Ctau), MUL(Ch, Stau));
            FLOAT XC = ADD(MUL(C, Ctau), MUL(S, Stau));
            FLOAT XS = SUB(MUL(S, Ctau), MUL(C, Stau));

            FLOAT CC = MUL(half, ADD(ws_k, ADD(MUL(C2, C2w), MUL(S2, S2w))));
            FLOAT SS = MUL(half, SUB(ws_k, ADD(MUL(C2, C2w), MUL(S2, S2w))));
            CC = SUB(CC, MUL(MUL(XC, XC), inv_ws));
            SS = SUB(SS, MUL(MUL(XS, XS), inv_ws));
            YC = SUB(YC, MUL(MUL(yws_k, XC), inv_ws));
            YS = SUB(YS, MUL(MUL(yws_k, XS), inv_ws));

            FLOAT dot = ADD(DIV(MUL(YC, YC), CC), DIV(MUL(YS, YS), SS));
            if (dot_scale) dot = MUL(dot, dscale_k);
            FLOAT condition_bound = FCAST(0.0);
            if (need_cond) {
                FLOAT c = MUL(C, inv_ws);
                FLOAT s = MUL(S, inv_ws);
                FLOAT c2 = MUL(C2, inv_ws);
                FLOAT s2 = MUL(S2, inv_ws);
                FLOAT c2_plus_s2 = ADD(MUL(c, c), MUL(s, s));
                FLOAT k1_abs2 = clamp_reflection_abs2(c2_plus_s2);
                FLOAT k2_num_r = SUB(ADD(c2, MUL(s, s)), MUL(c, c));
                FLOAT k2_num_i = SUB(s2, MUL(FCAST(2.0), MUL(c, s)));
                FLOAT k2_den = SUB(FCAST(1.0), c2_plus_s2);
                FLOAT max_abs2 = clamp_reflection_abs2(FCAST(10.0));
                FLOAT k2_abs2 = max_abs2;
                if (TO_DOUBLE(k2_den) > 0.0) {
                    k2_abs2 = clamp_reflection_abs2(DIV(ADD(MUL(k2_num_r, k2_num_r), MUL(k2_num_i, k2_num_i)), MUL(k2_den, k2_den)));
                }
                condition_bound = MUL(FCAST(3.0), MUL(reflection_condition_step(k1_abs2), reflection_condition_step(k2_abs2)));
                if (cond_b) OUT_BVEC_SET_LANE(c_out, lane, OUT_FROM_FLOAT(condition_bound));
            }
            OUT_BVEC_SET_LANE(p_out, lane, normalize_power_entry(dot, chi2_ref, condition_bound, normalization, 1, M, cyb, b));
        }
        OUT_BVEC_STORE(&power_b[k], p_out);
        if (cond_b) OUT_BVEC_STORE(&cond_b[k], c_out);
    }
    return CHI2PER_OK;
}

/* Reusable scratch for the Levinson/Zohar/Bareiss batch solvers across all
 * three precisions (VECF, VEC, VECDD via BREADTH_VEC).  The plan's block loop
 * solves one block at a time, so re-allocating this per block would put an
 * allocator call on the hot path; it is sized by the model order only. */
typedef struct {
    int norder;
    BREADTH_VEC *Rr, *Ri, *Yr, *Yi, *Xr, *Xi, *Ehr, *Ehi, *Ehpr, *Ehpi;
    BREADTH_VEC *BUr, *BUi, *BD, *Bur, *Bui, *Bvr, *Bvi, *Bwr, *Bwi;
} toep_workspace;

static void toep_workspace_free(toep_workspace *ws) {
    if (!ws) return;
    nanofft_aligned_free(ws->Rr);
    nanofft_aligned_free(ws->Ri);
    nanofft_aligned_free(ws->Yr);
    nanofft_aligned_free(ws->Yi);
    nanofft_aligned_free(ws->Xr);
    nanofft_aligned_free(ws->Xi);
    nanofft_aligned_free(ws->Ehr);
    nanofft_aligned_free(ws->Ehi);
    nanofft_aligned_free(ws->Ehpr);
    nanofft_aligned_free(ws->Ehpi);
    nanofft_aligned_free(ws->BUr);
    nanofft_aligned_free(ws->BUi);
    nanofft_aligned_free(ws->BD);
    nanofft_aligned_free(ws->Bur);
    nanofft_aligned_free(ws->Bui);
    nanofft_aligned_free(ws->Bvr);
    nanofft_aligned_free(ws->Bvi);
    nanofft_aligned_free(ws->Bwr);
    nanofft_aligned_free(ws->Bwi);
    memset(ws, 0, sizeof(*ws));
}

static int toep_workspace_init(toep_workspace *ws, int norder, int solver) {
    memset(ws, 0, sizeof(*ws));
    ws->norder = norder;
    ws->Rr = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder + 1, sizeof(BREADTH_VEC));
    ws->Ri = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder + 1, sizeof(BREADTH_VEC));
    ws->Yr = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    ws->Yi = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    ws->Xr = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    ws->Xi = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    ws->Ehr = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    ws->Ehi = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    ws->Ehpr = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    ws->Ehpi = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    if (!ws->Rr || !ws->Ri || !ws->Yr || !ws->Yi || !ws->Xr || !ws->Xi || !ws->Ehr || !ws->Ehi || !ws->Ehpr || !ws->Ehpi) {
        toep_workspace_free(ws);
        return CHI2PER_ERR_ALLOC;
    }
    if (solver == CHI2PER_SOLVER_BAREISS) {
        const size_t norder_sq = (size_t)norder * (size_t)norder;
        ws->BUr = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
        ws->BUi = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
        ws->BD = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
        ws->Bur = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
        ws->Bui = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
        ws->Bvr = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
        ws->Bvi = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
        ws->Bwr = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
        ws->Bwi = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
        if (!ws->BUr || !ws->BUi || !ws->BD || !ws->Bur || !ws->Bui || !ws->Bvr || !ws->Bvi || !ws->Bwr || !ws->Bwi) {
            toep_workspace_free(ws);
            return CHI2PER_ERR_ALLOC;
        }
    }
    return CHI2PER_OK;
}

static int solve_periodogram_ldlt_vec(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, FLOAT chi2_ref,
                                      int normalization, int M, const cybenko_model_t *cyb, OUT_FLOAT b, const FLOAT *dot_scale, OUT_FLOAT *power,
                                      OUT_FLOAT *cond) {
    if (normalization == TLS_NORM_REGULARIZED) return CHI2PER_ERR_SOLVER;
    const int norder = 2 * degree + 1;
    size_t norder_sq = (size_t)norder * (size_t)norder;

    BREADTH_VEC *XTX = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *XTy = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *X = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *L = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *D = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *Y = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *Z = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));

    if (!XTX || !XTy || !X || !L || !D || !Y || !Z) {
        nanofft_aligned_free(XTX);
        nanofft_aligned_free(XTy);
        nanofft_aligned_free(X);
        nanofft_aligned_free(L);
        nanofft_aligned_free(D);
        nanofft_aligned_free(Y);
        nanofft_aligned_free(Z);
        return CHI2PER_ERR_ALLOC;
    }

    int nan_count = 0;

    for (int base = 0; base < N; base += BREADTH_VEC_LEN) {
        for (int row = 0; row < norder; ++row) {
            for (int lane = 0; lane < BREADTH_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) idx = N - 1;
                BVEC_STORE_SET_LANE(XTy[row], lane, real_rhs_value(Syw, Cyw, N, idx, row));
            }

            for (int col = 0; col < norder; ++col) {
                for (int lane = 0; lane < BREADTH_VEC_LEN; ++lane) {
                    int idx = base + lane;
                    if (idx >= N) idx = N - 1;
                    BVEC_STORE_SET_LANE(XTX[(size_t)row * norder + col], lane, real_gram_value(Sw, Cw, N, idx, row, col));
                }
            }
        }

        BREADTH_VEC condition_bound = SOLVE_LDLT((size_t)norder, XTX, XTy, X, L, D, Y, Z);

        for (int lane = 0; lane < BREADTH_VEC_LEN; ++lane) {
            int idx = base + lane;
            if (idx >= N) continue;
            FLOAT cb = BVEC_STORE_GET_LANE(condition_bound, lane);
            if (cond) cond[idx] = OUT_FROM_FLOAT(cb);
            if (!cond && condition_bound_is_singular(cb, CHI2PER_SOLVER_LDLT)) {
                power[idx] = chi2_out_nan();
                ++nan_count;
                continue;
            }

            FLOAT dot = FCAST(0.0);
            for (int k = 0; k < norder; ++k) {
                dot = ADD(dot, MUL(BVEC_STORE_GET_LANE(XTy[k], lane), BVEC_STORE_GET_LANE(X[k], lane)));
            }
            if (dot_scale) dot = MUL(dot, dot_scale[idx]);
            power[idx] = normalize_power_entry(dot, chi2_ref, cb, normalization, degree, M, cyb, b);
            if (!cond && power_is_nan(power[idx])) ++nan_count;
        }
    }

    nanofft_aligned_free(XTX);
    nanofft_aligned_free(XTy);
    nanofft_aligned_free(X);
    nanofft_aligned_free(L);
    nanofft_aligned_free(D);
    nanofft_aligned_free(Y);
    nanofft_aligned_free(Z);
    return cond ? CHI2PER_OK : nan_count;
}

static int solve_periodogram_svd_vec(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, FLOAT chi2_ref, int normalization,
                                     int M, const cybenko_model_t *cyb, OUT_FLOAT b, const FLOAT *dot_scale, OUT_FLOAT *power, OUT_FLOAT *cond) {
    if (normalization == TLS_NORM_REGULARIZED) return CHI2PER_ERR_SOLVER;
    const int norder = 2 * degree + 1;
    size_t norder_sq = (size_t)norder * (size_t)norder;

    BREADTH_VEC *XTX = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *XTy = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *X = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *Q = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *S = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *D = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *Y = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));

    if (!XTX || !XTy || !X || !Q || !S || !D || !Y) {
        nanofft_aligned_free(XTX);
        nanofft_aligned_free(XTy);
        nanofft_aligned_free(X);
        nanofft_aligned_free(Q);
        nanofft_aligned_free(S);
        nanofft_aligned_free(D);
        nanofft_aligned_free(Y);
        return CHI2PER_ERR_ALLOC;
    }

    BREADTH_VEC max_cond;
    STORE_BVEC(&max_cond, BREADTH_SPLAT(cond ? FCAST(0.0) : FCAST(COND_SINGULARITY_THRESHOLD_SVD)));
    int nan_count = 0;

    for (int base = 0; base < N; base += BREADTH_VEC_LEN) {
        for (int row = 0; row < norder; ++row) {
            for (int lane = 0; lane < BREADTH_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) idx = N - 1;
                BVEC_STORE_SET_LANE(XTy[row], lane, real_rhs_value(Syw, Cyw, N, idx, row));
            }

            for (int col = 0; col < norder; ++col) {
                for (int lane = 0; lane < BREADTH_VEC_LEN; ++lane) {
                    int idx = base + lane;
                    if (idx >= N) idx = N - 1;
                    BVEC_STORE_SET_LANE(XTX[(size_t)row * norder + col], lane, real_gram_value(Sw, Cw, N, idx, row, col));
                }
            }
        }

        BREADTH_VEC condition_bound = SOLVE_SVD((size_t)norder, XTX, XTy, X, Q, S, D, Y, max_cond);

        for (int lane = 0; lane < BREADTH_VEC_LEN; ++lane) {
            int idx = base + lane;
            if (idx >= N) continue;
            FLOAT cb = BVEC_STORE_GET_LANE(condition_bound, lane);
            if (cond) cond[idx] = OUT_FROM_FLOAT(cb);
            if (!cond && condition_bound_is_invalid(cb)) {
                power[idx] = chi2_out_nan();
                ++nan_count;
                continue;
            }

            FLOAT dot = FCAST(0.0);
            for (int k = 0; k < norder; ++k) {
                dot = ADD(dot, MUL(BVEC_STORE_GET_LANE(XTy[k], lane), BVEC_STORE_GET_LANE(X[k], lane)));
            }
            if (dot_scale) dot = MUL(dot, dot_scale[idx]);
            power[idx] = normalize_power_entry(dot, chi2_ref, cb, normalization, degree, M, cyb, b);
            if (!cond && power_is_nan(power[idx])) ++nan_count;
        }
    }

    nanofft_aligned_free(XTX);
    nanofft_aligned_free(XTy);
    nanofft_aligned_free(X);
    nanofft_aligned_free(Q);
    nanofft_aligned_free(S);
    nanofft_aligned_free(D);
    nanofft_aligned_free(Y);
    return cond ? CHI2PER_OK : nan_count;
}

static int solve_periodogram_vec(const FLOAT *Sw, const FLOAT *Cw, const FLOAT *Syw, const FLOAT *Cyw, int N, int degree, int solver, FLOAT chi2_ref,
                                 int normalization, int M, const cybenko_model_t *cyb, OUT_FLOAT b, const FLOAT *dot_scale, OUT_FLOAT *power, OUT_FLOAT *cond,
                                 toep_workspace *ws) {
    if (solver == CHI2PER_SOLVER_LDLT)
        return solve_periodogram_ldlt_vec(Sw, Cw, Syw, Cyw, N, degree, chi2_ref, normalization, M, cyb, b, dot_scale, power, cond);
    if (solver == CHI2PER_SOLVER_SVD) return solve_periodogram_svd_vec(Sw, Cw, Syw, Cyw, N, degree, chi2_ref, normalization, M, cyb, b, dot_scale, power, cond);

    const int norder = 2 * degree + 1;
    if (!ws || ws->norder != norder || !ws->Rr) return CHI2PER_ERR_ALLOC;

    BREADTH_VEC *Rr = ws->Rr;
    BREADTH_VEC *Ri = ws->Ri;
    BREADTH_VEC *Yr = ws->Yr;
    BREADTH_VEC *Yi = ws->Yi;
    BREADTH_VEC *Xr = ws->Xr;
    BREADTH_VEC *Xi = ws->Xi;
    BREADTH_VEC *Ehr = ws->Ehr;
    BREADTH_VEC *Ehi = ws->Ehi;
    BREADTH_VEC *Ehpr = ws->Ehpr;
    BREADTH_VEC *Ehpi = ws->Ehpi;
    BREADTH_VEC *BUr = ws->BUr;
    BREADTH_VEC *BUi = ws->BUi;
    BREADTH_VEC *BD = ws->BD;
    BREADTH_VEC *Bur = ws->Bur;
    BREADTH_VEC *Bui = ws->Bui;
    BREADTH_VEC *Bvr = ws->Bvr;
    BREADTH_VEC *Bvi = ws->Bvi;
    BREADTH_VEC *Bwr = ws->Bwr;
    BREADTH_VEC *Bwi = ws->Bwi;
    if (solver == CHI2PER_SOLVER_BAREISS && (!BUr || !BUi || !BD || !Bur || !Bui || !Bvr || !Bvi || !Bwr || !Bwi)) return CHI2PER_ERR_ALLOC;

    int nan_count = 0;

    for (int base = 0; base < N; base += BREADTH_VEC_LEN) {
        for (int k = 0; k < norder; ++k) {
            int h = k - degree;
            for (int lane = 0; lane < BREADTH_VEC_LEN; ++lane) {
                int idx = base + lane;
                if (idx >= N) idx = N - 1;

                BVEC_STORE_SET_LANE(Rr[k], lane, Cw[(size_t)k * N + idx]);
                BVEC_STORE_SET_LANE(Ri[k], lane, NEG(Sw[(size_t)k * N + idx]));

                if (h == 0) {
                    BVEC_STORE_SET_LANE(Yr[k], lane, Cyw[idx]);
                    BVEC_STORE_SET_LANE(Yi[k], lane, FCAST(0.0));
                } else if (h > 0) {
                    BVEC_STORE_SET_LANE(Yr[k], lane, Cyw[(size_t)h * N + idx]);
                    BVEC_STORE_SET_LANE(Yi[k], lane, NEG(Syw[(size_t)h * N + idx]));
                } else {
                    int ah = -h;
                    BVEC_STORE_SET_LANE(Yr[k], lane, Cyw[(size_t)ah * N + idx]);
                    BVEC_STORE_SET_LANE(Yi[k], lane, Syw[(size_t)ah * N + idx]);
                }
            }
        }

        Rr[norder] = BVEC_STORE_ZERO;
        Ri[norder] = BVEC_STORE_ZERO;

        BREADTH_VEC condition_bound;
        if (solver == CHI2PER_SOLVER_LEVINSON) {
            condition_bound = SOLVE_LEVINSON((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ehr, Ehi, Ehpr, Ehpi);
        } else if (solver == CHI2PER_SOLVER_ZOHAR) {
            condition_bound = SOLVE_ZOHAR((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ehr, Ehi, Ehpr, Ehpi);
        } else if (solver == CHI2PER_SOLVER_BAREISS) {
            condition_bound = SOLVE_BAREISS((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, BUr, BUi, BD, Bur, Bui, Bvr, Bvi, Bwr, Bwi);
        } else {
            return CHI2PER_ERR_SOLVER;
        }

        for (int lane = 0; lane < BREADTH_VEC_LEN; ++lane) {
            int idx = base + lane;
            if (idx >= N) continue;
            FLOAT cb = BVEC_STORE_GET_LANE(condition_bound, lane);
            if (cond) cond[idx] = OUT_FROM_FLOAT(cb);
            if (!cond && condition_bound_is_singular(cb, solver)) {
                power[idx] = chi2_out_nan();
                ++nan_count;
                continue;
            }

            FLOAT dot = BVEC_STORE_GET_LANE(Xr[0], lane);
            if (dot_scale) dot = MUL(dot, dot_scale[idx]);
            power[idx] = normalize_power_entry(dot, chi2_ref, cb, normalization, degree, M, cyb, b);
            if (!cond && power_is_nan(power[idx])) ++nan_count;
        }
    }

    return cond ? CHI2PER_OK : nan_count;
}

static inline BREADTH_COMPUTE_VEC signed_vec_value(BREADTH_COMPUTE_VEC value, int sign) {
    if (sign < 0) return BREADTH_NEG(value);
    if (sign > 0) return value;
    return BREADTH_ZERO;
}

static inline BREADTH_COMPUTE_VEC real_gram_value_vec(const BREADTH_VEC *Sw, const BREADTH_VEC *Cw, int block, int idx, int row, int col) {
    const int m = real_term_harmonic(col);
    const int n = real_term_harmonic(row);
    const int col_sin = real_term_is_sin(col);
    const int row_sin = real_term_is_sin(row);
    const int diff = abs(m - n);
    const int sum = m + n;
    const BREADTH_COMPUTE_VEC half = BREADTH_SPLAT(FCAST(0.5));

    if (col_sin && row_sin)
        return BREADTH_MUL(half,
                           BREADTH_SUB(LOAD_BVEC(&Cw[(size_t)diff * (size_t)block + (size_t)idx]), LOAD_BVEC(&Cw[(size_t)sum * (size_t)block + (size_t)idx])));
    if (!col_sin && !row_sin)
        return BREADTH_MUL(half,
                           BREADTH_ADD(LOAD_BVEC(&Cw[(size_t)diff * (size_t)block + (size_t)idx]), LOAD_BVEC(&Cw[(size_t)sum * (size_t)block + (size_t)idx])));
    if (col_sin)
        return BREADTH_MUL(half, BREADTH_ADD(signed_vec_value(LOAD_BVEC(&Sw[(size_t)diff * (size_t)block + (size_t)idx]), m - n),
                                             LOAD_BVEC(&Sw[(size_t)sum * (size_t)block + (size_t)idx])));
    return BREADTH_MUL(half, BREADTH_ADD(signed_vec_value(LOAD_BVEC(&Sw[(size_t)diff * (size_t)block + (size_t)idx]), n - m),
                                         LOAD_BVEC(&Sw[(size_t)sum * (size_t)block + (size_t)idx])));
}

static inline BREADTH_VEC real_rhs_value_vec(const BREADTH_VEC *Syw, const BREADTH_VEC *Cyw, int block, int idx, int term) {
    const int h = real_term_harmonic(term);
    if (real_term_is_sin(term)) return Syw[(size_t)h * (size_t)block + (size_t)idx];
    return Cyw[(size_t)h * (size_t)block + (size_t)idx];
}

static int solve_periodogram_ldlt_breadth(const BREADTH_VEC *Sw, const BREADTH_VEC *Cw, const BREADTH_VEC *Syw, const BREADTH_VEC *Cyw, int block,
                                          int active_lanes, int degree, FLOAT chi2_ref, int normalization, int M, const cybenko_model_t *cyb, OUT_FLOAT b,
                                          const FLOAT *dot_scale, OUT_BVEC *power_b, OUT_BVEC *cond_b) {
    if (normalization == TLS_NORM_REGULARIZED) return CHI2PER_ERR_SOLVER;
    const int norder = 2 * degree + 1;
    const size_t norder_sq = (size_t)norder * (size_t)norder;

    BREADTH_VEC *XTX = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *XTy = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *X = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *L = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *D = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *Y = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *Z = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));

    if (!XTX || !XTy || !X || !L || !D || !Y || !Z) {
        nanofft_aligned_free(XTX);
        nanofft_aligned_free(XTy);
        nanofft_aligned_free(X);
        nanofft_aligned_free(L);
        nanofft_aligned_free(D);
        nanofft_aligned_free(Y);
        nanofft_aligned_free(Z);
        return CHI2PER_ERR_ALLOC;
    }

    for (int idx = 0; idx < block; ++idx) {
        for (int row = 0; row < norder; ++row) {
            XTy[row] = real_rhs_value_vec(Syw, Cyw, block, idx, row);
            for (int col = 0; col < norder; ++col)
                STORE_BVEC(&XTX[(size_t)row * (size_t)norder + (size_t)col], real_gram_value_vec(Sw, Cw, block, idx, row, col));
        }

        const BREADTH_VEC condition_bound = SOLVE_LDLT((size_t)norder, XTX, XTy, X, L, D, Y, Z);
        OUT_BVEC p_out = OUT_BVEC_ZERO;
        OUT_BVEC c_out = OUT_BVEC_ZERO;
        for (int lane = 0; lane < active_lanes; ++lane) {
            FLOAT cb = BVEC_STORE_GET_LANE(condition_bound, lane);
            if (cond_b) OUT_BVEC_SET_LANE(c_out, lane, OUT_FROM_FLOAT(cb));
            if (!cond_b && condition_bound_is_singular(cb, CHI2PER_SOLVER_LDLT)) {
                OUT_BVEC_SET_LANE(p_out, lane, chi2_out_nan());
                continue;
            }
            FLOAT dot = FCAST(0.0);
            for (int k = 0; k < norder; ++k) dot = ADD(dot, MUL(BVEC_STORE_GET_LANE(XTy[k], lane), BVEC_STORE_GET_LANE(X[k], lane)));
            if (dot_scale) dot = MUL(dot, dot_scale[idx]);
            OUT_BVEC_SET_LANE(p_out, lane, normalize_power_entry(dot, chi2_ref, cb, normalization, degree, M, cyb, b));
        }
        OUT_BVEC_STORE(&power_b[idx], p_out);
        if (cond_b) OUT_BVEC_STORE(&cond_b[idx], c_out);
    }

    nanofft_aligned_free(XTX);
    nanofft_aligned_free(XTy);
    nanofft_aligned_free(X);
    nanofft_aligned_free(L);
    nanofft_aligned_free(D);
    nanofft_aligned_free(Y);
    nanofft_aligned_free(Z);
    return CHI2PER_OK;
}

static int solve_periodogram_svd_breadth(const BREADTH_VEC *Sw, const BREADTH_VEC *Cw, const BREADTH_VEC *Syw, const BREADTH_VEC *Cyw, int block,
                                         int active_lanes, int degree, FLOAT chi2_ref, int normalization, int M, const cybenko_model_t *cyb, OUT_FLOAT b,
                                         const FLOAT *dot_scale, OUT_BVEC *power_b, OUT_BVEC *cond_b) {
    if (normalization == TLS_NORM_REGULARIZED) return CHI2PER_ERR_SOLVER;
    const int norder = 2 * degree + 1;
    const size_t norder_sq = (size_t)norder * (size_t)norder;

    BREADTH_VEC *XTX = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *XTy = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *X = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *Q = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *S = (BREADTH_VEC *)checked_aligned_malloc(norder_sq, sizeof(BREADTH_VEC));
    BREADTH_VEC *D = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));
    BREADTH_VEC *Y = (BREADTH_VEC *)checked_aligned_malloc((size_t)norder, sizeof(BREADTH_VEC));

    if (!XTX || !XTy || !X || !Q || !S || !D || !Y) {
        nanofft_aligned_free(XTX);
        nanofft_aligned_free(XTy);
        nanofft_aligned_free(X);
        nanofft_aligned_free(Q);
        nanofft_aligned_free(S);
        nanofft_aligned_free(D);
        nanofft_aligned_free(Y);
        return CHI2PER_ERR_ALLOC;
    }

    BREADTH_VEC max_cond;
    STORE_BVEC(&max_cond, BREADTH_SPLAT(cond_b ? FCAST(0.0) : FCAST(COND_SINGULARITY_THRESHOLD_SVD)));
    for (int idx = 0; idx < block; ++idx) {
        for (int row = 0; row < norder; ++row) {
            XTy[row] = real_rhs_value_vec(Syw, Cyw, block, idx, row);
            for (int col = 0; col < norder; ++col)
                STORE_BVEC(&XTX[(size_t)row * (size_t)norder + (size_t)col], real_gram_value_vec(Sw, Cw, block, idx, row, col));
        }

        const BREADTH_VEC condition_bound = SOLVE_SVD((size_t)norder, XTX, XTy, X, Q, S, D, Y, max_cond);
        OUT_BVEC p_out = OUT_BVEC_ZERO;
        OUT_BVEC c_out = OUT_BVEC_ZERO;
        for (int lane = 0; lane < active_lanes; ++lane) {
            FLOAT cb = BVEC_STORE_GET_LANE(condition_bound, lane);
            if (cond_b) OUT_BVEC_SET_LANE(c_out, lane, OUT_FROM_FLOAT(cb));
            if (!cond_b && condition_bound_is_invalid(cb)) {
                OUT_BVEC_SET_LANE(p_out, lane, chi2_out_nan());
                continue;
            }
            FLOAT dot = FCAST(0.0);
            for (int k = 0; k < norder; ++k) dot = ADD(dot, MUL(BVEC_STORE_GET_LANE(XTy[k], lane), BVEC_STORE_GET_LANE(X[k], lane)));
            if (dot_scale) dot = MUL(dot, dot_scale[idx]);
            OUT_BVEC_SET_LANE(p_out, lane, normalize_power_entry(dot, chi2_ref, cb, normalization, degree, M, cyb, b));
        }
        OUT_BVEC_STORE(&power_b[idx], p_out);
        if (cond_b) OUT_BVEC_STORE(&cond_b[idx], c_out);
    }

    nanofft_aligned_free(XTX);
    nanofft_aligned_free(XTy);
    nanofft_aligned_free(X);
    nanofft_aligned_free(Q);
    nanofft_aligned_free(S);
    nanofft_aligned_free(D);
    nanofft_aligned_free(Y);
    return CHI2PER_OK;
}

/* Zero-transpose breadth Toeplitz solver across all three precisions (VECF,
 * VEC, VECDD): every frequency bin feeds straight into the SIMD-batched solver
 * without lane gathering or transposing. */
static int solve_periodogram_breadth_vec(const BREADTH_VEC *Sw, const BREADTH_VEC *Cw, const BREADTH_VEC *Syw, const BREADTH_VEC *Cyw, int block,
                                         int active_lanes, int degree, int solver, FLOAT chi2_ref, int normalization, int M, const cybenko_model_t *cyb,
                                         OUT_FLOAT b, const FLOAT *dot_scale, OUT_BVEC *power_b, OUT_BVEC *cond_b, toep_workspace *ws) {
    if (solver == CHI2PER_SOLVER_LDLT)
        return solve_periodogram_ldlt_breadth(Sw, Cw, Syw, Cyw, block, active_lanes, degree, chi2_ref, normalization, M, cyb, b, dot_scale, power_b, cond_b);
    if (solver == CHI2PER_SOLVER_SVD)
        return solve_periodogram_svd_breadth(Sw, Cw, Syw, Cyw, block, active_lanes, degree, chi2_ref, normalization, M, cyb, b, dot_scale, power_b, cond_b);

    const int norder = 2 * degree + 1;
    if (!ws || ws->norder != norder || !ws->Rr) return CHI2PER_ERR_ALLOC;

    BREADTH_VEC *Rr = ws->Rr;
    BREADTH_VEC *Ri = ws->Ri;
    BREADTH_VEC *Yr = ws->Yr;
    BREADTH_VEC *Yi = ws->Yi;
    BREADTH_VEC *Xr = ws->Xr;
    BREADTH_VEC *Xi = ws->Xi;
    BREADTH_VEC *Ehr = ws->Ehr;
    BREADTH_VEC *Ehi = ws->Ehi;
    BREADTH_VEC *Ehpr = ws->Ehpr;
    BREADTH_VEC *Ehpi = ws->Ehpi;
    BREADTH_VEC *BUr = ws->BUr;
    BREADTH_VEC *BUi = ws->BUi;
    BREADTH_VEC *BD = ws->BD;
    BREADTH_VEC *Bur = ws->Bur;
    BREADTH_VEC *Bui = ws->Bui;
    BREADTH_VEC *Bvr = ws->Bvr;
    BREADTH_VEC *Bvi = ws->Bvi;
    BREADTH_VEC *Bwr = ws->Bwr;
    BREADTH_VEC *Bwi = ws->Bwi;
    if (solver == CHI2PER_SOLVER_BAREISS && (!BUr || !BUi || !BD || !Bur || !Bui || !Bvr || !Bvi || !Bwr || !Bwi)) return CHI2PER_ERR_ALLOC;

    const BREADTH_VEC s_zero = BVEC_STORE_ZERO;
    for (int idx = 0; idx < block; ++idx) {
        for (int k = 0; k < norder; ++k) {
            Rr[k] = Cw[(size_t)k * (size_t)block + (size_t)idx];
            STORE_BVEC(&Ri[k], BREADTH_NEG(LOAD_BVEC(&Sw[(size_t)k * (size_t)block + (size_t)idx])));
            const int h = k - degree;
            if (h == 0) {
                Yr[k] = Cyw[idx];
                Yi[k] = s_zero;
            } else if (h > 0) {
                Yr[k] = Cyw[(size_t)h * (size_t)block + (size_t)idx];
                STORE_BVEC(&Yi[k], BREADTH_NEG(LOAD_BVEC(&Syw[(size_t)h * (size_t)block + (size_t)idx])));
            } else {
                const int ah = -h;
                Yr[k] = Cyw[(size_t)ah * (size_t)block + (size_t)idx];
                Yi[k] = Syw[(size_t)ah * (size_t)block + (size_t)idx];
            }
        }
        Rr[norder] = s_zero;
        Ri[norder] = s_zero;

        BREADTH_VEC condition_bound;
        if (solver == CHI2PER_SOLVER_LEVINSON) {
            condition_bound = SOLVE_LEVINSON((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ehr, Ehi, Ehpr, Ehpi);
        } else if (solver == CHI2PER_SOLVER_ZOHAR) {
            condition_bound = SOLVE_ZOHAR((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, Ehr, Ehi, Ehpr, Ehpi);
        } else if (solver == CHI2PER_SOLVER_BAREISS) {
            condition_bound = SOLVE_BAREISS((size_t)norder, Rr, Ri, Yr, Yi, Xr, Xi, BUr, BUi, BD, Bur, Bui, Bvr, Bvi, Bwr, Bwi);
        } else {
            return CHI2PER_ERR_SOLVER;
        }

        OUT_BVEC p_out = OUT_BVEC_ZERO;
        OUT_BVEC c_out = OUT_BVEC_ZERO;
        for (int lane = 0; lane < active_lanes; ++lane) {
            FLOAT cb = BVEC_STORE_GET_LANE(condition_bound, lane);
            if (cond_b) OUT_BVEC_SET_LANE(c_out, lane, OUT_FROM_FLOAT(cb));
            if (!cond_b && condition_bound_is_singular(cb, solver)) {
                OUT_BVEC_SET_LANE(p_out, lane, chi2_out_nan());
                continue;
            }
            FLOAT dot = BVEC_STORE_GET_LANE(Xr[0], lane);
            if (dot_scale) dot = MUL(dot, dot_scale[idx]);
            OUT_BVEC_SET_LANE(p_out, lane, normalize_power_entry(dot, chi2_ref, cb, normalization, degree, M, cyb, b));
        }
        OUT_BVEC_STORE(&power_b[idx], p_out);
        if (cond_b) OUT_BVEC_STORE(&cond_b[idx], c_out);
    }
    return CHI2PER_OK;
}

#if defined(HALF)
static int fastchi2_core(const double *t, const float *y_f, const double *y_d, const float *dy_f, const double *dy_d, int M, double f0, double df, int N,
                         int degree, int backend, int solver, int normalization, int spreading, int vectorize, float *power, float *cond) {
    if (!t || (!y_f && !y_d) || (!dy_f && !dy_d) || !power || M <= 0 || N <= 0 || degree <= 0 || f0 < 0.0 || df <= 0.0) return CHI2PER_ERR_ARGUMENT;
#else
int CHI2_PREFIX(fastchi2_opts)(const TIME_INPUT_T *t, const FLOAT *y, const FLOAT *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                               int normalization, int spreading, int vectorize, OUT_FLOAT *power, OUT_FLOAT *cond) {
    if (!t || !y || !dy || !power || M <= 0 || N <= 0 || degree <= 0 || f0 < 0.0 || df <= 0.0) return CHI2PER_ERR_ARGUMENT;
#endif
    if (normalization < TLS_NORM_STANDARD || normalization > TLS_NORM_REGULARIZED) return CHI2PER_ERR_ARGUMENT;
    if (spreading < TLS_SPREAD_AUTO || spreading > TLS_SPREAD_PIECEWISE) return CHI2PER_ERR_ARGUMENT;
    if (vectorize < TLS_VEC_AUTO || vectorize > TLS_VEC_BREADTH) return CHI2PER_ERR_ARGUMENT;
#if defined(DOUBLE_DOUBLE)
    if (spreading == TLS_SPREAD_PIECEWISE) return CHI2PER_ERR_ARGUMENT;
    if (vectorize == TLS_VEC_DEPTH) return CHI2PER_ERR_ARGUMENT;
#endif
    if (backend != CHI2PER_BACKEND_PSWF43 && backend != CHI2PER_BACKEND_PSWF21 && backend != CHI2PER_BACKEND_LRA) return CHI2PER_ERR_BACKEND;
    if (degree != 1 && solver != CHI2PER_SOLVER_LEVINSON && solver != CHI2PER_SOLVER_ZOHAR && solver != CHI2PER_SOLVER_BAREISS &&
        solver != CHI2PER_SOLVER_LDLT && solver != CHI2PER_SOLVER_SVD)
        return CHI2PER_ERR_SOLVER;
    if (normalization == TLS_NORM_REGULARIZED && degree > 1 && (solver == CHI2PER_SOLVER_LDLT || solver == CHI2PER_SOLVER_SVD)) return CHI2PER_ERR_SOLVER;
    if (M < 2 * degree + 2) return CHI2PER_ERR_DEGENERATE;

    const bool use_pswf = (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21);
    const bool use_pswf43 = (backend == CHI2PER_BACKEND_PSWF43);
    const tls_plan_choice choice = select_periodogram_plan(N, M, degree, 0, backend, spreading, vectorize);
    int plan_block = choice.block;
    const int spread_mode = choice.spread_mode;
    const int vec_mode = choice.vec_mode;
    if (use_pswf43) plan_block = tls_pswf43_plan_len_from_base(plan_block);
    const int block = use_pswf43 ? tls_pswf43_output_len_for_plan(plan_block) : plan_block;
    const int nfft = use_pswf ? chi2_bitceil((unsigned int)(2 * plan_block)) : chi2_bitceil((unsigned int)block);
    const bool use_breadth = (vec_mode == TLS_VEC_BREADTH);
    const int ladder_levels = use_breadth ? 0 : tls_twiddle_ladder_levels(N, block);
    const int max_factor = 2 * degree;

    FLOAT *w = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *yw = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    NUFFT_INPUT_T *x = (NUFFT_INPUT_T *)checked_aligned_malloc((size_t)M, sizeof(NUFFT_INPUT_T));
    block_ladder ladder;
    trig_rows rows;
    trig_rows_breadth rows_b;
    memset(&ladder, 0, sizeof(ladder));
    memset(&rows, 0, sizeof(rows));
    memset(&rows_b, 0, sizeof(rows_b));
    toep_workspace workspace;
    memset(&workspace, 0, sizeof(workspace));
    FLOAT *out_r = use_breadth ? NULL : (FLOAT *)checked_aligned_malloc((size_t)block, sizeof(FLOAT));
    FLOAT *out_i = use_breadth ? NULL : (FLOAT *)checked_aligned_malloc((size_t)block, sizeof(FLOAT));
    OUT_FLOAT *power_block = use_breadth ? NULL : (OUT_FLOAT *)checked_aligned_malloc((size_t)block, sizeof(OUT_FLOAT));
    OUT_FLOAT *cond_block = (!use_breadth && cond) ? (OUT_FLOAT *)checked_aligned_malloc((size_t)block, sizeof(OUT_FLOAT)) : NULL;
    OUT_BVEC *power_b = use_breadth ? (OUT_BVEC *)checked_aligned_malloc((size_t)block, sizeof(OUT_BVEC)) : NULL;
    OUT_BVEC *cond_b = (use_breadth && cond) ? (OUT_BVEC *)checked_aligned_malloc((size_t)block, sizeof(OUT_BVEC)) : NULL;
    void *plan = NULL;
    int status = CHI2PER_OK;

    if (!w || !yw || !x || (!use_breadth && (!out_r || !out_i || !power_block || (cond && !cond_block))) || (use_breadth && (!power_b || (cond && !cond_b)))) {
        status = CHI2PER_ERR_ALLOC;
        goto cleanup;
    }

#if defined(HALF)
    double *w_d = (double *)checked_aligned_malloc((size_t)M, sizeof(double));
    if (!w_d) {
        status = CHI2PER_ERR_ALLOC;
        goto cleanup;
    }
    const double t_mid = 0.5 * (t[0] + t[M - 1]);
    double dy_min_d = dy_d ? dy_d[0] : (double)dy_f[0];
    for (int m = 0; m < M; ++m) {
        const double dm = dy_d ? dy_d[m] : (double)dy_f[m];
        if (dm <= 0.0) {
            nanofft_aligned_free(w_d);
            status = CHI2PER_ERR_ARGUMENT;
            goto cleanup;
        }
        if (dm < dy_min_d) dy_min_d = dm;
        x[m] = t[m] - t_mid;
    }

    double ws_raw_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double dm = dy_d ? dy_d[m] : (double)dy_f[m];
        const double inv_dy = dy_min_d / dm;
        w_d[m] = inv_dy * inv_dy;
        ws_raw_d += w_d[m];
    }
    if (ws_raw_d == 0.0) {
        nanofft_aligned_free(w_d);
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    /* Keep spreading bins well below FP16 max (65504) and FFT outputs well
     * above FP16 min normal (6.1e-5), then normalize NuFFT outputs to R[0]=1. */
    double w_target = sqrt((double)nfft);
    const double sqrt_m = sqrt((double)M);
    if (sqrt_m > w_target) w_target = sqrt_m;
    if (w_target > 1024.0) w_target = 1024.0;
    const double w_scale_d = w_target / ws_raw_d;
    const double y0_d = y_d ? y_d[0] : (double)y_f[0];
    double ws_d = 0.0;
    double sum_w2_d = 0.0;
    double wy_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = y_d ? y_d[m] : (double)y_f[m];
        w_d[m] *= w_scale_d;
        ws_d += w_d[m];
        sum_w2_d += w_d[m] * w_d[m];
        wy_d += w_d[m] * (ym - y0_d);
    }
    if (ws_d == 0.0) {
        nanofft_aligned_free(w_d);
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    const double dy_mean_d = wy_d / ws_d;
    double yc_max_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = y_d ? y_d[m] : (double)y_f[m];
        const double a = fabs((ym - y0_d) - dy_mean_d);
        if (a > yc_max_d) yc_max_d = a;
    }
    if (yc_max_d == 0.0) {
        nanofft_aligned_free(w_d);
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    double chi2_tmp_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = y_d ? y_d[m] : (double)y_f[m];
        const double yc_u = ((ym - y0_d) - dy_mean_d) / yc_max_d;
        chi2_tmp_d += w_d[m] * yc_u * yc_u;
    }
    if (chi2_tmp_d == 0.0) {
        nanofft_aligned_free(w_d);
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    const double inv_y_std_tmp_d = sqrt(ws_d / chi2_tmp_d);
    double chi2_ref_d = 0.0;
    double yws_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = y_d ? y_d[m] : (double)y_f[m];
        const double yc = (((ym - y0_d) - dy_mean_d) / yc_max_d) * inv_y_std_tmp_d;
        const double ywm = yc * w_d[m];
        w[m] = FCAST(w_d[m]);
        yw[m] = FCAST(ywm);
        yws_d += ywm;
        chi2_ref_d += yc * ywm;
    }
    nanofft_aligned_free(w_d);
    if (chi2_ref_d == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    const double row_scale_d = (double)nfft / ws_d;
    const bool merge_row_scale = use_pswf && (row_scale_d < 32768.0) && (tls_tuning_ihs_deconv() == TLS_IHS_DECONV_POST);

    if (backend == CHI2PER_BACKEND_PSWF43) {
        plan = NUFFT_PSWF_INIT_OPTS(M, plan_block, NUFFT_W43, df, max_factor, "43", spread_mode, vec_mode);
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        if (merge_row_scale) NUFFT_PSWF_SET_OUTPUT_SCALE((PSWF_PLAN_T *)plan, row_scale_d);
        NUFFT_PSWF_PRE((PSWF_PLAN_T *)plan, x);
    } else if (backend == CHI2PER_BACKEND_PSWF21) {
        plan = NUFFT_PSWF_INIT_OPTS(M, plan_block, NUFFT_W21, df, max_factor, "21", spread_mode, vec_mode);
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        if (merge_row_scale) NUFFT_PSWF_SET_OUTPUT_SCALE((PSWF_PLAN_T *)plan, row_scale_d);
        NUFFT_PSWF_PRE((PSWF_PLAN_T *)plan, x);
    } else {
        plan = NUFFT_LRA_INIT_OPTS(M, block, NUFFT_RANK, df, max_factor, vec_mode);
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_LRA_PRE((LRA_PLAN_T *)plan, x, M, block, NUFFT_RANK);
    }

    const double neff_w = (ws_d * ws_d) / sum_w2_d;
    cybenko_model_t cyb = cybenko_model_init(degree, neff_w);
    OUT_FLOAT b = (float)(0.5 * (neff_w - 1.0 - 2.0 * (double)degree));

    /* Rescale every NuFFT output row by nfft / ws_d so R[0] = 1.0 and
     * chi2_ref = 1.0 entering the FP16 solvers (prevents underflow of R^2).
     * For PSWF backends with merge_row_scale, this factor is folded into the
     * deconvolution table so only a single multiply pass is needed. */
    const FLOAT row_scale = FCAST(row_scale_d);
    const FLOAT ws_fft = FCAST(1.0);
    const FLOAT yws_fft = FCAST(yws_d / ws_d);
    FLOAT chi2_ref = FCAST(chi2_ref_d / ws_d);
#else
    TIME_INPUT_T t_mid = MUL(FCONST(0.5), ADD(t[0], t[M - 1]));
    FLOAT dy_min = dy[0];
    double dy_min_d = TO_DOUBLE(dy_min);
    for (int m = 0; m < M; ++m) {
        const double dm = TO_DOUBLE(dy[m]);
        if (dm <= 0.0) {
            status = CHI2PER_ERR_ARGUMENT;
            goto cleanup;
        }
        if (dm < dy_min_d) {
            dy_min_d = dm;
            dy_min = dy[m];
        }
        x[m] = time_to_nufft_input(SUB(t[m], t_mid));
    }

    FLOAT ws_raw = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        FLOAT inv_dy = DIV(dy_min, dy[m]);
        w[m] = MUL(inv_dy, inv_dy);
        ws_raw = ADD(ws_raw, w[m]);
    }
    if (TO_DOUBLE(ws_raw) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT w_scale = DIV(M_SQRT(FCAST(M)), ws_raw);
    FLOAT y0 = y[0];
    FLOAT ws = FCAST(0.0);
    FLOAT sum_w2 = FCAST(0.0);
    FLOAT wy = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        w[m] = MUL(w[m], w_scale);
        ws = ADD(ws, w[m]);
        sum_w2 = ADD(sum_w2, MUL(w[m], w[m]));
        wy = ADD(wy, MUL(w[m], SUB(y[m], y0)));
    }
    if (TO_DOUBLE(ws) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT dy_mean = DIV(wy, ws);
    double yc_max_d = 0.0;
    for (int m = 0; m < M; ++m) {
        FLOAT yc = SUB(SUB(y[m], y0), dy_mean);
        const double a = (double)M_FABS(yc);
        if (a > yc_max_d) yc_max_d = a;
    }
    if (yc_max_d == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT yc_max = FCAST(yc_max_d);
    FLOAT chi2_tmp = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        FLOAT yc_u = DIV(SUB(SUB(y[m], y0), dy_mean), yc_max);
        chi2_tmp = ADD(chi2_tmp, MUL(w[m], MUL(yc_u, yc_u)));
    }
    if (TO_DOUBLE(chi2_tmp) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT inv_y_std_tmp = M_SQRT(DIV(ws, chi2_tmp));
    FLOAT chi2_ref = FCAST(0.0);
    FLOAT yws = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        FLOAT yc = MUL(DIV(SUB(SUB(y[m], y0), dy_mean), yc_max), inv_y_std_tmp);
        yw[m] = MUL(yc, w[m]);
        yws = ADD(yws, yw[m]);
        chi2_ref = ADD(chi2_ref, MUL(yc, yw[m]));
    }

    if (TO_DOUBLE(chi2_ref) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    if (backend == CHI2PER_BACKEND_PSWF43) {
        plan = NUFFT_PSWF_INIT_OPTS(M, plan_block, NUFFT_W43, df, max_factor, "43", spread_mode, vec_mode);
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_PSWF_PRE((PSWF_PLAN_T *)plan, x);
    } else if (backend == CHI2PER_BACKEND_PSWF21) {
        plan = NUFFT_PSWF_INIT_OPTS(M, plan_block, NUFFT_W21, df, max_factor, "21", spread_mode, vec_mode);
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_PSWF_PRE((PSWF_PLAN_T *)plan, x);
    } else {
        plan = NUFFT_LRA_INIT_OPTS(M, block, NUFFT_RANK, df, max_factor, vec_mode);
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_LRA_PRE((LRA_PLAN_T *)plan, x, M, block, NUFFT_RANK);
    }

    /* All NLL and regularized calculations use the raw-weight effective
     * sample size (sum w)^2 / sum(w^2), not the raw measurement count M. */
    const double neff_w = TO_DOUBLE(DIV(MUL(ws, ws), sum_w2));
    cybenko_model_t cyb = cybenko_model_init(degree, neff_w);
#    if defined(DOUBLE_DOUBLE)
    OUT_FLOAT b = dd_mul(dd_make(0.5, 0.0), dd_make(neff_w - 1.0 - 2.0 * (double)degree, 0.0));
#    elif defined(DOUBLE)
    OUT_FLOAT b = 0.5 * (neff_w - 1.0 - 2.0 * (double)degree);
#    else
    OUT_FLOAT b = (float)(0.5 * (neff_w - 1.0 - 2.0 * (double)degree));
#    endif

    const FLOAT inv_nfft = DIV(FCAST(1.0), FCAST(nfft));
    const FLOAT ws_fft = MUL(ws, inv_nfft);
    const FLOAT yws_fft = MUL(yws, inv_nfft);
    chi2_ref = MUL(chi2_ref, inv_nfft);
#endif
    const int defer = (tls_tuning_defer() != 0 && use_pswf && NUFFT_PSWF_SLOTS_AVAILABLE((PSWF_PLAN_T *)plan)) ? 1 : 0;
    const size_t num_blocks = ((size_t)N + (size_t)block - 1u) / (size_t)block;
    int nan_count = 0;

    if (use_breadth) {
        status = trig_rows_breadth_init(&rows_b, block, max_factor, degree);
        if (status != CHI2PER_OK) goto cleanup;
        if (degree > 1 && (solver == CHI2PER_SOLVER_LEVINSON || solver == CHI2PER_SOLVER_ZOHAR || solver == CHI2PER_SOLVER_BAREISS)) {
            status = toep_workspace_init(&workspace, 2 * degree + 1, solver);
            if (status != CHI2PER_OK) goto cleanup;
        }
        const int slot_block = (defer && use_pswf) ? 1 : 0;
#if defined(HALF)
        const int mode = slot_block ? TLS_NUFFT_OUT_SLOTS : TLS_NUFFT_OUT_NATURAL;
        const FLOAT *slot_phi = NULL;
        const FLOAT *slot_dec = NULL;
#else
        const bool raw_slots = (slot_block && tls_tuning_ihs_deconv() == TLS_IHS_DECONV_POST);
        const int mode = raw_slots ? TLS_NUFFT_OUT_SLOTS_RAW : (slot_block ? TLS_NUFFT_OUT_SLOTS : TLS_NUFFT_OUT_NATURAL);
        const FLOAT *slot_phi = raw_slots ? NUFFT_PSWF_SLOT_PHI_HAT((PSWF_PLAN_T *)plan) : NULL;
        const FLOAT *slot_dec = raw_slots ? NUFFT_PSWF_SLOT_DECONV((PSWF_PLAN_T *)plan) : NULL;
#endif
        const BREADTH_COMPUTE_VEC v_ws = BREADTH_SPLAT(ws_fft);
        const BREADTH_COMPUTE_VEC v_yws = BREADTH_SPLAT(yws_fft);
        const BREADTH_COMPUTE_VEC v_zero = BREADTH_ZERO;
#if defined(HALF)
        const BREADTH_COMPUTE_VEC v_rscale = BREADTH_SPLAT(row_scale);
#endif

        for (size_t block_base = 0; block_base < num_blocks; block_base += (size_t)BREADTH_VEC_LEN) {
            const size_t rem_blocks = num_blocks - block_base;
            const int active_lanes = (rem_blocks < (size_t)BREADTH_VEC_LEN) ? (int)rem_blocks : BREADTH_VEC_LEN;
            const int base = (int)(block_base * (size_t)block);
            const int total_remaining = N - base;
            const int batch_max = active_lanes * block;
            const int batch_written = (total_remaining < batch_max) ? total_remaining : batch_max;

            if (slot_phi) {
                for (int k = 0; k < block; ++k) {
                    const FLOAT pk = slot_phi[k];
                    STORE_BVEC(&rows_b.Cw[k], BREADTH_MUL_S(v_ws, pk));
                    STORE_BVEC(&rows_b.Sw[k], v_zero);
                    STORE_BVEC(&rows_b.Cyw[k], BREADTH_MUL_S(v_yws, pk));
                    STORE_BVEC(&rows_b.Syw[k], v_zero);
                }
            } else {
                for (int k = 0; k < block; ++k) {
                    STORE_BVEC(&rows_b.Cw[k], v_ws);
                    STORE_BVEC(&rows_b.Sw[k], v_zero);
                    STORE_BVEC(&rows_b.Cyw[k], v_yws);
                    STORE_BVEC(&rows_b.Syw[k], v_zero);
                }
            }

            for (int q = 1; q <= max_factor; ++q) {
                BREADTH_VEC *cw_q = rows_b.Cw + (size_t)q * (size_t)block;
                BREADTH_VEC *sw_q = rows_b.Sw + (size_t)q * (size_t)block;
                if (q <= degree) {
                    BREADTH_VEC *cyw_q = rows_b.Cyw + (size_t)q * (size_t)block;
                    BREADTH_VEC *syw_q = rows_b.Syw + (size_t)q * (size_t)block;
                    status = execute_nufft_breadth_pair(plan, backend, x, w, yw, f0, block, (int)block_base, q, mode, cw_q, sw_q, cyw_q, syw_q);
                    if (status != CHI2PER_OK) goto cleanup;
#if defined(HALF)
                    if (!merge_row_scale) {
                        for (int k = 0; k < block; ++k) {
                            STORE_BVEC(&cw_q[k], BREADTH_MUL(LOAD_BVEC(&cw_q[k]), v_rscale));
                            STORE_BVEC(&sw_q[k], BREADTH_MUL(LOAD_BVEC(&sw_q[k]), v_rscale));
                            STORE_BVEC(&cyw_q[k], BREADTH_MUL(LOAD_BVEC(&cyw_q[k]), v_rscale));
                            STORE_BVEC(&syw_q[k], BREADTH_MUL(LOAD_BVEC(&syw_q[k]), v_rscale));
                        }
                    }
#endif
                } else {
                    status = execute_nufft_breadth(plan, backend, x, w, f0, block, (int)block_base, q, mode, cw_q, sw_q);
                    if (status != CHI2PER_OK) goto cleanup;
#if defined(HALF)
                    if (!merge_row_scale) {
                        for (int k = 0; k < block; ++k) {
                            STORE_BVEC(&cw_q[k], BREADTH_MUL(LOAD_BVEC(&cw_q[k]), v_rscale));
                            STORE_BVEC(&sw_q[k], BREADTH_MUL(LOAD_BVEC(&sw_q[k]), v_rscale));
                        }
                    }
#endif
                }
            }

            int block_status;
            if (degree == 1) {
                block_status = gls_impl_breadth(rows_b.Sw, rows_b.Cw, rows_b.Syw, rows_b.Cyw, block, active_lanes, ws_fft, yws_fft, chi2_ref, normalization, M,
                                                &cyb, b, slot_dec, power_b, cond_b);
            } else {
                block_status = solve_periodogram_breadth_vec(rows_b.Sw, rows_b.Cw, rows_b.Syw, rows_b.Cyw, block, active_lanes, degree, solver, chi2_ref,
                                                             normalization, M, &cyb, b, slot_dec, power_b, cond_b, &workspace);
            }
            if (block_status < 0) {
                status = block_status;
                goto cleanup;
            }

            if (slot_block) {
                NUFFT_PSWF_FINISH_SLOTS_BREADTH_OUT((PSWF_PLAN_T *)plan, power_b, power + base, block, active_lanes, total_remaining);
                if (cond) NUFFT_PSWF_FINISH_SLOTS_BREADTH_OUT((PSWF_PLAN_T *)plan, cond_b, cond + base, block, active_lanes, total_remaining);
            } else {
                for (int v = 0; v < active_lanes; ++v) {
                    const int rem = total_remaining - v * block;
                    if (rem <= 0) break;
                    const int count_v = (rem < block) ? rem : block;
                    OUT_FLOAT *pdst = power + base + (size_t)v * (size_t)block;
                    for (int k = 0; k < count_v; ++k) pdst[k] = OUT_BVEC_GET_LANE(power_b[k], v);
                    if (cond) {
                        OUT_FLOAT *cdst = cond + base + (size_t)v * (size_t)block;
                        for (int k = 0; k < count_v; ++k) cdst[k] = OUT_BVEC_GET_LANE(cond_b[k], v);
                    }
                }
            }
            if (cond == NULL) {
                for (int i = 0; i < batch_written; ++i) {
                    if (power_is_nan(power[base + i])) ++nan_count;
                }
            }
        }
        if (cond == NULL) status = nan_count;

        tls_tuning_report_work_bytes(
            (unsigned long long)((size_t)(2 * (max_factor + 1) + 2 * (degree + 1) + (cond ? 2 : 1)) * (size_t)block * sizeof(BREADTH_VEC)));
        goto cleanup;
    }

    status = trig_rows_init(&rows, block, max_factor, degree, INTERNAL_VEC_LEN);
    if (status == CHI2PER_OK) status = block_ladder_init(&ladder, x, w, yw, M, max_factor, degree, block, ladder_levels, backend, f0, df);
    if (status != CHI2PER_OK) goto cleanup;

    if (degree > 1 && (solver == CHI2PER_SOLVER_LEVINSON || solver == CHI2PER_SOLVER_ZOHAR || solver == CHI2PER_SOLVER_BAREISS)) {
        status = toep_workspace_init(&workspace, 2 * degree + 1, solver);
        if (status != CHI2PER_OK) goto cleanup;
    }

    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int base = (int)(block_idx * (size_t)block);
        const int count = (base + block <= N) ? block : (N - base);
        const int slot_block = (defer && count == block) ? 1 : 0;
#if defined(HALF)
        const int mode = slot_block ? TLS_NUFFT_OUT_SLOTS : TLS_NUFFT_OUT_NATURAL;
        const FLOAT *slot_phi = NULL;
        const FLOAT *slot_dec = NULL;
#else
        const bool raw_slots = (slot_block && tls_tuning_ihs_deconv() == TLS_IHS_DECONV_POST);
        const int mode = raw_slots ? TLS_NUFFT_OUT_SLOTS_RAW : (slot_block ? TLS_NUFFT_OUT_SLOTS : TLS_NUFFT_OUT_NATURAL);
        const FLOAT *slot_phi = raw_slots ? NUFFT_PSWF_SLOT_PHI_HAT((PSWF_PLAN_T *)plan) : NULL;
        const FLOAT *slot_dec = raw_slots ? NUFFT_PSWF_SLOT_DECONV((PSWF_PLAN_T *)plan) : NULL;
#endif

        block_ladder_advance(&ladder, (int)block_idx);

        if (slot_phi) {
            for (int k = 0; k < count; ++k) {
                const FLOAT pk = slot_phi[k];
                rows.Cw[k] = MUL(ws_fft, pk);
                rows.Sw[k] = FCAST(0.0);
                rows.Cyw[k] = MUL(yws_fft, pk);
                rows.Syw[k] = FCAST(0.0);
            }
        } else {
            for (int k = 0; k < count; ++k) {
                rows.Cw[k] = ws_fft;
                rows.Sw[k] = FCAST(0.0);
                rows.Cyw[k] = yws_fft;
                rows.Syw[k] = FCAST(0.0);
            }
        }

        for (int q = 1; q <= max_factor; ++q) {
            const size_t src_off = (size_t)(q - 1) * (size_t)ladder_levels * (size_t)M;
            FLOAT *cw_q = rows.Cw + (size_t)q * (size_t)rows.stride;
            FLOAT *sw_q = rows.Sw + (size_t)q * (size_t)rows.stride;
            if (q <= degree && ladder.srcy_r) {
                FLOAT *cyw_q = rows.Cyw + (size_t)q * (size_t)rows.stride;
                FLOAT *syw_q = rows.Syw + (size_t)q * (size_t)rows.stride;
                status = execute_nufft_block_pair(plan, backend, ladder.srcw_r + src_off, ladder.srcw_i + src_off, ladder.srcy_r + src_off,
                                                  ladder.srcy_i + src_off, cw_q, sw_q, cyw_q, syw_q, q, mode);
                if (status != CHI2PER_OK) goto cleanup;
#if defined(HALF)
                if (!merge_row_scale) {
                    for (int k = 0; k < count; ++k) {
                        cw_q[k] = MUL(cw_q[k], row_scale);
                        sw_q[k] = MUL(sw_q[k], row_scale);
                        cyw_q[k] = MUL(cyw_q[k], row_scale);
                        syw_q[k] = MUL(syw_q[k], row_scale);
                    }
                }
#endif
            } else {
                status = execute_nufft_block(plan, backend, ladder.srcw_r + src_off, ladder.srcw_i + src_off, cw_q, sw_q, q, mode);
                if (status != CHI2PER_OK) goto cleanup;
#if defined(HALF)
                if (!merge_row_scale) {
                    for (int k = 0; k < count; ++k) {
                        cw_q[k] = MUL(cw_q[k], row_scale);
                        sw_q[k] = MUL(sw_q[k], row_scale);
                    }
                }
#endif
            }
        }

        if (count != rows.stride) trig_rows_compact(&rows, count);

        int block_status;
        if (degree == 1) {
            block_status =
                gls_impl(rows.Sw, rows.Cw, rows.Syw, rows.Cyw, count, ws_fft, yws_fft, chi2_ref, normalization, M, &cyb, b, slot_dec, power_block, cond_block);
        } else {
            block_status = solve_periodogram_vec(rows.Sw, rows.Cw, rows.Syw, rows.Cyw, count, degree, solver, chi2_ref, normalization, M, &cyb, b, slot_dec,
                                                 power_block, cond_block, &workspace);
        }
        if (block_status < 0) {
            status = block_status;
            goto cleanup;
        }
        if (block_status > 0) nan_count += block_status;

        if (slot_block) {
            NUFFT_PSWF_FINISH_SLOTS_OUT((PSWF_PLAN_T *)plan, power_block, power + base, count);
            if (cond) NUFFT_PSWF_FINISH_SLOTS_OUT((PSWF_PLAN_T *)plan, cond_block, cond + base, count);
        } else {
            memcpy(power + base, power_block, (size_t)count * sizeof(OUT_FLOAT));
            if (cond) memcpy(cond + base, cond_block, (size_t)count * sizeof(OUT_FLOAT));
        }
    }
    if (cond == NULL) status = nan_count;

    tls_tuning_report_work_bytes((unsigned long long)((size_t)(2 * max_factor + 2 * degree) * (size_t)ladder_levels * (size_t)M * sizeof(FLOAT) +
                                                      (size_t)(2 * max_factor + degree + 2) * (size_t)rows.stride * sizeof(FLOAT) +
                                                      (size_t)3 * (size_t)block * sizeof(FLOAT)));

cleanup:
    if (plan && use_pswf)
        NUFFT_PSWF_FREE((PSWF_PLAN_T *)plan);
    else if (plan)
        NUFFT_LRA_FREE((LRA_PLAN_T *)plan);

    nanofft_aligned_free(w);
    nanofft_aligned_free(yw);
    nanofft_aligned_free(x);
    nanofft_aligned_free(out_r);
    nanofft_aligned_free(out_i);
    nanofft_aligned_free(power_block);
    nanofft_aligned_free(cond_block);
    nanofft_aligned_free(power_b);
    nanofft_aligned_free(cond_b);
    block_ladder_free(&ladder);
    trig_rows_free(&rows);
    trig_rows_breadth_free(&rows_b);
    toep_workspace_free(&workspace);
    return status;
}

#if defined(HALF)
#    if defined(TLS_HALF_HAS_NATIVE_EMBED) && !defined(HALF_NATIVE_EMBED)
static int tlsh_native_dispatch_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        unsigned long long f = tls_cpu_features();
        cached = (f & (TLS_CPU_AVX512FP16 | TLS_CPU_FP16)) ? 1 : 0;
    }
    return cached;
}
#    endif

int CHI2_PREFIX(fastchi2_opts)(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                               int normalization, int spreading, int vectorize, float *power, float *cond) {
#    if defined(TLS_HALF_HAS_NATIVE_EMBED) && !defined(HALF_NATIVE_EMBED)
    if (tlsh_native_dispatch_enabled())
        return tlsh_nat_fastchi2_opts(t, y, dy, M, f0, df, N, degree, backend, solver, normalization, spreading, vectorize, power, cond);
#    endif
    return fastchi2_core(t, y, NULL, dy, NULL, M, f0, df, N, degree, backend, solver, normalization, spreading, vectorize, power, cond);
}

int CHI2_PREFIX(fastchi2_opts_d)(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                                 int normalization, int spreading, int vectorize, float *power, float *cond) {
#    if defined(TLS_HALF_HAS_NATIVE_EMBED) && !defined(HALF_NATIVE_EMBED)
    if (tlsh_native_dispatch_enabled())
        return tlsh_nat_fastchi2_opts_d(t, y, dy, M, f0, df, N, degree, backend, solver, normalization, spreading, vectorize, power, cond);
#    endif
    return fastchi2_core(t, NULL, y, NULL, dy, M, f0, df, N, degree, backend, solver, normalization, spreading, vectorize, power, cond);
}

#    ifndef HALF_NATIVE_EMBED
int tlsh_fastchi2_spread(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                         int normalization, int spreading, float *power, float *cond) {
    return tlsh_fastchi2_opts(t, y, dy, M, f0, df, N, degree, backend, solver, normalization, spreading, tls_tuning_vectorize(), power, cond);
}

int tlsh_fastchi2_spread_d(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                           int normalization, int spreading, float *power, float *cond) {
    return tlsh_fastchi2_opts_d(t, y, dy, M, f0, df, N, degree, backend, solver, normalization, spreading, tls_tuning_vectorize(), power, cond);
}

int tlsh_fastchi2(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int solver, int normalization,
                  float *power, float *cond) {
    return tlsh_fastchi2_spread(t, y, dy, M, f0, df, N, degree, backend, solver, normalization, tls_tuning_spread(), power, cond);
}

int tlsh_fastchi2_d(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                    int normalization, float *power, float *cond) {
    return tlsh_fastchi2_spread_d(t, y, dy, M, f0, df, N, degree, backend, solver, normalization, tls_tuning_spread(), power, cond);
}

int tlsh_using_native_fp16(void) {
#        if NANOFFT_NATIVE_FP16
    return 1;
#        elif defined(TLS_HALF_HAS_NATIVE_EMBED)
    return tlsh_native_dispatch_enabled();
#        else
    return 0;
#        endif
}
#    endif

static int ihs_core(const double *t, const float *y_f, const double *y_d, const float *dy_f, const double *dy_d, int M, double f0, double df, int N, int degree,
                    int backend, int normalization, int spreading, int vectorize, float *power) {
    if (!t || (!y_f && !y_d) || (!dy_f && !dy_d) || !power || M <= 0 || N <= 0 || degree <= 0 || f0 < 0.0 || df <= 0.0) return CHI2PER_ERR_ARGUMENT;
#else
int CHI2_PREFIX(fastchi2_spread)(const TIME_INPUT_T *t, const FLOAT *y, const FLOAT *dy, int M, double f0, double df, int N, int degree, int backend,
                                 int solver, int normalization, int spreading, FLOAT *power, FLOAT *cond) {
#    if defined(DOUBLE_DOUBLE)
    int vectorize = (tls_tuning_vectorize() == TLS_VEC_BREADTH) ? TLS_VEC_BREADTH : TLS_VEC_AUTO;
#    else
    int vectorize = tls_tuning_vectorize();
#    endif
    return CHI2_PREFIX(fastchi2_opts)(t, y, dy, M, f0, df, N, degree, backend, solver, normalization, spreading, vectorize, power, cond);
}

int CHI2_PREFIX(fastchi2)(const TIME_INPUT_T *t, const FLOAT *y, const FLOAT *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                          int normalization, FLOAT *power, FLOAT *cond) {
#    if defined(DOUBLE_DOUBLE)
    int spreading = TLS_SPREAD_MINMAX;
#    else
    int spreading = tls_tuning_spread();
#    endif
    return CHI2_PREFIX(fastchi2_spread)(t, y, dy, M, f0, df, N, degree, backend, solver, normalization, spreading, power, cond);
}

int CHI2_PREFIX(ihs_opts)(const TIME_INPUT_T *t, const FLOAT *y, const FLOAT *dy, int M, double f0, double df, int N, int degree, int backend,
                          int normalization, int spreading, int vectorize, FLOAT *power) {
    if (!t || !y || !dy || !power || M <= 0 || N <= 0 || degree <= 0 || f0 < 0.0 || df <= 0.0) return CHI2PER_ERR_ARGUMENT;
#endif
    if (normalization != TLS_NORM_STANDARD && normalization != TLS_NORM_NLL && normalization != TLS_NORM_IHS_DEFAULT) return CHI2PER_ERR_ARGUMENT;
    if (spreading < TLS_SPREAD_AUTO || spreading > TLS_SPREAD_PIECEWISE) return CHI2PER_ERR_ARGUMENT;
    if (vectorize < TLS_VEC_AUTO || vectorize > TLS_VEC_BREADTH) return CHI2PER_ERR_ARGUMENT;
#if defined(DOUBLE_DOUBLE)
    if (spreading == TLS_SPREAD_PIECEWISE) return CHI2PER_ERR_ARGUMENT;
    if (vectorize == TLS_VEC_DEPTH) return CHI2PER_ERR_ARGUMENT;
#endif
    if (backend != CHI2PER_BACKEND_PSWF43 && backend != CHI2PER_BACKEND_PSWF21 && backend != CHI2PER_BACKEND_LRA) return CHI2PER_ERR_BACKEND;
    if (M < 2 * degree + 2) return CHI2PER_ERR_DEGENERATE;

    const bool use_pswf = (backend == CHI2PER_BACKEND_PSWF43 || backend == CHI2PER_BACKEND_PSWF21);
    const bool use_pswf43 = (backend == CHI2PER_BACKEND_PSWF43);
    const tls_plan_choice choice = select_periodogram_plan(N, M, degree, 1, backend, spreading, vectorize);
    int plan_block = choice.block;
    const int spread_mode = choice.spread_mode;
    const int vec_mode = choice.vec_mode;
    if (use_pswf43) plan_block = tls_pswf43_plan_len_from_base(plan_block);
    const int block = use_pswf43 ? tls_pswf43_output_len_for_plan(plan_block) : plan_block;
    const int nfft = use_pswf ? chi2_bitceil((unsigned int)(2 * plan_block)) : chi2_bitceil((unsigned int)block);
    const bool use_breadth = (vec_mode == TLS_VEC_BREADTH);
    const int ladder_levels = use_breadth ? 0 : tls_twiddle_ladder_levels(N, block);

    FLOAT *w = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    FLOAT *wy = (FLOAT *)checked_aligned_malloc((size_t)M, sizeof(FLOAT));
    NUFFT_INPUT_T *x = (NUFFT_INPUT_T *)checked_aligned_malloc((size_t)M, sizeof(NUFFT_INPUT_T));
    block_ladder ladder;
    memset(&ladder, 0, sizeof(ladder));
    FLOAT *out_r = use_breadth ? NULL : (FLOAT *)checked_aligned_malloc((size_t)block, sizeof(FLOAT));
    FLOAT *out_i = use_breadth ? NULL : (FLOAT *)checked_aligned_malloc((size_t)block, sizeof(FLOAT));
    void *plan = NULL;
    int status = CHI2PER_OK;

    if (!w || !wy || !x || (!use_breadth && (!out_r || !out_i))) {
        status = CHI2PER_ERR_ALLOC;
        goto cleanup;
    }

#if defined(HALF)
    double *w_d = (double *)checked_aligned_malloc((size_t)M, sizeof(double));
    if (!w_d) {
        status = CHI2PER_ERR_ALLOC;
        goto cleanup;
    }
    const double t_mid = 0.5 * (t[0] + t[M - 1]);
    double dy_min_d = dy_d ? dy_d[0] : (double)dy_f[0];
    for (int m = 0; m < M; ++m) {
        const double dm = dy_d ? dy_d[m] : (double)dy_f[m];
        if (dm <= 0.0) {
            nanofft_aligned_free(w_d);
            status = CHI2PER_ERR_ARGUMENT;
            goto cleanup;
        }
        if (dm < dy_min_d) dy_min_d = dm;
        x[m] = t[m] - t_mid;
    }

    double ws_raw_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double dm = dy_d ? dy_d[m] : (double)dy_f[m];
        const double inv_dy = dy_min_d / dm;
        w_d[m] = inv_dy * inv_dy;
        ws_raw_d += w_d[m];
    }
    if (ws_raw_d == 0.0) {
        nanofft_aligned_free(w_d);
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    double w_target = sqrt((double)nfft);
    const double sqrt_m = sqrt((double)M);
    if (sqrt_m > w_target) w_target = sqrt_m;
    if (w_target > 1024.0) w_target = 1024.0;
    const double w_scale_d = w_target / ws_raw_d;
    const double y0_d = y_d ? y_d[0] : (double)y_f[0];
    double ws_d = 0.0;
    double wy_mean_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = y_d ? y_d[m] : (double)y_f[m];
        w_d[m] *= w_scale_d;
        ws_d += w_d[m];
        wy_mean_d += w_d[m] * (ym - y0_d);
    }
    if (ws_d == 0.0) {
        nanofft_aligned_free(w_d);
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    const double dy_mean_d = wy_mean_d / ws_d;
    double yc_max_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = y_d ? y_d[m] : (double)y_f[m];
        const double a = fabs((ym - y0_d) - dy_mean_d);
        if (a > yc_max_d) yc_max_d = a;
    }
    if (yc_max_d == 0.0) {
        nanofft_aligned_free(w_d);
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    double chi2_tmp_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = y_d ? y_d[m] : (double)y_f[m];
        const double yc_u = ((ym - y0_d) - dy_mean_d) / yc_max_d;
        chi2_tmp_d += w_d[m] * yc_u * yc_u;
    }
    if (chi2_tmp_d == 0.0) {
        nanofft_aligned_free(w_d);
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    const double inv_y_std_tmp_d = sqrt(ws_d / chi2_tmp_d);
    double wy_sum_abs_d = 0.0;
    double wy_sq_sum_d = 0.0;
    double chi2_ref_d = 0.0;
    for (int m = 0; m < M; ++m) {
        const double ym = y_d ? y_d[m] : (double)y_f[m];
        const double yc = (((ym - y0_d) - dy_mean_d) / yc_max_d) * inv_y_std_tmp_d;
        const double wym = w_d[m] * yc;
        w[m] = FCAST(w_d[m]);
        wy[m] = FCAST(wym);
        wy_sum_abs_d += fabs(wym);
        wy_sq_sum_d += wym * wym;
        chi2_ref_d += wym * yc;
    }
    nanofft_aligned_free(w_d);
    if (chi2_ref_d == 0.0 || wy_sq_sum_d == 0.0 || wy_sum_abs_d == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    const double inv_nfft_d = 1.0 / (double)nfft;
    const double ws_fft_d = ws_d * inv_nfft_d;
    const double chi2_ref_fft_d = chi2_ref_d * inv_nfft_d;
    OUT_FLOAT standard_scale = (float)(2.0 / (ws_fft_d * chi2_ref_fft_d));
    const double norm_neff_d = (wy_sum_abs_d * wy_sum_abs_d) / wy_sq_sum_d;
    double nll_scale_d = 0.0;
    if (norm_neff_d > 0.0) {
        const double nfft_over_sum = (double)nfft / wy_sum_abs_d;
        nll_scale_d = norm_neff_d * nfft_over_sum * nfft_over_sum;
    }
    if (!double_is_finite_bits(nll_scale_d) || nll_scale_d < 0.0) nll_scale_d = 0.0;
    OUT_FLOAT nll_scale = (float)nll_scale_d;
#else
    TIME_INPUT_T t_mid = MUL(FCONST(0.5), ADD(t[0], t[M - 1]));
    FLOAT dy_min = dy[0];
    double dy_min_d = TO_DOUBLE(dy_min);
    for (int m = 0; m < M; ++m) {
        const double dm = TO_DOUBLE(dy[m]);
        if (dm <= 0.0) {
            status = CHI2PER_ERR_ARGUMENT;
            goto cleanup;
        }
        if (dm < dy_min_d) {
            dy_min_d = dm;
            dy_min = dy[m];
        }
        x[m] = time_to_nufft_input(SUB(t[m], t_mid));
    }

    FLOAT ws_raw = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        FLOAT inv_dy = DIV(dy_min, dy[m]);
        w[m] = MUL(inv_dy, inv_dy);
        ws_raw = ADD(ws_raw, w[m]);
    }
    if (TO_DOUBLE(ws_raw) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT w_scale = DIV(M_SQRT(FCAST(M)), ws_raw);
    FLOAT y0 = y[0];
    FLOAT ws = FCAST(0.0);
    FLOAT wy_mean = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        w[m] = MUL(w[m], w_scale);
        ws = ADD(ws, w[m]);
        wy_mean = ADD(wy_mean, MUL(w[m], SUB(y[m], y0)));
    }
    if (TO_DOUBLE(ws) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT dy_mean = DIV(wy_mean, ws);
    double yc_max_d = 0.0;
    for (int m = 0; m < M; ++m) {
        FLOAT yc = SUB(SUB(y[m], y0), dy_mean);
        const double a = (double)M_FABS(yc);
        if (a > yc_max_d) yc_max_d = a;
    }
    if (yc_max_d == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT yc_max = FCAST(yc_max_d);
    FLOAT chi2_tmp = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        FLOAT yc_u = DIV(SUB(SUB(y[m], y0), dy_mean), yc_max);
        chi2_tmp = ADD(chi2_tmp, MUL(w[m], MUL(yc_u, yc_u)));
    }
    if (TO_DOUBLE(chi2_tmp) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    FLOAT inv_y_std_tmp = M_SQRT(DIV(ws, chi2_tmp));
    FLOAT wy_sum_abs = FCAST(0.0);
    FLOAT wy_sq_sum = FCAST(0.0);
    FLOAT chi2_ref = FCAST(0.0);
    for (int m = 0; m < M; ++m) {
        FLOAT yc = MUL(DIV(SUB(SUB(y[m], y0), dy_mean), yc_max), inv_y_std_tmp);
        wy[m] = MUL(w[m], yc);
        wy_sum_abs = ADD(wy_sum_abs, FCAST(M_FABS(wy[m])));
        wy_sq_sum = ADD(wy_sq_sum, MUL(wy[m], wy[m]));
        chi2_ref = ADD(chi2_ref, MUL(wy[m], yc));
    }

    if (TO_DOUBLE(chi2_ref) == 0.0 || TO_DOUBLE(wy_sq_sum) == 0.0 || TO_DOUBLE(wy_sum_abs) == 0.0) {
        status = CHI2PER_ERR_DEGENERATE;
        goto cleanup;
    }

    const FLOAT inv_nfft = DIV(FCAST(1.0), FCAST(nfft));
    const FLOAT ws_fft = MUL(ws, inv_nfft);
    const FLOAT chi2_ref_fft = MUL(chi2_ref, inv_nfft);
    OUT_FLOAT standard_scale = DIV(FCAST(2.0), MUL(ws_fft, chi2_ref_fft));
    FLOAT norm_neff = DIV(MUL(wy_sum_abs, wy_sum_abs), wy_sq_sum);
    const double norm_neff_d = TO_DOUBLE(norm_neff);
    OUT_FLOAT nll_scale = FCAST(0.0);
    if (norm_neff_d > 0.0) {
        FLOAT nfft_over_sum = DIV(FCAST(nfft), wy_sum_abs);
        nll_scale = MUL(norm_neff, MUL(nfft_over_sum, nfft_over_sum));
    }
    if (!double_is_finite_bits(TO_DOUBLE(nll_scale)) || TO_DOUBLE(nll_scale) < 0.0) nll_scale = FCAST(0.0);
#endif

    if (backend == CHI2PER_BACKEND_PSWF43) {
#if defined(HALF)
        plan = NUFFT_PSWF_INIT_GEO_OPTS(M, plan_block, NUFFT_W43, df, degree, "43", spread_mode, vec_mode);
#else
        plan = NUFFT_PSWF_INIT_OPTS(M, plan_block, NUFFT_W43, df, degree, "43", spread_mode, vec_mode);
#endif
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_PSWF_PRE((PSWF_PLAN_T *)plan, x);
    } else if (backend == CHI2PER_BACKEND_PSWF21) {
#if defined(HALF)
        plan = NUFFT_PSWF_INIT_GEO_OPTS(M, plan_block, NUFFT_W21, df, degree, "21", spread_mode, vec_mode);
#else
        plan = NUFFT_PSWF_INIT_OPTS(M, plan_block, NUFFT_W21, df, degree, "21", spread_mode, vec_mode);
#endif
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_PSWF_PRE((PSWF_PLAN_T *)plan, x);
    } else {
        plan = NUFFT_LRA_INIT_OPTS(M, block, NUFFT_RANK, df, degree, vec_mode);
        if (!plan) {
            status = CHI2PER_ERR_ALLOC;
            goto cleanup;
        }
        NUFFT_LRA_PRE((LRA_PLAN_T *)plan, x, M, block, NUFFT_RANK);
    }

    const int defer = (tls_tuning_defer() != 0 && use_pswf && NUFFT_PSWF_SLOTS_AVAILABLE((PSWF_PLAN_T *)plan)) ? 1 : 0;
    if (use_breadth) {
        status = ihs_power_sweep_breadth(x, wy, f0, N, block, degree, plan, backend, power, defer, normalization, standard_scale, nll_scale, norm_neff_d);
    } else {
        status = block_ladder_init(&ladder, x, wy, NULL, M, degree, 0, block, ladder_levels, backend, f0, df);
        if (status == CHI2PER_OK) {
            status = ihs_power_sweep_depth(&ladder, N, block, degree, plan, backend, power, out_r, out_i, defer, normalization, standard_scale, nll_scale,
                                           norm_neff_d);
        }
    }

    if (status == CHI2PER_OK) {
        if (use_breadth) {
            tls_tuning_report_work_bytes((unsigned long long)((size_t)3 * (size_t)block * sizeof(BREADTH_VEC)));
        } else {
            tls_tuning_report_work_bytes(
                (unsigned long long)((size_t)degree * (size_t)ladder_levels * (size_t)M * 4u * sizeof(FLOAT) + (size_t)2 * (size_t)block * sizeof(FLOAT)));
        }
    }

cleanup:
    if (plan && use_pswf)
        NUFFT_PSWF_FREE((PSWF_PLAN_T *)plan);
    else if (plan)
        NUFFT_LRA_FREE((LRA_PLAN_T *)plan);

    nanofft_aligned_free(w);
    nanofft_aligned_free(wy);
    nanofft_aligned_free(x);
    nanofft_aligned_free(out_r);
    nanofft_aligned_free(out_i);
    block_ladder_free(&ladder);
    return status;
}

#if defined(HALF)
int CHI2_PREFIX(ihs_opts)(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                          int spreading, int vectorize, float *power) {
#    if defined(TLS_HALF_HAS_NATIVE_EMBED) && !defined(HALF_NATIVE_EMBED)
    if (tlsh_native_dispatch_enabled()) return tlsh_nat_ihs_opts(t, y, dy, M, f0, df, N, degree, backend, normalization, spreading, vectorize, power);
#    endif
    return ihs_core(t, y, NULL, dy, NULL, M, f0, df, N, degree, backend, normalization, spreading, vectorize, power);
}

int CHI2_PREFIX(ihs_opts_d)(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                            int spreading, int vectorize, float *power) {
#    if defined(TLS_HALF_HAS_NATIVE_EMBED) && !defined(HALF_NATIVE_EMBED)
    if (tlsh_native_dispatch_enabled()) return tlsh_nat_ihs_opts_d(t, y, dy, M, f0, df, N, degree, backend, normalization, spreading, vectorize, power);
#    endif
    return ihs_core(t, NULL, y, NULL, dy, M, f0, df, N, degree, backend, normalization, spreading, vectorize, power);
}

#    ifndef HALF_NATIVE_EMBED
int tlsh_ihs_spread(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                    int spreading, float *power) {
    return tlsh_ihs_opts(t, y, dy, M, f0, df, N, degree, backend, normalization, spreading, tls_tuning_vectorize(), power);
}

int tlsh_ihs_spread_d(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                      int spreading, float *power) {
    return tlsh_ihs_opts_d(t, y, dy, M, f0, df, N, degree, backend, normalization, spreading, tls_tuning_vectorize(), power);
}

int tlsh_ihs(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int normalization, float *power) {
    return tlsh_ihs_spread(t, y, dy, M, f0, df, N, degree, backend, normalization, tls_tuning_spread(), power);
}

int tlsh_ihs_d(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
               float *power) {
    return tlsh_ihs_spread_d(t, y, dy, M, f0, df, N, degree, backend, normalization, tls_tuning_spread(), power);
}
#    endif
#else
int CHI2_PREFIX(ihs_spread)(const TIME_INPUT_T *t, const FLOAT *y, const FLOAT *dy, int M, double f0, double df, int N, int degree, int backend,
                            int normalization, int spreading, FLOAT *power) {
#    if defined(DOUBLE_DOUBLE)
    int vectorize = (tls_tuning_vectorize() == TLS_VEC_BREADTH) ? TLS_VEC_BREADTH : TLS_VEC_AUTO;
#    else
    int vectorize = tls_tuning_vectorize();
#    endif
    return CHI2_PREFIX(ihs_opts)(t, y, dy, M, f0, df, N, degree, backend, normalization, spreading, vectorize, power);
}

int CHI2_PREFIX(ihs)(const TIME_INPUT_T *t, const FLOAT *y, const FLOAT *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                     FLOAT *power) {
#    if defined(DOUBLE_DOUBLE)
    int spreading = TLS_SPREAD_MINMAX;
#    else
    int spreading = tls_tuning_spread();
#    endif
    return CHI2_PREFIX(ihs_spread)(t, y, dy, M, f0, df, N, degree, backend, normalization, spreading, power);
}
#endif
