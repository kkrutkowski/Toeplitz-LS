#ifndef TLS_INTERNAL_H
#define TLS_INTERNAL_H

/*
 * Internal interfaces shared between the library translation units.
 *
 * This header is not part of the installed API; it only exists so chi2per.c
 * can drive the slot-order (deferred bit reversal) NuFFT output without
 * putting those entry points in the public nufft1.h.
 */

#include <nanofft.h>
#include <nufft1.h>

#ifdef __cplusplus
extern "C" {
#endif

/* How a PSWF NuFFT hands one block back to the caller. */
enum {
    TLS_NUFFT_OUT_NATURAL = 0, /* frequency order, Nout entries (classic contract) */
    TLS_NUFFT_OUT_SLOTS = 1    /* kept FFT positions only, also Nout entries */
};

/*
 * SLOTS writes the deconvolved spectrum of the kept FFT positions in slot
 * order (position p holds bin reverse_bits(p)); TLS_*_nufft_pswf_finish_slots()
 * turns such a block back into frequency order.  SLOTS is only honoured when
 * the plan found the periodic kept-bin pattern, which
 * TLS_*_nufft_pswf_slots_available() reports.
 */
#define TLS_INTERNAL_PSWF_DECL(export_attr, prefix, type, input_type, bvec_type, plan_type, lra_plan_type)                                                     \
    export_attr plan_type *prefix##_nufft_pswf_initialize_mode(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2], int spread_mode);  \
    export_attr plan_type *prefix##_nufft_pswf_initialize_opts(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2], int spread_mode,   \
                                                               int vec_mode);                                                                                  \
    export_attr void prefix##_nufft_pswf_execute_mode(plan_type *plan, const type *y_real, const type *y_imag, type *out_real, type *out_imag,                 \
                                                      int freq_factor, int mode);                                                                              \
    export_attr void prefix##_nufft_pswf_execute_breadth(plan_type *plan, const input_type *x, const type *val, double f0, int block, int block_base,          \
                                                         int freq_factor, int mode, bvec_type *out_real, bvec_type *out_imag);                                 \
    export_attr int prefix##_nufft_pswf_slots_available(plan_type *plan);                                                                                      \
    export_attr void prefix##_nufft_pswf_finish_slots(plan_type *plan, const type *compact, type *out, int count);                                             \
    export_attr void prefix##_nufft_pswf_finish_slots_breadth(plan_type *plan, const bvec_type *compact, type *out, int block, int active_lanes,               \
                                                              int total_remaining);                                                                            \
    export_attr lra_plan_type *prefix##_nufft_lra_initialize_opts(int max_Mpoints, int max_N, int max_rank, double df, int freq_factor, int vec_mode);         \
    export_attr void prefix##_nufft_lra_execute_breadth(const lra_plan_type *plan, const input_type *x, const type *val, double f0, int block, int block_base, \
                                                        int freq_factor, bvec_type *out_real, bvec_type *out_imag);

TLS_INTERNAL_PSWF_DECL(TLS_API, tlsf, float, double, VECF, tlsf_nufft_pswf_plan, tlsf_nufft_lra_plan)
TLS_INTERNAL_PSWF_DECL(TLS_API, tls, double, double, VEC, tls_nufft_pswf_plan, tls_nufft_lra_plan)
TLS_INTERNAL_PSWF_DECL(TLS_API, tlsdd, dd_t, dd_t, VECDD, tlsdd_nufft_pswf_plan, tlsdd_nufft_lra_plan)
TLS_INTERNAL_PSWF_DECL(TLS_API, tlsh, tls_f16_t, double, VECH, tlsh_nufft_pswf_plan, tlsh_nufft_lra_plan)
#ifdef HALF_NATIVE_EMBED
TLS_INTERNAL_PSWF_DECL(, tlsh_nat, tls_f16_t, double, VECH, tlsh_nat_nufft_pswf_plan, tlsh_nat_nufft_lra_plan)
#endif

#undef TLS_INTERNAL_PSWF_DECL

TLS_API int tlsf_nufft_pswf_bench_ppoly_gen(int w, const char upsamp[2], int reps);
TLS_API int tls_nufft_pswf_bench_ppoly_gen(int w, const char upsamp[2], int reps);
TLS_API int tlsh_nufft_pswf_bench_ppoly_gen(int w, const char upsamp[2], int reps);
#ifdef HALF_NATIVE_EMBED
int tlsh_nat_nufft_pswf_bench_ppoly_gen(int w, const char upsamp[2], int reps);
#endif

TLS_API void tlsh_nufft_pswf_finish_slots_f32(tlsh_nufft_pswf_plan *plan, const float *compact, float *out, int count);
TLS_API void tlsh_nufft_pswf_finish_slots_breadth_f32(tlsh_nufft_pswf_plan *plan, const float *compact, float *out, int block, int active_lanes,
                                                      int total_remaining);
#ifdef HALF_NATIVE_EMBED
void tlsh_nat_nufft_pswf_finish_slots_f32(tlsh_nat_nufft_pswf_plan *plan, const float *compact, float *out, int count);
void tlsh_nat_nufft_pswf_finish_slots_breadth_f32(tlsh_nat_nufft_pswf_plan *plan, const float *compact, float *out, int block, int active_lanes,
                                                  int total_remaining);
#endif

#if defined(HALF_NATIVE_EMBED) || defined(TLS_HALF_HAS_NATIVE_EMBED)
int tlsh_nat_fastchi2_opts(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                           int normalization, int spreading, int vectorize, float *power, float *cond);
int tlsh_nat_fastchi2_opts_d(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                             int normalization, int spreading, int vectorize, float *power, float *cond);
int tlsh_nat_ihs_opts(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                      int spreading, int vectorize, float *power);
int tlsh_nat_ihs_opts_d(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                        int spreading, int vectorize, float *power);
#endif

/* chi2per reports the size of the work arrays it allocated (benchmark stats). */
TLS_API void tls_tuning_report_work_bytes(unsigned long long bytes);

#ifdef __cplusplus
}
#endif

#endif /* TLS_INTERNAL_H */
