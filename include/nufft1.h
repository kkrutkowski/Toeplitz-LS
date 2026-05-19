#ifndef NUFFT1_H
#define NUFFT1_H

#include <nanofft.h>
#include <stdint.h>

/* =========================================================================
 * Single Precision API (float)
 * ========================================================================= */

typedef struct tlsf_nufft_lra_plan tlsf_nufft_lra_plan;

tlsf_nufft_lra_plan *tlsf_nufft_lra_initialize(int max_Mpoints, int max_N, int max_rank, double df, int freq_factor);
void tlsf_nufft_lra_precompute(tlsf_nufft_lra_plan *plan, const double *x, int Mpoints, int N, int rank);
void tlsf_nufft_lra_execute(const tlsf_nufft_lra_plan *plan, const float *y_real, const float *y_imag, float *out_real, float *out_imag, int freq_factor);
void tlsf_nufft_free_lra_plan(tlsf_nufft_lra_plan *plan);

typedef struct tlsf_nufft_pswf_plan tlsf_nufft_pswf_plan;

tlsf_nufft_pswf_plan *tlsf_nufft_pswf_initialize(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2]);
void tlsf_nufft_pswf_precompute(tlsf_nufft_pswf_plan *plan, const double *x);
void tlsf_nufft_pswf_execute(tlsf_nufft_pswf_plan *plan, const float *y_real, const float *y_imag, float *out_real, float *out_imag, int freq_factor);
void tlsf_nufft_free_pswf_plan(tlsf_nufft_pswf_plan *plan);

/* =========================================================================
 * Double Precision API (double)
 * ========================================================================= */

typedef struct tls_nufft_lra_plan tls_nufft_lra_plan;

tls_nufft_lra_plan *tls_nufft_lra_initialize(int max_Mpoints, int max_N, int max_rank, double df, int freq_factor);
void tls_nufft_lra_precompute(tls_nufft_lra_plan *plan, const double *x, int Mpoints, int N, int rank);
void tls_nufft_lra_execute(const tls_nufft_lra_plan *plan, const double *y_real, const double *y_imag, double *out_real, double *out_imag, int freq_factor);
void tls_nufft_free_lra_plan(tls_nufft_lra_plan *plan);

typedef struct tls_nufft_pswf_plan tls_nufft_pswf_plan;

tls_nufft_pswf_plan *tls_nufft_pswf_initialize(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2]);
void tls_nufft_pswf_precompute(tls_nufft_pswf_plan *plan, const double *x);
void tls_nufft_pswf_execute(tls_nufft_pswf_plan *plan, const double *y_real, const double *y_imag, double *out_real, double *out_imag, int freq_factor);
void tls_nufft_free_pswf_plan(tls_nufft_pswf_plan *plan);

/* =========================================================================
 * Double-Double Precision API (dd_t)
 * ========================================================================= */

typedef struct tlsdd_nufft_lra_plan tlsdd_nufft_lra_plan;

tlsdd_nufft_lra_plan *tlsdd_nufft_lra_initialize(int max_Mpoints, int max_N, int max_rank, double df, int freq_factor);
void tlsdd_nufft_lra_precompute(tlsdd_nufft_lra_plan *plan, const dd_t *x, int Mpoints, int N, int rank);
void tlsdd_nufft_lra_execute(const tlsdd_nufft_lra_plan *plan, const dd_t *y_real, const dd_t *y_imag, dd_t *out_real, dd_t *out_imag, int freq_factor);
void tlsdd_nufft_free_lra_plan(tlsdd_nufft_lra_plan *plan);

typedef struct tlsdd_nufft_pswf_plan tlsdd_nufft_pswf_plan;

tlsdd_nufft_pswf_plan *tlsdd_nufft_pswf_initialize(int Mpoints, int N, int w, double df, int freq_factor, const char upsamp[2]);
void tlsdd_nufft_pswf_precompute(tlsdd_nufft_pswf_plan *plan, const dd_t *x);
void tlsdd_nufft_pswf_execute(tlsdd_nufft_pswf_plan *plan, const dd_t *y_real, const dd_t *y_imag, dd_t *out_real, dd_t *out_imag, int freq_factor);
void tlsdd_nufft_free_pswf_plan(tlsdd_nufft_pswf_plan *plan);

#endif /* NUFFT1_H */
