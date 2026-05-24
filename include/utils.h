#ifndef TLS_UTILS_H
#define TLS_UTILS_H

#include <nanofft.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { TLS_UTIL_OK = 0, TLS_UTIL_ERR_ARGUMENT = -1, TLS_UTIL_ERR_ALLOC = -3 };

double tls_approximate_cost(int N, int M, int block, int degree, double alpha, double beta, double gamma, int backend);
int tls_optimize_plan_size(int N, int M, int degree, double alpha, double beta, double gamma, int backend);
int tls_pswf43_plan_len_from_base(int base_len);
int tls_pswf43_output_len_for_plan(int plan_len);
int tls_twiddle_ladder_levels(int N, int block);
int tls_twiddle_ladder_carry_level(size_t next_block, int levels);
double tls_twiddle_ladder_advance(int block, int level);

int tlsf_get_peaks(const float *freq, const float *power, const float *cond, int n, float threshold, float *out_freq, float *out_power, float *out_cond,
                   int *out_count);
int tls_get_peaks(const double *freq, const double *power, const double *cond, int n, double threshold, double *out_freq, double *out_power,
                  double *out_cond, int *out_count);
int tlsdd_get_peaks(const dd_t *freq, const dd_t *power, const dd_t *cond, int n, dd_t threshold, dd_t *out_freq, dd_t *out_power, dd_t *out_cond,
                    int *out_count);

#ifdef __cplusplus
}
#endif

#endif /* TLS_UTILS_H */

#if defined(FLOAT) && defined(FCAST) && defined(ADD) && defined(SUB) && defined(MUL) && !defined(NANOFFT_TRIPLE_ANGLE_DEFINED)
#    define NANOFFT_TRIPLE_ANGLE_DEFINED
static inline void nanofft_triple_angle(FLOAT c, FLOAT s, FLOAT *c3, FLOAT *s3) {
    *c3 = SUB(MUL(FCAST(4.0), MUL(MUL(c, c), c)), MUL(FCAST(3.0), c));
    *s3 = SUB(MUL(FCAST(3.0), s), MUL(FCAST(4.0), MUL(MUL(s, s), s)));
}
#endif
