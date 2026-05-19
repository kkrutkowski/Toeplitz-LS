#ifndef TLS_H
#define TLS_H

#include <nanofft.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { TLS_OK = 0, TLS_ERR_ARGUMENT = -1, TLS_ERR_BACKEND = -2, TLS_ERR_ALLOC = -3, TLS_ERR_DEGENERATE = -4, TLS_ERR_SOLVER = -5 };

enum { TLS_BACKEND_PSWF43 = 1, TLS_BACKEND_PSWF21 = 2, TLS_BACKEND_LRA = 3 };

enum { TLS_SOLVER_LEVINSON = 1, TLS_SOLVER_ZOHAR = 2, TLS_SOLVER_BAREISS = 3, TLS_SOLVER_LDLT = 4 };

int tlsf_fastchi2(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int solver, float *power,
                  float *cond);

int tls_fastchi2(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int solver, double *power,
                 double *cond);

int tlsdd_fastchi2(const dd_t *t, const dd_t *y, const dd_t *dy, int M, double f0, double df, int N, int degree, int backend, int solver, dd_t *power,
                   dd_t *cond);

#ifdef __cplusplus
}
#endif

#endif /* TLS_H */
