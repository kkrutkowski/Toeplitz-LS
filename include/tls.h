#ifndef TLS_H
#define TLS_H

#include <nanofft.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { TLS_OK = 0, TLS_ERR_ARGUMENT = -1, TLS_ERR_BACKEND = -2, TLS_ERR_ALLOC = -3, TLS_ERR_DEGENERATE = -4, TLS_ERR_SOLVER = -5 };

enum { TLS_BACKEND_PSWF43 = 1, TLS_BACKEND_PSWF21 = 2, TLS_BACKEND_LRA = 3 };

enum { TLS_SOLVER_LEVINSON = 1, TLS_SOLVER_ZOHAR = 2, TLS_SOLVER_BAREISS = 3, TLS_SOLVER_LDLT = 4, TLS_SOLVER_SVD = 5 };

/*
 * Periodogram output normalizations.
 *
 * TLS_NORM_STANDARD     coefficient of determination R^2.
 * TLS_NORM_ANOVA        ANOVA / F statistic, after Schwarzenberg-Czerny.
 * TLS_NORM_NLL          negative log false-alarm probability, natural log.
 * TLS_NORM_REGULARIZED  Laplace-regularized Cybenko statistic.
 *
 * The Toeplitz solvers report the condition upper bound n * P, with
 * n = 2 * degree + 1 and P the Cybenko reflection-coefficient product.
 * TLS_NORM_REGULARIZED uses P, not n * P.  For degree > 1 it requires one of
 * the Toeplitz solvers, since it needs the reflection-coefficient product.
 *
 * TLS_NORM_ASYMPTOTIC and TLS_NORM_BAYES are legacy aliases.
 */
enum {
    TLS_NORM_STANDARD = 0,
    TLS_NORM_ANOVA = 1,
    TLS_NORM_NLL = 2,
    TLS_NORM_REGULARIZED = 3,
    TLS_NORM_ASYMPTOTIC = TLS_NORM_ANOVA,
    TLS_NORM_BAYES = TLS_NORM_REGULARIZED
};

int tlsf_fastchi2(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int solver, int normalization,
                  float *power, float *cond);

int tls_fastchi2(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int solver, int normalization,
                 double *power, double *cond);

int tlsdd_fastchi2(const dd_t *t, const dd_t *y, const dd_t *dy, int M, double f0, double df, int N, int degree, int backend, int solver, int normalization,
                   dd_t *power, dd_t *cond);

/*
 * Incoherent harmonic sum (IHS) periodogram.
 *
 * The IHS path uses one NuFFT per harmonic (degree NuFFTs total) and
 * accumulates |sum_m w_m y_m exp(-2 pi i k f t_m)|^2 directly.  It does not
 * build or solve the Toeplitz normal equations.  Only two normalizations are
 * supported:
 *
 * TLS_NORM_STANDARD  sum of the uncorrelated per-harmonic R^2 estimates:
 *                    2 * sum_k |Z_k(f)|^2 / (sum_m w_m * chi2_ref).
 * TLS_NORM_NLL       IHS NLL: each harmonic power is normalized to the
 *                    Rayleigh scale and passed through Wilkie's finite-sample
 *                    correction before the harmonics are summed.  The sum is
 *                    then passed through the Erlang logarithmic correction
 *                    sum - ln(sum_{k=0}^{degree-1} sum^k / k!).
 *
 * Both entry points return only the power array; IHS has no Toeplitz
 * condition bound.
 */
int tlsf_ihs(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int normalization, float *power);

int tls_ihs(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int normalization, double *power);

int tlsdd_ihs(const dd_t *t, const dd_t *y, const dd_t *dy, int M, double f0, double df, int N, int degree, int backend, int normalization, dd_t *power);

#ifdef __cplusplus
}
#endif

#endif /* TLS_H */
