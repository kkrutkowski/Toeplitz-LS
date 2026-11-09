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
 * TLS_NORM_REGULARIZED  internal/reserved in 1.0; not part of the documented
 *                       public normalization set.
 * TLS_NORM_IHS_DEFAULT  IHS-only: Erlang correction without the finite-sample
 *                       Wilkie correction (the default IHS statistic).
 *
 * The Toeplitz solvers report the condition upper bound n * P, with
 * n = 2 * degree + 1 and P the Cybenko reflection-coefficient product.
 *
 * TLS_NORM_ASYMPTOTIC and TLS_NORM_BAYES are legacy aliases.
 * TLS_NORM_IHS_RAW and TLS_NORM_IHS_NLL are the IHS-only aliases.
 */
enum {
    TLS_NORM_STANDARD = 0,
    TLS_NORM_ANOVA = 1,
    TLS_NORM_NLL = 2,
    TLS_NORM_REGULARIZED = 3,
    TLS_NORM_IHS_DEFAULT = 4,
    TLS_NORM_ASYMPTOTIC = TLS_NORM_ANOVA,
    TLS_NORM_BAYES = TLS_NORM_REGULARIZED,
    TLS_NORM_IHS_RAW = TLS_NORM_STANDARD,
    TLS_NORM_IHS_NLL = TLS_NORM_NLL
};

TLS_API int tlsf_fastchi2(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                          int normalization, float *power, float *cond);

TLS_API int tls_fastchi2(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                         int normalization, double *power, double *cond);

TLS_API int tlsdd_fastchi2(const dd_t *t, const dd_t *y, const dd_t *dy, int M, double f0, double df, int N, int degree, int backend, int solver,
                           int normalization, dd_t *power, dd_t *cond);

/*
 * Incoherent harmonic sum (IHS) periodogram.
 *
 * The IHS path uses one NuFFT per harmonic (degree NuFFTs total) and
 * accumulates |sum_m w_m y_m exp(-2 pi i k f t_m)|^2 directly.  It does not
 * build or solve the Toeplitz normal equations.  Three normalizations are
 * supported:
 *
 * TLS_NORM_IHS_RAW        sum of the normalized per-harmonic Rayleigh powers
 *                         (no false-alarm calibration).
 * TLS_NORM_IHS_NLL        per-harmonic Wilkie finite-sample correction,
 *                         followed by the Erlang correction.
 * TLS_NORM_IHS_DEFAULT    Erlang correction only; the finite-sample correction
 *                         is intentionally omitted.
 *
 * All three entry points return only the power array; IHS has no Toeplitz
 * condition bound.
 */
TLS_API int tlsf_ihs(const double *t, const float *y, const float *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                     float *power);

TLS_API int tls_ihs(const double *t, const double *y, const double *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                    double *power);

TLS_API int tlsdd_ihs(const dd_t *t, const dd_t *y, const dd_t *dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                      dd_t *power);

#ifdef __cplusplus
}
#endif

#endif /* TLS_H */
