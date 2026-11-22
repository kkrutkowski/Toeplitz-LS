/* =============================================================================
 * linalg.h  —  Public API for the precision-generic Toeplitz solvers.
 *
 * Declarations for all three precision variants produced by linalg.c. All
 * three variants take SIMD vector batches (VECF, VEC, VECDD), solving one
 * independent system per SIMD lane.
 * =============================================================================
 */

#ifndef LINALG_H
#define LINALG_H

#include <nanofft.h> /* dd_t */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * float  (tlsf_)
 * ========================================================================= */

/** Dense real LDLT solver, single precision vector batch.
 *  Solves full row-major symmetric systems A x = b.
 *  Returns lane-wise Dmax/Dmin from the LDLT diagonal.
 *  Workspace: L length n*n; D, y, z length n each. */
TLS_API VECF tlsf_solve_ldlt(size_t n, const VECF *A, const VECF *b, VECF *x, VECF *L, VECF *D, VECF *y, VECF *z);

/** Levinson-Durbin recursion, single precision vector batch.
 *  Solves the Hermitian Toeplitz system T x = y.
 *  Returns a lane-wise upper condition bound from the reflection coefficients.
 *  Workspace: a_r, a_i — caller-owned, each length n; a_prev_r, a_prev_i are
 *  unused (retained for API compatibility; may be NULL). */
TLS_API VECF tlsf_solve_levinson(size_t n, const VECF *R_r, const VECF *R_i, const VECF *y_r, const VECF *y_i, VECF *x_r, VECF *x_i, VECF *a_r, VECF *a_i,
                                 VECF *a_prev_r, VECF *a_prev_i);

/** Bareiss fast Cholesky solver, single precision vector batch.
 *  Returns the same lane-wise reflection-coefficient condition bound used by
 *  Levinson. Workspace: U_r, U_i length n*n; D, u_r, u_i, v_r, v_i, w_r, w_i
 *  length n each. */
TLS_API VECF tlsf_solve_bareiss(size_t n, const VECF *R_r, const VECF *R_i, const VECF *y_r, const VECF *y_i, VECF *x_r, VECF *x_i, VECF *U_r, VECF *U_i,
                                VECF *D, VECF *u_r, VECF *u_i, VECF *v_r, VECF *v_i, VECF *w_r, VECF *w_i);

/** Zohar recursion, single precision vector batch.
 *  Returns lane-wise zeros.
 *  Workspace: e_hat_r, e_hat_i, e_hat_prev_r, e_hat_prev_i — length n each.
 *  Note: R must have at least n+1 elements allocated (guard slot). */
TLS_API VECF tlsf_solve_zohar(size_t n, const VECF *R_r, const VECF *R_i, const VECF *y_r, const VECF *y_i, VECF *s_r, VECF *s_i, VECF *e_hat_r, VECF *e_hat_i,
                              VECF *e_hat_prev_r, VECF *e_hat_prev_i);

/** Symmetric Jacobi spectral solver, single precision vector batch.
 *  Solves full row-major symmetric systems A x = b.
 *  Returns lane-wise spectral condition estimates.
 *  Workspace: Q, S length n*n; D, y length n each. */
TLS_API VECF tlsf_solve_svd(size_t n, const VECF *A, const VECF *b, VECF *x, VECF *Q, VECF *S, VECF *D, VECF *y, VECF max_cond);

/* =========================================================================
 * double  (tls_)
 * ========================================================================= */

TLS_API VEC tls_solve_ldlt(size_t n, const VEC *A, const VEC *b, VEC *x, VEC *L, VEC *D, VEC *y, VEC *z);

TLS_API VEC tls_solve_levinson(size_t n, const VEC *R_r, const VEC *R_i, const VEC *y_r, const VEC *y_i, VEC *x_r, VEC *x_i, VEC *a_r, VEC *a_i, VEC *a_prev_r,
                               VEC *a_prev_i);

TLS_API VEC tls_solve_bareiss(size_t n, const VEC *R_r, const VEC *R_i, const VEC *y_r, const VEC *y_i, VEC *x_r, VEC *x_i, VEC *U_r, VEC *U_i, VEC *D,
                              VEC *u_r, VEC *u_i, VEC *v_r, VEC *v_i, VEC *w_r, VEC *w_i);

TLS_API VEC tls_solve_zohar(size_t n, const VEC *R_r, const VEC *R_i, const VEC *y_r, const VEC *y_i, VEC *s_r, VEC *s_i, VEC *e_hat_r, VEC *e_hat_i,
                            VEC *e_hat_prev_r, VEC *e_hat_prev_i);

TLS_API VEC tls_solve_svd(size_t n, const VEC *A, const VEC *b, VEC *x, VEC *Q, VEC *S, VEC *D, VEC *y, VEC max_cond);

/* =========================================================================
 * double-double  (tlsdd_, SIMD VECDD batches)
 * ========================================================================= */

TLS_API VECDD tlsdd_solve_ldlt(size_t n, const VECDD *A, const VECDD *b, VECDD *x, VECDD *L, VECDD *D, VECDD *y, VECDD *z);

TLS_API VECDD tlsdd_solve_levinson(size_t n, const VECDD *R_r, const VECDD *R_i, const VECDD *y_r, const VECDD *y_i, VECDD *x_r, VECDD *x_i, VECDD *a_r,
                                   VECDD *a_i, VECDD *a_prev_r, VECDD *a_prev_i);

TLS_API VECDD tlsdd_solve_bareiss(size_t n, const VECDD *R_r, const VECDD *R_i, const VECDD *y_r, const VECDD *y_i, VECDD *x_r, VECDD *x_i, VECDD *U_r,
                                  VECDD *U_i, VECDD *D, VECDD *u_r, VECDD *u_i, VECDD *v_r, VECDD *v_i, VECDD *w_r, VECDD *w_i);

TLS_API VECDD tlsdd_solve_zohar(size_t n, const VECDD *R_r, const VECDD *R_i, const VECDD *y_r, const VECDD *y_i, VECDD *s_r, VECDD *s_i, VECDD *e_hat_r,
                                VECDD *e_hat_i, VECDD *e_hat_prev_r, VECDD *e_hat_prev_i);

TLS_API VECDD tlsdd_solve_svd(size_t n, const VECDD *A, const VECDD *b, VECDD *x, VECDD *Q, VECDD *S, VECDD *D, VECDD *y, VECDD max_cond);
#ifdef __cplusplus
}
#endif

#endif /* LINALG_H */
