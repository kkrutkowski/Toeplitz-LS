/* =============================================================================
 * linalg.h  —  Public API for the precision-generic Toeplitz solvers.
 *
 * Declarations for all three precision variants produced by linalg.c. The
 * float and double variants take GCC vector-extension batches, solving one
 * independent system per SIMD lane. The double-double variant is scalar
 * because its INTERNAL_VEC_LEN is 1.
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

/** Levinson-Durbin recursion, single precision vector batch.
 *  Solves the Hermitian Toeplitz system T x = y.
 *  Returns a lane-wise upper condition bound from the reflection coefficients.
 *  Workspace: a_r, a_i, a_prev_r, a_prev_i — caller-owned, each length n. */
VECF tlsf_solve_levinson(size_t n, const VECF *R_r, const VECF *R_i, const VECF *y_r, const VECF *y_i, VECF *x_r, VECF *x_i, VECF *a_r, VECF *a_i,
                         VECF *a_prev_r, VECF *a_prev_i);

/** Zohar recursion, single precision vector batch.
 *  Returns lane-wise zeros.
 *  Workspace: e_hat_r, e_hat_i, e_hat_prev_r, e_hat_prev_i — length n each.
 *  Note: R must have at least n+1 elements allocated (guard slot). */
VECF tlsf_solve_zohar(size_t n, const VECF *R_r, const VECF *R_i, const VECF *y_r, const VECF *y_i, VECF *s_r, VECF *s_i, VECF *e_hat_r, VECF *e_hat_i,
                      VECF *e_hat_prev_r, VECF *e_hat_prev_i);

/* =========================================================================
 * double  (tls_)
 * ========================================================================= */

VEC tls_solve_levinson(size_t n, const VEC *R_r, const VEC *R_i, const VEC *y_r, const VEC *y_i, VEC *x_r, VEC *x_i, VEC *a_r, VEC *a_i, VEC *a_prev_r,
                       VEC *a_prev_i);

VEC tls_solve_zohar(size_t n, const VEC *R_r, const VEC *R_i, const VEC *y_r, const VEC *y_i, VEC *s_r, VEC *s_i, VEC *e_hat_r, VEC *e_hat_i, VEC *e_hat_prev_r,
                    VEC *e_hat_prev_i);

/* =========================================================================
 * double-double  (tlsdd_)
 * ========================================================================= */

dd_t tlsdd_solve_levinson(size_t n, const dd_t *R_r, const dd_t *R_i, const dd_t *y_r, const dd_t *y_i, dd_t *x_r, dd_t *x_i, dd_t *a_r, dd_t *a_i,
                          dd_t *a_prev_r, dd_t *a_prev_i);

dd_t tlsdd_solve_zohar(size_t n, const dd_t *R_r, const dd_t *R_i, const dd_t *y_r, const dd_t *y_i, dd_t *s_r, dd_t *s_i, dd_t *e_hat_r, dd_t *e_hat_i,
                       dd_t *e_hat_prev_r, dd_t *e_hat_prev_i);

#ifdef __cplusplus
}
#endif

#endif /* LINALG_H */
