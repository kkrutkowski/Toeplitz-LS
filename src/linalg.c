// linalg.c  —  Precision-generic LDLT, SVD, Levinson-Durbin, Bareiss, and Zohar Toeplitz solvers

/* nanofft.h provides: dd_t and its arithmetic (dd_add/dd_sub/dd_mul/dd_div …),
 * VEC / VECF GCC vector types, VEC_LEN / VECF_LEN, and all the fast-trig
 * helpers that nanofft_precision.h macros reference internally.            */
#include <nanofft.h>
#include <stddef.h>

/* nanofft_precision.h provides – based on DOUBLE / DOUBLE_DOUBLE defines –:
 *   INTERNAL_VEC lane-batched type (VECF / VEC / dd_t)
 *   ADD SUB MUL  arithmetic on INTERNAL_VEC
 *   DIV NEG
 *   FCONST(x)    compile-time FLOAT literal
 *   FCAST(x)     runtime float  → FLOAT conversion                         */

#define NANOFFT_NEEDS_INTERNAL_VEC
#include <nanofft_precision.h>

/* -------------------------------------------------------------------------
 * tls_ naming family, separate from the nanofft_ family used by the FFT.
 * ------------------------------------------------------------------------- */
#if defined(DOUBLE_DOUBLE)
#    define TLS(name) tlsdd_##name
#elif defined(DOUBLE)
#    define TLS(name) tls_##name
#else
#    define TLS(name) tlsf_##name
#endif

#if defined(DOUBLE_DOUBLE)
#    define VCONST(x) FCONST(x)
static inline INTERNAL_VEC TLS(reflection_abs)(INTERNAL_VEC abs2) { return FCAST(sqrt(TO_DOUBLE(abs2))); }
#else
static inline INTERNAL_VEC TLS(vec_splat)(FLOAT value) {
    INTERNAL_VEC v;
    for (int lane = 0; lane < INTERNAL_VEC_LEN; lane++) v[lane] = value;
    return v;
}
#    define VCONST(x) TLS(vec_splat)(FCONST(x))
static inline INTERNAL_VEC TLS(reflection_abs)(INTERNAL_VEC abs2) { return M_SQRT(abs2); }
#endif

static inline INTERNAL_VEC TLS(reflection_condition_step)(INTERNAL_VEC abs2) {
    INTERNAL_VEC abs_reflection = TLS(reflection_abs)(abs2);
    return DIV(ADD(VCONST(1.0), abs_reflection), SUB(VCONST(1.0), abs_reflection));
}

static inline INTERNAL_VEC TLS(ldlt_diag_condition)(size_t n, const INTERNAL_VEC *restrict D) {
#if defined(DOUBLE_DOUBLE)
    INTERNAL_VEC dmin = D[0];
    INTERNAL_VEC dmax = D[0];
    for (size_t i = 1; i < n; i++) {
        if (TO_DOUBLE(D[i]) < TO_DOUBLE(dmin)) dmin = D[i];
        if (TO_DOUBLE(D[i]) > TO_DOUBLE(dmax)) dmax = D[i];
    }
    return DIV(dmax, dmin);
#else
    INTERNAL_VEC ratio;
    for (int lane = 0; lane < INTERNAL_VEC_LEN; lane++) {
        FLOAT dmin = D[0][lane];
        FLOAT dmax = D[0][lane];
        for (size_t i = 1; i < n; i++) {
            FLOAT d = D[i][lane];
            if (d < dmin) dmin = d;
            if (d > dmax) dmax = d;
        }
        ratio[lane] = dmax / dmin;
    }
    return ratio;
#endif
}

static inline FLOAT TLS(scalar_abs)(FLOAT value) { return TO_DOUBLE(value) < 0.0 ? NEG(value) : value; }

#if defined(DOUBLE_DOUBLE)
static inline FLOAT TLS(lane_get)(INTERNAL_VEC value, int lane) {
    (void)lane;
    return value;
}

static inline void TLS(lane_set)(INTERNAL_VEC *value, int lane, FLOAT lane_value) {
    (void)lane;
    *value = lane_value;
}
#else
static inline FLOAT TLS(lane_get)(INTERNAL_VEC value, int lane) { return value[lane]; }

static inline void TLS(lane_set)(INTERNAL_VEC *value, int lane, FLOAT lane_value) { (*value)[lane] = lane_value; }
#endif

static inline double TLS(svd_tolerance)(void) {
#if defined(DOUBLE_DOUBLE)
    return 1e-24;
#elif defined(DOUBLE)
    return 1e-12;
#else
    return 1e-5;
#endif
}

static inline int TLS(svd_max_sweeps)(void) {
#if defined(DOUBLE_DOUBLE)
    return 256;
#elif defined(DOUBLE)
    return 64;
#else
    return 32;
#endif
}

/* =========================================================================
 * tlsf/tls/tlsdd_solve_ldlt
 *
 * Dense real LDLT decomposition for full symmetric positive-definite systems
 * A x = b. A is row-major, with one independent real system per lane.
 *
 * Workspace (caller-allocated)
 *   L      unit-lower-triangular factor, length n*n
 *   D      diagonal, length n
 *   y, z   substitution workspaces, length n each
 *
 * Returns lane-wise Dmax/Dmin from the LDLT diagonal.
 * ========================================================================= */
INTERNAL_VEC TLS(solve_ldlt)(size_t n, const INTERNAL_VEC *restrict A, const INTERNAL_VEC *restrict b, INTERNAL_VEC *restrict x, INTERNAL_VEC *restrict L,
                             INTERNAL_VEC *restrict D, INTERNAL_VEC *restrict y, INTERNAL_VEC *restrict z) {
    for (size_t i = 0; i < n; i++) {
        size_t i_row_offset = i * n;
        for (size_t j = 0; j < i; j++) {
            size_t j_row_offset = j * n;
            INTERNAL_VEC sum = VCONST(0.0);
            for (size_t k = 0; k < j; k++) {
                sum = ADD(sum, MUL(MUL(L[i_row_offset + k], L[j_row_offset + k]), D[k]));
            }
            L[i_row_offset + j] = DIV(SUB(A[i_row_offset + j], sum), D[j]);
        }

        INTERNAL_VEC sum2 = VCONST(0.0);
        for (size_t k = 0; k < i; k++) {
            sum2 = ADD(sum2, MUL(MUL(L[i_row_offset + k], L[i_row_offset + k]), D[k]));
        }
        D[i] = SUB(A[i_row_offset + i], sum2);
        L[i_row_offset + i] = VCONST(1.0);
    }

    INTERNAL_VEC condition = TLS(ldlt_diag_condition)(n, D);

    for (size_t i = 0; i < n; i++) {
        size_t row_offset = i * n;
        INTERNAL_VEC sum = VCONST(0.0);
        for (size_t j = 0; j < i; j++) {
            sum = ADD(sum, MUL(L[row_offset + j], y[j]));
        }
        y[i] = SUB(b[i], sum);
    }

    for (size_t i = 0; i < n; i++) {
        z[i] = DIV(y[i], D[i]);
    }

    for (size_t i = n; i-- > 0;) {
        INTERNAL_VEC sum = VCONST(0.0);
        for (size_t j = i + 1; j < n; j++) {
            sum = ADD(sum, MUL(L[j * n + i], x[j]));
        }
        x[i] = SUB(z[i], sum);
    }

    return condition;
}

/* =========================================================================
 * tlsf/tls/tlsdd_solve_svd
 *
 * Symmetric Jacobi spectral solver for dense real symmetric systems A x = b.
 * A is row-major, with one independent real system per lane.
 *
 * Workspace (caller-allocated)
 *   Q       eigenvectors, length n*n
 *   S       mutable copy of A, length n*n
 *   D       eigenvalues, length n
 *   y       projected right-hand side, length n
 *
 * If max_cond > 0, negative eigenvalues and eigenvalues below
 * max_positive_eigenvalue / max_cond are filtered from the inverse.
 * If max_cond <= 0, every nonzero signed eigenvalue is inverted.
 *
 * Returns lane-wise spectral condition estimates. A negative returned
 * condition indicates that the Jacobi sweeps did not fully converge for that
 * lane within the precision-specific sweep limit.
 * ========================================================================= */
INTERNAL_VEC TLS(solve_svd)(size_t n, const INTERNAL_VEC *restrict A, const INTERNAL_VEC *restrict b, INTERNAL_VEC *restrict x, INTERNAL_VEC *restrict Q,
                            INTERNAL_VEC *restrict S, INTERNAL_VEC *restrict D, INTERNAL_VEC *restrict y, INTERNAL_VEC max_cond) {
    for (size_t i = 0; i < n * n; i++) {
        S[i] = A[i];
        Q[i] = VCONST(0.0);
    }
    for (size_t i = 0; i < n; i++) Q[i * n + i] = VCONST(1.0);

    double tol = TLS(svd_tolerance)();
    int max_sweeps = TLS(svd_max_sweeps)();
    int all_converged = 1;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        all_converged = 1;

        for (size_t p = 0; p < n; p++) {
            for (size_t q = p + 1; q < n; q++) {
                size_t pp = p * n + p;
                size_t qq = q * n + q;
                size_t pq = p * n + q;

                for (int lane = 0; lane < INTERNAL_VEC_LEN; lane++) {
                    FLOAT app = TLS(lane_get)(S[pp], lane);
                    FLOAT aqq = TLS(lane_get)(S[qq], lane);
                    FLOAT apq = TLS(lane_get)(S[pq], lane);
                    double app_d = TO_DOUBLE(app);
                    double aqq_d = TO_DOUBLE(aqq);
                    double apq_abs = fabs(TO_DOUBLE(apq));
                    double scale = fabs(app_d) + fabs(aqq_d);
                    if (scale == 0.0) scale = 1.0;
                    if (apq_abs <= tol * scale) continue;

                    all_converged = 0;

                    FLOAT tau = DIV(SUB(aqq, app), MUL(FCAST(2.0), apq));
                    FLOAT tau_abs = TLS(scalar_abs)(tau);
                    FLOAT denom = ADD(tau_abs, M_SQRT(ADD(FCAST(1.0), MUL(tau, tau))));
                    FLOAT t = DIV(FCAST(1.0), denom);
                    if (TO_DOUBLE(tau) < 0.0) t = NEG(t);
                    FLOAT c = DIV(FCAST(1.0), M_SQRT(ADD(FCAST(1.0), MUL(t, t))));
                    FLOAT s = MUL(t, c);
                    FLOAT c2 = MUL(c, c);
                    FLOAT s2 = MUL(s, s);
                    FLOAT cs = MUL(c, s);

                    for (size_t k = 0; k < n; k++) {
                        if (k == p || k == q) continue;

                        size_t kp = k * n + p;
                        size_t kq = k * n + q;
                        size_t pk = p * n + k;
                        size_t qk = q * n + k;
                        FLOAT akp = TLS(lane_get)(S[kp], lane);
                        FLOAT akq = TLS(lane_get)(S[kq], lane);
                        FLOAT new_kp = SUB(MUL(c, akp), MUL(s, akq));
                        FLOAT new_kq = ADD(MUL(s, akp), MUL(c, akq));
                        TLS(lane_set)(&S[kp], lane, new_kp);
                        TLS(lane_set)(&S[pk], lane, new_kp);
                        TLS(lane_set)(&S[kq], lane, new_kq);
                        TLS(lane_set)(&S[qk], lane, new_kq);
                    }

                    FLOAT two_cs_apq = MUL(FCAST(2.0), MUL(cs, apq));
                    FLOAT new_app = ADD(SUB(MUL(c2, app), two_cs_apq), MUL(s2, aqq));
                    FLOAT new_aqq = ADD(ADD(MUL(s2, app), two_cs_apq), MUL(c2, aqq));
                    TLS(lane_set)(&S[pp], lane, new_app);
                    TLS(lane_set)(&S[qq], lane, new_aqq);
                    TLS(lane_set)(&S[pq], lane, FCAST(0.0));
                    TLS(lane_set)(&S[q * n + p], lane, FCAST(0.0));

                    for (size_t k = 0; k < n; k++) {
                        size_t kp = k * n + p;
                        size_t kq = k * n + q;
                        FLOAT qkp = TLS(lane_get)(Q[kp], lane);
                        FLOAT qkq = TLS(lane_get)(Q[kq], lane);
                        TLS(lane_set)(&Q[kp], lane, SUB(MUL(c, qkp), MUL(s, qkq)));
                        TLS(lane_set)(&Q[kq], lane, ADD(MUL(s, qkp), MUL(c, qkq)));
                    }
                }
            }
        }

        if (all_converged) break;
    }

    int lane_converged[INTERNAL_VEC_LEN];
    for (int lane = 0; lane < INTERNAL_VEC_LEN; lane++) lane_converged[lane] = 1;

    for (size_t p = 0; p < n; p++) {
        for (size_t q = p + 1; q < n; q++) {
            size_t pp = p * n + p;
            size_t qq = q * n + q;
            size_t pq = p * n + q;
            for (int lane = 0; lane < INTERNAL_VEC_LEN; lane++) {
                FLOAT app = TLS(lane_get)(S[pp], lane);
                FLOAT aqq = TLS(lane_get)(S[qq], lane);
                FLOAT apq = TLS(lane_get)(S[pq], lane);
                double scale = fabs(TO_DOUBLE(app)) + fabs(TO_DOUBLE(aqq));
                if (scale == 0.0) scale = 1.0;
                if (fabs(TO_DOUBLE(apq)) > tol * scale) lane_converged[lane] = 0;
            }
        }
    }

    INTERNAL_VEC condition = VCONST(0.0);

    for (int lane = 0; lane < INTERNAL_VEC_LEN; lane++) {
        FLOAT max_abs = FCAST(0.0);
        FLOAT min_abs = FCAST(0.0);
        FLOAT max_positive = FCAST(0.0);
        int have_nonzero = 0;

        for (size_t i = 0; i < n; i++) {
            FLOAT lambda = TLS(lane_get)(S[i * n + i], lane);
            FLOAT abs_lambda = TLS(scalar_abs)(lambda);
            double abs_d = TO_DOUBLE(abs_lambda);
            if (TO_DOUBLE(lambda) > TO_DOUBLE(max_positive)) max_positive = lambda;
            if (abs_d > TO_DOUBLE(max_abs)) max_abs = abs_lambda;
            if (abs_d > 0.0 && (!have_nonzero || abs_d < TO_DOUBLE(min_abs))) {
                min_abs = abs_lambda;
                have_nonzero = 1;
            }
            TLS(lane_set)(&D[i], lane, lambda);
        }

        FLOAT lane_condition = have_nonzero ? DIV(max_abs, min_abs) : FCAST(INFINITY);
        if (!lane_converged[lane]) lane_condition = NEG(lane_condition);
        TLS(lane_set)(&condition, lane, lane_condition);

        FLOAT threshold = FCAST(0.0);
        FLOAT lane_max_cond = TLS(lane_get)(max_cond, lane);
        int filtered = TO_DOUBLE(lane_max_cond) > 0.0;
        if (filtered && TO_DOUBLE(max_positive) > 0.0) threshold = DIV(max_positive, lane_max_cond);

        for (size_t i = 0; i < n; i++) {
            FLOAT dot = FCAST(0.0);
            for (size_t j = 0; j < n; j++) {
                dot = ADD(dot, MUL(TLS(lane_get)(Q[j * n + i], lane), TLS(lane_get)(b[j], lane)));
            }

            FLOAT lambda = TLS(lane_get)(D[i], lane);
            FLOAT coeff = FCAST(0.0);
            if (filtered) {
                if (TO_DOUBLE(lambda) > 0.0 && TO_DOUBLE(lambda) >= TO_DOUBLE(threshold)) coeff = DIV(dot, lambda);
            } else if (TO_DOUBLE(lambda) != 0.0) {
                coeff = DIV(dot, lambda);
            }
            TLS(lane_set)(&y[i], lane, coeff);
        }

        for (size_t i = 0; i < n; i++) {
            FLOAT sum = FCAST(0.0);
            for (size_t j = 0; j < n; j++) {
                sum = ADD(sum, MUL(TLS(lane_get)(Q[i * n + j], lane), TLS(lane_get)(y[j], lane)));
            }
            TLS(lane_set)(&x[i], lane, sum);
        }
    }

    return condition;
}

/* =========================================================================
 * tlsf/tls/tlsdd_solve_levinson
 *
 * Levinson-Durbin recursion for INTERNAL_VEC_LEN independent complex
 * Hermitian Toeplitz systems T x = y. Each array element is one vector of
 * lane-wise systems:
 *   T_{ij} = R[|i-j|], R[0] real and positive, R[-k] = conj(R[k]).
 *
 * In / out
 *   R_r, R_i    Toeplitz row, length n
 *   y_r, y_i    right-hand side, length n
 *   x_r, x_i    solution, length n  (written on exit)
 *   return      lane-wise product Π (1 + |gamma_j|) / (1 - |gamma_j|)
 *
 * Workspace (caller-allocated, length n each)
 *   a_r, a_i          predictor polynomial
 *   a_prev_r, a_prev_i  copy from the previous order
 * ========================================================================= */
INTERNAL_VEC TLS(solve_levinson)(size_t n, const INTERNAL_VEC *restrict R_r, const INTERNAL_VEC *restrict R_i, const INTERNAL_VEC *restrict y_r,
                                 const INTERNAL_VEC *restrict y_i, INTERNAL_VEC *restrict x_r, INTERNAL_VEC *restrict x_i, INTERNAL_VEC *restrict a_r,
                                 INTERNAL_VEC *restrict a_i, INTERNAL_VEC *restrict a_prev_r, INTERNAL_VEC *restrict a_prev_i) {
    INTERNAL_VEC E = R_r[0];
    INTERNAL_VEC cond_bound = VCONST(1.0);

    x_r[0] = DIV(y_r[0], E);
    x_i[0] = DIV(y_i[0], E);

    /* a[0] is the leading (unit) coefficient, never modified. */
    a_r[0] = VCONST(1.0);
    a_i[0] = VCONST(0.0);

    for (size_t k = 1; k < n; k++) {
        /* ----- reflection coefficient: lambda = sum_{i=0}^{k-1} R[k-i] a[i] */
        INTERNAL_VEC lambda_r = VCONST(0.0);
        INTERNAL_VEC lambda_i = VCONST(0.0);

        for (size_t i = 0; i < k; i++) {
            INTERNAL_VEC rr = R_r[k - i], ri = R_i[k - i];
            INTERNAL_VEC ar = a_r[i], ai = a_i[i];
            lambda_r = ADD(lambda_r, SUB(MUL(rr, ar), MUL(ri, ai)));
            lambda_i = ADD(lambda_i, ADD(MUL(rr, ai), MUL(ri, ar)));
        }

        INTERNAL_VEC gamma_r = NEG(DIV(lambda_r, E));
        INTERNAL_VEC gamma_i = NEG(DIV(lambda_i, E));

        /* ----- snapshot current predictor before order-update */
        for (size_t i = 0; i < k; i++) {
            a_prev_r[i] = a_r[i];
            a_prev_i[i] = a_i[i];
        }

        /* ----- a[k] = gamma * conj(a_prev[0]) */
        INTERNAL_VEC ap0_r = a_prev_r[0];
        INTERNAL_VEC ap0_i = NEG(a_prev_i[0]);
        a_r[k] = SUB(MUL(gamma_r, ap0_r), MUL(gamma_i, ap0_i));
        a_i[k] = ADD(MUL(gamma_r, ap0_i), MUL(gamma_i, ap0_r));

        /* ----- a[i] += gamma * conj(a_prev[k-i]),  i = 1 … k-1 */
        for (size_t i = 1; i < k; i++) {
            INTERNAL_VEC ap_r = a_prev_r[k - i];
            INTERNAL_VEC ap_i = NEG(a_prev_i[k - i]);
            INTERNAL_VEC tr = SUB(MUL(gamma_r, ap_r), MUL(gamma_i, ap_i));
            INTERNAL_VEC ti = ADD(MUL(gamma_r, ap_i), MUL(gamma_i, ap_r));
            a_r[i] = ADD(a_prev_r[i], tr);
            a_i[i] = ADD(a_prev_i[i], ti);
        }

        /* ----- error energy: E *= (1 - |gamma|^2) */
        INTERNAL_VEC abs2_gamma = ADD(MUL(gamma_r, gamma_r), MUL(gamma_i, gamma_i));
        cond_bound = MUL(cond_bound, TLS(reflection_condition_step)(abs2_gamma));
        E = MUL(E, SUB(VCONST(1.0), abs2_gamma));

        /* ----- RHS residual: mu = y[k] - sum_{i=0}^{k-1} R[k-i] x[i] */
        INTERNAL_VEC mu_r = y_r[k];
        INTERNAL_VEC mu_i = y_i[k];

        for (size_t i = 0; i < k; i++) {
            INTERNAL_VEC rr = R_r[k - i], ri = R_i[k - i];
            INTERNAL_VEC xr = x_r[i], xi = x_i[i];
            mu_r = SUB(mu_r, SUB(MUL(rr, xr), MUL(ri, xi)));
            mu_i = SUB(mu_i, ADD(MUL(rr, xi), MUL(ri, xr)));
        }

        INTERNAL_VEC nu_r = DIV(mu_r, E);
        INTERNAL_VEC nu_i = DIV(mu_i, E);

        /* ----- solution update: x[i] += nu * conj(a[k-i]),  i = 0 … k */
        /* i = k: a[k-k] = a[0] = 1+0j, so x[k] = nu * 1 */
        INTERNAL_VEC a0_r = a_r[0];      /* always 1 */
        INTERNAL_VEC a0_i = NEG(a_i[0]); /* always 0 */
        x_r[k] = SUB(MUL(nu_r, a0_r), MUL(nu_i, a0_i));
        x_i[k] = ADD(MUL(nu_r, a0_i), MUL(nu_i, a0_r));

        for (size_t i = 0; i < k; i++) {
            INTERNAL_VEC ak_r = a_r[k - i];
            INTERNAL_VEC ak_i = NEG(a_i[k - i]);
            INTERNAL_VEC tr = SUB(MUL(nu_r, ak_r), MUL(nu_i, ak_i));
            INTERNAL_VEC ti = ADD(MUL(nu_r, ak_i), MUL(nu_i, ak_r));
            x_r[i] = ADD(x_r[i], tr);
            x_i[i] = ADD(x_i[i], ti);
        }
    }

    return cond_bound;
}

/* =========================================================================
 * tlsf/tls/tlsdd_solve_bareiss
 *
 * Bareiss fast Cholesky factorization for Hermitian Toeplitz systems T x = y.
 * Same system definition as solve_levinson above.
 *
 * Workspace (caller-allocated)
 *   U_r, U_i            upper triangular factor, length n*n each
 *   D                   real diagonal, length n
 *   u_r, u_i, v_r, v_i  Bareiss generators, length n each
 *   w_r, w_i            substitution workspace, length n each
 *
 * Returns the same lane-wise reflection-coefficient condition bound used by
 * solve_levinson.
 * ========================================================================= */
INTERNAL_VEC TLS(solve_bareiss)(size_t n, const INTERNAL_VEC *restrict R_r, const INTERNAL_VEC *restrict R_i, const INTERNAL_VEC *restrict y_r,
                                const INTERNAL_VEC *restrict y_i, INTERNAL_VEC *restrict x_r, INTERNAL_VEC *restrict x_i, INTERNAL_VEC *restrict U_r,
                                INTERNAL_VEC *restrict U_i, INTERNAL_VEC *restrict D, INTERNAL_VEC *restrict u_r, INTERNAL_VEC *restrict u_i,
                                INTERNAL_VEC *restrict v_r, INTERNAL_VEC *restrict v_i, INTERNAL_VEC *restrict w_r, INTERNAL_VEC *restrict w_i) {
    INTERNAL_VEC cond_bound = VCONST(1.0);

    for (size_t j = 0; j < n; j++) {
        u_r[j] = R_r[j];
        u_i[j] = R_i[j];
        if (j < n - 1) {
            v_r[j] = R_r[j + 1];
            v_i[j] = R_i[j + 1];
        } else {
            v_r[j] = VCONST(0.0);
            v_i[j] = VCONST(0.0);
        }
    }

    for (size_t k = 0; k < n; k++) {
        D[k] = u_r[0];

        size_t row_offset = k * n + k;
        for (size_t j = 0; j < n - k; j++) {
            U_r[row_offset + j] = u_r[j];
            U_i[row_offset + j] = u_i[j];
        }

        if (k == n - 1) break;

        INTERNAL_VEC inv_D = DIV(VCONST(1.0), D[k]);
        INTERNAL_VEC rho_r = MUL(v_r[0], inv_D);
        INTERNAL_VEC rho_i = MUL(v_i[0], inv_D);
        INTERNAL_VEC abs2_rho = ADD(MUL(rho_r, rho_r), MUL(rho_i, rho_i));
        cond_bound = MUL(cond_bound, TLS(reflection_condition_step)(abs2_rho));

        for (size_t j = 0; j < n - k - 1; j++) {
            INTERNAL_VEC uj_r = u_r[j];
            INTERNAL_VEC uj_i = u_i[j];
            INTERNAL_VEC vj_r = v_r[j];
            INTERNAL_VEC vj_i = v_i[j];

            INTERNAL_VEC up1_r = u_r[j + 1];
            INTERNAL_VEC up1_i = u_i[j + 1];
            INTERNAL_VEC vp1_r = v_r[j + 1];
            INTERNAL_VEC vp1_i = v_i[j + 1];

            u_r[j] = SUB(uj_r, ADD(MUL(rho_r, vj_r), MUL(rho_i, vj_i)));
            u_i[j] = SUB(uj_i, SUB(MUL(rho_r, vj_i), MUL(rho_i, vj_r)));

            v_r[j] = SUB(vp1_r, SUB(MUL(rho_r, up1_r), MUL(rho_i, up1_i)));
            v_i[j] = SUB(vp1_i, ADD(MUL(rho_r, up1_i), MUL(rho_i, up1_r)));
        }
    }

    for (size_t i = 0; i < n; i++) {
        w_r[i] = y_r[i];
        w_i[i] = y_i[i];
    }

    for (size_t j = 0; j < n; j++) {
        w_r[j] = DIV(w_r[j], D[j]);
        w_i[j] = DIV(w_i[j], D[j]);

        INTERNAL_VEC wj_r = w_r[j];
        INTERNAL_VEC wj_i = w_i[j];
        size_t row_offset = j * n;

        for (size_t i = j + 1; i < n; i++) {
            INTERNAL_VEC Uji_r = U_r[row_offset + i];
            INTERNAL_VEC Uji_i = U_i[row_offset + i];

            w_r[i] = SUB(w_r[i], SUB(MUL(Uji_r, wj_r), MUL(Uji_i, wj_i)));
            w_i[i] = SUB(w_i[i], ADD(MUL(Uji_r, wj_i), MUL(Uji_i, wj_r)));
        }
    }

    for (size_t r = n; r-- > 0;) {
        INTERNAL_VEC sum_r = VCONST(0.0);
        INTERNAL_VEC sum_i = VCONST(0.0);
        size_t row_offset = r * n;

        for (size_t j = r + 1; j < n; j++) {
            INTERNAL_VEC Uij_r = U_r[row_offset + j];
            INTERNAL_VEC Uij_i = U_i[row_offset + j];
            INTERNAL_VEC xj_r = x_r[j];
            INTERNAL_VEC xj_i = x_i[j];

            sum_r = ADD(sum_r, ADD(MUL(Uij_r, xj_r), MUL(Uij_i, xj_i)));
            sum_i = ADD(sum_i, SUB(MUL(Uij_r, xj_i), MUL(Uij_i, xj_r)));
        }

        x_r[r] = SUB(w_r[r], DIV(sum_r, D[r]));
        x_i[r] = SUB(w_i[r], DIV(sum_i, D[r]));
    }

    return cond_bound;
}

/* =========================================================================
 * tlsf/tls/tlsdd_solve_zohar
 *
 * Zohar's recursion (Zohar 1969, Hermitian variant) for T x = y.
 * Same system definition as solve_levinson above.
 *
 * Workspace (caller-allocated, length n each)
 *   e_hat_r, e_hat_i          backwards predictor
 *   e_hat_prev_r, e_hat_prev_i  snapshot from the previous order
 *
 * Returns the same lane-wise reflection-coefficient condition bound used by
 * solve_levinson, deriving |rho| from Zohar's lambda recurrence.
 *
 * Note: the loop accesses R[i+1] for i up to n-1, so the caller must ensure
 * R is allocated with at least n+1 slots (or accept the final iteration reads
 * one past the last defined element, as in the original code).
 * ========================================================================= */
INTERNAL_VEC TLS(solve_zohar)(size_t n, const INTERNAL_VEC *restrict R_r, const INTERNAL_VEC *restrict R_i, const INTERNAL_VEC *restrict y_r,
                              const INTERNAL_VEC *restrict y_i, INTERNAL_VEC *restrict s_r, INTERNAL_VEC *restrict s_i, INTERNAL_VEC *restrict e_hat_r,
                              INTERNAL_VEC *restrict e_hat_i, INTERNAL_VEC *restrict e_hat_prev_r, INTERNAL_VEC *restrict e_hat_prev_i) {
    INTERNAL_VEC inv_R0 = DIV(VCONST(1.0), R_r[0]);

    s_r[0] = MUL(y_r[0], inv_R0);
    s_i[0] = MUL(y_i[0], inv_R0);

    /* rho_{-1} = conj(R[1]) / R[0] */
    INTERNAL_VEC rho_m1_r = MUL(R_r[1], inv_R0);
    INTERNAL_VEC rho_m1_i = NEG(MUL(R_i[1], inv_R0));

    e_hat_r[0] = NEG(rho_m1_r);
    e_hat_i[0] = NEG(rho_m1_i);

    /* lambda = 1 - |rho_{-1}|^2 */
    INTERNAL_VEC abs2_rho_m1 = ADD(MUL(rho_m1_r, rho_m1_r), MUL(rho_m1_i, rho_m1_i));
    INTERNAL_VEC lambda = SUB(VCONST(1.0), abs2_rho_m1);
    INTERNAL_VEC cond_bound = VCONST(1.0);
    if (n > 1) cond_bound = TLS(reflection_condition_step)(abs2_rho_m1);

    for (size_t i = 1; i < n; i++) {
        /* ----- snapshot backwards predictor */
        for (size_t k = 0; k < i; k++) {
            e_hat_prev_r[k] = e_hat_r[k];
            e_hat_prev_i[k] = e_hat_i[k];
        }

        /* ----- theta = (y[i] - sum_{k<i} rho_{i-k} s[k]) / R[0]
         *        where rho_{j} = R[j] / R[0]  (un-conjugated)             */
        INTERNAL_VEC theta_r = MUL(y_r[i], inv_R0);
        INTERNAL_VEC theta_i = MUL(y_i[i], inv_R0);

        /* ----- eta   = -conj(R[i+1]) / R[0]
         *                - sum_{k<i} conj(rho_{k+1}) e_hat_prev[k]        */
        INTERNAL_VEC eta_r = NEG(MUL(R_r[i + 1], inv_R0)); /* -Re(R[i+1])/R[0] */
        INTERNAL_VEC eta_i = MUL(R_i[i + 1], inv_R0);      /* +Im(R[i+1])/R[0] → -conj */

        for (size_t k = 0; k < i; k++) {
            INTERNAL_VEC rho_ik_r = MUL(R_r[i - k], inv_R0);
            INTERNAL_VEC rho_ik_i = MUL(R_i[i - k], inv_R0);

            /* conj(rho_{k+1}) = (R_r[k+1] - j R_i[k+1]) / R[0] */
            INTERNAL_VEC rho_mk1_r = MUL(R_r[k + 1], inv_R0);
            INTERNAL_VEC rho_mk1_i = NEG(MUL(R_i[k + 1], inv_R0));

            theta_r = SUB(theta_r, SUB(MUL(s_r[k], rho_ik_r), MUL(s_i[k], rho_ik_i)));
            theta_i = SUB(theta_i, ADD(MUL(s_r[k], rho_ik_i), MUL(s_i[k], rho_ik_r)));

            eta_r = SUB(eta_r, SUB(MUL(rho_mk1_r, e_hat_prev_r[k]), MUL(rho_mk1_i, e_hat_prev_i[k])));
            eta_i = SUB(eta_i, ADD(MUL(rho_mk1_r, e_hat_prev_i[k]), MUL(rho_mk1_i, e_hat_prev_r[k])));
        }

        INTERNAL_VEC t_lam_r = DIV(theta_r, lambda);
        INTERNAL_VEC t_lam_i = DIV(theta_i, lambda);
        INTERNAL_VEC e_lam_r = DIV(eta_r, lambda);
        INTERNAL_VEC e_lam_i = DIV(eta_i, lambda);

        /* ----- simultaneous update of s and e_hat */
        for (size_t k = 0; k < i; k++) {
            /* s[k] += t_lam * e_hat_prev[k] */
            s_r[k] = ADD(s_r[k], SUB(MUL(t_lam_r, e_hat_prev_r[k]), MUL(t_lam_i, e_hat_prev_i[k])));
            s_i[k] = ADD(s_i[k], ADD(MUL(t_lam_r, e_hat_prev_i[k]), MUL(t_lam_i, e_hat_prev_r[k])));

            /* e_hat[k+1] = e_hat_prev[k] + e_lam * conj(e_hat_prev[i-1-k]) */
            size_t g_idx = i - 1 - k;
            INTERNAL_VEC g_r = e_hat_prev_r[g_idx];
            INTERNAL_VEC g_i = NEG(e_hat_prev_i[g_idx]);

            e_hat_r[k + 1] = ADD(e_hat_prev_r[k], SUB(MUL(e_lam_r, g_r), MUL(e_lam_i, g_i)));
            e_hat_i[k + 1] = ADD(e_hat_prev_i[k], ADD(MUL(e_lam_r, g_i), MUL(e_lam_i, g_r)));
        }

        s_r[i] = t_lam_r;
        s_i[i] = t_lam_i;
        e_hat_r[0] = e_lam_r;
        e_hat_i[0] = e_lam_i;

        /* ----- lambda update */
        INTERNAL_VEC eta_mag_sq = ADD(MUL(eta_r, eta_r), MUL(eta_i, eta_i));
        /* The final update prepares the next order, so it is not part of this n-by-n solve. */
        if (i + 1 < n) {
            INTERNAL_VEC abs2_reflection = ADD(MUL(e_lam_r, e_lam_r), MUL(e_lam_i, e_lam_i));
            cond_bound = MUL(cond_bound, TLS(reflection_condition_step)(abs2_reflection));
        }
        lambda = SUB(lambda, DIV(eta_mag_sq, lambda));
    }

    return cond_bound;
}
