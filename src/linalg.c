// linalg.c  —  Precision-generic LDLT, SVD, Levinson-Durbin, Bareiss, and Zohar Toeplitz solvers

/* nanofft.h provides: dd_t and its arithmetic (dd_add/dd_sub/dd_mul/dd_div …),
 * VEC / VECF GCC vector types, VEC_LEN / VECF_LEN, and all the fast-trig
 * helpers that nanofft_precision.h macros reference internally.            */
#include <linalg.h>
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
#elif defined(HALF)
#    ifdef HALF_NATIVE_EMBED
#        define TLS(name) tlsh_nat_##name
#    else
#        define TLS(name) tlsh_##name
#    endif
#else
#    define TLS(name) tlsf_##name
#endif

#if defined(DOUBLE_DOUBLE)
typedef VECDD LINALG_VEC;
typedef VECDD LINALG_STORE_VEC;
#    define LINALG_VEC_LEN VECDD_LEN
#    define LADD(a, b) vecdd_add((a), (b))
#    define LSUB(a, b) vecdd_sub((a), (b))
#    define LMUL(a, b) vecdd_mul((a), (b))
#    define LDIV(a, b) vecdd_div((a), (b))
#    define LNEG(a) vecdd_neg((a))
#    define LSQRT(a) vecdd_sqrt((a))
#    define VCONST(x) vecdd_splat_d((double)(x))
#    define VCAST(x) vecdd_splat_d((double)(x))
#    define L_LOAD(p) (*(p))
#    define L_STORE(p, v) (*(p) = (v))
#    define L_TO_STORE(v) (v)
#    define L_FROM_STORE(s) (s)
static inline FLOAT TLS(lane_get)(LINALG_VEC value, int lane) { return dd_make(value.hi[lane], value.lo[lane]); }
static inline void TLS(lane_set)(LINALG_VEC *value, int lane, FLOAT lane_value) {
    value->hi[lane] = lane_value.hi;
    value->lo[lane] = lane_value.lo;
}
static inline FLOAT TLS(store_lane_get)(const LINALG_STORE_VEC *p, int lane) { return dd_make(p->hi[lane], p->lo[lane]); }
static inline void TLS(store_lane_set)(LINALG_STORE_VEC *p, int lane, FLOAT lane_value) {
    p->hi[lane] = lane_value.hi;
    p->lo[lane] = lane_value.lo;
}
#elif defined(HALF) && !NANOFFT_NATIVE_FP16
typedef VECF LINALG_VEC;
typedef VECH LINALG_STORE_VEC;
#    define LINALG_VEC_LEN VECH_LEN
#    define LADD(a, b) ADD((a), (b))
#    define LSUB(a, b) SUB((a), (b))
#    define LMUL(a, b) MUL((a), (b))
#    define LDIV(a, b) DIV((a), (b))
#    define LNEG(a) NEG((a))
#    define LSQRT(a) M_SQRT((a))
#    define VCONST(x) IVEC_SPLAT(FCONST(x))
#    define VCAST(x) IVEC_SPLAT(FCAST(x))
#    define L_LOAD(p) vech_to_vecf(*(p))
#    define L_STORE(p, v) (*(p) = vecf_to_vech((v)))
#    define L_TO_STORE(v) vecf_to_vech((v))
#    define L_FROM_STORE(s) vech_to_vecf((s))
static inline FLOAT TLS(lane_get)(LINALG_VEC value, int lane) { return tls_f32_to_f16(value[lane]); }
static inline void TLS(lane_set)(LINALG_VEC *value, int lane, FLOAT lane_value) { (*value)[lane] = tls_f16_to_f32(lane_value); }
static inline FLOAT TLS(store_lane_get)(const LINALG_STORE_VEC *p, int lane) { return p->elems[lane]; }
static inline void TLS(store_lane_set)(LINALG_STORE_VEC *p, int lane, FLOAT lane_value) { p->elems[lane] = lane_value; }
#else
typedef INTERNAL_VEC LINALG_VEC;
typedef INTERNAL_VEC LINALG_STORE_VEC;
#    define LINALG_VEC_LEN INTERNAL_VEC_LEN
#    define LADD(a, b) ADD((a), (b))
#    define LSUB(a, b) SUB((a), (b))
#    define LMUL(a, b) MUL((a), (b))
#    define LDIV(a, b) DIV((a), (b))
#    define LNEG(a) NEG((a))
#    define LSQRT(a) M_SQRT((a))
#    define VCONST(x) IVEC_SPLAT(FCONST(x))
#    define VCAST(x) IVEC_SPLAT(FCAST(x))
#    define L_LOAD(p) (*(p))
#    define L_STORE(p, v) (*(p) = (v))
#    define L_TO_STORE(v) (v)
#    define L_FROM_STORE(s) (s)
#    if defined(HALF)
static inline FLOAT TLS(lane_get)(LINALG_VEC value, int lane) { return tls_fp16_to_f16(value[lane]); }
static inline void TLS(lane_set)(LINALG_VEC *value, int lane, FLOAT lane_value) { (*value)[lane] = tls_f16_to_fp16(lane_value); }
static inline FLOAT TLS(store_lane_get)(const LINALG_STORE_VEC *p, int lane) { return tls_fp16_to_f16((*p)[lane]); }
static inline void TLS(store_lane_set)(LINALG_STORE_VEC *p, int lane, FLOAT lane_value) { (*p)[lane] = tls_f16_to_fp16(lane_value); }
#    else
static inline FLOAT TLS(lane_get)(LINALG_VEC value, int lane) { return value[lane]; }
static inline void TLS(lane_set)(LINALG_VEC *value, int lane, FLOAT lane_value) { (*value)[lane] = lane_value; }
static inline FLOAT TLS(store_lane_get)(const LINALG_STORE_VEC *p, int lane) { return (*p)[lane]; }
static inline void TLS(store_lane_set)(LINALG_STORE_VEC *p, int lane, FLOAT lane_value) { (*p)[lane] = lane_value; }
#    endif
#endif

static inline LINALG_STORE_VEC TLS(ldlt_diag_condition)(size_t n, const LINALG_STORE_VEC *restrict D) {
    LINALG_VEC ratio = VCONST(0.0);
    for (int lane = 0; lane < LINALG_VEC_LEN; lane++) {
        FLOAT dmin = TLS(store_lane_get)(&D[0], lane);
        FLOAT dmax = dmin;
        for (size_t i = 1; i < n; i++) {
            FLOAT d = TLS(store_lane_get)(&D[i], lane);
            if (TO_DOUBLE(d) < TO_DOUBLE(dmin)) dmin = d;
            if (TO_DOUBLE(d) > TO_DOUBLE(dmax)) dmax = d;
        }
        TLS(lane_set)(&ratio, lane, DIV(dmax, dmin));
    }
    return L_TO_STORE(ratio);
}

static inline FLOAT TLS(scalar_abs)(FLOAT value) { return TO_DOUBLE(value) < 0.0 ? NEG(value) : value; }
static inline double TLS(svd_tolerance)(void) {
#if defined(DOUBLE_DOUBLE)
    return 1e-24;
#elif defined(DOUBLE)
    return 1e-12;
#elif defined(HALF)
    return 1e-3;
#else
    return 1e-5;
#endif
}

static inline int TLS(svd_max_sweeps)(void) {
#if defined(DOUBLE_DOUBLE)
    return 256;
#elif defined(DOUBLE)
    return 64;
#elif defined(HALF)
    return 16;
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
LINALG_STORE_VEC TLS(solve_ldlt)(size_t n, const LINALG_STORE_VEC *restrict A, const LINALG_STORE_VEC *restrict b, LINALG_STORE_VEC *restrict x,
                                 LINALG_STORE_VEC *restrict L, LINALG_STORE_VEC *restrict D, LINALG_STORE_VEC *restrict y, LINALG_STORE_VEC *restrict z) {
    for (size_t i = 0; i < n; i++) {
        size_t i_row_offset = i * n;
        for (size_t j = 0; j < i; j++) {
            size_t j_row_offset = j * n;
            LINALG_VEC sum = VCONST(0.0);
            for (size_t k = 0; k < j; k++) {
                sum = LADD(sum, LMUL(LMUL(L_LOAD(&L[i_row_offset + k]), L_LOAD(&L[j_row_offset + k])), L_LOAD(&D[k])));
            }
            L_STORE(&L[i_row_offset + j], LDIV(LSUB(L_LOAD(&A[i_row_offset + j]), sum), L_LOAD(&D[j])));
        }

        LINALG_VEC sum2 = VCONST(0.0);
        for (size_t k = 0; k < i; k++) {
            LINALG_VEC lik = L_LOAD(&L[i_row_offset + k]);
            sum2 = LADD(sum2, LMUL(LMUL(lik, lik), L_LOAD(&D[k])));
        }
        L_STORE(&D[i], LSUB(L_LOAD(&A[i_row_offset + i]), sum2));
        L_STORE(&L[i_row_offset + i], VCONST(1.0));
    }

    LINALG_STORE_VEC condition = TLS(ldlt_diag_condition)(n, D);

    for (size_t i = 0; i < n; i++) {
        size_t row_offset = i * n;
        LINALG_VEC sum = VCONST(0.0);
        for (size_t j = 0; j < i; j++) {
            sum = LADD(sum, LMUL(L_LOAD(&L[row_offset + j]), L_LOAD(&y[j])));
        }
        L_STORE(&y[i], LSUB(L_LOAD(&b[i]), sum));
    }

    for (size_t i = 0; i < n; i++) {
        L_STORE(&z[i], LDIV(L_LOAD(&y[i]), L_LOAD(&D[i])));
    }

    for (size_t i = n; i-- > 0;) {
        LINALG_VEC sum = VCONST(0.0);
        for (size_t j = i + 1; j < n; j++) {
            sum = LADD(sum, LMUL(L_LOAD(&L[j * n + i]), L_LOAD(&x[j])));
        }
        L_STORE(&x[i], LSUB(L_LOAD(&z[i]), sum));
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
LINALG_STORE_VEC TLS(solve_svd)(size_t n, const LINALG_STORE_VEC *restrict A, const LINALG_STORE_VEC *restrict b, LINALG_STORE_VEC *restrict x,
                                LINALG_STORE_VEC *restrict Q, LINALG_STORE_VEC *restrict S, LINALG_STORE_VEC *restrict D, LINALG_STORE_VEC *restrict y,
                                LINALG_STORE_VEC max_cond) {
    const LINALG_STORE_VEC s_zero = L_TO_STORE(VCONST(0.0));
    const LINALG_STORE_VEC s_one = L_TO_STORE(VCONST(1.0));
    for (size_t i = 0; i < n * n; i++) {
        S[i] = A[i];
        Q[i] = s_zero;
    }
    for (size_t i = 0; i < n; i++) Q[i * n + i] = s_one;

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

                for (int lane = 0; lane < LINALG_VEC_LEN; lane++) {
                    FLOAT app = TLS(store_lane_get)(&S[pp], lane);
                    FLOAT aqq = TLS(store_lane_get)(&S[qq], lane);
                    FLOAT apq = TLS(store_lane_get)(&S[pq], lane);
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
                        FLOAT akp = TLS(store_lane_get)(&S[kp], lane);
                        FLOAT akq = TLS(store_lane_get)(&S[kq], lane);
                        FLOAT new_kp = SUB(MUL(c, akp), MUL(s, akq));
                        FLOAT new_kq = ADD(MUL(s, akp), MUL(c, akq));
                        TLS(store_lane_set)(&S[kp], lane, new_kp);
                        TLS(store_lane_set)(&S[pk], lane, new_kp);
                        TLS(store_lane_set)(&S[kq], lane, new_kq);
                        TLS(store_lane_set)(&S[qk], lane, new_kq);
                    }

                    FLOAT two_cs_apq = MUL(FCAST(2.0), MUL(cs, apq));
                    FLOAT new_app = ADD(SUB(MUL(c2, app), two_cs_apq), MUL(s2, aqq));
                    FLOAT new_aqq = ADD(ADD(MUL(s2, app), two_cs_apq), MUL(c2, aqq));
                    TLS(store_lane_set)(&S[pp], lane, new_app);
                    TLS(store_lane_set)(&S[qq], lane, new_aqq);
                    TLS(store_lane_set)(&S[pq], lane, FCAST(0.0));
                    TLS(store_lane_set)(&S[q * n + p], lane, FCAST(0.0));

                    for (size_t k = 0; k < n; k++) {
                        size_t kp = k * n + p;
                        size_t kq = k * n + q;
                        FLOAT qkp = TLS(store_lane_get)(&Q[kp], lane);
                        FLOAT qkq = TLS(store_lane_get)(&Q[kq], lane);
                        TLS(store_lane_set)(&Q[kp], lane, SUB(MUL(c, qkp), MUL(s, qkq)));
                        TLS(store_lane_set)(&Q[kq], lane, ADD(MUL(s, qkp), MUL(c, qkq)));
                    }
                }
            }
        }

        if (all_converged) break;
    }

    int lane_converged[LINALG_VEC_LEN];
    for (int lane = 0; lane < LINALG_VEC_LEN; lane++) lane_converged[lane] = 1;

    for (size_t p = 0; p < n; p++) {
        for (size_t q = p + 1; q < n; q++) {
            size_t pp = p * n + p;
            size_t qq = q * n + q;
            size_t pq = p * n + q;
            for (int lane = 0; lane < LINALG_VEC_LEN; lane++) {
                FLOAT app = TLS(store_lane_get)(&S[pp], lane);
                FLOAT aqq = TLS(store_lane_get)(&S[qq], lane);
                FLOAT apq = TLS(store_lane_get)(&S[pq], lane);
                double scale = fabs(TO_DOUBLE(app)) + fabs(TO_DOUBLE(aqq));
                if (scale == 0.0) scale = 1.0;
                if (fabs(TO_DOUBLE(apq)) > tol * scale) lane_converged[lane] = 0;
            }
        }
    }

    LINALG_VEC condition = VCONST(0.0);

    for (int lane = 0; lane < LINALG_VEC_LEN; lane++) {
        FLOAT max_abs = FCAST(0.0);
        FLOAT min_abs = FCAST(0.0);
        FLOAT max_positive = FCAST(0.0);
        int have_nonzero = 0;

        for (size_t i = 0; i < n; i++) {
            FLOAT lambda = TLS(store_lane_get)(&S[i * n + i], lane);
            FLOAT abs_lambda = TLS(scalar_abs)(lambda);
            double abs_d = TO_DOUBLE(abs_lambda);
            if (TO_DOUBLE(lambda) > TO_DOUBLE(max_positive)) max_positive = lambda;
            if (abs_d > TO_DOUBLE(max_abs)) max_abs = abs_lambda;
            if (abs_d > 0.0 && (!have_nonzero || abs_d < TO_DOUBLE(min_abs))) {
                min_abs = abs_lambda;
                have_nonzero = 1;
            }
            TLS(store_lane_set)(&D[i], lane, lambda);
        }

        NANOFFT_DIAG_PUSH_NAN
        FLOAT lane_condition = have_nonzero ? DIV(max_abs, min_abs) : FCAST(INFINITY);
        NANOFFT_DIAG_POP
        if (!lane_converged[lane]) lane_condition = NEG(lane_condition);
        TLS(lane_set)(&condition, lane, lane_condition);

        FLOAT threshold = FCAST(0.0);
        FLOAT lane_max_cond = TLS(store_lane_get)(&max_cond, lane);
        int filtered = TO_DOUBLE(lane_max_cond) > 0.0;
        if (filtered && TO_DOUBLE(max_positive) > 0.0) threshold = DIV(max_positive, lane_max_cond);

        for (size_t i = 0; i < n; i++) {
            FLOAT dot = FCAST(0.0);
            for (size_t j = 0; j < n; j++) {
                dot = ADD(dot, MUL(TLS(store_lane_get)(&Q[j * n + i], lane), TLS(store_lane_get)(&b[j], lane)));
            }

            FLOAT lambda = TLS(store_lane_get)(&D[i], lane);
            FLOAT coeff = FCAST(0.0);
            if (filtered) {
                if (TO_DOUBLE(lambda) > 0.0 && TO_DOUBLE(lambda) >= TO_DOUBLE(threshold)) coeff = DIV(dot, lambda);
            } else if (TO_DOUBLE(lambda) != 0.0) {
                coeff = DIV(dot, lambda);
            }
            TLS(store_lane_set)(&y[i], lane, coeff);
        }

        for (size_t i = 0; i < n; i++) {
            FLOAT sum = FCAST(0.0);
            for (size_t j = 0; j < n; j++) {
                sum = ADD(sum, MUL(TLS(store_lane_get)(&Q[i * n + j], lane), TLS(store_lane_get)(&y[j], lane)));
            }
            TLS(store_lane_set)(&x[i], lane, sum);
        }
    }

    return L_TO_STORE(condition);
}

/* =========================================================================
 * tlsf/tls/tlsdd_solve_levinson
 *
 * Fused in-place Levinson-Durbin recursion for LINALG_VEC_LEN independent
 * complex Hermitian Toeplitz systems T x = y (quadratic form r2 = y^H T^-1 y):
 *   T_{ij} = R[|i-j|], R[0] real and positive, R[-k] = conj(R[k]).
 *
 * In / out
 *   R_r, R_i    Toeplitz row, length n
 *   y_r, y_i    right-hand side, length n
 *   x_r, x_i    x_r[0] = y^H T^-1 y, x_i[0] = 0
 *   return      lane-wise product n * Π (1 + |gamma_j|) / (1 - |gamma_j|)
 *
 * Workspace (caller-allocated)
 *   a_r, a_i            predictor polynomial, length n each
 *   a_prev_r, a_prev_i  unused (retained for API compatibility; may be NULL)
 * ========================================================================= */
LINALG_STORE_VEC TLS(solve_levinson)(size_t n, const LINALG_STORE_VEC *restrict R_r, const LINALG_STORE_VEC *restrict R_i, const LINALG_STORE_VEC *restrict y_r,
                                     const LINALG_STORE_VEC *restrict y_i, LINALG_STORE_VEC *restrict x_r, LINALG_STORE_VEC *restrict x_i,
                                     LINALG_STORE_VEC *restrict a_r, LINALG_STORE_VEC *restrict a_i, LINALG_STORE_VEC *restrict a_prev_r,
                                     LINALG_STORE_VEC *restrict a_prev_i) {
    (void)a_prev_r;
    (void)a_prev_i;

    const LINALG_VEC one = VCONST(1.0);
    const LINALG_VEC zero = VCONST(0.0);
    LINALG_VEC E = L_LOAD(&R_r[0]);
    LINALG_VEC inv_E = LDIV(one, E);
    LINALG_VEC y0r = L_LOAD(&y_r[0]);
    LINALG_VEC y0i = L_LOAD(&y_i[0]);
    LINALG_VEC r2 = LMUL(LADD(LMUL(y0r, y0r), LMUL(y0i, y0i)), inv_E);
    LINALG_VEC cond_num = VCAST((double)n);
    LINALG_VEC cond_den = one;

    L_STORE(&a_r[0], one);
    L_STORE(&a_i[0], zero);

    for (size_t k = 1; k < n; k++) {
        /* 4-way unrolled reflection inner product: lambda = sum_{i=0}^{k-1} R[k-i] a[i] */
        LINALG_VEC lr0 = L_LOAD(&R_r[k]), li0 = L_LOAD(&R_i[k]); /* i = 0: a[0] == 1 + 0i */
        LINALG_VEC lr1 = zero, li1 = zero;
        LINALG_VEC lr2 = zero, li2 = zero;
        LINALG_VEC lr3 = zero, li3 = zero;
        size_t i = 1;
        for (; i + 3 < k; i += 4) {
            LINALG_VEC r0r = L_LOAD(&R_r[k - i]), r0i = L_LOAD(&R_i[k - i]), a0r = L_LOAD(&a_r[i]), a0i = L_LOAD(&a_i[i]);
            lr0 = LADD(lr0, LSUB(LMUL(r0r, a0r), LMUL(r0i, a0i)));
            li0 = LADD(li0, LADD(LMUL(r0r, a0i), LMUL(r0i, a0r)));

            LINALG_VEC r1r = L_LOAD(&R_r[k - i - 1]), r1i = L_LOAD(&R_i[k - i - 1]), a1r = L_LOAD(&a_r[i + 1]), a1i = L_LOAD(&a_i[i + 1]);
            lr1 = LADD(lr1, LSUB(LMUL(r1r, a1r), LMUL(r1i, a1i)));
            li1 = LADD(li1, LADD(LMUL(r1r, a1i), LMUL(r1i, a1r)));

            LINALG_VEC r2r = L_LOAD(&R_r[k - i - 2]), r2i = L_LOAD(&R_i[k - i - 2]), a2r = L_LOAD(&a_r[i + 2]), a2i = L_LOAD(&a_i[i + 2]);
            lr2 = LADD(lr2, LSUB(LMUL(r2r, a2r), LMUL(r2i, a2i)));
            li2 = LADD(li2, LADD(LMUL(r2r, a2i), LMUL(r2i, a2r)));

            LINALG_VEC r3r = L_LOAD(&R_r[k - i - 3]), r3i = L_LOAD(&R_i[k - i - 3]), a3r = L_LOAD(&a_r[i + 3]), a3i = L_LOAD(&a_i[i + 3]);
            lr3 = LADD(lr3, LSUB(LMUL(r3r, a3r), LMUL(r3i, a3i)));
            li3 = LADD(li3, LADD(LMUL(r3r, a3i), LMUL(r3i, a3r)));
        }
        for (; i < k; i++) {
            LINALG_VEC rr = L_LOAD(&R_r[k - i]), ri = L_LOAD(&R_i[k - i]), ar = L_LOAD(&a_r[i]), ai = L_LOAD(&a_i[i]);
            lr0 = LADD(lr0, LSUB(LMUL(rr, ar), LMUL(ri, ai)));
            li0 = LADD(li0, LADD(LMUL(rr, ai), LMUL(ri, ar)));
        }
        LINALG_VEC lambda_r = LADD(LADD(lr0, lr1), LADD(lr2, lr3));
        LINALG_VEC lambda_i = LADD(LADD(li0, li1), LADD(li2, li3));

        LINALG_VEC gamma_r = LNEG(LMUL(lambda_r, inv_E));
        LINALG_VEC gamma_i = LNEG(LMUL(lambda_i, inv_E));

        /* Start inv_E = 1 / E and sqrt(|gamma|^2) before the fused update + xi loop
         * so both overlap with the O(k) FMA work; defer cond division to the end. */
        LINALG_VEC abs2_gamma = LADD(LMUL(gamma_r, gamma_r), LMUL(gamma_i, gamma_i));
        LINALG_VEC abs_gamma = LSQRT(abs2_gamma);
        E = LMUL(E, LSUB(one, abs2_gamma));
        inv_E = LDIV(one, E);

        /* a[k] = gamma (since a[0] == 1 + 0i); initialize xi with i=0 and i=k terms. */
        L_STORE(&a_r[k], gamma_r);
        L_STORE(&a_i[k], gamma_i);
        LINALG_VEC xr0 = LADD(L_LOAD(&y_r[k]), LSUB(LMUL(gamma_r, y0r), LMUL(gamma_i, y0i)));
        LINALG_VEC xi0 = LADD(L_LOAD(&y_i[k]), LADD(LMUL(gamma_r, y0i), LMUL(gamma_i, y0r)));
        LINALG_VEC xr1 = zero, xi1 = zero;

        /* Paired in-place symmetric update of (a[lo], a[hi]) fused with xi accumulation. */
        size_t lo = 1, hi = k - 1;
        for (; lo < hi; ++lo, --hi) {
            LINALG_VEC alr = L_LOAD(&a_r[lo]), ali = L_LOAD(&a_i[lo]);
            LINALG_VEC ahr = L_LOAD(&a_r[hi]), ahi = L_LOAD(&a_i[hi]);

            LINALG_VEC nlr = LADD(alr, LADD(LMUL(gamma_r, ahr), LMUL(gamma_i, ahi)));
            LINALG_VEC nli = LADD(ali, LSUB(LMUL(gamma_i, ahr), LMUL(gamma_r, ahi)));
            LINALG_VEC nhr = LADD(ahr, LADD(LMUL(gamma_r, alr), LMUL(gamma_i, ali)));
            LINALG_VEC nhi = LADD(ahi, LSUB(LMUL(gamma_i, alr), LMUL(gamma_r, ali)));

            L_STORE(&a_r[lo], nlr);
            L_STORE(&a_i[lo], nli);
            L_STORE(&a_r[hi], nhr);
            L_STORE(&a_i[hi], nhi);

            LINALG_VEC yhr = L_LOAD(&y_r[hi]), yhi = L_LOAD(&y_i[hi]);
            LINALG_VEC ylr = L_LOAD(&y_r[lo]), yli = L_LOAD(&y_i[lo]);
            xr0 = LADD(xr0, LSUB(LMUL(nlr, yhr), LMUL(nli, yhi)));
            xi0 = LADD(xi0, LADD(LMUL(nlr, yhi), LMUL(nli, yhr)));
            xr1 = LADD(xr1, LSUB(LMUL(nhr, ylr), LMUL(nhi, yli)));
            xi1 = LADD(xi1, LADD(LMUL(nhr, yli), LMUL(nhi, ylr)));
        }
        if (lo == hi) {
            LINALG_VEC alr = L_LOAD(&a_r[lo]), ali = L_LOAD(&a_i[lo]);
            LINALG_VEC nlr = LADD(alr, LADD(LMUL(gamma_r, alr), LMUL(gamma_i, ali)));
            LINALG_VEC nli = LADD(ali, LSUB(LMUL(gamma_i, alr), LMUL(gamma_r, ali)));
            L_STORE(&a_r[lo], nlr);
            L_STORE(&a_i[lo], nli);

            LINALG_VEC ylr = L_LOAD(&y_r[lo]), yli = L_LOAD(&y_i[lo]);
            xr0 = LADD(xr0, LSUB(LMUL(nlr, ylr), LMUL(nli, yli)));
            xi0 = LADD(xi0, LADD(LMUL(nlr, yli), LMUL(nli, ylr)));
        }

        cond_num = LMUL(cond_num, LADD(one, abs_gamma));
        cond_den = LMUL(cond_den, LSUB(one, abs_gamma));

        LINALG_VEC xr = LADD(xr0, xr1);
        LINALG_VEC xi = LADD(xi0, xi1);
        LINALG_VEC abs2_xi = LADD(LMUL(xr, xr), LMUL(xi, xi));
        r2 = LADD(r2, LMUL(abs2_xi, inv_E));
    }

    L_STORE(&x_r[0], r2);
    L_STORE(&x_i[0], zero);
    return L_TO_STORE(LDIV(cond_num, cond_den));
}

/* =========================================================================
 * tlsf/tls/tlsdd_solve_bareiss
 *
 * Bareiss fast Cholesky factorization for Hermitian Toeplitz systems T x = y.
 * Same system definition as solve_levinson above.
 *
 * Workspace (caller-allocated)
 *   U_r, U_i            upper triangular factor, length n*n each
 *   D                   real inverse diagonal scratch, length n
 *   u_r, u_i, v_r, v_i  Bareiss generators, length n each
 *   w_r, w_i            substitution workspace, length n each
 *
 * Returns the same lane-wise reflection-coefficient condition bound used by
 * solve_levinson.
 * ========================================================================= */
LINALG_STORE_VEC TLS(solve_bareiss)(size_t n, const LINALG_STORE_VEC *restrict R_r, const LINALG_STORE_VEC *restrict R_i, const LINALG_STORE_VEC *restrict y_r,
                                    const LINALG_STORE_VEC *restrict y_i, LINALG_STORE_VEC *restrict x_r, LINALG_STORE_VEC *restrict x_i,
                                    LINALG_STORE_VEC *restrict U_r, LINALG_STORE_VEC *restrict U_i, LINALG_STORE_VEC *restrict D,
                                    LINALG_STORE_VEC *restrict u_r, LINALG_STORE_VEC *restrict u_i, LINALG_STORE_VEC *restrict v_r,
                                    LINALG_STORE_VEC *restrict v_i, LINALG_STORE_VEC *restrict w_r, LINALG_STORE_VEC *restrict w_i) {
    const LINALG_VEC one = VCONST(1.0);
    const LINALG_VEC zero = VCONST(0.0);
    const LINALG_STORE_VEC s_zero = L_TO_STORE(zero);
    LINALG_VEC cond_num = VCAST((double)n);
    LINALG_VEC cond_den = one;

    for (size_t j = 0; j < n; j++) {
        u_r[j] = R_r[j];
        u_i[j] = R_i[j];
        if (j < n - 1) {
            v_r[j] = R_r[j + 1];
            v_i[j] = R_i[j + 1];
        } else {
            v_r[j] = s_zero;
            v_i[j] = s_zero;
        }
    }

    for (size_t k = 0; k < n; k++) {
        /* Save 1 / D[k] directly in D[k] so forward substitution avoids a second
         * pass of n divisions. */
        LINALG_VEC inv_D = LDIV(one, L_LOAD(&u_r[0]));
        L_STORE(&D[k], inv_D);

        size_t row_offset = k * n + k;
        for (size_t j = 0; j < n - k; j++) {
            U_r[row_offset + j] = u_r[j];
            U_i[row_offset + j] = u_i[j];
        }

        if (k == n - 1) break;

        LINALG_VEC rho_r = LMUL(L_LOAD(&v_r[0]), inv_D);
        LINALG_VEC rho_i = LMUL(L_LOAD(&v_i[0]), inv_D);
        LINALG_VEC abs2_rho = LADD(LMUL(rho_r, rho_r), LMUL(rho_i, rho_i));
        LINALG_VEC abs_rho = LSQRT(abs2_rho);

        for (size_t j = 0; j < n - k - 1; j++) {
            LINALG_VEC uj_r = L_LOAD(&u_r[j]);
            LINALG_VEC uj_i = L_LOAD(&u_i[j]);
            LINALG_VEC vj_r = L_LOAD(&v_r[j]);
            LINALG_VEC vj_i = L_LOAD(&v_i[j]);

            LINALG_VEC up1_r = L_LOAD(&u_r[j + 1]);
            LINALG_VEC up1_i = L_LOAD(&u_i[j + 1]);
            LINALG_VEC vp1_r = L_LOAD(&v_r[j + 1]);
            LINALG_VEC vp1_i = L_LOAD(&v_i[j + 1]);

            L_STORE(&u_r[j], LSUB(uj_r, LADD(LMUL(rho_r, vj_r), LMUL(rho_i, vj_i))));
            L_STORE(&u_i[j], LSUB(uj_i, LSUB(LMUL(rho_r, vj_i), LMUL(rho_i, vj_r))));

            L_STORE(&v_r[j], LSUB(vp1_r, LSUB(LMUL(rho_r, up1_r), LMUL(rho_i, up1_i))));
            L_STORE(&v_i[j], LSUB(vp1_i, LADD(LMUL(rho_r, up1_i), LMUL(rho_i, up1_r))));
        }

        cond_num = LMUL(cond_num, LADD(one, abs_rho));
        cond_den = LMUL(cond_den, LSUB(one, abs_rho));
    }

    for (size_t i = 0; i < n; i++) {
        w_r[i] = y_r[i];
        w_i[i] = y_i[i];
    }

    /* Forward substitution fused with quadratic-form accumulation:
     *   |w_raw|^2 / D_j = Re(w_raw * conj(w_raw / D_j)). */
    LINALG_VEC r2 = zero;
    for (size_t j = 0; j < n; j++) {
        LINALG_VEC wraw_r = L_LOAD(&w_r[j]);
        LINALG_VEC wraw_i = L_LOAD(&w_i[j]);
        LINALG_VEC inv_Dj = L_LOAD(&D[j]);
        LINALG_VEC wj_r = LMUL(wraw_r, inv_Dj);
        LINALG_VEC wj_i = LMUL(wraw_i, inv_Dj);
        r2 = LADD(r2, LADD(LMUL(wraw_r, wj_r), LMUL(wraw_i, wj_i)));

        size_t row_offset = j * n;
        for (size_t i = j + 1; i < n; i++) {
            LINALG_VEC Uji_r = L_LOAD(&U_r[row_offset + i]);
            LINALG_VEC Uji_i = L_LOAD(&U_i[row_offset + i]);

            L_STORE(&w_r[i], LSUB(L_LOAD(&w_r[i]), LSUB(LMUL(Uji_r, wj_r), LMUL(Uji_i, wj_i))));
            L_STORE(&w_i[i], LSUB(L_LOAD(&w_i[i]), LADD(LMUL(Uji_r, wj_i), LMUL(Uji_i, wj_r))));
        }
    }

    L_STORE(&x_r[0], r2);
    L_STORE(&x_i[0], zero);
    return L_TO_STORE(LDIV(cond_num, cond_den));
}

/* =========================================================================
 * tlsf/tls/tlsdd_solve_zohar
 *
 * Zohar's recursion (Zohar 1969, Hermitian variant) for T x = y.
 * Same system definition as solve_levinson above.
 *
 * Workspace (caller-allocated, length n each)
 *   e_hat_r, e_hat_i            backwards predictor (ping-ponged with prev)
 *   e_hat_prev_r, e_hat_prev_i  alternate buffer (no per-step copy)
 *
 * Returns the same lane-wise reflection-coefficient condition bound used by
 * solve_levinson, deriving |rho| from Zohar's lambda recurrence.
 * ========================================================================= */
LINALG_STORE_VEC TLS(solve_zohar)(size_t n, const LINALG_STORE_VEC *restrict R_r, const LINALG_STORE_VEC *restrict R_i, const LINALG_STORE_VEC *restrict y_r,
                                  const LINALG_STORE_VEC *restrict y_i, LINALG_STORE_VEC *restrict s_r, LINALG_STORE_VEC *restrict s_i,
                                  LINALG_STORE_VEC *restrict e_hat_r, LINALG_STORE_VEC *restrict e_hat_i, LINALG_STORE_VEC *restrict e_hat_prev_r,
                                  LINALG_STORE_VEC *restrict e_hat_prev_i) {
    const LINALG_VEC one = VCONST(1.0);
    const LINALG_VEC zero = VCONST(0.0);
    LINALG_VEC inv_R0 = LDIV(one, L_LOAD(&R_r[0]));

    LINALG_VEC y0r = L_LOAD(&y_r[0]);
    LINALG_VEC y0i = L_LOAD(&y_i[0]);
    LINALG_VEC s0r = LMUL(y0r, inv_R0);
    LINALG_VEC s0i = LMUL(y0i, inv_R0);
    L_STORE(&s_r[0], s0r);
    L_STORE(&s_i[0], s0i);

    LINALG_VEC cond_num = VCAST((double)n);
    LINALG_VEC cond_den = one;

    if (n <= 1) {
        L_STORE(&s_r[0], LADD(LMUL(y0r, s0r), LMUL(y0i, s0i)));
        L_STORE(&s_i[0], zero);
        return L_TO_STORE(cond_num);
    }

    /* rho_{-1} = conj(R[1]) / R[0] */
    LINALG_VEC rho_m1_r = LMUL(L_LOAD(&R_r[1]), inv_R0);
    LINALG_VEC rho_m1_i = LNEG(LMUL(L_LOAD(&R_i[1]), inv_R0));

    LINALG_STORE_VEC *cur_r = e_hat_r;
    LINALG_STORE_VEC *cur_i = e_hat_i;
    LINALG_STORE_VEC *prev_r = e_hat_prev_r;
    LINALG_STORE_VEC *prev_i = e_hat_prev_i;

    L_STORE(&cur_r[0], LNEG(rho_m1_r));
    L_STORE(&cur_i[0], LNEG(rho_m1_i));

    /* lambda = 1 - |rho_{-1}|^2 */
    LINALG_VEC abs2_rho_m1 = LADD(LMUL(rho_m1_r, rho_m1_r), LMUL(rho_m1_i, rho_m1_i));
    LINALG_VEC abs_rho_m1 = LSQRT(abs2_rho_m1);
    cond_num = LMUL(cond_num, LADD(one, abs_rho_m1));
    cond_den = LMUL(cond_den, LSUB(one, abs_rho_m1));
    LINALG_VEC lambda = LSUB(one, abs2_rho_m1);

    for (size_t i = 1; i < n; i++) {
        /* Ping-pong predictor buffers: previous order is now in prev_r/prev_i. */
        LINALG_STORE_VEC *tmp_r = prev_r;
        LINALG_STORE_VEC *tmp_i = prev_i;
        prev_r = cur_r;
        prev_i = cur_i;
        cur_r = tmp_r;
        cur_i = tmp_i;

        /* Overlap inv_lambda = 1 / lambda with the O(i) inner dot products, and
         * factor inv_R0 outside the sums to avoid 2*n^2 redundant multiplies. */
        LINALG_VEC inv_lambda = LDIV(one, lambda);
        LINALG_VEC inv_R0_lam = LMUL(inv_R0, inv_lambda);

        LINALG_VEC theta_raw_r = L_LOAD(&y_r[i]);
        LINALG_VEC theta_raw_i = L_LOAD(&y_i[i]);
        LINALG_VEC eta_raw_r = zero;
        LINALG_VEC eta_raw_i = zero;

        if (i + 1 < n) {
            eta_raw_r = LNEG(L_LOAD(&R_r[i + 1]));
            eta_raw_i = L_LOAD(&R_i[i + 1]); /* -conj(R[i+1]) */

            for (size_t k = 0; k < i; k++) {
                LINALG_VEC rik_r = L_LOAD(&R_r[i - k]), rik_i = L_LOAD(&R_i[i - k]);
                LINALG_VEC rmk_r = L_LOAD(&R_r[k + 1]), rmk_i = LNEG(L_LOAD(&R_i[k + 1])); /* conj(R[k+1]) */
                LINALG_VEC sk_r = L_LOAD(&s_r[k]), sk_i = L_LOAD(&s_i[k]);
                LINALG_VEC pk_r = L_LOAD(&prev_r[k]), pk_i = L_LOAD(&prev_i[k]);

                theta_raw_r = LSUB(theta_raw_r, LSUB(LMUL(sk_r, rik_r), LMUL(sk_i, rik_i)));
                theta_raw_i = LSUB(theta_raw_i, LADD(LMUL(sk_r, rik_i), LMUL(sk_i, rik_r)));

                eta_raw_r = LSUB(eta_raw_r, LSUB(LMUL(rmk_r, pk_r), LMUL(rmk_i, pk_i)));
                eta_raw_i = LSUB(eta_raw_i, LADD(LMUL(rmk_r, pk_i), LMUL(rmk_i, pk_r)));
            }
        } else {
            for (size_t k = 0; k < i; k++) {
                LINALG_VEC rik_r = L_LOAD(&R_r[i - k]), rik_i = L_LOAD(&R_i[i - k]);
                LINALG_VEC sk_r = L_LOAD(&s_r[k]), sk_i = L_LOAD(&s_i[k]);

                theta_raw_r = LSUB(theta_raw_r, LSUB(LMUL(sk_r, rik_r), LMUL(sk_i, rik_i)));
                theta_raw_i = LSUB(theta_raw_i, LADD(LMUL(sk_r, rik_i), LMUL(sk_i, rik_r)));
            }
        }

        LINALG_VEC t_lam_r = LMUL(theta_raw_r, inv_R0_lam);
        LINALG_VEC t_lam_i = LMUL(theta_raw_i, inv_R0_lam);

        /* Simultaneous update of s */
        for (size_t k = 0; k < i; k++) {
            LINALG_VEC pk_r = L_LOAD(&prev_r[k]), pk_i = L_LOAD(&prev_i[k]);
            L_STORE(&s_r[k], LADD(L_LOAD(&s_r[k]), LSUB(LMUL(t_lam_r, pk_r), LMUL(t_lam_i, pk_i))));
            L_STORE(&s_i[k], LADD(L_LOAD(&s_i[k]), LADD(LMUL(t_lam_r, pk_i), LMUL(t_lam_i, pk_r))));
        }
        L_STORE(&s_r[i], t_lam_r);
        L_STORE(&s_i[i], t_lam_i);

        if (i + 1 < n) {
            LINALG_VEC e_lam_r = LMUL(eta_raw_r, inv_R0_lam);
            LINALG_VEC e_lam_i = LMUL(eta_raw_i, inv_R0_lam);
            LINALG_VEC abs2_refl = LADD(LMUL(e_lam_r, e_lam_r), LMUL(e_lam_i, e_lam_i));
            LINALG_VEC abs_refl = LSQRT(abs2_refl);

            for (size_t k = 0; k < i; k++) {
                size_t g_idx = i - 1 - k;
                LINALG_VEC g_r = L_LOAD(&prev_r[g_idx]);
                LINALG_VEC g_i = LNEG(L_LOAD(&prev_i[g_idx]));

                L_STORE(&cur_r[k + 1], LADD(L_LOAD(&prev_r[k]), LSUB(LMUL(e_lam_r, g_r), LMUL(e_lam_i, g_i))));
                L_STORE(&cur_i[k + 1], LADD(L_LOAD(&prev_i[k]), LADD(LMUL(e_lam_r, g_i), LMUL(e_lam_i, g_r))));
            }
            L_STORE(&cur_r[0], e_lam_r);
            L_STORE(&cur_i[0], e_lam_i);

            cond_num = LMUL(cond_num, LADD(one, abs_refl));
            cond_den = LMUL(cond_den, LSUB(one, abs_refl));

            /* lambda *= (1 - |e_lam|^2)  (identical to lambda - |eta|^2 / lambda) */
            lambda = LMUL(lambda, LSUB(one, abs2_refl));
        }
    }

    LINALG_VEC r2 = zero;
    for (size_t k = 0; k < n; k++) {
        r2 = LADD(r2, LADD(LMUL(L_LOAD(&y_r[k]), L_LOAD(&s_r[k])), LMUL(L_LOAD(&y_i[k]), L_LOAD(&s_i[k]))));
    }
    L_STORE(&s_r[0], r2);
    L_STORE(&s_i[0], zero);
    return L_TO_STORE(LDIV(cond_num, cond_den));
}
