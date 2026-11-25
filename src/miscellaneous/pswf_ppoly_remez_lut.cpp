/*
 * pswf_ppoly_remez_lut.cpp
 *
 * Offline float128 piecewise minimax polynomial generator for PSWF_0^c(z)
 * and FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution constants.
 *
 * Optimization pipeline per panel j in [0, ceil(w/2) - 1]:
 *   1. Standard Chebyshev nodes L^2 regression
 *   2. Uniform nodes L^4 IRLS
 *   3. Uniform nodes L^8 IRLS
 *   4. Uniform nodes L^12 IRLS
 *   5. Uniform nodes L^16 IRLS
 *   6. Remez exchange fine-tuning with golden-section subgrid extremum refinement
 *
 * Preserves the 1/4-above-1, 3/4-below-1 log-range scaling 2^e from src/nufft1.c
 * (pswf_kernel_scale_exp) across:
 *   - FP16 (h, 2xFP16): w in {4, 5},       PSWF21 & PSWF43
 *   - FP32 (f, float):  w in {8, 9},       PSWF21 & PSWF43
 *   - DF   (df, 2xFP32): w in {14, 15, 16}, PSWF21 & PSWF43
 *   - FP64 (d, double): w in {16, 17, 18}, PSWF21 & PSWF43
 */

#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using boost::multiprecision::float128;

namespace {

constexpr int BOUWKAMP_TERMS = 120;

float128 pi_q() {
    static const float128 v = boost::multiprecision::acos(float128(-1));
    return v;
}

// Power-of-two scaling exponent from src/nufft1.c:pswf_kernel_scale_exp
// Places 1/4 of the kernel log-range above 1 and 3/4 below 1.
int pswf_kernel_scale_exp(int w, bool mode43, int max_exp) {
    if (w <= 0) return 0;
    const double c =
        3.14159265358979323846 * (double)w * (mode43 ? 0.625 : 0.75) - 0.05;
    if (!(c > 0.0)) return 0;
    const double log_ratio = c - 0.5 * std::log(4.0 * 3.14159265358979323846 * c);
    if (!(log_ratio > 0.0)) return 0;
    int e = (int)std::floor((0.25 * log_ratio) / 0.6931471805599453094 + 0.5);
    if (e < 0) e = 0;
    if (e > max_exp) e = max_exp;
    return e;
}

// Dense Gaussian elimination with partial pivoting in float128
bool solve_linear_system(std::vector<float128> A,
                         std::vector<float128> b,
                         int n,
                         std::vector<float128> &x) {
    using boost::multiprecision::abs;
    x.assign(n, float128(0));
    for (int k = 0; k < n; ++k) {
        int piv = k;
        float128 best = abs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            float128 v = abs(A[i * n + k]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best == float128(0)) return false;
        if (piv != k) {
            for (int j = k; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]);
            std::swap(b[k], b[piv]);
        }
        const float128 akk = A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            const float128 factor = A[i * n + k] / akk;
            A[i * n + k] = float128(0);
            for (int j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        float128 s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * x[j];
        x[i] = s / A[i * n + i];
    }
    return true;
}

// Weighted least-squares solver via modified Gram-Schmidt QR with re-orthogonalization
bool solve_weighted_ls_qr(const std::vector<float128> &V,
                          const std::vector<float128> &y,
                          const std::vector<float128> &w,
                          int n_samples,
                          int n_cols,
                          float128 ridge,
                          std::vector<float128> &coeff) {
    using boost::multiprecision::sqrt;
    const int m_rows = (ridge > float128(0)) ? (n_samples + n_cols) : n_samples;
    std::vector<float128> Q((size_t)n_cols * (size_t)m_rows, float128(0));
    std::vector<float128> R((size_t)n_cols * (size_t)n_cols, float128(0));
    std::vector<float128> rhs((size_t)m_rows, float128(0));

    for (int i = 0; i < n_samples; ++i) {
        const float128 sw = sqrt(w[i]);
        rhs[i] = sw * y[i];
        const float128 *row = &V[(size_t)i * (size_t)n_cols];
        for (int j = 0; j < n_cols; ++j) {
            Q[(size_t)j * (size_t)m_rows + (size_t)i] = sw * row[j];
        }
    }
    if (ridge > float128(0)) {
        for (int j = 0; j < n_cols; ++j) {
            Q[(size_t)j * (size_t)m_rows + (size_t)(n_samples + j)] = ridge;
        }
    }

    for (int j = 0; j < n_cols; ++j) {
        float128 *qj = &Q[(size_t)j * (size_t)m_rows];
        for (int pass = 0; pass < 2; ++pass) {
            for (int k = 0; k < j; ++k) {
                const float128 *qk = &Q[(size_t)k * (size_t)m_rows];
                float128 dot = 0;
                for (int i = 0; i < m_rows; ++i) dot += qk[i] * qj[i];
                R[(size_t)k * (size_t)n_cols + (size_t)j] += dot;
                for (int i = 0; i < m_rows; ++i) qj[i] -= dot * qk[i];
            }
        }
        float128 norm2 = 0;
        for (int i = 0; i < m_rows; ++i) norm2 += qj[i] * qj[i];
        const float128 norm = sqrt(norm2);
        if (!(norm > float128(0))) return false;
        R[(size_t)j * (size_t)n_cols + (size_t)j] = norm;
        const float128 inv = float128(1) / norm;
        for (int i = 0; i < m_rows; ++i) qj[i] *= inv;
    }

    std::vector<float128> qt_b(n_cols, float128(0));
    for (int j = 0; j < n_cols; ++j) {
        const float128 *qj = &Q[(size_t)j * (size_t)m_rows];
        float128 s = 0;
        for (int i = 0; i < m_rows; ++i) s += qj[i] * rhs[i];
        qt_b[j] = s;
    }

    coeff.assign(n_cols, float128(0));
    for (int j = n_cols - 1; j >= 0; --j) {
        float128 s = qt_b[j];
        for (int k = j + 1; k < n_cols; ++k) {
            s -= R[(size_t)j * (size_t)n_cols + (size_t)k] * coeff[k];
        }
        coeff[j] = s / R[(size_t)j * (size_t)n_cols + (size_t)j];
    }
    return true;
}

// High-precision Bouwkamp solver for PSWF_0^c(z) / PSWF_0^c(0) in float128
struct PswfOracle {
    float128 c;
    int scale_exp;
    float128 scale_pow2;
    std::vector<float128> leg_c; // Even Legendre coefficients for normalized PSWF0
    int active_terms;
    float128 lambda0;            // Integral_{-1}^1 (psi0(z)/psi0(0)) dz = 2 * leg_c[0]

    void init(int w, bool mode43, int max_exp) {
        using boost::multiprecision::abs;
        using boost::multiprecision::ldexp;
        using boost::multiprecision::sqrt;
        c = pi_q() * float128(w) * (mode43 ? float128("0.625") : float128("0.75")) -
            float128("0.05");
        scale_exp = pswf_kernel_scale_exp(w, mode43, max_exp);
        scale_pow2 = ldexp(float128(1), scale_exp);

        const int M = BOUWKAMP_TERMS;
        const float128 c2 = c * c;
        std::vector<float128> diag(M, float128(0));
        std::vector<float128> off(M - 1, float128(0));

        for (int k = 0; k < M; ++k) {
            const float128 n = float128(2 * k);
            const float128 num = float128(2) * n * (n + float128(1)) - float128(1);
            const float128 den = (float128(2) * n - float128(1)) * (float128(2) * n + float128(3));
            diag[k] = n * (n + float128(1)) + c2 * (num / den);
            if (k + 1 < M) {
                const float128 a = (n + float128(2)) * (n + float128(1)) /
                                   ((float128(2) * n + float128(3)) *
                                    sqrt((float128(2) * n + float128(1)) *
                                         (float128(2) * n + float128(5))));
                off[k] = c2 * a;
            }
        }

        // Tridiagonal Thomas solve for inverse iteration
        std::vector<float128> v(M, float128(0));
        v[0] = float128(1);
        float128 shift = 0;
        for (int it = 0; it < 30; ++it) {
            std::vector<float128> c_prime(M - 1, float128(0));
            std::vector<float128> d_prime(M, float128(0));
            float128 denom = diag[0] - shift;
            c_prime[0] = off[0] / denom;
            d_prime[0] = v[0] / denom;
            for (int i = 1; i < M; ++i) {
                denom = (diag[i] - shift) - off[i - 1] * c_prime[i - 1];
                if (i + 1 < M) c_prime[i] = off[i] / denom;
                d_prime[i] = (v[i] - off[i - 1] * d_prime[i - 1]) / denom;
            }
            std::vector<float128> w_vec(M);
            w_vec[M - 1] = d_prime[M - 1];
            for (int i = M - 2; i >= 0; --i) {
                w_vec[i] = d_prime[i] - c_prime[i] * w_vec[i + 1];
            }
            float128 nrm2 = 0;
            for (int i = 0; i < M; ++i) nrm2 += w_vec[i] * w_vec[i];
            const float128 inv = float128(1) / sqrt(nrm2);
            for (int i = 0; i < M; ++i) v[i] = w_vec[i] * inv;

            if (it >= 3) {
                float128 rq = 0;
                for (int i = 0; i < M; ++i) {
                    float128 Av_i = diag[i] * v[i];
                    if (i > 0) Av_i += off[i - 1] * v[i - 1];
                    if (i + 1 < M) Av_i += off[i] * v[i + 1];
                    rq += v[i] * Av_i;
                }
                shift = rq;
            }
        }
        if (v[0] < float128(0)) {
            for (int i = 0; i < M; ++i) v[i] = -v[i];
        }

        leg_c.resize(M);
        for (int k = 0; k < M; ++k) {
            leg_c[k] = v[k] * sqrt(float128(2 * k) + float128("0.5"));
        }
        active_terms = M;
        const float128 psi0_at_0 = eval_unnormalized(float128(0));
        for (int k = 0; k < M; ++k) {
            leg_c[k] /= psi0_at_0;
            if (k > 8 && abs(leg_c[k]) > float128("1e-36")) {
                active_terms = k + 1;
            }
        }
        active_terms = std::min(M, active_terms + 2);
        lambda0 = float128(2) * leg_c[0];
    }

    float128 eval_unnormalized(float128 z) const {
        const int max_n = 2 * (active_terms - 1);
        float128 p_nm2 = 1;
        float128 p_nm1 = z;
        float128 acc = leg_c[0] * p_nm2;
        for (int n = 2; n <= max_n; ++n) {
            const float128 p_n =
                (float128(2 * n - 1) * z * p_nm1 - float128(n - 1) * p_nm2) /
                float128(n);
            if ((n & 1) == 0) {
                acc += leg_c[n / 2] * p_n;
            }
            p_nm2 = p_nm1;
            p_nm1 = p_n;
        }
        return acc;
    }

    float128 eval_scaled(float128 z) const {
        return scale_pow2 * eval_unnormalized(z);
    }
};

float128 eval_poly(const std::vector<float128> &c, float128 u) {
    float128 acc = 0;
    for (int k = (int)c.size() - 1; k >= 0; --k) {
        acc = acc * u + c[k];
    }
    return acc;
}

// 6-stage minimax polynomial solver on t in [-1, 1]:
//   Stage 1: Standard Chebyshev nodes L^2 regression
//   Stage 2: Uniform nodes L^4 IRLS
//   Stage 3: Uniform nodes L^8 IRLS
//   Stage 4: Uniform nodes L^12 IRLS
//   Stage 5: Uniform nodes L^16 IRLS
//   Stage 6: Remez exchange fine-tune with local-extremum detection + golden-section refinement
template <typename Func>
std::vector<float128> fit_minimax_6stage_unit(Func f,
                                              int deg,
                                              float128 *out_max_err = nullptr,
                                              float128 *out_cheb_err = nullptr) {
    using boost::multiprecision::abs;
    using boost::multiprecision::cos;
    using boost::multiprecision::pow;

    const int n_terms = deg + 1;

    // Stage 1: Standard Chebyshev nodes L^2 regression
    const int n_cheb = std::max(256, 24 * n_terms);
    std::vector<float128> u_cheb(n_cheb);
    std::vector<float128> y_cheb(n_cheb);
    std::vector<float128> w_cheb(n_cheb, float128(1));
    std::vector<float128> V_cheb((size_t)n_cheb * (size_t)n_terms);

    for (int i = 0; i < n_cheb; ++i) {
        const float128 theta =
            pi_q() * (float128(2 * i + 1)) / float128(2 * n_cheb);
        const float128 u = -cos(theta);
        u_cheb[i] = u;
        y_cheb[i] = f(u);
        float128 p = 1;
        for (int k = 0; k < n_terms; ++k) {
            V_cheb[(size_t)i * (size_t)n_terms + (size_t)k] = p;
            p *= u;
        }
    }

    std::vector<float128> coeff(n_terms, float128(0));
    solve_weighted_ls_qr(
        V_cheb, y_cheb, w_cheb, n_cheb, n_terms, float128(0), coeff);

    // Dense uniform grid on [-1, 1] for Stages 2..6
    const int n_unif = 1025;
    std::vector<float128> u_unif(n_unif);
    std::vector<float128> y_unif(n_unif);
    std::vector<float128> V_unif((size_t)n_unif * (size_t)n_terms);

    for (int i = 0; i < n_unif; ++i) {
        const float128 u =
            float128(-1) + float128(2 * i) / float128(n_unif - 1);
        u_unif[i] = u;
        y_unif[i] = f(u);
        float128 p = 1;
        for (int k = 0; k < n_terms; ++k) {
            V_unif[(size_t)i * (size_t)n_terms + (size_t)k] = p;
            p *= u;
        }
    }

    auto compute_max_err = [&](const std::vector<float128> &c) -> float128 {
        float128 mx = 0;
        for (int i = 0; i < n_unif; ++i) {
            const float128 err = abs(eval_poly(c, u_unif[i]) - y_unif[i]);
            if (err > mx) mx = err;
        }
        return mx;
    };

    float128 best_err = compute_max_err(coeff);
    if (out_cheb_err) *out_cheb_err = best_err;
    std::vector<float128> best_coeff = coeff;

    // Stages 2..5: Uniform nodes L^4 -> L^8 -> L^12 -> L^16 IRLS
    const int p_schedule[4] = {4, 8, 12, 16};
    std::vector<float128> weights(n_unif, float128(1));

    for (int s_idx = 0; s_idx < 4; ++s_idx) {
        const float128 p_norm = float128(p_schedule[s_idx]);
        const float128 mix = (p_schedule[s_idx] <= 8) ? float128("0.5")
                                                      : float128("0.35");
        for (int iter = 0; iter < 20; ++iter) {
            std::vector<float128> res(n_unif);
            float128 max_r = 0;
            for (int i = 0; i < n_unif; ++i) {
                const float128 r = abs(eval_poly(coeff, u_unif[i]) - y_unif[i]);
                res[i] = r;
                if (r > max_r) max_r = r;
            }
            if (max_r < best_err) {
                best_err = max_r;
                best_coeff = coeff;
            }
            if (!(max_r > float128(0))) break;

            const float128 eps_r = max_r * float128("1e-4");
            float128 w_sum = 0;
            for (int i = 0; i < n_unif; ++i) {
                const float128 r_reg =
                    std::max(res[i], eps_r) / max_r;
                const float128 w_new = pow(r_reg, p_norm - float128(2));
                weights[i] =
                    (float128(1) - mix) * weights[i] + mix * w_new;
                w_sum += weights[i];
            }
            const float128 w_scale = float128(n_unif) / w_sum;
            for (int i = 0; i < n_unif; ++i) weights[i] *= w_scale;

            std::vector<float128> cand;
            if (!solve_weighted_ls_qr(V_unif,
                                      y_unif,
                                      weights,
                                      n_unif,
                                      n_terms,
                                      float128("1e-28"),
                                      cand)) {
                break;
            }
            for (int k = 0; k < n_terms; ++k) {
                coeff[k] =
                    (float128(1) - mix) * coeff[k] + mix * cand[k];
            }
        }
    }
    coeff = best_coeff;

    // Stage 6: Remez exchange fine-tuning using local extrema (endpoints + interior turning points)
    const int n_ext = n_terms + 1;
    const float128 phi =
        (boost::multiprecision::sqrt(float128(5)) - float128(1)) *
        float128("0.5");

    for (int rem_it = 0; rem_it < 15; ++rem_it) {
        std::vector<float128> r_grid(n_unif);
        for (int i = 0; i < n_unif; ++i) {
            r_grid[i] = eval_poly(coeff, u_unif[i]) - y_unif[i];
        }

        // Collect all local extrema of r_grid (including endpoints 0 and n_unif - 1)
        std::vector<int> ext_indices;
        ext_indices.push_back(0);
        for (int i = 1; i + 1 < n_unif; ++i) {
            const float128 d_prev = r_grid[i] - r_grid[i - 1];
            const float128 d_next = r_grid[i + 1] - r_grid[i];
            if ((d_prev >= float128(0) && d_next <= float128(0)) ||
                (d_prev <= float128(0) && d_next >= float128(0))) {
                if (abs(d_prev) > float128(0) || abs(d_next) > float128(0)) {
                    ext_indices.push_back(i);
                }
            }
        }
        ext_indices.push_back(n_unif - 1);

        struct Extremum {
            float128 u;
            float128 r;
            int sign;
        };

        auto refine_idx = [&](int idx) -> Extremum {
            float128 u_best = u_unif[idx];
            float128 r_best = r_grid[idx];
            if (idx > 0 && idx + 1 < n_unif) {
                float128 lo = u_unif[idx - 1];
                float128 hi = u_unif[idx + 1];
                float128 x1 = hi - phi * (hi - lo);
                float128 x2 = lo + phi * (hi - lo);
                float128 f1 = abs(eval_poly(coeff, x1) - f(x1));
                float128 f2 = abs(eval_poly(coeff, x2) - f(x2));
                for (int g = 0; g < 18; ++g) {
                    if (f1 > f2) {
                        hi = x2;
                        x2 = x1;
                        f2 = f1;
                        x1 = hi - phi * (hi - lo);
                        f1 = abs(eval_poly(coeff, x1) - f(x1));
                    } else {
                        lo = x1;
                        x1 = x2;
                        f1 = f2;
                        x2 = lo + phi * (hi - lo);
                        f2 = abs(eval_poly(coeff, x2) - f(x2));
                    }
                }
                const float128 u_mid = float128("0.5") * (lo + hi);
                const float128 r_mid = eval_poly(coeff, u_mid) - f(u_mid);
                if (abs(r_mid) > abs(r_best)) {
                    u_best = u_mid;
                    r_best = r_mid;
                }
            }
            const int sgn = (r_best >= float128(0)) ? 1 : -1;
            return {u_best, r_best, sgn};
        };

        // Collapse adjacent same-sign extrema by keeping the larger magnitude one
        std::vector<Extremum> alt_ext;
        for (int idx : ext_indices) {
            Extremum e = refine_idx(idx);
            if (alt_ext.empty() || alt_ext.back().sign != e.sign) {
                alt_ext.push_back(e);
            } else if (abs(e.r) > abs(alt_ext.back().r)) {
                alt_ext.back() = e;
            }
        }

        if ((int)alt_ext.size() < n_ext) break;
        while ((int)alt_ext.size() > n_ext) {
            if (abs(alt_ext.front().r) <= abs(alt_ext.back().r)) {
                alt_ext.erase(alt_ext.begin());
            } else {
                alt_ext.pop_back();
            }
        }

        // Solve exact Remez equioscillation system:
        // sum_{k=0}^deg a_k u_m^k + (-1)^m * s0 * E = f(u_m)
        const int sys_n = n_ext;
        std::vector<float128> A((size_t)sys_n * (size_t)sys_n, float128(0));
        std::vector<float128> rhs(sys_n, float128(0));
        const float128 s0 = (alt_ext[0].r >= float128(0)) ? float128(-1)
                                                          : float128(1);
        for (int m = 0; m < sys_n; ++m) {
            float128 p = 1;
            for (int k = 0; k < n_terms; ++k) {
                A[m * sys_n + k] = p;
                p *= alt_ext[m].u;
            }
            A[m * sys_n + n_terms] = (m % 2 == 0) ? s0 : -s0;
            rhs[m] = f(alt_ext[m].u);
        }

        std::vector<float128> sol;
        if (!solve_linear_system(A, rhs, sys_n, sol)) break;
        std::vector<float128> cand(sol.begin(), sol.begin() + n_terms);
        const float128 cand_err = compute_max_err(cand);
        if (cand_err <= best_err * float128("1.0001")) {
            coeff = cand;
            if (cand_err < best_err) {
                best_err = cand_err;
                best_coeff = cand;
            }
        } else {
            break;
        }
    }

    if (out_max_err) *out_max_err = best_err;
    return best_coeff;
}

// Simulated Dekker double-float (df_t = 2xFP32) arithmetic on host for exact verification
struct HostDf {
    float hi;
    float lo;
};

HostDf host_df_from_q(float128 q) {
    float hi = (float)q;
    float lo = (float)(q - float128(hi));
    return {hi, lo};
}

HostDf host_two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return {s, e};
}

HostDf host_quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return {s, e};
}

HostDf host_df_add(HostDf a, HostDf b) {
    HostDf s = host_two_sum(a.hi, b.hi);
    s.lo += (a.lo + b.lo);
    return host_quick_two_sum(s.hi, s.lo);
}

HostDf host_df_mul(HostDf a, HostDf b) {
    float p = a.hi * b.hi;
    float e = std::fma(a.hi, b.hi, -p);
    e += a.hi * b.lo + a.lo * b.hi;
    return host_quick_two_sum(p, e);
}

struct PanelPoly {
    std::vector<float128> even_c; // E_j(t) coefficients in t = u^2
    std::vector<float128> odd_c;  // O_j(t) coefficients in t = u^2
    float128 max_err_q;
    float128 cheb_err_q;
};

struct ConfigTable {
    std::string prec_name;
    int w;
    bool mode43;
    int deg;          // Even polynomial degree d = 2*K
    int n_even;       // K + 1
    int n_odd;        // K
    int scale_exp;
    float128 c;
    float128 prefac;      // Closed-form self-FT prefactor = hat{phi}_{half}(0) / phi(0)
    float128 inv_prefac;  // 1 / prefac
    float128 eta_scale;   // pi * w / c (so eta_k = eta_scale * (k - output_shift) / N_fft)
    std::vector<PanelPoly> half_panels; // ceil(w/2) panels: j = 0 .. floor((w-1)/2)
    double max_target_err;
    double max_deconv_rel_err;
};

ConfigTable build_config(const std::string &prec_name,
                         int w,
                         bool mode43,
                         int deg,
                         int max_exp) {
    using boost::multiprecision::abs;
    PswfOracle oracle;
    oracle.init(w, mode43, max_exp);

    ConfigTable cfg;
    cfg.prec_name = prec_name;
    cfg.w = w;
    cfg.mode43 = mode43;
    cfg.deg = deg;
    cfg.n_even = deg / 2 + 1;
    cfg.n_odd = deg / 2;
    cfg.scale_exp = oracle.scale_exp;
    cfg.c = oracle.c;

    const int n_left = w / 2;
    const int n_half = (w + 1) / 2;
    cfg.half_panels.resize(n_half);

    // Fit all half-panels j = 0 .. n_half - 1 directly on u in [-1, 1]
    for (int j = 0; j < n_half; ++j) {
        auto f_panel = [&](float128 u) -> float128 {
            const float128 z =
                (u + float128(2 * j + 1 - w)) / float128(w);
            return oracle.eval_scaled(z);
        };
        float128 max_err = 0, cheb_err = 0;
        std::vector<float128> full_c =
            fit_minimax_6stage_unit(f_panel, deg, &max_err, &cheb_err);

        PanelPoly pp;
        pp.even_c.resize(cfg.n_even);
        pp.odd_c.resize(cfg.n_odd);
        for (int m = 0; m < cfg.n_even; ++m) pp.even_c[m] = full_c[2 * m];
        for (int m = 0; m < cfg.n_odd; ++m) {
            // For odd w, the center panel j == n_left is symmetric on [-1, 1], so odd terms vanish
            pp.odd_c[m] = (j < n_left) ? full_c[2 * m + 1] : float128(0);
        }
        pp.max_err_q = max_err;
        pp.cheb_err_q = cheb_err;
        cfg.half_panels[j] = pp;
    }

    // Closed-form integral of the piecewise polynomial matching src/nufft1.c:2035:
    // hat{phi}_{half}(0) = w * int_0^1 phi(z) dz = (w/2) * int_{-1}^1 phi(z) dz
    //                    = (1/2) * sum_{j=0}^{w-1} int_{-1}^1 P_j(u) du
    //                    = sum_{j=0}^{w-1} sum_{m=0}^{K} a_{j, 2m} / (2m + 1)
    float128 poly_int_half = 0;
    for (int j = 0; j < n_left; ++j) {
        float128 panel_int = 0;
        for (int m = 0; m < cfg.n_even; ++m) {
            panel_int += cfg.half_panels[j].even_c[m] / float128(2 * m + 1);
        }
        poly_int_half += float128(2) * panel_int;
    }
    if (w % 2 == 1) {
        const int j_mid = w / 2;
        float128 panel_int = 0;
        for (int m = 0; m < cfg.n_even; ++m) {
            panel_int += cfg.half_panels[j_mid].even_c[m] / float128(2 * m + 1);
        }
        poly_int_half += panel_int;
    }

    cfg.prefac = poly_int_half / oracle.scale_pow2;
    cfg.inv_prefac = float128(1) / cfg.prefac;
    cfg.eta_scale = pi_q() * float128(w) / cfg.c;

    // Simulate target-precision Horner evaluation across 500 test points per panel
    double max_t_err = 0.0;
    for (int j = 0; j < n_half; ++j) {
        const PanelPoly &pp = cfg.half_panels[j];
        for (int i = 0; i <= 500; ++i) {
            const float128 u_q =
                float128(-1) + float128(2 * i) / float128(500);
            const float128 z_q =
                (u_q + float128(2 * j + 1 - w)) / float128(w);
            const float128 exact_v = oracle.eval_scaled(z_q);

            float128 approx_v = 0;
            if (prec_name == "fp16") {
                _Float16 u_h = (_Float16)(float)u_q;
                _Float16 t_h = u_h * u_h;
                _Float16 e_acc = (_Float16)(float)pp.even_c[cfg.n_even - 1];
                for (int m = cfg.n_even - 2; m >= 0; --m) {
                    e_acc = e_acc * t_h + (_Float16)(float)pp.even_c[m];
                }
                _Float16 o_acc = (_Float16)0.0f;
                if (j < n_left) {
                    o_acc = (_Float16)(float)pp.odd_c[cfg.n_odd - 1];
                    for (int m = cfg.n_odd - 2; m >= 0; --m) {
                        o_acc = o_acc * t_h + (_Float16)(float)pp.odd_c[m];
                    }
                }
                _Float16 p_h = e_acc + u_h * o_acc;
                approx_v = float128((float)p_h);
            } else if (prec_name == "fp32") {
                float u_f = (float)u_q;
                float t_f = u_f * u_f;
                float e_acc = (float)pp.even_c[cfg.n_even - 1];
                for (int m = cfg.n_even - 2; m >= 0; --m) {
                    e_acc = std::fma(e_acc, t_f, (float)pp.even_c[m]);
                }
                float o_acc = 0.0f;
                if (j < n_left) {
                    o_acc = (float)pp.odd_c[cfg.n_odd - 1];
                    for (int m = cfg.n_odd - 2; m >= 0; --m) {
                        o_acc = std::fma(o_acc, t_f, (float)pp.odd_c[m]);
                    }
                }
                float p_f = std::fma(u_f, o_acc, e_acc);
                approx_v = float128(p_f);
            } else if (prec_name == "df") {
                HostDf u_df = host_df_from_q(u_q);
                HostDf t_df = host_df_mul(u_df, u_df);
                HostDf e_acc = host_df_from_q(pp.even_c[cfg.n_even - 1]);
                for (int m = cfg.n_even - 2; m >= 0; --m) {
                    e_acc = host_df_add(host_df_mul(e_acc, t_df),
                                        host_df_from_q(pp.even_c[m]));
                }
                HostDf o_acc = {0.0f, 0.0f};
                if (j < n_left) {
                    o_acc = host_df_from_q(pp.odd_c[cfg.n_odd - 1]);
                    for (int m = cfg.n_odd - 2; m >= 0; --m) {
                        o_acc = host_df_add(host_df_mul(o_acc, t_df),
                                            host_df_from_q(pp.odd_c[m]));
                    }
                }
                HostDf p_df = host_df_add(e_acc, host_df_mul(u_df, o_acc));
                approx_v = float128(p_df.hi) + float128(p_df.lo);
            } else {
                double u_d = (double)u_q;
                double t_d = u_d * u_d;
                double e_acc = (double)pp.even_c[cfg.n_even - 1];
                for (int m = cfg.n_even - 2; m >= 0; --m) {
                    e_acc = std::fma(e_acc, t_d, (double)pp.even_c[m]);
                }
                double o_acc = 0.0;
                if (j < n_left) {
                    o_acc = (double)pp.odd_c[cfg.n_odd - 1];
                    for (int m = cfg.n_odd - 2; m >= 0; --m) {
                        o_acc = std::fma(o_acc, t_d, (double)pp.odd_c[m]);
                    }
                }
                double p_d = std::fma(u_d, o_acc, e_acc);
                approx_v = float128(p_d);
            }

            double rel_err =
                (double)(abs(approx_v - exact_v) / oracle.scale_pow2);
            if (rel_err > max_t_err) max_t_err = rel_err;
        }
    }
    cfg.max_target_err = max_t_err;

    // Verify self-FT deconvolution identity against exact float128 Legendre eigenvalue
    const float128 eta_max =
        mode43 ? (float128("0.375") * cfg.eta_scale)
               : (float128("0.25") * cfg.eta_scale);
    double max_dec_err = 0.0;
    for (int i = 0; i <= 500; ++i) {
        const float128 eta =
            eta_max * float128(i) / float128(500);
        const float128 exact_ft =
            (float128(w) * float128("0.5") * oracle.lambda0) *
            oracle.eval_scaled(eta);

        const float128 s =
            float128("0.5") * float128(w) * (float128(1) - eta);
        int j = (int)s;
        if (j >= n_half) j = n_half - 1;
        if (j < 0) j = 0;
        const float128 u = float128(2) * (s - float128(j)) - float128(1);
        const float128 t = u * u;
        const PanelPoly &pp = cfg.half_panels[j];
        float128 e_v = eval_poly(pp.even_c, t);
        float128 o_v = (j < n_left) ? eval_poly(pp.odd_c, t) : float128(0);
        const float128 poly_ft = cfg.prefac * (e_v + u * o_v);
        const double r_err = (double)(abs(poly_ft - exact_ft) / abs(exact_ft));
        if (r_err > max_dec_err) max_dec_err = r_err;
    }
    cfg.max_deconv_rel_err = max_dec_err;

    return cfg;
}

std::string fmt_f32(float v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9ef", (double)v);
    return std::string(buf);
}

std::string fmt_f64(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17e", v);
    return std::string(buf);
}

void emit_header(const std::string &path,
                 const std::vector<ConfigTable> &configs) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to open output file: " << path << "\n";
        std::exit(1);
    }

    out << "#ifndef SYCL_NUFFT_PSWF_COEFFS_HPP\n";
    out << "#define SYCL_NUFFT_PSWF_COEFFS_HPP\n\n";
    out << "// Auto-generated by src/miscellaneous/pswf_ppoly_remez_lut.cpp\n";
    out << "// Hardcoded piecewise minimax polynomials for PSWF_0^c(z) (scaled by 2^e)\n";
    out << "// and FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution constants.\n";
    out << "// Pipeline: Chebyshev L^2 -> Uniform L^4 -> L^8 -> L^12 -> L^16 IRLS -> Remez.\n\n";
    out << "#include \"common/math.hpp\"\n\n";
    out << "namespace sycl_nufft {\n\n";
    out << "#ifndef SYCL_NUFFT_PSWF_MODE_DEFINED\n";
    out << "#define SYCL_NUFFT_PSWF_MODE_DEFINED 1\n";
    out << "enum class PswfMode : int {\n";
    out << "    PSWF21 = 0, // sigma = 2.0,  N_out = N,     output_shift = N/2,   N_fft = bitceil(2N)\n";
    out << "    PSWF43 = 1  // sigma = 4/3,  N_out = 3N/2,  output_shift = 3N/4,  N_fft = bitceil(2N)\n";
    out << "};\n";
    out << "#endif\n\n";

    out << "template <int W, int NEven, int NOdd>\n";
    out << "struct PswfPiecewiseTableF16 {\n";
    out << "    static constexpr int width = W;\n";
    out << "    static constexpr int n_half = (W + 1) / 2;\n";
    out << "    static constexpr int n_left = W / 2;\n";
    out << "    static constexpr int n_even = NEven;\n";
    out << "    static constexpr int n_odd = NOdd;\n";
    out << "    int scale_exp;\n";
    out << "    float c;\n";
    out << "    float prefac;\n";
    out << "    float inv_prefac;\n";
    out << "    float eta_scale;\n";
    out << "    float even[n_half][NEven];\n";
    out << "    float odd[n_half][NOdd];\n";
    out << "};\n\n";

    out << "template <int W, int NEven, int NOdd>\n";
    out << "struct PswfPiecewiseTableF32 {\n";
    out << "    static constexpr int width = W;\n";
    out << "    static constexpr int n_half = (W + 1) / 2;\n";
    out << "    static constexpr int n_left = W / 2;\n";
    out << "    static constexpr int n_even = NEven;\n";
    out << "    static constexpr int n_odd = NOdd;\n";
    out << "    int scale_exp;\n";
    out << "    float c;\n";
    out << "    float prefac;\n";
    out << "    float inv_prefac;\n";
    out << "    float eta_scale;\n";
    out << "    float even[n_half][NEven];\n";
    out << "    float odd[n_half][NOdd];\n";
    out << "};\n\n";

    out << "template <int W, int NEven, int NOdd>\n";
    out << "struct PswfPiecewiseTableDF {\n";
    out << "    static constexpr int width = W;\n";
    out << "    static constexpr int n_half = (W + 1) / 2;\n";
    out << "    static constexpr int n_left = W / 2;\n";
    out << "    static constexpr int n_even = NEven;\n";
    out << "    static constexpr int n_odd = NOdd;\n";
    out << "    int scale_exp;\n";
    out << "    sycl_fft::df_t c;\n";
    out << "    sycl_fft::df_t prefac;\n";
    out << "    sycl_fft::df_t inv_prefac;\n";
    out << "    sycl_fft::df_t eta_scale;\n";
    out << "    sycl_fft::df_t even[n_half][NEven];\n";
    out << "    sycl_fft::df_t odd[n_half][NOdd];\n";
    out << "};\n\n";

    out << "template <int W, int NEven, int NOdd>\n";
    out << "struct PswfPiecewiseTableF64 {\n";
    out << "    static constexpr int width = W;\n";
    out << "    static constexpr int n_half = (W + 1) / 2;\n";
    out << "    static constexpr int n_left = W / 2;\n";
    out << "    static constexpr int n_even = NEven;\n";
    out << "    static constexpr int n_odd = NOdd;\n";
    out << "    int scale_exp;\n";
    out << "    double c;\n";
    out << "    double prefac;\n";
    out << "    double inv_prefac;\n";
    out << "    double eta_scale;\n";
    out << "    double even[n_half][NEven];\n";
    out << "    double odd[n_half][NOdd];\n";
    out << "};\n\n";

    for (size_t idx = 0; idx < configs.size(); idx += 2) {
        const ConfigTable &c21 = configs[idx];
        const ConfigTable &c43 = configs[idx + 1];
        const int w = c21.w;
        const int ne = c21.n_even;
        const int no = c21.n_odd;
        const int nh = (w + 1) / 2;

        std::string struct_type;
        if (c21.prec_name == "fp16") struct_type = "PswfPiecewiseTableF16";
        else if (c21.prec_name == "fp32") struct_type = "PswfPiecewiseTableF32";
        else if (c21.prec_name == "df") struct_type = "PswfPiecewiseTableDF";
        else struct_type = "PswfPiecewiseTableF64";

        out << "// " << c21.prec_name << " w=" << w
            << " (deg=" << c21.deg << ", scale_exp={21:" << c21.scale_exp
            << ", 43:" << c43.scale_exp << "})\n";
        out << "inline constexpr " << struct_type << "<" << w << ", " << ne
            << ", " << no << "> kPswfTable_" << c21.prec_name << "_w" << w
            << "[2] = {\n";

        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            const ConfigTable &cfg = (m_idx == 0) ? c21 : c43;
            out << "    // Mode " << (m_idx == 0 ? "PSWF21" : "PSWF43") << "\n";
            out << "    {\n";
            out << "        " << cfg.scale_exp << ",\n";

            if (cfg.prec_name == "fp16" || cfg.prec_name == "fp32") {
                out << "        " << fmt_f32((float)cfg.c) << ",\n";
                out << "        " << fmt_f32((float)cfg.prefac) << ",\n";
                out << "        " << fmt_f32((float)cfg.inv_prefac) << ",\n";
                out << "        " << fmt_f32((float)cfg.eta_scale) << ",\n";
                out << "        {\n";
                for (int j = 0; j < nh; ++j) {
                    out << "            {";
                    for (int m = 0; m < ne; ++m) {
                        float v = (cfg.prec_name == "fp16")
                                      ? (float)((_Float16)(float)cfg.half_panels[j].even_c[m])
                                      : (float)cfg.half_panels[j].even_c[m];
                        out << fmt_f32(v) << (m + 1 < ne ? ", " : "");
                    }
                    out << "}" << (j + 1 < nh ? ",\n" : "\n");
                }
                out << "        },\n";
                out << "        {\n";
                for (int j = 0; j < nh; ++j) {
                    out << "            {";
                    for (int m = 0; m < no; ++m) {
                        float v = (cfg.prec_name == "fp16")
                                      ? (float)((_Float16)(float)cfg.half_panels[j].odd_c[m])
                                      : (float)cfg.half_panels[j].odd_c[m];
                        out << fmt_f32(v) << (m + 1 < no ? ", " : "");
                    }
                    out << "}" << (j + 1 < nh ? ",\n" : "\n");
                }
                out << "        }\n";
            } else if (cfg.prec_name == "df") {
                auto fmt_df = [&](float128 q) -> std::string {
                    HostDf d = host_df_from_q(q);
                    return "{" + fmt_f32(d.hi) + ", " + fmt_f32(d.lo) + "}";
                };
                out << "        " << fmt_df(cfg.c) << ",\n";
                out << "        " << fmt_df(cfg.prefac) << ",\n";
                out << "        " << fmt_df(cfg.inv_prefac) << ",\n";
                out << "        " << fmt_df(cfg.eta_scale) << ",\n";
                out << "        {\n";
                for (int j = 0; j < nh; ++j) {
                    out << "            {";
                    for (int m = 0; m < ne; ++m) {
                        out << fmt_df(cfg.half_panels[j].even_c[m])
                            << (m + 1 < ne ? ", " : "");
                    }
                    out << "}" << (j + 1 < nh ? ",\n" : "\n");
                }
                out << "        },\n";
                out << "        {\n";
                for (int j = 0; j < nh; ++j) {
                    out << "            {";
                    for (int m = 0; m < no; ++m) {
                        out << fmt_df(cfg.half_panels[j].odd_c[m])
                            << (m + 1 < no ? ", " : "");
                    }
                    out << "}" << (j + 1 < nh ? ",\n" : "\n");
                }
                out << "        }\n";
            } else {
                out << "        " << fmt_f64((double)cfg.c) << ",\n";
                out << "        " << fmt_f64((double)cfg.prefac) << ",\n";
                out << "        " << fmt_f64((double)cfg.inv_prefac) << ",\n";
                out << "        " << fmt_f64((double)cfg.eta_scale) << ",\n";
                out << "        {\n";
                for (int j = 0; j < nh; ++j) {
                    out << "            {";
                    for (int m = 0; m < ne; ++m) {
                        out << fmt_f64((double)cfg.half_panels[j].even_c[m])
                            << (m + 1 < ne ? ", " : "");
                    }
                    out << "}" << (j + 1 < nh ? ",\n" : "\n");
                }
                out << "        },\n";
                out << "        {\n";
                for (int j = 0; j < nh; ++j) {
                    out << "            {";
                    for (int m = 0; m < no; ++m) {
                        out << fmt_f64((double)cfg.half_panels[j].odd_c[m])
                            << (m + 1 < no ? ", " : "");
                    }
                    out << "}" << (j + 1 < nh ? ",\n" : "\n");
                }
                out << "        }\n";
            }
            out << "    }" << (m_idx == 0 ? ",\n" : "\n");
        }
        out << "};\n\n";
    }

    // Emit device-inline evaluators for spreading weights and GL-free self-FT deconvolution
    out << R"CPP(// ============================================================================
// Device-Inline Piecewise Minimax PSWF0 Evaluators (Spreading & GL-Free Deconv)
// ============================================================================

template <int W>
SYCL_FFT_FORCE_INLINE const auto& get_pswf_table_f16(PswfMode mode) {
    const int idx = (mode == PswfMode::PSWF43) ? 1 : 0;
    if constexpr (W == 4) return kPswfTable_fp16_w4[idx];
    else { static_assert(W == 5, "Unsupported FP16 PSWF width"); return kPswfTable_fp16_w5[idx]; }
}

template <int W>
SYCL_FFT_FORCE_INLINE const auto& get_pswf_table_f32(PswfMode mode) {
    const int idx = (mode == PswfMode::PSWF43) ? 1 : 0;
    if constexpr (W == 8) return kPswfTable_fp32_w8[idx];
    else { static_assert(W == 9, "Unsupported FP32 PSWF width"); return kPswfTable_fp32_w9[idx]; }
}

template <int W>
SYCL_FFT_FORCE_INLINE const auto& get_pswf_table_df(PswfMode mode) {
    const int idx = (mode == PswfMode::PSWF43) ? 1 : 0;
    if constexpr (W == 14) return kPswfTable_df_w14[idx];
    else if constexpr (W == 15) return kPswfTable_df_w15[idx];
    else { static_assert(W == 16, "Unsupported DF PSWF width"); return kPswfTable_df_w16[idx]; }
}

template <int W>
SYCL_FFT_FORCE_INLINE const auto& get_pswf_table_f64(PswfMode mode) {
    const int idx = (mode == PswfMode::PSWF43) ? 1 : 0;
    if constexpr (W == 16) return kPswfTable_fp64_w16[idx];
    else if constexpr (W == 17) return kPswfTable_fp64_w17[idx];
    else { static_assert(W == 18, "Unsupported FP64 PSWF width"); return kPswfTable_fp64_w18[idx]; }
}

// Evaluate all W spreading weights in native FP16 (sycl::half) using half-panel symmetry
template <int W, int NEven, int NOdd>
SYCL_FFT_FORCE_INLINE void eval_pswf_spread_weights_f16(
    sycl::half u,
    const PswfPiecewiseTableF16<W, NEven, NOdd>& tbl,
    sycl::half (&w_out)[W]) {
    const sycl::half t = u * u;
    constexpr int n_left = W / 2;
    #pragma unroll
    for (int j = 0; j < n_left; ++j) {
        sycl::half e_acc = static_cast<sycl::half>(tbl.even[j][NEven - 1]);
        #pragma unroll
        for (int m = NEven - 2; m >= 0; --m) {
            e_acc = e_acc * t + static_cast<sycl::half>(tbl.even[j][m]);
        }
        sycl::half o_acc = static_cast<sycl::half>(tbl.odd[j][NOdd - 1]);
        #pragma unroll
        for (int m = NOdd - 2; m >= 0; --m) {
            o_acc = o_acc * t + static_cast<sycl::half>(tbl.odd[j][m]);
        }
        const sycl::half uo = u * o_acc;
        w_out[j] = e_acc + uo;
        w_out[W - 1 - j] = e_acc - uo;
    }
    if constexpr ((W & 1) != 0) {
        constexpr int j_mid = W / 2;
        sycl::half e_acc = static_cast<sycl::half>(tbl.even[j_mid][NEven - 1]);
        #pragma unroll
        for (int m = NEven - 2; m >= 0; --m) {
            e_acc = e_acc * t + static_cast<sycl::half>(tbl.even[j_mid][m]);
        }
        w_out[j_mid] = e_acc;
    }
}

// Evaluate all W spreading weights in FP32 using hardware FMA and half-panel symmetry
template <int W, int NEven, int NOdd>
SYCL_FFT_FORCE_INLINE void eval_pswf_spread_weights_f32(
    float u,
    const PswfPiecewiseTableF32<W, NEven, NOdd>& tbl,
    float (&w_out)[W]) {
    const float t = u * u;
    constexpr int n_left = W / 2;
    #pragma unroll
    for (int j = 0; j < n_left; ++j) {
        float e_acc = tbl.even[j][NEven - 1];
        #pragma unroll
        for (int m = NEven - 2; m >= 0; --m) {
            e_acc = sycl::fma(e_acc, t, tbl.even[j][m]);
        }
        float o_acc = tbl.odd[j][NOdd - 1];
        #pragma unroll
        for (int m = NOdd - 2; m >= 0; --m) {
            o_acc = sycl::fma(o_acc, t, tbl.odd[j][m]);
        }
        const float uo = u * o_acc;
        w_out[j] = e_acc + uo;
        w_out[W - 1 - j] = e_acc - uo;
    }
    if constexpr ((W & 1) != 0) {
        constexpr int j_mid = W / 2;
        float e_acc = tbl.even[j_mid][NEven - 1];
        #pragma unroll
        for (int m = NEven - 2; m >= 0; --m) {
            e_acc = sycl::fma(e_acc, t, tbl.even[j_mid][m]);
        }
        w_out[j_mid] = e_acc;
    }
}

// Evaluate all W spreading weights in Dekker Double-Float (df_t) using half-panel symmetry.
// Unrolling (j, m) makes all table accesses compile-time constants (zero .local stack loads),
// while SYCL_FFT_DF_INLINE on df_add/df_mul prevents IR bloat.
template <int W, int NEven, int NOdd>
SYCL_FFT_DF_INLINE void eval_pswf_spread_weights_df(
    sycl_fft::df_t u,
    const PswfPiecewiseTableDF<W, NEven, NOdd>& tbl,
    sycl_fft::df_t (&w_out)[W]) {
    using namespace sycl_fft;
    const df_t t = df_sqr(u);
    constexpr int n_left = W / 2;
    #pragma unroll
    for (int j = 0; j < n_left; ++j) {
        df_t e_acc = tbl.even[j][NEven - 1];
        #pragma unroll
        for (int m = NEven - 2; m >= 0; --m) {
            e_acc = df_add(df_mul(e_acc, t), tbl.even[j][m]);
        }
        df_t o_acc = tbl.odd[j][NOdd - 1];
        #pragma unroll
        for (int m = NOdd - 2; m >= 0; --m) {
            o_acc = df_add(df_mul(o_acc, t), tbl.odd[j][m]);
        }
        const df_t uo = df_mul(u, o_acc);
        w_out[j] = df_add(e_acc, uo);
        w_out[W - 1 - j] = df_sub(e_acc, uo);
    }
    if constexpr ((W & 1) != 0) {
        constexpr int j_mid = W / 2;
        df_t e_acc = tbl.even[j_mid][NEven - 1];
        #pragma unroll
        for (int m = NEven - 2; m >= 0; --m) {
            e_acc = df_add(df_mul(e_acc, t), tbl.even[j_mid][m]);
        }
        w_out[j_mid] = e_acc;
    }
}

// Evaluate all W spreading weights in FP64 using hardware FMA and half-panel symmetry
template <int W, int NEven, int NOdd>
SYCL_FFT_FORCE_INLINE void eval_pswf_spread_weights_f64(
    double u,
    const PswfPiecewiseTableF64<W, NEven, NOdd>& tbl,
    double (&w_out)[W]) {
    const double t = u * u;
    constexpr int n_left = W / 2;
    #pragma unroll
    for (int j = 0; j < n_left; ++j) {
        double e_acc = tbl.even[j][NEven - 1];
        #pragma unroll
        for (int m = NEven - 2; m >= 0; --m) {
            e_acc = sycl::fma(e_acc, t, tbl.even[j][m]);
        }
        double o_acc = tbl.odd[j][NOdd - 1];
        #pragma unroll
        for (int m = NOdd - 2; m >= 0; --m) {
            o_acc = sycl::fma(o_acc, t, tbl.odd[j][m]);
        }
        const double uo = u * o_acc;
        w_out[j] = e_acc + uo;
        w_out[W - 1 - j] = e_acc - uo;
    }
    if constexpr ((W & 1) != 0) {
        constexpr int j_mid = W / 2;
        double e_acc = tbl.even[j_mid][NEven - 1];
        #pragma unroll
        for (int m = NEven - 2; m >= 0; --m) {
            e_acc = sycl::fma(e_acc, t, tbl.even[j_mid][m]);
        }
        w_out[j_mid] = e_acc;
    }
}

// FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution evaluators:
//   deconv(k') = inv_prefac / phi(eta_{k'}),  eta_{k'} = eta_scale * |k'| / N_fft.
// Note: FP16 evaluates the 1D scalar ratio in FP32 and rounds once to binary16
// (cvt.rn.f16.f32) so that no FP16->FP32 promotion (cvt.f32.f16) is ever emitted.
template <int W, int NEven, int NOdd>
SYCL_FFT_FORCE_INLINE sycl::half eval_pswf_deconv_f16(
    int abs_k_prime,
    float inv_nfft,
    const PswfPiecewiseTableF16<W, NEven, NOdd>& tbl) {
    constexpr int n_half = (W + 1) / 2;
    constexpr int n_left = W / 2;
    const float eta = (tbl.eta_scale * inv_nfft) * static_cast<float>(abs_k_prime);
    const float s = 0.5f * static_cast<float>(W) * (1.0f - eta);
    int j = static_cast<int>(s);
    if (j >= n_half) j = n_half - 1;
    if (j < 0) j = 0;
    const float u = 2.0f * (s - static_cast<float>(j)) - 1.0f;
    const float t = u * u;
    float e_acc = 0.0f;
    float o_acc = 0.0f;
    #pragma unroll
    for (int J = 0; J < n_half; ++J) {
        if (j == J) {
            e_acc = tbl.even[J][NEven - 1];
            #pragma unroll
            for (int m = NEven - 2; m >= 0; --m) {
                e_acc = sycl::fma(e_acc, t, tbl.even[J][m]);
            }
            if (J < n_left) {
                o_acc = tbl.odd[J][NOdd - 1];
                #pragma unroll
                for (int m = NOdd - 2; m >= 0; --m) {
                    o_acc = sycl::fma(o_acc, t, tbl.odd[J][m]);
                }
            }
        }
    }
    const float phi = sycl::fma(u, o_acc, e_acc);
    return static_cast<sycl::half>(tbl.inv_prefac / phi);
}

template <int W, int NEven, int NOdd>
SYCL_FFT_FORCE_INLINE float eval_pswf_deconv_f32(
    int abs_k_prime,
    float inv_nfft,
    const PswfPiecewiseTableF32<W, NEven, NOdd>& tbl) {
    constexpr int n_half = (W + 1) / 2;
    constexpr int n_left = W / 2;
    const float eta = (tbl.eta_scale * inv_nfft) * static_cast<float>(abs_k_prime);
    const float s = 0.5f * static_cast<float>(W) * (1.0f - eta);
    int j = static_cast<int>(s);
    if (j >= n_half) j = n_half - 1;
    if (j < 0) j = 0;
    const float u = 2.0f * (s - static_cast<float>(j)) - 1.0f;
    const float t = u * u;
    float e_acc = 0.0f;
    float o_acc = 0.0f;
    #pragma unroll
    for (int J = 0; J < n_half; ++J) {
        if (j == J) {
            e_acc = tbl.even[J][NEven - 1];
            #pragma unroll
            for (int m = NEven - 2; m >= 0; --m) {
                e_acc = sycl::fma(e_acc, t, tbl.even[J][m]);
            }
            if (J < n_left) {
                o_acc = tbl.odd[J][NOdd - 1];
                #pragma unroll
                for (int m = NOdd - 2; m >= 0; --m) {
                    o_acc = sycl::fma(o_acc, t, tbl.odd[J][m]);
                }
            }
        }
    }
    const float phi = sycl::fma(u, o_acc, e_acc);
    return tbl.inv_prefac / phi;
}

template <int W, int NEven, int NOdd>
SYCL_FFT_DF_INLINE sycl_fft::df_t eval_pswf_deconv_df(
    int abs_k_prime,
    float inv_nfft,
    const PswfPiecewiseTableDF<W, NEven, NOdd>& tbl) {
    using namespace sycl_fft;
    constexpr int n_half = (W + 1) / 2;
    constexpr int n_left = W / 2;
    const df_t eta = df_scale_pow2(df_mul_f(tbl.eta_scale, static_cast<float>(abs_k_prime)), inv_nfft);
    const df_t one_minus_eta = df_sub(df_make(1.0f, 0.0f), eta);
    const df_t s = df_scale_pow2(df_mul_f(one_minus_eta, static_cast<float>(W)), 0.5f);
    int j = static_cast<int>(s.hi);
    if (static_cast<float>(j) > s.hi || (static_cast<float>(j) == s.hi && s.lo < 0.0f)) --j;
    if (j >= n_half) j = n_half - 1;
    if (j < 0) j = 0;
    const df_t s_minus_j = df_sub(s, df_make(static_cast<float>(j), 0.0f));
    const df_t u = df_sub(df_scale_pow2(s_minus_j, 2.0f), df_make(1.0f, 0.0f));
    const df_t t = df_sqr(u);
    df_t e_acc = df_make(0.0f, 0.0f);
    df_t o_acc = df_make(0.0f, 0.0f);
    #pragma unroll
    for (int J = 0; J < n_half; ++J) {
        if (j == J) {
            e_acc = tbl.even[J][NEven - 1];
            #pragma unroll
            for (int m = NEven - 2; m >= 0; --m) {
                e_acc = df_add(df_mul(e_acc, t), tbl.even[J][m]);
            }
            if (J < n_left) {
                o_acc = tbl.odd[J][NOdd - 1];
                #pragma unroll
                for (int m = NOdd - 2; m >= 0; --m) {
                    o_acc = df_add(df_mul(o_acc, t), tbl.odd[J][m]);
                }
            }
        }
    }
    const df_t phi = df_add(e_acc, df_mul(u, o_acc));
    return df_div(tbl.inv_prefac, phi);
}

template <int W, int NEven, int NOdd>
SYCL_FFT_FORCE_INLINE double eval_pswf_deconv_f64(
    int abs_k_prime,
    double inv_nfft,
    const PswfPiecewiseTableF64<W, NEven, NOdd>& tbl) {
    constexpr int n_half = (W + 1) / 2;
    constexpr int n_left = W / 2;
    const double eta = (tbl.eta_scale * inv_nfft) * static_cast<double>(abs_k_prime);
    const double s = 0.5 * static_cast<double>(W) * (1.0 - eta);
    int j = static_cast<int>(s);
    if (j >= n_half) j = n_half - 1;
    if (j < 0) j = 0;
    const double u = 2.0 * (s - static_cast<double>(j)) - 1.0;
    const double t = u * u;
    double e_acc = 0.0;
    double o_acc = 0.0;
    #pragma unroll
    for (int J = 0; J < n_half; ++J) {
        if (j == J) {
            e_acc = tbl.even[J][NEven - 1];
            #pragma unroll
            for (int m = NEven - 2; m >= 0; --m) {
                e_acc = sycl::fma(e_acc, t, tbl.even[J][m]);
            }
            if (J < n_left) {
                o_acc = tbl.odd[J][NOdd - 1];
                #pragma unroll
                for (int m = NOdd - 2; m >= 0; --m) {
                    o_acc = sycl::fma(o_acc, t, tbl.odd[J][m]);
                }
            }
        }
    }
    const double phi = sycl::fma(u, o_acc, e_acc);
    return tbl.inv_prefac / phi;
}
)CPP";

    out << "} // namespace sycl_nufft\n\n";
    out << "#endif // SYCL_NUFFT_PSWF_COEFFS_HPP\n";
}

} // namespace

int main(int argc, char **argv) {
    std::string out_path = "../../offload/sycl/nufft/pswf_coeffs.hpp";
    if (argc > 1) out_path = argv[1];

    struct Spec {
        std::string prec;
        int w;
        int deg;
        int max_exp;
    };

    const std::vector<Spec> specs = {
        // FP16: w in {4, 5}, deg = 6 (4 even, 3 odd) -> minimax error < 2e-6 << FP16 eps
        {"fp16", 4, 6, 3},
        {"fp16", 5, 6, 3},
        // FP32: w in {8, 9}, deg = 8 (5 even, 4 odd) -> minimax error < 3e-9 << FP32 eps
        {"fp32", 8, 8, 6},
        {"fp32", 9, 8, 6},
        // DF: w in {14, 15, 16}, deg = 12 (7 even, 6 odd) -> minimax error < 8e-16 << DF eps
        {"df", 14, 12, 13},
        {"df", 15, 12, 13},
        {"df", 16, 12, 13},
        // FP64: w in {16, 17, 18}, deg = 14 (8 even, 7 odd) -> minimax error < 8e-19 << FP64 eps
        {"fp64", 16, 14, 13},
        {"fp64", 17, 14, 13},
        {"fp64", 18, 14, 13},
    };

    std::vector<ConfigTable> configs;
    std::printf("=========================================================================================\n");
    std::printf(" %-5s | w  | Mode   | deg | 2^e | Cheb L2 Err  | Remez L_inf  | Target Err   | Self-FT Err\n",
                "Prec");
    std::printf("-----------------------------------------------------------------------------------------\n");

    for (const auto &sp : specs) {
        for (int m = 0; m < 2; ++m) {
            const bool mode43 = (m == 1);
            ConfigTable cfg =
                build_config(sp.prec, sp.w, mode43, sp.deg, sp.max_exp);

            double max_cheb = 0.0;
            double max_remez = 0.0;
            const double scale = std::ldexp(1.0, cfg.scale_exp);
            for (const auto &pp : cfg.half_panels) {
                double ce = (double)pp.cheb_err_q / scale;
                double re = (double)pp.max_err_q / scale;
                if (ce > max_cheb) max_cheb = ce;
                if (re > max_remez) max_remez = re;
            }

            std::printf(" %-5s | %-2d | %-6s | %-3d | 2^%-2d| %.3e  | %.3e  | %.3e  | %.3e\n",
                        sp.prec.c_str(),
                        sp.w,
                        mode43 ? "PSWF43" : "PSWF21",
                        sp.deg,
                        cfg.scale_exp,
                        max_cheb,
                        max_remez,
                        cfg.max_target_err,
                        cfg.max_deconv_rel_err);
            configs.push_back(std::move(cfg));
        }
    }
    std::printf("=========================================================================================\n");

    emit_header(out_path, configs);
    std::printf("Wrote hardcoded piecewise minimax coefficient header to: %s\n",
                out_path.c_str());
    return 0;
}
