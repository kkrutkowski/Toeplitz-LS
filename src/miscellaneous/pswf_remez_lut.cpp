#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Boost Multiprecision for __float128 wrapper
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/float128.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

using float128 = boost::multiprecision::float128;
using namespace Eigen;

typedef Matrix<float128, Dynamic, Dynamic> MatrixXmp;
typedef Matrix<float128, Dynamic, 1> VectorXmp;

// ---------------------------------------------------------
// Math Utilities & High-Precision PSWF_0^c
// ---------------------------------------------------------
float128 legendre(int n, float128 x) {
    if (n == 0) return float128(1.0);
    if (n == 1) return x;
    float128 p0 = 1.0, p1 = x, p2;
    for (int i = 1; i < n; ++i) {
        p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1);
        p0 = p1;
        p1 = p2;
    }
    return p2;
}

float128 legendre_der(int n, float128 x) {
    if (n == 0) return float128(0.0);
    return n * (x * legendre(n, x) - legendre(n - 1, x)) / (x * x - 1.0);
}

VectorXmp get_pswf0_coeffs(float128 c, int n_terms = 50) {
    MatrixXmp M = MatrixXmp::Zero(n_terms, n_terms);
    float128 c2 = c * c;

    for (int i = 0; i < n_terms; ++i) {
        float128 k = 2 * i;
        float128 beta0 = (i == 0) ? float128(1.0) / 3.0 : ((k + 1) * (k + 1) / (2 * k + 3) + k * k / (2 * k - 1)) / (2 * k + 1);

        M(i, i) = k * (k + 1) + c2 * beta0;

        if (i < n_terms - 1) {
            float128 gamma0 = (k + 1) * (k + 2) / ((2 * k + 1) * (2 * k + 3));
            float128 scale = sqrt((k + 0.5) / (k + 2.5));
            M(i, i + 1) = c2 * gamma0 * scale;
            M(i + 1, i) = M(i, i + 1);
        }
    }

    SelfAdjointEigenSolver<MatrixXmp> eigensolver(M);
    VectorXmp evals = eigensolver.eigenvalues();
    MatrixXmp evecs = eigensolver.eigenvectors();

    int min_idx = 0;
    float128 min_val = evals(0);
    for (int i = 1; i < n_terms; ++i) {
        if (evals(i) < min_val) {
            min_val = evals(i);
            min_idx = i;
        }
    }

    VectorXmp eigenvector = evecs.col(min_idx);
    VectorXmp workdata(n_terms);
    for (int i = 0; i < n_terms; ++i) {
        workdata(i) = eigenvector(i) * sqrt(2 * i + 0.5);
    }
    return workdata;
}

float128 pswf0_eval(float128 x, const VectorXmp& workdata) {
    float128 val = 0.0;
    for (int j = 0; j < workdata.size(); ++j) {
        val += workdata(j) * legendre(2 * j, x);
    }
    return val;
}

float128 pswf0_der_eval(float128 x, const VectorXmp& workdata) {
    float128 val = 0.0;
    for (int j = 0; j < workdata.size(); ++j) {
        val += workdata(j) * legendre_der(2 * j, x);
    }
    return val;
}

// ---------------------------------------------------------
// Error Evaluation & Solvers
// ---------------------------------------------------------
float128 evaluate_dense_max_error(const VectorXmp& coeffs, std::function<float128(float128)> fx, std::function<float128(float128)> Wx,
                                  const std::vector<int>& powers, int eval_nodes) {
    float128 max_err = 0.0;
    for (int k = 0; k <= eval_nodes; ++k) {
        float128 x = float128(k) / float128(eval_nodes);
        float128 px = 0.0;
        for (size_t i = 0; i < powers.size(); ++i) {
            px += coeffs(i) * pow(x, powers[i]);
        }
        float128 err = abs(Wx(x) * px - fx(x));
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float128 bisection(std::function<float128(float128)> f, float128 a, float128 b, float128 tol = 1e-33) {
    float128 fa = f(a), fb = f(b);

    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fa * fb > 0) return (a + b) / 2.0;  // Fail-safe if bracket is lost

    while ((b - a) > tol) {
        float128 mid = (a + b) / 2.0;
        float128 fmid = f(mid);
        if (fmid == 0.0) return mid;

        if (fa * fmid < 0) {
            b = mid;
            fb = fmid;
        } else {
            a = mid;
            fa = fmid;
        }
    }
    return (a + b) / 2.0;
}

// ---------------------------------------------------------
// Context Wrapper
// ---------------------------------------------------------
struct EpsContext {
    std::string name;
    double eps;    // Used only to choose coefficient print precision.
    int w;         // Window width, supplied explicitly by the user.
    double sigma;  // Oversampling factor, supplied explicitly by the user.
    float128 c_param;
    VectorXmp workdata;
    float128 xv0;
};

EpsContext build_context(std::string name, double eps, int w, double sigma) {
    EpsContext ctx;
    ctx.name = name;
    ctx.eps = eps;
    ctx.w = w;
    ctx.sigma = sigma;
    ctx.c_param =
        boost::math::constants::pi<float128>() * float128(ctx.w) * (float128(1.0) - (float128(1.0) / (float128(2.0) * float128(ctx.sigma)))) - float128(0.05);
    int n_terms = static_cast<int>(ctx.c_param) + 60;
    ctx.workdata = get_pswf0_coeffs(ctx.c_param, n_terms);
    ctx.xv0 = float128(1.0) / pswf0_eval(float128(0.0), ctx.workdata);
    return ctx;
}

// ---------------------------------------------------------
// Execution Phases
// ---------------------------------------------------------
VectorXmp run_regression_single(EpsContext& ctx, int degree_z, int reg_nodes, int eval_nodes) {
    std::cout << "--- 1. IRLS Regression (L2 Cheb -> L4 -> L8 -> L12 -> L16) for " << ctx.name << " ---\n";
    std::cout << "Input w     : " << ctx.w << "\n";
    std::cout << "Input sigma : " << ctx.sigma << "\n";
    std::cout << "Computed c  : " << static_cast<double>(ctx.c_param) << "\n\n";

    auto fx = [&](float128 x) { return pswf0_eval(x, ctx.workdata) * ctx.xv0; };
    auto Wx = [&](float128 x) { return exp(-ctx.c_param * x * x / 2.0); };

    std::vector<int> powers_x;
    for (int i = 0; i <= degree_z; ++i) powers_x.push_back(2 * i);
    int n_unknowns = static_cast<int>(powers_x.size());

    MatrixXmp A_ls(reg_nodes, n_unknowns);
    VectorXmp Y_ls(reg_nodes);
    float128 pi = boost::math::constants::pi<float128>();

    // Stage 1: L2 least squares on Chebyshev nodes in z = x^2
    for (int k = 0; k < reg_nodes; ++k) {
        float128 z = 0.5 * (1.0 - cos((2 * k + 1) * pi / (2 * reg_nodes)));
        float128 x = sqrt(z);
        float128 gauss_val = Wx(x);
        Y_ls(k) = fx(x);
        for (int j = 0; j < n_unknowns; ++j) {
            A_ls(k, j) = gauss_val * pow(x, powers_x[j]);
        }
    }

    VectorXmp an = A_ls.completeOrthogonalDecomposition().solve(Y_ls);
    float128 err = evaluate_dense_max_error(an, fx, Wx, powers_x, eval_nodes);
    std::cout << "Target Z-Degree " << std::setw(2) << degree_z
              << " | Stage L2  (Cheb) Max Err: " << std::scientific << std::setprecision(5)
              << static_cast<double>(err) << "\n";

    // Precompute uniform nodes and targets for L^p IRLS stages
    std::vector<float128> ux(reg_nodes), uf(reg_nodes), uw(reg_nodes);
    MatrixXmp U_basis(reg_nodes, n_unknowns);
    for (int k = 0; k < reg_nodes; ++k) {
        ux[k] = float128(k) / float128(reg_nodes - 1);
        uf[k] = fx(ux[k]);
        uw[k] = Wx(ux[k]);
        for (int j = 0; j < n_unknowns; ++j) {
            U_basis(k, j) = uw[k] * pow(ux[k], powers_x[j]);
        }
    }

    // Stages 2-5: L^p IRLS on uniform nodes with Kahng damping alpha = 1 / (p - 1)
    const int lp_stages[4] = {4, 8, 12, 16};
    for (int s = 0; s < 4; ++s) {
        int p = lp_stages[s];
        float128 alpha = float128(1.0) / float128(p - 1);
        float128 half_exp = float128(p - 2) / float128(2.0);

        for (int it = 0; it < 12; ++it) {
            std::vector<float128> r(reg_nodes);
            float128 r_max = 0.0;
            for (int k = 0; k < reg_nodes; ++k) {
                float128 pred = 0.0;
                for (int j = 0; j < n_unknowns; ++j) pred += U_basis(k, j) * an(j);
                r[k] = pred - uf[k];
                float128 ar = abs(r[k]);
                if (ar > r_max) r_max = ar;
            }
            if (r_max == 0.0) break;

            for (int k = 0; k < reg_nodes; ++k) {
                float128 ratio = abs(r[k]) / r_max;
                if (ratio < float128(1e-3)) ratio = float128(1e-3);
                float128 row_scale = pow(ratio, half_exp);
                Y_ls(k) = row_scale * uf[k];
                for (int j = 0; j < n_unknowns; ++j) {
                    A_ls(k, j) = row_scale * U_basis(k, j);
                }
            }

            VectorXmp a_wls = A_ls.completeOrthogonalDecomposition().solve(Y_ls);
            VectorXmp a_new = (float128(1.0) - alpha) * an + alpha * a_wls;

            float128 step_inf = 0.0, an_inf = 0.0;
            for (int j = 0; j < n_unknowns; ++j) {
                float128 dj = abs(a_new(j) - an(j));
                float128 aj = abs(an(j));
                if (dj > step_inf) step_inf = dj;
                if (aj > an_inf) an_inf = aj;
            }
            an = a_new;
            if (an_inf > 0.0 && step_inf <= float128(1e-20) * an_inf) break;
        }

        err = evaluate_dense_max_error(an, fx, Wx, powers_x, eval_nodes);
        std::cout << "Target Z-Degree " << std::setw(2) << degree_z
                  << " | Stage L" << std::setw(2) << std::left << p << std::right
                  << " (Unif) Max Err: " << std::scientific << std::setprecision(5)
                  << static_cast<double>(err) << "\n";
    }
    std::cout << "\n";
    return an;
}

void run_remez_single(EpsContext& ctx, int d, VectorXmp initial_an, int eval_nodes) {
    std::cout << "--- 2. Remez Optimization for " << ctx.name << " ---\n";
    std::cout << ">>> Testing Remez for Z-degree " << d << "...\n";

    std::vector<int> powers_x;
    for (int i = 0; i <= d; ++i) powers_x.push_back(2 * i);
    int n_unknowns = static_cast<int>(powers_x.size());
    float128 pi = boost::math::constants::pi<float128>();

    auto fx = [&](float128 x) { return pswf0_eval(x, ctx.workdata) * ctx.xv0; };
    auto fx_der = [&](float128 x) { return pswf0_der_eval(x, ctx.workdata) * ctx.xv0; };
    auto Wx = [&](float128 x) { return exp(-ctx.c_param * x * x / 2.0); };
    auto Wx_der = [&](float128 x) { return -ctx.c_param * x * exp(-ctx.c_param * x * x / 2.0); };

    VectorXmp an = initial_an;

    // Seed alternation nodes from local extrema of the L16 residual on [0, 1]
    std::vector<float128> grid_x(eval_nodes + 1), grid_r(eval_nodes + 1);
    for (int k = 0; k <= eval_nodes; ++k) {
        float128 x = float128(k) / float128(eval_nodes);
        grid_x[k] = x;
        float128 px = 0.0;
        for (int j = 0; j < n_unknowns; ++j) px += an(j) * pow(x, powers_x[j]);
        grid_r[k] = Wx(x) * px - fx(x);
    }

    struct Extremum {
        float128 x;
        float128 r;
    };
    std::vector<Extremum> alt_extrema;
    for (int k = 0; k <= eval_nodes; ++k) {
        float128 rk = grid_r[k];
        if (rk == 0.0) continue;
        bool is_ext = false;
        if (k == 0) {
            is_ext = (abs(rk) >= abs(grid_r[1]));
        } else if (k == eval_nodes) {
            is_ext = (abs(rk) >= abs(grid_r[eval_nodes - 1]));
        } else if ((rk > 0.0 && rk >= grid_r[k - 1] && rk >= grid_r[k + 1]) ||
                   (rk < 0.0 && rk <= grid_r[k - 1] && rk <= grid_r[k + 1])) {
            is_ext = true;
        }
        if (!is_ext) continue;
        if (!alt_extrema.empty() && ((alt_extrema.back().r > 0.0) == (rk > 0.0))) {
            if (abs(rk) > abs(alt_extrema.back().r)) {
                alt_extrema.back() = {grid_x[k], rk};
            }
        } else {
            alt_extrema.push_back({grid_x[k], rk});
        }
    }

    std::vector<float128> xn(n_unknowns + 1);
    if (static_cast<int>(alt_extrema.size()) >= n_unknowns + 1) {
        // Pick contiguous alternating window of length n_unknowns + 1 with largest min |r|
        size_t best_start = 0;
        float128 best_min_r = -1.0;
        for (size_t st = 0; st + n_unknowns < alt_extrema.size(); ++st) {
            float128 cur_min = abs(alt_extrema[st].r);
            for (int j = 1; j <= n_unknowns; ++j) {
                float128 ar = abs(alt_extrema[st + j].r);
                if (ar < cur_min) cur_min = ar;
            }
            if (cur_min > best_min_r) {
                best_min_r = cur_min;
                best_start = st;
            }
        }
        for (int k = 0; k <= n_unknowns; ++k) xn[k] = alt_extrema[best_start + k].x;
        std::cout << "  Seeded " << (n_unknowns + 1) << " alternation nodes from L16 residual extrema.\n";
    } else {
        for (int k = 0; k <= n_unknowns; ++k) {
            float128 z = 0.5 * (1.0 - cos(k * pi / n_unknowns));
            xn[k] = sqrt(z);
        }
        std::sort(xn.begin(), xn.end());
        xn[0] = 0.0;
        xn.back() = 1.0;
        std::cout << "  Seeded " << (n_unknowns + 1) << " Chebyshev fallback nodes.\n";
    }
    float128 min_true_err = 1e300;
    VectorXmp best_coeffs = an;
    float128 prev_true_err = -1.0;
    float128 prev_ref_e = -1.0;

    for (int iter = 0; iter < 50; ++iter) {
        MatrixXmp A(n_unknowns + 1, n_unknowns + 1);
        VectorXmp F(n_unknowns + 1);

        for (int r = 0; r <= n_unknowns; ++r) {
            float128 xi = xn[r];
            float128 w_val = Wx(xi);
            for (int c = 0; c < n_unknowns; ++c) {
                A(r, c) = w_val * pow(xi, powers_x[c]);
            }
            A(r, n_unknowns) = (r % 2 == 0) ? -1.0 : 1.0;
            F(r) = fx(xi);
        }

        VectorXmp sol = A.fullPivHouseholderQr().solve(F);
        for (int i = 0; i < n_unknowns; ++i) an(i) = sol(i);
        float128 current_e = abs(sol(n_unknowns));

        float128 true_max_err = evaluate_dense_max_error(an, fx, Wx, powers_x, eval_nodes);

        // Safeguard tracking for lowest error (prevents rebounds)
        if (true_max_err < min_true_err) {
            min_true_err = true_max_err;
            best_coeffs = an;
        }

        std::cout << "  Iter " << std::setw(2) << iter << " | True Max Err: " << std::scientific << std::setprecision(5) << static_cast<double>(true_max_err)
                  << " | Ref E: " << static_cast<double>(current_e) << " | Coeffs: ";
        for (int i = 0; i < n_unknowns; ++i) {
            std::cout << std::scientific << std::setprecision(3) << static_cast<double>(an(i)) << " ";
        }
        std::cout << "\n";

        // Check for divergence
        if (iter > 0 && true_max_err > prev_true_err && current_e > prev_ref_e) {
            std::cout << "    --> [Approximation failed to converge] Error and Ref E increased simultaneously!\n";
            break;
        }

        prev_true_err = true_max_err;
        prev_ref_e = current_e;

        auto err_func = [&](float128 x) {
            float128 px = 0.0;
            for (int i = 0; i < n_unknowns; ++i) px += an(i) * pow(x, powers_x[i]);
            return Wx(x) * px - fx(x);
        };

        auto err_der_func = [&](float128 x) {
            float128 px = 0.0, px_der = 0.0;
            for (int i = 0; i < n_unknowns; ++i) {
                px += an(i) * pow(x, powers_x[i]);
                if (powers_x[i] > 0) px_der += an(i) * powers_x[i] * pow(x, powers_x[i] - 1);
            }
            return (Wx_der(x) * px + Wx(x) * px_der) - fx_der(x);
        };

        std::vector<float128> xn_roots;
        for (int i = 0; i < n_unknowns; ++i) {
            xn_roots.push_back(bisection(err_func, xn[i], xn[i + 1]));
        }

        std::vector<float128> xn_new(n_unknowns + 1);

        // Derivative based maximization
        if (err_der_func(0.0) * err_der_func(xn_roots[0]) < 0) {
            xn_new[0] = bisection(err_der_func, 0.0, xn_roots[0]);
        } else {
            xn_new[0] = 0.0;
        }

        for (int i = 0; i < n_unknowns - 1; ++i) {
            xn_new[i + 1] = bisection(err_der_func, xn_roots[i], xn_roots[i + 1]);
        }

        if (err_der_func(xn_roots.back()) * err_der_func(1.0) < 0) {
            xn_new[n_unknowns] = bisection(err_der_func, xn_roots.back(), 1.0);
        } else {
            xn_new[n_unknowns] = 1.0;
        }

        float128 max_diff = 0;
        for (int i = 0; i <= n_unknowns; ++i) {
            float128 diff = abs(xn_new[i] - xn[i]);
            if (diff > max_diff) max_diff = diff;
            xn[i] = xn_new[i];
        }

        if (max_diff < 1e-30 && iter > 2) {
            break;
        }
    }

    std::cout << "\n========================================================\n";
    std::cout << "FINAL OPTIMAL MINIMAX POLYNOMIAL for " << ctx.name << "\n";
    std::cout << "Optimal Z-Degree : " << d << "\n";
    std::cout << "Lowest Max Error : " << std::scientific << std::setprecision(5) << static_cast<double>(min_true_err) << "\n";
    std::cout << "Coefficients for Q(z):\n";

    int print_prec = std::ceil(-log10(ctx.eps)) + 2;
    std::cout << "{";
    for (int i = 0; i <= d; ++i) {
        std::cout << std::scientific << std::setprecision(print_prec) << static_cast<double>(best_coeffs(i));
        if (i < d) std::cout << ", ";
    }
    std::cout << "}\n";
    std::cout << "========================================================\n\n";
}

int main(int argc, char* argv[]) {
    Eigen::setNbThreads(16);
    Eigen::initParallel();

    double cli_eps = 1e-16;  // Used only for coefficient print precision.
    int cli_w = 0;
    double cli_sigma = 0.0;
    bool have_w = false;
    bool have_sigma = false;
    int cli_deg = 26;  // Default to Double Precision bounds
    int cli_reg_nodes = 250;
    int cli_eval_nodes = 500;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("-deg=") == 0) cli_deg = std::stoi(arg.substr(5));
        if (arg.find("-eps=") == 0) cli_eps = std::stod(arg.substr(5));
        if (arg.find("-w=") == 0) {
            cli_w = std::stoi(arg.substr(3));
            have_w = true;
        }
        if (arg.find("-sigma=") == 0) {
            cli_sigma = std::stod(arg.substr(7));
            have_sigma = true;
        }
        if (arg.find("-reg_nodes=") == 0) cli_reg_nodes = std::stoi(arg.substr(11));
        if (arg.find("-eval_nodes=") == 0) cli_eval_nodes = std::stoi(arg.substr(12));
    }

    if (!have_w || !have_sigma || cli_w <= 0 || cli_sigma <= 0.0) {
        std::cerr << "Usage: " << argv[0] << " -w=<positive integer> -sigma=<positive value> [-deg=N] [-eps=E] [-reg_nodes=N] [-eval_nodes=N]\n";
        return 1;
    }

    std::cout << "========================================================\n";
    std::cout << "=== Eigen/Float128 COD Weighted Minimax              ===\n";
    std::cout << "========================================================\n\n";
    std::cout << "Running Custom Config: -deg=" << cli_deg << " -w=" << cli_w << " -sigma=" << cli_sigma << " -eps=" << cli_eps
              << " -reg_nodes=" << cli_reg_nodes << " -eval_nodes=" << cli_eval_nodes << "\n\n";

    EpsContext ctx_custom = build_context("Custom CLI", cli_eps, cli_w, cli_sigma);

    VectorXmp initial_coeffs = run_regression_single(ctx_custom, cli_deg, cli_reg_nodes, cli_eval_nodes);

    run_remez_single(ctx_custom, cli_deg, initial_coeffs, cli_eval_nodes);

    return 0;
}
