/* =============================================================================
 * eigen_zohar_bench.cpp
 *
 * FastChi2-style Hermitian Toeplitz benchmark:
 *   - vector-batched Zohar solvers from src/linalg.c
 *   - Eigen FullPivHouseholderQR and LDLT on the equivalent real block system
 *   - cpp_bin_float_quad FullPivHouseholderQR as the reference solution
 * =============================================================================
 */

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <boost/multiprecision/cpp_bin_float.hpp>

#include <linalg.h>

using mp_quad = boost::multiprecision::cpp_bin_float_quad;

static constexpr double kPi = 3.141592653589793238462643383279502884;
static constexpr int kWarmupPasses = 16;

struct ToeplitzSystem {
  std::vector<double> Rr;
  std::vector<double> Ri;
  std::vector<double> Yr;
  std::vector<double> Yi;
};

struct QuadSolution {
  std::vector<mp_quad> xr;
  std::vector<mp_quad> xi;
};

static double tls_clock(void) {
  using clock = std::chrono::steady_clock;
  static const auto start = clock::now();
  return std::chrono::duration<double>(clock::now() - start).count();
}

template <typename T> static T scalar_cast(double value) { return T(value); }

static dd_t dd_from_double_local(double value) { return dd_make(value, 0.0); }

static mp_quad dd_to_mp_quad(dd_t value) {
  return mp_quad(value.hi) + mp_quad(value.lo);
}

template <typename Scalar>
static Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
solve_eigen_block(const ToeplitzSystem &sys, bool use_ldlt) {
  const int n = (int)sys.Yr.size();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A(2 * n, 2 * n);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(2 * n);

  A.setZero();
  for (int r = 0; r < n; r++) {
    b(r) = scalar_cast<Scalar>(sys.Yr[r]);
    b(n + r) = scalar_cast<Scalar>(sys.Yi[r]);

    for (int c = 0; c < n; c++) {
      const int k = std::abs(r - c);
      const Scalar re = scalar_cast<Scalar>(sys.Rr[k]);
      const Scalar im = scalar_cast<Scalar>((r >= c) ? sys.Ri[k] : -sys.Ri[k]);

      A(r, c) = re;
      A(r, n + c) = -im;
      A(n + r, c) = im;
      A(n + r, n + c) = re;
    }
  }

  if (use_ldlt)
    return A.ldlt().solve(b);
  return A.fullPivHouseholderQr().solve(b);
}

static QuadSolution solve_quad_reference(const ToeplitzSystem &sys) {
  const int n = (int)sys.Yr.size();
  Eigen::Matrix<mp_quad, Eigen::Dynamic, 1> x =
      solve_eigen_block<mp_quad>(sys, false);

  QuadSolution ref;
  ref.xr.resize(n);
  ref.xi.resize(n);
  for (int i = 0; i < n; i++) {
    ref.xr[i] = x(i);
    ref.xi[i] = x(n + i);
  }
  return ref;
}

template <typename Scalar>
static double
rel_error_block_solution(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x,
                         const QuadSolution &ref) {
  const int n = (int)ref.xr.size();
  mp_quad num = 0;
  mp_quad den = 0;
  for (int i = 0; i < n; i++) {
    const mp_quad xr = mp_quad(x(i));
    const mp_quad xi = mp_quad(x(n + i));
    const mp_quad dr = xr - ref.xr[i];
    const mp_quad di = xi - ref.xi[i];
    num += dr * dr + di * di;
    den += ref.xr[i] * ref.xr[i] + ref.xi[i] * ref.xi[i];
  }
  return std::sqrt((double)(num / (den + mp_quad("1e-300"))));
}

static std::vector<ToeplitzSystem> make_fastchi2_systems(int nterms,
                                                         int num_systems) {
  const int n = 2 * nterms + 1;
  const int sample_count = std::max(64, 10 * n);
  const double f0 = 0.1;
  const double df = 0.0075;

  std::mt19937 rng(42 + 101 * nterms);
  std::uniform_real_distribution<double> t_dist(0.0, 100.0);
  std::uniform_real_distribution<double> dy_dist(0.1, 0.5);
  std::normal_distribution<double> noise_dist(0.0, 1.0);

  std::vector<double> t(sample_count), y(sample_count), dy(sample_count),
      w(sample_count);
  for (int i = 0; i < sample_count; i++) {
    t[i] = t_dist(rng);
    dy[i] = dy_dist(rng);
    const double clean = 1.7 * std::sin(2.0 * kPi * 1.234 * t[i]) +
                         0.6 * std::cos(2.0 * kPi * 0.431 * t[i]);
    y[i] = clean + dy[i] * noise_dist(rng);
    w[i] = 1.0 / (dy[i] * dy[i]);
  }

  double ws = 0.0;
  double wy = 0.0;
  for (int i = 0; i < sample_count; i++) {
    ws += w[i];
    wy += w[i] * y[i];
  }
  const double y_mean = wy / ws;
  for (int i = 0; i < sample_count; i++)
    y[i] -= y_mean;

  std::vector<ToeplitzSystem> systems(num_systems);
  for (int s = 0; s < num_systems; s++) {
    const double freq = f0 + df * (double)s;
    std::vector<double> Sw(n, 0.0), Cw(n, 0.0);
    std::vector<double> Syw(nterms + 1, 0.0), Cyw(nterms + 1, 0.0);

    Cw[0] = ws;
    for (int i = 0; i < sample_count; i++)
      Cyw[0] += w[i] * y[i];

    for (int k = 1; k < n; k++) {
      for (int i = 0; i < sample_count; i++) {
        const double angle = 2.0 * kPi * (double)k * freq * t[i];
        const double sv = std::sin(angle);
        const double cv = std::cos(angle);
        Sw[k] += w[i] * sv;
        Cw[k] += w[i] * cv;
        if (k <= nterms) {
          Syw[k] += w[i] * y[i] * sv;
          Cyw[k] += w[i] * y[i] * cv;
        }
      }
    }

    ToeplitzSystem sys;
    sys.Rr.assign(n + 1, 0.0);
    sys.Ri.assign(n + 1, 0.0);
    sys.Yr.assign(n, 0.0);
    sys.Yi.assign(n, 0.0);

    for (int k = 0; k < n; k++) {
      sys.Rr[k] = Cw[k];
      sys.Ri[k] = -Sw[k];

      const int h = k - nterms;
      if (h == 0) {
        sys.Yr[k] = Cyw[0];
      } else if (h > 0) {
        sys.Yr[k] = Cyw[h];
        sys.Yi[k] = -Syw[h];
      } else {
        sys.Yr[k] = Cyw[-h];
        sys.Yi[k] = Syw[-h];
      }
    }
    systems[s] = std::move(sys);
  }

  return systems;
}

static void print_result(const char *label, int n, int lanes, int calls,
                         double t_zohar, double t_qr, double t_ldlt,
                         double err_zohar, double err_qr, double err_ldlt) {
  const double systems = (double)calls * (double)lanes;
  std::printf("[%-13s] N=%-4d lanes=%-2d calls=%-5d Zohar: %9.6f s "
              "(%8.2f ns/system)  Eigen QR: %9.6f s  LDLT: %9.6f s  "
              "rel-err(Z/QR/LDLT): %.3e / %.3e / %.3e\n",
              label, n, lanes, calls, t_zohar, t_zohar * 1e9 / systems, t_qr,
              t_ldlt, err_zohar, err_qr, err_ldlt);
}

static void benchmark_float(int n, const std::vector<ToeplitzSystem> &systems,
                            const std::vector<QuadSolution> &refs) {
  const int lanes = VECF_LEN;
  const int calls = ((int)systems.size() + lanes - 1) / lanes;
  std::vector<VECF> Rr(calls * (n + 1)), Ri(calls * (n + 1));
  std::vector<VECF> Yr(calls * n), Yi(calls * n), Xr(calls * n), Xi(calls * n),
      EhR(calls * n), EhI(calls * n), EhPrevR(calls * n), EhPrevI(calls * n);

  for (int call = 0; call < calls; call++) {
    for (int lane = 0; lane < lanes; lane++) {
      const int idx = std::min(call * lanes + lane, (int)systems.size() - 1);
      for (int k = 0; k <= n; k++) {
        Rr[call * (n + 1) + k][lane] = (float)systems[idx].Rr[k];
        Ri[call * (n + 1) + k][lane] = (float)systems[idx].Ri[k];
      }
      for (int k = 0; k < n; k++) {
        Yr[call * n + k][lane] = (float)systems[idx].Yr[k];
        Yi[call * n + k][lane] = (float)systems[idx].Yi[k];
      }
    }
  }

  for (int pass = 0; pass < kWarmupPasses; pass++)
    for (int call = 0; call < calls; call++)
      tlsf_solve_zohar(n, &Rr[call * (n + 1)], &Ri[call * (n + 1)],
                       &Yr[call * n], &Yi[call * n], &Xr[call * n],
                       &Xi[call * n], &EhR[call * n], &EhI[call * n],
                       &EhPrevR[call * n], &EhPrevI[call * n]);

  double t0 = tls_clock();
  for (int call = 0; call < calls; call++)
    tlsf_solve_zohar(n, &Rr[call * (n + 1)], &Ri[call * (n + 1)], &Yr[call * n],
                     &Yi[call * n], &Xr[call * n], &Xi[call * n],
                     &EhR[call * n], &EhI[call * n], &EhPrevR[call * n],
                     &EhPrevI[call * n]);
  const double t_zohar = tls_clock() - t0;

  mp_quad z_num = 0, z_den = 0, qr_num = 0, qr_den = 0, ldlt_num = 0,
          ldlt_den = 0;
  double t_qr = 0.0, t_ldlt = 0.0;
  for (int s = 0; s < (int)systems.size(); s++) {
    const int call = s / lanes;
    const int lane = s % lanes;
    for (int k = 0; k < n; k++) {
      const mp_quad xr = (double)Xr[call * n + k][lane];
      const mp_quad xi = (double)Xi[call * n + k][lane];
      const mp_quad dr = xr - refs[s].xr[k];
      const mp_quad di = xi - refs[s].xi[k];
      z_num += dr * dr + di * di;
      z_den += refs[s].xr[k] * refs[s].xr[k] + refs[s].xi[k] * refs[s].xi[k];
    }

    t0 = tls_clock();
    auto x_qr = solve_eigen_block<float>(systems[s], false);
    t_qr += tls_clock() - t0;
    t0 = tls_clock();
    auto x_ldlt = solve_eigen_block<float>(systems[s], true);
    t_ldlt += tls_clock() - t0;

    const double e_qr = rel_error_block_solution(x_qr, refs[s]);
    const double e_ldlt = rel_error_block_solution(x_ldlt, refs[s]);
    qr_num += mp_quad(e_qr) * mp_quad(e_qr);
    qr_den += 1;
    ldlt_num += mp_quad(e_ldlt) * mp_quad(e_ldlt);
    ldlt_den += 1;
  }

  print_result("float", n, lanes, calls, t_zohar, t_qr, t_ldlt,
               std::sqrt((double)(z_num / (z_den + mp_quad("1e-300")))),
               std::sqrt((double)(qr_num / qr_den)),
               std::sqrt((double)(ldlt_num / ldlt_den)));
}

static void benchmark_double(int n, const std::vector<ToeplitzSystem> &systems,
                             const std::vector<QuadSolution> &refs) {
  const int lanes = VEC_LEN;
  const int calls = ((int)systems.size() + lanes - 1) / lanes;
  std::vector<VEC> Rr(calls * (n + 1)), Ri(calls * (n + 1));
  std::vector<VEC> Yr(calls * n), Yi(calls * n), Xr(calls * n), Xi(calls * n),
      EhR(calls * n), EhI(calls * n), EhPrevR(calls * n), EhPrevI(calls * n);

  for (int call = 0; call < calls; call++) {
    for (int lane = 0; lane < lanes; lane++) {
      const int idx = std::min(call * lanes + lane, (int)systems.size() - 1);
      for (int k = 0; k <= n; k++) {
        Rr[call * (n + 1) + k][lane] = systems[idx].Rr[k];
        Ri[call * (n + 1) + k][lane] = systems[idx].Ri[k];
      }
      for (int k = 0; k < n; k++) {
        Yr[call * n + k][lane] = systems[idx].Yr[k];
        Yi[call * n + k][lane] = systems[idx].Yi[k];
      }
    }
  }

  for (int pass = 0; pass < kWarmupPasses; pass++)
    for (int call = 0; call < calls; call++)
      tls_solve_zohar(n, &Rr[call * (n + 1)], &Ri[call * (n + 1)],
                      &Yr[call * n], &Yi[call * n], &Xr[call * n],
                      &Xi[call * n], &EhR[call * n], &EhI[call * n],
                      &EhPrevR[call * n], &EhPrevI[call * n]);

  double t0 = tls_clock();
  for (int call = 0; call < calls; call++)
    tls_solve_zohar(n, &Rr[call * (n + 1)], &Ri[call * (n + 1)], &Yr[call * n],
                    &Yi[call * n], &Xr[call * n], &Xi[call * n], &EhR[call * n],
                    &EhI[call * n], &EhPrevR[call * n], &EhPrevI[call * n]);
  const double t_zohar = tls_clock() - t0;

  mp_quad z_num = 0, z_den = 0, qr_num = 0, qr_den = 0, ldlt_num = 0,
          ldlt_den = 0;
  double t_qr = 0.0, t_ldlt = 0.0;
  for (int s = 0; s < (int)systems.size(); s++) {
    const int call = s / lanes;
    const int lane = s % lanes;
    for (int k = 0; k < n; k++) {
      const mp_quad xr = Xr[call * n + k][lane];
      const mp_quad xi = Xi[call * n + k][lane];
      const mp_quad dr = xr - refs[s].xr[k];
      const mp_quad di = xi - refs[s].xi[k];
      z_num += dr * dr + di * di;
      z_den += refs[s].xr[k] * refs[s].xr[k] + refs[s].xi[k] * refs[s].xi[k];
    }

    t0 = tls_clock();
    auto x_qr = solve_eigen_block<double>(systems[s], false);
    t_qr += tls_clock() - t0;
    t0 = tls_clock();
    auto x_ldlt = solve_eigen_block<double>(systems[s], true);
    t_ldlt += tls_clock() - t0;

    const double e_qr = rel_error_block_solution(x_qr, refs[s]);
    const double e_ldlt = rel_error_block_solution(x_ldlt, refs[s]);
    qr_num += mp_quad(e_qr) * mp_quad(e_qr);
    qr_den += 1;
    ldlt_num += mp_quad(e_ldlt) * mp_quad(e_ldlt);
    ldlt_den += 1;
  }

  print_result("double", n, lanes, calls, t_zohar, t_qr, t_ldlt,
               std::sqrt((double)(z_num / (z_den + mp_quad("1e-300")))),
               std::sqrt((double)(qr_num / qr_den)),
               std::sqrt((double)(ldlt_num / ldlt_den)));
}

static void benchmark_dd(int n, const std::vector<ToeplitzSystem> &systems,
                         const std::vector<QuadSolution> &refs) {
  const int calls = (int)systems.size();
  std::vector<dd_t> Rr(calls * (n + 1)), Ri(calls * (n + 1));
  std::vector<dd_t> Yr(calls * n), Yi(calls * n), Xr(calls * n), Xi(calls * n),
      EhR(calls * n), EhI(calls * n), EhPrevR(calls * n), EhPrevI(calls * n);

  for (int call = 0; call < calls; call++) {
    for (int k = 0; k <= n; k++) {
      Rr[call * (n + 1) + k] = dd_from_double_local(systems[call].Rr[k]);
      Ri[call * (n + 1) + k] = dd_from_double_local(systems[call].Ri[k]);
    }
    for (int k = 0; k < n; k++) {
      Yr[call * n + k] = dd_from_double_local(systems[call].Yr[k]);
      Yi[call * n + k] = dd_from_double_local(systems[call].Yi[k]);
    }
  }

  for (int pass = 0; pass < kWarmupPasses; pass++)
    for (int call = 0; call < calls; call++)
      tlsdd_solve_zohar(n, &Rr[call * (n + 1)], &Ri[call * (n + 1)],
                        &Yr[call * n], &Yi[call * n], &Xr[call * n],
                        &Xi[call * n], &EhR[call * n], &EhI[call * n],
                        &EhPrevR[call * n], &EhPrevI[call * n]);

  double t0 = tls_clock();
  for (int call = 0; call < calls; call++)
    tlsdd_solve_zohar(n, &Rr[call * (n + 1)], &Ri[call * (n + 1)],
                      &Yr[call * n], &Yi[call * n], &Xr[call * n],
                      &Xi[call * n], &EhR[call * n], &EhI[call * n],
                      &EhPrevR[call * n], &EhPrevI[call * n]);
  const double t_zohar = tls_clock() - t0;

  mp_quad z_num = 0, z_den = 0, ldlt_num = 0, ldlt_den = 0;
  double t_qr = 0.0, t_ldlt = 0.0;
  for (int s = 0; s < calls; s++) {
    for (int k = 0; k < n; k++) {
      const mp_quad xr = dd_to_mp_quad(Xr[s * n + k]);
      const mp_quad xi = dd_to_mp_quad(Xi[s * n + k]);
      const mp_quad dr = xr - refs[s].xr[k];
      const mp_quad di = xi - refs[s].xi[k];
      z_num += dr * dr + di * di;
      z_den += refs[s].xr[k] * refs[s].xr[k] + refs[s].xi[k] * refs[s].xi[k];
    }

    t0 = tls_clock();
    (void)solve_eigen_block<mp_quad>(systems[s], false);
    t_qr += tls_clock() - t0;
    t0 = tls_clock();
    auto x_ldlt = solve_eigen_block<mp_quad>(systems[s], true);
    t_ldlt += tls_clock() - t0;

    const double e_ldlt = rel_error_block_solution(x_ldlt, refs[s]);
    ldlt_num += mp_quad(e_ldlt) * mp_quad(e_ldlt);
    ldlt_den += 1;
  }

  print_result("double-double", n, 1, calls, t_zohar, t_qr, t_ldlt,
               std::sqrt((double)(z_num / (z_den + mp_quad("1e-300")))), 0.0,
               std::sqrt((double)(ldlt_num / ldlt_den)));
}

int main(int argc, char **argv) {
  const int max_nterms = (argc > 1) ? std::atoi(argv[1]) : 12;
  const int num_systems = (argc > 2) ? std::atoi(argv[2]) : 1024;

  std::printf("Zohar vs Eigen FastChi2 Toeplitz benchmark (systems = %d)\n",
              num_systems);
  std::printf("N = 2*nterms+1. Reference is Eigen FullPivHouseholderQR with "
              "cpp_bin_float_quad.\n");
  std::printf("---------------------------------------------------------------"
              "---------------------------------------------------------\n");

  for (int nterms = 1; nterms <= max_nterms; nterms++) {
    const int n = 2 * nterms + 1;
    std::vector<ToeplitzSystem> systems =
        make_fastchi2_systems(nterms, num_systems);

    std::vector<QuadSolution> refs;
    refs.reserve(systems.size());
    for (const ToeplitzSystem &sys : systems)
      refs.push_back(solve_quad_reference(sys));

    benchmark_float(n, systems, refs);
    benchmark_double(n, systems, refs);
    benchmark_dd(n, systems, refs);
    std::printf("\n");
  }

  return 0;
}
