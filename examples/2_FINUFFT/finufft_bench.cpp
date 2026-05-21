#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// FINUFFT header
#include <finufft.h>

// Your microlibrary headers
extern "C" {
#include <nufft1.h>
#include <quadmath.h>
}

// Helper timer function
static double now_seconds() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// Relative error template for split arrays (LRA / PSWF)
template <typename T>
static double compute_rel_err(int N, const dd_t *ref_r, const dd_t *ref_i,
                              const T *test_r, const T *test_i) {
  __float128 sum_rel = 0.0Q;
  for (int i = 0; i < N; ++i) {
    __float128 a_r = (__float128)ref_r[i].hi + (__float128)ref_r[i].lo;
    __float128 a_i = (__float128)ref_i[i].hi + (__float128)ref_i[i].lo;
    __float128 b_r = test_r[i];
    __float128 b_i = test_i[i];

    __float128 diff_r = a_r - b_r;
    __float128 diff_i = a_i - b_i;
    __float128 abs_err = sqrtq((diff_r * diff_r) + (diff_i * diff_i));
    __float128 denom = sqrtq((a_r * a_r) + (a_i * a_i));
    sum_rel += (denom > 0.0Q) ? abs_err / denom : abs_err;
  }
  return (double)(sum_rel / N);
}

// Relative error template for std::complex arrays (FINUFFT)
template <typename T>
static double compute_rel_err_complex(int N, const dd_t *ref_r,
                                      const dd_t *ref_i,
                                      const std::complex<T> *test) {
  __float128 sum_rel = 0.0Q;
  for (int i = 0; i < N; ++i) {
    __float128 a_r = (__float128)ref_r[i].hi + (__float128)ref_r[i].lo;
    __float128 a_i = (__float128)ref_i[i].hi + (__float128)ref_i[i].lo;
    __float128 b_r = test[i].real();
    __float128 b_i = test[i].imag();

    __float128 diff_r = a_r - b_r;
    __float128 diff_i = a_i - b_i;
    __float128 abs_err = sqrtq((diff_r * diff_r) + (diff_i * diff_i));
    __float128 denom = sqrtq((a_r * a_r) + (a_i * a_i));
    sum_rel += (denom > 0.0Q) ? abs_err / denom : abs_err;
  }
  return (double)(sum_rel / N);
}

int main() {
  srand(1);

  int Mpoints = 256 * 64;
  int rank_f = 9;
  int w_f = 8;
  int rank_d = 16;
  int w_d = 16;
  int rank_dd = 27;

  double df = 1.0;
  int max_ff = 1; // Test multiple freq factors just like nufft1_bench.c

  int iflag = 1; // +1 exponential sign matches your custom code implementation

  // =====================================================================================================
  // 1. SINGLE PRECISION SWEEP (FLOAT)
  // =====================================================================================================
  std::cout << "\n============================================================="
               "===========================================================\n";
  std::cout << " SINGLE PRECISION (FLOAT) BENCHMARK: Mpoints=" << Mpoints
            << " | Rank=" << rank_f << " | Width=" << w_f
            << " | max_ff=" << max_ff << "\n";
  std::cout << "==============================================================="
               "=========================================================\n";
  std::cout << "      N | FIN Plan(s) FIN Exec(s)   FIN Err | PSW Plan(s) PSW "
               "Exec(s)   PSW Err | LRA Plan(s) LRA "
               "Exec(s)   LRA Err \n";
  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------\n";

  for (int N = 256; N <= 1048576; N *= 2) {
    int64_t N_modes[] = {(int64_t)N};

    std::vector<double> x(Mpoints);
    std::vector<dd_t> x_dd(Mpoints);
    std::vector<float> ys_r_f(Mpoints), ys_i_f(Mpoints);
    std::vector<dd_t> ys_r_dd(Mpoints), ys_i_dd(Mpoints);

    for (int i = 0; i < Mpoints; ++i) {
      double r = 2.0 * rand() / RAND_MAX - 1.0;
      double im = 2.0 * rand() / RAND_MAX - 1.0;
      double pos = (double)rand() / RAND_MAX;

      x[i] = pos;
      x_dd[i] = dd_make(pos, 0.0);
      ys_r_f[i] = (float)r;
      ys_i_f[i] = (float)im;

      ys_r_dd[i] = dd_make(r, 0.0);
      ys_i_dd[i] = dd_make(im, 0.0);
    }

    std::vector<float> out_lra_r_f(N * max_ff), out_lra_i_f(N * max_ff);
    std::vector<float> out_pswf_r_f(N * max_ff), out_pswf_i_f(N * max_ff);
    std::vector<std::complex<float>> out_finufftf(N * max_ff);
    std::vector<dd_t> out_lra_r_dd(N * max_ff), out_lra_i_dd(N * max_ff);

    // --- Ground Truth (Double-Double) ---
    auto plan_lra_dd =
        tlsdd_nufft_lra_initialize(Mpoints, N, rank_dd, df, max_ff);
    tlsdd_nufft_lra_precompute(plan_lra_dd, x_dd.data(), Mpoints, N, rank_dd);
    for (int ff = 1; ff <= max_ff; ++ff) {
      size_t offset = (ff - 1) * N;
      tlsdd_nufft_lra_execute(plan_lra_dd, ys_r_dd.data(), ys_i_dd.data(),
                              out_lra_r_dd.data() + offset,
                              out_lra_i_dd.data() + offset, ff);
    }
    tlsdd_nufft_free_lra_plan(plan_lra_dd);

    // --- FINUFFT Float ---
    finufft_opts opts_f;
    opts_f.nthreads = 1;
    finufft_default_opts(&opts_f);
    float tol_f = 1.2e-7;

    double t_fin_plan_f = 0.0;
    double t_fin_exec_f = 0.0;

    for (int ff = 1; ff <= max_ff; ++ff) {
      std::vector<float> x_fin(Mpoints);
      std::vector<std::complex<float>> y_fin(Mpoints);
      for (int i = 0; i < Mpoints; ++i) {
        double pos_scaled = x[i] * df * ff;
        double wrapped = fmod(pos_scaled, 1.0);
        if (wrapped < 0)
          wrapped += 1.0;
        x_fin[i] = (float)(wrapped * 2.0 * M_PI);

        double phase = fmod((double)N * pos_scaled, 2.0) * M_PI;
        double cos_p = cos(phase);
        double sin_p = sin(phase);
        y_fin[i] =
            std::complex<float>((float)(ys_r_f[i] * cos_p - ys_i_f[i] * sin_p),
                                (float)(ys_r_f[i] * sin_p + ys_i_f[i] * cos_p));
      }

      finufftf_plan plan_f;
      double t0 = now_seconds();
      finufftf_makeplan(1, 1, N_modes, iflag, 1, tol_f, &plan_f, &opts_f);
      finufftf_setpts(plan_f, Mpoints, x_fin.data(), nullptr, nullptr, 0,
                      nullptr, nullptr, nullptr);
      t_fin_plan_f += now_seconds() - t0;

      size_t offset = (ff - 1) * N;
      finufftf_execute(plan_f, y_fin.data(),
                       out_finufftf.data() + offset); // Warmup

      t0 = now_seconds();
      finufftf_execute(plan_f, y_fin.data(), out_finufftf.data() + offset);
      t_fin_exec_f += now_seconds() - t0;

      finufftf_destroy(plan_f);
    }

    // --- Custom LRA Float ---
    double t0 = now_seconds();
    auto plan_lra_f = tlsf_nufft_lra_initialize(Mpoints, N, rank_f, df, max_ff);
    tlsf_nufft_lra_precompute(plan_lra_f, x.data(), Mpoints, N, rank_f);
    double t_lra_plan_f = now_seconds() - t0;

    tlsf_nufft_lra_execute(plan_lra_f, ys_r_f.data(), ys_i_f.data(),
                           out_lra_r_f.data(), out_lra_i_f.data(), 1); // Warmup

    double t_lra_exec_f = 0.0;
    for (int ff = 1; ff <= max_ff; ++ff) {
      size_t offset = (ff - 1) * N;
      t0 = now_seconds();
      tlsf_nufft_lra_execute(plan_lra_f, ys_r_f.data(), ys_i_f.data(),
                             out_lra_r_f.data() + offset,
                             out_lra_i_f.data() + offset, ff);
      t_lra_exec_f += now_seconds() - t0;
    }
    tlsf_nufft_free_lra_plan(plan_lra_f);

    // --- Custom PSWF Float ---
    t0 = now_seconds();
    auto plan_pswf_f =
        tlsf_nufft_pswf_initialize(Mpoints, N, w_f, df, max_ff, "21");
    tlsf_nufft_pswf_precompute(plan_pswf_f, x.data());
    double t_pswf_plan_f = now_seconds() - t0;

    tlsf_nufft_pswf_execute(plan_pswf_f, ys_r_f.data(), ys_i_f.data(),
                            out_pswf_r_f.data(), out_pswf_i_f.data(),
                            1); // Warmup

    double t_pswf_exec_f = 0.0;
    for (int ff = 1; ff <= max_ff; ++ff) {
      size_t offset = (ff - 1) * N;
      t0 = now_seconds();
      tlsf_nufft_pswf_execute(plan_pswf_f, ys_r_f.data(), ys_i_f.data(),
                              out_pswf_r_f.data() + offset,
                              out_pswf_i_f.data() + offset, ff);
      t_pswf_exec_f += now_seconds() - t0;
    }
    tlsf_nufft_free_pswf_plan(plan_pswf_f);

    // Errors
    double err_fin_f =
        compute_rel_err_complex(N * max_ff, out_lra_r_dd.data(),
                                out_lra_i_dd.data(), out_finufftf.data());
    double err_lra_f =
        compute_rel_err(N * max_ff, out_lra_r_dd.data(), out_lra_i_dd.data(),
                        out_lra_r_f.data(), out_lra_i_f.data());
    double err_pswf_f =
        compute_rel_err(N * max_ff, out_lra_r_dd.data(), out_lra_i_dd.data(),
                        out_pswf_r_f.data(), out_pswf_i_f.data());

    std::cout << std::setw(7) << N << " | " << std::scientific
              << std::setprecision(2) << std::setw(9) << t_fin_plan_f << "  "
              << std::setw(11) << t_fin_exec_f << "  " << std::scientific
              << std::setprecision(1) << std::setw(9) << err_fin_f << " | "
              << std::scientific << std::setprecision(2) << std::setw(9)
              << t_pswf_plan_f << " " << std::setw(11) << t_pswf_exec_f << "   "
              << std::scientific << std::setprecision(1) << std::setw(9)
              << err_pswf_f << " | " << std::scientific << std::setprecision(2)
              << std::setw(9) << t_lra_plan_f << "  " << std::setw(11)
              << t_lra_exec_f << "  " << std::scientific << std::setprecision(1)
              << std::setw(9) << err_lra_f << "\n";
  }

  // =====================================================================================================
  // 2. DOUBLE PRECISION SWEEP
  // =====================================================================================================
  std::cout << "\n============================================================="
               "===========================================================\n";
  std::cout << " DOUBLE PRECISION BENCHMARK: Mpoints=" << Mpoints
            << " | Rank=" << rank_d << " | Width=" << w_d
            << " | max_ff=" << max_ff << "\n";
  std::cout << "==============================================================="
               "=========================================================\n";
  std::cout << "      N | FIN Plan(s) FIN Exec(s)   FIN Err | PSW Plan(s) PSW "
               "Exec(s)   PSW Err | LRA Plan(s) LRA "
               "Exec(s)   LRA Err\n";
  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------\n";

  for (int N = 256; N <= 1048576; N *= 2) {
    int64_t N_modes[] = {(int64_t)N};

    std::vector<double> x(Mpoints);
    std::vector<dd_t> x_dd(Mpoints);
    std::vector<double> ys_r_d(Mpoints), ys_i_d(Mpoints);
    std::vector<dd_t> ys_r_dd(Mpoints), ys_i_dd(Mpoints);

    for (int i = 0; i < Mpoints; ++i) {
      double r = 2.0 * rand() / RAND_MAX - 1.0;
      double im = 2.0 * rand() / RAND_MAX - 1.0;
      double pos = (double)rand() / RAND_MAX;

      x[i] = pos;
      x_dd[i] = dd_make(pos, 0.0);
      ys_r_d[i] = r;
      ys_i_d[i] = im;

      ys_r_dd[i] = dd_make(r, 0.0);
      ys_i_dd[i] = dd_make(im, 0.0);
    }

    std::vector<double> out_lra_r_d(N * max_ff), out_lra_i_d(N * max_ff);
    std::vector<double> out_pswf_r_d(N * max_ff), out_pswf_i_d(N * max_ff);
    std::vector<std::complex<double>> out_finufft(N * max_ff);
    std::vector<dd_t> out_lra_r_dd(N * max_ff), out_lra_i_dd(N * max_ff);

    // --- Ground Truth (Double-Double) ---
    auto plan_lra_dd =
        tlsdd_nufft_lra_initialize(Mpoints, N, rank_dd, df, max_ff);
    tlsdd_nufft_lra_precompute(plan_lra_dd, x_dd.data(), Mpoints, N, rank_dd);
    for (int ff = 1; ff <= max_ff; ++ff) {
      size_t offset = (ff - 1) * N;
      tlsdd_nufft_lra_execute(plan_lra_dd, ys_r_dd.data(), ys_i_dd.data(),
                              out_lra_r_dd.data() + offset,
                              out_lra_i_dd.data() + offset, ff);
    }
    tlsdd_nufft_free_lra_plan(plan_lra_dd);

    // --- FINUFFT Double ---
    finufft_opts opts_d;
    opts_d.nthreads = 1;
    finufft_default_opts(&opts_d);
    double tol_d = 8e-16;

    double t_fin_plan_d = 0.0;
    double t_fin_exec_d = 0.0;

    for (int ff = 1; ff <= max_ff; ++ff) {
      std::vector<double> x_fin(Mpoints);
      std::vector<std::complex<double>> y_fin(Mpoints);
      for (int i = 0; i < Mpoints; ++i) {
        double pos_scaled = x[i] * df * ff;
        double wrapped = fmod(pos_scaled, 1.0);
        if (wrapped < 0)
          wrapped += 1.0;
        x_fin[i] = wrapped * 2.0 * M_PI;

        double phase = fmod((double)N * pos_scaled, 2.0) * M_PI;
        double cos_p = cos(phase);
        double sin_p = sin(phase);
        y_fin[i] = std::complex<double>(ys_r_d[i] * cos_p - ys_i_d[i] * sin_p,
                                        ys_r_d[i] * sin_p + ys_i_d[i] * cos_p);
      }

      finufft_plan plan_d;
      double t0 = now_seconds();
      finufft_makeplan(1, 1, N_modes, iflag, 1, tol_d, &plan_d, &opts_d);
      finufft_setpts(plan_d, Mpoints, x_fin.data(), nullptr, nullptr, 0,
                     nullptr, nullptr, nullptr);
      t_fin_plan_d += now_seconds() - t0;

      size_t offset = (ff - 1) * N;
      finufft_execute(plan_d, y_fin.data(),
                      out_finufft.data() + offset); // Warmup

      t0 = now_seconds();
      finufft_execute(plan_d, y_fin.data(), out_finufft.data() + offset);
      t_fin_exec_d += now_seconds() - t0;

      finufft_destroy(plan_d);
    }

    // --- Custom LRA Double ---
    double t0 = now_seconds();
    auto plan_lra_d = tls_nufft_lra_initialize(Mpoints, N, rank_d, df, max_ff);
    tls_nufft_lra_precompute(plan_lra_d, x.data(), Mpoints, N, rank_d);
    double t_lra_plan_d = now_seconds() - t0;

    tls_nufft_lra_execute(plan_lra_d, ys_r_d.data(), ys_i_d.data(),
                          out_lra_r_d.data(), out_lra_i_d.data(), 1); // Warmup

    double t_lra_exec_d = 0.0;
    for (int ff = 1; ff <= max_ff; ++ff) {
      size_t offset = (ff - 1) * N;
      t0 = now_seconds();
      tls_nufft_lra_execute(plan_lra_d, ys_r_d.data(), ys_i_d.data(),
                            out_lra_r_d.data() + offset,
                            out_lra_i_d.data() + offset, ff);
      t_lra_exec_d += now_seconds() - t0;
    }
    tls_nufft_free_lra_plan(plan_lra_d);

    // --- Custom PSWF Double ---
    t0 = now_seconds();
    auto plan_pswf_d =
        tls_nufft_pswf_initialize(Mpoints, N, w_d, df, max_ff, "21");
    tls_nufft_pswf_precompute(plan_pswf_d, x.data());
    double t_pswf_plan_d = now_seconds() - t0;

    tls_nufft_pswf_execute(plan_pswf_d, ys_r_d.data(), ys_i_d.data(),
                           out_pswf_r_d.data(), out_pswf_i_d.data(),
                           1); // Warmup

    double t_pswf_exec_d = 0.0;
    for (int ff = 1; ff <= max_ff; ++ff) {
      size_t offset = (ff - 1) * N;
      t0 = now_seconds();
      tls_nufft_pswf_execute(plan_pswf_d, ys_r_d.data(), ys_i_d.data(),
                             out_pswf_r_d.data() + offset,
                             out_pswf_i_d.data() + offset, ff);
      t_pswf_exec_d += now_seconds() - t0;
    }
    tls_nufft_free_pswf_plan(plan_pswf_d);

    // Errors
    double err_fin_d =
        compute_rel_err_complex(N * max_ff, out_lra_r_dd.data(),
                                out_lra_i_dd.data(), out_finufft.data());
    double err_lra_d =
        compute_rel_err(N * max_ff, out_lra_r_dd.data(), out_lra_i_dd.data(),
                        out_lra_r_d.data(), out_lra_i_d.data());
    double err_pswf_d =
        compute_rel_err(N * max_ff, out_lra_r_dd.data(), out_lra_i_dd.data(),
                        out_pswf_r_d.data(), out_pswf_i_d.data());

    std::cout << std::setw(7) << N << " | " << std::scientific
              << std::setprecision(2) << std::setw(9) << t_fin_plan_d << " "
              << std::setw(11) << t_fin_exec_d << "   " << std::scientific
              << std::setprecision(1) << std::setw(9) << err_fin_d << " | "
              << std::scientific << std::setprecision(2) << std::setw(9)
              << t_pswf_plan_d << " " << std::setw(11) << t_pswf_exec_d
              << std::scientific << "   " << std::setprecision(1)
              << std::setw(9) << err_pswf_d << " | " << std::scientific
              << std::setprecision(2) << std::setw(9) << t_lra_plan_d << "  "
              << std::setw(11) << t_lra_exec_d << "   " << std::scientific
              << std::setprecision(1) << std::setw(9) << err_lra_d << "\n";
  }

  return 0;
}
