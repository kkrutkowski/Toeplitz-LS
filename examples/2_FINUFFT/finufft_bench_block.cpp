#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <finufft.h>

#include "finufft_bench_config.h"

extern "C" {
#include <nanofft.h>
#include <nufft1.h>
#include <scaling.h>
}

static double now_seconds() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

template <typename T>
static double compute_rel_err(int N, const dd_t *ref_r, const dd_t *ref_i,
                              const T *test_r, const T *test_i) {
  double sum_rel = 0.0;
  for (int i = 0; i < N; ++i) {
    double a_r = dd_to_double(ref_r[i]);
    double a_i = dd_to_double(ref_i[i]);
    double diff_r =
        dd_to_double(dd_sub(ref_r[i], dd_make((double)test_r[i], 0.0)));
    double diff_i =
        dd_to_double(dd_sub(ref_i[i], dd_make((double)test_i[i], 0.0)));
    double abs_err = sqrt((diff_r * diff_r) + (diff_i * diff_i));
    double denom = sqrt((a_r * a_r) + (a_i * a_i));
    sum_rel += (denom > 0.0) ? abs_err / denom : abs_err;
  }
  return sum_rel / N;
}

template <typename T>
static double compute_rel_err_complex(int N, const dd_t *ref_r,
                                      const dd_t *ref_i,
                                      const std::complex<T> *test) {
  double sum_rel = 0.0;
  for (int i = 0; i < N; ++i) {
    double a_r = dd_to_double(ref_r[i]);
    double a_i = dd_to_double(ref_i[i]);
    double diff_r =
        dd_to_double(dd_sub(ref_r[i], dd_make((double)test[i].real(), 0.0)));
    double diff_i =
        dd_to_double(dd_sub(ref_i[i], dd_make((double)test[i].imag(), 0.0)));
    double abs_err = sqrt((diff_r * diff_r) + (diff_i * diff_i));
    double denom = sqrt((a_r * a_r) + (a_i * a_i));
    sum_rel += (denom > 0.0) ? abs_err / denom : abs_err;
  }
  return sum_rel / N;
}

enum {
  kBackendPswf43 = 1,
  kBackendPswf21 = 2,
};

enum class PswfMode {
  k21,
  k43,
};

static inline int bitceil(double x) {
  return x <= 1.0 ? 1 : 1 << (1 + (int)(log2(x)));
}

static double approximate_cost(int N, int M, int block, int degree,
                               double alpha, double beta, double gamma,
                               int backend) {
  int block_eff = block;
  double beta_eff = beta;
  if (backend == kBackendPswf43) {
    block_eff += block_eff >> 1;
    beta_eff *= 1.5 * (9.0 / 8.0);
  }

  double N_eff = block_eff * ceil((double)N / block_eff);
  double gamma_eff =
      gamma * ((double)((2 * degree) + 1)) / (double)((3 * degree) + 1);
  if (backend == kBackendPswf43) {
    gamma_eff *= 1.5;
  }

  double cost = N_eff * pow((double)block, alpha);
  cost += beta_eff * (N_eff - (double)(block_eff)) * (double)(M) /
          (double)(block_eff);
  cost += gamma_eff * block_eff;
  return cost;
}

static int optimize_plan_size(int N, int M, int degree, double alpha,
                              double beta, double gamma, int backend) {
  double start = pow((beta * (double)M / alpha), 1.0 / (alpha + 1.0));
  int block = bitceil(start);
  int n_cap = bitceil((double)N);

  if (block > n_cap)
    block = n_cap;
  if (block < 1)
    block = 1;

  double best =
      approximate_cost(N, M, block, degree, alpha, beta, gamma, backend);
  while (block > 1) {
    int next = block >> 1;
    double next_cost =
        approximate_cost(N, M, next, degree, alpha, beta, gamma, backend);
    if (next_cost >= best)
      break;
    block = next;
    best = next_cost;
  }
  return block;
}

static int pswf43_plan_len_from_base(int base_len) {
  int plan_len = base_len;
  if (plan_len < 4)
    plan_len = 4;
  return (plan_len + 3) & ~3;
}

static int pswf43_output_len_for_plan(int plan_len) {
  return plan_len + (plan_len >> 1);
}

static int pswf_backend(PswfMode mode) {
  return mode == PswfMode::k21 ? kBackendPswf21 : kBackendPswf43;
}

static const char *pswf_upsamp(PswfMode mode) {
  return mode == PswfMode::k21 ? "21" : "43";
}

struct BaseSamples {
  std::vector<double> x;
  std::vector<double> y_real;
  std::vector<double> y_imag;
};

static BaseSamples make_base_samples(int Mpoints) {
  BaseSamples samples;
  samples.x.resize(Mpoints);
  samples.y_real.resize(Mpoints);
  samples.y_imag.resize(Mpoints);

  for (int i = 0; i < Mpoints; ++i) {
    samples.y_real[i] = 2.0 * rand() / RAND_MAX - 1.0;
    samples.y_imag[i] = 2.0 * rand() / RAND_MAX - 1.0;
    samples.x[i] = (double)rand() / RAND_MAX;
  }

  return samples;
}

template <typename Real>
static std::vector<Real> cast_real(const BaseSamples &s) {
  std::vector<Real> out(s.y_real.size());
  for (size_t i = 0; i < s.y_real.size(); ++i) {
    out[i] = (Real)s.y_real[i];
  }
  return out;
}

template <typename Real>
static std::vector<Real> cast_imag(const BaseSamples &s) {
  std::vector<Real> out(s.y_imag.size());
  for (size_t i = 0; i < s.y_imag.size(); ++i) {
    out[i] = (Real)s.y_imag[i];
  }
  return out;
}

static std::vector<dd_t> cast_dd(const std::vector<double> &in) {
  std::vector<dd_t> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = dd_make(in[i], 0.0);
  }
  return out;
}

static void triple_angle(double c, double s, double &c3, double &s3) {
  c3 = (4.0 * c * c * c) - (3.0 * c);
  s3 = (3.0 * s) - (4.0 * s * s * s);
}

static void compute_block_delta(PswfMode mode, double x, double df,
                                int output_block_len, float &delta_real,
                                float &delta_imag) {
  (void)mode;
  double phase_delta = x * df * (double)output_block_len;
  delta_real = (float)cos2pi(phase_delta);
  delta_imag = (float)sin2pi(phase_delta);
}

static void compute_block_delta(PswfMode mode, double x, double df,
                                int output_block_len, double &delta_real,
                                double &delta_imag) {
  if (mode == PswfMode::k43) {
    double phase_delta = x * df * (double)(output_block_len / 3);
    double c = cos2pi(phase_delta);
    double s = sin2pi(phase_delta);
    triple_angle(c, s, delta_real, delta_imag);
    return;
  }

  double phase_delta = x * df * (double)output_block_len;
  delta_real = cos2pi(phase_delta);
  delta_imag = sin2pi(phase_delta);
}

template <typename Real> struct SplitOutput {
  std::vector<Real> real;
  std::vector<Real> imag;
};

template <typename Real> struct BenchRow {
  int N = 0;
  int pswf21_block = 0;
  int pswf43_block = 0;
  double fin_plan = 0.0;
  double fin_exec = 0.0;
  double fin_err = 0.0;
  double pswf21_plan = 0.0;
  double pswf21_exec = 0.0;
  double pswf21_err = 0.0;
  double pswf43_plan = 0.0;
  double pswf43_exec = 0.0;
  double pswf43_err = 0.0;
};

template <typename Real> struct BenchTraits;

template <> struct BenchTraits<float> {
  using real_type = float;
  using coord_type = float;
  using complex_type = std::complex<float>;
  using finufft_plan_type = finufftf_plan;
  using pswf_plan_type = tlsf_nufft_pswf_plan;

  static constexpr const char *section_title = "SINGLE PRECISION (FLOAT)";
  static constexpr int width21 = 8;
  static constexpr int width43 = 9;
  static constexpr double alpha = F_ALPHA;
  static constexpr double beta = F_PSWF_BETA;
  static constexpr double gamma = F_PSWF_GAMMA;
  static constexpr float tolerance = 1.2e-7f;

  static int makeplan(int64_t *n_modes, int iflag, finufft_plan_type *plan,
                      finufft_opts *opts) {
    return finufftf_makeplan(1, 1, n_modes, iflag, 1, tolerance, plan, opts);
  }

  static int setpts(finufft_plan_type plan, int Mpoints, coord_type *x_fin) {
    return finufftf_setpts(plan, Mpoints, x_fin, nullptr, nullptr, 0, nullptr,
                           nullptr, nullptr);
  }

  static int execute(finufft_plan_type plan, complex_type *y_fin,
                     complex_type *out) {
    return finufftf_execute(plan, y_fin, out);
  }

  static void destroy(finufft_plan_type plan) { finufftf_destroy(plan); }

  static pswf_plan_type *pswf_initialize(int Mpoints, int plan_len, int width,
                                         double df, int max_ff,
                                         const char *upsamp) {
    return tlsf_nufft_pswf_initialize(Mpoints, plan_len, width, df, max_ff,
                                      upsamp);
  }

  static void pswf_precompute(pswf_plan_type *plan,
                              const std::vector<double> &x) {
    tlsf_nufft_pswf_precompute(plan, x.data());
  }

  static void pswf_execute(pswf_plan_type *plan, const real_type *src_real,
                           const real_type *src_imag, real_type *out_real,
                           real_type *out_imag, int freq_factor) {
    tlsf_nufft_pswf_execute(plan, src_real, src_imag, out_real, out_imag,
                            freq_factor);
  }

  static void pswf_destroy(pswf_plan_type *plan) {
    tlsf_nufft_free_pswf_plan(plan);
  }
};

template <> struct BenchTraits<double> {
  using real_type = double;
  using coord_type = double;
  using complex_type = std::complex<double>;
  using finufft_plan_type = finufft_plan;
  using pswf_plan_type = tls_nufft_pswf_plan;

  static constexpr const char *section_title = "DOUBLE PRECISION";
  static constexpr int width21 = 16;
  static constexpr int width43 = 18;
  static constexpr double alpha = D_ALPHA;
  static constexpr double beta = D_PSWF_BETA;
  static constexpr double gamma = D_PSWF_GAMMA;
  static constexpr double tolerance = 8e-16;

  static int makeplan(int64_t *n_modes, int iflag, finufft_plan_type *plan,
                      finufft_opts *opts) {
    return finufft_makeplan(1, 1, n_modes, iflag, 1, tolerance, plan, opts);
  }

  static int setpts(finufft_plan_type plan, int Mpoints, coord_type *x_fin) {
    return finufft_setpts(plan, Mpoints, x_fin, nullptr, nullptr, 0, nullptr,
                          nullptr, nullptr);
  }

  static int execute(finufft_plan_type plan, complex_type *y_fin,
                     complex_type *out) {
    return finufft_execute(plan, y_fin, out);
  }

  static void destroy(finufft_plan_type plan) { finufft_destroy(plan); }

  static pswf_plan_type *pswf_initialize(int Mpoints, int plan_len, int width,
                                         double df, int max_ff,
                                         const char *upsamp) {
    return tls_nufft_pswf_initialize(Mpoints, plan_len, width, df, max_ff,
                                     upsamp);
  }

  static void pswf_precompute(pswf_plan_type *plan,
                              const std::vector<double> &x) {
    tls_nufft_pswf_precompute(plan, x.data());
  }

  static void pswf_execute(pswf_plan_type *plan, const real_type *src_real,
                           const real_type *src_imag, real_type *out_real,
                           real_type *out_imag, int freq_factor) {
    tls_nufft_pswf_execute(plan, src_real, src_imag, out_real, out_imag,
                           freq_factor);
  }

  static void pswf_destroy(pswf_plan_type *plan) {
    tls_nufft_free_pswf_plan(plan);
  }
};

template <> struct BenchTraits<dd_t> {
  using real_type = dd_t;
  using pswf_plan_type = tlsdd_nufft_pswf_plan;

  static constexpr int width21 = 32;
  static constexpr int width43 = 36;
  static constexpr double alpha = DD_ALPHA;
  static constexpr double beta = DD_PSWF_BETA;
  static constexpr double gamma = DD_PSWF_GAMMA;

  static pswf_plan_type *pswf_initialize(int Mpoints, int plan_len, int width,
                                         double df, int max_ff,
                                         const char *upsamp) {
    return tlsdd_nufft_pswf_initialize(Mpoints, plan_len, width, df, max_ff,
                                       upsamp);
  }

  static void pswf_precompute(pswf_plan_type *plan,
                              const std::vector<dd_t> &x) {
    tlsdd_nufft_pswf_precompute(plan, x.data());
  }

  static void pswf_execute(pswf_plan_type *plan, const real_type *src_real,
                           const real_type *src_imag, real_type *out_real,
                           real_type *out_imag, int freq_factor) {
    tlsdd_nufft_pswf_execute(plan, src_real, src_imag, out_real, out_imag,
                             freq_factor);
  }

  static void pswf_destroy(pswf_plan_type *plan) {
    tlsdd_nufft_free_pswf_plan(plan);
  }
};

static void require_status(int status, const std::string &what) {
  if (status != 0) {
    throw std::runtime_error(what + " failed with status " +
                             std::to_string(status));
  }
}

template <typename Traits> static int width_for_mode(PswfMode mode) {
  return mode == PswfMode::k21 ? Traits::width21 : Traits::width43;
}

template <typename Traits>
static int optimized_plan_len_for_mode(PswfMode mode, int N, int Mpoints) {
  int base_len = optimize_plan_size(N, Mpoints, 0, Traits::alpha, Traits::beta,
                                    Traits::gamma, pswf_backend(mode));
  if (mode == PswfMode::k21) {
    return base_len;
  }
  return pswf43_plan_len_from_base(base_len);
}

static int output_block_len_for_mode(PswfMode mode, int plan_len) {
  if (mode == PswfMode::k21) {
    return plan_len;
  }
  return pswf43_output_len_for_plan(plan_len);
}

template <typename Traits>
static void execute_pswf_block_sweep(
    typename Traits::pswf_plan_type *plan,
    const std::vector<typename Traits::real_type> &input_real,
    const std::vector<typename Traits::real_type> &input_imag,
    const std::vector<typename Traits::real_type> &delta_real,
    const std::vector<typename Traits::real_type> &delta_imag, int N,
    int output_block_len, std::vector<typename Traits::real_type> &work_real,
    std::vector<typename Traits::real_type> &work_imag,
    std::vector<typename Traits::real_type> &block_real,
    std::vector<typename Traits::real_type> &block_imag,
    SplitOutput<typename Traits::real_type> &out) {
  using real_type = typename Traits::real_type;

  work_real = input_real;
  work_imag = input_imag;

  for (int base = 0; base < N; base += output_block_len) {
    Traits::pswf_execute(plan, work_real.data(), work_imag.data(),
                         block_real.data(), block_imag.data(), 1);

    int count = std::min(output_block_len, N - base);
    std::copy_n(block_real.data(), count, out.real.data() + base);
    std::copy_n(block_imag.data(), count, out.imag.data() + base);

    if (base + output_block_len < N) {
      for (size_t m = 0; m < work_real.size(); ++m) {
        real_type yr = work_real[m];
        real_type yi = work_imag[m];
        real_type dr = delta_real[m];
        real_type di = delta_imag[m];
        work_real[m] = yr * dr - yi * di;
        work_imag[m] = yr * di + yi * dr;
      }
    }
  }
}

template <typename Traits> struct FinufftResult {
  double plan_time = 0.0;
  double exec_time = 0.0;
  std::vector<typename Traits::complex_type> out;
};

template <typename Traits>
static FinufftResult<Traits> run_finufft_naive(int N, const BaseSamples &base,
                                               double df, int iflag,
                                               int max_ff) {
  using real_type = typename Traits::real_type;
  using coord_type = typename Traits::coord_type;
  using complex_type = typename Traits::complex_type;

  FinufftResult<Traits> result;
  result.out.resize(N * max_ff);

  int64_t N_modes[] = {(int64_t)N};
  finufft_opts opts;
  opts.nthreads = 1;
  finufft_default_opts(&opts);

  for (int ff = 1; ff <= max_ff; ++ff) {
    std::vector<coord_type> x_fin(base.x.size());
    std::vector<complex_type> y_fin(base.x.size());
    for (size_t i = 0; i < base.x.size(); ++i) {
      double pos_scaled = base.x[i] * df * ff;
      double wrapped = fmod(pos_scaled, 1.0);
      if (wrapped < 0.0)
        wrapped += 1.0;
      x_fin[i] = (coord_type)(wrapped * 2.0 * M_PI);

      double phase = fmod((double)N * pos_scaled, 2.0) * M_PI;
      double cos_p = cos(phase);
      double sin_p = sin(phase);
      y_fin[i] = complex_type(
          (real_type)(base.y_real[i] * cos_p - base.y_imag[i] * sin_p),
          (real_type)(base.y_real[i] * sin_p + base.y_imag[i] * cos_p));
    }

    typename Traits::finufft_plan_type plan;
    double t0 = now_seconds();
    require_status(Traits::makeplan(N_modes, iflag, &plan, &opts),
                   "FINUFFT makeplan");
    require_status(Traits::setpts(plan, (int)base.x.size(), x_fin.data()),
                   "FINUFFT setpts");
    result.plan_time += now_seconds() - t0;

    size_t offset = (size_t)(ff - 1) * N;
    require_status(
        Traits::execute(plan, y_fin.data(), result.out.data() + offset),
        "FINUFFT warmup execute");

    t0 = now_seconds();
    require_status(
        Traits::execute(plan, y_fin.data(), result.out.data() + offset),
        "FINUFFT execute");
    result.exec_time += now_seconds() - t0;

    Traits::destroy(plan);
  }

  return result;
}

template <typename Traits> struct PswfBlockedResult {
  int output_block_len = 0;
  double plan_time = 0.0;
  double exec_time = 0.0;
  SplitOutput<typename Traits::real_type> out;
};

template <typename Traits>
static PswfBlockedResult<Traits> run_pswf_blocked(PswfMode mode, int N,
                                                  const BaseSamples &base,
                                                  double df, int max_ff) {
  using real_type = typename Traits::real_type;

  std::vector<real_type> input_real = cast_real<real_type>(base);
  std::vector<real_type> input_imag = cast_imag<real_type>(base);

  PswfBlockedResult<Traits> result;
  int plan_len =
      optimized_plan_len_for_mode<Traits>(mode, N, (int)base.x.size());
  result.output_block_len = output_block_len_for_mode(mode, plan_len);
  result.out.real.resize(N);
  result.out.imag.resize(N);

  double t0 = now_seconds();
  typename Traits::pswf_plan_type *plan = Traits::pswf_initialize(
      (int)base.x.size(), plan_len, width_for_mode<Traits>(mode), df, max_ff,
      pswf_upsamp(mode));
  if (!plan) {
    throw std::runtime_error("PSWF initialize failed");
  }
  Traits::pswf_precompute(plan, base.x);

  std::vector<real_type> delta_real(base.x.size());
  std::vector<real_type> delta_imag(base.x.size());
  for (size_t i = 0; i < base.x.size(); ++i) {
    compute_block_delta(mode, base.x[i], df, result.output_block_len,
                        delta_real[i], delta_imag[i]);
  }
  result.plan_time = now_seconds() - t0;

  std::vector<real_type> work_real(base.x.size());
  std::vector<real_type> work_imag(base.x.size());
  std::vector<real_type> block_real(result.output_block_len);
  std::vector<real_type> block_imag(result.output_block_len);

  execute_pswf_block_sweep<Traits>(plan, input_real, input_imag, delta_real,
                                   delta_imag, N, result.output_block_len,
                                   work_real, work_imag, block_real, block_imag,
                                   result.out);

  t0 = now_seconds();
  execute_pswf_block_sweep<Traits>(plan, input_real, input_imag, delta_real,
                                   delta_imag, N, result.output_block_len,
                                   work_real, work_imag, block_real, block_imag,
                                   result.out);
  result.exec_time = now_seconds() - t0;

  Traits::pswf_destroy(plan);
  return result;
}

static SplitOutput<dd_t> run_dd_pswf21_reference(int N, const BaseSamples &base,
                                                 double df, int max_ff) {
  std::vector<dd_t> x_dd = cast_dd(base.x);
  std::vector<dd_t> input_real = cast_dd(base.y_real);
  std::vector<dd_t> input_imag = cast_dd(base.y_imag);

  int plan_len = optimized_plan_len_for_mode<BenchTraits<dd_t>>(
      PswfMode::k21, N, (int)base.x.size());
  int output_block_len = output_block_len_for_mode(PswfMode::k21, plan_len);

  auto *plan = BenchTraits<dd_t>::pswf_initialize(
      (int)base.x.size(), plan_len,
      width_for_mode<BenchTraits<dd_t>>(PswfMode::k21), df, max_ff,
      pswf_upsamp(PswfMode::k21));
  if (!plan) {
    throw std::runtime_error("Double-double PSWF21 initialize failed");
  }
  BenchTraits<dd_t>::pswf_precompute(plan, x_dd);

  std::vector<dd_t> delta_real(base.x.size());
  std::vector<dd_t> delta_imag(base.x.size());
  dd_t df_dd = dd_make(df, 0.0);
  dd_t block_dd = dd_make((double)output_block_len, 0.0);
  for (size_t i = 0; i < base.x.size(); ++i) {
    dd_t phase_delta = dd_mul(dd_mul(x_dd[i], df_dd), block_dd);
    delta_real[i] = cos2pidd(phase_delta);
    delta_imag[i] = sin2pidd(phase_delta);
  }

  std::vector<dd_t> block_real(output_block_len);
  std::vector<dd_t> block_imag(output_block_len);
  std::vector<dd_t> work_real(base.x.size());
  std::vector<dd_t> work_imag(base.x.size());

  SplitOutput<dd_t> out;
  out.real.resize(N);
  out.imag.resize(N);

  work_real = input_real;
  work_imag = input_imag;
  for (int base_idx = 0; base_idx < N; base_idx += output_block_len) {
    BenchTraits<dd_t>::pswf_execute(plan, work_real.data(), work_imag.data(),
                                    block_real.data(), block_imag.data(), 1);

    int count = std::min(output_block_len, N - base_idx);
    std::copy_n(block_real.data(), count, out.real.data() + base_idx);
    std::copy_n(block_imag.data(), count, out.imag.data() + base_idx);

    if (base_idx + output_block_len < N) {
      for (size_t m = 0; m < work_real.size(); ++m) {
        dd_t yr = work_real[m];
        dd_t yi = work_imag[m];
        dd_t dr = delta_real[m];
        dd_t di = delta_imag[m];
        work_real[m] = dd_sub(dd_mul(yr, dr), dd_mul(yi, di));
        work_imag[m] = dd_add(dd_mul(yr, di), dd_mul(yi, dr));
      }
    }
  }

  BenchTraits<dd_t>::pswf_destroy(plan);
  return out;
}

template <typename Traits>
static BenchRow<typename Traits::real_type>
run_case(int N, int Mpoints, double df, int iflag, int max_ff) {
  BenchRow<typename Traits::real_type> row;
  row.N = N;

  BaseSamples base = make_base_samples(Mpoints);
  SplitOutput<dd_t> ref = run_dd_pswf21_reference(N, base, df, max_ff);
  FinufftResult<Traits> fin =
      run_finufft_naive<Traits>(N, base, df, iflag, max_ff);
  PswfBlockedResult<Traits> pswf21 =
      run_pswf_blocked<Traits>(PswfMode::k21, N, base, df, max_ff);
  PswfBlockedResult<Traits> pswf43 =
      run_pswf_blocked<Traits>(PswfMode::k43, N, base, df, max_ff);

  row.pswf21_block = pswf21.output_block_len;
  row.pswf43_block = pswf43.output_block_len;
  row.fin_plan = fin.plan_time;
  row.fin_exec = fin.exec_time;
  row.fin_err = compute_rel_err_complex(N, ref.real.data(), ref.imag.data(),
                                        fin.out.data());
  row.pswf21_plan = pswf21.plan_time;
  row.pswf21_exec = pswf21.exec_time;
  row.pswf21_err =
      compute_rel_err(N, ref.real.data(), ref.imag.data(),
                      pswf21.out.real.data(), pswf21.out.imag.data());
  row.pswf43_plan = pswf43.plan_time;
  row.pswf43_exec = pswf43.exec_time;
  row.pswf43_err =
      compute_rel_err(N, ref.real.data(), ref.imag.data(),
                      pswf43.out.real.data(), pswf43.out.imag.data());

  return row;
}

template <typename Traits>
static void run_section(int Mpoints, double df, int iflag, int max_ff) {
  std::cout << "\n============================================================="
               "=========================================================\n";
  std::cout << ' ' << Traits::section_title
            << " BLOCK BENCHMARK: Mpoints=" << Mpoints << " | max_ff=" << max_ff
            << " | ref=blocked DD PSWF21\n";
  std::cout << "==============================================================="
               "========================================================\n";
  std::cout
      << " Errors: all methods use the same blocked DD-PSWF21 reference\n";
  std::cout << "      N | 21Blk 43Blk | FIN Plan(s) FIN Exec(s)   FIN Err | "
               "21 Plan(s) 21 Exec(s)    21 Err | 43 Plan(s) 43 Exec(s)    "
               "43 Err\n";
  std::cout << "---------------------------------------------------------------"
               "--------------------------------------------------------\n";

  for (int N = kBenchmarkNMin; N <= kBenchmarkNMax; N *= 2) {
    BenchRow<typename Traits::real_type> row =
        run_case<Traits>(N, Mpoints, df, iflag, max_ff);
    std::cout << std::setw(7) << row.N << " | " << std::setw(5)
              << row.pswf21_block << ' ' << std::setw(5) << row.pswf43_block
              << " | " << std::scientific << std::setprecision(2)
              << std::setw(11) << row.fin_plan << ' ' << std::setw(11)
              << row.fin_exec << ' ' << std::setprecision(1) << std::setw(9)
              << row.fin_err << " | " << std::setprecision(2) << std::setw(11)
              << row.pswf21_plan << ' ' << std::setw(11) << row.pswf21_exec
              << ' ' << std::setprecision(1) << std::setw(9) << row.pswf21_err
              << " | " << std::setprecision(2) << std::setw(11)
              << row.pswf43_plan << ' ' << std::setw(11) << row.pswf43_exec
              << ' ' << std::setprecision(1) << std::setw(9) << row.pswf43_err
              << '\n';
  }
}

int main() {
  try {
    srand(1);

    int Mpoints = kBenchmarkM;
    double df = 1.0;
    int max_ff = 1;
    int iflag = 1;

    run_section<BenchTraits<float>>(Mpoints, df, iflag, max_ff);
    run_section<BenchTraits<double>>(Mpoints, df, iflag, max_ff);
  } catch (const std::exception &ex) {
    std::cerr << "finufft_bench_block error: " << ex.what() << '\n';
    return 1;
  }

  return 0;
}
