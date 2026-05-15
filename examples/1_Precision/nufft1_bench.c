#include <math.h>
#include <nufft1.h>
#include <quadmath.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void nufft1_naive_q(const double *x, const __float128 *y_real,
                           const __float128 *y_imag, int Mpoints, int N,
                           __float128 *out_real, __float128 *out_imag,
                           double df, int freq_factor) {
  const __float128 two_pi = 2.0Q * M_PIq;
  for (int k = 0; k < N; ++k) {
    __float128 acc_r = 0.0Q, acc_i = 0.0Q, c_r = 0.0Q, c_i = 0.0Q;
    for (int j = 0; j < Mpoints; ++j) {
      __float128 yr = y_real[j];
      __float128 yi = y_imag[j];
      // Include df and freq_factor into exact formulation
      __float128 theta =
          two_pi * (__float128)x[j] * (__float128)(k * df * freq_factor);
      __float128 cos_t = cosq(theta);
      __float128 sin_t = sinq(theta);
      __float128 term_r = (yr * cos_t) - (yi * sin_t);
      __float128 term_i = (yr * sin_t) + (yi * cos_t);

      __float128 yk_r = term_r - c_r;
      __float128 yk_i = term_i - c_i;
      __float128 tr = acc_r + yk_r;
      __float128 ti = acc_i + yk_i;
      c_r = (tr - acc_r) - yk_r;
      c_i = (ti - acc_i) - yk_i;
      acc_r = tr;
      acc_i = ti;
    }
    out_real[k] = acc_r;
    out_imag[k] = acc_i;
  }
}

enum bench_precision { BENCH_FLOAT, BENCH_DOUBLE, BENCH_DD };

__attribute__((optimize("no-fast-math", "no-associative-math",
                        "no-reciprocal-math",
                        "no-unsafe-math-optimizations"))) static __float128
bench_value(const void *values, enum bench_precision precision, int i) {
  if (precision == BENCH_DD) {
    const dd_t *v = (const dd_t *)values;
    return (__float128)v[i].hi + (__float128)v[i].lo;
  }
  if (precision == BENCH_DOUBLE)
    return (__float128)((const double *)values)[i];
  return (__float128)((const float *)values)[i];
}

__attribute__((optimize("no-fast-math", "no-associative-math",
                        "no-reciprocal-math",
                        "no-unsafe-math-optimizations"))) static double
compute_avg_rel(const __float128 *ref_r, const __float128 *ref_i, int N,
                int num_y, void *test_r, void *test_i,
                enum bench_precision precision) {
  __float128 sum_rel = 0.0Q;
  for (int i = 0; i < N * num_y; ++i) {
    __float128 a_r = ref_r[i];
    __float128 a_i = ref_i[i];
    __float128 b_r = bench_value(test_r, precision, i);
    __float128 b_i = bench_value(test_i, precision, i);

    __float128 diff_r = a_r - b_r;
    __float128 diff_i = a_i - b_i;
    __float128 abs_err = sqrtq((diff_r * diff_r) + (diff_i * diff_i));
    __float128 denom = sqrtq((a_r * a_r) + (a_i * a_i));
    sum_rel += (denom > 0.0Q) ? abs_err / denom : abs_err;
  }
  return (double)(sum_rel / (N * num_y));
}

int main() {
  srand(1);
  int Mpoints = 200;
  int N = 256;

  // Explicitly separated parameters
  int rank_f = 9;
  int w_f = 8;
  int rank_d = 16;
  int w_d = 16;
  int rank_dd = 27;
  int w_dd = 32; // EoS fallback

  double df = 1.0;
  int max_ff = 2; // Test with max_ff=2 to evaluate freq_factors of 1 and 2

  double *x = (double *)malloc(Mpoints * sizeof(double));
  dd_t *x_dd = (dd_t *)malloc(Mpoints * sizeof(dd_t));
  for (int i = 0; i < Mpoints; ++i) {
    x[i] = (double)rand() / RAND_MAX;
    x_dd[i] = dd_make(x[i], 0.0);
  }

  printf(
      "======================================================================"
      "========================================"
      "==========\n");
  printf(" DEFAULT SWEEP: Mpoints=%d N=%d rank_f=%d w_f=%d rank_d=%d w_d=%d "
         "rank_dd=%d w_dd=%d df=%.2f freq_factors=%d\n",
         Mpoints, N, rank_f, w_f, rank_d, w_d, rank_dd, w_dd, df, max_ff);
  printf(
      "======================================================================"
      "========================================"
      "==========\n");
  printf(
      " num_y |  Naive (s) |   LRA_f (s) |  LRA_f Err |  PSWF_f (s) | PSWF_f "
      "Err |   LRA_d (s) |  LRA_d Err |  "
      "PSWF_d (s) | PSWF_d Err |  LRA_dd (s) | LRA_dd Err | PSWF_dd (s) | "
      "PSWF_dd Err\n");
  printf(
      "----------------------------------------------------------------------"
      "----------------------------------------"
      "----------\n");

  for (int num_y = 10; num_y <= 100; num_y += 30) {
    size_t total_pts = (size_t)num_y * Mpoints;
    size_t total_out = (size_t)num_y * max_ff * N;

    float *ys_r_f = malloc(total_pts * sizeof(float));
    float *ys_i_f = malloc(total_pts * sizeof(float));
    double *ys_r_d = malloc(total_pts * sizeof(double));
    double *ys_i_d = malloc(total_pts * sizeof(double));
    dd_t *ys_r_dd = malloc(total_pts * sizeof(dd_t));
    dd_t *ys_i_dd = malloc(total_pts * sizeof(dd_t));
    __float128 *ys_r_q = malloc(total_pts * sizeof(__float128));
    __float128 *ys_i_q = malloc(total_pts * sizeof(__float128));

    for (size_t i = 0; i < total_pts; ++i) {
      double r = 2.0 * rand() / RAND_MAX - 1.0;
      double im = 2.0 * rand() / RAND_MAX - 1.0;
      ys_r_f[i] = (float)r;
      ys_i_f[i] = (float)im;
      ys_r_d[i] = r;
      ys_i_d[i] = im;
      ys_r_dd[i] = dd_make(r, 0.0);
      ys_i_dd[i] = dd_make(im, 0.0);
      ys_r_q[i] = (__float128)r;
      ys_i_q[i] = (__float128)im;
    }

    float *out_lra_r_f = malloc(total_out * sizeof(float));
    float *out_lra_i_f = malloc(total_out * sizeof(float));
    float *out_pswf_r_f = malloc(total_out * sizeof(float));
    float *out_pswf_i_f = malloc(total_out * sizeof(float));

    double *out_lra_r_d = malloc(total_out * sizeof(double));
    double *out_lra_i_d = malloc(total_out * sizeof(double));
    double *out_pswf_r_d = malloc(total_out * sizeof(double));
    double *out_pswf_i_d = malloc(total_out * sizeof(double));
    dd_t *out_lra_r_dd = malloc(total_out * sizeof(dd_t));
    dd_t *out_lra_i_dd = malloc(total_out * sizeof(dd_t));
    dd_t *out_pswf_r_dd = malloc(total_out * sizeof(dd_t));
    dd_t *out_pswf_i_dd = malloc(total_out * sizeof(dd_t));

    __float128 *out_naive_r_q = malloc(total_out * sizeof(__float128));
    __float128 *out_naive_i_q = malloc(total_out * sizeof(__float128));

    double t0 = now_seconds();
    tlsf_nufft_lra_plan *plan_lra_f =
        tlsf_nufft_lra_initialize(Mpoints, N, rank_f, df, max_ff);
    tlsf_nufft_lra_precompute(plan_lra_f, x, Mpoints, N, rank_f);
    for (int j = 0; j < num_y; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tlsf_nufft_lra_execute(plan_lra_f, ys_r_f + j * Mpoints,
                               ys_i_f + j * Mpoints, out_lra_r_f + out_idx * N,
                               out_lra_i_f + out_idx * N, ff);
      }
    }
    tlsf_nufft_free_lra_plan(plan_lra_f);
    double t_lra_f = now_seconds() - t0;

    t0 = now_seconds();
    tlsf_nufft_pswf_plan *plan_pswf_f =
        tlsf_nufft_pswf_initialize(Mpoints, N, w_f, df, max_ff);
    tlsf_nufft_pswf_precompute(plan_pswf_f, x);
    for (int j = 0; j < num_y; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tlsf_nufft_pswf_execute(
            plan_pswf_f, ys_r_f + j * Mpoints, ys_i_f + j * Mpoints,
            out_pswf_r_f + out_idx * N, out_pswf_i_f + out_idx * N, ff);
      }
    }
    tlsf_nufft_free_pswf_plan(plan_pswf_f);
    double t_pswf_f = now_seconds() - t0;

    t0 = now_seconds();
    tls_nufft_lra_plan *plan_lra_d =
        tls_nufft_lra_initialize(Mpoints, N, rank_d, df, max_ff);
    tls_nufft_lra_precompute(plan_lra_d, x, Mpoints, N, rank_d);
    for (int j = 0; j < num_y; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tls_nufft_lra_execute(plan_lra_d, ys_r_d + j * Mpoints,
                              ys_i_d + j * Mpoints, out_lra_r_d + out_idx * N,
                              out_lra_i_d + out_idx * N, ff);
      }
    }
    tls_nufft_free_lra_plan(plan_lra_d);
    double t_lra_d = now_seconds() - t0;

    t0 = now_seconds();
    tls_nufft_pswf_plan *plan_pswf_d =
        tls_nufft_pswf_initialize(Mpoints, N, w_d, df, max_ff);
    tls_nufft_pswf_precompute(plan_pswf_d, x);
    for (int j = 0; j < num_y; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tls_nufft_pswf_execute(plan_pswf_d, ys_r_d + j * Mpoints,
                               ys_i_d + j * Mpoints, out_pswf_r_d + out_idx * N,
                               out_pswf_i_d + out_idx * N, ff);
      }
    }
    tls_nufft_free_pswf_plan(plan_pswf_d);
    double t_pswf_d = now_seconds() - t0;

    t0 = now_seconds();
    tlsdd_nufft_lra_plan *plan_lra_dd =
        tlsdd_nufft_lra_initialize(Mpoints, N, rank_dd, df, max_ff);
    tlsdd_nufft_lra_precompute(plan_lra_dd, x_dd, Mpoints, N, rank_dd);
    for (int j = 0; j < num_y; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tlsdd_nufft_lra_execute(
            plan_lra_dd, ys_r_dd + j * Mpoints, ys_i_dd + j * Mpoints,
            out_lra_r_dd + out_idx * N, out_lra_i_dd + out_idx * N, ff);
      }
    }
    tlsdd_nufft_free_lra_plan(plan_lra_dd);
    double t_lra_dd = now_seconds() - t0;

    t0 = now_seconds();
    tlsdd_nufft_pswf_plan *plan_pswf_dd =
        tlsdd_nufft_pswf_initialize(Mpoints, N, w_dd, df, max_ff);
    tlsdd_nufft_pswf_precompute(plan_pswf_dd, x_dd);
    for (int j = 0; j < num_y; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tlsdd_nufft_pswf_execute(
            plan_pswf_dd, ys_r_dd + j * Mpoints, ys_i_dd + j * Mpoints,
            out_pswf_r_dd + out_idx * N, out_pswf_i_dd + out_idx * N, ff);
      }
    }
    tlsdd_nufft_free_pswf_plan(plan_pswf_dd);
    double t_pswf_dd = now_seconds() - t0;

    t0 = now_seconds();
    for (int j = 0; j < num_y; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        nufft1_naive_q(x, ys_r_q + j * Mpoints, ys_i_q + j * Mpoints, Mpoints,
                       N, out_naive_r_q + out_idx * N,
                       out_naive_i_q + out_idx * N, df, ff);
      }
    }
    double t_naive = now_seconds() - t0;

    double err_lra_f =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y * max_ff,
                        out_lra_r_f, out_lra_i_f, BENCH_FLOAT);
    double err_pswf_f =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y * max_ff,
                        out_pswf_r_f, out_pswf_i_f, BENCH_FLOAT);
    double err_lra_d =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y * max_ff,
                        out_lra_r_d, out_lra_i_d, BENCH_DOUBLE);
    double err_pswf_d =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y * max_ff,
                        out_pswf_r_d, out_pswf_i_d, BENCH_DOUBLE);
    double err_lra_dd =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y * max_ff,
                        out_lra_r_dd, out_lra_i_dd, BENCH_DD);
    double err_pswf_dd =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y * max_ff,
                        out_pswf_r_dd, out_pswf_i_dd, BENCH_DD);

    printf(" %5d | %10.6f | %11.6f | %10.2e | %11.6f | %10.2e | %11.6f | "
           "%10.2e | %11.6f | %10.2e | %11.6f | %10.2e | %11.6f | %10.2e\n",
           num_y, t_naive, t_lra_f, err_lra_f, t_pswf_f, err_pswf_f, t_lra_d,
           err_lra_d, t_pswf_d, err_pswf_d, t_lra_dd, err_lra_dd, t_pswf_dd,
           err_pswf_dd);

    free(ys_r_f);
    free(ys_i_f);
    free(ys_r_d);
    free(ys_i_d);
    free(ys_r_dd);
    free(ys_i_dd);
    free(ys_r_q);
    free(ys_i_q);
    free(out_lra_r_f);
    free(out_lra_i_f);
    free(out_pswf_r_f);
    free(out_pswf_i_f);
    free(out_lra_r_d);
    free(out_lra_i_d);
    free(out_pswf_r_d);
    free(out_pswf_i_d);
    free(out_lra_r_dd);
    free(out_lra_i_dd);
    free(out_pswf_r_dd);
    free(out_pswf_i_dd);
    free(out_naive_r_q);
    free(out_naive_i_q);
  }

  printf(
      "\n===================================================================="
      "========================================"
      "============\n");
  printf(" RANK/WIDTH SWEEP: num_y=10 \n");
  printf(
      "======================================================================"
      "========================================"
      "==========\n");
  printf(
      "  Rank |  Naive (s) |   LRA_f (s) |  LRA_f Err |  PSWF_f (s) | PSWF_f "
      "Err |   LRA_d (s) |  LRA_d Err |  "
      "PSWF_d (s) | PSWF_d Err |  LRA_dd (s) | LRA_dd Err | PSWF_dd (s) | "
      "PSWF_dd Err\n");
  printf(
      "----------------------------------------------------------------------"
      "----------------------------------------"
      "----------\n");

  int num_y_fixed = 10;
  size_t total_pts = (size_t)num_y_fixed * Mpoints;
  size_t total_out = (size_t)num_y_fixed * max_ff * N;

  float *ys_r_f = malloc(total_pts * sizeof(float));
  float *ys_i_f = malloc(total_pts * sizeof(float));
  double *ys_r_d = malloc(total_pts * sizeof(double));
  double *ys_i_d = malloc(total_pts * sizeof(double));
  dd_t *ys_r_dd = malloc(total_pts * sizeof(dd_t));
  dd_t *ys_i_dd = malloc(total_pts * sizeof(dd_t));
  __float128 *ys_r_q = malloc(total_pts * sizeof(__float128));
  __float128 *ys_i_q = malloc(total_pts * sizeof(__float128));

  for (size_t i = 0; i < total_pts; ++i) {
    double r = 2.0 * rand() / RAND_MAX - 1.0;
    double im = 2.0 * rand() / RAND_MAX - 1.0;
    ys_r_f[i] = (float)r;
    ys_i_f[i] = (float)im;
    ys_r_d[i] = r;
    ys_i_d[i] = im;
    ys_r_dd[i] = dd_make(r, 0.0);
    ys_i_dd[i] = dd_make(im, 0.0);
    ys_r_q[i] = (__float128)r;
    ys_i_q[i] = (__float128)im;
  }

  float *out_lra_r_f = malloc(total_out * sizeof(float));
  float *out_lra_i_f = malloc(total_out * sizeof(float));
  float *out_pswf_r_f = malloc(total_out * sizeof(float));
  float *out_pswf_i_f = malloc(total_out * sizeof(float));

  double *out_lra_r_d = malloc(total_out * sizeof(double));
  double *out_lra_i_d = malloc(total_out * sizeof(double));
  double *out_pswf_r_d = malloc(total_out * sizeof(double));
  double *out_pswf_i_d = malloc(total_out * sizeof(double));
  dd_t *out_lra_r_dd = malloc(total_out * sizeof(dd_t));
  dd_t *out_lra_i_dd = malloc(total_out * sizeof(dd_t));
  dd_t *out_pswf_r_dd = malloc(total_out * sizeof(dd_t));
  dd_t *out_pswf_i_dd = malloc(total_out * sizeof(dd_t));

  __float128 *out_naive_r_q = malloc(total_out * sizeof(__float128));
  __float128 *out_naive_i_q = malloc(total_out * sizeof(__float128));

  // Precompute stored Naive baseline for all freq_factors
  double t0 = now_seconds();
  for (int j = 0; j < num_y_fixed; j++) {
    for (int ff = 1; ff <= max_ff; ++ff) {
      size_t out_idx = (size_t)j * max_ff + (ff - 1);
      nufft1_naive_q(x, ys_r_q + j * Mpoints, ys_i_q + j * Mpoints, Mpoints, N,
                     out_naive_r_q + out_idx * N, out_naive_i_q + out_idx * N,
                     df, ff);
    }
  }
  double t_naive_stored = now_seconds() - t0;

  for (int r = 1; r <= 32; ++r) {
    // LRA Single
    t0 = now_seconds();
    tlsf_nufft_lra_plan *plan_lra_f =
        tlsf_nufft_lra_initialize(Mpoints, N, r, df, max_ff);
    tlsf_nufft_lra_precompute(plan_lra_f, x, Mpoints, N, r);
    for (int j = 0; j < num_y_fixed; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tlsf_nufft_lra_execute(plan_lra_f, ys_r_f + j * Mpoints,
                               ys_i_f + j * Mpoints, out_lra_r_f + out_idx * N,
                               out_lra_i_f + out_idx * N, ff);
      }
    }
    tlsf_nufft_free_lra_plan(plan_lra_f);
    double t_lra_f = now_seconds() - t0;

    // PSWF Single
    t0 = now_seconds();
    tlsf_nufft_pswf_plan *plan_pswf_f =
        tlsf_nufft_pswf_initialize(Mpoints, N, r, df, max_ff);
    tlsf_nufft_pswf_precompute(plan_pswf_f, x);
    for (int j = 0; j < num_y_fixed; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tlsf_nufft_pswf_execute(
            plan_pswf_f, ys_r_f + j * Mpoints, ys_i_f + j * Mpoints,
            out_pswf_r_f + out_idx * N, out_pswf_i_f + out_idx * N, ff);
      }
    }
    tlsf_nufft_free_pswf_plan(plan_pswf_f);
    double t_pswf_f = now_seconds() - t0;

    // LRA Double
    t0 = now_seconds();
    tls_nufft_lra_plan *plan_lra_d =
        tls_nufft_lra_initialize(Mpoints, N, r, df, max_ff);
    tls_nufft_lra_precompute(plan_lra_d, x, Mpoints, N, r);
    for (int j = 0; j < num_y_fixed; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tls_nufft_lra_execute(plan_lra_d, ys_r_d + j * Mpoints,
                              ys_i_d + j * Mpoints, out_lra_r_d + out_idx * N,
                              out_lra_i_d + out_idx * N, ff);
      }
    }
    tls_nufft_free_lra_plan(plan_lra_d);
    double t_lra_d = now_seconds() - t0;

    // PSWF Double
    t0 = now_seconds();
    tls_nufft_pswf_plan *plan_pswf_d =
        tls_nufft_pswf_initialize(Mpoints, N, r, df, max_ff);
    tls_nufft_pswf_precompute(plan_pswf_d, x);
    for (int j = 0; j < num_y_fixed; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tls_nufft_pswf_execute(plan_pswf_d, ys_r_d + j * Mpoints,
                               ys_i_d + j * Mpoints, out_pswf_r_d + out_idx * N,
                               out_pswf_i_d + out_idx * N, ff);
      }
    }
    tls_nufft_free_pswf_plan(plan_pswf_d);
    double t_pswf_d = now_seconds() - t0;

    // LRA Double-Double
    t0 = now_seconds();
    tlsdd_nufft_lra_plan *plan_lra_dd =
        tlsdd_nufft_lra_initialize(Mpoints, N, r, df, max_ff);
    tlsdd_nufft_lra_precompute(plan_lra_dd, x_dd, Mpoints, N, r);
    for (int j = 0; j < num_y_fixed; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tlsdd_nufft_lra_execute(
            plan_lra_dd, ys_r_dd + j * Mpoints, ys_i_dd + j * Mpoints,
            out_lra_r_dd + out_idx * N, out_lra_i_dd + out_idx * N, ff);
      }
    }
    tlsdd_nufft_free_lra_plan(plan_lra_dd);
    double t_lra_dd = now_seconds() - t0;

    // PSWF Double-Double
    t0 = now_seconds();
    tlsdd_nufft_pswf_plan *plan_pswf_dd =
        tlsdd_nufft_pswf_initialize(Mpoints, N, r, df, max_ff);
    tlsdd_nufft_pswf_precompute(plan_pswf_dd, x_dd);
    for (int j = 0; j < num_y_fixed; j++) {
      for (int ff = 1; ff <= max_ff; ++ff) {
        size_t out_idx = (size_t)j * max_ff + (ff - 1);
        tlsdd_nufft_pswf_execute(
            plan_pswf_dd, ys_r_dd + j * Mpoints, ys_i_dd + j * Mpoints,
            out_pswf_r_dd + out_idx * N, out_pswf_i_dd + out_idx * N, ff);
      }
    }
    tlsdd_nufft_free_pswf_plan(plan_pswf_dd);
    double t_pswf_dd = now_seconds() - t0;

    // Errors (comparing to the stored naive calculation over all ff
    // multipliers)
    double err_lra_f =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y_fixed * max_ff,
                        out_lra_r_f, out_lra_i_f, BENCH_FLOAT);
    double err_pswf_f =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y_fixed * max_ff,
                        out_pswf_r_f, out_pswf_i_f, BENCH_FLOAT);
    double err_lra_d =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y_fixed * max_ff,
                        out_lra_r_d, out_lra_i_d, BENCH_DOUBLE);
    double err_pswf_d =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y_fixed * max_ff,
                        out_pswf_r_d, out_pswf_i_d, BENCH_DOUBLE);
    double err_lra_dd =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y_fixed * max_ff,
                        out_lra_r_dd, out_lra_i_dd, BENCH_DD);
    double err_pswf_dd =
        compute_avg_rel(out_naive_r_q, out_naive_i_q, N, num_y_fixed * max_ff,
                        out_pswf_r_dd, out_pswf_i_dd, BENCH_DD);

    printf(" %5d | %10.6f | %11.6f | %10.2e | %11.6f | %10.2e | %11.6f | "
           "%10.2e | %11.6f | %10.2e | %11.6f | %10.2e | %11.6f | %10.2e\n",
           r, t_naive_stored, t_lra_f, err_lra_f, t_pswf_f, err_pswf_f, t_lra_d,
           err_lra_d, t_pswf_d, err_pswf_d, t_lra_dd, err_lra_dd, t_pswf_dd,
           err_pswf_dd);
  }

  free(ys_r_f);
  free(ys_i_f);
  free(ys_r_d);
  free(ys_i_d);
  free(ys_r_dd);
  free(ys_i_dd);
  free(ys_r_q);
  free(ys_i_q);
  free(out_lra_r_f);
  free(out_lra_i_f);
  free(out_pswf_r_f);
  free(out_pswf_i_f);
  free(out_lra_r_d);
  free(out_lra_i_d);
  free(out_pswf_r_d);
  free(out_pswf_i_d);
  free(out_lra_r_dd);
  free(out_lra_i_dd);
  free(out_pswf_r_dd);
  free(out_pswf_i_dd);
  free(out_naive_r_q);
  free(out_naive_i_q);

  free(x);
  free(x_dd);
  return 0;
}
