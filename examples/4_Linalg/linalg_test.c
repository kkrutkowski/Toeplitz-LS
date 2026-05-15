/* =============================================================================
 * linalg_test.c - Driver for vector-batched Levinson vs Zohar benchmarks.
 * =============================================================================
 */

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <linalg.h>

#define WARMUP_CALLS 64

static double tls_clock(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static void *tls_alloc(size_t count, size_t size, size_t align) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, align, count * size) != 0) {
    fprintf(stderr, "allocation failed\n");
    abort();
  }
  return ptr;
}

static size_t solver_calls_for(size_t num_systems, size_t lanes) {
  return (num_systems + lanes - 1) / lanes;
}

static double rand_signed_unit(void) {
  return ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

static VECF vecf_splat(float value) {
  VECF v;
  for (int lane = 0; lane < VECF_LEN; lane++)
    v[lane] = value;
  return v;
}

static VEC vec_splat(double value) {
  VEC v;
  for (int lane = 0; lane < VEC_LEN; lane++)
    v[lane] = value;
  return v;
}

static dd_t dd_from_double(double value) {
  return (dd_t){value, 0.0};
}

static dd_t dd_neg_local(dd_t value) { return dd_make(-value.hi, -value.lo); }

static void print_result(const char *label, size_t n, size_t lanes,
                         size_t calls, double t_lev, double t_zoh,
                         double rel_diff) {
  double systems = (double)calls * (double)lanes;
  printf("[%-13s] N=%-4zu lanes=%-2zu calls=%-5zu Levinson: %9.6f s "
         "(%8.2f ns/system)  Zohar: %9.6f s (%8.2f ns/system)  rel-diff: "
         "%.3e\n",
         label, n, lanes, calls, t_lev, t_lev * 1e9 / systems, t_zoh,
         t_zoh * 1e9 / systems, rel_diff);
}

typedef struct {
  int f_levinson_faster;
  int d_levinson_faster;
  int dd_levinson_faster;
} benchmark_result;

static benchmark_result benchmark_solvers(size_t d, size_t num_systems) {
  size_t n = 2 * d + 1;
  double scale = 0.45 / (double)n;
  benchmark_result result = {0, 0, 0};

  {
    size_t calls = solver_calls_for(num_systems, VECF_LEN);

    VECF *R_r = tls_alloc(n + 1, sizeof(*R_r), __alignof__(VECF));
    VECF *R_i = tls_alloc(n + 1, sizeof(*R_i), __alignof__(VECF));
    VECF *y_r = tls_alloc(n, sizeof(*y_r), __alignof__(VECF));
    VECF *y_i = tls_alloc(n, sizeof(*y_i), __alignof__(VECF));
    VECF *x_exact_r = tls_alloc(n, sizeof(*x_exact_r), __alignof__(VECF));
    VECF *x_exact_i = tls_alloc(n, sizeof(*x_exact_i), __alignof__(VECF));
    VECF *x_lev_r = tls_alloc(n, sizeof(*x_lev_r), __alignof__(VECF));
    VECF *x_lev_i = tls_alloc(n, sizeof(*x_lev_i), __alignof__(VECF));
    VECF *x_zoh_r = tls_alloc(n, sizeof(*x_zoh_r), __alignof__(VECF));
    VECF *x_zoh_i = tls_alloc(n, sizeof(*x_zoh_i), __alignof__(VECF));
    VECF *a_r = tls_alloc(n, sizeof(*a_r), __alignof__(VECF));
    VECF *a_i = tls_alloc(n, sizeof(*a_i), __alignof__(VECF));
    VECF *a_prev_r = tls_alloc(n, sizeof(*a_prev_r), __alignof__(VECF));
    VECF *a_prev_i = tls_alloc(n, sizeof(*a_prev_i), __alignof__(VECF));
    VECF *e_hat_r = tls_alloc(n, sizeof(*e_hat_r), __alignof__(VECF));
    VECF *e_hat_i = tls_alloc(n, sizeof(*e_hat_i), __alignof__(VECF));
    VECF *e_hat_prev_r = tls_alloc(n, sizeof(*e_hat_prev_r), __alignof__(VECF));
    VECF *e_hat_prev_i = tls_alloc(n, sizeof(*e_hat_prev_i), __alignof__(VECF));

    R_r[0] = vecf_splat((float)n);
    R_i[0] = vecf_splat(0.0f);
    for (size_t k = 1; k <= n; k++) {
      VECF rr, ri;
      for (int lane = 0; lane < VECF_LEN; lane++) {
        rr[lane] = (float)(rand_signed_unit() * scale);
        ri[lane] = (float)(rand_signed_unit() * scale);
      }
      R_r[k] = rr;
      R_i[k] = ri;
    }

    for (size_t i = 0; i < n; i++) {
      VECF xr, xi;
      for (int lane = 0; lane < VECF_LEN; lane++) {
        xr[lane] = (float)rand_signed_unit();
        xi[lane] = (float)rand_signed_unit();
      }
      x_exact_r[i] = xr;
      x_exact_i[i] = xi;
    }

    for (size_t i = 0; i < n; i++) {
      VECF sr = vecf_splat(0.0f);
      VECF si = vecf_splat(0.0f);
      for (size_t j = 0; j < n; j++) {
        VECF Hr, Hi;
        if (i >= j) {
          Hr = R_r[i - j];
          Hi = R_i[i - j];
        } else {
          Hr = R_r[j - i];
          Hi = -R_i[j - i];
        }
        sr += Hr * x_exact_r[j] - Hi * x_exact_i[j];
        si += Hr * x_exact_i[j] + Hi * x_exact_r[j];
      }
      y_r[i] = sr;
      y_i[i] = si;
    }

    size_t warmup = calls < WARMUP_CALLS ? calls : WARMUP_CALLS;
    for (size_t call = 0; call < warmup; call++)
      tlsf_solve_levinson(n, R_r, R_i, y_r, y_i, x_lev_r, x_lev_i, a_r, a_i,
                          a_prev_r, a_prev_i);

    double t0 = tls_clock();
    for (size_t call = 0; call < calls; call++) {
      tlsf_solve_levinson(n, R_r, R_i, y_r, y_i, x_lev_r, x_lev_i, a_r, a_i,
                          a_prev_r, a_prev_i);
    }
    double t_lev = tls_clock() - t0;

    for (size_t call = 0; call < warmup; call++)
      tlsf_solve_zohar(n, R_r, R_i, y_r, y_i, x_zoh_r, x_zoh_i, e_hat_r,
                       e_hat_i, e_hat_prev_r, e_hat_prev_i);

    t0 = tls_clock();
    for (size_t call = 0; call < calls; call++) {
      tlsf_solve_zohar(n, R_r, R_i, y_r, y_i, x_zoh_r, x_zoh_i, e_hat_r,
                       e_hat_i, e_hat_prev_r, e_hat_prev_i);
    }
    double t_zoh = tls_clock() - t0;
    result.f_levinson_faster = t_lev < t_zoh;

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
      for (int lane = 0; lane < VECF_LEN; lane++) {
        double dr = (double)x_lev_r[i][lane] - (double)x_zoh_r[i][lane];
        double di = (double)x_lev_i[i][lane] - (double)x_zoh_i[i][lane];
        double lr = (double)x_lev_r[i][lane];
        double li = (double)x_lev_i[i][lane];
        num += dr * dr + di * di;
        den += lr * lr + li * li;
      }
    }
    print_result("float", n, VECF_LEN, calls, t_lev, t_zoh,
                 sqrt(num / (den + 1e-300)));

    free(R_r);
    free(R_i);
    free(y_r);
    free(y_i);
    free(x_exact_r);
    free(x_exact_i);
    free(x_lev_r);
    free(x_lev_i);
    free(x_zoh_r);
    free(x_zoh_i);
    free(a_r);
    free(a_i);
    free(a_prev_r);
    free(a_prev_i);
    free(e_hat_r);
    free(e_hat_i);
    free(e_hat_prev_r);
    free(e_hat_prev_i);
  }

  {
    size_t calls = solver_calls_for(num_systems, VEC_LEN);

    VEC *R_r = tls_alloc(n + 1, sizeof(*R_r), __alignof__(VEC));
    VEC *R_i = tls_alloc(n + 1, sizeof(*R_i), __alignof__(VEC));
    VEC *y_r = tls_alloc(n, sizeof(*y_r), __alignof__(VEC));
    VEC *y_i = tls_alloc(n, sizeof(*y_i), __alignof__(VEC));
    VEC *x_exact_r = tls_alloc(n, sizeof(*x_exact_r), __alignof__(VEC));
    VEC *x_exact_i = tls_alloc(n, sizeof(*x_exact_i), __alignof__(VEC));
    VEC *x_lev_r = tls_alloc(n, sizeof(*x_lev_r), __alignof__(VEC));
    VEC *x_lev_i = tls_alloc(n, sizeof(*x_lev_i), __alignof__(VEC));
    VEC *x_zoh_r = tls_alloc(n, sizeof(*x_zoh_r), __alignof__(VEC));
    VEC *x_zoh_i = tls_alloc(n, sizeof(*x_zoh_i), __alignof__(VEC));
    VEC *a_r = tls_alloc(n, sizeof(*a_r), __alignof__(VEC));
    VEC *a_i = tls_alloc(n, sizeof(*a_i), __alignof__(VEC));
    VEC *a_prev_r = tls_alloc(n, sizeof(*a_prev_r), __alignof__(VEC));
    VEC *a_prev_i = tls_alloc(n, sizeof(*a_prev_i), __alignof__(VEC));
    VEC *e_hat_r = tls_alloc(n, sizeof(*e_hat_r), __alignof__(VEC));
    VEC *e_hat_i = tls_alloc(n, sizeof(*e_hat_i), __alignof__(VEC));
    VEC *e_hat_prev_r = tls_alloc(n, sizeof(*e_hat_prev_r), __alignof__(VEC));
    VEC *e_hat_prev_i = tls_alloc(n, sizeof(*e_hat_prev_i), __alignof__(VEC));

    R_r[0] = vec_splat((double)n);
    R_i[0] = vec_splat(0.0);
    for (size_t k = 1; k <= n; k++) {
      VEC rr, ri;
      for (int lane = 0; lane < VEC_LEN; lane++) {
        rr[lane] = rand_signed_unit() * scale;
        ri[lane] = rand_signed_unit() * scale;
      }
      R_r[k] = rr;
      R_i[k] = ri;
    }

    for (size_t i = 0; i < n; i++) {
      VEC xr, xi;
      for (int lane = 0; lane < VEC_LEN; lane++) {
        xr[lane] = rand_signed_unit();
        xi[lane] = rand_signed_unit();
      }
      x_exact_r[i] = xr;
      x_exact_i[i] = xi;
    }

    for (size_t i = 0; i < n; i++) {
      VEC sr = vec_splat(0.0);
      VEC si = vec_splat(0.0);
      for (size_t j = 0; j < n; j++) {
        VEC Hr, Hi;
        if (i >= j) {
          Hr = R_r[i - j];
          Hi = R_i[i - j];
        } else {
          Hr = R_r[j - i];
          Hi = -R_i[j - i];
        }
        sr += Hr * x_exact_r[j] - Hi * x_exact_i[j];
        si += Hr * x_exact_i[j] + Hi * x_exact_r[j];
      }
      y_r[i] = sr;
      y_i[i] = si;
    }

    size_t warmup = calls < WARMUP_CALLS ? calls : WARMUP_CALLS;
    for (size_t call = 0; call < warmup; call++)
      tls_solve_levinson(n, R_r, R_i, y_r, y_i, x_lev_r, x_lev_i, a_r, a_i,
                         a_prev_r, a_prev_i);

    double t0 = tls_clock();
    for (size_t call = 0; call < calls; call++) {
      tls_solve_levinson(n, R_r, R_i, y_r, y_i, x_lev_r, x_lev_i, a_r, a_i,
                         a_prev_r, a_prev_i);
    }
    double t_lev = tls_clock() - t0;

    for (size_t call = 0; call < warmup; call++)
      tls_solve_zohar(n, R_r, R_i, y_r, y_i, x_zoh_r, x_zoh_i, e_hat_r, e_hat_i,
                      e_hat_prev_r, e_hat_prev_i);

    t0 = tls_clock();
    for (size_t call = 0; call < calls; call++) {
      tls_solve_zohar(n, R_r, R_i, y_r, y_i, x_zoh_r, x_zoh_i, e_hat_r, e_hat_i,
                      e_hat_prev_r, e_hat_prev_i);
    }
    double t_zoh = tls_clock() - t0;
    result.d_levinson_faster = t_lev < t_zoh;

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
      for (int lane = 0; lane < VEC_LEN; lane++) {
        double dr = x_lev_r[i][lane] - x_zoh_r[i][lane];
        double di = x_lev_i[i][lane] - x_zoh_i[i][lane];
        double lr = x_lev_r[i][lane];
        double li = x_lev_i[i][lane];
        num += dr * dr + di * di;
        den += lr * lr + li * li;
      }
    }
    print_result("double", n, VEC_LEN, calls, t_lev, t_zoh,
                 sqrt(num / (den + 1e-300)));

    free(R_r);
    free(R_i);
    free(y_r);
    free(y_i);
    free(x_exact_r);
    free(x_exact_i);
    free(x_lev_r);
    free(x_lev_i);
    free(x_zoh_r);
    free(x_zoh_i);
    free(a_r);
    free(a_i);
    free(a_prev_r);
    free(a_prev_i);
    free(e_hat_r);
    free(e_hat_i);
    free(e_hat_prev_r);
    free(e_hat_prev_i);
  }

  {
    size_t calls = solver_calls_for(num_systems, 1);

    dd_t *R_r = tls_alloc(n + 1, sizeof(*R_r), __alignof__(dd_t));
    dd_t *R_i = tls_alloc(n + 1, sizeof(*R_i), __alignof__(dd_t));
    dd_t *y_r = tls_alloc(n, sizeof(*y_r), __alignof__(dd_t));
    dd_t *y_i = tls_alloc(n, sizeof(*y_i), __alignof__(dd_t));
    dd_t *x_exact_r = tls_alloc(n, sizeof(*x_exact_r), __alignof__(dd_t));
    dd_t *x_exact_i = tls_alloc(n, sizeof(*x_exact_i), __alignof__(dd_t));
    dd_t *x_lev_r = tls_alloc(n, sizeof(*x_lev_r), __alignof__(dd_t));
    dd_t *x_lev_i = tls_alloc(n, sizeof(*x_lev_i), __alignof__(dd_t));
    dd_t *x_zoh_r = tls_alloc(n, sizeof(*x_zoh_r), __alignof__(dd_t));
    dd_t *x_zoh_i = tls_alloc(n, sizeof(*x_zoh_i), __alignof__(dd_t));
    dd_t *a_r = tls_alloc(n, sizeof(*a_r), __alignof__(dd_t));
    dd_t *a_i = tls_alloc(n, sizeof(*a_i), __alignof__(dd_t));
    dd_t *a_prev_r = tls_alloc(n, sizeof(*a_prev_r), __alignof__(dd_t));
    dd_t *a_prev_i = tls_alloc(n, sizeof(*a_prev_i), __alignof__(dd_t));
    dd_t *e_hat_r = tls_alloc(n, sizeof(*e_hat_r), __alignof__(dd_t));
    dd_t *e_hat_i = tls_alloc(n, sizeof(*e_hat_i), __alignof__(dd_t));
    dd_t *e_hat_prev_r = tls_alloc(n, sizeof(*e_hat_prev_r), __alignof__(dd_t));
    dd_t *e_hat_prev_i = tls_alloc(n, sizeof(*e_hat_prev_i), __alignof__(dd_t));

    R_r[0] = dd_from_double((double)n);
    R_i[0] = dd_from_double(0.0);
    for (size_t k = 1; k <= n; k++) {
      R_r[k] = dd_from_double(rand_signed_unit() * scale);
      R_i[k] = dd_from_double(rand_signed_unit() * scale);
    }

    for (size_t i = 0; i < n; i++) {
      x_exact_r[i] = dd_from_double(rand_signed_unit());
      x_exact_i[i] = dd_from_double(rand_signed_unit());
    }

    for (size_t i = 0; i < n; i++) {
      dd_t sr = dd_from_double(0.0);
      dd_t si = dd_from_double(0.0);
      for (size_t j = 0; j < n; j++) {
        dd_t Hr, Hi;
        if (i >= j) {
          Hr = R_r[i - j];
          Hi = R_i[i - j];
        } else {
          Hr = R_r[j - i];
          Hi = dd_neg_local(R_i[j - i]);
        }
        sr = dd_add(sr,
                    dd_sub(dd_mul(Hr, x_exact_r[j]), dd_mul(Hi, x_exact_i[j])));
        si = dd_add(si,
                    dd_add(dd_mul(Hr, x_exact_i[j]), dd_mul(Hi, x_exact_r[j])));
      }
      y_r[i] = sr;
      y_i[i] = si;
    }

    size_t warmup = calls < WARMUP_CALLS ? calls : WARMUP_CALLS;
    for (size_t call = 0; call < warmup; call++)
      tlsdd_solve_levinson(n, R_r, R_i, y_r, y_i, x_lev_r, x_lev_i, a_r, a_i,
                           a_prev_r, a_prev_i);

    double t0 = tls_clock();
    for (size_t call = 0; call < calls; call++) {
      tlsdd_solve_levinson(n, R_r, R_i, y_r, y_i, x_lev_r, x_lev_i, a_r, a_i,
                           a_prev_r, a_prev_i);
    }
    double t_lev = tls_clock() - t0;

    for (size_t call = 0; call < warmup; call++)
      tlsdd_solve_zohar(n, R_r, R_i, y_r, y_i, x_zoh_r, x_zoh_i, e_hat_r,
                        e_hat_i, e_hat_prev_r, e_hat_prev_i);

    t0 = tls_clock();
    for (size_t call = 0; call < calls; call++) {
      tlsdd_solve_zohar(n, R_r, R_i, y_r, y_i, x_zoh_r, x_zoh_i, e_hat_r,
                        e_hat_i, e_hat_prev_r, e_hat_prev_i);
    }
    double t_zoh = tls_clock() - t0;
    result.dd_levinson_faster = t_lev < t_zoh;

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
      double dr = dd_to_double(x_lev_r[i]) - dd_to_double(x_zoh_r[i]);
      double di = dd_to_double(x_lev_i[i]) - dd_to_double(x_zoh_i[i]);
      double lr = dd_to_double(x_lev_r[i]);
      double li = dd_to_double(x_lev_i[i]);
      num += dr * dr + di * di;
      den += lr * lr + li * li;
    }
    print_result("double-double", n, 1, calls, t_lev, t_zoh,
                 sqrt(num / (den + 1e-300)));

    free(R_r);
    free(R_i);
    free(y_r);
    free(y_i);
    free(x_exact_r);
    free(x_exact_i);
    free(x_lev_r);
    free(x_lev_i);
    free(x_zoh_r);
    free(x_zoh_i);
    free(a_r);
    free(a_i);
    free(a_prev_r);
    free(a_prev_i);
    free(e_hat_r);
    free(e_hat_i);
    free(e_hat_prev_r);
    free(e_hat_prev_i);
  }

  return result;
}

#ifdef SAVE
static int save_linalg_params(int f_max_d, int d_max_d, int dd_max_d) {
  const char *fname = "linalg_params.mk";
  FILE *f = fopen(fname, "w");
  if (!f) {
    perror("fopen linalg_params.mk");
    return 1;
  }

  fprintf(f, "F_LINALG_DEFS := -DF_LINALG_LEVINSON_MAX_D=%d\n", f_max_d);
  fprintf(f, "D_LINALG_DEFS := -DD_LINALG_LEVINSON_MAX_D=%d\n", d_max_d);
  fprintf(f, "DD_LINALG_DEFS := -DDD_LINALG_LEVINSON_MAX_D=%d\n", dd_max_d);
  fclose(f);

  printf("[SAVE] Written linalg params to %s\n", fname);
  return 0;
}
#endif

int main(void) {
  srand(42);

  static const size_t d_sizes[] = {1, 2,  3,  4,  5,  6,  7, 8,
                                   9, 10, 11, 12, 13, 14, 15, 32, 48, 64, 96, 128};
  int num_sizes = (int)(sizeof(d_sizes) / sizeof(d_sizes[0]));
  size_t num_systems = 16000;
  int f_levinson_max_d = 0;
  int d_levinson_max_d = 0;
  int dd_levinson_max_d = 0;

  /* Pending candidate: the first d where Zohar won.  Confirmed only after
   * Zohar also wins at the immediately following d (consecutive requirement).
   */
  int f_zohar_candidate = 0;
  int d_zohar_candidate = 0;
  int dd_zohar_candidate = 0;

  printf("Levinson vs Zohar vector-batched benchmark (N = 2d+1, systems = "
         "%zu)\n",
         num_systems);
  printf("The call count is ceil(systems / SIMD lanes) for each precision.\n");
  printf("---------------------------------------------------------------------"
         "--------------------------------------------------\n");

  for (int i = 0; i < num_sizes; i++) {
    benchmark_result result = benchmark_solvers(d_sizes[i], num_systems);

    /* float ---------------------------------------------------------------- */
    if (!result.f_levinson_faster) {
      if (f_zohar_candidate && !f_levinson_max_d)
        f_levinson_max_d =
            f_zohar_candidate; /* confirmed: two consecutive wins */
      if (!f_levinson_max_d)
        f_zohar_candidate =
            (int)d_sizes[i]; /* first win — wait for confirmation */
    } else {
      f_zohar_candidate = 0; /* Levinson won again — reset */
    }

    /* double --------------------------------------------------------------- */
    if (!result.d_levinson_faster) {
      if (d_zohar_candidate && !d_levinson_max_d)
        d_levinson_max_d = d_zohar_candidate;
      if (!d_levinson_max_d)
        d_zohar_candidate = (int)d_sizes[i];
    } else {
      d_zohar_candidate = 0;
    }

    /* double-double -------------------------------------------------------- */
    if (!result.dd_levinson_faster) {
      if (dd_zohar_candidate && !dd_levinson_max_d)
        dd_levinson_max_d = dd_zohar_candidate;
      if (!dd_levinson_max_d)
        dd_zohar_candidate = (int)d_sizes[i];
    } else {
      dd_zohar_candidate = 0;
    }

    printf("\n");
  }

  /* Fallback: Levinson was fastest (or never confirmed consecutive wins)
   * across all tested sizes — use the last tested d as the upper bound.
   * Note: use d_sizes[num_sizes-1], not num_sizes, so this stays correct
   * if the table is ever extended. */
  if (!f_levinson_max_d)
    f_levinson_max_d = (int)d_sizes[num_sizes - 1];
  if (!d_levinson_max_d)
    d_levinson_max_d = (int)d_sizes[num_sizes - 1];
  if (!dd_levinson_max_d)
    dd_levinson_max_d = (int)d_sizes[num_sizes - 1];

#ifdef SAVE
  if (save_linalg_params(f_levinson_max_d, d_levinson_max_d,
                         dd_levinson_max_d) != 0)
    return 1;
#endif

  return 0;
}
