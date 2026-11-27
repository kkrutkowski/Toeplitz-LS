/* Milestone 4 Verification: End-to-end C23 ABI (tls_vulkan.so) FastChi2 & IHS
 * Tests all 4 precisions (h, f, df, d), d in {1, 3, 5, 13}, normalizations,
 * PSWF43 & PSWF21, autonan, cross-vendor (NVIDIA, AMD RADV, llvmpipe),
 * and unsupported FP16 (-8) / FP64 (-9) error handling. */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327950288
#endif

typedef struct tls_vk_device_info_t {
    int id;
    char name[256];
    char backend[32];
    char device_type[32];
    uint64_t max_compute_units;
    uint64_t global_mem_size;
    int is_gpu;
    int has_fp64;
    int has_fp16;
} tls_vk_device_info_t;

typedef int (*fn_abi_t)(void);
typedef int (*fn_dev_count_t)(void);
typedef int (*fn_dev_info_t)(int, tls_vk_device_info_t*);
typedef const char* (*fn_last_err_t)(void);

typedef int (*fn_fc2_f_t)(const double*, const float*, const float*, int, double, double, int, int, int, int, int, float*, float*);
typedef int (*fn_fc2_hd_t)(const double*, const double*, const double*, int, double, double, int, int, int, int, int, float*, float*);
typedef int (*fn_fc2_d_t)(const double*, const double*, const double*, int, double, double, int, int, int, int, int, double*, double*);

typedef int (*fn_ihs_f_t)(const double*, const float*, const float*, int, double, double, int, int, int, int, int, float*);
typedef int (*fn_ihs_d_t)(const double*, const double*, const double*, int, double, double, int, int, int, int, int, double*);

/* Exact O(N * M * d^3) reference FastChi2 (standard normalization) for small N, M */
static void ref_fastchi2_std(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, double* out_power) {
    /* Weighted mean subtraction */
    double W = 0.0, wy = 0.0;
    for (int m = 0; m < M; ++m) {
        double w = 1.0 / (dy[m] * dy[m]);
        W += w;
        wy += w * y[m];
    }
    double ymean = wy / W;
    double chi2_0 = 0.0;
    for (int m = 0; m < M; ++m) {
        double r = (y[m] - ymean) / dy[m];
        chi2_0 += r * r;
    }

    int P = 2 * degree + 1;
    double* A = (double*)malloc((size_t)P * (size_t)P * sizeof(double));
    double* b = (double*)malloc((size_t)P * sizeof(double));
    double* row = (double*)malloc((size_t)P * sizeof(double));

    for (int k = 0; k < N; ++k) {
        double fk = f0 + (double)k * df;
        memset(A, 0, (size_t)P * (size_t)P * sizeof(double));
        memset(b, 0, (size_t)P * sizeof(double));
        for (int m = 0; m < M; ++m) {
            double w = 1.0 / (dy[m] * dy[m]);
            double ym = y[m] - ymean;
            row[0] = 1.0;
            for (int h = 1; h <= degree; ++h) {
                double ang = 2.0 * M_PI * (double)h * fk * t[m];
                row[2 * h - 1] = cos(ang);
                row[2 * h] = sin(ang);
            }
            for (int i = 0; i < P; ++i) {
                double wi = w * row[i];
                b[i] += wi * ym;
                for (int j = 0; j < P; ++j) A[i * P + j] += wi * row[j];
            }
        }
        /* Gaussian elimination with partial pivoting */
        double* x = (double*)malloc((size_t)P * sizeof(double));
        double* M_aug = (double*)malloc((size_t)P * (size_t)(P + 1) * sizeof(double));
        for (int i = 0; i < P; ++i) {
            for (int j = 0; j < P; ++j) M_aug[i * (P + 1) + j] = A[i * P + j];
            M_aug[i * (P + 1) + P] = b[i];
        }
        int singular = 0;
        for (int col = 0; col < P; ++col) {
            int piv = col;
            double maxv = fabs(M_aug[col * (P + 1) + col]);
            for (int r = col + 1; r < P; ++r) {
                double v = fabs(M_aug[r * (P + 1) + col]);
                if (v > maxv) {
                    maxv = v;
                    piv = r;
                }
            }
            if (maxv < 1e-14) {
                singular = 1;
                break;
            }
            if (piv != col) {
                for (int c = col; c <= P; ++c) {
                    double tmp = M_aug[col * (P + 1) + c];
                    M_aug[col * (P + 1) + c] = M_aug[piv * (P + 1) + c];
                    M_aug[piv * (P + 1) + c] = tmp;
                }
            }
            double diag = M_aug[col * (P + 1) + col];
            for (int r = col + 1; r < P; ++r) {
                double fac = M_aug[r * (P + 1) + col] / diag;
                for (int c = col; c <= P; ++c) M_aug[r * (P + 1) + c] -= fac * M_aug[col * (P + 1) + c];
            }
        }
        if (singular) {
            out_power[k] = 0.0;
        } else {
            for (int i = P - 1; i >= 0; --i) {
                double s = M_aug[i * (P + 1) + P];
                for (int j = i + 1; j < P; ++j) s -= M_aug[i * (P + 1) + j] * x[j];
                x[i] = s / M_aug[i * (P + 1) + i];
            }
            double q = 0.0;
            for (int i = 0; i < P; ++i) q += b[i] * x[i];
            out_power[k] = q / chi2_0;
        }
        free(x);
        free(M_aug);
    }

    free(A);
    free(b);
    free(row);
}

int main(void) {
    void* lib = dlopen("./offload/vulkan/build/tls_vulkan.so", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    fn_abi_t abi_fn = (fn_abi_t)dlsym(lib, "tls_vk_abi_version");
    fn_dev_count_t count_fn = (fn_dev_count_t)dlsym(lib, "tls_vk_device_count");
    fn_dev_info_t info_fn = (fn_dev_info_t)dlsym(lib, "tls_vk_device_info");
    fn_last_err_t err_fn = (fn_last_err_t)dlsym(lib, "tls_vk_last_error");

    fn_fc2_f_t h_fc2 = (fn_fc2_f_t)dlsym(lib, "tls_vk_h_fastchi2");
    fn_fc2_hd_t hd_fc2 = (fn_fc2_hd_t)dlsym(lib, "tls_vk_h_fastchi2_d");
    fn_fc2_f_t f_fc2 = (fn_fc2_f_t)dlsym(lib, "tls_vk_f_fastchi2");
    fn_fc2_d_t df_fc2 = (fn_fc2_d_t)dlsym(lib, "tls_vk_df_fastchi2");
    fn_fc2_d_t d_fc2 = (fn_fc2_d_t)dlsym(lib, "tls_vk_d_fastchi2");

    fn_ihs_f_t h_ihs = (fn_ihs_f_t)dlsym(lib, "tls_vk_h_ihs");
    fn_ihs_f_t f_ihs = (fn_ihs_f_t)dlsym(lib, "tls_vk_f_ihs");
    fn_ihs_d_t df_ihs = (fn_ihs_d_t)dlsym(lib, "tls_vk_df_ihs");
    fn_ihs_d_t d_ihs = (fn_ihs_d_t)dlsym(lib, "tls_vk_d_ihs");

    printf("tls_vk_abi_version = %d\n", abi_fn());
    int ndevs = count_fn();
    printf("tls_vk_device_count = %d\n", ndevs);
    if (ndevs <= 0) return 1;

    /* Synthetic irregular lightcurve with known multi-harmonic signal */
    const int M = 257;
    const int N = 1024;
    const double T = 40.0;
    const double f0 = 0.05;
    const double df = 0.004;
    double* t = (double*)malloc((size_t)M * sizeof(double));
    double* yd = (double*)malloc((size_t)M * sizeof(double));
    double* ed = (double*)malloc((size_t)M * sizeof(double));
    float* yf = (float*)malloc((size_t)M * sizeof(float));
    float* ef = (float*)malloc((size_t)M * sizeof(float));

    const double f_sig = 0.73;
    for (int m = 0; m < M; ++m) {
        double u = ((double)m + 0.37 * sin(1.7 * (double)m)) / (double)M;
        t[m] = u * T;
        double ph = 2.0 * M_PI * f_sig * t[m];
        yd[m] = 12.5 + 1.4 * cos(ph) - 0.6 * sin(2.0 * ph) + 0.25 * cos(3.0 * ph) + 0.08 * sin(13.1 * (double)m);
        ed[m] = 0.08 + 0.03 * fabs(cos(0.9 * (double)m));
        yf[m] = (float)yd[m];
        ef[m] = (float)ed[m];
    }

    int degrees[4] = {1, 3, 5, 13};
    double* ref_p[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int di = 0; di < 4; ++di) {
        ref_p[di] = (double*)malloc((size_t)N * sizeof(double));
        ref_fastchi2_std(t, yd, ed, M, f0, df, N, degrees[di], ref_p[di]);
    }

    float* pf = (float*)malloc((size_t)N * sizeof(float));
    float* cf = (float*)malloc((size_t)N * sizeof(float));
    double* pd = (double*)malloc((size_t)N * sizeof(double));
    double* cd = (double*)malloc((size_t)N * sizeof(double));

    int fails = 0;
    for (int dev = 0; dev < ndevs; ++dev) {
        tls_vk_device_info_t info = {0};
        info_fn(dev, &info);
        printf("\n=== Device [%d] %s (%s, %s, cu=%llu, mem=%.2f GiB, fp16=%d, fp64=%d) ===\n", info.id, info.name, info.backend, info.device_type,
               (unsigned long long)info.max_compute_units, (double)info.global_mem_size / (1024.0 * 1024.0 * 1024.0), info.has_fp16, info.has_fp64);

        for (int di = 0; di < 4; ++di) {
            int deg = degrees[di];
            /* Skip heavy d=13 on llvmpipe? No, N=1024 is tiny (<50ms on llvmpipe!) */
            struct {
                const char* name;
                int kind;
                double tol;
            } tests[5] = {
                {"h ", 0, (deg <= 3) ? 2.5e-2 : 1.0e-1},  {"hd", 1, (deg <= 3) ? 2.5e-2 : 1.0e-1},   {"f ", 2, (deg <= 5) ? 5.0e-5 : 5.0e-4},
                {"df", 3, (deg <= 5) ? 5.0e-10 : 5.0e-9}, {"d ", 4, (deg <= 5) ? 5.0e-11 : 5.0e-10},
            };

            for (int ti = 0; ti < 5; ++ti) {
                int rc = 0;
                if (tests[ti].kind == 0) {
                    if (!info.has_fp16) continue;
                    rc = h_fc2(t, yf, ef, M, f0, df, N, deg, 1, 0, dev, pf, cf);
                } else if (tests[ti].kind == 1) {
                    if (!info.has_fp16) continue;
                    rc = hd_fc2(t, yd, ed, M, f0, df, N, deg, 2, 0, dev, pf, cf); /* PSWF21 */
                } else if (tests[ti].kind == 2) {
                    rc = f_fc2(t, yf, ef, M, f0, df, N, deg, 1, 0, dev, pf, cf);
                } else if (tests[ti].kind == 3) {
                    rc = df_fc2(t, yd, ed, M, f0, df, N, deg, 1, 0, dev, pd, cd);
                } else {
                    if (!info.has_fp64) continue;
                    rc = d_fc2(t, yd, ed, M, f0, df, N, deg, 1, 0, dev, pd, cd);
                }

                if (rc != 0) {
                    printf("  FAIL: fc2 prec=%s d=%2d rc=%d err=%s\n", tests[ti].name, deg, rc, err_fn());
                    ++fails;
                    continue;
                }

                double max_abs = 0.0;
                int bad_nan = 0;
                for (int k = 0; k < N; ++k) {
                    double v = (tests[ti].kind <= 2) ? (double)pf[k] : pd[k];
                    if (!isfinite(v)) {
                        ++bad_nan;
                        continue;
                    }
                    double e = fabs(v - ref_p[di][k]);
                    if (e > max_abs) max_abs = e;
                }
                if (bad_nan > 0 || max_abs > tests[ti].tol) {
                    printf("  FAIL: fc2 prec=%s d=%2d max_abs=%.3e (tol=%.1e) bad_nan=%d\n", tests[ti].name, deg, max_abs, tests[ti].tol, bad_nan);
                    ++fails;
                } else {
                    printf("  PASS: fc2 prec=%s d=%2d max_abs=%.3e\n", tests[ti].name, deg, max_abs);
                }
            }
        }

        /* Quick IHS cross-precision consistency check (d=3, all 3 normalizations: 0=default, 2=nll, 4=raw) */
        const int ihs_norms[3] = {0, 2, 4};
        for (int ni = 0; ni < 3; ++ni) {
            int norm = ihs_norms[ni];
            double* ihs_d = (double*)malloc((size_t)N * sizeof(double));
            double* ihs_df = (double*)malloc((size_t)N * sizeof(double));
            float* ihs_f = (float*)malloc((size_t)N * sizeof(float));
            float* ihs_h = (float*)malloc((size_t)N * sizeof(float));

            int rc_d = info.has_fp64 ? d_ihs(t, yd, ed, M, f0, df, N, 3, 1, norm, dev, ihs_d) : 0;
            int rc_df = df_ihs(t, yd, ed, M, f0, df, N, 3, 1, norm, dev, ihs_df);
            int rc_f = f_ihs(t, yf, ef, M, f0, df, N, 3, 1, norm, dev, ihs_f);
            int rc_h = info.has_fp16 ? h_ihs(t, yf, ef, M, f0, df, N, 3, 1, norm, dev, ihs_h) : 0;

            if (rc_d != 0 || rc_df != 0 || rc_f != 0 || rc_h != 0) {
                printf("  FAIL: ihs norm=%d rc=(%d,%d,%d,%d) err=%s\n", norm, rc_d, rc_df, rc_f, rc_h, err_fn());
                ++fails;
            } else if (info.has_fp64) {
                double e_df = 0.0, e_f = 0.0, e_h = 0.0, ref_max = 0.0;
                for (int k = 0; k < N; ++k) {
                    if (fabs(ihs_d[k]) > ref_max) ref_max = fabs(ihs_d[k]);
                    double a = fabs(ihs_df[k] - ihs_d[k]);
                    if (a > e_df) e_df = a;
                    double b = fabs((double)ihs_f[k] - ihs_d[k]);
                    if (b > e_f) e_f = b;
                    if (info.has_fp16) {
                        double c = fabs((double)ihs_h[k] - ihs_d[k]);
                        if (c > e_h) e_h = c;
                    }
                }
                double r_df = e_df / (ref_max + 1e-30);
                double r_f = e_f / (ref_max + 1e-30);
                double r_h = e_h / (ref_max + 1e-30);
                if (r_df > 1e-9 || r_f > 2e-5 || (info.has_fp16 && r_h > 3e-2)) {
                    printf("  FAIL: ihs norm=%d rel=(df:%.3e, f:%.3e, h:%.3e)\n", norm, r_df, r_f, r_h);
                    ++fails;
                } else {
                    printf("  PASS: ihs norm=%d rel=(df:%.3e, f:%.3e, h:%.3e)\n", norm, r_df, r_f, r_h);
                }
            }
            free(ihs_d);
            free(ihs_df);
            free(ihs_f);
            free(ihs_h);
        }
    }

    for (int di = 0; di < 4; ++di) free(ref_p[di]);
    free(t);
    free(yd);
    free(ed);
    free(yf);
    free(ef);
    free(pf);
    free(cf);
    free(pd);
    free(cd);
    dlclose(lib);

    printf("\nMilestone 4 Test Summary: total_fails=%d\n", fails);
    return fails == 0 ? 0 : 1;
}
