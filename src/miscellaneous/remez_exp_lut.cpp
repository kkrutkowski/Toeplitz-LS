/*
 * remez_exp_lut.cpp - coefficient generator for the library's vector exp().
 *
 * The library cannot call the platform's vectorised exponential: GCC lowers a
 * lane loop of expf()/exp() to libmvec (_ZGVdN4v_exp), which glibc provides
 * but musl, macOS libSystem and the MSVC CRT do not, so the same sources would
 * link differently per platform.  Instead the vector kernel reduces the
 * argument and evaluates this Remez polynomial, which makes the vector path
 * identical everywhere and removes the libmvec dependency entirely.
 *
 * Pipeline (IRLS progression followed by Remez exchange):
 *
 *   1. L2 least-squares regression on Chebyshev nodes at target degree;
 *   2. L^p iteratively reweighted least squares on uniform nodes for
 *      p = 4 -> 8 -> 12 -> 16 with Kahng damping alpha = 1 / (p - 1);
 *   3. true Remez exchange iterations at the target degree, seeded with the
 *      alternation points of the L16 IRLS fit.
 * The relative error |p(r) exp(-r) - 1| is minimised on r in [-ln2/2, ln2/2],
 * which is the exact range left by the x = k ln2 + r reduction the kernel
 * performs.  All arithmetic is boost::multiprecision::float128 (113-bit
 * mantissa), so the coefficients are correct far beyond the target precision.
 *
 * Usage:
 *   g++ -O2 remez_exp_lut.cpp -o remez_exp -I/usr/include/eigen3 -lquadmath
 *   ./remez_exp --emit ../../include/nanofft_exp_coeffs.h
 *   ./remez_exp --verify ../../include/nanofft_exp_coeffs.h
 */

#include <boost/multiprecision/float128.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using float128 = boost::multiprecision::float128;
using Vec = std::vector<float128>;
using Mat = std::vector<Vec>; /* row-major */

/* Dense Gauss-Jordan elimination with partial pivoting.  The systems here are
 * at most ~20x20 and the arithmetic is 113-bit, so a plain elimination is both
 * accurate enough and one dependency lighter than a general decomposition
 * library (Eigen's decompositions assume numeric_limits support that
 * boost's float128 backend does not fully provide). */
static Vec solve_dense(Mat a, Vec b) {
    const int n = (int)b.size();
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (abs(a[r][col]) > abs(a[piv][col])) piv = r;
        if (piv != col) {
            std::swap(a[col], a[piv]);
            std::swap(b[col], b[piv]);
        }
        const float128 d = a[col][col];
        if (d == 0) continue;
        for (int c = col; c < n; ++c) a[col][c] /= d;
        b[col] /= d;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const float128 f = a[r][col];
            if (f == 0) continue;
            for (int c = col; c < n; ++c) a[r][c] -= f * a[col][c];
            b[r] -= f * b[col];
        }
    }
    return b;
}

static const float128 LN2 = log(float128(2));

/* -------------------------------------------------------------------------
 * Polynomial helpers (Horner, coefficient order a[0] + a[1] r + ...)
 * ------------------------------------------------------------------------- */
static float128 poly_eval(const std::vector<float128> &a, float128 r) {
    float128 p = a.back();
    for (int i = (int)a.size() - 2; i >= 0; --i) p = p * r + a[i];
    return p;
}

/* Relative error of the fit at r. */
static float128 rel_error(const std::vector<float128> &a, float128 r) { return poly_eval(a, r) / exp(r) - 1; }

/* Chebyshev nodes of the first kind mapped to [-half, half]. */
static std::vector<float128> chebyshev_nodes(int count, float128 half) {
    std::vector<float128> nodes(count);
    for (int k = 0; k < count; ++k) {
        float128 z = cos(float128(3.141592653589793238462643383279502884L) * (float128(2 * k + 1) / float128(2 * count)));
        nodes[k] = half * z;
    }
    return nodes;
}

/* Weighted least-squares fit through the given samples. */
static std::vector<float128> weighted_lsq(const std::vector<float128> &xs, const std::vector<float128> &ys, const std::vector<float128> &ws, int degree) {
    /* Normal equations (A^T W A) a = A^T W y; 113-bit arithmetic absorbs the
     * squaring of the condition number for these low degrees. */
    const int m = degree + 1;
    Mat ata(m, Vec(m, 0));
    Vec atb(m, 0);
    for (size_t i = 0; i < xs.size(); ++i) {
        Vec pw(m);
        float128 xp = 1;
        for (int j = 0; j < m; ++j) {
            pw[j] = xp;
            xp *= xs[i];
        }
        for (int r = 0; r < m; ++r) {
            atb[r] += pw[r] * ws[i] * ys[i];
            for (int c = 0; c < m; ++c) ata[r][c] += pw[r] * ws[i] * pw[c];
        }
    }
    return solve_dense(ata, atb);
}

/* L^p IRLS on uniform nodes with Kahng damping alpha = 1 / (p - 1). */
static std::vector<float128> irls_lp_stage(const std::vector<float128> &xs, const std::vector<float128> &ys, const std::vector<float128> &base_weights,
                                           int degree, int p, int iterations, std::vector<float128> coeffs) {
    const float128 alpha = float128(1) / float128(p - 1);
    const float128 exp_p = float128(p - 2);
    std::vector<float128> weights(xs.size());
    for (int it = 0; it < iterations; ++it) {
        float128 worst = 0;
        std::vector<float128> errs(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) {
            errs[i] = abs(poly_eval(coeffs, xs[i]) / ys[i] - 1);
            if (errs[i] > worst) worst = errs[i];
        }
        if (worst == 0) break;
        for (size_t i = 0; i < xs.size(); ++i) {
            float128 ratio = std::max(errs[i] / worst, float128(1e-3));
            weights[i] = base_weights[i] * pow(ratio, exp_p);
        }
        std::vector<float128> a_wls = weighted_lsq(xs, ys, weights, degree);
        float128 step_inf = 0, c_inf = 0;
        for (size_t j = 0; j < coeffs.size(); ++j) {
            float128 nxt = (float128(1) - alpha) * coeffs[j] + alpha * a_wls[j];
            step_inf = std::max(step_inf, abs(nxt - coeffs[j]));
            c_inf = std::max(c_inf, abs(coeffs[j]));
            coeffs[j] = nxt;
        }
        if (c_inf > 0 && step_inf <= float128(1e-20) * c_inf) break;
    }
    return coeffs;
}

/* One Remez linear system: p(x_i) + (-1)^i E / w_i = f(x_i). */
static void remez_step(const std::vector<float128> &xs, int degree, std::vector<float128> &coeffs, float128 &err) {
    const int n = degree + 2;
    Mat a(n, Vec(n, 0));
    Vec b(n, 0);
    for (int i = 0; i < n; ++i) {
        float128 xp = 1;
        for (int j = 0; j <= degree; ++j) {
            a[i][j] = xp;
            xp *= xs[i];
        }
        float128 w = exp(xs[i]); /* weight 1/f in the relative-error form */
        a[i][n - 1] = (i % 2 == 0 ? float128(1) : float128(-1)) / w;
        b[i] = w; /* f(x_i) */
    }
    Vec sol = solve_dense(a, b);
    coeffs.assign(sol.begin(), sol.begin() + degree + 1);
    err = sol[n - 1];
}

/* Sample |relative error| and return the alternating extrema. */
static std::vector<float128> find_extrema(const std::vector<float128> &coeffs, float128 half, int samples) {
    std::vector<float128> xs(samples);
    std::vector<float128> es(samples);
    for (int i = 0; i < samples; ++i) {
        xs[i] = -half + (float128(2) * half * i) / float128(samples - 1);
        es[i] = rel_error(coeffs, xs[i]);
    }
    std::vector<float128> extrema;
    for (int i = 0; i < samples; ++i) {
        const bool left_ok = (i == 0) || (abs(es[i]) >= abs(es[i - 1]));
        const bool right_ok = (i == samples - 1) || (abs(es[i]) >= abs(es[i + 1]));
        const bool sign_ok = (i == 0) || (i == samples - 1) || ((es[i] > 0) == (es[i - 1] > 0)) || abs(es[i]) >= abs(es[i - 1]);
        if (left_ok && right_ok && sign_ok) extrema.push_back(xs[i]);
    }
    return extrema;
}

/* True Remez exchange, seeded with the given reference points. */
static std::vector<float128> remez(const std::vector<float128> &seed_points, int degree, float128 half, int iterations, float128 &final_err) {
    std::vector<float128> xs = seed_points;
    std::vector<float128> coeffs;
    float128 err = 0;
    int stable = 0;
    for (int it = 0; it < iterations; ++it) {
        if ((int)xs.size() != degree + 2) break;
        remez_step(xs, degree, coeffs, err);
        std::vector<float128> extrema = find_extrema(coeffs, half, 4001);
        if ((int)extrema.size() < degree + 2) break;
        /* keep the degree+2 largest-|error| extrema, in ascending x order */
        std::vector<std::pair<float128, float128>> scored;
        for (auto &x : extrema) scored.emplace_back(abs(rel_error(coeffs, x)), x);
        std::sort(scored.begin(), scored.end(), [](const auto &p, const auto &q) { return p.first > q.first; });
        scored.resize(degree + 2);
        std::sort(scored.begin(), scored.end(), [](const auto &p, const auto &q) { return p.second < q.second; });
        std::vector<float128> next;
        for (auto &s : scored) next.push_back(s.second);
        if (next == xs) {
            if (++stable > 1) break;
        } else {
            stable = 0;
        }
        xs = next;
    }
    final_err = err;
    return coeffs;
}

/* -------------------------------------------------------------------------
 * Accuracy report: max relative error and max ulp error.
 * ------------------------------------------------------------------------- */
struct accuracy {
    float128 max_rel;
    double max_ulp;
    float128 at;
};

/* mantissa bits of the target precision: 24 for float, 53 for double */
static accuracy measure(const std::vector<float128> &coeffs, float128 lo, float128 hi, int samples, int mantissa_bits = 112) {
    accuracy acc{0, 0, 0};
    for (int i = 0; i <= samples; ++i) {
        float128 r = lo + (hi - lo) * float128(i) / float128(samples);
        float128 p = poly_eval(coeffs, r);
        float128 t = exp(r);
        float128 rel = abs(p / t - 1);
        if (rel > acc.max_rel) {
            acc.max_rel = rel;
            acc.at = r;
        }
        /* One ulp is at most t * 2^-mantissa_bits. */
        float128 ulp = t * ldexp(float128(1), -mantissa_bits);
        double u = (double)(rel * t / ulp);
        if (u > acc.max_ulp) acc.max_ulp = u;
    }
    return acc;
}

/* -------------------------------------------------------------------------
 * Coefficient emission
 * ------------------------------------------------------------------------- */
static std::string hex_double(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%a", v);
    return buf;
}

static void emit_header(const char *path, const std::vector<float128> &c_half, const std::vector<float128> &c_float,
                        const std::vector<float128> &c_double) {
    FILE *f = std::fopen(path, "w");
    if (!f) {
        std::perror("fopen");
        std::exit(1);
    }
    std::fprintf(f,
                 "/* Generated by src/miscellaneous/remez_exp_lut.cpp -- do not edit.\n"
                 " *\n"
                 " * Remez minimax coefficients of exp(r) on r in [-ln2/2, ln2/2], in the\n"
                 " * relative-error sense, together with the two-term ln2 split the\n"
                 " * argument reduction needs.  Regenerate with\n"
                 " *   make -C src/miscellaneous exp-coeffs\n"
                 " */\n"
                 "#ifndef NANOFFT_EXP_COEFFS_H\n"
                 "#define NANOFFT_EXP_COEFFS_H\n\n");

    std::fprintf(f, "/* half (IEEE binary16): degree %d */\n", (int)c_half.size() - 1);
    std::fprintf(f, "#define NANOFFT_EXPH_DEGREE %d\n", (int)c_half.size() - 1);
    std::fprintf(f, "static const float nanofft_exph_coeffs[NANOFFT_EXPH_DEGREE + 1] = {");
    for (size_t i = 0; i < c_half.size(); ++i) {
        double v = (double)c_half[i];
        std::fprintf(f, "%s%s", i ? ", " : "", hex_double(v).c_str());
        if (i && (i % 3) == 0 && i + 1 < c_half.size()) std::fprintf(f, "\n    ");
    }
    std::fprintf(f, "};\n\n");

    std::fprintf(f, "/* float: degree %d */\n", (int)c_float.size() - 1);
    std::fprintf(f, "#define NANOFFT_EXPF_DEGREE %d\n", (int)c_float.size() - 1);
    std::fprintf(f, "static const float nanofft_expf_coeffs[NANOFFT_EXPF_DEGREE + 1] = {");
    for (size_t i = 0; i < c_float.size(); ++i) {
        double v = (double)c_float[i];
        std::fprintf(f, "%s%s", i ? ", " : "", hex_double(v).c_str());
        if (i && (i % 3) == 0 && i + 1 < c_float.size()) std::fprintf(f, "\n    ");
    }
    std::fprintf(f, "};\n\n");

    std::fprintf(f, "/* double: degree %d */\n", (int)c_double.size() - 1);
    std::fprintf(f, "#define NANOFFT_EXP_DEGREE %d\n", (int)c_double.size() - 1);
    std::fprintf(f, "static const double nanofft_exp_coeffs[NANOFFT_EXP_DEGREE + 1] = {");
    for (size_t i = 0; i < c_double.size(); ++i) {
        double v = (double)c_double[i];
        std::fprintf(f, "%s%s", i ? ", " : "", hex_double(v).c_str());
        if (i && (i % 3) == 0 && i + 1 < c_double.size()) std::fprintf(f, "\n    ");
    }
    std::fprintf(f, "};\n\n");

    /* ln2 split: the high part has enough low mantissa bits zero so that
     * k * hi is exact for every k the reduction can produce. */
    const double ln2_hi_d = 0x1.62e42fee00000p-1; /* 21 low mantissa bits zero */
    const double ln2_lo_d = (double)(LN2 - (float128)ln2_hi_d);
    const float ln2_hi_f = 0x1.62e4p-1f; /* 10 low mantissa bits zero */
    const float ln2_lo_f = (float)(LN2 - (float128)ln2_hi_f);
    const double ln2_hi_h = 0x1.64p-1; /* 5 low FP16 mantissa bits zero; exact for |k| <= 24 */
    const double ln2_lo_h = (double)(LN2 - (float128)ln2_hi_h);

    std::fprintf(f,
                 "/* ln2 = hi + lo, hi chosen so that k * hi is exact in the reduction\n"
                 " * for every |k| the exponential can produce. */\n"
                 "#define NANOFFT_EXP_LN2_HI_H %s\n"
                 "#define NANOFFT_EXP_LN2_LO_H %s\n"
                 "#define NANOFFT_EXP_INV_LN2_H %s\n"
                 "#define NANOFFT_EXP_LN2_HI_F %s\n"
                 "#define NANOFFT_EXP_LN2_LO_F %s\n"
                 "#define NANOFFT_EXP_INV_LN2_F %s\n"
                 "#define NANOFFT_EXP_LN2_HI %s\n"
                 "#define NANOFFT_EXP_LN2_LO %s\n"
                 "#define NANOFFT_EXP_INV_LN2 %s\n",
                 hex_double(ln2_hi_h).c_str(), hex_double(ln2_lo_h).c_str(), hex_double((double)(1 / LN2)).c_str(),
                 hex_double((double)ln2_hi_f).c_str(), hex_double((double)ln2_lo_f).c_str(), hex_double((double)(1 / LN2)).c_str(),
                 hex_double(ln2_hi_d).c_str(), hex_double(ln2_lo_d).c_str(), hex_double((double)(1 / LN2)).c_str());

    std::fprintf(f, "\n#endif /* NANOFFT_EXP_COEFFS_H */\n");
    std::fclose(f);
    std::printf("[emit] wrote %s\n", path);
}

/* Exact IEEE 754 binary16 round-to-nearest-even helper for FP16 kernel simulation. */
static uint16_t f32_to_f16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t exp = (u >> 23) & 0xffu;
    uint32_t frac = u & 0x7fffffu;
    if (exp == 0xffu) {
        return (uint16_t)(sign | 0x7c00u | (frac ? (0x0200u | (frac >> 13)) : 0u));
    }
    int e = (int)exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        uint32_t m = frac | 0x800000u;
        int shift = 14 - e;
        uint32_t round_bit = 1u << (shift - 1);
        uint32_t sticky = m & (round_bit - 1u);
        uint32_t out = m >> shift;
        if ((m & round_bit) && (sticky || (out & 1u))) out++;
        return (uint16_t)(sign | out);
    }
    uint32_t round_bit = 0x1000u;
    uint32_t sticky = frac & 0x0fffu;
    uint32_t out = frac >> 13;
    if ((frac & round_bit) && (sticky || (out & 1u))) {
        out++;
        if (out == 0x0400u) {
            out = 0;
            e++;
            if (e >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | out);
}

static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t frac = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            int e = -14;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                e--;
            }
            frac &= 0x03ffu;
            out = sign | ((uint32_t)(e + 127) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

static float f16_round(float x) { return f16_bits_to_f32(f32_to_f16_bits(x)); }

/* Simulated kernel: reduction + polynomial + scaling in the target precision,
 * compared against float128 exp.  This is the number that matters. */
static void report_kernel(const std::vector<float128> &ch, const std::vector<float128> &cf, const std::vector<float128> &cd) {
    const float128 half = LN2 / 2;
    accuracy ah = measure(ch, -half, half, 20000, 11);
    accuracy af = measure(cf, -half, half, 20000, 24);
    accuracy ad = measure(cd, -half, half, 20000, 53);
    std::printf("[poly] half   deg %2d  max_rel %.3e  max_ulp %.2f\n", (int)ch.size() - 1, (double)ah.max_rel, ah.max_ulp);
    std::printf("[poly] float  deg %2d  max_rel %.3e  max_ulp %.2f\n", (int)cf.size() - 1, (double)af.max_rel, af.max_ulp);
    std::printf("[poly] double deg %2d  max_rel %.3e  max_ulp %.2f\n", (int)cd.size() - 1, (double)ad.max_rel, ad.max_ulp);

    /* Kernel accuracy in ulps of the target type, over the whole finite range
     * (denormal results included: their ulp is the subnormal step, so they are
     * judged on the same scale).  Saturated results (0 / inf) are correct by
     * construction and skipped. */
    double worst_f_ulp = 0, at_f = 0;
    for (int i = 0; i <= 200000; ++i) {
        double x = -103.0 + 191.0 * i / 200000.0; /* float: exp under/overflows here */
        float xf = (float)x;
        float kf = std::nearbyint(xf * (float)(1.0 / 0.693147180559945309417232121458176568));
        float r = xf - kf * (float)6.93145751953125e-01f;
        r = r - kf * (float)(0.693147180559945309417232121458176568 - 6.93145751953125e-01f);
        float pf = (float)cf.back();
        for (int j = (int)cf.size() - 2; j >= 0; --j) pf = pf * r + (float)cf[j];
        int k1 = (int)std::max(-126.0f, std::min(126.0f, kf));
        int k2 = (int)std::max(-126.0f, std::min(126.0f, kf - (float)k1));
        float s1, s2;
        int b1 = (k1 + 127) << 23;
        int b2 = (k2 + 127) << 23;
        std::memcpy(&s1, &b1, 4);
        std::memcpy(&s2, &b2, 4);
        float got = pf * s1 * s2;
        float want = (float)exp((float128)xf);
        if (want == 0.0f || std::isinf(want)) continue;
        float step = std::nextafterf(want, INFINITY) - want;
        double ulps = std::fabs((double)(got - want)) / (double)step;
        if (ulps > worst_f_ulp) {
            worst_f_ulp = ulps;
            at_f = x;
        }
    }

    double worst_d_ulp = 0, at_d = 0;
    for (int i = 0; i <= 200000; ++i) {
        double x = -708.0 + 1416.0 * i / 200000.0;
        double kd = std::nearbyint(x * (1.0 / 0.693147180559945309417232121458176568));
        double r = x - kd * 6.93147180369123816490e-01;
        r = r - kd * 1.90821492927058770002e-10;
        double pd = (double)cd.back();
        for (int j = (int)cd.size() - 2; j >= 0; --j) pd = pd * r + (double)cd[j];
        long long k1 = (long long)std::max(-1022.0, std::min(1023.0, kd));
        long long k2 = (long long)std::max(-1022.0, std::min(1023.0, kd - (double)k1));
        double s1, s2;
        unsigned long long b1 = (unsigned long long)(k1 + 1023) << 52;
        unsigned long long b2 = (unsigned long long)(k2 + 1023) << 52;
        std::memcpy(&s1, &b1, 8);
        std::memcpy(&s2, &b2, 8);
        double got = pd * s1 * s2;
        double want = (double)exp((float128)x);
        if (want == 0.0 || std::isinf(want)) continue;
        double step = std::nextafter(want, INFINITY) - want;
        double ulps = std::fabs(got - want) / step;
        if (ulps > worst_d_ulp) {
            worst_d_ulp = ulps;
            at_d = x;
        }
    }

    /* Exhaustive IEEE binary16 kernel simulation over [-16, 11] */
    double worst_h_ulp = 0, at_h = 0;
    const float ln2_hi_h = f16_round((float)0x1.64p-1);
    const float ln2_lo_h = f16_round((float)(LN2 - (float128)0x1.64p-1));
    const float inv_ln2_h = f16_round((float)(1 / LN2));
    for (uint32_t bits = 0; bits < 65536u; ++bits) {
        if ((bits & 0x7c00u) == 0x7c00u) continue;
        float xh = f16_bits_to_f32((uint16_t)bits);
        if (xh < -16.0f || xh > 11.0f) continue;
        float kh = f16_round(std::nearbyint(f16_round(xh * inv_ln2_h)));
        float r = f16_round(xh - f16_round(kh * ln2_hi_h));
        r = f16_round(r - f16_round(kh * ln2_lo_h));
        float ph = f16_round((float)ch.back());
        for (int j = (int)ch.size() - 2; j >= 0; --j) {
            ph = f16_round(f16_round(ph * r) + f16_round((float)ch[j]));
        }
        int k1 = (int)std::max(-14.0f, std::min(15.0f, kh));
        int k2 = (int)std::max(-14.0f, std::min(15.0f, kh - (float)k1));
        float s1 = f16_bits_to_f32((uint16_t)((k1 + 15) << 10));
        float s2 = f16_bits_to_f32((uint16_t)((k2 + 15) << 10));
        uint16_t got_bits = f32_to_f16_bits(f16_round(f16_round(ph * s1) * s2));
        uint16_t want_bits = f32_to_f16_bits((float)exp((float128)xh));
        if ((want_bits & 0x7fffu) == 0u || (want_bits & 0x7c00u) == 0x7c00u) continue;
        double ulps = std::fabs((double)(int)got_bits - (double)(int)want_bits);
        if (ulps > worst_h_ulp) {
            worst_h_ulp = ulps;
            at_h = (double)xh;
        }
    }

    std::printf("[kernel] half   max %5.2f ulp at x=%.6g\n", worst_h_ulp, at_h);
    std::printf("[kernel] float  max %5.2f ulp at x=%.6g\n", worst_f_ulp, at_f);
    std::printf("[kernel] double max %5.2f ulp at x=%.6g\n", worst_d_ulp, at_d);
}

static std::vector<float128> fit_chain(int degree, int samples_per_unit) {
    const float128 half = LN2 / 2;
    /* Stage 1: L2 least-squares regression on Chebyshev nodes at target degree. */
    const int n_cheb = std::max(4 * degree + 8, 64);
    std::vector<float128> cheb = chebyshev_nodes(n_cheb, half);
    std::vector<float128> cy(n_cheb), cw(n_cheb);
    for (int i = 0; i < n_cheb; ++i) {
        cy[i] = exp(cheb[i]);
        cw[i] = 1 / (cy[i] * cy[i]);
    }
    std::vector<float128> seed = weighted_lsq(cheb, cy, cw, degree);
    std::printf("[init] deg %2d L2  (Cheb): max_rel %.3e\n", degree, (double)measure(seed, -half, half, 4000).max_rel);

    /* Stages 2-5: L^p IRLS on uniform nodes for p = 4 -> 8 -> 12 -> 16. */
    const int samples = std::max(200, samples_per_unit * (degree + 1));
    std::vector<float128> xs(samples), ys(samples), ws(samples);
    for (int i = 0; i < samples; ++i) {
        xs[i] = -half + float128(2) * half * i / float128(samples - 1);
        ys[i] = exp(xs[i]);
        ws[i] = 1 / (ys[i] * ys[i]);
    }
    const int lp_stages[4] = {4, 8, 12, 16};
    for (int s = 0; s < 4; ++s) {
        int p = lp_stages[s];
        seed = irls_lp_stage(xs, ys, ws, degree, p, 12, seed);
        std::printf("[init] deg %2d L%-2d (Unif): max_rel %.3e\n", degree, p, (double)measure(seed, -half, half, 4000).max_rel);
    }

    /* Stage 6: true Remez at the target degree, seeded by the L16 fit's extrema. */
    std::vector<float128> extrema = find_extrema(seed, half, 4001);
    std::vector<float128> ref;
    if ((int)extrema.size() >= degree + 2) {
        for (int i = 0; i < degree + 2; ++i) {
            size_t idx = (size_t)((double)i * (extrema.size() - 1) / (double)(degree + 1));
            ref.push_back(extrema[idx]);
        }
    } else {
        for (int i = 0; i < degree + 2; ++i) ref.push_back(-half + float128(2) * half * i / float128(degree + 1));
    }
    float128 err = 0;
    std::vector<float128> coeffs = remez(ref, degree, half, 60, err);
    accuracy acc = measure(coeffs, -half, half, 20000);
    std::printf("[remez] degree %2d: minimax |E| %.3e, achieved max_rel %.3e, max_ulp %.2f\n", degree, (double)abs(err), (double)acc.max_rel, acc.max_ulp);
    return coeffs;
}

int main(int argc, char **argv) {
    const float128 half = LN2 / 2;
    std::printf("range [-ln2/2, ln2/2] = [%.18g, %.18g]\n", (double)-half, (double)half);

    std::vector<float128> ch = fit_chain(3, 200);
    std::vector<float128> cf = fit_chain(6, 200);
    std::vector<float128> cd = fit_chain(13, 200);

    report_kernel(ch, cf, cd);

    if (argc > 1 && std::strcmp(argv[1], "--emit") == 0 && argc > 2) emit_header(argv[2], ch, cf, cd);
    return 0;
}
