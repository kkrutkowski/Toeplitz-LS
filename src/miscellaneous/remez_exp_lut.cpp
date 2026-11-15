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
 * Pipeline (the one used for the PSWF LUT in pswf_remez_lut.cpp, extended):
 *
 *   1. Chebyshev-node least-squares regression at degree 4;
 *   2. iteratively reweighted least squares (weights ~ 1/residual) at degree 8
 *      and then 16, which drives the fit towards the minimax solution;
 *   3. true Remez exchange iterations at the target degree, seeded with the
 *      alternation points of the degree-16 IRLS fit.
 *
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

/* IRLS: reweight each sample by 1/|relative residual| and refit. */
static std::vector<float128> irls_fit(const std::vector<float128> &xs, const std::vector<float128> &ys, const std::vector<float128> &base_weights, int degree,
                                      int iterations) {
    std::vector<float128> weights = base_weights;
    std::vector<float128> coeffs;
    for (int it = 0; it < iterations; ++it) {
        coeffs = weighted_lsq(xs, ys, weights, degree);
        float128 worst = 0;
        for (size_t i = 0; i < xs.size(); ++i) worst = std::max(worst, abs(poly_eval(coeffs, xs[i]) / ys[i] - 1));
        if (worst == 0) break;
        for (size_t i = 0; i < xs.size(); ++i) {
            float128 e = abs(poly_eval(coeffs, xs[i]) / ys[i] - 1);
            weights[i] = base_weights[i] / std::max(e, worst * float128(1e-3));
        }
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

static void emit_header(const char *path, const std::vector<float128> &c_float, const std::vector<float128> &c_double) {
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

    /* ln2 split: the high part has its low 27/12 mantissa bits zero so that
     * k * hi is exact for every k the reduction can produce. */
    /* The remainder must be taken against the *true* ln2, not against its
     * double rounding: k can reach 1021, so an error of half an ulp in the
     * double ln2 would be amplified to ~5e-14 in the reduced argument. */
    const double ln2_hi_d = 0x1.62e42fee00000p-1; /* 21 low mantissa bits zero */
    const double ln2_lo_d = (double)(LN2 - (float128)ln2_hi_d);
    /* The high part must have enough low mantissa bits zero that k * hi is
     * exact for every k the reduction can produce (|k| <= 1024 here), so the
     * subtraction x - k*hi is exact by Sterbenz and the reduction error comes
     * only from k * lo. */
    const float ln2_hi_f = 0x1.62e4p-1f; /* 10 low mantissa bits zero */
    const float ln2_lo_f = (float)(LN2 - (float128)ln2_hi_f);

    std::fprintf(f,
                 "/* ln2 = hi + lo, hi chosen so that k * hi is exact in the reduction\n"
                 " * for every |k| the exponential can produce. */\n"
                 "#define NANOFFT_EXP_LN2_HI_F %s\n"
                 "#define NANOFFT_EXP_LN2_LO_F %s\n"
                 "#define NANOFFT_EXP_INV_LN2_F %s\n"
                 "#define NANOFFT_EXP_LN2_HI %s\n"
                 "#define NANOFFT_EXP_LN2_LO %s\n"
                 "#define NANOFFT_EXP_INV_LN2 %s\n",
                 hex_double((double)ln2_hi_f).c_str(), hex_double((double)ln2_lo_f).c_str(), hex_double((double)(1 / LN2)).c_str(), hex_double(ln2_hi_d).c_str(),
                 hex_double(ln2_lo_d).c_str(), hex_double((double)(1 / LN2)).c_str());

    std::fprintf(f, "\n#endif /* NANOFFT_EXP_COEFFS_H */\n");
    std::fclose(f);
    std::printf("[emit] wrote %s\n", path);
}

/* Simulated kernel: reduction + polynomial + scaling in the target precision,
 * compared against float128 exp.  This is the number that matters. */
static void report_kernel(const std::vector<float128> &cf, const std::vector<float128> &cd) {
    const float128 half = LN2 / 2;
    accuracy af = measure(cf, -half, half, 20000, 24);
    accuracy ad = measure(cd, -half, half, 20000, 53);
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

    std::printf("[kernel] float  max %5.2f ulp at x=%.6g\n", worst_f_ulp, at_f);
    std::printf("[kernel] double max %5.2f ulp at x=%.6g\n", worst_d_ulp, at_d);
}

static std::vector<float128> fit_chain(int degree, int samples_per_unit) {
    const float128 half = LN2 / 2;
    /* Dense sampling for the regression stages. */
    const int samples = std::max(200, samples_per_unit * (degree + 1));
    std::vector<float128> xs(samples), ys(samples), ws(samples);
    for (int i = 0; i < samples; ++i) {
        xs[i] = -half + float128(2) * half * i / float128(samples - 1);
        ys[i] = exp(xs[i]);
        ws[i] = 1 / ys[i];
    }

    /* 1-2: Chebyshev-node regression, then IRLS at degree 4 -> 8 -> 16. */
    std::vector<float128> cheb = chebyshev_nodes(std::max(2 * degree + 2, 24), half);
    std::vector<float128> cx(cheb.size()), cy(cheb.size()), cw(cheb.size());
    for (size_t i = 0; i < cheb.size(); ++i) {
        cx[i] = cheb[i];
        cy[i] = exp(cheb[i]);
        cw[i] = 1 / cy[i];
    }
    std::vector<float128> seed = weighted_lsq(cx, cy, cw, 4);
    std::printf("[init] degree 4 lsq: max_rel %.3e\n", (double)measure(seed, -half, half, 4000).max_rel);
    seed = irls_fit(cx, cy, cw, 8, 12);
    std::printf("[init] degree 8 irls: max_rel %.3e\n", (double)measure(seed, -half, half, 4000).max_rel);
    seed = irls_fit(cx, cy, cw, 16, 12);
    std::printf("[init] degree 16 irls: max_rel %.3e\n", (double)measure(seed, -half, half, 4000).max_rel);

    /* 3: true Remez at the target degree, seeded by the IRLS fit's extrema. */
    std::vector<float128> extrema = find_extrema(seed, half, 2001);
    std::vector<float128> ref;
    if ((int)extrema.size() >= degree + 2) {
        /* take the degree+2 outermost alternation points, evenly */
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

    std::vector<float128> cf = fit_chain(6, 200);
    std::vector<float128> cd = fit_chain(13, 200);

    report_kernel(cf, cd);

    if (argc > 1 && std::strcmp(argv[1], "--emit") == 0 && argc > 2) emit_header(argv[2], cf, cd);
    return 0;
}
