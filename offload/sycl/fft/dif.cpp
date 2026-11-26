// Quadruple-Precision SYCL DIF (Sande-Tukey) Radix-2 FFT Implementation.
// Compiled multiple times with -D flags into isolated HCF object modules:
//   Precision (defaults to SYCL_FFT_DF when no half/float/double is selected):
//     -DSYCL_FFT_HALF | -DSYCL_FFT_FLOAT | -DSYCL_FFT_DOUBLE | -DSYCL_FFT_DF
//   Size split (optional; omit to compile both small & large kernels + DifFftPlan):
//     -DSYCL_FFT_INST_SMALL_ONLY | -DSYCL_FFT_INST_LARGE_ONLY
#if !defined(SYCL_FFT_HALF) && !defined(SYCL_FFT_FLOAT) && \
    !defined(SYCL_FFT_DOUBLE) && !defined(SYCL_FFT_DF)
#  define SYCL_FFT_DF 1
#endif

#include "fft/dif.hpp"

#include <algorithm>
#include <chrono>
#include <new>
#include <stdexcept>

namespace sycl_fft {

// ============================================================================
// Exact Compile-Time W_32^k & W_256^k FFT Twiddle Tables (FFT-internal)
// ============================================================================
constexpr double kCos32[16] = {
     1.00000000000000000e+00,  9.80785280403230431e-01,  9.23879532511286738e-01,  8.31469612302545236e-01,
     7.07106781186547573e-01,  5.55570233019602178e-01,  3.82683432365089782e-01,  1.95090322016128276e-01,
     0.00000000000000000e+00, -1.95090322016128276e-01, -3.82683432365089782e-01, -5.55570233019602178e-01,
    -7.07106781186547573e-01, -8.31469612302545236e-01, -9.23879532511286738e-01, -9.80785280403230431e-01
};

constexpr double kSin32Neg[16] = {
     0.00000000000000000e+00, -1.95090322016128276e-01, -3.82683432365089782e-01, -5.55570233019602178e-01,
    -7.07106781186547573e-01, -8.31469612302545236e-01, -9.23879532511286738e-01, -9.80785280403230431e-01,
    -1.00000000000000000e+00, -9.80785280403230431e-01, -9.23879532511286738e-01, -8.31469612302545236e-01,
    -7.07106781186547573e-01, -5.55570233019602178e-01, -3.82683432365089782e-01, -1.95090322016128276e-01
};

constexpr float kCos32f[16] = {
    static_cast<float>(kCos32[0]),  static_cast<float>(kCos32[1]),  static_cast<float>(kCos32[2]),  static_cast<float>(kCos32[3]),
    static_cast<float>(kCos32[4]),  static_cast<float>(kCos32[5]),  static_cast<float>(kCos32[6]),  static_cast<float>(kCos32[7]),
    static_cast<float>(kCos32[8]),  static_cast<float>(kCos32[9]),  static_cast<float>(kCos32[10]), static_cast<float>(kCos32[11]),
    static_cast<float>(kCos32[12]), static_cast<float>(kCos32[13]), static_cast<float>(kCos32[14]), static_cast<float>(kCos32[15])
};

constexpr float kSin32Negf[16] = {
    static_cast<float>(kSin32Neg[0]),  static_cast<float>(kSin32Neg[1]),  static_cast<float>(kSin32Neg[2]),  static_cast<float>(kSin32Neg[3]),
    static_cast<float>(kSin32Neg[4]),  static_cast<float>(kSin32Neg[5]),  static_cast<float>(kSin32Neg[6]),  static_cast<float>(kSin32Neg[7]),
    static_cast<float>(kSin32Neg[8]),  static_cast<float>(kSin32Neg[9]),  static_cast<float>(kSin32Neg[10]), static_cast<float>(kSin32Neg[11]),
    static_cast<float>(kSin32Neg[12]), static_cast<float>(kSin32Neg[13]), static_cast<float>(kSin32Neg[14]), static_cast<float>(kSin32Neg[15])
};

constexpr df_t kCos32df[16] = {
    df_from_double(kCos32[0]),  df_from_double(kCos32[1]),  df_from_double(kCos32[2]),  df_from_double(kCos32[3]),
    df_from_double(kCos32[4]),  df_from_double(kCos32[5]),  df_from_double(kCos32[6]),  df_from_double(kCos32[7]),
    df_from_double(kCos32[8]),  df_from_double(kCos32[9]),  df_from_double(kCos32[10]), df_from_double(kCos32[11]),
    df_from_double(kCos32[12]), df_from_double(kCos32[13]), df_from_double(kCos32[14]), df_from_double(kCos32[15])
};

constexpr df_t kSin32Negdf[16] = {
    df_from_double(kSin32Neg[0]),  df_from_double(kSin32Neg[1]),  df_from_double(kSin32Neg[2]),  df_from_double(kSin32Neg[3]),
    df_from_double(kSin32Neg[4]),  df_from_double(kSin32Neg[5]),  df_from_double(kSin32Neg[6]),  df_from_double(kSin32Neg[7]),
    df_from_double(kSin32Neg[8]),  df_from_double(kSin32Neg[9]),  df_from_double(kSin32Neg[10]), df_from_double(kSin32Neg[11]),
    df_from_double(kSin32Neg[12]), df_from_double(kSin32Neg[13]), df_from_double(kSin32Neg[14]), df_from_double(kSin32Neg[15])
};

template <int Sign>
constexpr float twiddle32_ref(int idx32) {
    return kCos32f[idx32];
}

template <int Sign>
constexpr float twiddle32_imf(int idx32) {
    return (Sign < 0) ? kSin32Negf[idx32] : -kSin32Negf[idx32];
}

constexpr f16x2_t kCos32h[16] = {
    f16x2_splat_const(kCos32f[0]),  f16x2_splat_const(kCos32f[1]),  f16x2_splat_const(kCos32f[2]),  f16x2_splat_const(kCos32f[3]),
    f16x2_splat_const(kCos32f[4]),  f16x2_splat_const(kCos32f[5]),  f16x2_splat_const(kCos32f[6]),  f16x2_splat_const(kCos32f[7]),
    f16x2_splat_const(kCos32f[8]),  f16x2_splat_const(kCos32f[9]),  f16x2_splat_const(kCos32f[10]), f16x2_splat_const(kCos32f[11]),
    f16x2_splat_const(kCos32f[12]), f16x2_splat_const(kCos32f[13]), f16x2_splat_const(kCos32f[14]), f16x2_splat_const(kCos32f[15])
};

constexpr f16x2_t kSin32Negh[16] = {
    f16x2_splat_const(kSin32Negf[0]),  f16x2_splat_const(kSin32Negf[1]),  f16x2_splat_const(kSin32Negf[2]),  f16x2_splat_const(kSin32Negf[3]),
    f16x2_splat_const(kSin32Negf[4]),  f16x2_splat_const(kSin32Negf[5]),  f16x2_splat_const(kSin32Negf[6]),  f16x2_splat_const(kSin32Negf[7]),
    f16x2_splat_const(kSin32Negf[8]),  f16x2_splat_const(kSin32Negf[9]),  f16x2_splat_const(kSin32Negf[10]), f16x2_splat_const(kSin32Negf[11]),
    f16x2_splat_const(kSin32Negf[12]), f16x2_splat_const(kSin32Negf[13]), f16x2_splat_const(kSin32Negf[14]), f16x2_splat_const(kSin32Negf[15])
};

template <int Sign>
SYCL_FFT_FORCE_INLINE SYCL_FFT_F16X2_CONSTEXPR f16x2_t twiddle32_reh(int idx32) {
    return kCos32h[idx32];
}

template <int Sign>
SYCL_FFT_FORCE_INLINE SYCL_FFT_F16X2_CONSTEXPR f16x2_t twiddle32_imh(int idx32) {
    return (Sign < 0) ? kSin32Negh[idx32] : -kSin32Negh[idx32];
}

template <int Sign>
constexpr double twiddle32_re(int idx32) {
    return kCos32[idx32];
}

template <int Sign>
constexpr double twiddle32_im(int idx32) {
    return (Sign < 0) ? kSin32Neg[idx32] : -kSin32Neg[idx32];
}

template <int Sign>
constexpr df_t twiddle32_redf(int idx32) {
    return kCos32df[idx32];
}

template <int Sign>
constexpr df_t twiddle32_imdf(int idx32) {
    return (Sign < 0) ? kSin32Negdf[idx32]
                      : df_t{-kSin32Negdf[idx32].hi, -kSin32Negdf[idx32].lo};
}

// Exact compile-time W_256^k = exp(-i 2pi k / 256) table (k = 0..15).
// Also serves W_128^k (idx = 2*k, k=0..7) and W_64^k (idx = 4*k, k=0..3) with zero trig ops.
constexpr double kW256Re[16] = {
     1.00000000000000000e+00,  9.99698818696204250e-01,  9.98795456205172405e-01,  9.97290456678690207e-01,
     9.95184726672196929e-01,  9.92479534598709967e-01,  9.89176509964781014e-01,  9.85277642388941222e-01,
     9.80785280403230431e-01,  9.75702130038528570e-01,  9.70031253194543974e-01,  9.63776065795439840e-01,
     9.56940335732208824e-01,  9.49528180593036675e-01,  9.41544065183020806e-01,  9.32992798834738846e-01
};

constexpr double kW256ImNeg[16] = {
     0.00000000000000000e+00, -2.45412285229122881e-02, -4.90676743274180149e-02, -7.35645635996674263e-02,
    -9.80171403295606036e-02, -1.22410675199216196e-01, -1.46730474455361748e-01, -1.70961888760301217e-01,
    -1.95090322016128276e-01, -2.19101240156869798e-01, -2.42980179903263899e-01, -2.66712757474898365e-01,
    -2.90284677254462387e-01, -3.13681740398891462e-01, -3.36889853392220051e-01, -3.59895036534988166e-01
};

constexpr float kW256Ref[16] = {
    static_cast<float>(kW256Re[0]),  static_cast<float>(kW256Re[1]),  static_cast<float>(kW256Re[2]),  static_cast<float>(kW256Re[3]),
    static_cast<float>(kW256Re[4]),  static_cast<float>(kW256Re[5]),  static_cast<float>(kW256Re[6]),  static_cast<float>(kW256Re[7]),
    static_cast<float>(kW256Re[8]),  static_cast<float>(kW256Re[9]),  static_cast<float>(kW256Re[10]), static_cast<float>(kW256Re[11]),
    static_cast<float>(kW256Re[12]), static_cast<float>(kW256Re[13]), static_cast<float>(kW256Re[14]), static_cast<float>(kW256Re[15])
};

constexpr float kW256ImNegf[16] = {
    static_cast<float>(kW256ImNeg[0]),  static_cast<float>(kW256ImNeg[1]),  static_cast<float>(kW256ImNeg[2]),  static_cast<float>(kW256ImNeg[3]),
    static_cast<float>(kW256ImNeg[4]),  static_cast<float>(kW256ImNeg[5]),  static_cast<float>(kW256ImNeg[6]),  static_cast<float>(kW256ImNeg[7]),
    static_cast<float>(kW256ImNeg[8]),  static_cast<float>(kW256ImNeg[9]),  static_cast<float>(kW256ImNeg[10]), static_cast<float>(kW256ImNeg[11]),
    static_cast<float>(kW256ImNeg[12]), static_cast<float>(kW256ImNeg[13]), static_cast<float>(kW256ImNeg[14]), static_cast<float>(kW256ImNeg[15])
};

constexpr df_t kW256Redf[16] = {
    df_from_double(kW256Re[0]),  df_from_double(kW256Re[1]),  df_from_double(kW256Re[2]),  df_from_double(kW256Re[3]),
    df_from_double(kW256Re[4]),  df_from_double(kW256Re[5]),  df_from_double(kW256Re[6]),  df_from_double(kW256Re[7]),
    df_from_double(kW256Re[8]),  df_from_double(kW256Re[9]),  df_from_double(kW256Re[10]), df_from_double(kW256Re[11]),
    df_from_double(kW256Re[12]), df_from_double(kW256Re[13]), df_from_double(kW256Re[14]), df_from_double(kW256Re[15])
};

constexpr df_t kW256ImNegdf[16] = {
    df_from_double(kW256ImNeg[0]),  df_from_double(kW256ImNeg[1]),  df_from_double(kW256ImNeg[2]),  df_from_double(kW256ImNeg[3]),
    df_from_double(kW256ImNeg[4]),  df_from_double(kW256ImNeg[5]),  df_from_double(kW256ImNeg[6]),  df_from_double(kW256ImNeg[7]),
    df_from_double(kW256ImNeg[8]),  df_from_double(kW256ImNeg[9]),  df_from_double(kW256ImNeg[10]), df_from_double(kW256ImNeg[11]),
    df_from_double(kW256ImNeg[12]), df_from_double(kW256ImNeg[13]), df_from_double(kW256ImNeg[14]), df_from_double(kW256ImNeg[15])
};

template <int Sign>
constexpr float twiddle256_ref(int idx) { return kW256Ref[idx]; }
template <int Sign>
constexpr float twiddle256_imf(int idx) { return (Sign < 0) ? kW256ImNegf[idx] : -kW256ImNegf[idx]; }
template <int Sign>
constexpr double twiddle256_re(int idx) { return kW256Re[idx]; }
template <int Sign>
constexpr double twiddle256_im(int idx) { return (Sign < 0) ? kW256ImNeg[idx] : -kW256ImNeg[idx]; }
template <int Sign>
constexpr df_t twiddle256_redf(int idx) { return kW256Redf[idx]; }
template <int Sign>
constexpr df_t twiddle256_imdf(int idx) {
    return (Sign < 0) ? kW256ImNegdf[idx]
                      : df_t{-kW256ImNegdf[idx].hi, -kW256ImNegdf[idx].lo};
}

#include "common/precision.hpp"

// ============================================================================
// Pure Radix-2 DIF stages in registers on E = 2^LogE elements (no inter-thread
// twiddles; exact compile-time W_32 constants with W_2, W_4, W_8 specializations).
// ============================================================================
template <int LogE>
SYCL_FFT_DF_INLINE void SYCL_FFT_SUFFIX(dif_radix2_reg_base)(
    SYCL_FFT_REAL (&r_re)[1 << LogE], SYCL_FFT_REAL (&r_im)[1 << LogE], bool pos_sign) {
    using Real = SYCL_FFT_REAL;
    constexpr int E = 1 << LogE;
    static_assert(LogE <= 5, "W_32 table supports up to LogE == 5 (E == 32)");

    unroll_for<LogE>([&](auto s_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int S = s_ic.value;
        constexpr int H = E >> (S + 1);
        constexpr int Step = H << 1;
        constexpr int Groups = E / Step;

        unroll_for<H>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int M = m_ic.value;
            constexpr int Idx32 = M * (32 / Step);

            unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int A = g_ic.value * Step + M;
                constexpr int B = A + H;
                const Real ar = SYCL_FFT_SCALE_HALF(r_re[A]), ai = SYCL_FFT_SCALE_HALF(r_im[A]);
                const Real br = SYCL_FFT_SCALE_HALF(r_re[B]), bi = SYCL_FFT_SCALE_HALF(r_im[B]);
                const Real dr = SYCL_FFT_SUB(ar, br);
                const Real di = SYCL_FFT_SUB(ai, bi);
                r_re[A] = SYCL_FFT_ADD(ar, br);
                r_im[A] = SYCL_FFT_ADD(ai, bi);

                if constexpr (M == 0) {
                    r_re[B] = dr;
                    r_im[B] = di;
                } else if constexpr (Idx32 == 8) {
                    r_re[B] = SYCL_FFT_COND_NEG(di, pos_sign);
                    r_im[B] = SYCL_FFT_COND_NEG(dr, !pos_sign);
                } else if constexpr (Idx32 == 4) {
                    SYCL_FFT_CMUL_SQRT1_2(false, pos_sign, dr, di, r_re[B], r_im[B]);
                } else if constexpr (Idx32 == 12) {
                    SYCL_FFT_CMUL_SQRT1_2(true, pos_sign, dr, di, r_re[B], r_im[B]);
                } else {
                    const Real cr = SYCL_FFT_TWIDDLE32_RE(-1, Idx32);
                    const Real ci = SYCL_FFT_COND_NEG(SYCL_FFT_TWIDDLE32_IM(-1, Idx32), pos_sign);
                    SYCL_FFT_CMUL(dr, di, cr, ci, r_re[B], r_im[B]);
                }
            });
        });
    });
}

// ============================================================================
// Factored Bit-Reversed Twiddle Application:
//   Multiplies slot P (0..E-1) by W_{E*T}^{t * bitrev_{LogE}(P)} in strictly
//   ascending P order and invokes on_elem(p_ic, out_re, out_im).
//   Eliminates 12.5-37% of complex multiplies vs unfactored DIF stages and
//   evaluates only ONE octant-1 polynomial trig call (t / (E*T) < 1/8 for E >= 8).
// ============================================================================
template <int LogE, int LogT, typename OnElem>
SYCL_FFT_DF_INLINE void SYCL_FFT_SUFFIX(apply_bitrev_twiddles_with)(
    const SYCL_FFT_REAL (&r_re)[1 << LogE],
    const SYCL_FFT_REAL (&r_im)[1 << LogE],
    int t,
    bool pos_sign,
    OnElem&& on_elem) {
    using Real = SYCL_FFT_REAL;
    using AngleT = SYCL_FFT_ANGLE_T;
    constexpr int E = 1 << LogE;
    constexpr int T = 1 << LogT;
    constexpr AngleT inv_m = static_cast<AngleT>(1.0) / static_cast<AngleT>(E * T);

    on_elem(std::integral_constant<int, 0>{}, r_re[0], r_im[0]);

#if SYCL_FFT_IS_HALF
    // Evaluate twiddle recurrences in scalar FP32 (eliminating FP16 squaring phase drift)
    // and splat exact binary16 twiddles right before the 2x FP16 complex multiply.
    auto tw_cmul = [](float ar, float ai, float br, float bi, float& or_, float& oi) SYCL_FFT_LAMBDA_INLINE {
        or_ = sycl::fma(ar, br, -(ai * bi));
        oi  = sycl::fma(ar, bi,   ai * br);
    };
    auto tw_csqr = [](float ar, float ai, float& or_, float& oi) SYCL_FFT_LAMBDA_INLINE {
        or_ = sycl::fma(ar, ar, -(ai * ai));
        oi  = (2.0f * ar) * ai;
    };

    float w1r, w1i;
    if constexpr (LogE >= 4 && LogE + LogT <= 8) {
        constexpr int Scale = 256 / (E * T);
        const int idx = t * Scale;
        w1r = twiddle256_ref<-1>(idx);
        w1i = twiddle256_imf<-1>(idx);
    } else if constexpr (LogE >= 3) {
        sincos2pif_oct1<-1>(static_cast<float>(t) * inv_m, w1r, w1i);
    } else {
        sincos2pif_half<-1>(static_cast<float>(t) * inv_m, w1r, w1i);
    }
    w1i = SYCL_FFT_BIT_CAST(float,
        SYCL_FFT_BIT_CAST(uint32_t, w1i) ^ (static_cast<uint32_t>(pos_sign) << 31));
    float w2r, w2i, w3r, w3i, w4r, w4i;
    tw_csqr(w1r, w1i, w2r, w2i);
    tw_cmul(w2r, w2i, w1r, w1i, w3r, w3i);
    tw_csqr(w2r, w2i, w4r, w4i);

    if constexpr (LogE == 3) {
        const float lo_r[4] = {1.0f, w2r, w1r, w3r};
        const float lo_i[4] = {0.0f, w2i, w1i, w3i};
        unroll_for<7>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 1;
            constexpr int B = P & 1;
            float wr, wi;
            if constexpr (A == 0) {
                wr = w4r; wi = w4i;
            } else if constexpr (B == 0) {
                wr = lo_r[A]; wi = lo_i[A];
            } else {
                tw_cmul(lo_r[A], lo_i[A], w4r, w4i, wr, wi);
            }
            const f16_scalar_t hwr = static_cast<f16_scalar_t>(wr), hwi = static_cast<f16_scalar_t>(wi);
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], SYCL_FFT_F16X2_SPLAT_H(hwr), SYCL_FFT_F16X2_SPLAT_H(hwi), or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 4) {
        float w8r, w8i, w12r, w12i;
        tw_csqr(w4r, w4i, w8r, w8i);
        tw_cmul(w8r, w8i, w4r, w4i, w12r, w12i);
        const float lo_r[4] = {1.0f, w2r, w1r, w3r};
        const float lo_i[4] = {0.0f, w2i, w1i, w3i};
        const float hi_r[4] = {1.0f, w8r, w4r, w12r};
        const float hi_i[4] = {0.0f, w8i, w4i, w12i};
        unroll_for<15>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2;
            constexpr int B = P & 3;
            float wr, wi;
            if constexpr (A == 0) {
                wr = hi_r[B]; wi = hi_i[B];
            } else if constexpr (B == 0) {
                wr = lo_r[A]; wi = lo_i[A];
            } else {
                tw_cmul(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi);
            }
            const f16_scalar_t hwr = static_cast<f16_scalar_t>(wr), hwi = static_cast<f16_scalar_t>(wi);
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], SYCL_FFT_F16X2_SPLAT_H(hwr), SYCL_FFT_F16X2_SPLAT_H(hwi), or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 5) {
        float w5r, w5i, w6r, w6i, w7r, w7i;
        float w8r, w8i, w16r, w16i, w24r, w24i;
        tw_cmul(w4r, w4i, w1r, w1i, w5r, w5i);
        tw_cmul(w4r, w4i, w2r, w2i, w6r, w6i);
        tw_cmul(w4r, w4i, w3r, w3i, w7r, w7i);
        tw_csqr(w4r, w4i, w8r, w8i);
        tw_csqr(w8r, w8i, w16r, w16i);
        tw_cmul(w16r, w16i, w8r, w8i, w24r, w24i);
        const float lo_r[8] = {1.0f, w4r, w2r, w6r, w1r, w5r, w3r, w7r};
        const float lo_i[8] = {0.0f, w4i, w2i, w6i, w1i, w5i, w3i, w7i};
        const float hi_r[4] = {1.0f, w16r, w8r, w24r};
        const float hi_i[4] = {0.0f, w16i, w8i, w24i};
        unroll_for<31>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2;
            constexpr int B = P & 3;
            float wr, wi;
            if constexpr (A == 0) {
                wr = hi_r[B]; wi = hi_i[B];
            } else if constexpr (B == 0) {
                wr = lo_r[A]; wi = lo_i[A];
            } else {
                tw_cmul(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi);
            }
            const f16_scalar_t hwr = static_cast<f16_scalar_t>(wr), hwi = static_cast<f16_scalar_t>(wi);
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], SYCL_FFT_F16X2_SPLAT_H(hwr), SYCL_FFT_F16X2_SPLAT_H(hwi), or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    }
#else
    Real w1r, w1i;
    if constexpr (LogE >= 4 && LogE + LogT <= 8) {
        // Exact small-angle table lookup (W_64, W_128, W_256) -> zero trig ops!
        constexpr int Scale = 256 / (E * T);
        const int idx = t * Scale;
        w1r = SYCL_FFT_TWIDDLE256_RE(-1, idx);
        w1i = SYCL_FFT_TWIDDLE256_IM(-1, idx);
    } else if constexpr (LogE >= 3) {
        SYCL_FFT_SINCOS2PI_OCT1(-1, static_cast<AngleT>(t) * inv_m, w1r, w1i);
    } else {
        SYCL_FFT_SINCOS2PI_HALF(-1, static_cast<AngleT>(t) * inv_m, w1r, w1i);
    }
    w1i = SYCL_FFT_COND_NEG(w1i, pos_sign);
    Real w2r, w2i, w3r, w3i, w4r, w4i;
    SYCL_FFT_CSQR(w1r, w1i, w2r, w2i);
    SYCL_FFT_CMUL(w2r, w2i, w1r, w1i, w3r, w3i);
    SYCL_FFT_CSQR(w2r, w2i, w4r, w4i);

    if constexpr (LogE == 3) {
        const Real lo_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real lo_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        unroll_for<7>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 1; // 0..3
            constexpr int B = P & 1;  // 0..1 -> {1, w4}
            Real wr, wi;
            if constexpr (A == 0) {
                wr = w4r;
                wi = w4i;
            } else if constexpr (B == 0) {
                wr = lo_r[A];
                wi = lo_i[A];
            } else {
                SYCL_FFT_CMUL(lo_r[A], lo_i[A], w4r, w4i, wr, wi);
            }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 4) {
        Real w8r, w8i, w12r, w12i;
        SYCL_FFT_CSQR(w4r, w4i, w8r, w8i);
        SYCL_FFT_CMUL(w8r, w8i, w4r, w4i, w12r, w12i);
        const Real lo_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real lo_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        const Real hi_r[4] = {SYCL_FFT_ONE, w8r, w4r, w12r};
        const Real hi_i[4] = {SYCL_FFT_ZERO, w8i, w4i, w12i};
        unroll_for<15>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2;
            constexpr int B = P & 3;
            Real wr, wi;
            if constexpr (A == 0) {
                wr = hi_r[B];
                wi = hi_i[B];
            } else if constexpr (B == 0) {
                wr = lo_r[A];
                wi = lo_i[A];
            } else {
                SYCL_FFT_CMUL(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi);
            }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 5) {
        Real w8r, w8i, w16r, w16i, w24r, w24i;
        SYCL_FFT_CSQR(w4r, w4i, w8r, w8i);
        SYCL_FFT_CSQR(w8r, w8i, w16r, w16i);
        SYCL_FFT_CMUL(w16r, w16i, w8r, w8i, w24r, w24i);
        const Real hi_r[4] = {SYCL_FFT_ONE, w16r, w8r, w24r};
        const Real hi_i[4] = {SYCL_FFT_ZERO, w16i, w8i, w24i};

#if SYCL_FFT_WIDE
        // Minimal-register factorization for 8-byte reals (FP64/DF):
        // compute lo_a on the fly per A = 0..7 using only {w1, w2, w3, w4} + hi[4].
        const Real base_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real base_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        unroll_for<8>([&](auto a_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int A = a_ic.value;
            constexpr int Ahi = A >> 1;
            constexpr int A0 = A & 1;
            Real lor, loi;
            if constexpr (A == 0) {
                lor = SYCL_FFT_ONE;
                loi = SYCL_FFT_ZERO;
            } else if constexpr (Ahi == 0) {
                lor = w4r;
                loi = w4i;
            } else if constexpr (A0 == 0) {
                lor = base_r[Ahi];
                loi = base_i[Ahi];
            } else {
                SYCL_FFT_CMUL(base_r[Ahi], base_i[Ahi], w4r, w4i, lor, loi);
            }
            unroll_for<4>([&](auto b_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int B = b_ic.value;
                constexpr int P = (A << 2) | B;
                if constexpr (P > 0) {
                    Real wr, wi;
                    if constexpr (A == 0) {
                        wr = hi_r[B];
                        wi = hi_i[B];
                    } else if constexpr (B == 0) {
                        wr = lor;
                        wi = loi;
                    } else {
                        SYCL_FFT_CMUL(lor, loi, hi_r[B], hi_i[B], wr, wi);
                    }
                    Real or_, oi;
                    SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
                    on_elem(std::integral_constant<int, P>{}, or_, oi);
                }
            });
        });
#else
        Real w5r, w5i, w6r, w6i, w7r, w7i;
        SYCL_FFT_CMUL(w4r, w4i, w1r, w1i, w5r, w5i);
        SYCL_FFT_CMUL(w4r, w4i, w2r, w2i, w6r, w6i);
        SYCL_FFT_CMUL(w4r, w4i, w3r, w3i, w7r, w7i);
        const Real lo_r[8] = {SYCL_FFT_ONE, w4r, w2r, w6r, w1r, w5r, w3r, w7r};
        const Real lo_i[8] = {SYCL_FFT_ZERO, w4i, w2i, w6i, w1i, w5i, w3i, w7i};
        unroll_for<31>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2;
            constexpr int B = P & 3;
            Real wr, wi;
            if constexpr (A == 0) {
                wr = hi_r[B];
                wi = hi_i[B];
            } else if constexpr (B == 0) {
                wr = lo_r[A];
                wi = lo_i[A];
            } else {
                SYCL_FFT_CMUL(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi);
            }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
#endif
    }
#endif
}

// Runtime-stride twiddle helper for collapsed Pass0Functor<LogN1, true>
// (inv_m = 1 / (N1 * N2) passed as kernel argument; LogE >= 3 guarantees t * inv_m < 1/8).
template <int LogE, typename OnElem>
SYCL_FFT_DF_INLINE void SYCL_FFT_SUFFIX(apply_bitrev_twiddles_runtime)(
    const SYCL_FFT_REAL (&r_re)[1 << LogE],
    const SYCL_FFT_REAL (&r_im)[1 << LogE],
    int t,
    SYCL_FFT_ANGLE_T inv_m,
    bool pos_sign,
    OnElem&& on_elem) {
    using Real = SYCL_FFT_REAL;
    static_assert(LogE >= 3 && LogE <= 5, "apply_bitrev_twiddles_runtime expects LogE in 3..5");

    on_elem(std::integral_constant<int, 0>{}, r_re[0], r_im[0]);

#if SYCL_FFT_IS_HALF
    auto tw_cmul = [](float ar, float ai, float br, float bi, float& or_, float& oi) SYCL_FFT_LAMBDA_INLINE {
        or_ = sycl::fma(ar, br, -(ai * bi));
        oi  = sycl::fma(ar, bi,   ai * br);
    };
    auto tw_csqr = [](float ar, float ai, float& or_, float& oi) SYCL_FFT_LAMBDA_INLINE {
        or_ = sycl::fma(ar, ar, -(ai * ai));
        oi  = (2.0f * ar) * ai;
    };
    float w1r, w1i;
    sincos2pif_oct1<-1>(static_cast<float>(t) * inv_m, w1r, w1i);
    w1i = SYCL_FFT_BIT_CAST(float,
        SYCL_FFT_BIT_CAST(uint32_t, w1i) ^ (static_cast<uint32_t>(pos_sign) << 31));
    float w2r, w2i, w3r, w3i, w4r, w4i;
    tw_csqr(w1r, w1i, w2r, w2i);
    tw_cmul(w2r, w2i, w1r, w1i, w3r, w3i);
    tw_csqr(w2r, w2i, w4r, w4i);

    if constexpr (LogE == 3) {
        const float lo_r[4] = {1.0f, w2r, w1r, w3r};
        const float lo_i[4] = {0.0f, w2i, w1i, w3i};
        unroll_for<7>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 1, B = P & 1;
            float wr, wi;
            if constexpr (A == 0) { wr = w4r; wi = w4i; }
            else if constexpr (B == 0) { wr = lo_r[A]; wi = lo_i[A]; }
            else { tw_cmul(lo_r[A], lo_i[A], w4r, w4i, wr, wi); }
            const f16_scalar_t hwr = static_cast<f16_scalar_t>(wr), hwi = static_cast<f16_scalar_t>(wi);
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], SYCL_FFT_F16X2_SPLAT_H(hwr), SYCL_FFT_F16X2_SPLAT_H(hwi), or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 4) {
        float w8r, w8i, w12r, w12i;
        tw_csqr(w4r, w4i, w8r, w8i);
        tw_cmul(w8r, w8i, w4r, w4i, w12r, w12i);
        const float lo_r[4] = {1.0f, w2r, w1r, w3r};
        const float lo_i[4] = {0.0f, w2i, w1i, w3i};
        const float hi_r[4] = {1.0f, w8r, w4r, w12r};
        const float hi_i[4] = {0.0f, w8i, w4i, w12i};
        unroll_for<15>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2, B = P & 3;
            float wr, wi;
            if constexpr (A == 0) { wr = hi_r[B]; wi = hi_i[B]; }
            else if constexpr (B == 0) { wr = lo_r[A]; wi = lo_i[A]; }
            else { tw_cmul(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi); }
            const f16_scalar_t hwr = static_cast<f16_scalar_t>(wr), hwi = static_cast<f16_scalar_t>(wi);
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], SYCL_FFT_F16X2_SPLAT_H(hwr), SYCL_FFT_F16X2_SPLAT_H(hwi), or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 5) {
        float w5r, w5i, w6r, w6i, w7r, w7i, w8r, w8i, w16r, w16i, w24r, w24i;
        tw_cmul(w4r, w4i, w1r, w1i, w5r, w5i);
        tw_cmul(w4r, w4i, w2r, w2i, w6r, w6i);
        tw_cmul(w4r, w4i, w3r, w3i, w7r, w7i);
        tw_csqr(w4r, w4i, w8r, w8i);
        tw_csqr(w8r, w8i, w16r, w16i);
        tw_cmul(w16r, w16i, w8r, w8i, w24r, w24i);
        const float lo_r[8] = {1.0f, w4r, w2r, w6r, w1r, w5r, w3r, w7r};
        const float lo_i[8] = {0.0f, w4i, w2i, w6i, w1i, w5i, w3i, w7i};
        const float hi_r[4] = {1.0f, w16r, w8r, w24r};
        const float hi_i[4] = {0.0f, w16i, w8i, w24i};
        unroll_for<31>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2, B = P & 3;
            float wr, wi;
            if constexpr (A == 0) { wr = hi_r[B]; wi = hi_i[B]; }
            else if constexpr (B == 0) { wr = lo_r[A]; wi = lo_i[A]; }
            else { tw_cmul(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi); }
            const f16_scalar_t hwr = static_cast<f16_scalar_t>(wr), hwi = static_cast<f16_scalar_t>(wi);
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], SYCL_FFT_F16X2_SPLAT_H(hwr), SYCL_FFT_F16X2_SPLAT_H(hwi), or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    }
#else
    using AngleT = SYCL_FFT_ANGLE_T;
    Real w1r, w1i;
    SYCL_FFT_SINCOS2PI_OCT1(-1, static_cast<AngleT>(t) * inv_m, w1r, w1i);
    w1i = SYCL_FFT_COND_NEG(w1i, pos_sign);
    Real w2r, w2i, w3r, w3i, w4r, w4i;
    SYCL_FFT_CSQR(w1r, w1i, w2r, w2i);
    SYCL_FFT_CMUL(w2r, w2i, w1r, w1i, w3r, w3i);
    SYCL_FFT_CSQR(w2r, w2i, w4r, w4i);

    if constexpr (LogE == 3) {
        const Real lo_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real lo_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        unroll_for<7>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 1, B = P & 1;
            Real wr, wi;
            if constexpr (A == 0) { wr = w4r; wi = w4i; }
            else if constexpr (B == 0) { wr = lo_r[A]; wi = lo_i[A]; }
            else { SYCL_FFT_CMUL(lo_r[A], lo_i[A], w4r, w4i, wr, wi); }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 4) {
        Real w8r, w8i, w12r, w12i;
        SYCL_FFT_CSQR(w4r, w4i, w8r, w8i);
        SYCL_FFT_CMUL(w8r, w8i, w4r, w4i, w12r, w12i);
        const Real lo_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real lo_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        const Real hi_r[4] = {SYCL_FFT_ONE, w8r, w4r, w12r};
        const Real hi_i[4] = {SYCL_FFT_ZERO, w8i, w4i, w12i};
        unroll_for<15>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2, B = P & 3;
            Real wr, wi;
            if constexpr (A == 0) { wr = hi_r[B]; wi = hi_i[B]; }
            else if constexpr (B == 0) { wr = lo_r[A]; wi = lo_i[A]; }
            else { SYCL_FFT_CMUL(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi); }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 5) {
        Real w8r, w8i, w16r, w16i, w24r, w24i;
        SYCL_FFT_CSQR(w4r, w4i, w8r, w8i);
        SYCL_FFT_CSQR(w8r, w8i, w16r, w16i);
        SYCL_FFT_CMUL(w16r, w16i, w8r, w8i, w24r, w24i);
        const Real hi_r[4] = {SYCL_FFT_ONE, w16r, w8r, w24r};
        const Real hi_i[4] = {SYCL_FFT_ZERO, w16i, w8i, w24i};
#if SYCL_FFT_WIDE
        const Real base_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real base_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        unroll_for<8>([&](auto a_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int A = a_ic.value, Ahi = A >> 1, A0 = A & 1;
            Real lor, loi;
            if constexpr (A == 0) { lor = SYCL_FFT_ONE; loi = SYCL_FFT_ZERO; }
            else if constexpr (Ahi == 0) { lor = w4r; loi = w4i; }
            else if constexpr (A0 == 0) { lor = base_r[Ahi]; loi = base_i[Ahi]; }
            else { SYCL_FFT_CMUL(base_r[Ahi], base_i[Ahi], w4r, w4i, lor, loi); }
            unroll_for<4>([&](auto b_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int B = b_ic.value, P = (A << 2) | B;
                if constexpr (P > 0) {
                    Real wr, wi;
                    if constexpr (A == 0) { wr = hi_r[B]; wi = hi_i[B]; }
                    else if constexpr (B == 0) { wr = lor; wi = loi; }
                    else { SYCL_FFT_CMUL(lor, loi, hi_r[B], hi_i[B], wr, wi); }
                    Real or_, oi;
                    SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
                    on_elem(std::integral_constant<int, P>{}, or_, oi);
                }
            });
        });
#else
        Real w5r, w5i, w6r, w6i, w7r, w7i;
        SYCL_FFT_CMUL(w4r, w4i, w1r, w1i, w5r, w5i);
        SYCL_FFT_CMUL(w4r, w4i, w2r, w2i, w6r, w6i);
        SYCL_FFT_CMUL(w4r, w4i, w3r, w3i, w7r, w7i);
        const Real lo_r[8] = {SYCL_FFT_ONE, w4r, w2r, w6r, w1r, w5r, w3r, w7r};
        const Real lo_i[8] = {SYCL_FFT_ZERO, w4i, w2i, w6i, w1i, w5i, w3i, w7i};
        unroll_for<31>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2, B = P & 3;
            Real wr, wi;
            if constexpr (A == 0) { wr = hi_r[B]; wi = hi_i[B]; }
            else if constexpr (B == 0) { wr = lo_r[A]; wi = lo_i[A]; }
            else { SYCL_FFT_CMUL(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi); }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
#endif
    }
#endif
}

template <int LogE, int LogT>
SYCL_FFT_DF_INLINE void SYCL_FFT_SUFFIX(dif_radix2_reg)(
    SYCL_FFT_REAL (&r_re)[1 << LogE], SYCL_FFT_REAL (&r_im)[1 << LogE], int t, bool pos_sign) {
    SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogE>(r_re, r_im, pos_sign);
    if constexpr (LogT > 0) {
        SYCL_FFT_SUFFIX(apply_bitrev_twiddles_with)<LogE, LogT>(
            r_re, r_im, t, pos_sign, [&](auto p_ic, SYCL_FFT_REAL vr, SYCL_FFT_REAL vi) SYCL_FFT_LAMBDA_INLINE {
                constexpr int P = p_ic.value;
                if constexpr (P > 0) {
                    r_re[P] = vr;
                    r_im[P] = vi;
                }
            });
    }
}

// ============================================================================
// Warp-Convergent Sub-Group Radix-2 DIF across W = 2^LogW lanes (LogW <= 5):
//   Unified branchless butterfly (single ADD across all 32 lanes via sign-bit XOR),
//   single polynomial trig call for widest stage + negated squaring for H = 8, 4.
// ============================================================================
template <int LogW, int E>
SYCL_FFT_DF_INLINE void SYCL_FFT_SUFFIX(dif_radix2_subgroup)(
    const sycl::sub_group& sg, SYCL_FFT_REAL (&r_re)[E], SYCL_FFT_REAL (&r_im)[E], int lane, bool pos_sign) {
    using Real = SYCL_FFT_REAL;
    constexpr int W = 1 << LogW;
    const int l = lane & (W - 1);

    unroll_for<LogW>([&](auto s_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int S = s_ic.value;
        constexpr int H = W >> (S + 1);
        const bool is_hi = (l & H) != 0;
        const int k = l & (H - 1);

        Real tw_re = SYCL_FFT_ONE;
        Real tw_im = SYCL_FFT_ZERO;
        if constexpr (H > 2) {
            // Exact W_32 table lookup (W <= 32 -> zero polynomial trig & zero squaring!)
            const int idx32 = k * (16 / H);
            tw_re = SYCL_FFT_TWIDDLE32_RE(-1, idx32);
            tw_im = SYCL_FFT_COND_NEG(SYCL_FFT_TWIDDLE32_IM(-1, idx32), pos_sign);
        }
        unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int M = m_ic.value;
            const Real my_re = SYCL_FFT_SCALE_HALF(r_re[M]);
            const Real my_im = SYCL_FFT_SCALE_HALF(r_im[M]);
            const Real p_re = SYCL_FFT_SHUFFLE_XOR(sg, my_re, H);
            const Real p_im = SYCL_FFT_SHUFFLE_XOR(sg, my_im, H);

            // Unified warp-convergent add/sub: p + (is_hi ? -my : my)
            const Real s_re = SYCL_FFT_COND_NEG(my_re, is_hi);
            const Real s_im = SYCL_FFT_COND_NEG(my_im, is_hi);
            Real u_re = SYCL_FFT_ADD(p_re, s_re);
            Real u_im = SYCL_FFT_ADD(p_im, s_im);

            if constexpr (H == 1) {
                r_re[M] = u_re;
                r_im[M] = u_im;
            } else if constexpr (H == 2) {
                if (is_hi && k != 0) {
                    r_re[M] = SYCL_FFT_COND_NEG(u_im, pos_sign);
                    r_im[M] = SYCL_FFT_COND_NEG(u_re, !pos_sign);
                } else {
                    r_re[M] = u_re;
                    r_im[M] = u_im;
                }
            } else {
                if (is_hi) {
                    Real tr, ti;
                    SYCL_FFT_CMUL(u_re, u_im, tw_re, tw_im, tr, ti);
                    r_re[M] = tr;
                    r_im[M] = ti;
                } else {
                    r_re[M] = u_re;
                    r_im[M] = u_im;
                }
            }
        });
    });
}

// ============================================================================
// Sub-Tile Synchronization Helper:
//   Uses a fast sub-group barrier whenever the device's verified sub-group
//   covers the W-lane tile linearly; falls back to a work-group barrier
//   otherwise.
// ============================================================================
template <int W>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(sync_sub_tile)(
    const sycl::nd_item<1>& item, const sycl::sub_group& sg, bool sg_tile_sync_ok) {
    if (sg_tile_sync_ok && sg.get_max_local_range()[0] >= static_cast<size_t>(W)) {
        sycl::group_barrier(sg);
    } else {
        sycl::group_barrier(item.get_group());
    }
}

// ============================================================================
// Local-Memory Staging Holder (Split: 2 accessors, Seq: 1 accessor)
//   Preserves zero-overhead separate-accessor codegen for Split staging.
// ============================================================================
template <typename SmemT, StagingPolicy Staging>
struct SYCL_FFT_SUFFIX(SmemHolder);

template <typename SmemT>
struct SYCL_FFT_SUFFIX(SmemHolder)<SmemT, StagingPolicy::Split> {
    sycl::local_accessor<SmemT, 1> re;
    sycl::local_accessor<SmemT, 1> im;
    SYCL_FFT_SUFFIX(SmemHolder)(size_t elems, sycl::handler& h)
        : re(sycl::range<1>(elems), h), im(sycl::range<1>(elems), h) {}
};

template <typename SmemT>
struct SYCL_FFT_SUFFIX(SmemHolder)<SmemT, StagingPolicy::Seq> {
    sycl::local_accessor<SmemT, 1> buf;
    SYCL_FFT_SUFFIX(SmemHolder)(size_t elems, sycl::handler& h)
        : buf(sycl::range<1>(elems), h) {}
};

// ============================================================================
// Unified Parameterized Single-Pass DIF Kernel Body
//   Covers Split/Seq staging, Affine/XorT1 transpose-1 layouts,
//   SubgroupShuffle / TwoTransposeXor / TwoTransposeStride phase-3 engines,
//   single-chunk and 2-chunk (optional Chunk-1 g_out stashing) execution, and
//   compile-time BankModel padding/stride derivation.
// ============================================================================
template <
    int LogN,
    int LogE,
    int LogChunks,
    int B,
    StagingPolicy Staging,
    T1LayoutPolicy T1Layout,
    Phase3Policy Phase3,
    bool StashChunk1,
    BankModelKind BankKind>
struct SYCL_FFT_SUFFIX(SinglePassBody) {
    using Real     = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT    = SYCL_FFT_SMEM_T;
    using SmemH    = SYCL_FFT_SUFFIX(SmemHolder)<SmemT, Staging>;

    static constexpr int N        = 1 << LogN;
    static constexpr int E        = 1 << LogE;
    static constexpr int LogT     = LogN - LogE;
    static constexpr int T        = 1 << LogT;
    static constexpr int Chunks   = 1 << LogChunks;
    static constexpr int LogChunk = LogE - LogChunks;
    static constexpr int Chunk    = 1 << LogChunk;
    static constexpr int LogW     = LogT - LogChunk;
    static constexpr int W        = 1 << LogW;
    static constexpr int WG       = B * T;

    static_assert(LogChunks == 0 || LogChunks == 1, "LogChunks must be 0 or 1");
    static_assert(LogChunks == 0 || B == 1, "Multi-batch WG requires LogChunks == 0");
    static_assert(!StashChunk1 || LogChunks == 1, "StashChunk1 requires LogChunks == 1");
    static_assert(LogW >= 0 && LogW <= 5, "W must fit in 32 lanes");
    static_assert(Phase3 != Phase3Policy::TwoTransposeXor || Chunk == W,
                  "TwoTransposeXor requires Chunk == W");
    static_assert(Phase3 != Phase3Policy::TwoTransposeStride || (Chunk >= W && (Chunk % W) == 0),
                  "TwoTransposeStride requires Chunk to be a multiple of W");
    static_assert(T1Layout != T1LayoutPolicy::XorT1 || (LogChunks == 0 && W < Chunk),
                  "XorT1 requires LogChunks == 0 and W < Chunk");

    static constexpr BankModel kBank = bank_model_from_kind(BankKind);
    static constexpr int kElemBytes = static_cast<int>(sizeof(SmemT));
    static constexpr int Stride2 =
        (Phase3 == Phase3Policy::TwoTransposeStride)
            ? find_min_conflict_stride2(kBank, kElemBytes, W)
            : 0;
    static constexpr int MinRow =
        (Phase3 == Phase3Policy::TwoTransposeStride)
            ? ((Chunk * Stride2 > T) ? (Chunk * Stride2) : T)
            : T;
    static constexpr int RowStride =
        (T1Layout == T1LayoutPolicy::XorT1)
            ? T
            : ((Staging == StagingPolicy::Seq && Phase3 != Phase3Policy::TwoTransposeStride)
                   ? T
                   : find_min_conflict_row_stride(kBank, kElemBytes, T, W, MinRow, Stride2, Chunk));
    static constexpr int FftSmem = Chunk * RowStride;
    static constexpr size_t SmemElemsPerBuf = static_cast<size_t>(B) * static_cast<size_t>(FftSmem);
    static constexpr size_t SmemBuffers = (Staging == StagingPolicy::Split) ? size_t{2} : size_t{1};
    static constexpr size_t SmemBytes = SmemBuffers * SmemElemsPerBuf * sizeof(SmemT);
    static constexpr size_t ReqSG =
        (Phase3 == Phase3Policy::SubgroupShuffle) ? static_cast<size_t>(W) : size_t{1};

#if SYCL_FFT_IS_HALF
    using DeconvScalarT = f16_scalar_t;
#else
    using DeconvScalarT = Real;
#endif

    const ComplexT*      in;
    ComplexT*            out;
    size_t               bs;
    bool                 sg_tile_sync_ok;
    bool                 pos_sign;
    int                  compact_mode;
    int                  log_n1;
    const DeconvScalarT* d_slot_deconv;
    bool                 apply_deconv;
    SmemH                smem;

    SYCL_FFT_FORCE_INLINE static void load_elem(
        const ComplexT* g0, const ComplexT* g1, int idx, bool valid0, bool valid1,
        Real& re_out, Real& im_out) {
#if SYCL_FFT_PACKED_BATCH == 2
        f16x2_load_pair(g0, g1, idx, valid0, valid1, re_out, im_out);
#else
        (void)g1;
        (void)valid1;
        if (valid0) {
            SYCL_FFT_LOAD_COMPLEX(g0, idx, re_out, im_out);
        } else {
            re_out = SYCL_FFT_ZERO;
            im_out = SYCL_FFT_ZERO;
        }
#endif
    }

    SYCL_FFT_FORCE_INLINE static void store_elem(
        ComplexT* g0, ComplexT* g1, int idx, bool valid0, bool valid1,
        Real re_in, Real im_in) {
#if SYCL_FFT_PACKED_BATCH == 2
        f16x2_store_pair(g0, g1, idx, valid0, valid1, re_in, im_in);
#else
        (void)g1;
        (void)valid1;
        if (valid0) {
            SYCL_FFT_STORE_COMPLEX(g0, idx, re_in, im_in);
        }
#endif
    }

    SYCL_FFT_FORCE_INLINE Real load_deconv_val(
        const DeconvScalarT* d_dec0, const DeconvScalarT* d_dec1, int ci) const {
#if SYCL_FFT_PACKED_BATCH == 2
        return (log_n1 == 0)
            ? SYCL_FFT_F16X2_SPLAT_H(d_dec0[ci])
            : f16x2_make(d_dec0[ci], d_dec1[ci]);
#else
        (void)d_dec1;
        return d_dec0[ci];
#endif
    }

    // Hoisted branch-free compact-slot filtering + optional deconvolution epilogue.
    // Because out_base = (sub * T + l) with T divisible by 8 and W >= 4 power of 2,
    // slot validity (l & 3 for PSWF21; l & 7 for PSWF43 when W >= 8) is INVARIANT across
    // all M = 0..Chunk-1, eliminating per-element branches and enabling pipelined LDG/STG.
    template <typename GetValFn>
    SYCL_FFT_FORCE_INLINE void emit_chunk_epilogue(
        ComplexT* g0, ComplexT* g1,
        const DeconvScalarT* d_dec0, const DeconvScalarT* d_dec1,
        int out_base, int l, bool valid0, bool valid1,
        GetValFn&& get_val) const {
        static_assert(W >= 4, "SinglePassBody requires W >= 4");
        if (compact_mode == 0) {
            unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int M = m_ic.value;
                Real vr, vi;
                get_val(m_ic, vr, vi);
                store_elem(g0, g1, out_base + M * W, valid0, valid1, vr, vi);
            });
            return;
        }
        if constexpr (W >= 8) {
            bool keep = false;
            int ci0 = 0;
            int ci_stride = 0;
            if (compact_mode == 1) {
                const int r = l & 3;
                keep = (r == 0 || r == 3);
                ci0 = ((out_base >> 2) << 1) + (r == 3 ? 1 : 0);
                ci_stride = W >> 1;
            } else {
                const int r = l & 7;
                keep = (r != 1 && r != 6);
                const int off = r - (r >= 2 ? 1 : 0) - (r >= 7 ? 1 : 0);
                ci0 = (out_base >> 3) * 6 + off;
                ci_stride = (W >> 3) * 6;
            }
            if (!keep) return;
            if (apply_deconv) {
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    const int ci = ci0 + M * ci_stride;
                    Real vr, vi;
                    get_val(m_ic, vr, vi);
                    const Real dv = load_deconv_val(d_dec0, d_dec1, ci);
                    store_elem(g0, g1, ci, valid0, valid1, SYCL_FFT_MUL(vr, dv), SYCL_FFT_MUL(vi, dv));
                });
            } else {
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    const int ci = ci0 + M * ci_stride;
                    Real vr, vi;
                    get_val(m_ic, vr, vi);
                    store_elem(g0, g1, ci, valid0, valid1, vr, vi);
                });
            }
        } else {
            // W == 4 (LogN == 10, LogE == 4)
            if (compact_mode == 1) {
                if (l == 0 || l == 3) {
                    const int ci0 = ((out_base >> 2) << 1) + (l == 3 ? 1 : 0);
                    if (apply_deconv) {
                        unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = m_ic.value;
                            const int ci = ci0 + M * 2;
                            Real vr, vi;
                            get_val(m_ic, vr, vi);
                            const Real dv = load_deconv_val(d_dec0, d_dec1, ci);
                            store_elem(g0, g1, ci, valid0, valid1, SYCL_FFT_MUL(vr, dv), SYCL_FFT_MUL(vi, dv));
                        });
                    } else {
                        unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = m_ic.value;
                            const int ci = ci0 + M * 2;
                            Real vr, vi;
                            get_val(m_ic, vr, vi);
                            store_elem(g0, g1, ci, valid0, valid1, vr, vi);
                        });
                    }
                }
            } else {
                const int blk0 = (out_base >> 3) * 6;
                if (l != 1) {
                    const int off_e = (l >= 2) ? (l - 1) : l;
                    if (apply_deconv) {
                        unroll_for<Chunk / 2>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = 2 * k_ic.value;
                            const int ci = blk0 + k_ic.value * 6 + off_e;
                            Real vr, vi;
                            get_val(std::integral_constant<int, M>{}, vr, vi);
                            const Real dv = load_deconv_val(d_dec0, d_dec1, ci);
                            store_elem(g0, g1, ci, valid0, valid1, SYCL_FFT_MUL(vr, dv), SYCL_FFT_MUL(vi, dv));
                        });
                    } else {
                        unroll_for<Chunk / 2>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = 2 * k_ic.value;
                            const int ci = blk0 + k_ic.value * 6 + off_e;
                            Real vr, vi;
                            get_val(std::integral_constant<int, M>{}, vr, vi);
                            store_elem(g0, g1, ci, valid0, valid1, vr, vi);
                        });
                    }
                }
                if (l != 2) {
                    const int off_o = (l == 3) ? 5 : (l + 3);
                    if (apply_deconv) {
                        unroll_for<Chunk / 2>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = 2 * k_ic.value + 1;
                            const int ci = blk0 + k_ic.value * 6 + off_o;
                            Real vr, vi;
                            get_val(std::integral_constant<int, M>{}, vr, vi);
                            const Real dv = load_deconv_val(d_dec0, d_dec1, ci);
                            store_elem(g0, g1, ci, valid0, valid1, SYCL_FFT_MUL(vr, dv), SYCL_FFT_MUL(vi, dv));
                        });
                    } else {
                        unroll_for<Chunk / 2>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = 2 * k_ic.value + 1;
                            const int ci = blk0 + k_ic.value * 6 + off_o;
                            Real vr, vi;
                            get_val(std::integral_constant<int, M>{}, vr, vi);
                            store_elem(g0, g1, ci, valid0, valid1, vr, vi);
                        });
                    }
                }
            }
        }
    }

    SYCL_FFT_DF_INLINE void finish_chunk(
        const sycl::nd_item<1>& item,
        const sycl::sub_group& sg,
        Real (&c_re)[Chunk],
        Real (&c_im)[Chunk],
        ComplexT* g_out0,
        ComplexT* g_out1,
        const DeconvScalarT* d_dec0,
        const DeconvScalarT* d_dec1,
        int chunk_idx,
        int smem_base,
        int t,
        int sub,
        int l,
        bool valid0,
        bool valid1) const {
        const int row_base = smem_base + sub * RowStride;

        // ---- Transpose 1 ----
        if constexpr (Staging == StagingPolicy::Split) {
            unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int M = m_ic.value;
                const int w_idx = (T1Layout == T1LayoutPolicy::XorT1)
                    ? (smem_base + M * T + ((sub ^ M) << LogW) + l)
                    : (smem_base + M * RowStride + t);
                smem.re[w_idx] = SYCL_FFT_TO_SMEM(c_re[M]);
                smem.im[w_idx] = SYCL_FFT_TO_SMEM(c_im[M]);
            });
            sycl::group_barrier(item.get_group());
            unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int M = m_ic.value;
                const int r_idx = (T1Layout == T1LayoutPolicy::XorT1)
                    ? (smem_base + sub * T + ((M ^ sub) << LogW) + l)
                    : (row_base + M * W + l);
                c_re[M] = SYCL_FFT_FROM_SMEM(smem.re[r_idx]);
                c_im[M] = SYCL_FFT_FROM_SMEM(smem.im[r_idx]);
            });
        } else {
            unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int M = m_ic.value;
                const int w_idx = (T1Layout == T1LayoutPolicy::XorT1)
                    ? (smem_base + M * T + ((sub ^ M) << LogW) + l)
                    : (smem_base + M * RowStride + t);
                smem.buf[w_idx] = SYCL_FFT_TO_SMEM(c_re[M]);
            });
            sycl::group_barrier(item.get_group());
            unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int M = m_ic.value;
                const int r_idx = (T1Layout == T1LayoutPolicy::XorT1)
                    ? (smem_base + sub * T + ((M ^ sub) << LogW) + l)
                    : (row_base + M * W + l);
                c_re[M] = SYCL_FFT_FROM_SMEM(smem.buf[r_idx]);
            });
            sycl::group_barrier(item.get_group());

            unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int M = m_ic.value;
                const int w_idx = (T1Layout == T1LayoutPolicy::XorT1)
                    ? (smem_base + M * T + ((sub ^ M) << LogW) + l)
                    : (smem_base + M * RowStride + t);
                smem.buf[w_idx] = SYCL_FFT_TO_SMEM(c_im[M]);
            });
            sycl::group_barrier(item.get_group());
            unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int M = m_ic.value;
                const int r_idx = (T1Layout == T1LayoutPolicy::XorT1)
                    ? (smem_base + sub * T + ((M ^ sub) << LogW) + l)
                    : (row_base + M * W + l);
                c_im[M] = SYCL_FFT_FROM_SMEM(smem.buf[r_idx]);
            });
        }

        // ---- Phase 2: Register DIF across Chunk elements with inter-lane twiddles ----
        SYCL_FFT_SUFFIX(dif_radix2_reg)<LogChunk, LogW>(c_re, c_im, l, pos_sign);

        const int out_base = (LogChunks == 0)
            ? (sub * T + l)
            : ((chunk_idx * Chunk + sub) * T + l);

        // ---- Phase 3 & Coalesced Store ----
        if constexpr (Phase3 == Phase3Policy::SubgroupShuffle) {
            SYCL_FFT_SUFFIX(dif_radix2_subgroup)<LogW, Chunk>(sg, c_re, c_im, l, pos_sign);
            if (valid0) {
                emit_chunk_epilogue(
                    g_out0, g_out1, d_dec0, d_dec1, out_base, l, valid0, valid1,
                    [&](auto m_ic, Real& vr, Real& vi) SYCL_FFT_LAMBDA_INLINE {
                        vr = c_re[m_ic.value];
                        vi = c_im[m_ic.value];
                    });
            }
        } else if constexpr (Phase3 == Phase3Policy::TwoTransposeXor) {
            // Transpose 2 (XOR-swizzled, Chunk == W)
            SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
            if constexpr (Staging == StagingPolicy::Split) {
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    const int idx = row_base + M * W + (l ^ M);
                    smem.re[idx] = SYCL_FFT_TO_SMEM(c_re[M]);
                    smem.im[idx] = SYCL_FFT_TO_SMEM(c_im[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    const int idx = row_base + l * W + (K ^ l);
                    c_re[K] = SYCL_FFT_FROM_SMEM(smem.re[idx]);
                    c_im[K] = SYCL_FFT_FROM_SMEM(smem.im[idx]);
                });
            } else {
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem.buf[row_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(c_re[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    c_re[K] = SYCL_FFT_FROM_SMEM(smem.buf[row_base + l * W + (K ^ l)]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);

                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem.buf[row_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(c_im[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    c_im[K] = SYCL_FFT_FROM_SMEM(smem.buf[row_base + l * W + (K ^ l)]);
                });
            }

            SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogW>(c_re, c_im, pos_sign);

            // Transpose 3 (coalesced global store via XOR-swizzled smem)
            SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
            if constexpr (Staging == StagingPolicy::Split) {
                unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    const int idx = row_base + l * W + (K ^ l);
                    smem.re[idx] = SYCL_FFT_TO_SMEM(c_re[K]);
                    smem.im[idx] = SYCL_FFT_TO_SMEM(c_im[K]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                if (valid0) {
                    emit_chunk_epilogue(
                        g_out0, g_out1, d_dec0, d_dec1, out_base, l, valid0, valid1,
                        [&](auto m_ic, Real& vr, Real& vi) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = m_ic.value;
                            const int idx = row_base + M * W + (l ^ M);
                            vr = SYCL_FFT_FROM_SMEM(smem.re[idx]);
                            vi = SYCL_FFT_FROM_SMEM(smem.im[idx]);
                        });
                }
            } else {
                unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    smem.buf[row_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(c_re[K]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    c_re[M] = SYCL_FFT_FROM_SMEM(smem.buf[row_base + M * W + (l ^ M)]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);

                unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    smem.buf[row_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(c_im[K]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    c_im[M] = SYCL_FFT_FROM_SMEM(smem.buf[row_base + M * W + (l ^ M)]);
                });
                if (valid0) {
                    emit_chunk_epilogue(
                        g_out0, g_out1, d_dec0, d_dec1, out_base, l, valid0, valid1,
                        [&](auto m_ic, Real& vr, Real& vi) SYCL_FFT_LAMBDA_INLINE {
                            vr = c_re[m_ic.value];
                            vi = c_im[m_ic.value];
                        });
                }
            }
        } else {
            // Phase3Policy::TwoTransposeStride (Chunk == Groups * W)
            constexpr int Groups = Chunk / W;
            Real blk_re[Groups][W];
            Real blk_im[Groups][W];

            SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
            if constexpr (Staging == StagingPolicy::Split) {
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    const int idx = row_base + M * Stride2 + l;
                    smem.re[idx] = SYCL_FFT_TO_SMEM(c_re[M]);
                    smem.im[idx] = SYCL_FFT_TO_SMEM(c_im[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int G = g_ic.value;
                    const int off = row_base + (l + G * W) * Stride2;
                    unroll_for<W>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int K = k_ic.value;
                        blk_re[G][K] = SYCL_FFT_FROM_SMEM(smem.re[off + K]);
                        blk_im[G][K] = SYCL_FFT_FROM_SMEM(smem.im[off + K]);
                    });
                });
            } else {
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem.buf[row_base + M * Stride2 + l] = SYCL_FFT_TO_SMEM(c_re[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int G = g_ic.value;
                    const int off = row_base + (l + G * W) * Stride2;
                    unroll_for<W>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int K = k_ic.value;
                        blk_re[G][K] = SYCL_FFT_FROM_SMEM(smem.buf[off + K]);
                    });
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);

                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem.buf[row_base + M * Stride2 + l] = SYCL_FFT_TO_SMEM(c_im[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int G = g_ic.value;
                    const int off = row_base + (l + G * W) * Stride2;
                    unroll_for<W>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int K = k_ic.value;
                        blk_im[G][K] = SYCL_FFT_FROM_SMEM(smem.buf[off + K]);
                    });
                });
            }

            unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int G = g_ic.value;
                SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogW>(blk_re[G], blk_im[G], pos_sign);
            });

            SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
            if constexpr (Staging == StagingPolicy::Split) {
                unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int G = g_ic.value;
                    const int off = row_base + (l + G * W) * Stride2;
                    unroll_for<W>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int K = k_ic.value;
                        smem.re[off + K] = SYCL_FFT_TO_SMEM(blk_re[G][K]);
                        smem.im[off + K] = SYCL_FFT_TO_SMEM(blk_im[G][K]);
                    });
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                if (valid0) {
                    emit_chunk_epilogue(
                        g_out0, g_out1, d_dec0, d_dec1, out_base, l, valid0, valid1,
                        [&](auto m_ic, Real& vr, Real& vi) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = m_ic.value;
                            const int idx = row_base + M * Stride2 + l;
                            vr = SYCL_FFT_FROM_SMEM(smem.re[idx]);
                            vi = SYCL_FFT_FROM_SMEM(smem.im[idx]);
                        });
                }
            } else {
                unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int G = g_ic.value;
                    const int off = row_base + (l + G * W) * Stride2;
                    unroll_for<W>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int K = k_ic.value;
                        smem.buf[off + K] = SYCL_FFT_TO_SMEM(blk_re[G][K]);
                    });
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    c_re[M] = SYCL_FFT_FROM_SMEM(smem.buf[row_base + M * Stride2 + l]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);

                unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int G = g_ic.value;
                    const int off = row_base + (l + G * W) * Stride2;
                    unroll_for<W>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int K = k_ic.value;
                        smem.buf[off + K] = SYCL_FFT_TO_SMEM(blk_im[G][K]);
                    });
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg, sg_tile_sync_ok);
                unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    c_im[M] = SYCL_FFT_FROM_SMEM(smem.buf[row_base + M * Stride2 + l]);
                });
                if (valid0) {
                    emit_chunk_epilogue(
                        g_out0, g_out1, d_dec0, d_dec1, out_base, l, valid0, valid1,
                        [&](auto m_ic, Real& vr, Real& vi) SYCL_FFT_LAMBDA_INLINE {
                            vr = c_re[m_ic.value];
                            vi = c_im[m_ic.value];
                        });
                }
            }
        }
    }

    void run(sycl::nd_item<1> item) const {
        constexpr size_t PB = static_cast<size_t>(SYCL_FFT_PACKED_BATCH);
        const size_t wg_id = item.get_group_linear_id();
        const int tid = static_cast<int>(item.get_local_linear_id());
        const int b_local = (B == 1) ? 0 : (tid >> LogT);
        const int t = (B == 1) ? tid : (tid & (T - 1));
        const size_t pair_idx = (B == 1) ? wg_id : (wg_id * B + static_cast<size_t>(b_local));
        const size_t batch0 = pair_idx * PB;
        const bool valid0 = (B == 1 && PB == 1) || (batch0 < bs);
        const bool valid1 = (PB == 2) && (batch0 + 1 < bs);

        const size_t sub_nout = (compact_mode == 0)
            ? static_cast<size_t>(N)
            : ((compact_mode == 1) ? static_cast<size_t>(N >> 1)
                                   : static_cast<size_t>((N >> 2) * 3));
        const size_t n1_mask = (size_t{1} << log_n1) - size_t{1};
        const size_t p0_0 = batch0 & n1_mask;
        const size_t p0_1 = (batch0 + 1) & n1_mask;

        const ComplexT* g_in0 = valid0 ? (in + batch0 * N) : in;
        const ComplexT* g_in1 = valid1 ? (in + (batch0 + 1) * N) : g_in0;
        ComplexT* g_out0 = valid0 ? (out + batch0 * sub_nout) : out;
        ComplexT* g_out1 = valid1 ? (out + (batch0 + 1) * sub_nout) : g_out0;
        const DeconvScalarT* d_dec0 = apply_deconv ? (d_slot_deconv + p0_0 * sub_nout) : d_slot_deconv;
        const DeconvScalarT* d_dec1 = apply_deconv ? (d_slot_deconv + p0_1 * sub_nout) : d_slot_deconv;
        const int smem_base = (B == 1) ? 0 : (b_local * FftSmem);
        const int sub = t >> LogW;
        const int l = t & (W - 1);
        const auto sg = item.get_sub_group();

        if constexpr (LogChunks == 0) {
            Real r_re[E], r_im[E];
            if (valid0) {
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    load_elem(g_in0, g_in1, t + M * T, valid0, valid1, r_re[M], r_im[M]);
                });
            } else {
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_re[M] = SYCL_FFT_ZERO;
                    r_im[M] = SYCL_FFT_ZERO;
                });
            }
            SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT>(r_re, r_im, t, pos_sign);
            finish_chunk(item, sg, r_re, r_im, g_out0, g_out1, d_dec0, d_dec1, 0, smem_base, t, sub, l, valid0, valid1);
        } else if constexpr (StashChunk1) {
            // Stream chunk 0 (P=0..Chunk-1) into registers and stash chunk 1 (P=Chunk..E-1)
            // into full-length scratch slots (g_out for full FFT; mutable g_in for compact FFT).
            ComplexT* g_stash0 = (compact_mode == 0) ? g_out0 : const_cast<ComplexT*>(g_in0);
            ComplexT* g_stash1 = (compact_mode == 0) ? g_out1 : const_cast<ComplexT*>(g_in1);
            Real c_re[Chunk], c_im[Chunk];
            {
                Real r_re[E], r_im[E];
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    load_elem(g_in0, g_in1, t + M * T, valid0, valid1, r_re[M], r_im[M]);
                });
                SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogE>(r_re, r_im, pos_sign);
                SYCL_FFT_SUFFIX(apply_bitrev_twiddles_with)<LogE, LogT>(
                    r_re, r_im, t, pos_sign, [&](auto p_ic, Real vr, Real vi) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int P = p_ic.value;
                        if constexpr (P < Chunk) {
                            c_re[P] = vr;
                            c_im[P] = vi;
                        } else {
                            store_elem(g_stash0, g_stash1, (P * T) + t, valid0, valid1, vr, vi);
                        }
                    });
            }
            #pragma nounroll
            for (int chunk_idx = 0; chunk_idx < 2; ++chunk_idx) {
                if (chunk_idx > 0) {
                    sycl::group_barrier(item.get_group());
                    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        load_elem(g_stash0, g_stash1, ((Chunk + M) * T) + t, valid0, valid1, c_re[M], c_im[M]);
                    });
                }
                finish_chunk(item, sg, c_re, c_im, g_out0, g_out1, d_dec0, d_dec1, chunk_idx, smem_base, t, sub, l, valid0, valid1);
            }
        } else {
            Real r_re[E], r_im[E];
            unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int M = m_ic.value;
                load_elem(g_in0, g_in1, t + M * T, valid0, valid1, r_re[M], r_im[M]);
            });
            SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT>(r_re, r_im, t, pos_sign);

            Real c_re[Chunk], c_im[Chunk];
            #pragma nounroll
            for (int chunk_idx = 0; chunk_idx < 2; ++chunk_idx) {
                if (chunk_idx == 0) {
                    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        c_re[M] = r_re[M];
                        c_im[M] = r_im[M];
                    });
                } else {
                    sycl::group_barrier(item.get_group());
                    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        c_re[M] = r_re[Chunk + M];
                        c_im[M] = r_im[Chunk + M];
                    });
                }
                finish_chunk(item, sg, c_re, c_im, g_out0, g_out1, d_dec0, d_dec1, chunk_idx, smem_base, t, sub, l, valid0, valid1);
            }
        }
    }
};

// Named kernel functor wrapper supporting optional [[sycl::reqd_sub_group_size(S)]]
template <typename BodyT, int ReqdSG>
struct SYCL_FFT_SUFFIX(SinglePassFunctor) {
    BodyT body;
    void operator()(sycl::nd_item<1> item) const {
        body.run(item);
    }
};

#if SYCL_FFT_HAS_REQD_SG_SIZE
#define SYCL_FFT_DEFINE_PINNED_FUNCTOR(S)                                                \
    template <typename BodyT>                                                            \
    struct SYCL_FFT_SUFFIX(SinglePassFunctor)<BodyT, S> {                                \
        BodyT body;                                                                      \
        [[sycl::reqd_sub_group_size(S)]]                                                 \
        void operator()(sycl::nd_item<1> item) const {                                   \
            body.run(item);                                                              \
        }                                                                                \
    };
SYCL_FFT_DEFINE_PINNED_FUNCTOR(8)
SYCL_FFT_DEFINE_PINNED_FUNCTOR(16)
SYCL_FFT_DEFINE_PINNED_FUNCTOR(32)
SYCL_FFT_DEFINE_PINNED_FUNCTOR(64)
#undef SYCL_FFT_DEFINE_PINNED_FUNCTOR
#endif

// Self-describing single-pass kernel specification: single source of truth for
// requirements (WG, smem_bytes, req_sg, policy flags), kernel_bundle probe, and launch.
template <
    int LogN,
    int LogE,
    int LogChunks,
    int B,
    StagingPolicy Staging,
    T1LayoutPolicy T1Layout,
    Phase3Policy Phase3,
    bool StashChunk1,
    BankModelKind BankKind,
    int ReqdSG>
struct SYCL_FFT_SUFFIX(SinglePassSpec) {
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT    = SYCL_FFT_SMEM_T;
    using BodyT    = SYCL_FFT_SUFFIX(SinglePassBody)<
        LogN, LogE, LogChunks, B, Staging, T1Layout, Phase3, StashChunk1, BankKind>;
    using FunctorT = SYCL_FFT_SUFFIX(SinglePassFunctor)<BodyT, ReqdSG>;

    static constexpr int           kLogN          = LogN;
    static constexpr size_t        kWgSize        = static_cast<size_t>(BodyT::WG);
    static constexpr size_t        kSmemBytes     = BodyT::SmemBytes;
    static constexpr size_t        kReqSG         = BodyT::ReqSG;
    static constexpr int           kPinnedSG      = ReqdSG;
    static constexpr bool          kUsesWide2T    = (Phase3 != Phase3Policy::SubgroupShuffle);
    static constexpr bool          kUsesStash8192 = StashChunk1;
    static constexpr BankModelKind kBankKind      = BankKind;

    static void launch(
        sycl::queue& q, const ComplexT* in, ComplexT* out, size_t bs,
        bool sg_tile_sync_ok, bool pos_sign,
        const FftCompactParams& ep = FftCompactParams{}) {
        if (bs == 0) return;
        constexpr size_t PB = static_cast<size_t>(SYCL_FFT_PACKED_BATCH);
        const size_t num_pairs = (bs + PB - 1) / PB;
        const size_t num_wg = (num_pairs + static_cast<size_t>(B) - 1) / static_cast<size_t>(B);
        q.submit([&](sycl::handler& h) {
            typename BodyT::SmemH smem(BodyT::SmemElemsPerBuf, h);
            BodyT body{
                in, out, bs, sg_tile_sync_ok, pos_sign,
                ep.compact_mode, ep.log_n1,
                static_cast<const typename BodyT::DeconvScalarT*>(ep.d_slot_deconv),
                ep.apply_deconv,
                smem};
            h.parallel_for(
                sycl::nd_range<1>(num_wg * kWgSize, kWgSize),
                FunctorT{body});
        });
    }

    static KernelProbeInfo probe(sycl::queue& q, const sycl::device& dev) {
        return probe_kernel_bundle<FunctorT>(q, dev);
    }
};

template <typename SpecT>
constexpr SYCL_FFT_SUFFIX(SinglePassCandidate) SYCL_FFT_SUFFIX(make_candidate)(const char* id) {
    return SYCL_FFT_SUFFIX(SinglePassCandidate){
        id,
        SpecT::kLogN,
        SpecT::kWgSize,
        SpecT::kSmemBytes,
        SpecT::kReqSG,
        SpecT::kPinnedSG,
        SpecT::kUsesWide2T,
        SpecT::kUsesStash8192,
        SpecT::kBankKind,
        &SpecT::launch,
        &SpecT::probe
    };
}

#if !defined(SYCL_FFT_INST_LARGE_ONLY)
template <BankModelKind BK, int ReqdSG>
static void SYCL_FFT_SUFFIX(append_candidates_small_for_bank)(
    int logN, std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)>& out) {
    constexpr auto Split = StagingPolicy::Split;
    constexpr auto Seq   = StagingPolicy::Seq;
    constexpr auto Aff   = T1LayoutPolicy::Affine;
#if SYCL_FFT_WIDE
    constexpr auto XorT1 = T1LayoutPolicy::XorT1;
#endif
    constexpr auto Shuf  = Phase3Policy::SubgroupShuffle;
    constexpr auto Xor2T = Phase3Policy::TwoTransposeXor;
    constexpr auto Str2T = Phase3Policy::TwoTransposeStride;

#if !SYCL_FFT_WIDE
    switch (logN) {
        case 10:
            if constexpr (ReqdSG == 0 || ReqdSG >= 16) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<10, 3, 0, 2, Split, Aff, Shuf, false, BK, ReqdSG>>(
                        "split_10_e3_b2_aff"));
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<10, 3, 0, 1, Split, Aff, Shuf, false, BK, ReqdSG>>(
                        "split_10_e3_b1_aff"));
            }
            if constexpr (ReqdSG == 0 || ReqdSG >= 4) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<10, 4, 0, 1, Split, Aff, Shuf, false, BK, ReqdSG>>(
                        "split_10_e4_b1_aff"));
            }
            if constexpr (ReqdSG == 0) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<10, 4, 0, 1, Seq, Aff, Str2T, false, BK, 0>>(
                        "seq_10_e4_b1_2tstr"));
            }
            break;
        case 11:
            if constexpr (ReqdSG == 0 || ReqdSG >= 32) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<11, 3, 0, 1, Split, Aff, Shuf, false, BK, ReqdSG>>(
                        "split_11_e3_b1_aff"));
            }
            if constexpr (ReqdSG == 0 || ReqdSG >= 8) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<11, 4, 0, 1, Split, Aff, Shuf, false, BK, ReqdSG>>(
                        "split_11_e4_b1_aff"));
            }
            if constexpr (ReqdSG == 0) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<11, 4, 0, 1, Split, Aff, Str2T, false, BK, 0>>(
                        "split_11_e4_b1_2tstr"));
            }
            break;
        case 12:
            if constexpr (ReqdSG == 0 || ReqdSG >= 16) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<12, 4, 0, 1, Split, Aff, Shuf, false, BK, ReqdSG>>(
                        "split_12_e4_b1_aff"));
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<12, 4, 0, 1, Seq, Aff, Shuf, false, BK, ReqdSG>>(
                        "seq_12_e4_b1_sg"));
            }
            if constexpr (ReqdSG == 0) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<12, 4, 0, 1, Seq, Aff, Xor2T, false, BK, 0>>(
                        "seq_12_e4_b1_2txor"));
            }
            break;
        default:
            break;
    }
#else
    switch (logN) {
        case 10:
            if constexpr (ReqdSG == 0) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<10, 4, 0, 1, Seq, Aff, Str2T, false, BK, 0>>(
                        "seq_10_e4_b1_2tstr"));
            }
            if constexpr (ReqdSG == 0 || ReqdSG >= 4) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<10, 4, 0, 1, Split, XorT1, Shuf, false, BK, ReqdSG>>(
                        "split_10_e4_b1_xor"));
            }
            break;
        case 11:
            if constexpr (ReqdSG == 0) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<11, 4, 0, 1, Split, Aff, Str2T, false, BK, 0>>(
                        "wide_11_e4_b1_2tstr"));
            }
            if constexpr (ReqdSG == 0 || ReqdSG >= 8) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<11, 4, 0, 1, Seq, Aff, Shuf, false, BK, ReqdSG>>(
                        "seq_11_e4_b1_sg"));
            }
            break;
        case 12:
            if constexpr (ReqdSG == 0) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<12, 4, 0, 1, Seq, Aff, Xor2T, false, BK, 0>>(
                        "wide_12_e4_b1_2txor"));
            }
            if constexpr (ReqdSG == 0 || ReqdSG >= 16) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<12, 4, 0, 1, Seq, Aff, Shuf, false, BK, ReqdSG>>(
                        "seq_12_e4_b1_sg"));
            }
            break;
        default:
            break;
    }
#endif
}
#endif // !SYCL_FFT_INST_LARGE_ONLY

#if !defined(SYCL_FFT_INST_SMALL_ONLY)
template <BankModelKind BK, int ReqdSG>
static void SYCL_FFT_SUFFIX(append_candidates_large_for_bank)(
    int logN, std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)>& out) {
    constexpr auto Seq   = StagingPolicy::Seq;
    constexpr auto Aff   = T1LayoutPolicy::Affine;
#if !SYCL_FFT_WIDE
    constexpr auto Shuf  = Phase3Policy::SubgroupShuffle;
#endif
    constexpr auto Xor2T = Phase3Policy::TwoTransposeXor;

#if !SYCL_FFT_WIDE
    switch (logN) {
        case 13:
            if constexpr (ReqdSG == 0 || ReqdSG >= 32) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<13, 4, 0, 1, Seq, Aff, Shuf, false, BK, ReqdSG>>(
                        "seq_13_e4_b1_sg"));
            }
            if constexpr (ReqdSG == 0 || ReqdSG >= 16) {
                // Low-WG (256), 16 KiB smem, SG=16 fallback for register-limited / Intel GPUs
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<13, 5, 1, 1, Seq, Aff, Shuf, false, BK, ReqdSG>>(
                        "seq_13_e5_c2_sg"));
            }
            if constexpr (ReqdSG == 0) {
                // Low-WG (256), 16 KiB smem, SG=1 (no shuffles) fallback
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<13, 5, 1, 1, Seq, Aff, Xor2T, false, BK, 0>>(
                        "seq_13_e5_c2_2txor"));
            }
            break;
        case 14:
            if constexpr (ReqdSG == 0 || ReqdSG >= 32) {
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<14, 5, 1, 1, Seq, Aff, Shuf, false, BK, ReqdSG>>(
                        "seq_14_e5_c2_sg"));
            }
            break;
        default:
            break;
    }
#else
    switch (logN) {
        case 13:
            if constexpr (ReqdSG == 0) {
#if SYCL_FFT_IS_DF
                // Default heuristic prefers stashing chunk 1 on DF; both variants exist for probe/autotune.
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<13, 5, 1, 1, Seq, Aff, Xor2T, true, BK, 0>>(
                        "wide_13_e5_c2_xor_stash"));
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<13, 5, 1, 1, Seq, Aff, Xor2T, false, BK, 0>>(
                        "wide_13_e5_c2_xor_nostash"));
#else
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<13, 5, 1, 1, Seq, Aff, Xor2T, false, BK, 0>>(
                        "wide_13_e5_c2_xor_nostash"));
                out.push_back(SYCL_FFT_SUFFIX(make_candidate)<
                    SYCL_FFT_SUFFIX(SinglePassSpec)<13, 5, 1, 1, Seq, Aff, Xor2T, true, BK, 0>>(
                        "wide_13_e5_c2_xor_stash"));
#endif
            }
            break;
        default:
            break;
    }
#endif
}
#endif // !SYCL_FFT_INST_SMALL_ONLY

#if defined(SYCL_FFT_INST_LARGE_ONLY)
std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)> SYCL_FFT_SUFFIX(get_single_pass_candidates_small)(
    int logN, BankModelKind bank_kind);
#else
#if defined(SYCL_FFT_INST_SMALL_ONLY)
std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)> SYCL_FFT_SUFFIX(get_single_pass_candidates_small)(
#else
static std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)> SYCL_FFT_SUFFIX(get_single_pass_candidates_small)(
#endif
    int logN, BankModelKind bank_kind) {
    std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)> cands;
#if defined(SYCL_FFT_ENABLE_BANK32X8)
    if (bank_kind == BankModelKind::B32x8) {
        SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x8, 0>(logN, cands);
#if SYCL_FFT_HAS_REQD_SG_SIZE
        SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x8, 8>(logN, cands);
        SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x8, 16>(logN, cands);
        SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x8, 32>(logN, cands);
        SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x8, 64>(logN, cands);
#endif
        return cands;
    }
#else
    (void)bank_kind;
#endif
    SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x4, 0>(logN, cands);
#if SYCL_FFT_HAS_REQD_SG_SIZE
    SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x4, 8>(logN, cands);
    SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x4, 16>(logN, cands);
    SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x4, 32>(logN, cands);
    SYCL_FFT_SUFFIX(append_candidates_small_for_bank)<BankModelKind::B32x4, 64>(logN, cands);
#endif
    return cands;
}
#endif // SYCL_FFT_INST_LARGE_ONLY

#if !defined(SYCL_FFT_INST_SMALL_ONLY)
static std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)> SYCL_FFT_SUFFIX(get_single_pass_candidates_large)(
    int logN, BankModelKind bank_kind) {
    std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)> cands;
#if defined(SYCL_FFT_ENABLE_BANK32X8)
    if (bank_kind == BankModelKind::B32x8) {
        SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x8, 0>(logN, cands);
#if SYCL_FFT_HAS_REQD_SG_SIZE
        SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x8, 8>(logN, cands);
        SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x8, 16>(logN, cands);
        SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x8, 32>(logN, cands);
        SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x8, 64>(logN, cands);
#endif
        return cands;
    }
#else
    (void)bank_kind;
#endif
    SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x4, 0>(logN, cands);
#if SYCL_FFT_HAS_REQD_SG_SIZE
    SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x4, 8>(logN, cands);
    SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x4, 16>(logN, cands);
    SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x4, 32>(logN, cands);
    SYCL_FFT_SUFFIX(append_candidates_large_for_bank)<BankModelKind::B32x4, 64>(logN, cands);
#endif
    return cands;
}

static std::vector<SYCL_FFT_SUFFIX(SinglePassCandidate)> SYCL_FFT_SUFFIX(get_single_pass_candidates)(
    int logN, BankModelKind bank_kind) {
    if (logN <= 12) {
        return SYCL_FFT_SUFFIX(get_single_pass_candidates_small)(logN, bank_kind);
    }
    return SYCL_FFT_SUFFIX(get_single_pass_candidates_large)(logN, bank_kind);
}

// ============================================================================
// 2-Pass DIF Pass 0 & Zero-Smem Multi-Stage Global Fallback Building Block:
//   Register-only, ZERO shared memory, ZERO barriers, ZERO sub-group shuffles.
//   100% in-place safe (each thread reads & writes strictly its own N1 strided slots).
//   Supports LogN2 == 0 as the twiddle-free final stage of multi-stage fallback.
// ============================================================================
template <int LogN1, bool HasTwiddles, bool UseVolatileStore = false>
struct SYCL_FFT_SUFFIX(Pass0Functor) {
    using Real     = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;
    using AngleT   = SYCL_FFT_ANGLE_T;
    static constexpr int N1 = 1 << LogN1;

    const ComplexT* in;
    ComplexT*       tmp;
    size_t          chunk_bs;
    int             log_n2;
    AngleT          inv_m;
    bool            pos_sign;

    void operator()(sycl::nd_item<1> item) const {
        constexpr size_t PB = static_cast<size_t>(SYCL_FFT_PACKED_BATCH);
        const int n2 = HasTwiddles ? (1 << log_n2) : 1;
        const size_t n = static_cast<size_t>(N1) << (HasTwiddles ? log_n2 : 0);
        const size_t gid = item.get_global_linear_id();
        const size_t pair_idx = HasTwiddles ? (gid >> log_n2) : gid;
        const size_t batch0 = pair_idx * PB;
        const bool valid0 = (PB == 1) || (batch0 < chunk_bs);
        const bool valid1 = (PB == 2) && (batch0 + 1 < chunk_bs);
        const int j = HasTwiddles ? static_cast<int>(gid & static_cast<size_t>(n2 - 1)) : 0;

        const ComplexT* g_in0 = valid0 ? (in + batch0 * n + j) : in;
        const ComplexT* g_in1 = valid1 ? (in + (batch0 + 1) * n + j) : g_in0;
        ComplexT* g_tmp0 = valid0 ? (tmp + batch0 * n + j) : tmp;
        ComplexT* g_tmp1 = valid1 ? (tmp + (batch0 + 1) * n + j) : g_tmp0;

        Real r_re[N1], r_im[N1];
        unroll_for<N1>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int M = m_ic.value;
#if SYCL_FFT_PACKED_BATCH == 2
            f16x2_load_pair(g_in0, g_in1, M * n2, valid0, valid1, r_re[M], r_im[M]);
#else
            (void)g_in1; (void)valid0; (void)valid1;
            SYCL_FFT_LOAD_COMPLEX(g_in0, M * n2, r_re[M], r_im[M]);
#endif
        });

        SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogN1>(r_re, r_im, pos_sign);

        if constexpr (HasTwiddles) {
            SYCL_FFT_SUFFIX(apply_bitrev_twiddles_runtime)<LogN1>(
                r_re, r_im, j, inv_m, pos_sign, [&](auto p_ic, Real vr, Real vi) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int P = p_ic.value;
#if SYCL_FFT_PACKED_BATCH == 2
                    f16x2_store_pair(g_tmp0, g_tmp1, P * n2, valid0, valid1, vr, vi);
#else
                    (void)g_tmp1;
                    SYCL_FFT_STORE_COMPLEX(g_tmp0, P * n2, vr, vi);
#endif
                });
        } else if constexpr (UseVolatileStore) {
            // Store scalar lanes via volatile pointer on CUDA/NVPTX to prevent LLVM-21
            // LoadStoreVectorizer from merging contiguous LogN2==0 stores into invalid
            // 256-bit PTX '.b4' instructions on sm_120.
            unroll_for<N1>([&](auto p_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int P = p_ic.value;
#if SYCL_FFT_IS_DF
                volatile float* dst = reinterpret_cast<volatile float*>(g_tmp0 + P);
                dst[0] = r_re[P].hi;
                dst[1] = r_re[P].lo;
                dst[2] = r_im[P].hi;
                dst[3] = r_im[P].lo;
#elif SYCL_FFT_IS_DOUBLE
                volatile double* dst = reinterpret_cast<volatile double*>(g_tmp0 + P);
                dst[0] = r_re[P];
                dst[1] = r_im[P];
#elif SYCL_FFT_IS_HALF
                const uint32_t re_u32 = SYCL_FFT_BIT_CAST(uint32_t, r_re[P]);
                const uint32_t im_u32 = SYCL_FFT_BIT_CAST(uint32_t, r_im[P]);
                if (valid0) {
                    volatile uint16_t* dst0 = reinterpret_cast<volatile uint16_t*>(g_tmp0 + P);
                    dst0[0] = static_cast<uint16_t>(re_u32 & 0xFFFFu);
                    dst0[1] = static_cast<uint16_t>(im_u32 & 0xFFFFu);
                }
                if (valid1) {
                    volatile uint16_t* dst1 = reinterpret_cast<volatile uint16_t*>(g_tmp1 + P);
                    dst1[0] = static_cast<uint16_t>(re_u32 >> 16);
                    dst1[1] = static_cast<uint16_t>(im_u32 >> 16);
                }
#else
                volatile float* dst = reinterpret_cast<volatile float*>(g_tmp0 + P);
                dst[0] = r_re[P];
                dst[1] = r_im[P];
#endif
            });
        } else {
            // Clean coalesced/vectorizable stores for non-CUDA backends (HIP, Level Zero, CPU).
            unroll_for<N1>([&](auto p_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int P = p_ic.value;
#if SYCL_FFT_PACKED_BATCH == 2
                f16x2_store_pair(g_tmp0, g_tmp1, P, valid0, valid1, r_re[P], r_im[P]);
#else
                (void)g_tmp1;
                SYCL_FFT_STORE_COMPLEX(g_tmp0, P, r_re[P], r_im[P]);
#endif
            });
        }
    }
};

template <int LogN1, bool HasTwiddles, bool UseVolatileStore = false>
static void SYCL_FFT_SUFFIX(launch_pass0_reg)(
    sycl::queue& q,
    const SYCL_FFT_COMPLEX* in,
    SYCL_FFT_COMPLEX* tmp,
    int log_n2,
    size_t chunk_bs,
    bool pos_sign,
    size_t wg_size = 256) {
    using FunctorT = SYCL_FFT_SUFFIX(Pass0Functor)<LogN1, HasTwiddles, UseVolatileStore>;
    using AngleT = SYCL_FFT_ANGLE_T;
    constexpr size_t PB = static_cast<size_t>(SYCL_FFT_PACKED_BATCH);

    if (chunk_bs == 0) return;
    const size_t n2 = HasTwiddles ? (size_t{1} << log_n2) : size_t{1};
    const size_t num_pairs = (chunk_bs + PB - 1) / PB;
    const size_t total_threads = num_pairs * n2;
    size_t wg = 1;
    const size_t cap = std::min(std::max<size_t>(1, wg_size), total_threads);
    while ((wg << 1) <= cap) wg <<= 1;
    while (wg > 1 && (total_threads % wg) != 0) wg >>= 1;

    const AngleT inv_m = HasTwiddles
        ? (static_cast<AngleT>(1.0) / static_cast<AngleT>(size_t{1} << (LogN1 + log_n2)))
        : static_cast<AngleT>(0.0);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(total_threads, wg),
            FunctorT{in, tmp, chunk_bs, HasTwiddles ? log_n2 : 0, inv_m, pos_sign});
    });
}

static void SYCL_FFT_SUFFIX(dispatch_pass0_stage)(
    sycl::queue& q,
    int log_n1,
    int log_n2,
    const SYCL_FFT_COMPLEX* in,
    SYCL_FFT_COMPLEX* out,
    size_t bs,
    bool pos_sign,
    size_t wg_size,
    bool use_volatile_store = false) {
    if (log_n2 == 0) {
        if (use_volatile_store) {
            switch (log_n1) {
                case 3: SYCL_FFT_SUFFIX(launch_pass0_reg)<3, false, true>(q, in, out, 0, bs, pos_sign, wg_size); return;
                case 4: SYCL_FFT_SUFFIX(launch_pass0_reg)<4, false, true>(q, in, out, 0, bs, pos_sign, wg_size); return;
                case 5: SYCL_FFT_SUFFIX(launch_pass0_reg)<5, false, true>(q, in, out, 0, bs, pos_sign, wg_size); return;
                default: break;
            }
        } else {
            switch (log_n1) {
                case 3: SYCL_FFT_SUFFIX(launch_pass0_reg)<3, false, false>(q, in, out, 0, bs, pos_sign, wg_size); return;
                case 4: SYCL_FFT_SUFFIX(launch_pass0_reg)<4, false, false>(q, in, out, 0, bs, pos_sign, wg_size); return;
                case 5: SYCL_FFT_SUFFIX(launch_pass0_reg)<5, false, false>(q, in, out, 0, bs, pos_sign, wg_size); return;
                default: break;
            }
        }
    } else if (log_n2 > 0) {
        switch (log_n1) {
            case 3: SYCL_FFT_SUFFIX(launch_pass0_reg)<3, true, false>(q, in, out, log_n2, bs, pos_sign, wg_size); return;
            case 4: SYCL_FFT_SUFFIX(launch_pass0_reg)<4, true, false>(q, in, out, log_n2, bs, pos_sign, wg_size); return;
            case 5: SYCL_FFT_SUFFIX(launch_pass0_reg)<5, true, false>(q, in, out, log_n2, bs, pos_sign, wg_size); return;
            default: break;
        }
    }
    throw std::invalid_argument("Unsupported Pass-0 stage factorization");
}

static KernelProbeInfo SYCL_FFT_SUFFIX(probe_pass0_stage)(
    sycl::queue& q, const sycl::device& dev, int log_n1, int log_n2) {
    if (log_n2 <= 0) return KernelProbeInfo{};
    switch (log_n1) {
        case 3: return probe_kernel_bundle<SYCL_FFT_SUFFIX(Pass0Functor)<3, true>>(q, dev);
        case 4: return probe_kernel_bundle<SYCL_FFT_SUFFIX(Pass0Functor)<4, true>>(q, dev);
        case 5: return probe_kernel_bundle<SYCL_FFT_SUFFIX(Pass0Functor)<5, true>>(q, dev);
        default: return KernelProbeInfo{};
    }
}

// Zero-smem, zero-shuffle multi-stage decomposition for logN = 10..18
static int SYCL_FFT_SUFFIX(multistage_plan_factors)(int logN, int (&stages)[4]) {
    switch (logN) {
        case 10: stages[0] = 4; stages[1] = 3; stages[2] = 3; return 3;
        case 11: stages[0] = 4; stages[1] = 4; stages[2] = 3; return 3;
        case 12: stages[0] = 4; stages[1] = 4; stages[2] = 4; return 3;
        case 13: stages[0] = 5; stages[1] = 4; stages[2] = 4; return 3;
        case 14: stages[0] = 5; stages[1] = 5; stages[2] = 4; return 3;
        case 15: stages[0] = 5; stages[1] = 5; stages[2] = 5; return 3;
        case 16: stages[0] = 4; stages[1] = 4; stages[2] = 4; stages[3] = 4; return 4;
        case 17: stages[0] = 5; stages[1] = 4; stages[2] = 4; stages[3] = 4; return 4;
        case 18: stages[0] = 5; stages[1] = 5; stages[2] = 4; stages[3] = 4; return 4;
        default: return 0;
    }
}

static void SYCL_FFT_SUFFIX(execute_multistage_global)(
    sycl::queue& q,
    const SYCL_FFT_COMPLEX* in,
    SYCL_FFT_COMPLEX* out,
    int logN,
    size_t bs,
    size_t chunk_bs,
    bool pos_sign,
    size_t wg_size,
    bool use_volatile_store) {
    if (bs == 0) return;
    int stages[4] = {};
    const int nstages = SYCL_FFT_SUFFIX(multistage_plan_factors)(logN, stages);
    if (nstages == 0) {
        throw std::invalid_argument("Unsupported multi-stage logN");
    }
    const size_t N = size_t{1} << logN;
    const size_t eff_chunk = std::max<size_t>(1, chunk_bs);
    size_t done = 0;
    while (done < bs) {
        const size_t cur = std::min(eff_chunk, bs - done);
        const SYCL_FFT_COMPLEX* c_in = in + done * N;
        SYCL_FFT_COMPLEX* c_out = out + done * N;

        size_t sub_bs = cur;
        int rem = logN;
        for (int s = 0; s < nstages; ++s) {
            const int r = stages[s];
            rem -= r;
            const SYCL_FFT_COMPLEX* src = (s == 0) ? c_in : c_out;
            SYCL_FFT_SUFFIX(dispatch_pass0_stage)(
                q, r, rem, src, c_out, sub_bs, pos_sign, wg_size, use_volatile_store);
            sub_bs <<= r;
        }
        done += cur;
    }
}

// ============================================================================
// Adaptive Hardware-Generalized DifFftPlan Member Definitions (logN = 10..18)
// ============================================================================
template <int Sign>
SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::SYCL_FFT_SUFFIX(DifFftPlan)(
    sycl::queue& q, int logN, size_t max_bs, const FftLaunchConfig& cfg)
    : q_(q), logN_(logN), N_((logN >= 0 && logN <= 30) ? (size_t{1} << logN) : size_t{0}),
      dev_cfg_(DeviceFftConfig::query(q.get_device())), launch_cfg_(cfg) {
    if (!q_.is_in_order()) {
        throw std::invalid_argument("DifFftPlan requires an in-order sycl::queue");
    }
    if (logN < 10 || logN > 18) {
        throw std::invalid_argument("Supported logN range is 10..18");
    }
    d_dummy_deconv_ = UsmPtr<ComplexT>::alloc_device(1, q_);
    configure_plan(max_bs);
}

template <int Sign>
SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::SYCL_FFT_SUFFIX(DifFftPlan)(
    sycl::queue& q, int logN, size_t max_bs, size_t l2_chunk_bytes)
    : SYCL_FFT_SUFFIX(DifFftPlan)(q, logN, max_bs, make_chunk_config(l2_chunk_bytes)) {}

template <int Sign>
SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::~SYCL_FFT_SUFFIX(DifFftPlan)() {
    if (scratch_ || d_dummy_deconv_) {
        try {
            q_.wait();
        } catch (...) {
        }
        if (scratch_) {
            sycl::free(scratch_, q_);
            scratch_ = nullptr;
        }
    }
}

template <int Sign>
void SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::execute_raw(
    const ComplexT* in, ComplexT* out, size_t bs, size_t chunk_bs_override) {
    if (bs == 0) return;
    const FftCompactParams ep_full{0, 0, d_dummy_deconv_.get(), false};
    if (exec_mode_ == PlanExecMode::SinglePass) {
        single_pass_cand_.launch(q_, in, out, bs, sg_tile_sync_ok_, kPosSign, ep_full);
        return;
    }
    const size_t eff_chunk = (chunk_bs_override > 0)
        ? std::min(chunk_bs_, chunk_bs_override)
        : chunk_bs_;
    if (exec_mode_ == PlanExecMode::MultiStageGlobal) {
        SYCL_FFT_SUFFIX(execute_multistage_global)(
            q_, in, out, logN_, bs, eff_chunk, kPosSign, pass0_wg_size_,
            dev_cfg_.backend_tag == "cuda");
        return;
    }
    // PlanExecMode::TwoPass
    const size_t n1 = size_t{1} << log_n1_;
    const bool use_vol = (dev_cfg_.backend_tag == "cuda");
    size_t done = 0;
    while (done < bs) {
        const size_t cur = std::min(eff_chunk, bs - done);
        const ComplexT* c_in = in + done * N_;
        ComplexT* c_out = out + done * N_;
        SYCL_FFT_SUFFIX(dispatch_pass0_stage)(
            q_, log_n1_, log_n2_, c_in, scratch_, cur, kPosSign, pass0_wg_size_, use_vol);
        single_pass_cand_.launch(q_, scratch_, c_out, cur * n1, sg_tile_sync_ok_, kPosSign, ep_full);
        done += cur;
    }
}

template <int Sign>
void SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::execute_raw_compact(
    ComplexT* in, ComplexT* out, size_t bs, int compact_mode,
    const void* d_slot_deconv, size_t chunk_bs_override) {
    if (bs == 0) return;
    if (compact_mode != 1 && compact_mode != 2) {
        throw std::invalid_argument("DifFftPlan::execute_raw_compact: compact_mode must be 1 (PSWF21) or 2 (PSWF43)");
    }
    if (exec_mode_ == PlanExecMode::MultiStageGlobal) {
        throw std::runtime_error("DifFftPlan::execute_raw_compact: unsupported in MultiStageGlobal mode");
    }
    const bool apply_dec = (d_slot_deconv != nullptr);
    const void* dec_ptr = apply_dec ? d_slot_deconv : static_cast<const void*>(d_dummy_deconv_.get());
    if (exec_mode_ == PlanExecMode::SinglePass) {
        const FftCompactParams ep{compact_mode, 0, dec_ptr, apply_dec};
        single_pass_cand_.launch(q_, in, out, bs, sg_tile_sync_ok_, kPosSign, ep);
        return;
    }
    // PlanExecMode::TwoPass: Pass-0 into scratch_, Pass-1 fuses compact filtering + deconv into out
    const size_t eff_chunk = (chunk_bs_override > 0)
        ? std::min(chunk_bs_, chunk_bs_override)
        : chunk_bs_;
    const size_t n1 = size_t{1} << log_n1_;
    const size_t full_nout = (compact_mode == 1) ? (N_ >> 1) : ((N_ >> 2) * 3);
    const FftCompactParams ep{compact_mode, log_n1_, dec_ptr, apply_dec};
    const bool use_vol = (dev_cfg_.backend_tag == "cuda");
    size_t done = 0;
    while (done < bs) {
        const size_t cur = std::min(eff_chunk, bs - done);
        const ComplexT* c_in = in + done * N_;
        ComplexT* c_out = out + done * full_nout;
        SYCL_FFT_SUFFIX(dispatch_pass0_stage)(
            q_, log_n1_, log_n2_, c_in, scratch_, cur, kPosSign, pass0_wg_size_, use_vol);
        single_pass_cand_.launch(q_, scratch_, c_out, cur * n1, sg_tile_sync_ok_, kPosSign, ep);
        done += cur;
    }
}

template <int Sign>
void SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::execute(
    const ComplexT* in, ComplexT* out, size_t bs, size_t chunk_bs_override) {
    if (bs == 0) return;
    execute_raw(in, out, bs, chunk_bs_override);
    bit_reverse(out, bs);
}

template <int Sign>
void SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::bit_reverse(
    const ComplexT* src, ComplexT* dst, size_t bs) {
    ensure_bitrev_plan().execute(src, dst, bs);
}

template <int Sign>
void SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::bit_reverse(ComplexT* buf, size_t bs) {
    ensure_bitrev_plan().execute_in_place(buf, bs);
}

template <int Sign>
const char* SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::bitrev_variant_name(bool in_place) {
    return ensure_bitrev_plan().variant_name(in_place);
}

template <int Sign>
bool SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::bitrev_cache_hit(bool in_place) {
    return ensure_bitrev_plan().cache_hit(in_place);
}

template <int Sign>
BitRevPlan<SYCL_FFT_COMPLEX>& SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::bitrev_plan() {
    return ensure_bitrev_plan();
}

template <int Sign>
FftLaunchConfig SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::make_chunk_config(size_t l2_chunk_bytes) {
    FftLaunchConfig c{};
    c.l2_chunk_bytes = l2_chunk_bytes;
    return c;
}

template <int Sign>
std::string SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::format_multistage_name(int logN) {
    int stages[4] = {};
    const int k = SYCL_FFT_SUFFIX(multistage_plan_factors)(logN, stages);
    std::string s = "ms_";
    for (int i = 0; i < k; ++i) {
        if (i > 0) s.push_back('x');
        s += std::to_string(1 << stages[i]);
    }
    return s;
}

template <int Sign>
bool SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::fits_candidate(
    const Candidate& c, bool apply_variant_override, KernelProbeInfo& out_kinfo) const {
    out_kinfo = KernelProbeInfo{};
    if (!dev_cfg_.has_native_local_mem && !launch_cfg_.allow_emulated_local_mem) return false;
    if (c.wg_size > dev_cfg_.max_work_group_size) return false;
    if (c.smem_bytes > dev_cfg_.local_mem_size) return false;
    if (c.uses_wide_2t && !launch_cfg_.allow_wide_kernels) return false;

#if SYCL_FFT_HAS_REQD_SG_SIZE
    if (c.req_sg > 1) {
        const size_t wanted_sg = dev_cfg_.pick_sub_group_size(c.req_sg);
        const bool has_pinned = (wanted_sg == 8 || wanted_sg == 16 || wanted_sg == 32 || wanted_sg == 64);
        if (has_pinned) {
            if (c.pinned_sg != static_cast<int>(wanted_sg)) return false;
        } else {
            if (c.pinned_sg != 0 || c.req_sg > effective_sg_size_ || !sg_linear_ok_) return false;
        }
    } else if (c.pinned_sg != 0) {
        return false;
    }
#else
    if (c.pinned_sg != 0) return false;
    if (c.req_sg > 1 && (c.req_sg > effective_sg_size_ || !sg_linear_ok_)) return false;
#endif

#if SYCL_FFT_WIDE
    if (c.logN == 13 && launch_cfg_.stash_8192_override >= 0) {
        if (c.uses_stash_8192 != (launch_cfg_.stash_8192_override != 0)) return false;
    }
#endif
    if (apply_variant_override && !launch_cfg_.variant_override.empty()) {
        if (launch_cfg_.variant_override != c.variant_id) return false;
    }

    out_kinfo = c.probe_bundle(q_, q_.get_device());
    if (out_kinfo.available) {
        if (out_kinfo.max_wg_size > 0 && c.wg_size > out_kinfo.max_wg_size) return false;
        if (out_kinfo.compile_sg_size > 0 && c.req_sg > static_cast<size_t>(out_kinfo.compile_sg_size)) {
            return false;
        }
    }
    return true;
}

template <int Sign>
std::vector<typename SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::FilteredCandidate>
SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::collect_candidates(int n, bool apply_variant_override) const {
    std::vector<FilteredCandidate> valid;
    const auto all = SYCL_FFT_SUFFIX(get_single_pass_candidates)(n, dev_cfg_.bank_model);
    for (const auto& c : all) {
        KernelProbeInfo ki{};
        if (fits_candidate(c, apply_variant_override, ki)) {
            valid.push_back(FilteredCandidate{c, ki});
        }
    }
    return valid;
}

template <int Sign>
bool SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::find_by_id(
    const std::vector<FilteredCandidate>& valid, const std::string& id, Candidate& out) {
    if (id.empty()) return false;
    for (const auto& fc : valid) {
        if (id == fc.cand.variant_id) {
            out = fc.cand;
            return true;
        }
    }
    return false;
}

template <int Sign>
typename SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::Candidate
SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::pick_heuristic(const std::vector<FilteredCandidate>& valid) {
    // Prefer zero private_mem_size (no register spills) when bundle probe is available.
    for (const auto& fc : valid) {
        if (fc.kinfo.available && fc.kinfo.private_mem_bytes == 0) {
            return fc.cand;
        }
    }
    return valid.front().cand;
}

template <int Sign>
std::vector<typename SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::FilteredCandidate>
SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::shortlist_single_pass(
    const std::vector<FilteredCandidate>& valid) const {
    if (valid.size() <= 1) return valid;
    std::vector<FilteredCandidate> primary;
#if !SYCL_FFT_WIDE
    // When sub-group shuffles are verified, prefer SubgroupShuffle over SG=1 2-transpose fallbacks
    if (sg_linear_ok_) {
        for (const auto& fc : valid) {
            if (!fc.cand.uses_wide_2t) primary.push_back(fc);
        }
    }
#else
    // On DF/FP64, 128-reg 2-transpose kernels consistently beat 206-218-reg SubgroupShuffle
    // kernels when local memory suffices; skip cold-JITting the shuffle fallbacks when 2T fits.
    for (const auto& fc : valid) {
        if (fc.cand.uses_wide_2t) primary.push_back(fc);
    }
#endif
    const auto& pool = primary.empty() ? valid : primary;
    const size_t limit = std::min<size_t>(pool.size(), 2);
    return std::vector<FilteredCandidate>(pool.begin(), pool.begin() + limit);
}

template <int Sign>
size_t SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::representative_tune_bs(int n) const {
    const size_t fft_bytes = (size_t{1} << n) * sizeof(ComplexT);
    const size_t eff_l2 = dev_cfg_.effective_l2_bytes();
    size_t max_bytes = size_t{32} * 1024 * 1024;
    if (dev_cfg_.global_mem_size > 0) {
        max_bytes = std::min(max_bytes, std::max(size_t{4} * fft_bytes, dev_cfg_.global_mem_size / 16));
    }
    if (dev_cfg_.max_mem_alloc_size > 0) {
        max_bytes = std::min(max_bytes, std::max(size_t{4} * fft_bytes, dev_cfg_.max_mem_alloc_size / 8));
    }
    const size_t target = std::clamp<size_t>(
        std::max<size_t>(eff_l2, size_t{16} * 1024 * 1024),
        size_t{4} * fft_bytes,
        std::max(size_t{4} * fft_bytes, max_bytes));
    return std::max<size_t>(4, target / fft_bytes);
}

template <int Sign>
bool SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::tune_single_pass(
    int n,
    const std::vector<FilteredCandidate>& valid,
    Candidate& out_winner,
    double& out_best_ms,
    double* out_margin_pct,
    std::string* out_runner_up) {
    if (valid.empty()) return false;
    const auto shortlisted = shortlist_single_pass(valid);
    const size_t sub_n = size_t{1} << n;
    size_t tune_bs = representative_tune_bs(n);
    UsmPtr<ComplexT> d_in, d_out;
    while (tune_bs >= 2) {
        d_in = UsmPtr<ComplexT>::alloc_device(tune_bs * sub_n, q_);
        d_out = d_in ? UsmPtr<ComplexT>::alloc_device(tune_bs * sub_n, q_) : UsmPtr<ComplexT>{};
        if (d_in && d_out) break;
        d_in.reset();
        d_out.reset();
        tune_bs >>= 1;
    }
    if (!d_in || !d_out) {
        out_winner = pick_heuristic(shortlisted);
        out_best_ms = 0.0;
        return true;
    }

    // Fill with non-zero byte pattern to avoid GPU hardware zero-page compression
    q_.memset(d_in.get(), 0x3c, tune_bs * sub_n * sizeof(ComplexT)).wait();

    // Always pass non-null d_dummy_deconv_ (with apply_deconv=false) so AdaptiveCpp SSCP's
    // pointer-alias hash matches execute_raw, SlotBitReversed, and SlotBitReversedRaw identically.
    const FftCompactParams ep_full{0, 0, d_dummy_deconv_.get(), false};

    // Single-candidate fast path: 1-shot safety launch only (no multi-rep timing loop).
    if (shortlisted.size() == 1) {
        try {
            shortlisted[0].cand.launch(q_, d_in.get(), d_out.get(), tune_bs, sg_tile_sync_ok_, kPosSign, ep_full);
            q_.wait_and_throw();
            out_winner = shortlisted[0].cand;
            out_best_ms = 0.0;
            return true;
        } catch (...) {
            // Fall through to full valid list below
        }
    }

    auto more_conservative = [&](size_t a, size_t b) -> bool {
        const auto& fa = shortlisted[a];
        const auto& fb = shortlisted[b];
        const bool a_nospill = fa.kinfo.available && fa.kinfo.private_mem_bytes == 0;
        const bool b_nospill = fb.kinfo.available && fb.kinfo.private_mem_bytes == 0;
        if (a_nospill != b_nospill) return a_nospill;
        if (fa.cand.smem_bytes != fb.cand.smem_bytes) return fa.cand.smem_bytes < fb.cand.smem_bytes;
        return a < b;
    };

    const auto bench = run_interleaved_bench(
        shortlisted.size(),
        [&](size_t c) {
            shortlisted[c].cand.launch(q_, d_in.get(), d_out.get(), tune_bs, sg_tile_sync_ok_, kPosSign, ep_full);
            return true;
        },
        [&]() { q_.wait_and_throw(); },
        more_conservative,
        /*num_rounds=*/5,
        /*target_slice_ms=*/2.0,
        /*max_batch_reps=*/8);

    bool found = bench.ok;
    double best_ms = bench.chosen_ms;
    if (bench.ok) {
        out_winner = shortlisted[bench.chosen_idx].cand;
        if (out_margin_pct) *out_margin_pct = bench.margin_pct;
        if (out_runner_up && bench.has_runner_up) {
            *out_runner_up = shortlisted[bench.runner_up_idx].cand.variant_id;
        }
    }
    // If every shortlisted contender failed to launch, fall back to remaining valid candidates.
    if (!found && shortlisted.size() < valid.size()) {
        for (const auto& fc : valid) {
            try {
                fc.cand.launch(q_, d_in.get(), d_out.get(), tune_bs, sg_tile_sync_ok_, kPosSign, ep_full);
                q_.wait_and_throw();
                found = true;
                out_winner = fc.cand;
                best_ms = 0.0;
                break;
            } catch (...) {
            }
        }
    }
    out_best_ms = best_ms;
    return found;
}

template <int Sign>
bool SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::resolve_single_pass_for(
    int n,
    bool apply_variant_override,
    bool allow_cache,
    bool force_tune,
    Candidate& out_cand,
    bool& out_cache_hit,
    double* out_ms) {
    out_cache_hit = false;
    const auto valid = collect_candidates(n, apply_variant_override);
    if (valid.empty()) return false;

    const std::string key = AutotuneCache::make_key(dev_cfg_, SYCL_FFT_PREC_TAG, n, "fft", Sign);
    if (allow_cache && !force_tune) {
        AutotuneEntry cached{};
        if (AutotuneCache::get(cache_path_, key, cached) &&
            cached.mode == PlanExecMode::SinglePass &&
            find_by_id(valid, cached.variant_id, out_cand)) {
            out_cache_hit = true;
            return true;
        }
    }

    if (launch_cfg_.autotune == AutotuneMode::Off && !force_tune) {
        out_cand = pick_heuristic(valid);
        return true;
    }

    double best_ms = 0.0;
    double margin_pct = 0.0;
    std::string runner_up;
    if (!tune_single_pass(n, valid, out_cand, best_ms, &margin_pct, &runner_up)) {
        return false;
    }
    if (out_ms) *out_ms = best_ms;
    if ((allow_cache || force_tune) && !launch_cfg_.has_plan_overrides()) {
        AutotuneEntry e{};
        e.mode = PlanExecMode::SinglePass;
        e.variant_id = out_cand.variant_id;
        e.log_n1 = 0;
        e.pass0_wg = 0;
        e.l2_chunk_bytes = 0;
        e.margin_pct = margin_pct;
        e.runner_up_id = runner_up;
        AutotuneCache::put(cache_path_, key, e);
    }
    return true;
}

template <int Sign>
void SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::setup_multistage(size_t max_bs) {
    exec_mode_ = PlanExecMode::MultiStageGlobal;
    variant_name_ = format_multistage_name(logN_);
    const size_t fft_bytes = N_ * sizeof(ComplexT);
    const size_t l2_bytes = (launch_cfg_.l2_chunk_bytes > 0)
        ? launch_cfg_.l2_chunk_bytes
        : dev_cfg_.compute_l2_chunk_bytes(
              fft_bytes,
              size_t{16},
              is_compute_heavy_,
              launch_cfg_.l2_budget_num,
              launch_cfg_.l2_budget_den,
              launch_cfg_.min_waves_per_cu);
    chunk_bs_ = std::min(max_bs, std::max<size_t>(1, l2_bytes / fft_bytes));
    if (chunk_bs_ == 0) chunk_bs_ = 1;
    const size_t reg_wg_cap = SYCL_FFT_WIDE ? 256 : 512;
    pass0_wg_size_ = (launch_cfg_.pass0_wg_size > 0)
        ? dev_cfg_.clamp_pow2_wg(N_, launch_cfg_.pass0_wg_size, reg_wg_cap)
        : dev_cfg_.default_wg_size(N_, std::min<size_t>(256, reg_wg_cap));
    // Multi-stage executes strictly in-place on output buffer -> zero scratch needed.
    scratch_elems_ = 0;
    scratch_ = nullptr;
}

template <int Sign>
void SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::configure_plan(size_t max_bs) {
    // 1. Hardware capability & numerical safety guards
#if SYCL_FFT_IS_DOUBLE
    if (!dev_cfg_.has_fp64) {
        throw std::runtime_error(
            "FP64 FFT requested on device without sycl::aspect::fp64 (" +
            dev_cfg_.device_name + "); use DifFftPlandf (double-float) instead");
    }
#endif
#if SYCL_FFT_IS_DF
    if (!launch_cfg_.skip_df_kat) {
        const auto kat = probe_df_kat(q_, dev_cfg_);
        if (!kat.df_kat_passed) {
            throw std::runtime_error(
                "Double-Float Known-Answer Test failed on " + dev_cfg_.device_name +
                ": " + kat.df_kat_error);
        }
    }
#endif
#if SYCL_FFT_IS_HALF
    if (!launch_cfg_.skip_fp16_kat) {
        const auto kat = probe_fp16_kat(q_, dev_cfg_);
        if (!kat.fp16_kat_passed) {
            throw std::runtime_error(
                "FP16 Known-Answer Test failed on " + dev_cfg_.device_name +
                ": " + kat.fp16_kat_error);
        }
    }
#endif

    // 2. Always-on cached hardware probes (sub-group topology & FP64:FP32 ratio)
    const auto sg_probe = probe_subgroup_topology(q_, dev_cfg_);
    effective_sg_size_ = sg_probe.effective_sg_size;
    sg_linear_ok_ = sg_probe.sg_linear_ok && sg_probe.sg_xor_ok;
    sg_tile_sync_ok_ = sg_probe.sg_linear_ok;

#if SYCL_FFT_IS_DOUBLE
    if (launch_cfg_.fp64_compute_heavy_override >= 0) {
        is_compute_heavy_ = (launch_cfg_.fp64_compute_heavy_override != 0);
    } else {
        const auto fp64_probe = probe_fp64_ratio(q_, dev_cfg_);
        fp64_to_fp32_ratio_ = fp64_probe.fp64_to_fp32_ratio;
        is_compute_heavy_ = fp64_probe.fp64_compute_heavy;
    }
#else
    is_compute_heavy_ = (launch_cfg_.fp64_compute_heavy_override == 1);
#endif

    cache_path_ = AutotuneCache::resolve_cache_path(launch_cfg_.cache_path);
    bitrev_wg_size_ = (launch_cfg_.bitrev_wg_size > 0)
        ? dev_cfg_.clamp_pow2_wg(N_, launch_cfg_.bitrev_wg_size)
        : dev_cfg_.default_wg_size(N_, 256);
    // BitRevPlan is constructed lazily via ensure_bitrev_plan() so raw-FFT / NuFFT-1
    // callers never pay bit-reversal planning or LUT-state allocation overhead.

    // 3. Refuse smem-transpose kernels when !has_native_local_mem (or forced multistage)
    if (launch_cfg_.force_multistage ||
        (!dev_cfg_.has_native_local_mem && !launch_cfg_.allow_emulated_local_mem)) {
        if (!launch_cfg_.allow_multistage_fallback) {
            throw std::runtime_error(
                "Device lacks native local memory (" + dev_cfg_.device_name +
                ") and multi-stage global fallback is disabled");
        }
        setup_multistage(max_bs);
        return;
    }

    // Derive single-pass ceiling from descriptor tables
    int max_single = 9;
    for (int n = 10; n <= 14; ++n) {
        if (!collect_candidates(n, false).empty()) max_single = n;
    }
    two_pass_min_logN_ = (launch_cfg_.two_pass_min_logN > 0)
        ? launch_cfg_.two_pass_min_logN
        : (max_single + 1);

    const bool can_try_single = (logN_ < two_pass_min_logN_) &&
                                !collect_candidates(logN_, true).empty();
    const bool can_use_cache = (launch_cfg_.autotune == AutotuneMode::Auto) &&
                               !launch_cfg_.has_plan_overrides();
    const bool force_tune = (launch_cfg_.autotune == AutotuneMode::Force);
    const std::string plan_key = AutotuneCache::make_key(dev_cfg_, SYCL_FFT_PREC_TAG, logN_, "fft", Sign);

    // 4. Fast path: check persistent disk autotune cache for this logN
    if (can_use_cache) {
        AutotuneEntry cached{};
        if (AutotuneCache::get(cache_path_, plan_key, cached)) {
            if (cached.mode == PlanExecMode::SinglePass && can_try_single) {
                const auto valid = collect_candidates(logN_, true);
                if (find_by_id(valid, cached.variant_id, single_pass_cand_)) {
                    exec_mode_ = PlanExecMode::SinglePass;
                    variant_name_ = single_pass_cand_.variant_id;
                    cache_hit_ = true;
                    return;
                }
            } else if (cached.mode == PlanExecMode::TwoPass &&
                       cached.log_n1 >= 3 && cached.log_n1 <= 5) {
                const int n2 = logN_ - cached.log_n1;
                if (n2 >= 10 && n2 <= 13) {
                    const auto valid2 = collect_candidates(n2, false);
                    if (find_by_id(valid2, cached.variant_id, single_pass_cand_)) {
                        exec_mode_ = PlanExecMode::TwoPass;
                        log_n1_ = cached.log_n1;
                        log_n2_ = n2;
                        variant_name_ = single_pass_cand_.variant_id;
                        const size_t n2_sz = size_t{1} << log_n2_;
                        pass0_wg_size_ = dev_cfg_.clamp_pow2_wg(
                            n2_sz, cached.pass0_wg > 0 ? cached.pass0_wg : 256);
                        const size_t fft_bytes = N_ * sizeof(ComplexT);
                        const size_t l2_bytes = cached.l2_chunk_bytes > 0
                            ? cached.l2_chunk_bytes
                            : dev_cfg_.compute_l2_chunk_bytes(
                                  fft_bytes, size_t{1} << log_n1_, is_compute_heavy_,
                                  launch_cfg_.l2_budget_num, launch_cfg_.l2_budget_den,
                                  launch_cfg_.min_waves_per_cu);
                        chunk_bs_ = std::min(max_bs, std::max<size_t>(1, l2_bytes / fft_bytes));
                        if (chunk_bs_ == 0) chunk_bs_ = 1;
                        scratch_elems_ = chunk_bs_ * N_;
                        scratch_ = sycl::malloc_device<ComplexT>(scratch_elems_, q_);
                        if (!scratch_) throw std::bad_alloc();
                        cache_hit_ = true;
                        return;
                    }
                }
            }
        }
    }

    // 5. Try Single-Pass
    if (can_try_single) {
        bool sub_hit = false;
        if (resolve_single_pass_for(logN_, true, can_use_cache, force_tune, single_pass_cand_, sub_hit)) {
            exec_mode_ = PlanExecMode::SinglePass;
            variant_name_ = single_pass_cand_.variant_id;
            cache_hit_ = sub_hit;
            return;
        }
    }

    // 6. Two-Pass Planning & Autotuning
    struct SplitOption {
        int       n1;
        int       n2;
        Candidate pass1_cand;
    };
    std::vector<SplitOption> splits;
    auto try_add_split = [&](int n1) {
        const int n2 = logN_ - n1;
        if (n1 < 3 || n1 > 5 || n2 < 10 || n2 > 13) return;
        Candidate c{};
        bool sub_hit = false;
        if (resolve_single_pass_for(
                n2, false, !launch_cfg_.has_plan_overrides(), false, c, sub_hit)) {
            splits.push_back(SplitOption{n1, n2, c});
        }
    };

    if (launch_cfg_.two_pass_log_n1 > 0) {
        try_add_split(launch_cfg_.two_pass_log_n1);
    } else {
        int order[3] = {4, 5, 3};
#if SYCL_FFT_IS_DF
        if (logN_ == 17) {
            order[0] = 5; order[1] = 4; order[2] = 3;
        }
#endif
        for (int n1 : order) {
            if (splits.size() >= 2) break;
            if ((logN_ - n1) >= 11) try_add_split(n1);
        }
        if (splits.size() < 2) {
            for (int n1 : order) {
                if (splits.size() >= 2) break;
                if ((logN_ - n1) == 10) try_add_split(n1);
            }
        }
    }

    if (splits.empty()) {
        if (launch_cfg_.allow_multistage_fallback) {
            setup_multistage(max_bs);
            return;
        }
        throw std::runtime_error(
            "Device capabilities cannot satisfy any single-pass or 2-pass FFT split for this logN");
    }

    const size_t fft_bytes = N_ * sizeof(ComplexT);
    int best_idx = 0;
    size_t best_pass0_wg = 256;
    size_t best_l2_bytes = 0;

    auto default_pass0_wg_for = [&](int n1, int n2) -> size_t {
        const size_t n2_sz = size_t{1} << n2;
        size_t reg_wg_cap = (SYCL_FFT_WIDE && n1 >= 5) ? 256 : 512;
        const auto p0_info = SYCL_FFT_SUFFIX(probe_pass0_stage)(q_, q_.get_device(), n1, n2);
        if (p0_info.available && p0_info.max_wg_size > 0) {
            reg_wg_cap = std::min(reg_wg_cap, p0_info.max_wg_size);
        }
        return (launch_cfg_.pass0_wg_size > 0)
            ? dev_cfg_.clamp_pow2_wg(n2_sz, launch_cfg_.pass0_wg_size, reg_wg_cap)
            : dev_cfg_.default_wg_size(n2_sz, std::min<size_t>(256, reg_wg_cap));
    };

    if (launch_cfg_.autotune == AutotuneMode::Off) {
        best_idx = 0;
        best_pass0_wg = default_pass0_wg_for(splits[0].n1, splits[0].n2);
        best_l2_bytes = (launch_cfg_.l2_chunk_bytes > 0)
            ? launch_cfg_.l2_chunk_bytes
            : dev_cfg_.compute_l2_chunk_bytes(
                  fft_bytes, size_t{1} << splits[0].n1, is_compute_heavy_,
                  launch_cfg_.l2_budget_num, launch_cfg_.l2_budget_den,
                  launch_cfg_.min_waves_per_cu);
    } else {
        // Autotune (split, Pass-0 WG, L2 chunk size) on a representative working set
        size_t tune_bs = representative_tune_bs(logN_);
        tune_bs = std::max<size_t>(4, tune_bs);
        ComplexT* d_in = nullptr;
        ComplexT* d_out = nullptr;
        ComplexT* d_tmp = nullptr;
        while (tune_bs >= 2) {
            d_in = sycl::malloc_device<ComplexT>(tune_bs * N_, q_);
            d_out = d_in ? sycl::malloc_device<ComplexT>(tune_bs * N_, q_) : nullptr;
            d_tmp = d_out ? sycl::malloc_device<ComplexT>(tune_bs * N_, q_) : nullptr;
            if (d_in && d_out && d_tmp) break;
            if (d_in) sycl::free(d_in, q_);
            if (d_out) sycl::free(d_out, q_);
            if (d_tmp) sycl::free(d_tmp, q_);
            d_in = d_out = d_tmp = nullptr;
            tune_bs >>= 1;
        }

        if (!d_in || !d_out || !d_tmp) {
            best_idx = 0;
            best_pass0_wg = default_pass0_wg_for(splits[0].n1, splits[0].n2);
            best_l2_bytes = (launch_cfg_.l2_chunk_bytes > 0)
                ? launch_cfg_.l2_chunk_bytes
                : dev_cfg_.compute_l2_chunk_bytes(
                      fft_bytes, size_t{1} << splits[0].n1, is_compute_heavy_,
                      launch_cfg_.l2_budget_num, launch_cfg_.l2_budget_den,
                      launch_cfg_.min_waves_per_cu);
        } else {
            q_.memset(d_in, 0x3c, tune_bs * N_ * sizeof(ComplexT)).wait();

            auto time_two_pass_cfg = [&](const SplitOption& sp, size_t p0_wg, size_t c_bs, double& out_ms) -> bool {
                const size_t n1_sz = size_t{1} << sp.n1;
                auto run_once = [&]() {
                    size_t done = 0;
                    const bool use_vol = (dev_cfg_.backend_tag == "cuda");
                    const FftCompactParams ep_full{0, 0, d_dummy_deconv_.get(), false};
                    while (done < tune_bs) {
                        const size_t cur = std::min(c_bs, tune_bs - done);
                        SYCL_FFT_SUFFIX(dispatch_pass0_stage)(
                            q_, sp.n1, sp.n2, d_in + done * N_, d_tmp, cur, kPosSign, p0_wg, use_vol);
                        sp.pass1_cand.launch(
                            q_, d_tmp, d_out + done * N_, cur * n1_sz, sg_tile_sync_ok_, kPosSign, ep_full);
                        done += cur;
                    }
                };
                try {
                    const auto w0 = std::chrono::steady_clock::now();
                    run_once();
                    q_.wait_and_throw();
                    const double warm_ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
                    const int reps = (warm_ms > 10.0) ? 3 : 5;
                    double samples[5] = {};
                    for (int r = 0; r < reps; ++r) {
                        const auto t0 = std::chrono::steady_clock::now();
                        run_once();
                        q_.wait_and_throw();
                        samples[r] = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
                    }
                    std::sort(samples, samples + reps);
                    out_ms = samples[reps / 2];
                    return true;
                } catch (...) {
                    return false;
                }
            };

            // Factored smart search:
            //   Phase A: compare <=2 shortlisted splits at their analytical default Pass-0 WG.
            //   Phase B: on the winning split only, check 1 alternate Pass-0 WG (0 new JIT).
            //   Phase C: on the winning (split, WG), check 2 L2 chunk sizes (0 new JIT).
            double best_ms = 1e30;
            bool any_ok = false;
            for (size_t si = 0; si < splits.size(); ++si) {
                const auto& sp = splits[si];
                const size_t def_wg = default_pass0_wg_for(sp.n1, sp.n2);
                const size_t base_l2 = (launch_cfg_.l2_chunk_bytes > 0)
                    ? launch_cfg_.l2_chunk_bytes
                    : dev_cfg_.compute_l2_chunk_bytes(
                          fft_bytes, size_t{1} << sp.n1, is_compute_heavy_,
                          launch_cfg_.l2_budget_num, launch_cfg_.l2_budget_den,
                          launch_cfg_.min_waves_per_cu);
                const size_t base_cbs = std::min(tune_bs, std::max<size_t>(1, base_l2 / fft_bytes));
                double ms = 0.0;
                if (time_two_pass_cfg(sp, def_wg, base_cbs, ms) && (!any_ok || ms < best_ms)) {
                    any_ok = true;
                    best_ms = ms;
                    best_idx = static_cast<int>(si);
                    best_pass0_wg = def_wg;
                    best_l2_bytes = base_l2;
                }
            }

            if (any_ok && launch_cfg_.pass0_wg_size == 0) {
                const auto& best_sp = splits[static_cast<size_t>(best_idx)];
                const size_t n2_sz = size_t{1} << best_sp.n2;
                const size_t alt_target = SYCL_FFT_WIDE ? size_t{128} : size_t{512};
                const size_t alt_wg = dev_cfg_.clamp_pow2_wg(n2_sz, alt_target);
                if (alt_wg != best_pass0_wg) {
                    const size_t base_cbs = std::min(tune_bs, std::max<size_t>(1, best_l2_bytes / fft_bytes));
                    double ms = 0.0;
                    if (time_two_pass_cfg(best_sp, alt_wg, base_cbs, ms) && ms < best_ms) {
                        best_ms = ms;
                        best_pass0_wg = alt_wg;
                    }
                }
            }

            if (any_ok && launch_cfg_.l2_chunk_bytes == 0) {
                const auto& best_sp = splits[static_cast<size_t>(best_idx)];
                const size_t base_l2 = best_l2_bytes;
                const size_t max_l2_cap = std::min(
                    tune_bs * fft_bytes,
                    dev_cfg_.max_mem_alloc_size > 0
                        ? std::max(fft_bytes, dev_cfg_.max_mem_alloc_size / 4)
                        : (tune_bs * fft_bytes));
                const size_t base_cbs = std::min(tune_bs, std::max<size_t>(1, base_l2 / fft_bytes));
                const size_t alt_bytes = std::clamp<size_t>(
                    base_l2 * 2, fft_bytes, std::max(fft_bytes, max_l2_cap));
                const size_t alt_cbs = std::min(tune_bs, std::max<size_t>(1, alt_bytes / fft_bytes));
                if (alt_cbs != base_cbs) {
                    double ms = 0.0;
                    if (time_two_pass_cfg(best_sp, best_pass0_wg, alt_cbs, ms) && ms < best_ms) {
                        best_ms = ms;
                        best_l2_bytes = alt_cbs * fft_bytes;
                    }
                }
            }
            sycl::free(d_tmp, q_);
            sycl::free(d_out, q_);
            sycl::free(d_in, q_);

            if (!any_ok) {
                if (launch_cfg_.allow_multistage_fallback) {
                    setup_multistage(max_bs);
                    return;
                }
                throw std::runtime_error("All 2-pass FFT candidate launches failed on device");
            }
        }
    }

    const auto& chosen = splits[static_cast<size_t>(best_idx)];
    exec_mode_ = PlanExecMode::TwoPass;
    log_n1_ = chosen.n1;
    log_n2_ = chosen.n2;
    single_pass_cand_ = chosen.pass1_cand;
    variant_name_ = single_pass_cand_.variant_id;
    pass0_wg_size_ = best_pass0_wg;

    chunk_bs_ = std::min(max_bs, std::max<size_t>(1, best_l2_bytes / fft_bytes));
    if (chunk_bs_ == 0) chunk_bs_ = 1;
    scratch_elems_ = chunk_bs_ * N_;
    scratch_ = sycl::malloc_device<ComplexT>(scratch_elems_, q_);
    if (!scratch_) throw std::bad_alloc();

    if (launch_cfg_.autotune != AutotuneMode::Off && !launch_cfg_.has_plan_overrides()) {
        AutotuneEntry e{};
        e.mode = PlanExecMode::TwoPass;
        e.variant_id = single_pass_cand_.variant_id;
        e.log_n1 = log_n1_;
        e.pass0_wg = pass0_wg_size_;
        e.l2_chunk_bytes = best_l2_bytes;
        AutotuneCache::put(cache_path_, plan_key, e);
    }
}

template <int Sign>
BitRevPlan<SYCL_FFT_COMPLEX>& SYCL_FFT_SUFFIX(DifFftPlan)<Sign>::ensure_bitrev_plan() {
    if (!bitrev_plan_) {
        bitrev_plan_ = std::make_unique<BitRevPlan<ComplexT>>(
            q_, logN_, SYCL_FFT_PREC_TAG, dev_cfg_, launch_cfg_);
    }
    return *bitrev_plan_;
}

// Explicit instantiations for the selected precision
#if SYCL_FFT_ENABLE_NEG_SIGN
template class SYCL_FFT_SUFFIX(DifFftPlan)<-1>;
#endif
#if SYCL_FFT_ENABLE_POS_SIGN
template class SYCL_FFT_SUFFIX(DifFftPlan)<+1>;
#endif

#endif // !SYCL_FFT_INST_SMALL_ONLY

} // namespace sycl_fft
