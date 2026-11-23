#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace sycl_fft {

#if defined(__GNUC__) || defined(__clang__)
#define SYCL_FFT_FORCE_INLINE __attribute__((always_inline)) inline
#define SYCL_FFT_LAMBDA_INLINE __attribute__((always_inline))
#else
#define SYCL_FFT_FORCE_INLINE inline
#define SYCL_FFT_LAMBDA_INLINE
#endif

// Exact IEEE-754 ordering guard for Dekker/Knuth compensated arithmetic.
// Prevents -O3 reassociation and implicit FMA contraction from collapsing
// error-free transformations (explicit sycl::fma calls still emit FFMA).
#if defined(__clang__)
#define SYCL_FFT_DF_EXACT_FP \
    _Pragma("clang fp reassociate(off)") \
    _Pragma("clang fp contract(off)")
#else
#define SYCL_FFT_DF_EXACT_FP
#endif

#if defined(__ACPP_ENABLE_CUDA_TARGET__)
#define SYCL_FFT_HOST_DEVICE __host__ __device__
#else
#define SYCL_FFT_HOST_DEVICE
#endif

// Exact round-to-nearest-even FP32 ops that neither LLVM NVPTX ISel nor ptxas --fmad
// may fuse into FMA (critical for Dekker TwoProd / Knuth TwoSum).
SYCL_FFT_HOST_DEVICE SYCL_FFT_FORCE_INLINE float df_fadd_rn(float a, float b) {
    SYCL_FFT_DF_EXACT_FP
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

SYCL_FFT_HOST_DEVICE SYCL_FFT_FORCE_INLINE float df_fsub_rn(float a, float b) {
    SYCL_FFT_DF_EXACT_FP
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}

SYCL_FFT_HOST_DEVICE SYCL_FFT_FORCE_INLINE float df_fmul_rn(float a, float b) {
    SYCL_FFT_DF_EXACT_FP
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

// ============================================================================
// Core Compile-Time Unrolling & Bit-Reversal Helpers
// ============================================================================
template <int N, typename Fn, int... Is>
SYCL_FFT_FORCE_INLINE void unroll_for_impl(Fn&& fn, std::integer_sequence<int, Is...>) {
    (fn(std::integral_constant<int, Is>{}), ...);
}

template <int N, typename Fn>
SYCL_FFT_FORCE_INLINE void unroll_for(Fn&& fn) {
    unroll_for_impl<N>(std::forward<Fn>(fn), std::make_integer_sequence<int, N>{});
}

constexpr uint32_t bit_reverse_const(uint32_t x, int bits) {
    uint32_t r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | ((x >> i) & 1u);
    }
    return r;
}

SYCL_FFT_FORCE_INLINE uint32_t bit_reverse(uint32_t x, int bits) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    x = (x << 16) | (x >> 16);
    return x >> (32 - bits);
}

// ============================================================================
// Triple-Precision Complex & Double-Float (DF) Types
//   FP32: Complexf  (sycl::float2,  8 B)
//   FP64: Complex   (sycl::double2, 16 B)
//   DF:   Complexdf ({df_t re, im}, 16 B, ~47-48 bit mantissa via 2 x fp32)
// ============================================================================
using Complexf = sycl::float2;
using Complex  = sycl::double2;

struct alignas(8) df_t {
    float hi;
    float lo;
};

struct alignas(16) Complexdf {
    df_t re;
    df_t im;
};

static_assert(sizeof(df_t) == 8 && alignof(df_t) == 8 && std::is_trivially_copyable_v<df_t>,
              "df_t must be 8-byte aligned trivially copyable pair of floats");
static_assert(sizeof(Complexdf) == 16 && alignof(Complexdf) == 16 && std::is_trivially_copyable_v<Complexdf>,
              "Complexdf must be 16-byte aligned trivially copyable for 128-bit LDG/STG");

// ============================================================================
// Dekker / Knuth Double-Float (df_t) Arithmetic Primitives
// ============================================================================
constexpr df_t df_make(float hi, float lo) {
    return df_t{hi, lo};
}

constexpr df_t df_from_double(double d) {
    const float hi = static_cast<float>(d);
    const float lo = static_cast<float>(d - static_cast<double>(hi));
    return df_t{hi, lo};
}

SYCL_FFT_FORCE_INLINE double df_to_double(df_t a) {
    return static_cast<double>(a.hi) + static_cast<double>(a.lo);
}

SYCL_FFT_FORCE_INLINE df_t df_neg(df_t a) {
    return df_t{-a.hi, -a.lo};
}

SYCL_FFT_FORCE_INLINE df_t df_scale_pow2(df_t a, float s) {
    return df_t{a.hi * s, a.lo * s};
}

SYCL_FFT_FORCE_INLINE df_t df_select(df_t a, df_t b, uint32_t mask) {
    const uint32_t hi = (sycl::bit_cast<uint32_t>(a.hi) & mask) |
                        (sycl::bit_cast<uint32_t>(b.hi) & ~mask);
    const uint32_t lo = (sycl::bit_cast<uint32_t>(a.lo) & mask) |
                        (sycl::bit_cast<uint32_t>(b.lo) & ~mask);
    return df_t{sycl::bit_cast<float>(hi), sycl::bit_cast<float>(lo)};
}

SYCL_FFT_FORCE_INLINE df_t df_xor_sign(df_t a, uint32_t sign_bit) {
    return df_t{
        sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(a.hi) ^ sign_bit),
        sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(a.lo) ^ sign_bit)
    };
}

// Knuth TwoSum (6 FP32 ops): a + b = r + e exactly
SYCL_FFT_FORCE_INLINE void df_two_sum(float a, float b, float& r, float& e) {
    SYCL_FFT_DF_EXACT_FP
    r = df_fadd_rn(a, b);
    const float v = df_fsub_rn(r, a);
    e = df_fadd_rn(df_fsub_rn(a, df_fsub_rn(r, v)), df_fsub_rn(b, v));
}

// Dekker QuickTwoSum (3 FP32 ops, requires |x| >= |y|): x + y = r + e exactly
SYCL_FFT_FORCE_INLINE void df_two_sum_quick(float x, float y, float& r, float& e) {
    SYCL_FFT_DF_EXACT_FP
    r = df_fadd_rn(x, y);
    e = df_fsub_rn(y, df_fsub_rn(r, x));
}

// Knuth TwoDiff (6 FP32 ops): x - y = r + e exactly
SYCL_FFT_FORCE_INLINE void df_two_difference(float x, float y, float& r, float& e) {
    SYCL_FFT_DF_EXACT_FP
    r = df_fsub_rn(x, y);
    const float v = df_fsub_rn(r, x);
    e = df_fsub_rn(df_fsub_rn(x, df_fsub_rn(r, v)), df_fadd_rn(y, v));
}

// Dekker TwoProd via hardware FMA (2 FP32 ops): x * y = r + e exactly
SYCL_FFT_FORCE_INLINE void df_two_product(float x, float y, float& r, float& e) {
    SYCL_FFT_DF_EXACT_FP
    r = df_fmul_rn(x, y);
    e = sycl::fma(x, y, -r);
}

// DF + DF (11 FP32 ops)
SYCL_FFT_FORCE_INLINE df_t df_add(df_t a, df_t b) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_sum(a.hi, b.hi, r, e);
    e = df_fadd_rn(df_fadd_rn(e, a.lo), b.lo);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// DF - DF (11 FP32 ops)
SYCL_FFT_FORCE_INLINE df_t df_sub(df_t a, df_t b) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_difference(a.hi, b.hi, r, e);
    e = df_fsub_rn(df_fadd_rn(e, a.lo), b.lo);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// DF * DF (7 FP32 ops)
SYCL_FFT_FORCE_INLINE df_t df_mul(df_t a, df_t b) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_product(a.hi, b.hi, r, e);
    e = sycl::fma(a.hi, b.lo, e);
    e = sycl::fma(a.lo, b.hi, e);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// DF * float (6 FP32 ops)
SYCL_FFT_FORCE_INLINE df_t df_mul_f(df_t a, float b) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_product(a.hi, b, r, e);
    e = sycl::fma(a.lo, b, e);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// DF * DF squaring (6 FP32 ops)
SYCL_FFT_FORCE_INLINE df_t df_sqr(df_t a) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_product(a.hi, a.hi, r, e);
    e = sycl::fma(df_fadd_rn(a.hi, a.hi), a.lo, e);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// Fused DF complex multiply (38 FP32 ops vs 50 unfused):
// Feeds un-renormalized two-product pairs directly into two_diff/two_sum + quick_two_sum.
SYCL_FFT_FORCE_INLINE void df_cmul_fused(
    df_t ar, df_t ai, df_t br, df_t bi, df_t& out_r, df_t& out_i) {
    SYCL_FFT_DF_EXACT_FP
    float p1h, p1l, p2h, p2l;
    df_two_product(ar.hi, br.hi, p1h, p1l);
    p1l = sycl::fma(ar.lo, br.hi, sycl::fma(ar.hi, br.lo, p1l));

    df_two_product(ai.hi, bi.hi, p2h, p2l);
    p2l = sycl::fma(ai.lo, bi.hi, sycl::fma(ai.hi, bi.lo, p2l));

    float rr, re;
    df_two_difference(p1h, p2h, rr, re);
    re = df_fsub_rn(df_fadd_rn(re, p1l), p2l);
    df_two_sum_quick(rr, re, out_r.hi, out_r.lo);

    float q1h, q1l, q2h, q2l;
    df_two_product(ar.hi, bi.hi, q1h, q1l);
    q1l = sycl::fma(ar.lo, bi.hi, sycl::fma(ar.hi, bi.lo, q1l));

    df_two_product(ai.hi, br.hi, q2h, q2l);
    q2l = sycl::fma(ai.lo, br.hi, sycl::fma(ai.hi, br.lo, q2l));

    float ir, ie;
    df_two_sum(q1h, q2h, ir, ie);
    ie = df_fadd_rn(df_fadd_rn(ie, q1l), q2l);
    df_two_sum_quick(ir, ie, out_i.hi, out_i.lo);
}

// Fused DF complex square (26 FP32 ops vs 38 for df_cmul_fused):
// out_r = ar^2 - ai^2, out_i = 2 * ar * ai
SYCL_FFT_FORCE_INLINE void df_csqr_fused(df_t ar, df_t ai, df_t& out_r, df_t& out_i) {
    SYCL_FFT_DF_EXACT_FP
    float p1h, p1l, p2h, p2l;
    df_two_product(ar.hi, ar.hi, p1h, p1l);
    p1l = sycl::fma(df_fadd_rn(ar.hi, ar.hi), ar.lo, p1l);

    df_two_product(ai.hi, ai.hi, p2h, p2l);
    p2l = sycl::fma(df_fadd_rn(ai.hi, ai.hi), ai.lo, p2l);

    float rr, re;
    df_two_difference(p1h, p2h, rr, re);
    re = df_fsub_rn(df_fadd_rn(re, p1l), p2l);
    df_two_sum_quick(rr, re, out_r.hi, out_r.lo);

    out_i = df_scale_pow2(df_mul(ar, ai), 2.0f);
}

// ============================================================================
// Exact Compile-Time W_32^k = exp(-i 2pi k / 32) Tables (k = 0..15)
// ============================================================================
constexpr float kSqrt1_2f = 0.70710678118654752440f;
constexpr double kSqrt1_2 = 0.70710678118654752440;
constexpr df_t kSqrt1_2df = df_from_double(kSqrt1_2);

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

// Specialized W_8^{1,3} rotation: (dr + i*di) * ((NegRe ? -1 : 1) + i*(NegIm ? -1 : 1)) * sqrt(1/2)
template <bool NegRe, bool NegIm>
SYCL_FFT_FORCE_INLINE void df_cmul_sqrt1_2(df_t dr, df_t di, df_t& out_r, df_t& out_i) {
    // (dr + i*di) * (sr + i*si) * kSqrt1_2 = ((sr*dr - si*di) + i*(si*dr + sr*di)) * kSqrt1_2
    df_t tr, ti;
    if constexpr (!NegRe && NegIm) {
        // W_8^1 (Sign=-1): (1 - i)/sqrt(2) -> (dr + di) + i*(di - dr)
        tr = df_add(dr, di);
        ti = df_sub(di, dr);
    } else if constexpr (NegRe && NegIm) {
        // W_8^3 (Sign=-1): (-1 - i)/sqrt(2) -> (di - dr) + i*(-dr - di)
        tr = df_sub(di, dr);
        ti = df_neg(df_add(dr, di));
    } else if constexpr (!NegRe && !NegIm) {
        // W_8^1 (Sign=+1): (1 + i)/sqrt(2) -> (dr - di) + i*(dr + di)
        tr = df_sub(dr, di);
        ti = df_add(dr, di);
    } else {
        // W_8^3 (Sign=+1): (-1 + i)/sqrt(2) -> (-dr - di) + i*(dr - di)
        tr = df_neg(df_add(dr, di));
        ti = df_sub(dr, di);
    }
    out_r = df_mul(tr, kSqrt1_2df);
    out_i = df_mul(ti, kSqrt1_2df);
}

// ============================================================================
// FP32 Branchless Polynomial Trig (f suffix, minimax on [0, 0.25])
// ============================================================================
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pif_q1(float f, float& c_out, float& s_out) {
    const float f2 = f * f;
    float ps = sycl::fma(f2, 39.536706065730207835f, -76.549782293595742666f);
    ps = sycl::fma(ps, f2, 81.601004073261773523f);
    ps = sycl::fma(ps, f2, -41.341655031416278077f);
    ps = sycl::fma(ps, f2, 6.283185160089477443f);
    const float s_mag = ps * f;

    float pc = sycl::fma(f2, 56.242380464873243260f, -85.240330322699427860f);
    pc = sycl::fma(pc, f2, 64.934590626780991246f);
    pc = sycl::fma(pc, f2, -19.739171434702393619f);
    pc = sycl::fma(pc, f2, 0.999999953466670136f);

    c_out = pc;
    s_out = (Sign < 0) ? -s_mag : s_mag;
}

template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pif_oct1(float f, float& c_out, float& s_out) {
    sincos2pif_q1<Sign>(f, c_out, s_out);
}

template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pif_half(float f, float& c_out, float& s_out) {
    const uint32_t gt_qtr_bit = static_cast<uint32_t>(f > 0.25f);
    const uint32_t gt_qtr_mask = 0u - gt_qtr_bit;
    const float f_fold = 0.5f - f;
    f = sycl::bit_cast<float>(
        (sycl::bit_cast<uint32_t>(f_fold) & gt_qtr_mask) |
        (sycl::bit_cast<uint32_t>(f) & ~gt_qtr_mask));

    float c_mag, s_signed;
    sincos2pif_q1<Sign>(f, c_mag, s_signed);
    c_out = sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(c_mag) ^ (gt_qtr_bit << 31));
    s_out = s_signed;
}

template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pif_branchless(float x, float& c_out, float& s_out) {
    float f = x - sycl::floor(x);
    const uint32_t ge_half_bit = static_cast<uint32_t>(f >= 0.5f);
    const uint32_t ge_half_mask = 0u - ge_half_bit;
    f -= sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(0.5f) & ge_half_mask);

    const uint32_t gt_qtr_bit = static_cast<uint32_t>(f > 0.25f);
    const uint32_t gt_qtr_mask = 0u - gt_qtr_bit;
    const float f_fold = 0.5f - f;
    f = sycl::bit_cast<float>(
        (sycl::bit_cast<uint32_t>(f_fold) & gt_qtr_mask) |
        (sycl::bit_cast<uint32_t>(f) & ~gt_qtr_mask));

    float c_mag, s_base;
    sincos2pif_q1<1>(f, c_mag, s_base);
    const uint32_t s_flip = (ge_half_bit ^ static_cast<uint32_t>(Sign < 0)) << 31;
    const uint32_t c_flip = (ge_half_bit ^ gt_qtr_bit) << 31;
    c_out = sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(c_mag) ^ c_flip);
    s_out = sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(s_base) ^ s_flip);
}

SYCL_FFT_FORCE_INLINE float cos2pif_branchless(float x) {
    float c, s;
    sincos2pif_branchless<1>(x, c, s);
    return c;
}

SYCL_FFT_FORCE_INLINE float sin2pif_branchless(float x) {
    float c, s;
    sincos2pif_branchless<1>(x, c, s);
    return s;
}

SYCL_FFT_FORCE_INLINE float cos2pif(float x) { return cos2pif_branchless(x); }
SYCL_FFT_FORCE_INLINE float sin2pif(float x) { return sin2pif_branchless(x); }

// ============================================================================
// FP64 Branchless Polynomial Trig (raw unsuffixed, minimax on [0, 0.125])
//   sin: degree 13 (7 terms in f^2), max err <= 1.11e-16 (0.5 ULP)
//   cos: degree 14 (8 terms in f^2), max err <= 1.11e-16 (0.5 ULP)
// ============================================================================
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pi_oct1(double f, double& c_out, double& s_out) {
    const double f2 = f * f;

    double ps = sycl::fma(f2, 3.78086895930229572e+00, -1.50938042099871144e+01);
    ps = sycl::fma(ps, f2,  4.20586850201608939e+01);
    ps = sycl::fma(ps, f2, -7.67058597042745447e+01);
    ps = sycl::fma(ps, f2,  8.16052492759480401e+01);
    ps = sycl::fma(ps, f2, -4.13417022403996341e+01);
    ps = sycl::fma(ps, f2,  6.28318530717958623e+00);
    const double s_mag = ps * f;

    double pc = sycl::fma(f2, -1.69684947029188260e+00,  7.90309168775169724e+00);
    pc = sycl::fma(pc, f2, -2.64262509086221371e+01);
    pc = sycl::fma(pc, f2,  6.02446413288672957e+01);
    pc = sycl::fma(pc, f2, -8.54568172065227145e+01);
    pc = sycl::fma(pc, f2,  6.49393940226679547e+01);
    pc = sycl::fma(pc, f2, -1.97392088021787160e+01);
    pc = sycl::fma(pc, f2,  1.00000000000000000e+00);

    c_out = pc;
    s_out = (Sign < 0) ? -s_mag : s_mag;
}

template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pi_half(double f, double& c_out, double& s_out) {
    // Valid for f in [0, 0.5): branchless quadrant + octant fold
    const uint64_t gt_qtr_bit = static_cast<uint64_t>(f > 0.25);
    const uint64_t gt_qtr_mask = 0ull - gt_qtr_bit;
    const double f_qtr = 0.5 - f;
    f = sycl::bit_cast<double>(
        (sycl::bit_cast<uint64_t>(f_qtr) & gt_qtr_mask) |
        (sycl::bit_cast<uint64_t>(f) & ~gt_qtr_mask));

    const uint64_t gt_oct_bit = static_cast<uint64_t>(f > 0.125);
    const uint64_t gt_oct_mask = 0ull - gt_oct_bit;
    const double f_oct = 0.25 - f;
    f = sycl::bit_cast<double>(
        (sycl::bit_cast<uint64_t>(f_oct) & gt_oct_mask) |
        (sycl::bit_cast<uint64_t>(f) & ~gt_oct_mask));

    double c_oct, s_oct;
    sincos2pi_oct1<1>(f, c_oct, s_oct);

    const uint64_t c_oct_u = sycl::bit_cast<uint64_t>(c_oct);
    const uint64_t s_oct_u = sycl::bit_cast<uint64_t>(s_oct);
    const uint64_t c_q1_u = (s_oct_u & gt_oct_mask) | (c_oct_u & ~gt_oct_mask);
    const uint64_t s_q1_u = (c_oct_u & gt_oct_mask) | (s_oct_u & ~gt_oct_mask);

    c_out = sycl::bit_cast<double>(c_q1_u ^ (gt_qtr_bit << 63));
    const uint64_t s_flip = static_cast<uint64_t>(Sign < 0) << 63;
    s_out = sycl::bit_cast<double>(s_q1_u ^ s_flip);
}

template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pi_branchless(double x, double& c_out, double& s_out) {
    double f = x - sycl::floor(x);
    const uint64_t ge_half_bit = static_cast<uint64_t>(f >= 0.5);
    const uint64_t ge_half_mask = 0ull - ge_half_bit;
    f -= sycl::bit_cast<double>(sycl::bit_cast<uint64_t>(0.5) & ge_half_mask);

    const uint64_t gt_qtr_bit = static_cast<uint64_t>(f > 0.25);
    const uint64_t gt_qtr_mask = 0ull - gt_qtr_bit;
    const double f_qtr = 0.5 - f;
    f = sycl::bit_cast<double>(
        (sycl::bit_cast<uint64_t>(f_qtr) & gt_qtr_mask) |
        (sycl::bit_cast<uint64_t>(f) & ~gt_qtr_mask));

    const uint64_t gt_oct_bit = static_cast<uint64_t>(f > 0.125);
    const uint64_t gt_oct_mask = 0ull - gt_oct_bit;
    const double f_oct = 0.25 - f;
    f = sycl::bit_cast<double>(
        (sycl::bit_cast<uint64_t>(f_oct) & gt_oct_mask) |
        (sycl::bit_cast<uint64_t>(f) & ~gt_oct_mask));

    double c_oct, s_oct;
    sincos2pi_oct1<1>(f, c_oct, s_oct);

    const uint64_t c_oct_u = sycl::bit_cast<uint64_t>(c_oct);
    const uint64_t s_oct_u = sycl::bit_cast<uint64_t>(s_oct);
    const uint64_t c_q1_u = (s_oct_u & gt_oct_mask) | (c_oct_u & ~gt_oct_mask);
    const uint64_t s_q1_u = (c_oct_u & gt_oct_mask) | (s_oct_u & ~gt_oct_mask);

    const uint64_t c_flip = (ge_half_bit ^ gt_qtr_bit) << 63;
    const uint64_t s_flip = (ge_half_bit ^ static_cast<uint64_t>(Sign < 0)) << 63;
    c_out = sycl::bit_cast<double>(c_q1_u ^ c_flip);
    s_out = sycl::bit_cast<double>(s_q1_u ^ s_flip);
}

SYCL_FFT_FORCE_INLINE double cos2pi_branchless(double x) {
    double c, s;
    sincos2pi_branchless<1>(x, c, s);
    return c;
}

SYCL_FFT_FORCE_INLINE double sin2pi_branchless(double x) {
    double c, s;
    sincos2pi_branchless<1>(x, c, s);
    return s;
}

SYCL_FFT_FORCE_INLINE double cos2pi(double x) { return cos2pi_branchless(x); }
SYCL_FFT_FORCE_INLINE double sin2pi(double x) { return sin2pi_branchless(x); }

// ============================================================================
// Double-Float (DF) Branchless Polynomial Trig (df suffix, minimax on [0, 0.125])
//   Truncated 2 degrees relative to FP64:
//   sin: degree 11 (6 terms in f^2), max err <= 5.77e-15 (~47.3 bits)
//   cos: degree 12 (7 terms in f^2), max err <= 3.22e-15 (~48.1 bits)
//   Hybrid FP32-top + 5 DF Horner steps for minimum FP32 op count!
// ============================================================================
namespace detail {

constexpr df_t kSinDf0 = df_from_double( 6.28318530717955959e+00);
constexpr df_t kSinDf1 = df_from_double(-4.13417022402757794e+01);
constexpr df_t kSinDf2 = df_from_double( 8.16052491834839344e+01);
constexpr df_t kSinDf3 = df_from_double(-7.67058344584512923e+01);
constexpr df_t kSinDf4 = df_from_double( 4.20555692559449881e+01);
constexpr float kSinDf5 = static_cast<float>(-1.49165651188191184e+01);

constexpr df_t kCosDf0 = df_from_double( 1.00000000000000000e+00);
constexpr df_t kCosDf1 = df_from_double(-1.97392088021784211e+01);
constexpr df_t kCosDf2 = df_from_double( 6.49393940223653772e+01);
constexpr df_t kCosDf3 = df_from_double(-8.54568170903411470e+01);
constexpr df_t kCosDf4 = df_from_double( 6.02446200859302934e+01);
constexpr float kCosDf5 = static_cast<float>(-2.64242570348826895e+01);
constexpr float kCosDf6 = static_cast<float>( 7.81029096814419077e+00);

} // namespace detail

// Fast path for exact dyadic float angles f = j / 2^k in [0, 0.125] (lo == 0)
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pidf_oct1_f(float f, df_t& c_out, df_t& s_out) {
    df_t f2;
    df_two_product(f, f, f2.hi, f2.lo);

    // sin(2*pi*f) = f * P_sin(f^2): top coeff in FP32, 5 DF Horner steps, then * f
    df_t ps = df_add(df_mul_f(f2, detail::kSinDf5), detail::kSinDf4);
    ps = df_add(df_mul(ps, f2), detail::kSinDf3);
    ps = df_add(df_mul(ps, f2), detail::kSinDf2);
    ps = df_add(df_mul(ps, f2), detail::kSinDf1);
    ps = df_add(df_mul(ps, f2), detail::kSinDf0);
    const df_t s_mag = df_mul_f(ps, f);

    // cos(2*pi*f) = P_cos(f^2): top 2 coeffs in FP32 FMA, 5 DF Horner steps
    const float pc_top = sycl::fma(detail::kCosDf6, f2.hi, detail::kCosDf5);
    df_t pc = df_add(df_mul_f(f2, pc_top), detail::kCosDf4);
    pc = df_add(df_mul(pc, f2), detail::kCosDf3);
    pc = df_add(df_mul(pc, f2), detail::kCosDf2);
    pc = df_add(df_mul(pc, f2), detail::kCosDf1);
    pc = df_add(df_mul(pc, f2), detail::kCosDf0);

    c_out = pc;
    s_out = (Sign < 0) ? df_neg(s_mag) : s_mag;
}

// Fast path for exact dyadic float angles f = j / 2^k in [0, 0.5)
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pidf_half_f(float f, df_t& c_out, df_t& s_out) {
    // Exact float quadrant & octant folds (0.5f - f and 0.25f - f are exact for dyadic floats)
    const uint32_t gt_qtr_bit = static_cast<uint32_t>(f > 0.25f);
    const uint32_t gt_qtr_mask = 0u - gt_qtr_bit;
    const float f_qtr = 0.5f - f;
    f = sycl::bit_cast<float>(
        (sycl::bit_cast<uint32_t>(f_qtr) & gt_qtr_mask) |
        (sycl::bit_cast<uint32_t>(f) & ~gt_qtr_mask));

    const uint32_t gt_oct_bit = static_cast<uint32_t>(f > 0.125f);
    const uint32_t gt_oct_mask = 0u - gt_oct_bit;
    const float f_oct = 0.25f - f;
    f = sycl::bit_cast<float>(
        (sycl::bit_cast<uint32_t>(f_oct) & gt_oct_mask) |
        (sycl::bit_cast<uint32_t>(f) & ~gt_oct_mask));

    df_t c_oct, s_oct;
    sincos2pidf_oct1_f<1>(f, c_oct, s_oct);

    const df_t c_q1 = df_select(s_oct, c_oct, gt_oct_mask);
    const df_t s_q1 = df_select(c_oct, s_oct, gt_oct_mask);

    c_out = df_xor_sign(c_q1, gt_qtr_bit << 31);
    s_out = (Sign < 0) ? df_neg(s_q1) : s_q1;
}

// General df_t input on [0, 0.125]
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pidf_oct1(df_t f, df_t& c_out, df_t& s_out) {
    const df_t f2 = df_sqr(f);

    df_t ps = df_add(df_mul_f(f2, detail::kSinDf5), detail::kSinDf4);
    ps = df_add(df_mul(ps, f2), detail::kSinDf3);
    ps = df_add(df_mul(ps, f2), detail::kSinDf2);
    ps = df_add(df_mul(ps, f2), detail::kSinDf1);
    ps = df_add(df_mul(ps, f2), detail::kSinDf0);
    const df_t s_mag = df_mul(ps, f);

    const float pc_top = sycl::fma(detail::kCosDf6, f2.hi, detail::kCosDf5);
    df_t pc = df_add(df_mul_f(f2, pc_top), detail::kCosDf4);
    pc = df_add(df_mul(pc, f2), detail::kCosDf3);
    pc = df_add(df_mul(pc, f2), detail::kCosDf2);
    pc = df_add(df_mul(pc, f2), detail::kCosDf1);
    pc = df_add(df_mul(pc, f2), detail::kCosDf0);

    c_out = pc;
    s_out = (Sign < 0) ? df_neg(s_mag) : s_mag;
}

// General branchless df_t sincos2pi for arbitrary real df_t x
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pidf_branchless(df_t x, df_t& c_out, df_t& s_out) {
    const float k = sycl::floor(x.hi);
    df_t f = df_sub(x, df_make(k, 0.0f));
    // Handle boundary where x.hi is integer and x.lo < 0 (or rounds to 1.0)
    const uint32_t lt_zero_mask = 0u - static_cast<uint32_t>(f.hi < 0.0f || (f.hi == 0.0f && f.lo < 0.0f));
    f = df_add(f, df_select(df_make(1.0f, 0.0f), df_make(0.0f, 0.0f), lt_zero_mask));
    const uint32_t ge_one_mask = 0u - static_cast<uint32_t>(f.hi > 1.0f || (f.hi == 1.0f && f.lo >= 0.0f));
    f = df_sub(f, df_select(df_make(1.0f, 0.0f), df_make(0.0f, 0.0f), ge_one_mask));

    const uint32_t ge_half_bit = static_cast<uint32_t>(f.hi > 0.5f || (f.hi == 0.5f && f.lo >= 0.0f));
    const uint32_t ge_half_mask = 0u - ge_half_bit;
    f = df_sub(f, df_select(df_make(0.5f, 0.0f), df_make(0.0f, 0.0f), ge_half_mask));

    const uint32_t gt_qtr_bit = static_cast<uint32_t>(f.hi > 0.25f || (f.hi == 0.25f && f.lo > 0.0f));
    const uint32_t gt_qtr_mask = 0u - gt_qtr_bit;
    f = df_select(df_sub(df_make(0.5f, 0.0f), f), f, gt_qtr_mask);

    const uint32_t gt_oct_bit = static_cast<uint32_t>(f.hi > 0.125f || (f.hi == 0.125f && f.lo > 0.0f));
    const uint32_t gt_oct_mask = 0u - gt_oct_bit;
    f = df_select(df_sub(df_make(0.25f, 0.0f), f), f, gt_oct_mask);

    df_t c_oct, s_oct;
    sincos2pidf_oct1<1>(f, c_oct, s_oct);

    const df_t c_q1 = df_select(s_oct, c_oct, gt_oct_mask);
    const df_t s_q1 = df_select(c_oct, s_oct, gt_oct_mask);

    const uint32_t c_flip = (ge_half_bit ^ gt_qtr_bit) << 31;
    const uint32_t s_flip = (ge_half_bit ^ static_cast<uint32_t>(Sign < 0)) << 31;
    c_out = df_xor_sign(c_q1, c_flip);
    s_out = df_xor_sign(s_q1, s_flip);
}

SYCL_FFT_FORCE_INLINE df_t cos2pidf_branchless(df_t x) {
    df_t c, s;
    sincos2pidf_branchless<1>(x, c, s);
    return c;
}

SYCL_FFT_FORCE_INLINE df_t sin2pidf_branchless(df_t x) {
    df_t c, s;
    sincos2pidf_branchless<1>(x, c, s);
    return s;
}

SYCL_FFT_FORCE_INLINE df_t cos2pidf(df_t x) { return cos2pidf_branchless(x); }
SYCL_FFT_FORCE_INLINE df_t sin2pidf(df_t x) { return sin2pidf_branchless(x); }

} // namespace sycl_fft
