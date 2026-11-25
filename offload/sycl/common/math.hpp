#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <vector>

namespace sycl_fft {

#if defined(__GNUC__) || defined(__clang__)
#define SYCL_FFT_FORCE_INLINE __attribute__((always_inline)) inline
#define SYCL_FFT_LAMBDA_INLINE __attribute__((always_inline))
#else
#define SYCL_FFT_FORCE_INLINE inline
#define SYCL_FFT_LAMBDA_INLINE
#endif

// Under AdaptiveCpp SSCP, bitcode is extracted before mem2reg/SROA after AlwaysInliner.
// Keeping multi-op DF arithmetic helpers plain `inline` prevents pre-mem2reg alloca/IR
// explosion in HCF while LLVM's JIT -O3 still inlines them completely into target PTX/ISA.
#if defined(SYCL_IMPLEMENTATION_ACPP) || defined(SYCL_IMPLEMENTATION_HIPSYCL)
#define SYCL_FFT_DF_INLINE inline
#else
#define SYCL_FFT_DF_INLINE SYCL_FFT_FORCE_INLINE
#endif

#if defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
#define SYCL_FFT_BIT_CAST(Dst, src) __builtin_bit_cast(Dst, (src))
#else
#define SYCL_FFT_BIT_CAST(Dst, src) ::sycl::bit_cast<Dst>(src)
#endif
#else
#define SYCL_FFT_BIT_CAST(Dst, src) ::sycl::bit_cast<Dst>(src)
#endif

template <typename Dst, typename Src>
SYCL_FFT_FORCE_INLINE constexpr Dst sycl_fft_bit_cast(const Src& src) noexcept {
    return SYCL_FFT_BIT_CAST(Dst, src);
}

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
// Note: Under AdaptiveCpp SSCP (--acpp-targets=generic), __CUDA_ARCH__ is not defined
// in the frontend, so SYCL_FFT_DF_EXACT_FP (clang fp contract(off) + reassociate(off))
// emits non-contracted LLVM IR fadd/fsub/fmul which NVPTX JIT lowers to *.rn.f32.
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
// Quadruple-Precision Complex Types
//   FP16: Complexh  (sycl::half2,    4 B, paired 2x FP16 compute via f16x2_t)
//   FP32: Complexf  (sycl::float2,   8 B)
//   FP64: Complex   (sycl::double2, 16 B)
//   DF:   Complexdf ({df_t re, im}, 16 B, ~47-48 bit mantissa via 2 x fp32)
// ============================================================================
using Complexh = sycl::half2;
using Complexf = sycl::float2;
using Complex  = sycl::double2;

static_assert(sizeof(Complexh) == 4 && alignof(Complexh) == 4 && std::is_trivially_copyable_v<Complexh>,
              "Complexh must be 4-byte aligned trivially copyable pair of binary16");

#if !defined(SYCL_FFT_DISABLE_NATIVE_F16X2) && defined(__clang__) && defined(__FLT16_MANT_DIG__) && \
    (defined(SYCL_IMPLEMENTATION_ACPP) || defined(SYCL_IMPLEMENTATION_HIPSYCL))
#define SYCL_FFT_USE_CLANG_F16X2 1
#else
#define SYCL_FFT_USE_CLANG_F16X2 0
#endif

#if SYCL_FFT_USE_CLANG_F16X2
using f16_scalar_t = _Float16;
using f16x2_t = _Float16 __attribute__((ext_vector_type(2)));
#else
using f16_scalar_t = sycl::half;
using f16x2_t = sycl::half2;
#endif

static_assert(sizeof(f16x2_t) == 4 && alignof(f16x2_t) == 4 && std::is_trivially_copyable_v<f16x2_t>,
              "f16x2_t must be 4-byte trivially copyable for 32-bit SMEM & shuffle");

SYCL_FFT_FORCE_INLINE constexpr uint32_t f16x2_to_u32(f16x2_t v) {
    return SYCL_FFT_BIT_CAST(uint32_t, v);
}

SYCL_FFT_FORCE_INLINE constexpr f16x2_t u32_to_f16x2(uint32_t u) {
    return SYCL_FFT_BIT_CAST(f16x2_t, u);
}

SYCL_FFT_FORCE_INLINE constexpr f16x2_t f16x2_make(f16_scalar_t lo, f16_scalar_t hi) {
#if SYCL_FFT_USE_CLANG_F16X2
    return f16x2_t{lo, hi};
#else
    sycl::half2 v{};
    v.x() = lo;
    v.y() = hi;
    return v;
#endif
}

SYCL_FFT_FORCE_INLINE constexpr f16x2_t f16x2_splat(f16_scalar_t v) {
    return f16x2_make(v, v);
}

SYCL_FFT_FORCE_INLINE constexpr f16x2_t f16x2_splat_f(float v) {
    const f16_scalar_t h = static_cast<f16_scalar_t>(v);
    return f16x2_make(h, h);
}

#if SYCL_FFT_USE_CLANG_F16X2
#define SYCL_FFT_F16X2_SPLAT_H(h) (f16x2_t{(h), (h)})
#define SYCL_FFT_F16X2_SPLAT_F(v) \
    (f16x2_t{static_cast<f16_scalar_t>(v), static_cast<f16_scalar_t>(v)})
#if defined(__has_builtin)
#if __has_builtin(__builtin_elementwise_fma)
#define SYCL_FFT_F16X2_FMA(a, b, c) __builtin_elementwise_fma((a), (b), (c))
#else
#define SYCL_FFT_F16X2_FMA(a, b, c) ((a) * (b) + (c))
#endif
#else
#define SYCL_FFT_F16X2_FMA(a, b, c) ((a) * (b) + (c))
#endif
#else
#define SYCL_FFT_F16X2_SPLAT_H(h) f16x2_splat((h))
#define SYCL_FFT_F16X2_SPLAT_F(v) f16x2_splat_f((v))
#define SYCL_FFT_F16X2_FMA(a, b, c) sycl::fma((a), (b), (c))
#endif

constexpr f16x2_t kZero_f16x2    = f16x2_splat_f(0.0f);
constexpr f16x2_t kOne_f16x2     = f16x2_splat_f(1.0f);
constexpr f16x2_t kHalf_f16x2    = f16x2_splat_f(0.5f);
constexpr f16x2_t kTwo_f16x2     = f16x2_splat_f(2.0f);
constexpr f16x2_t kSqrt1_2_f16x2 = f16x2_splat_f(0.7071067811865475244f);
SYCL_FFT_FORCE_INLINE f16x2_t f16x2_add(f16x2_t a, f16x2_t b) { return a + b; }
SYCL_FFT_FORCE_INLINE f16x2_t f16x2_sub(f16x2_t a, f16x2_t b) { return a - b; }
SYCL_FFT_FORCE_INLINE f16x2_t f16x2_neg(f16x2_t a) { return -a; }
SYCL_FFT_FORCE_INLINE f16x2_t f16x2_mul(f16x2_t a, f16x2_t b) { return a * b; }

SYCL_FFT_FORCE_INLINE f16x2_t f16x2_fma(f16x2_t a, f16x2_t b, f16x2_t c) {
#if SYCL_FFT_USE_CLANG_F16X2
#if defined(__has_builtin)
#if __has_builtin(__builtin_elementwise_fma)
    return __builtin_elementwise_fma(a, b, c);
#else
    return a * b + c;
#endif
#else
    return a * b + c;
#endif
#else
    return sycl::fma(a, b, c);
#endif
}

SYCL_FFT_FORCE_INLINE f16x2_t f16x2_scale_half(f16x2_t a) {
    return a * kHalf_f16x2;
}

SYCL_FFT_FORCE_INLINE f16x2_t f16x2_cond_neg(f16x2_t v, bool cond) {
    const uint32_t mask = cond ? 0x80008000u : 0u;
    return SYCL_FFT_BIT_CAST(f16x2_t, SYCL_FFT_BIT_CAST(uint32_t, v) ^ mask);
}

SYCL_FFT_FORCE_INLINE void f16x2_cmul(
    f16x2_t ar, f16x2_t ai, f16x2_t br, f16x2_t bi, f16x2_t& or_, f16x2_t& oi) {
    or_ = f16x2_fma(ar, br, -(ai * bi));
    oi  = f16x2_fma(ar, bi,   ai * br);
}

SYCL_FFT_FORCE_INLINE void f16x2_csqr(
    f16x2_t ar, f16x2_t ai, f16x2_t& or_, f16x2_t& oi) {
    or_ = f16x2_fma(ar, ar, -(ai * ai));
    oi  = (kTwo_f16x2 * ar) * ai;
}

template <bool NegRe>
SYCL_FFT_FORCE_INLINE void f16x2_cmul_sqrt1_2_rt(
    f16x2_t dr, f16x2_t di, bool pos_sign, f16x2_t& or_, f16x2_t& oi) {
    const uint32_t mask = pos_sign ? 0x80008000u : 0u;
    const f16x2_t s_dr = SYCL_FFT_BIT_CAST(f16x2_t, SYCL_FFT_BIT_CAST(uint32_t, dr) ^ mask);
    const f16x2_t s_di = SYCL_FFT_BIT_CAST(f16x2_t, SYCL_FFT_BIT_CAST(uint32_t, di) ^ mask);
    if constexpr (!NegRe) {
        or_ = (dr + s_di) * kSqrt1_2_f16x2;
        oi  = (di - s_dr) * kSqrt1_2_f16x2;
    } else {
        or_ = (s_di - dr) * kSqrt1_2_f16x2;
        oi  = -(s_dr + di) * kSqrt1_2_f16x2;
    }
}

template <bool NegRe, bool NegIm>
SYCL_FFT_FORCE_INLINE void f16x2_cmul_sqrt1_2(
    f16x2_t dr, f16x2_t di, f16x2_t& or_, f16x2_t& oi) {
    f16x2_cmul_sqrt1_2_rt<NegRe>(dr, di, !NegIm, or_, oi);
}

// Pack 2 independent complex binary16 values (batch 0 and batch 1) into SoA f16x2_t registers:
//   re = {g0.re, g1.re}, im = {g0.im, g1.im}.
SYCL_FFT_FORCE_INLINE void f16x2_load_pair(
    const Complexh* g0, const Complexh* g1, int idx, bool valid0, bool valid1,
    f16x2_t& re_out, f16x2_t& im_out) {
    const uint32_t u0 = valid0 ? reinterpret_cast<const uint32_t*>(g0)[idx] : 0u;
    const uint32_t u1 = valid1 ? reinterpret_cast<const uint32_t*>(g1)[idx] : 0u;
    const uint32_t re_u32 = (u0 & 0xFFFFu) | ((u1 & 0xFFFFu) << 16);
    const uint32_t im_u32 = (u0 >> 16) | (u1 & 0xFFFF0000u);
    re_out = SYCL_FFT_BIT_CAST(f16x2_t, re_u32);
    im_out = SYCL_FFT_BIT_CAST(f16x2_t, im_u32);
}

SYCL_FFT_FORCE_INLINE void f16x2_store_pair(
    Complexh* g0, Complexh* g1, int idx, bool valid0, bool valid1,
    f16x2_t re_in, f16x2_t im_in) {
    const uint32_t re_u32 = SYCL_FFT_BIT_CAST(uint32_t, re_in);
    const uint32_t im_u32 = SYCL_FFT_BIT_CAST(uint32_t, im_in);
    if (valid0) {
        reinterpret_cast<uint32_t*>(g0)[idx] = (re_u32 & 0xFFFFu) | ((im_u32 & 0xFFFFu) << 16);
    }
    if (valid1) {
        reinterpret_cast<uint32_t*>(g1)[idx] = (re_u32 >> 16) | (im_u32 & 0xFFFF0000u);
    }
}

SYCL_FFT_FORCE_INLINE Complexh complexh_from_doubles(double r, double i) {
    Complexh out{};
    out.x() = static_cast<sycl::half>(static_cast<float>(r));
    out.y() = static_cast<sycl::half>(static_cast<float>(i));
    return out;
}
SYCL_FFT_FORCE_INLINE void complexh_to_long_doubles(Complexh v, long double& r, long double& i) {
    r = static_cast<long double>(static_cast<float>(v.x()));
    i = static_cast<long double>(static_cast<float>(v.y()));
}
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
    SYCL_FFT_DF_EXACT_FP
    return df_t{a.hi * s, a.lo * s};
}

SYCL_FFT_FORCE_INLINE df_t df_select(df_t a, df_t b, uint32_t mask) {
    const uint32_t hi = (SYCL_FFT_BIT_CAST(uint32_t, a.hi) & mask) |
                        (SYCL_FFT_BIT_CAST(uint32_t, b.hi) & ~mask);
    const uint32_t lo = (SYCL_FFT_BIT_CAST(uint32_t, a.lo) & mask) |
                        (SYCL_FFT_BIT_CAST(uint32_t, b.lo) & ~mask);
    return df_t{SYCL_FFT_BIT_CAST(float, hi), SYCL_FFT_BIT_CAST(float, lo)};
}

SYCL_FFT_FORCE_INLINE df_t df_xor_sign(df_t a, uint32_t sign_bit) {
    return df_t{
        SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, a.hi) ^ sign_bit),
        SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, a.lo) ^ sign_bit)
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
SYCL_FFT_DF_INLINE df_t df_add(df_t a, df_t b) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_sum(a.hi, b.hi, r, e);
    e = df_fadd_rn(df_fadd_rn(e, a.lo), b.lo);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// DF - DF (11 FP32 ops)
SYCL_FFT_DF_INLINE df_t df_sub(df_t a, df_t b) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_difference(a.hi, b.hi, r, e);
    e = df_fsub_rn(df_fadd_rn(e, a.lo), b.lo);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// DF * DF (7 FP32 ops)
SYCL_FFT_DF_INLINE df_t df_mul(df_t a, df_t b) {
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
SYCL_FFT_DF_INLINE df_t df_mul_f(df_t a, float b) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_product(a.hi, b, r, e);
    e = sycl::fma(a.lo, b, e);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// DF * DF squaring (6 FP32 ops)
SYCL_FFT_DF_INLINE df_t df_sqr(df_t a) {
    SYCL_FFT_DF_EXACT_FP
    float r, e;
    df_two_product(a.hi, a.hi, r, e);
    e = sycl::fma(df_fadd_rn(a.hi, a.hi), a.lo, e);
    df_t out;
    df_two_sum_quick(r, e, out.hi, out.lo);
    return out;
}

// DF / DF division (~48-bit accuracy via 2-step Newton/remainder refinement)
SYCL_FFT_DF_INLINE df_t df_div(df_t a, df_t b) {
    SYCL_FFT_DF_EXACT_FP
    const float q1 = a.hi / b.hi;
    const df_t r1 = df_sub(a, df_mul_f(b, q1));
    const float q2 = r1.hi / b.hi;
    const df_t r2 = df_sub(r1, df_mul_f(b, q2));
    const float q3 = r2.hi / b.hi;
    df_t out;
    df_two_sum_quick(q1, q2, out.hi, out.lo);
    return df_add(out, df_make(q3, 0.0f));
}

SYCL_FFT_FORCE_INLINE df_t df_abs(df_t a) {
    return (a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f)) ? df_neg(a) : a;
}

SYCL_FFT_DF_INLINE df_t df_floor(df_t a) {
    SYCL_FFT_DF_EXACT_FP
    const float f_hi = sycl::floor(a.hi);
    const float f_lo = (f_hi == a.hi) ? sycl::floor(a.lo) : 0.0f;
    df_t out;
    df_two_sum(f_hi, f_lo, out.hi, out.lo);
    return out;
}

SYCL_FFT_DF_INLINE df_t df_ceil(df_t a) {
    SYCL_FFT_DF_EXACT_FP
    const float c_hi = sycl::ceil(a.hi);
    const float c_lo = (c_hi == a.hi) ? sycl::ceil(a.lo) : 0.0f;
    df_t out;
    df_two_sum(c_hi, c_lo, out.hi, out.lo);
    return out;
}

// Fused DF complex multiply (38 FP32 ops vs 50 unfused):
// Feeds un-renormalized two-product pairs directly into two_diff/two_sum + quick_two_sum.
SYCL_FFT_DF_INLINE void df_cmul_fused(
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
SYCL_FFT_DF_INLINE void df_csqr_fused(df_t ar, df_t ai, df_t& out_r, df_t& out_i) {
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
// Common Irrational Constants
// ============================================================================
constexpr float kSqrt1_2f = 0.70710678118654752440f;
constexpr double kSqrt1_2 = 0.70710678118654752440;
constexpr df_t kSqrt1_2df = df_from_double(kSqrt1_2);

// Specialized W_8^{1,3} rotation with runtime sign:
//   NegRe=false: W_8^{\pm 1} = (1 \pm i)/sqrt(2)
//   NegRe=true:  W_8^{\pm 3} = (-1 \pm i)/sqrt(2)
template <bool NegRe>
SYCL_FFT_DF_INLINE void df_cmul_sqrt1_2_rt(
    df_t dr, df_t di, bool pos_sign, df_t& out_r, df_t& out_i) {
    const uint32_t sbit = static_cast<uint32_t>(pos_sign) << 31;
    const df_t s_dr = df_xor_sign(dr, sbit);
    const df_t s_di = df_xor_sign(di, sbit);
    df_t tr, ti;
    if constexpr (!NegRe) {
        tr = df_add(dr, s_di);
        ti = df_sub(di, s_dr);
    } else {
        tr = df_sub(s_di, dr);
        ti = df_neg(df_add(s_dr, di));
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
    f = SYCL_FFT_BIT_CAST(float,
        (SYCL_FFT_BIT_CAST(uint32_t, f_fold) & gt_qtr_mask) |
        (SYCL_FFT_BIT_CAST(uint32_t, f) & ~gt_qtr_mask));

    float c_mag, s_signed;
    sincos2pif_q1<Sign>(f, c_mag, s_signed);
    c_out = SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, c_mag) ^ (gt_qtr_bit << 31));
    s_out = s_signed;
}

template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pif_branchless(float x, float& c_out, float& s_out) {
    float f = x - sycl::floor(x);
    const uint32_t ge_half_bit = static_cast<uint32_t>(f >= 0.5f);
    const uint32_t ge_half_mask = 0u - ge_half_bit;
    f -= SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, 0.5f) & ge_half_mask);

    const uint32_t gt_qtr_bit = static_cast<uint32_t>(f > 0.25f);
    const uint32_t gt_qtr_mask = 0u - gt_qtr_bit;
    const float f_fold = 0.5f - f;
    f = SYCL_FFT_BIT_CAST(float,
        (SYCL_FFT_BIT_CAST(uint32_t, f_fold) & gt_qtr_mask) |
        (SYCL_FFT_BIT_CAST(uint32_t, f) & ~gt_qtr_mask));

    float c_mag, s_base;
    sincos2pif_q1<1>(f, c_mag, s_base);
    const uint32_t s_flip = (ge_half_bit ^ static_cast<uint32_t>(Sign < 0)) << 31;
    const uint32_t c_flip = (ge_half_bit ^ gt_qtr_bit) << 31;
    c_out = SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, c_mag) ^ c_flip);
    s_out = SYCL_FFT_BIT_CAST(float, SYCL_FFT_BIT_CAST(uint32_t, s_base) ^ s_flip);
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
// FP16 Standalone Branchless Trig (evaluated in FP32, rounded to binary16 <= 0.5 ULP)
// ============================================================================
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pih_branchless(sycl::half x, sycl::half& c_out, sycl::half& s_out) {
    float c = 0.0f, s = 0.0f;
    sincos2pif_branchless<Sign>(static_cast<float>(x), c, s);
    c_out = static_cast<sycl::half>(c);
    s_out = static_cast<sycl::half>(s);
}

SYCL_FFT_FORCE_INLINE sycl::half cos2pih(sycl::half x) {
    return static_cast<sycl::half>(cos2pif_branchless(static_cast<float>(x)));
}

SYCL_FFT_FORCE_INLINE sycl::half sin2pih(sycl::half x) {
    return static_cast<sycl::half>(sin2pif_branchless(static_cast<float>(x)));
}

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
    f = SYCL_FFT_BIT_CAST(double,
        (SYCL_FFT_BIT_CAST(uint64_t, f_qtr) & gt_qtr_mask) |
        (SYCL_FFT_BIT_CAST(uint64_t, f) & ~gt_qtr_mask));

    const uint64_t gt_oct_bit = static_cast<uint64_t>(f > 0.125);
    const uint64_t gt_oct_mask = 0ull - gt_oct_bit;
    const double f_oct = 0.25 - f;
    f = SYCL_FFT_BIT_CAST(double,
        (SYCL_FFT_BIT_CAST(uint64_t, f_oct) & gt_oct_mask) |
        (SYCL_FFT_BIT_CAST(uint64_t, f) & ~gt_oct_mask));

    double c_oct, s_oct;
    sincos2pi_oct1<1>(f, c_oct, s_oct);

    const uint64_t c_oct_u = SYCL_FFT_BIT_CAST(uint64_t, c_oct);
    const uint64_t s_oct_u = SYCL_FFT_BIT_CAST(uint64_t, s_oct);
    const uint64_t c_q1_u = (s_oct_u & gt_oct_mask) | (c_oct_u & ~gt_oct_mask);
    const uint64_t s_q1_u = (c_oct_u & gt_oct_mask) | (s_oct_u & ~gt_oct_mask);

    c_out = SYCL_FFT_BIT_CAST(double, c_q1_u ^ (gt_qtr_bit << 63));
    const uint64_t s_flip = static_cast<uint64_t>(Sign < 0) << 63;
    s_out = SYCL_FFT_BIT_CAST(double, s_q1_u ^ s_flip);
}

template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pi_branchless(double x, double& c_out, double& s_out) {
    double f = x - sycl::floor(x);
    const uint64_t ge_half_bit = static_cast<uint64_t>(f >= 0.5);
    const uint64_t ge_half_mask = 0ull - ge_half_bit;
    f -= SYCL_FFT_BIT_CAST(double, SYCL_FFT_BIT_CAST(uint64_t, 0.5) & ge_half_mask);

    const uint64_t gt_qtr_bit = static_cast<uint64_t>(f > 0.25);
    const uint64_t gt_qtr_mask = 0ull - gt_qtr_bit;
    const double f_qtr = 0.5 - f;
    f = SYCL_FFT_BIT_CAST(double,
        (SYCL_FFT_BIT_CAST(uint64_t, f_qtr) & gt_qtr_mask) |
        (SYCL_FFT_BIT_CAST(uint64_t, f) & ~gt_qtr_mask));

    const uint64_t gt_oct_bit = static_cast<uint64_t>(f > 0.125);
    const uint64_t gt_oct_mask = 0ull - gt_oct_bit;
    const double f_oct = 0.25 - f;
    f = SYCL_FFT_BIT_CAST(double,
        (SYCL_FFT_BIT_CAST(uint64_t, f_oct) & gt_oct_mask) |
        (SYCL_FFT_BIT_CAST(uint64_t, f) & ~gt_oct_mask));

    double c_oct, s_oct;
    sincos2pi_oct1<1>(f, c_oct, s_oct);

    const uint64_t c_oct_u = SYCL_FFT_BIT_CAST(uint64_t, c_oct);
    const uint64_t s_oct_u = SYCL_FFT_BIT_CAST(uint64_t, s_oct);
    const uint64_t c_q1_u = (s_oct_u & gt_oct_mask) | (c_oct_u & ~gt_oct_mask);
    const uint64_t s_q1_u = (c_oct_u & gt_oct_mask) | (s_oct_u & ~gt_oct_mask);

    const uint64_t c_flip = (ge_half_bit ^ gt_qtr_bit) << 63;
    const uint64_t s_flip = (ge_half_bit ^ static_cast<uint64_t>(Sign < 0)) << 63;
    c_out = SYCL_FFT_BIT_CAST(double, c_q1_u ^ c_flip);
    s_out = SYCL_FFT_BIT_CAST(double, s_q1_u ^ s_flip);
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
    f = SYCL_FFT_BIT_CAST(float,
        (SYCL_FFT_BIT_CAST(uint32_t, f_qtr) & gt_qtr_mask) |
        (SYCL_FFT_BIT_CAST(uint32_t, f) & ~gt_qtr_mask));

    const uint32_t gt_oct_bit = static_cast<uint32_t>(f > 0.125f);
    const uint32_t gt_oct_mask = 0u - gt_oct_bit;
    const float f_oct = 0.25f - f;
    f = SYCL_FFT_BIT_CAST(float,
        (SYCL_FFT_BIT_CAST(uint32_t, f_oct) & gt_oct_mask) |
        (SYCL_FFT_BIT_CAST(uint32_t, f) & ~gt_oct_mask));

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

// General branchless df_t sincos2pi for arbitrary real df_t x (even when |x| >= 2^24)
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pidf_branchless(df_t x, df_t& c_out, df_t& s_out) {
    df_t f = df_sub(x, df_floor(x));
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
