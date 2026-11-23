#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sycl_fft {

using Complex = sycl::float2;

constexpr float kTwoPi = 6.28318530717958647692f;

#if defined(__GNUC__) || defined(__clang__)
#define SYCL_FFT_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define SYCL_FFT_FORCE_INLINE inline
#endif

// Compile-time unroll helper: guarantees every index I is an integral_constant,
// preventing register-array spilling to local memory across all SYCL backends.
template <int N, typename Fn, int... Is>
SYCL_FFT_FORCE_INLINE void unroll_for_impl(Fn&& fn, std::integer_sequence<int, Is...>) {
    (fn(std::integral_constant<int, Is>{}), ...);
}

template <int N, typename Fn>
SYCL_FFT_FORCE_INLINE void unroll_for(Fn&& fn) {
    unroll_for_impl<N>(std::forward<Fn>(fn), std::make_integer_sequence<int, N>{});
}

// Compile-time bit-reverse of lowest `Bits` bits of `x`.
constexpr uint32_t bit_reverse_const(uint32_t x, int bits) {
    uint32_t r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | ((x >> i) & 1u);
    }
    return r;
}

// Fast runtime bit-reverse of lowest `bits` bits of `x`.
SYCL_FFT_FORCE_INLINE uint32_t bit_reverse(uint32_t x, int bits) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    x = (x << 16) | (x >> 16);
    return x >> (32 - bits);
}

// Exact compile-time cos/sin table for W_32^k = exp(-i 2pi k / 32), k = 0..15.
// For Sign = +1, negate imaginary part. Any W_{2^p}^m with 2^p <= 32 maps to k = m * (32 / 2^p).
constexpr float kCos32[16] = {
     1.0000000000000000f,  0.9807852804032304f,  0.9238795325112867f,  0.8314696123025452f,
     0.7071067811865476f,  0.5555702330196023f,  0.3826834323650898f,  0.1950903220161283f,
     0.0000000000000000f, -0.1950903220161282f, -0.3826834323650897f, -0.5555702330196020f,
    -0.7071067811865475f, -0.8314696123025453f, -0.9238795325112867f, -0.9807852804032304f
};

constexpr float kSin32Neg[16] = {
     0.0000000000000000f, -0.1950903220161282f, -0.3826834323650898f, -0.5555702330196022f,
    -0.7071067811865475f, -0.8314696123025452f, -0.9238795325112867f, -0.9807852804032304f,
    -1.0000000000000000f, -0.9807852804032304f, -0.9238795325112867f, -0.8314696123025452f,
    -0.7071067811865476f, -0.5555702330196022f, -0.3826834323650898f, -0.1950903220161286f
};

template <int Sign>
constexpr float twiddle32_re(int idx32) {
    return kCos32[idx32];
}

template <int Sign>
constexpr float twiddle32_im(int idx32) {
    return (Sign < 0) ? kSin32Neg[idx32] : -kSin32Neg[idx32];
}

// ============================================================================
// Branchless polynomial cos(2*pi*x) and Sign*sin(2*pi*x) using exact fp32
// minimax polynomials from include/nanofft.h and bitwise sign-bit folding.
// Runs entirely on the FP32 FMA pipes (no SFU bottleneck, no device libm).
// ============================================================================
template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pif_q1(float f, float& c_out, float& s_out) {
    // Valid for f in [0, 0.25] (first quadrant; zero reduction needed!)
    const float f2 = f * f;
    float ps = f2 * 39.536706065730207835f - 76.549782293595742666f;
    ps = ps * f2 + 81.601004073261773523f;
    ps = ps * f2 - 41.341655031416278077f;
    ps = ps * f2 + 6.283185160089477443f;
    const float s_mag = ps * f;

    float pc = f2 * 56.242380464873243260f - 85.240330322699427860f;
    pc = pc * f2 + 64.934590626780991246f;
    pc = pc * f2 - 19.739171434702393619f;
    pc = pc * f2 + 0.999999953466670136f;

    c_out = pc;
    s_out = (Sign < 0) ? -s_mag : s_mag;
}

template <int Sign = -1>
SYCL_FFT_FORCE_INLINE void sincos2pif_half(float f, float& c_out, float& s_out) {
    // Branchless bitwise quadrant fold for f in [0, 0.5)
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
    // General branchless bitwise quadrant fold for arbitrary real x
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

// ============================================================================
// Pure Radix-2 DIF (Sande-Tukey) stages in registers on E = 2^LogE elements
// held by a thread at stride T = 2^LogT (r[m] is index m*T + t in sub-FFT of size E*T).
// Fully compile-time unrolled with constexpr twiddle literals -> zero libm calls,
// zero register-array spills on both generic and cuda:sm_120 targets!
// ============================================================================
template <int LogE, int LogT, int Sign>
SYCL_FFT_FORCE_INLINE void dif_radix2_reg(float (&r_re)[1 << LogE], float (&r_im)[1 << LogE], int t) {
    constexpr int E = 1 << LogE;
    constexpr int T = 1 << LogT;
    static_assert(LogE <= 5, "kCos32 table supports up to LogE == 5 (E == 32)");

    float wt_re = 1.0f;
    float wt_im = 0.0f;
    if constexpr (LogT > 0) {
        // t < T and E >= 4 => t / (E * T) < 1 / E <= 0.25 -> pure first-quadrant polynomial!
        constexpr float inv_m = 1.0f / static_cast<float>(E * T);
        if constexpr (E >= 4) {
            sincos2pif_q1<Sign>(static_cast<float>(t) * inv_m, wt_re, wt_im);
        } else {
            sincos2pif_half<Sign>(static_cast<float>(t) * inv_m, wt_re, wt_im);
        }
    }

    unroll_for<LogE>([&](auto s_ic) {
        constexpr int S = s_ic.value;
        constexpr int H = E >> (S + 1);
        constexpr int Step = H << 1;
        constexpr int Groups = E / Step;

        unroll_for<H>([&](auto m_ic) {
            constexpr int M = m_ic.value;
            constexpr int Idx32 = M * (32 / Step);
            constexpr float cr = twiddle32_re<Sign>(Idx32);
            constexpr float ci = twiddle32_im<Sign>(Idx32);

            float wr = cr, wi = ci;
            if constexpr (LogT > 0) {
                if constexpr (M == 0) {
                    wr = wt_re;
                    wi = wt_im;
                } else if constexpr (Idx32 == 8) {
                    // W_{Step}^{Step/4} = Sign * i
                    if constexpr (Sign < 0) {
                        wr = wt_im;
                        wi = -wt_re;
                    } else {
                        wr = -wt_im;
                        wi = wt_re;
                    }
                } else {
                    wr = wt_re * cr - wt_im * ci;
                    wi = wt_re * ci + wt_im * cr;
                }
            }

            unroll_for<Groups>([&](auto g_ic) {
                constexpr int A = g_ic.value * Step + M;
                constexpr int B = A + H;
                const float ar = r_re[A], ai = r_im[A];
                const float br = r_re[B], bi = r_im[B];
                const float dr = ar - br;
                const float di = ai - bi;
                r_re[A] = ar + br;
                r_im[A] = ai + bi;

                if constexpr (LogT == 0 && M == 0) {
                    r_re[B] = dr;
                    r_im[B] = di;
                } else if constexpr (LogT == 0 && Idx32 == 8) {
                    if constexpr (Sign < 0) {
                        r_re[B] = di;
                        r_im[B] = -dr;
                    } else {
                        r_re[B] = -di;
                        r_im[B] = dr;
                    }
                } else {
                    r_re[B] = dr * wr - di * wi;
                    r_im[B] = dr * wi + di * wr;
                }
            });
        });

        if constexpr (LogT > 0 && S + 1 < LogE) {
            const float nre = wt_re * wt_re - wt_im * wt_im;
            const float nim = 2.0f * wt_re * wt_im;
            wt_re = nre;
            wt_im = nim;
        }
    });
}

// ============================================================================
// Pure Radix-2 DIF stages across W = 2^LogW consecutive sub-group lanes (LogW <= 5),
// applied to E independent W-point sub-FFTs held in r_re[0..E-1], r_im[0..E-1].
// Fully compile-time unrolled -> zero runtime branches on stage width, zero spills!
// ============================================================================
template <int LogW, int E, int Sign>
SYCL_FFT_FORCE_INLINE void dif_radix2_subgroup(const sycl::sub_group& sg, float (&r_re)[E], float (&r_im)[E], int lane) {
    constexpr int W = 1 << LogW;
    const int l = lane & (W - 1);

    // Single branchless polynomial sincos2pif_half for the widest stage (H = W/2),
    // then square for subsequent stages (H = 8, 4). When is_hi is true for stage H,
    // the squared twiddle has an exact -1 factor: W_{2H}^{l & (H-1)} = -(W_{4H}^{l & (2H-1)})^2!
    float tw_re = 1.0f, tw_im = 0.0f;
    if constexpr (LogW >= 3) {
        constexpr int H0 = W >> 1;
        constexpr float inv_w = 1.0f / static_cast<float>(W);
        sincos2pif_half<Sign>(static_cast<float>(l & (H0 - 1)) * inv_w, tw_re, tw_im);
    }

    unroll_for<LogW>([&](auto s_ic) {
        constexpr int S = s_ic.value;
        constexpr int H = W >> (S + 1);
        const bool is_hi = (l & H) != 0;
        const int k = l & (H - 1);

        float wr = tw_re, wi = tw_im;
        if constexpr (S > 0 && H > 2) {
            // Negated square: exact W_{2H}^{l & (H-1)} on lanes where (l & H) != 0
            const float sq_re = tw_re * tw_re - tw_im * tw_im;
            const float sq_im = 2.0f * tw_re * tw_im;
            tw_re = -sq_re;
            tw_im = -sq_im;
            wr = tw_re;
            wi = tw_im;
        }

        unroll_for<E>([&](auto m_ic) {
            constexpr int M = m_ic.value;
            const float my_re = r_re[M];
            const float my_im = r_im[M];
            const float p_re = sycl::permute_group_by_xor(sg, my_re, static_cast<uint32_t>(H));
            const float p_im = sycl::permute_group_by_xor(sg, my_im, static_cast<uint32_t>(H));

            if (!is_hi) {
                r_re[M] = my_re + p_re;
                r_im[M] = my_im + p_im;
            } else {
                const float dr = p_re - my_re;
                const float di = p_im - my_im;
                if constexpr (H == 1) {
                    r_re[M] = dr;
                    r_im[M] = di;
                } else if constexpr (H == 2) {
                    if (k == 0) {
                        r_re[M] = dr;
                        r_im[M] = di;
                    } else {
                        if constexpr (Sign < 0) {
                            r_re[M] = di;
                            r_im[M] = -dr;
                        } else {
                            r_re[M] = -di;
                            r_im[M] = dr;
                        }
                    }
                } else {
                    r_re[M] = dr * wr - di * wi;
                    r_im[M] = dr * wi + di * wr;
                }
            }
        });
    });
}

// ============================================================================
// Single-Pass Kernel A (LogN = 10, 11, 12):
//   N = 2^LogN = E * T, with T = 2^LogT threads per FFT, B FFTs per workgroup.
//   Phase 1: LogE register DIF stages -> E independent T-point sub-FFTs.
//   Smem transpose: Split smem_re, smem_im (1 barrier, zero bank conflicts).
//   Phase 2: LogE register DIF stages on W = T/E threads per sub-FFT (W <= 32).
//   Phase 3: LogW sub-group shuffle DIF stages (zero smem, zero barrier).
// ============================================================================
template <int LogN, int LogE, int B, int Sign>
void launch_single_pass_split_smem(sycl::queue& q, const Complex* in, Complex* out, size_t bs) {
    constexpr int N = 1 << LogN;
    constexpr int E = 1 << LogE;
    constexpr int LogT = LogN - LogE;
    constexpr int T = 1 << LogT;
    constexpr int LogW = LogT - LogE;
    static_assert(LogW >= 0 && LogW <= 5, "T_sub must fit within a 32-lane sub-group");
    constexpr int W = 1 << LogW;
    constexpr int Pad = (W == 32) ? 0 : 16;
    constexpr int RowStride = T + Pad;
    constexpr int FftSmem = E * RowStride;
    constexpr int WG = B * T;

    if (bs == 0) return;
    const size_t num_wg = (bs + B - 1) / B;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem_re(sycl::range<1>(B * FftSmem), h);
        sycl::local_accessor<float, 1> smem_im(sycl::range<1>(B * FftSmem), h);

        h.parallel_for(
            sycl::nd_range<1>(num_wg * WG, WG),
            [=](sycl::nd_item<1> item) {
                const size_t wg_id = item.get_group_linear_id();
                const int tid = static_cast<int>(item.get_local_linear_id());
                const int b_local = tid >> LogT;
                const int t = tid & (T - 1);
                const size_t batch = wg_id * B + static_cast<size_t>(b_local);
                const bool valid = (B == 1) || (batch < bs);

                const Complex* g_in = in + batch * N;
                Complex* g_out = out + batch * N;

                float r_re[E], r_im[E];
                if (valid) {
                    unroll_for<E>([&](auto m_ic) {
                        constexpr int M = m_ic.value;
                        const Complex v = g_in[t + M * T];
                        r_re[M] = v.x();
                        r_im[M] = v.y();
                    });
                } else {
                    unroll_for<E>([&](auto m_ic) {
                        constexpr int M = m_ic.value;
                        r_re[M] = 0.0f;
                        r_im[M] = 0.0f;
                    });
                }
                dif_radix2_reg<LogE, LogT, Sign>(r_re, r_im, t);

                const int smem_base = b_local * FftSmem;
                unroll_for<E>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    smem_re[smem_base + M * RowStride + t] = r_re[M];
                    smem_im[smem_base + M * RowStride + t] = r_im[M];
                });

                sycl::group_barrier(item.get_group());

                const int sub = t >> LogW;
                const int l = t & (W - 1);
                const int row_off = smem_base + sub * RowStride + l;
                unroll_for<E>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    r_re[M] = smem_re[row_off + M * W];
                    r_im[M] = smem_im[row_off + M * W];
                });

                dif_radix2_reg<LogE, LogW, Sign>(r_re, r_im, l);

                auto sg = item.get_sub_group();
                dif_radix2_subgroup<LogW, E, Sign>(sg, r_re, r_im, l);

                if (valid) {
                    const int out_base = (sub << LogE) * W + l;
                    unroll_for<E>([&](auto m_ic) {
                        constexpr int M = m_ic.value;
                        g_out[out_base + M * W] = Complex{r_re[M], r_im[M]};
                    });
                }
            });
    });
}

// ============================================================================
// Single-Pass Kernel B (LogN = 13: N = 8192):
//   T = 512 threads/WG, E = 16 elements/thread.
//   Uses ONLY 32 KiB shared memory (real-then-imag sequential transpose),
//   fitting well inside the standard 48 KiB SYCL local memory limit!
// ============================================================================
template <int Sign>
void launch_single_pass_8192(sycl::queue& q, const Complex* in, Complex* out, size_t bs) {
    constexpr int N = 8192;
    constexpr int T = 512;
    constexpr int E = 16;
    constexpr int LogE = 4;
    constexpr int LogT = 9;
    constexpr int LogW = 5;
    constexpr int W = 32;
    constexpr int SmemFloats = E * T; // 32 KiB

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(SmemFloats), h);

        h.parallel_for(
            sycl::nd_range<1>(bs * T, T),
            [=](sycl::nd_item<1> item) {
                const size_t batch = item.get_group_linear_id();
                const int t = static_cast<int>(item.get_local_linear_id());
                const Complex* g_in = in + batch * N;
                Complex* g_out = out + batch * N;

                float r_re[E], r_im[E];
                unroll_for<E>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    const Complex v = g_in[t + M * T];
                    r_re[M] = v.x();
                    r_im[M] = v.y();
                });

                dif_radix2_reg<LogE, LogT, Sign>(r_re, r_im, t);

                const int sub = t >> LogW;
                const int l = t & (W - 1);

                unroll_for<E>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    smem[M * T + t] = r_re[M];
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    r_re[M] = smem[sub * T + M * W + l];
                });
                sycl::group_barrier(item.get_group());

                unroll_for<E>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    smem[M * T + t] = r_im[M];
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    r_im[M] = smem[sub * T + M * W + l];
                });

                dif_radix2_reg<LogE, LogW, Sign>(r_re, r_im, l);

                auto sg = item.get_sub_group();
                dif_radix2_subgroup<LogW, E, Sign>(sg, r_re, r_im, l);

                const int out_base = sub * 512 + l;
                unroll_for<E>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    g_out[out_base + M * W] = Complex{r_re[M], r_im[M]};
                });
            });
    });
}

// ============================================================================
// Single-Pass Kernel C (LogN = 14: N = 16384):
//   T = 512 threads/WG, E = 32 elements/thread (64 float regs).
//   Step 1: 5 register DIF stages -> 32 independent 512-point sub-FFTs.
//   Step 2: 2 chunks of 16 sub-FFTs via ONLY 32 KiB smem (real-then-imag).
//   SINGLE GLOBAL PASS FOR N = 16384!
// ============================================================================
template <int C, int Sign>
SYCL_FFT_FORCE_INLINE void finish_chunk_16384(
    const sycl::nd_item<1>& item,
    const sycl::sub_group& sg,
    const sycl::local_accessor<float, 1>& smem,
    const float (&r_re)[32],
    const float (&r_im)[32],
    Complex* g_out,
    int t,
    int sub,
    int l) {
    constexpr int T = 512;
    constexpr int W = 32;
    constexpr int LogW = 5;
    constexpr int Chunk = 16;

    if constexpr (C > 0) {
        sycl::group_barrier(item.get_group());
    }
    float c_re[Chunk], c_im[Chunk];
    #pragma unroll
    for (int m = 0; m < Chunk; ++m) {
        smem[m * T + t] = r_re[C * Chunk + m];
    }
    sycl::group_barrier(item.get_group());
    #pragma unroll
    for (int m = 0; m < Chunk; ++m) {
        c_re[m] = smem[sub * T + m * W + l];
    }
    sycl::group_barrier(item.get_group());

    #pragma unroll
    for (int m = 0; m < Chunk; ++m) {
        smem[m * T + t] = r_im[C * Chunk + m];
    }
    sycl::group_barrier(item.get_group());
    #pragma unroll
    for (int m = 0; m < Chunk; ++m) {
        c_im[m] = smem[sub * T + m * W + l];
    }

    dif_radix2_reg<4, LogW, Sign>(c_re, c_im, l);
    dif_radix2_subgroup<LogW, Chunk, Sign>(sg, c_re, c_im, l);

    const int out_base = (C * Chunk + sub) * 512 + l;
    #pragma unroll
    for (int m = 0; m < Chunk; ++m) {
        g_out[out_base + m * W] = Complex{c_re[m], c_im[m]};
    }
}

template <int Sign>
void launch_single_pass_16384(sycl::queue& q, const Complex* in, Complex* out, size_t bs) {
    constexpr int N = 16384;
    constexpr int T = 512;
    constexpr int E = 32;
    constexpr int LogE = 5;
    constexpr int LogT = 9;
    constexpr int LogW = 5;
    constexpr int W = 32;
    constexpr int Chunk = 16;
    constexpr int SmemFloats = Chunk * T; // 32 KiB

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(SmemFloats), h);

        h.parallel_for(
            sycl::nd_range<1>(bs * T, T),
            [=](sycl::nd_item<1> item) {
                const size_t batch = item.get_group_linear_id();
                const int t = static_cast<int>(item.get_local_linear_id());
                const Complex* g_in = in + batch * N;
                Complex* g_out = out + batch * N;

                float r_re[E], r_im[E];
                #pragma unroll
                for (int m = 0; m < E; ++m) {
                    const Complex v = g_in[t + m * T];
                    r_re[m] = v.x();
                    r_im[m] = v.y();
                }

                dif_radix2_reg<LogE, LogT, Sign>(r_re, r_im, t);

                const int sub = t >> LogW;
                const int l = t & (W - 1);
                auto sg = item.get_sub_group();

                finish_chunk_16384<0, Sign>(item, sg, smem, r_re, r_im, g_out, t, sub, l);
                finish_chunk_16384<1, Sign>(item, sg, smem, r_re, r_im, g_out, t, sub, l);
            });
    });
}

// ============================================================================
// 2-Pass DIF Pass 0 (Register-only, ZERO shared memory, ZERO barriers):
//   N = N1 * N2, N1 = 2^LogN1 (16, 32), N2 = 2^LogN2.
//   multiplies slot p by W_N^{j * bitrev(p)}, and stores monotonically p = 0..N1-1.
// ============================================================================
template <int LogN1, int LogN2, int Sign>
void launch_pass0_reg(sycl::queue& q, const Complex* in, Complex* tmp, size_t chunk_bs) {
    constexpr int N1 = 1 << LogN1;
    constexpr int N2 = 1 << LogN2;
    constexpr int N = N1 * N2;
    constexpr int WG = 256;
    const size_t total_threads = chunk_bs * N2;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(total_threads, WG),
            [=](sycl::nd_item<1> item) {
                const size_t gid = item.get_global_linear_id();
                const size_t batch = gid >> LogN2;
                const int j = static_cast<int>(gid & (N2 - 1));

                const Complex* g_in = in + batch * N + j;
                Complex* g_tmp = tmp + batch * N + j;

                float r_re[N1], r_im[N1];
                unroll_for<N1>([&](auto m_ic) {
                    constexpr int M = m_ic.value;
                    const Complex v = g_in[M * N2];
                    r_re[M] = v.x();
                    r_im[M] = v.y();
                });

                // Plain N1-point natural-to-reversed DIF Radix-2 FFT in registers (zero trig calls!)
                dif_radix2_reg<LogN1, 0, Sign>(r_re, r_im, 0);

                // Multiply by W_N^{j * bitrev_{LogN1}(p)} and store in strictly ascending p order.
                // Since j < N2 and N1 >= 8, j / N < 1/8 < 0.25 -> pure first-quadrant polynomial!
                float w1r, w1i;
                sincos2pif_q1<Sign>(static_cast<float>(j) * (1.0f / static_cast<float>(N)), w1r, w1i);
                const float w2r = w1r * w1r - w1i * w1i;
                const float w2i = 2.0f * w1r * w1i;
                const float w3r = w2r * w1r - w2i * w1i;
                const float w3i = w2r * w1i + w2i * w1r;
                const float w4r = w2r * w2r - w2i * w2i;
                const float w4i = 2.0f * w2r * w2i;

                g_tmp[0] = Complex{r_re[0], r_im[0]};

                if constexpr (LogN1 == 4) {
                    const float w8r = w4r * w4r - w4i * w4i;
                    const float w8i = 2.0f * w4r * w4i;
                    const float w12r = w8r * w4r - w8i * w4i;
                    const float w12i = w8r * w4i + w8i * w4r;
                    const float lo_r[4] = {1.0f, w2r, w1r, w3r};
                    const float lo_i[4] = {0.0f, w2i, w1i, w3i};
                    const float hi_r[4] = {1.0f, w8r, w4r, w12r};
                    const float hi_i[4] = {0.0f, w8i, w4i, w12i};
                    unroll_for<15>([&](auto idx_ic) {
                        constexpr int P = idx_ic.value + 1;
                        constexpr int A = P >> 2;
                        constexpr int B = P & 3;
                        float wr, wi;
                        if constexpr (A == 0) {
                            wr = hi_r[B];
                            wi = hi_i[B];
                        } else if constexpr (B == 0) {
                            wr = lo_r[A];
                            wi = lo_i[A];
                        } else {
                            wr = lo_r[A] * hi_r[B] - lo_i[A] * hi_i[B];
                            wi = lo_r[A] * hi_i[B] + lo_i[A] * hi_r[B];
                        }
                        const float vr = r_re[P], vi = r_im[P];
                        g_tmp[P * N2] = Complex{vr * wr - vi * wi, vr * wi + vi * wr};
                    });
                } else if constexpr (LogN1 == 5) {
                    const float w8r = w4r * w4r - w4i * w4i;
                    const float w8i = 2.0f * w4r * w4i;
                    const float w16r = w8r * w8r - w8i * w8i;
                    const float w16i = 2.0f * w8r * w8i;
                    const float w24r = w16r * w8r - w16i * w8i;
                    const float w24i = w16r * w8i + w16i * w8r;
                    const float hi_r[4] = {1.0f, w16r, w8r, w24r};
                    const float hi_i[4] = {0.0f, w16i, w8i, w24i};
                    const float w5r = w4r * w1r - w4i * w1i;
                    const float w5i = w4r * w1i + w4i * w1r;
                    const float w6r = w4r * w2r - w4i * w2i;
                    const float w6i = w4r * w2i + w4i * w2r;
                    const float w7r = w4r * w3r - w4i * w3i;
                    const float w7i = w4r * w3i + w4i * w3r;
                    const float lo_r[8] = {1.0f, w4r, w2r, w6r, w1r, w5r, w3r, w7r};
                    const float lo_i[8] = {0.0f, w4i, w2i, w6i, w1i, w5i, w3i, w7i};
                    unroll_for<31>([&](auto idx_ic) {
                        constexpr int P = idx_ic.value + 1;
                        constexpr int A = P >> 2;
                        constexpr int B = P & 3;
                        float wr, wi;
                        if constexpr (A == 0) {
                            wr = hi_r[B];
                            wi = hi_i[B];
                        } else if constexpr (B == 0) {
                            wr = lo_r[A];
                            wi = lo_i[A];
                        } else {
                            wr = lo_r[A] * hi_r[B] - lo_i[A] * hi_i[B];
                            wi = lo_r[A] * hi_i[B] + lo_i[A] * hi_r[B];
                        }
                        const float vr = r_re[P], vi = r_im[P];
                        g_tmp[P * N2] = Complex{vr * wr - vi * wi, vr * wi + vi * wr};
                    });
                }
            });
    });
}

// ============================================================================
// GPU Bit-Reversal Permutation Kernel (for correctness verification)
// ============================================================================
inline void launch_bit_reverse(sycl::queue& q, const Complex* src, Complex* dst, int logN, size_t bs) {
    const size_t N = size_t{1} << logN;
    const size_t total = N * bs;
    constexpr size_t WG = 256;

    if (src == dst) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(total, WG), [=](sycl::nd_item<1> item) {
                const size_t gid = item.get_global_linear_id();
                const size_t batch = gid >> logN;
                const uint32_t idx = static_cast<uint32_t>(gid & (N - 1));
                const uint32_t rev = bit_reverse(idx, logN);
                if (idx < rev) {
                    Complex* base = dst + batch * N;
                    const Complex a = base[idx];
                    const Complex b = base[rev];
                    base[idx] = b;
                    base[rev] = a;
                }
            });
        });
    } else {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(total, WG), [=](sycl::nd_item<1> item) {
                const size_t gid = item.get_global_linear_id();
                const size_t batch = gid >> logN;
                const uint32_t idx = static_cast<uint32_t>(gid & (N - 1));
                const uint32_t rev = bit_reverse(idx, logN);
                dst[batch * N + idx] = src[batch * N + rev];
            });
        });
    }
}

// ============================================================================
// Plan & Dispatcher (logN = 10..18)
// ============================================================================
template <int Sign = -1>
class DifFftPlan {
public:
    DifFftPlan(sycl::queue& q, int logN, size_t max_bs, size_t l2_chunk_bytes = 6 * 1024 * 1024)
        : q_(q), logN_(logN), N_(size_t{1} << logN) {
        if (logN < 10 || logN > 18) {
            throw std::invalid_argument("Supported logN range is 10..18");
        }
        if (logN >= 15) {
            const size_t fft_bytes = N_ * sizeof(Complex);
            size_t chunk_bs = std::max<size_t>(1, l2_chunk_bytes / fft_bytes);
            chunk_bs_ = std::min(max_bs, chunk_bs);
            if (chunk_bs_ == 0) chunk_bs_ = 1;
            scratch_elems_ = chunk_bs_ * N_;
            scratch_ = sycl::malloc_device<Complex>(scratch_elems_, q_);
            if (!scratch_) throw std::bad_alloc();
        }
    }

    ~DifFftPlan() {
        if (scratch_) sycl::free(scratch_, q_);
    }

    DifFftPlan(const DifFftPlan&) = delete;
    DifFftPlan& operator=(const DifFftPlan&) = delete;

    // Bit-reversal-free (natural-to-reversed) DIF Radix-2 FFT
    void execute_raw(const Complex* in, Complex* out, size_t bs) {
        switch (logN_) {
            case 10:
                launch_single_pass_split_smem<10, 3, 2, Sign>(q_, in, out, bs);
                break;
            case 11:
                launch_single_pass_split_smem<11, 3, 1, Sign>(q_, in, out, bs);
                break;
            case 12:
                launch_single_pass_split_smem<12, 4, 1, Sign>(q_, in, out, bs);
                break;
            case 13:
                launch_single_pass_8192<Sign>(q_, in, out, bs);
                break;
            case 14:
                launch_single_pass_16384<Sign>(q_, in, out, bs);
                break;
            case 15:
                execute_two_pass<4, 11>(in, out, bs);
                break;
            case 16:
                execute_two_pass<4, 12>(in, out, bs);
                break;
            case 17:
                execute_two_pass<4, 13>(in, out, bs);
                break;
            case 18:
                execute_two_pass<5, 13>(in, out, bs);
                break;
            default:
                throw std::invalid_argument("Unsupported logN");
        }
    }

    // Full natural-to-natural DIF Radix-2 FFT (for correctness verification)
    void execute(const Complex* in, Complex* out, size_t bs) {
        execute_raw(in, out, bs);
        launch_bit_reverse(q_, out, out, logN_, bs);
    }

    size_t chunk_bs() const { return chunk_bs_; }

private:
    template <int LogN1, int LogN2>
    void execute_two_pass(const Complex* in, Complex* out, size_t bs) {
        constexpr size_t N1 = size_t{1} << LogN1;
        size_t done = 0;
        while (done < bs) {
            const size_t cur = std::min(chunk_bs_, bs - done);
            const Complex* c_in = in + done * N_;
            Complex* c_out = out + done * N_;

            launch_pass0_reg<LogN1, LogN2, Sign>(q_, c_in, scratch_, cur);

            if constexpr (LogN2 == 11) {
                launch_single_pass_split_smem<11, 3, 1, Sign>(q_, scratch_, c_out, cur * N1);
            } else if constexpr (LogN2 == 12) {
                launch_single_pass_split_smem<12, 4, 1, Sign>(q_, scratch_, c_out, cur * N1);
            } else if constexpr (LogN2 == 13) {
                launch_single_pass_8192<Sign>(q_, scratch_, c_out, cur * N1);
            }
            done += cur;
        }
    }

    sycl::queue& q_;
    int logN_;
    size_t N_;
    size_t chunk_bs_ = 0;
    size_t scratch_elems_ = 0;
    Complex* scratch_ = nullptr;
};

} // namespace sycl_fft
