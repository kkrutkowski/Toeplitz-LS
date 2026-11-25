#pragma once

// Quadruple-Precision SYCL GPU-Accelerated Type-1 NuFFT (NuFFT-1) Umbrella Header:
//   - Hardcoded piecewise minimax PSWF_0^c polynomial spreading kernels:
//       FP16 (2xFP16, h suffix):   w in {4, 5},       x-axis gridding in df_t (2xFP32)
//       FP32 (float,  f suffix):   w in {8, 9},       x-axis gridding in df_t (2xFP32)
//       DF   (2xFP32, df suffix):  w in {14, 15, 16}, x-axis gridding in df_t (2xFP32)
//       FP64 (double, no suffix):  w in {16, 17, 18}, x-axis gridding in double (FP64)
//   - FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution pass
//   - Primary fast path: Natural-to-bit-reversed (SlotBitReversed) transform
//   - Verification path: Natural frequency order (Natural) + host slot un-permutation

#include "sycl_fft_dif.hpp"
#include "sycl_nufft_pswf_coeffs.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sycl_nufft {

enum class NufftOutOrder : int {
    SlotBitReversed = 0, // Primary fast path: raw DIF slot order (no bit-reversal pass)
    Natural         = 1  // Verification path: natural frequency order k in [0, N_out)
};

SYCL_FFT_FORCE_INLINE constexpr int next_pow2_int(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

SYCL_FFT_FORCE_INLINE constexpr int int_log2_pow2(int n) {
    int l = 0;
    while ((1 << l) < n) ++l;
    return l;
}

SYCL_FFT_FORCE_INLINE constexpr int nufft1_nout(int N, PswfMode mode) {
    return (mode == PswfMode::PSWF43) ? (N + (N >> 1)) : N;
}

SYCL_FFT_FORCE_INLINE constexpr int nufft1_output_shift(int N, PswfMode mode) {
    return (mode == PswfMode::PSWF43) ? (3 * (N >> 2)) : (N >> 1);
}

// Maps compact slot index i in [0, N_out) to raw DIF FFT slot p in [0, N_fft)
// matching src/nufft1.c:pswf_build_slot_tables:
//   PSWF21: period = 4, nvalid = 2, offsets = {0, 3} (dropped top-2-bit quadrants {1,2} -> bitrev2 {2,1})
//   PSWF43: period = 8, nvalid = 6, offsets = {0, 2, 3, 4, 5, 7} (dropped top-3-bit octants {3,4} -> bitrev3 {6,1})
SYCL_FFT_FORCE_INLINE constexpr int nufft1_compact_idx_to_slot(int i, PswfMode mode) {
    if (mode == PswfMode::PSWF21) {
        return ((i >> 1) << 2) + ((i & 1) ? 3 : 0);
    } else {
        const int q = i / 6;
        const int r = i - 6 * q;
        const int off = r + (r >= 1 ? 1 : 0) + (r >= 5 ? 1 : 0);
        return (q << 3) + off;
    }
}

// Maps raw DIF FFT slot p in [0, N_fft) to centered frequency index k' = k - output_shift
// in [-output_shift, N_out - output_shift)
SYCL_FFT_FORCE_INLINE int nufft1_slot_to_k_prime(
    int slot_p, int log_nfft, int nfft, int output_shift) {
    const int bin = static_cast<int>(
        sycl_fft::bit_reverse(static_cast<uint32_t>(slot_p), log_nfft));
    const int first_bin = nfft - output_shift;
    return (bin >= first_bin) ? (bin - nfft) : bin;
}

// Maps raw DIF FFT slot p in [0, N_fft) to natural frequency index k in [0, N_out)
SYCL_FFT_FORCE_INLINE int nufft1_slot_to_natural_idx(
    int slot_p, int log_nfft, int nfft, int output_shift) {
    return nufft1_slot_to_k_prime(slot_p, log_nfft, nfft, output_shift) + output_shift;
}

// Host helper for accuracy verification ONLY (never included in benchmark timings):
// Converts compact SlotBitReversed output [bs][N_out] to Natural frequency order [bs][N_out].
template <typename ComplexT>
inline void unpermute_slots_to_natural_host(
    const ComplexT* h_slots,
    ComplexT* h_nat,
    int N,
    PswfMode mode,
    size_t bs) {
    const int nfft = next_pow2_int(2 * N);
    const int log_nfft = int_log2_pow2(nfft);
    const int nout = nufft1_nout(N, mode);
    const int output_shift = nufft1_output_shift(N, mode);

    for (size_t b = 0; b < bs; ++b) {
        const ComplexT* src = h_slots + b * static_cast<size_t>(nout);
        ComplexT* dst = h_nat + b * static_cast<size_t>(nout);
        for (int i = 0; i < nout; ++i) {
            const int slot_p = nufft1_compact_idx_to_slot(i, mode);
            const int k = nufft1_slot_to_natural_idx(slot_p, log_nfft, nfft, output_shift);
            if (k >= 0 && k < nout) {
                dst[k] = src[i];
            }
        }
    }
}

} // namespace sycl_nufft

#if !defined(SYCL_FFT_ONLY_FLOAT) && !defined(SYCL_FFT_ONLY_DOUBLE) && !defined(SYCL_FFT_ONLY_DF)
#undef SYCL_FFT_DOUBLE
#undef SYCL_FFT_DF
#define SYCL_FFT_HALF
#include "sycl_fft_precision.hpp"
#include "sycl_nufft1_impl.hpp"
#undef SYCL_FFT_HALF
#endif

#if !defined(SYCL_FFT_ONLY_HALF) && !defined(SYCL_FFT_ONLY_DOUBLE) && !defined(SYCL_FFT_ONLY_DF)
#undef SYCL_FFT_HALF
#undef SYCL_FFT_DOUBLE
#undef SYCL_FFT_DF
#include "sycl_fft_precision.hpp"
#include "sycl_nufft1_impl.hpp"
#endif

#if !defined(SYCL_FFT_ONLY_HALF) && !defined(SYCL_FFT_ONLY_FLOAT) && !defined(SYCL_FFT_ONLY_DF)
#undef SYCL_FFT_HALF
#undef SYCL_FFT_DF
#define SYCL_FFT_DOUBLE
#include "sycl_fft_precision.hpp"
#include "sycl_nufft1_impl.hpp"
#undef SYCL_FFT_DOUBLE
#endif

#if !defined(SYCL_FFT_ONLY_HALF) && !defined(SYCL_FFT_ONLY_FLOAT) && !defined(SYCL_FFT_ONLY_DOUBLE)
#undef SYCL_FFT_HALF
#undef SYCL_FFT_DOUBLE
#define SYCL_FFT_DF
#include "sycl_fft_precision.hpp"
#include "sycl_nufft1_impl.hpp"
#undef SYCL_FFT_DF
#endif
