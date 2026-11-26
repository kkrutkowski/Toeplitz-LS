#pragma once

// Quadruple-Precision SYCL GPU-Accelerated Type-1 NuFFT (NuFFT-1) Umbrella Header:
//   - Hardcoded piecewise minimax PSWF_0^c polynomial spreading kernels:
//       FP16 (2xFP16, h suffix):   w in {4, 5},       x-axis gridding in df_t (2xFP32)
//       FP32 (float,  f suffix):   w in {8, 9},       x-axis gridding in df_t (2xFP32)
//       DF   (2xFP32, df suffix):  w in {14, 15, 16}, x-axis gridding in df_t (2xFP32)
//       FP64 (double, no suffix):  w in {16, 17, 18}, x-axis gridding in double (FP64)
//   - Hardware-generalized launch parameters derived from DeviceFftConfig + BankModel
//     + persistent AutotuneCache (zero hardcoded GPU/Blackwell tuning constants)
//   - Decoupled execution modules (aux / spread / deconv / zero-HCF host plan)
//   - FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution pass
//   - Primary fast path: Natural-to-bit-reversed (SlotBitReversed) transform

#include "fft/dif.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sycl_nufft {

#ifndef SYCL_NUFFT_PSWF_MODE_DEFINED
#define SYCL_NUFFT_PSWF_MODE_DEFINED 1
enum class PswfMode : int {
    PSWF21 = 0, // sigma = 2.0,  N_out = N,     output_shift = N/2,   N_fft = bitceil(2N)
    PSWF43 = 1  // sigma = 4/3,  N_out = 3N/2,  output_shift = 3N/4,  N_fft = bitceil(2N)
};
#endif

enum class NufftOutOrder : int {
    SlotBitReversed    = 0, // Primary fast path: compact DIF slot order, deconvolved (fused FFT epilogue)
    Natural            = 1, // Verification path: natural frequency order k in [0, N_out), deconvolved
    SlotBitReversedRaw = 2  // Deferred deconvolution: compact DIF slot order WITHOUT dividing by phi_hat
};

// Hardware-generalized NuFFT-1 launch configuration (all fields auto-derived when 0/empty)
struct Nufft1LaunchConfig {
    sycl_fft::FftLaunchConfig fft_cfg{};
    std::string spread_variant_override;  // e.g. "smem_wg128_win256" or "collab_wg128"
    std::string deconv_variant_override;  // e.g. "deconv_wg64_bt64"
    size_t      chunk_bytes_override = 0; // 0 = auto from DeviceFftConfig L2 & CUs
    bool        force_collab_spread  = false;
    bool        use_geo_scale_fp16   = false; // true = geometric-mean-1 FP16 PSWF table (used by FP16 IHS)

    bool has_spread_overrides() const {
        return !spread_variant_override.empty() || force_collab_spread ||
               fft_cfg.allow_emulated_local_mem;
    }

    bool has_deconv_overrides() const {
        return !deconv_variant_override.empty();
    }
};

// Runtime spreading execution parameters (derived dynamically from DeviceFftConfig & BankModel)
struct SpreadConfig {
    bool use_smem            = true;
    bool use_reg_acc         = false;
    bool use_cpu_direct      = false;
    bool use_precomp_weights = false;
    int  wg_size             = 64;
    int  t_j                 = 8;
    int  b_per_wg            = 8;
    int  win_elems           = 1024;
    int  num_wins            = 1;
    int  grid_stride         = 1024;
    int  tile_m              = 32;
    int  transpose_tile      = 16;
    int  transpose_stride    = 17;
};

// Runtime deconvolution execution parameters (derived dynamically from DeviceFftConfig)
struct DeconvConfig {
    int wg_i       = 64;
    int batch_tile = 32;
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
//   PSWF21: period = 4, nvalid = 2, offsets = {0, 3}
//   PSWF43: period = 8, nvalid = 6, offsets = {0, 2, 3, 4, 5, 7}
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

namespace detail {

// --- Auxiliary kernels (sycl_nufft1_aux.cpp) ---
void launch_precompute_pointsh(
    sycl::queue& q, const sycl_fft::df_t* d_x, sycl_fft::df_t df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43, bool pos_sign,
    int* d_first, sycl_fft::f16_scalar_t* d_u, sycl_fft::Complexh* d_shift,
    size_t wg_size);

void launch_precompute_pointsf(
    sycl::queue& q, const sycl_fft::df_t* d_x, sycl_fft::df_t df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43, bool pos_sign,
    int* d_first, float* d_u, sycl_fft::Complexf* d_shift,
    size_t wg_size);

void launch_precompute_points(
    sycl::queue& q, const double* d_x, double df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43, bool pos_sign,
    int* d_first, double* d_u, sycl_fft::Complex* d_shift,
    size_t wg_size);

void launch_precompute_pointsdf(
    sycl::queue& q, const sycl_fft::df_t* d_x, sycl_fft::df_t df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43, bool pos_sign,
    int* d_first, sycl_fft::df_t* d_u, sycl_fft::Complexdf* d_shift,
    size_t wg_size);

void launch_build_window_buckets(
    sycl::queue& q, const int* d_first, int M, int nfft, int w,
    int win_elems, int num_wins,
    int* d_bucket_counts, int* d_bucket_off, int* d_bucket_m,
    size_t wg_size);

// Raw-size deduplicated grid transpose + L2 zeroing:
//   - 4B: FP32 (float -> Complexf)
//   - 8B: shared identically by FP64 (double -> Complex) and DF (df_t -> Complexdf)
//   - f16pair: FP16 2xSIMD paired unpack (f16x2_t -> 2 x Complexh)
void launch_transpose_and_zero_4b(
    sycl::queue& q, uint32_t* d_grid_T_re, uint32_t* d_grid_T_im,
    uint64_t* d_fft_buf, int nfft, size_t chunk_bs, size_t packed_stride,
    int tile, int stride);

void launch_transpose_and_zero_8b(
    sycl::queue& q, uint64_t* d_grid_T_re, uint64_t* d_grid_T_im,
    sycl::ulong2* d_fft_buf, int nfft, size_t chunk_bs, size_t packed_stride,
    int tile, int stride);

void launch_transpose_and_zero_f16pair(
    sycl::queue& q, sycl_fft::f16x2_t* d_grid_T_re, sycl_fft::f16x2_t* d_grid_T_im,
    sycl_fft::Complexh* d_fft_buf, int nfft, size_t chunk_bs, size_t packed_stride,
    int tile, int stride);

// --- Device-resident PSWF table uploaders (avoids 1 KiB by-value kernel capture;
//     applies exact geometric-mean-1 rescaling for FP32/FP64/DF, retains 2^e for FP16 unless geo_scale=true) ---
uint8_t* upload_pswf_tableh(sycl::queue& q, int w, PswfMode mode, bool geo_scale = false);
uint8_t* upload_pswf_tablef(sycl::queue& q, int w, PswfMode mode);
uint8_t* upload_pswf_table(sycl::queue& q, int w, PswfMode mode);
uint8_t* upload_pswf_tabledf(sycl::queue& q, int w, PswfMode mode);

// --- L2-resident precomputed spreading weight evaluators (called once in set_points) ---
void launch_precompute_weightsh(
    sycl::queue& q, int w, const void* d_tbl,
    const sycl_fft::f16_scalar_t* d_u, sycl_fft::f16_scalar_t* d_weights,
    int M, size_t wg_size);

void launch_precompute_weightsf(
    sycl::queue& q, int w, const void* d_tbl,
    const float* d_u, float* d_weights,
    int M, size_t wg_size);

void launch_precompute_weights(
    sycl::queue& q, int w, const void* d_tbl,
    const double* d_u, double* d_weights,
    int M, size_t wg_size);

void launch_precompute_weightsdf(
    sycl::queue& q, int w, const void* d_tbl,
    const sycl_fft::df_t* d_u, sycl_fft::df_t* d_weights,
    int M, size_t wg_size);

// --- Bucket-order geometry packers (once per set_points when num_wins > 1) ---
void launch_pack_bucket_geomh(
    sycl::queue& q, int w, int num_wins, int max_entries,
    const int* d_bucket_off, const int* d_bucket_m,
    const int* d_first, const sycl_fft::Complexh* d_shift, const sycl_fft::f16_scalar_t* d_weights,
    int* d_b_first, sycl_fft::Complexh* d_b_shift, sycl_fft::f16_scalar_t* d_b_weights,
    size_t wg_size);

void launch_pack_bucket_geomf(
    sycl::queue& q, int w, int num_wins, int max_entries,
    const int* d_bucket_off, const int* d_bucket_m,
    const int* d_first, const sycl_fft::Complexf* d_shift, const float* d_weights,
    int* d_b_first, sycl_fft::Complexf* d_b_shift, float* d_b_weights,
    size_t wg_size);

void launch_pack_bucket_geom(
    sycl::queue& q, int w, int num_wins, int max_entries,
    const int* d_bucket_off, const int* d_bucket_m,
    const int* d_first, const sycl_fft::Complex* d_shift, const double* d_weights,
    int* d_b_first, sycl_fft::Complex* d_b_shift, double* d_b_weights,
    size_t wg_size);

void launch_pack_bucket_geomdf(
    sycl::queue& q, int w, int num_wins, int max_entries,
    const int* d_bucket_off, const int* d_bucket_m,
    const int* d_first, const sycl_fft::Complexdf* d_shift, const sycl_fft::df_t* d_weights,
    int* d_b_first, sycl_fft::Complexdf* d_b_shift, sycl_fft::df_t* d_b_weights,
    size_t wg_size);

// --- Decoupled spreading stage launchers (nufft/spread.cpp) ---
void launch_spread_smem_winh(
    sycl::queue& q, int w, const void* d_tbl, const sycl_fft::f16_scalar_t* d_weights,
    const int* d_first, const sycl_fft::f16_scalar_t* d_u, const sycl_fft::Complexh* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const sycl_fft::Complexh* d_y, sycl_fft::Complexh* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_collabh(
    sycl::queue& q, int w, const void* d_tbl, const sycl_fft::f16_scalar_t* d_weights,
    const int* d_first, const sycl_fft::f16_scalar_t* d_u, const sycl_fft::Complexh* d_shift,
    const sycl_fft::Complexh* d_y,
    sycl_fft::f16x2_t* d_grid_T_re, sycl_fft::f16x2_t* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_cpu_directh(
    sycl::queue& q, int w, const void* d_tbl, const sycl_fft::f16_scalar_t* d_weights,
    const int* d_first, const sycl_fft::f16_scalar_t* d_u, const sycl_fft::Complexh* d_shift,
    const sycl_fft::Complexh* d_y, sycl_fft::Complexh* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs);

void launch_spread_smem_winf(
    sycl::queue& q, int w, const void* d_tbl, const float* d_weights,
    const int* d_first, const float* d_u, const sycl_fft::Complexf* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const sycl_fft::Complexf* d_y, sycl_fft::Complexf* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_collabf(
    sycl::queue& q, int w, const void* d_tbl, const float* d_weights,
    const int* d_first, const float* d_u, const sycl_fft::Complexf* d_shift,
    const sycl_fft::Complexf* d_y,
    float* d_grid_T_re, float* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_cpu_directf(
    sycl::queue& q, int w, const void* d_tbl, const float* d_weights,
    const int* d_first, const float* d_u, const sycl_fft::Complexf* d_shift,
    const sycl_fft::Complexf* d_y, sycl_fft::Complexf* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs);

void launch_spread_smem_win(
    sycl::queue& q, int w, const void* d_tbl, const double* d_weights,
    const int* d_first, const double* d_u, const sycl_fft::Complex* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const sycl_fft::Complex* d_y, sycl_fft::Complex* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_collab(
    sycl::queue& q, int w, const void* d_tbl, const double* d_weights,
    const int* d_first, const double* d_u, const sycl_fft::Complex* d_shift,
    const sycl_fft::Complex* d_y,
    double* d_grid_T_re, double* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_cpu_direct(
    sycl::queue& q, int w, const void* d_tbl, const double* d_weights,
    const int* d_first, const double* d_u, const sycl_fft::Complex* d_shift,
    const sycl_fft::Complex* d_y, sycl_fft::Complex* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs);

void launch_spread_smem_windf(
    sycl::queue& q, int w, const void* d_tbl, const sycl_fft::df_t* d_weights,
    const int* d_first, const sycl_fft::df_t* d_u, const sycl_fft::Complexdf* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const sycl_fft::Complexdf* d_y, sycl_fft::Complexdf* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_collabdf(
    sycl::queue& q, int w, const void* d_tbl, const sycl_fft::df_t* d_weights,
    const int* d_first, const sycl_fft::df_t* d_u, const sycl_fft::Complexdf* d_shift,
    const sycl_fft::Complexdf* d_y,
    sycl_fft::df_t* d_grid_T_re, sycl_fft::df_t* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_cpu_directdf(
    sycl::queue& q, int w, const void* d_tbl, const sycl_fft::df_t* d_weights,
    const int* d_first, const sycl_fft::df_t* d_u, const sycl_fft::Complexdf* d_shift,
    const sycl_fft::Complexdf* d_y, sycl_fft::Complexdf* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs);

// SYCL 2020 kernel_bundle / private_mem_size spill & WG probe for register-resident spreading
sycl_fft::KernelProbeInfo probe_spread_reg_kernelh(
    sycl::queue& q, const sycl::device& dev, int w, const SpreadConfig& cfg);
sycl_fft::KernelProbeInfo probe_spread_reg_kernelf(
    sycl::queue& q, const sycl::device& dev, int w, const SpreadConfig& cfg);
sycl_fft::KernelProbeInfo probe_spread_reg_kernel(
    sycl::queue& q, const sycl::device& dev, int w, const SpreadConfig& cfg);
sycl_fft::KernelProbeInfo probe_spread_reg_kerneldf(
    sycl::queue& q, const sycl::device& dev, int w, const SpreadConfig& cfg);

// --- Decoupled GL-free deconvolution stage launchers & reusable LUT builders (nufft/deconv.cpp) ---
// table_kind: 0 = slot_phi_hat (compact), 1 = slot_deconv (compact), 2 = nat_deconv (natural)
void launch_build_deconv_tableh(
    sycl::queue& q, int w, const void* d_tbl,
    int nfft, int log_nfft, int nout, int output_shift,
    PswfMode mode, int table_kind, sycl_fft::f16_scalar_t* d_out,
    double output_scale = 1.0);

void launch_build_deconv_tableh_f32(
    sycl::queue& q, int w, const void* d_tbl,
    int nfft, int log_nfft, int nout, int output_shift,
    PswfMode mode, int table_kind, float* d_out,
    double output_scale = 1.0);

void launch_build_deconv_tablef(
    sycl::queue& q, int w, const void* d_tbl,
    int nfft, int log_nfft, int nout, int output_shift,
    PswfMode mode, int table_kind, float* d_out,
    double output_scale = 1.0);

void launch_build_deconv_table(
    sycl::queue& q, int w, const void* d_tbl,
    int nfft, int log_nfft, int nout, int output_shift,
    PswfMode mode, int table_kind, double* d_out,
    double output_scale = 1.0);

void launch_build_deconv_tabledf(
    sycl::queue& q, int w, const void* d_tbl,
    int nfft, int log_nfft, int nout, int output_shift,
    PswfMode mode, int table_kind, sycl_fft::df_t* d_out,
    double output_scale = 1.0);

void launch_deconv_emith(
    sycl::queue& q, int w, const void* d_tbl,
    const sycl_fft::Complexh* d_fft_buf, sycl_fft::Complexh* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg,
    double output_scale = 1.0);

void launch_deconv_emitf(
    sycl::queue& q, int w, const void* d_tbl,
    const sycl_fft::Complexf* d_fft_buf, sycl_fft::Complexf* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg,
    double output_scale = 1.0);

void launch_deconv_emit(
    sycl::queue& q, int w, const void* d_tbl,
    const sycl_fft::Complex* d_fft_buf, sycl_fft::Complex* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg,
    double output_scale = 1.0);

void launch_deconv_emitdf(
    sycl::queue& q, int w, const void* d_tbl,
    const sycl_fft::Complexdf* d_fft_buf, sycl_fft::Complexdf* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg,
    double output_scale = 1.0);

} // namespace detail

namespace detail {
// --- Precision & sign tags for Nufft1PlanBase ---
template <int Sign = SYCL_FFT_DEFAULT_SIGN>
struct HalfPlanTraits {
    using ComplexT = sycl_fft::Complexh;
    using ScalarT  = sycl_fft::f16_scalar_t;
    using RealT    = sycl_fft::f16x2_t;
    using XCoordT  = sycl_fft::df_t;
    using FftPlanT = sycl_fft::DifFftPlanh<Sign>;
    static constexpr int    kSign    = Sign;
    static constexpr bool   kPosSign = (Sign > 0);
    static constexpr size_t kLanes   = 2;
    static constexpr bool   kIsFp64  = false;
    static constexpr bool   kIsHalf  = true;
    static constexpr bool   kIsDf    = false;
    static constexpr const char* kPrecTag = "h";
    static bool is_valid_width(int w) { return w == 4 || w == 5; }
};

template <int Sign = SYCL_FFT_DEFAULT_SIGN>
struct FloatPlanTraits {
    using ComplexT = sycl_fft::Complexf;
    using ScalarT  = float;
    using RealT    = float;
    using XCoordT  = sycl_fft::df_t;
    using FftPlanT = sycl_fft::DifFftPlanf<Sign>;
    static constexpr int    kSign    = Sign;
    static constexpr bool   kPosSign = (Sign > 0);
    static constexpr size_t kLanes   = 1;
    static constexpr bool   kIsFp64  = false;
    static constexpr bool   kIsHalf  = false;
    static constexpr bool   kIsDf    = false;
    static constexpr const char* kPrecTag = "f";
    static bool is_valid_width(int w) { return w == 8 || w == 9; }
};

template <int Sign = SYCL_FFT_DEFAULT_SIGN>
struct DoublePlanTraits {
    using ComplexT = sycl_fft::Complex;
    using ScalarT  = double;
    using RealT    = double;
    using XCoordT  = double;
    using FftPlanT = sycl_fft::DifFftPlan<Sign>;
    static constexpr int    kSign    = Sign;
    static constexpr bool   kPosSign = (Sign > 0);
    static constexpr size_t kLanes   = 1;
    static constexpr bool   kIsFp64  = true;
    static constexpr bool   kIsHalf  = false;
    static constexpr bool   kIsDf    = false;
    static constexpr const char* kPrecTag = "d";
    static bool is_valid_width(int w) { return w == 16 || w == 17 || w == 18; }
};

template <int Sign = SYCL_FFT_DEFAULT_SIGN>
struct DfPlanTraits {
    using ComplexT = sycl_fft::Complexdf;
    using ScalarT  = sycl_fft::df_t;
    using RealT    = sycl_fft::df_t;
    using XCoordT  = sycl_fft::df_t;
    using FftPlanT = sycl_fft::DifFftPlandf<Sign>;
    static constexpr int    kSign    = Sign;
    static constexpr bool   kPosSign = (Sign > 0);
    static constexpr size_t kLanes   = 1;
    static constexpr bool   kIsFp64  = false;
    static constexpr bool   kIsHalf  = false;
    static constexpr bool   kIsDf    = true;
    static constexpr const char* kPrecTag = "df";
    static bool is_valid_width(int w) { return w == 14 || w == 15 || w == 16; }
};

} // namespace detail

// Pure-host orchestrated NuFFT-1 plan (instantiated in nufft/plan.cpp, 0 KiB HCF).
// Requires an in-order sycl::queue and power-of-2 N in [512, 131072] (enforced upfront).
template <typename Traits>
class Nufft1PlanBase {
public:
    using ComplexT = typename Traits::ComplexT;
    using ScalarT  = typename Traits::ScalarT;
    using RealT    = typename Traits::RealT;
    using XCoordT  = typename Traits::XCoordT;
    using FftPlanT = typename Traits::FftPlanT;
    static constexpr int  kSign    = Traits::kSign;
    static constexpr bool kPosSign = Traits::kPosSign;

    Nufft1PlanBase(
        sycl::queue& q,
        int M,
        int N,
        int w,
        PswfMode mode,
        size_t max_bs,
        const Nufft1LaunchConfig& nufft_cfg);

    Nufft1PlanBase(
        sycl::queue& q,
        int M,
        int N,
        int w,
        PswfMode mode,
        size_t max_bs,
        const sycl_fft::FftLaunchConfig& fft_cfg = {})
        : Nufft1PlanBase(q, M, N, w, mode, max_bs, make_default_cfg(fft_cfg)) {}

    ~Nufft1PlanBase();

    Nufft1PlanBase(const Nufft1PlanBase&) = delete;
    Nufft1PlanBase& operator=(const Nufft1PlanBase&) = delete;

    // Precompute high-precision x-axis gridding (FP64 for FP64; Dekker df_t for DF/FP32/FP16)
    // and build O(M) window buckets when windowed SMEM spreading uses num_wins > 1.
    void set_points(const XCoordT* d_x, XCoordT df, int freq_factor = 1);

    // Resizable overload for reusing a single cached Nufft1Plan across lightcurves of varying M
    // with zero device reallocations whenever M <= M_cap_.
    void set_points(const XCoordT* d_x, int M, XCoordT df, int freq_factor = 1);

    // Execute NuFFT-1 for a batch of `bs` signals d_y [bs][M] -> d_out [bs][N_out].
    void execute(
        const ComplexT* d_y,
        ComplexT* d_out,
        size_t bs,
        NufftOutOrder order = NufftOutOrder::SlotBitReversed);

    // Convenience single-call entrypoint combining set_points + execute
    void execute(
        const XCoordT* d_x,
        XCoordT df,
        const ComplexT* d_y,
        ComplexT* d_out,
        size_t bs,
        int freq_factor = 1,
        NufftOutOrder order = NufftOutOrder::SlotBitReversed) {
        set_points(d_x, df, freq_factor);
        execute(d_y, d_out, bs, order);
    }

    // Optional multiplicative output scale folded into deconvolution (for FP16, folded in FP32
    // prior to the single FP16 cast to absorb Toeplitz row_scale without an extra rounding pass).
    void set_output_scale(double scale);
    double output_scale() const { return output_scale_; }

    // Lazy read-only device LUTs (length nout_) for deferred deconvolution and multi-NuFFT
    // index-wise combination with a single final bit-reversal pass:
    //   d_slot_phi_hat():     phi_hat(k'_i) in compact SlotBitReversed order
    //   d_slot_deconv():      output_scale / phi_hat(k'_i) in compact SlotBitReversed order
    //   d_deconv():           output_scale / phi_hat(k') in Natural frequency order
    //   d_slot_deconv_f32():  FP32-precision output_scale / phi_hat(k'_i) (FP16 IHS deferred deconv)
    const ScalarT* d_slot_phi_hat();
    const ScalarT* d_slot_deconv();
    const ScalarT* d_deconv();
    const float*   d_slot_deconv_f32();

    int M() const { return M_; }
    int M_capacity() const { return M_cap_; }
    int N() const { return N_; }
    int w() const { return w_; }
    int nfft() const { return nfft_; }
    int log_nfft() const { return log_nfft_; }
    int nout() const { return nout_; }
    int output_shift() const { return output_shift_; }
    PswfMode mode() const { return mode_; }
    size_t chunk_bs() const { return chunk_bs_; }
    FftPlanT& fft_plan() { return fft_plan_; }

    const char* spread_variant_name() const { return spread_variant_name_.c_str(); }
    const char* deconv_variant_name() const { return deconv_variant_name_.c_str(); }
    bool spread_cache_hit() const { return spread_cache_hit_; }
    bool deconv_cache_hit() const { return deconv_cache_hit_; }
    int  spread_num_wins() const { return spread_cfg_.num_wins; }
    int  spread_win_elems() const { return spread_cfg_.win_elems; }
    const SpreadConfig& spread_config() const { return spread_cfg_; }
    const DeconvConfig& deconv_config() const { return deconv_cfg_; }

private:
    static int validate_and_log_nfft(sycl::queue& q, int M, int N, int w, PswfMode mode) {
        if (!q.is_in_order()) {
            throw std::invalid_argument("Nufft1Plan: requires an in-order sycl::queue");
        }
        if (M <= 0) {
            throw std::invalid_argument("Nufft1Plan: M must be positive");
        }
        if (N < 512 || N > (1 << 17) || (N & (N - 1)) != 0) {
            throw std::invalid_argument(
                "Nufft1Plan: N must be a power of 2 in [512, 131072]");
        }
        if (mode == PswfMode::PSWF43 && (N % 4) != 0) {
            throw std::invalid_argument("Nufft1Plan: PSWF43 requires N divisible by 4");
        }
        if (!Traits::is_valid_width(w)) {
            throw std::invalid_argument("Nufft1Plan: unsupported kernel width w for this precision");
        }
        return int_log2_pow2(2 * N);
    }

    static Nufft1LaunchConfig make_default_cfg(const sycl_fft::FftLaunchConfig& fft_cfg) {
        Nufft1LaunchConfig c{};
        c.fft_cfg = fft_cfg;
        return c;
    }

    void allocate_collab_grid();
    void allocate_window_buckets();
    void ensure_point_capacity(int M);

    sycl::queue&       q_;
    int                M_;
    int                M_cap_;
    int                N_;
    int                w_;
    PswfMode           mode_;
    int                log_nfft_;
    int                nfft_;
    int                nout_;
    int                output_shift_;
    size_t             max_bs_;
    size_t             chunk_bs_;
    size_t             packed_stride_;
    Nufft1LaunchConfig launch_cfg_;

    FftPlanT           fft_plan_;

    SpreadConfig       spread_cfg_{};
    DeconvConfig       deconv_cfg_{};
    std::string        spread_variant_name_;
    std::string        deconv_variant_name_;
    bool               spread_cache_hit_    = false;
    bool               deconv_cache_hit_    = false;
    bool               use_precomp_weights_ = false;
    size_t             precompute_wg_       = 64;
    double             output_scale_        = 1.0;

    sycl_fft::UsmPtr<uint8_t>  d_pswf_tbl_;
    sycl_fft::UsmPtr<ScalarT>  d_weights_;
    sycl_fft::UsmPtr<ScalarT>  d_slot_phi_hat_;
    sycl_fft::UsmPtr<ScalarT>  d_slot_deconv_;
    sycl_fft::UsmPtr<ScalarT>  d_nat_deconv_;
    sycl_fft::UsmPtr<float>    d_slot_deconv_f32_;
    sycl_fft::UsmPtr<int>      d_first_;
    sycl_fft::UsmPtr<ScalarT>  d_u_;
    sycl_fft::UsmPtr<ComplexT> d_shift_;
    sycl_fft::UsmPtr<int>      d_bucket_counts_;
    sycl_fft::UsmPtr<int>      d_bucket_off_;
    sycl_fft::UsmPtr<int>      d_bucket_m_;
    sycl_fft::UsmPtr<int>      d_b_first_;
    sycl_fft::UsmPtr<ComplexT> d_b_shift_;
    sycl_fft::UsmPtr<ScalarT>  d_b_weights_;
    sycl_fft::UsmPtr<RealT>    d_grid_T_re_;
    sycl_fft::UsmPtr<RealT>    d_grid_T_im_;
    sycl_fft::UsmPtr<ComplexT> d_fft_buf_;
};

#if SYCL_FFT_ENABLE_POS_SIGN
extern template class Nufft1PlanBase<detail::HalfPlanTraits<+1>>;
extern template class Nufft1PlanBase<detail::FloatPlanTraits<+1>>;
extern template class Nufft1PlanBase<detail::DoublePlanTraits<+1>>;
extern template class Nufft1PlanBase<detail::DfPlanTraits<+1>>;
#endif
#if SYCL_FFT_ENABLE_NEG_SIGN
extern template class Nufft1PlanBase<detail::HalfPlanTraits<-1>>;
extern template class Nufft1PlanBase<detail::FloatPlanTraits<-1>>;
extern template class Nufft1PlanBase<detail::DoublePlanTraits<-1>>;
extern template class Nufft1PlanBase<detail::DfPlanTraits<-1>>;
#endif

template <int Sign = SYCL_FFT_DEFAULT_SIGN>
using Nufft1PlanhSign  = Nufft1PlanBase<detail::HalfPlanTraits<Sign>>;
template <int Sign = SYCL_FFT_DEFAULT_SIGN>
using Nufft1PlanfSign  = Nufft1PlanBase<detail::FloatPlanTraits<Sign>>;
template <int Sign = SYCL_FFT_DEFAULT_SIGN>
using Nufft1PlanSign   = Nufft1PlanBase<detail::DoublePlanTraits<Sign>>;
template <int Sign = SYCL_FFT_DEFAULT_SIGN>
using Nufft1PlandfSign = Nufft1PlanBase<detail::DfPlanTraits<Sign>>;

using Nufft1Planh  = Nufft1PlanhSign<SYCL_FFT_DEFAULT_SIGN>;
using Nufft1Planf  = Nufft1PlanfSign<SYCL_FFT_DEFAULT_SIGN>;
using Nufft1Plan   = Nufft1PlanSign<SYCL_FFT_DEFAULT_SIGN>;
using Nufft1Plandf = Nufft1PlandfSign<SYCL_FFT_DEFAULT_SIGN>;

} // namespace sycl_nufft
