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
    SlotBitReversed = 0, // Primary fast path: raw DIF slot order (no bit-reversal pass)
    Natural         = 1  // Verification path: natural frequency order k in [0, N_out)
};

// Hardware-generalized NuFFT-1 launch configuration (all fields auto-derived when 0/empty)
struct Nufft1LaunchConfig {
    sycl_fft::FftLaunchConfig fft_cfg{};
    std::string spread_variant_override;  // e.g. "smem_wg128_win256" or "collab_wg128"
    std::string deconv_variant_override;  // e.g. "deconv_wg64_bt64"
    size_t      chunk_bytes_override = 0; // 0 = auto from DeviceFftConfig L2 & CUs
    bool        force_collab_spread  = false;

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
    bool use_smem         = true;
    int  wg_size          = 64;
    int  t_j              = 8;
    int  b_per_wg         = 8;
    int  win_elems        = 1024;
    int  num_wins         = 1;
    int  grid_stride      = 1024;
    int  tile_m           = 32;
    int  transpose_tile   = 16;
    int  transpose_stride = 17;
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
    int M, int nfft, int w, int output_shift, bool mode43,
    int* d_first, sycl_fft::f16_scalar_t* d_u, sycl_fft::Complexh* d_shift,
    size_t wg_size);

void launch_precompute_pointsf(
    sycl::queue& q, const sycl_fft::df_t* d_x, sycl_fft::df_t df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43,
    int* d_first, float* d_u, sycl_fft::Complexf* d_shift,
    size_t wg_size);

void launch_precompute_points(
    sycl::queue& q, const double* d_x, double df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43,
    int* d_first, double* d_u, sycl_fft::Complex* d_shift,
    size_t wg_size);

void launch_precompute_pointsdf(
    sycl::queue& q, const sycl_fft::df_t* d_x, sycl_fft::df_t df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43,
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

// --- Decoupled spreading stage launchers (sycl_nufft1_spread_{fp16,fp32,fp64,df}.cpp) ---
void launch_spread_smem_winh(
    sycl::queue& q, int w,
    const int* d_first, const sycl_fft::f16_scalar_t* d_u, const sycl_fft::Complexh* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const sycl_fft::Complexh* d_y, sycl_fft::Complexh* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_collabh(
    sycl::queue& q, int w,
    const int* d_first, const sycl_fft::f16_scalar_t* d_u, const sycl_fft::Complexh* d_shift,
    const sycl_fft::Complexh* d_y,
    sycl_fft::f16x2_t* d_grid_T_re, sycl_fft::f16x2_t* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_smem_winf(
    sycl::queue& q, int w,
    const int* d_first, const float* d_u, const sycl_fft::Complexf* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const sycl_fft::Complexf* d_y, sycl_fft::Complexf* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_collabf(
    sycl::queue& q, int w,
    const int* d_first, const float* d_u, const sycl_fft::Complexf* d_shift,
    const sycl_fft::Complexf* d_y,
    float* d_grid_T_re, float* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_smem_win(
    sycl::queue& q, int w,
    const int* d_first, const double* d_u, const sycl_fft::Complex* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const sycl_fft::Complex* d_y, sycl_fft::Complex* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_collab(
    sycl::queue& q, int w,
    const int* d_first, const double* d_u, const sycl_fft::Complex* d_shift,
    const sycl_fft::Complex* d_y,
    double* d_grid_T_re, double* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_smem_windf(
    sycl::queue& q, int w,
    const int* d_first, const sycl_fft::df_t* d_u, const sycl_fft::Complexdf* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const sycl_fft::Complexdf* d_y, sycl_fft::Complexdf* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg);

void launch_spread_collabdf(
    sycl::queue& q, int w,
    const int* d_first, const sycl_fft::df_t* d_u, const sycl_fft::Complexdf* d_shift,
    const sycl_fft::Complexdf* d_y,
    sycl_fft::df_t* d_grid_T_re, sycl_fft::df_t* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg);

// --- Decoupled GL-free deconvolution stage launchers (sycl_nufft1_deconv_{fp16,fp32,fp64,df}.cpp) ---
void launch_deconv_emith(
    sycl::queue& q, int w,
    const sycl_fft::Complexh* d_fft_buf, sycl_fft::Complexh* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg);

void launch_deconv_emitf(
    sycl::queue& q, int w,
    const sycl_fft::Complexf* d_fft_buf, sycl_fft::Complexf* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg);

void launch_deconv_emit(
    sycl::queue& q, int w,
    const sycl_fft::Complex* d_fft_buf, sycl_fft::Complex* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg);

void launch_deconv_emitdf(
    sycl::queue& q, int w,
    const sycl_fft::Complexdf* d_fft_buf, sycl_fft::Complexdf* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg);

} // namespace detail

namespace detail {
// --- Precision tags for Nufft1PlanBase ---
struct HalfPlanTraits {
    using ComplexT = sycl_fft::Complexh;
    using ScalarT  = sycl_fft::f16_scalar_t;
    using RealT    = sycl_fft::f16x2_t;
    using XCoordT  = sycl_fft::df_t;
    using FftPlanT = sycl_fft::DifFftPlanh<+1>;
    static constexpr size_t kLanes   = 2;
    static constexpr bool   kIsFp64  = false;
    static constexpr bool   kIsHalf  = true;
    static constexpr bool   kIsDf    = false;
    static constexpr const char* kPrecTag = "h";
    static bool is_valid_width(int w) { return w == 4 || w == 5; }
};

struct FloatPlanTraits {
    using ComplexT = sycl_fft::Complexf;
    using ScalarT  = float;
    using RealT    = float;
    using XCoordT  = sycl_fft::df_t;
    using FftPlanT = sycl_fft::DifFftPlanf<+1>;
    static constexpr size_t kLanes   = 1;
    static constexpr bool   kIsFp64  = false;
    static constexpr bool   kIsHalf  = false;
    static constexpr bool   kIsDf    = false;
    static constexpr const char* kPrecTag = "f";
    static bool is_valid_width(int w) { return w == 8 || w == 9; }
};

struct DoublePlanTraits {
    using ComplexT = sycl_fft::Complex;
    using ScalarT  = double;
    using RealT    = double;
    using XCoordT  = double;
    using FftPlanT = sycl_fft::DifFftPlan<+1>;
    static constexpr size_t kLanes   = 1;
    static constexpr bool   kIsFp64  = true;
    static constexpr bool   kIsHalf  = false;
    static constexpr bool   kIsDf    = false;
    static constexpr const char* kPrecTag = "d";
    static bool is_valid_width(int w) { return w == 16 || w == 17 || w == 18; }
};

struct DfPlanTraits {
    using ComplexT = sycl_fft::Complexdf;
    using ScalarT  = sycl_fft::df_t;
    using RealT    = sycl_fft::df_t;
    using XCoordT  = sycl_fft::df_t;
    using FftPlanT = sycl_fft::DifFftPlandf<+1>;
    static constexpr size_t kLanes   = 1;
    static constexpr bool   kIsFp64  = false;
    static constexpr bool   kIsHalf  = false;
    static constexpr bool   kIsDf    = true;
    static constexpr const char* kPrecTag = "df";
    static bool is_valid_width(int w) { return w == 14 || w == 15 || w == 16; }
};

} // namespace detail

// Pure-host orchestrated NuFFT-1 plan (instantiated in sycl_nufft1_plan.cpp, 0 KiB HCF)
template <typename Traits>
class Nufft1PlanBase {
public:
    using ComplexT = typename Traits::ComplexT;
    using ScalarT  = typename Traits::ScalarT;
    using RealT    = typename Traits::RealT;
    using XCoordT  = typename Traits::XCoordT;
    using FftPlanT = typename Traits::FftPlanT;

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

    int M() const { return M_; }
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
    static Nufft1LaunchConfig make_default_cfg(const sycl_fft::FftLaunchConfig& fft_cfg) {
        Nufft1LaunchConfig c{};
        c.fft_cfg = fft_cfg;
        return c;
    }

    void allocate_collab_grid();
    void allocate_window_buckets();

    sycl::queue&       q_;
    int                M_;
    int                N_;
    int                w_;
    PswfMode           mode_;
    int                nfft_;
    int                log_nfft_;
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
    bool               spread_cache_hit_ = false;
    bool               deconv_cache_hit_ = false;
    size_t             precompute_wg_    = 64;

    int*               d_first_          = nullptr;
    ScalarT*           d_u_              = nullptr;
    ComplexT*          d_shift_          = nullptr;
    int*               d_bucket_counts_  = nullptr;
    int*               d_bucket_off_     = nullptr;
    int*               d_bucket_m_       = nullptr;
    RealT*             d_grid_T_re_      = nullptr;
    RealT*             d_grid_T_im_      = nullptr;
    ComplexT*          d_fft_buf_        = nullptr;
};

extern template class Nufft1PlanBase<detail::HalfPlanTraits>;
extern template class Nufft1PlanBase<detail::FloatPlanTraits>;
extern template class Nufft1PlanBase<detail::DoublePlanTraits>;
extern template class Nufft1PlanBase<detail::DfPlanTraits>;

using Nufft1Planh  = Nufft1PlanBase<detail::HalfPlanTraits>;
using Nufft1Planf  = Nufft1PlanBase<detail::FloatPlanTraits>;
using Nufft1Plan   = Nufft1PlanBase<detail::DoublePlanTraits>;
using Nufft1Plandf = Nufft1PlanBase<detail::DfPlanTraits>;

} // namespace sycl_nufft
