// Decoupled SYCL Type-1 NuFFT Gauss-Legendre-Free Deconvolution Module.
// Compiled once per precision with -D flags so each precision's deconvolution
// kernels live in their own minimal HCF module, decoupled from spreading & FFT:
//   -DSYCL_FFT_HALF | -DSYCL_FFT_FLOAT | -DSYCL_FFT_DOUBLE | -DSYCL_FFT_DF
// Defaults to SYCL_FFT_DF (Double-Float) when no half/float/double is selected.
// ZERO hardcoded GPU/Blackwell constants: wg_i and batch_tile come from DeconvConfig.

#if !defined(SYCL_FFT_HALF) && !defined(SYCL_FFT_FLOAT) && \
    !defined(SYCL_FFT_DOUBLE) && !defined(SYCL_FFT_DF)
#  define SYCL_FFT_DF 1
#endif

#include "nufft/nufft1.hpp"
#include "nufft/pswf_coeffs.hpp"
#include "common/precision.hpp"

namespace sycl_nufft {
namespace detail {

using namespace sycl_fft;

#if defined(SYCL_FFT_HALF)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexh
#define SYCL_NUFFT_REAL_T    sycl_fft::f16x2_t
#define SYCL_NUFFT_SCALAR_T  sycl_fft::f16_scalar_t
#define SYCL_NUFFT_LANES     2
#elif defined(SYCL_FFT_DOUBLE)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complex
#define SYCL_NUFFT_REAL_T    double
#define SYCL_NUFFT_SCALAR_T  double
#define SYCL_NUFFT_LANES     1
#elif defined(SYCL_FFT_DF)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexdf
#define SYCL_NUFFT_REAL_T    sycl_fft::df_t
#define SYCL_NUFFT_SCALAR_T  sycl_fft::df_t
#define SYCL_NUFFT_LANES     1
#else
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexf
#define SYCL_NUFFT_REAL_T    float
#define SYCL_NUFFT_SCALAR_T  float
#define SYCL_NUFFT_LANES     1
#endif

// Reusable 1D LUT builder (length nout) for fused FFT epilogue, deferred deconvolution,
// and multi-NuFFT index-wise combination with a single final bit-reversal pass:
//   table_kind == 0: d_out[i] = phi_hat(k'_i) in compact SlotBitReversed order
//   table_kind == 1: d_out[i] = output_scale / phi_hat(k'_i) in compact SlotBitReversed order
//   table_kind == 2: d_out[i] = output_scale / phi_hat(k') in Natural frequency order
template <int W>
inline void SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)(
    sycl::queue& q,
    const void* d_tbl_raw,
    int nfft,
    int log_nfft,
    int nout,
    int output_shift,
    PswfMode mode,
    int table_kind,
    SYCL_NUFFT_SCALAR_T* d_out,
    double output_scale) {
    using ScalarT = SYCL_NUFFT_SCALAR_T;
#if defined(SYCL_FFT_HALF)
    using TableT = std::decay_t<decltype(get_pswf_table_f16<W>(PswfMode::PSWF21))>;
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
    const float scale_f = static_cast<float>(output_scale);
#elif defined(SYCL_FFT_DOUBLE)
    using TableT = std::decay_t<decltype(get_pswf_table_f64<W>(PswfMode::PSWF21))>;
    const double inv_nfft = 1.0 / static_cast<double>(nfft);
    const double scale_d = output_scale;
#elif defined(SYCL_FFT_DF)
    using TableT = std::decay_t<decltype(get_pswf_table_df<W>(PswfMode::PSWF21))>;
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
    const bool apply_scale = (output_scale != 1.0);
    const sycl_fft::df_t scale_df = sycl_fft::df_from_double(output_scale);
#else
    using TableT = std::decay_t<decltype(get_pswf_table_f32<W>(PswfMode::PSWF21))>;
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
    const float scale_f = static_cast<float>(output_scale);
#endif
    const TableT* d_tbl = static_cast<const TableT*>(d_tbl_raw);
    const size_t wg = 64;
    const size_t num_wg = (static_cast<size_t>(nout) + wg - 1) / wg;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * wg, wg),
            [=](sycl::nd_item<1> item) {
                const int i = static_cast<int>(item.get_global_linear_id());
                if (i >= nout) return;

                int k_prime = 0;
                if (table_kind == 2) {
                    k_prime = i - output_shift;
                } else {
                    const int slot_p = nufft1_compact_idx_to_slot(i, mode);
                    k_prime = nufft1_slot_to_k_prime(slot_p, log_nfft, nfft, output_shift);
                }
                const int abs_k_prime = (k_prime >= 0) ? k_prime : -k_prime;

                ScalarT val{};
                if (table_kind == 0) {
#if defined(SYCL_FFT_HALF)
                    val = eval_pswf_phi_hat_f16<W>(abs_k_prime, inv_nfft, *d_tbl);
#elif defined(SYCL_FFT_DOUBLE)
                    val = eval_pswf_phi_hat_f64<W>(abs_k_prime, inv_nfft, *d_tbl);
#elif defined(SYCL_FFT_DF)
                    val = eval_pswf_phi_hat_df<W>(abs_k_prime, inv_nfft, *d_tbl);
#else
                    val = eval_pswf_phi_hat_f32<W>(abs_k_prime, inv_nfft, *d_tbl);
#endif
                } else {
#if defined(SYCL_FFT_HALF)
                    val = eval_pswf_deconv_f16<W>(abs_k_prime, inv_nfft, *d_tbl, scale_f);
#elif defined(SYCL_FFT_DOUBLE)
                    val = eval_pswf_deconv_f64<W>(abs_k_prime, inv_nfft, *d_tbl) * scale_d;
#elif defined(SYCL_FFT_DF)
                    val = eval_pswf_deconv_df<W>(abs_k_prime, inv_nfft, *d_tbl);
                    if (apply_scale) {
                        val = sycl_fft::df_mul(val, scale_df);
                    }
#else
                    val = eval_pswf_deconv_f32<W>(abs_k_prime, inv_nfft, *d_tbl) * scale_f;
#endif
                }
                d_out[i] = val;
            });
    });
}

// FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution pass templated on width W:
//   - Used for Natural output order and MultiStageGlobal fallback (SinglePass/TwoPass
//     SlotBitReversed & SlotBitReversedRaw fuse directly into the FFT final store).
template <int W>
inline void SYCL_FFT_SUFFIX(launch_deconv_emit_impl)(
    sycl::queue& q,
    const void* d_tbl_raw,
    const SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    SYCL_NUFFT_COMPLEX_T* d_out,
    int nfft,
    int log_nfft,
    int nout,
    int output_shift,
    size_t total_nout_stride,
    size_t chunk_bs,
    PswfMode mode,
    NufftOutOrder order,
    const DeconvConfig& cfg,
    double output_scale = 1.0) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    using RealT    = SYCL_NUFFT_REAL_T;
    (void)sizeof(ScalarT);
    (void)sizeof(ComplexT);

    constexpr int kLanes = SYCL_NUFFT_LANES;
    const size_t wg_i       = static_cast<size_t>(std::max(1, cfg.wg_i));
    const size_t batch_tile = static_cast<size_t>(std::max(1, cfg.batch_tile));
    const size_t bs_packed  = (chunk_bs + kLanes - 1) / kLanes;
    const size_t groups_i   = (static_cast<size_t>(nout) + wg_i - 1) / wg_i;
    const size_t groups_b   = (bs_packed + batch_tile - 1) / batch_tile;

#if defined(SYCL_FFT_HALF)
    using TableT = std::decay_t<decltype(get_pswf_table_f16<W>(PswfMode::PSWF21))>;
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
    const float scale_f = static_cast<float>(output_scale);
#elif defined(SYCL_FFT_DOUBLE)
    using TableT = std::decay_t<decltype(get_pswf_table_f64<W>(PswfMode::PSWF21))>;
    const double inv_nfft = 1.0 / static_cast<double>(nfft);
    const double scale_d = output_scale;
#elif defined(SYCL_FFT_DF)
    using TableT = std::decay_t<decltype(get_pswf_table_df<W>(PswfMode::PSWF21))>;
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
    const bool apply_scale = (output_scale != 1.0);
    const sycl_fft::df_t scale_df = sycl_fft::df_from_double(output_scale);
#else
    using TableT = std::decay_t<decltype(get_pswf_table_f32<W>(PswfMode::PSWF21))>;
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
    const float scale_f = static_cast<float>(output_scale);
#endif
    const TableT* d_tbl = static_cast<const TableT*>(d_tbl_raw);
    const bool raw_mode = (order == NufftOutOrder::SlotBitReversedRaw);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_b, groups_i * wg_i),
                              sycl::range<2>(1, wg_i)),
            [=](sycl::nd_item<2> item) {
                const size_t b_tile_idx = item.get_global_id(0);
                const int i = static_cast<int>(item.get_global_id(1));
                if (i >= nout) return;

                int slot_p = 0;
                int k_prime = 0;
                if (order == NufftOutOrder::SlotBitReversed || raw_mode) {
                    slot_p = nufft1_compact_idx_to_slot(i, mode);
                    k_prime = nufft1_slot_to_k_prime(slot_p, log_nfft, nfft, output_shift);
                } else {
                    k_prime = i - output_shift;
                    const int bin = (k_prime >= 0) ? k_prime : (nfft + k_prime);
                    slot_p = static_cast<int>(
                        sycl_fft::bit_reverse(static_cast<uint32_t>(bin), log_nfft));
                }
                const int abs_k_prime = (k_prime >= 0) ? k_prime : -k_prime;

                RealT deconv_v = SYCL_FFT_ONE;
                if (!raw_mode) {
#if defined(SYCL_FFT_HALF)
                    const ScalarT deconv_s = eval_pswf_deconv_f16<W>(abs_k_prime, inv_nfft, *d_tbl, scale_f);
                    deconv_v = SYCL_FFT_F16X2_SPLAT_H(deconv_s);
#elif defined(SYCL_FFT_DOUBLE)
                    deconv_v = eval_pswf_deconv_f64<W>(abs_k_prime, inv_nfft, *d_tbl) * scale_d;
#elif defined(SYCL_FFT_DF)
                    deconv_v = eval_pswf_deconv_df<W>(abs_k_prime, inv_nfft, *d_tbl);
                    if (apply_scale) {
                        deconv_v = sycl_fft::df_mul(deconv_v, scale_df);
                    }
#else
                    deconv_v = eval_pswf_deconv_f32<W>(abs_k_prime, inv_nfft, *d_tbl) * scale_f;
#endif
                }

                const size_t bp_start = b_tile_idx * batch_tile;
                const size_t bp_end   = sycl::min(bs_packed, bp_start + batch_tile);

                for (size_t bp = bp_start; bp < bp_end; ++bp) {
#if defined(SYCL_FFT_HALF)
                    const size_t b0 = 2 * bp;
                    const size_t b1 = 2 * bp + 1;
                    const bool v0 = (b0 < chunk_bs);
                    const bool v1 = (b1 < chunk_bs);
                    const ComplexT* src0 = d_fft_buf + b0 * static_cast<size_t>(nfft);
                    const ComplexT* src1 = v1 ? (d_fft_buf + b1 * static_cast<size_t>(nfft)) : src0;
                    ComplexT* dst0 = d_out + b0 * total_nout_stride;
                    ComplexT* dst1 = v1 ? (d_out + b1 * total_nout_stride) : dst0;
                    RealT fr, fi;
                    sycl_fft::f16x2_load_pair(src0, src1, slot_p, v0, v1, fr, fi);
                    if (!raw_mode) {
                        fr = SYCL_FFT_MUL(fr, deconv_v);
                        fi = SYCL_FFT_MUL(fi, deconv_v);
                    }
                    sycl_fft::f16x2_store_pair(dst0, dst1, i, v0, v1, fr, fi);
#elif defined(SYCL_FFT_DF)
                    const ComplexT fval = d_fft_buf[bp * static_cast<size_t>(nfft) + static_cast<size_t>(slot_p)];
                    d_out[bp * total_nout_stride + static_cast<size_t>(i)] = raw_mode
                        ? fval
                        : ComplexT{SYCL_FFT_MUL(fval.re, deconv_v), SYCL_FFT_MUL(fval.im, deconv_v)};
#else
                    const ComplexT fval = d_fft_buf[bp * static_cast<size_t>(nfft) + static_cast<size_t>(slot_p)];
                    d_out[bp * total_nout_stride + static_cast<size_t>(i)] = raw_mode
                        ? fval
                        : ComplexT(SYCL_FFT_MUL(fval.x(), deconv_v), SYCL_FFT_MUL(fval.y(), deconv_v));
#endif
                }
            });
    });
}

void SYCL_FFT_SUFFIX(launch_build_deconv_table)(
    sycl::queue& q, int w, const void* d_tbl,
    int nfft, int log_nfft, int nout, int output_shift,
    PswfMode mode, int table_kind, SYCL_NUFFT_SCALAR_T* d_out,
    double output_scale) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<4>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<5>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    }
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<16>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else if (w == 17) {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<17>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<18>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    }
#elif defined(SYCL_FFT_DF)
    if (w == 14) {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<14>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else if (w == 15) {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<15>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<16>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    }
#else
    if (w == 8) {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<8>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else {
        SYCL_FFT_SUFFIX(launch_build_deconv_table_impl)<9>(
            q, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    }
#endif
}

void SYCL_FFT_SUFFIX(launch_deconv_emit)(
    sycl::queue& q, int w, const void* d_tbl,
    const SYCL_NUFFT_COMPLEX_T* d_fft_buf, SYCL_NUFFT_COMPLEX_T* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg,
    double output_scale) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<4>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<5>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    }
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<16>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else if (w == 17) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<17>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<18>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    }
#elif defined(SYCL_FFT_DF)
    if (w == 14) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<14>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else if (w == 15) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<15>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<16>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    }
#else
    if (w == 8) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<8>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<9>(
            q, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    }
#endif
}

#undef SYCL_NUFFT_COMPLEX_T
#undef SYCL_NUFFT_REAL_T
#undef SYCL_NUFFT_SCALAR_T
#undef SYCL_NUFFT_LANES

} // namespace detail
} // namespace sycl_nufft
