// Multi-includable SYCL Type-1 NuFFT Gauss-Legendre-Free Deconvolution Module.
// Included once per precision by sycl_nufft1_deconv_{fp16,fp32,fp64,df}.cpp
// so each precision's deconvolution kernels live in their own minimal HCF module,
// decoupled from spreading kernels and FFT kernels.
// ZERO hardcoded GPU/Blackwell constants: wg_i and batch_tile come from DeconvConfig.

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

// FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution pass templated on width W:
//   - Evaluates deconv(k') = inv_prefac / phi(eta_{k'}) ONCE per thread in registers
//     (zero global LUT reads, zero Gauss-Legendre quadrature) and streams across
//     cfg.batch_tile packed transforms (amortizing polynomial + division cost across batch).
template <int W>
inline void SYCL_FFT_SUFFIX(launch_deconv_emit_impl)(
    sycl::queue& q,
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
    const DeconvConfig& cfg) {
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
    const auto tbl = get_pswf_table_f16<W>(mode);
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
#elif defined(SYCL_FFT_DOUBLE)
    const auto tbl = get_pswf_table_f64<W>(mode);
    const double inv_nfft = 1.0 / static_cast<double>(nfft);
#elif defined(SYCL_FFT_DF)
    const auto tbl = get_pswf_table_df<W>(mode);
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
#else
    const auto tbl = get_pswf_table_f32<W>(mode);
    const float inv_nfft = 1.0f / static_cast<float>(nfft);
#endif

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
                if (order == NufftOutOrder::SlotBitReversed) {
                    slot_p = nufft1_compact_idx_to_slot(i, mode);
                    k_prime = nufft1_slot_to_k_prime(slot_p, log_nfft, nfft, output_shift);
                } else {
                    k_prime = i - output_shift;
                    const int bin = (k_prime >= 0) ? k_prime : (nfft + k_prime);
                    slot_p = static_cast<int>(
                        sycl_fft::bit_reverse(static_cast<uint32_t>(bin), log_nfft));
                }
                const int abs_k_prime = (k_prime >= 0) ? k_prime : -k_prime;

#if defined(SYCL_FFT_HALF)
                const ScalarT deconv_s = eval_pswf_deconv_f16<W>(abs_k_prime, inv_nfft, tbl);
                const RealT deconv_v = SYCL_FFT_F16X2_SPLAT_H(deconv_s);
#elif defined(SYCL_FFT_DOUBLE)
                const RealT deconv_v = eval_pswf_deconv_f64<W>(abs_k_prime, inv_nfft, tbl);
#elif defined(SYCL_FFT_DF)
                const RealT deconv_v = eval_pswf_deconv_df<W>(abs_k_prime, inv_nfft, tbl);
#else
                const RealT deconv_v = eval_pswf_deconv_f32<W>(abs_k_prime, inv_nfft, tbl);
#endif

                const size_t bp_start = b_tile_idx * batch_tile;
                const size_t bp_end   = sycl::min(bs_packed, bp_start + batch_tile);

                for (size_t bp = bp_start; bp < bp_end; ++bp) {
#if defined(SYCL_FFT_HALF)
                    const size_t b0 = 2 * bp;
                    const size_t b1 = 2 * bp + 1;
                    RealT fr, fi;
                    sycl_fft::f16x2_load_pair(
                        d_fft_buf + b0 * static_cast<size_t>(nfft),
                        d_fft_buf + b1 * static_cast<size_t>(nfft),
                        slot_p,
                        b0 < chunk_bs,
                        b1 < chunk_bs,
                        fr,
                        fi);
                    fr = SYCL_FFT_MUL(fr, deconv_v);
                    fi = SYCL_FFT_MUL(fi, deconv_v);
                    sycl_fft::f16x2_store_pair(
                        d_out + b0 * total_nout_stride,
                        d_out + b1 * total_nout_stride,
                        i,
                        b0 < chunk_bs,
                        b1 < chunk_bs,
                        fr,
                        fi);
#elif defined(SYCL_FFT_DF)
                    const ComplexT fval = d_fft_buf[bp * static_cast<size_t>(nfft) + static_cast<size_t>(slot_p)];
                    d_out[bp * total_nout_stride + static_cast<size_t>(i)] = ComplexT{
                        SYCL_FFT_MUL(fval.re, deconv_v),
                        SYCL_FFT_MUL(fval.im, deconv_v)
                    };
#else
                    const ComplexT fval = d_fft_buf[bp * static_cast<size_t>(nfft) + static_cast<size_t>(slot_p)];
                    d_out[bp * total_nout_stride + static_cast<size_t>(i)] = ComplexT(
                        SYCL_FFT_MUL(fval.x(), deconv_v),
                        SYCL_FFT_MUL(fval.y(), deconv_v));
#endif
                }
            });
    });
}

void SYCL_FFT_SUFFIX(launch_deconv_emit)(
    sycl::queue& q, int w,
    const SYCL_NUFFT_COMPLEX_T* d_fft_buf, SYCL_NUFFT_COMPLEX_T* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<4>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<5>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    }
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<16>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    } else if (w == 17) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<17>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<18>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    }
#elif defined(SYCL_FFT_DF)
    if (w == 14) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<14>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    } else if (w == 15) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<15>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<16>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    }
#else
    if (w == 8) {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<8>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_deconv_emit_impl)<9>(
            q, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift,
            total_nout_stride, chunk_bs, mode, order, cfg);
    }
#endif
}

#undef SYCL_NUFFT_COMPLEX_T
#undef SYCL_NUFFT_REAL_T
#undef SYCL_NUFFT_SCALAR_T
#undef SYCL_NUFFT_LANES

} // namespace detail
} // namespace sycl_nufft
