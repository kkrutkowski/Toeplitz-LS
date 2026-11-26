// Decoupled SYCL Type-1 NuFFT Spreading Kernel Module.
// Compiled once per precision with -D flags so each precision's spreading kernels
// live in their own lean HCF module:
//   -DSYCL_FFT_HALF | -DSYCL_FFT_FLOAT | -DSYCL_FFT_DOUBLE | -DSYCL_FFT_DF
// Defaults to SYCL_FFT_DF (Double-Float) when no half/float/double is selected.
//
// Features:
//   1. Precision-specific PSWF table upload: exact geometric-mean-1 rescaling for
//      FP32/FP64/DF (sqrt(phi_hat(0) * phi_hat(edge)) == 1), preserving subnormal-safe
//      2^e scaling for FP16.
//   2. L2-resident precomputed d_weights[M * W] evaluator (eliminates redundant per-b_group
//      PSWF polynomial evaluations and frees l_w SMEM in the hot spreading kernel).
//   3. Bank-conflict-free transposed SMEM layouts:
//        - l_sy[k * (b_per_wg + 1) + lane_b]: 0 conflicts in Phase 1b & Phase 2 + coalesced d_y
//        - l_w[j * (tile_m + 1) + k]: 0 conflicts in on-the-fly fallback
//   4. Optional multi-bin-per-lane (t_j = prev_pow2(W) < W) for w in {5, 9, 17, 18}
//      to eliminate 37.5%-47% lane underutilization and double b_per_wg.
//   5. Barrier-free, zero-SMEM direct CPU spreading path when !has_native_local_mem.

#if !defined(SYCL_FFT_HALF) && !defined(SYCL_FFT_FLOAT) && \
    !defined(SYCL_FFT_DOUBLE) && !defined(SYCL_FFT_DF)
#  define SYCL_FFT_DF 1
#endif

#include "nufft/nufft1.hpp"
#include "nufft/pswf_coeffs.hpp"
#include "common/precision.hpp"

#include <type_traits>

namespace sycl_nufft {
namespace detail {

using namespace sycl_fft;

#if defined(SYCL_FFT_HALF)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexh
#define SYCL_NUFFT_REAL_T    sycl_fft::f16x2_t
#define SYCL_NUFFT_SCALAR_T  sycl_fft::f16_scalar_t
#define SYCL_NUFFT_LANES     2
#define SYCL_NUFFT_FMA(a, b, c) SYCL_FFT_F16X2_FMA((a), (b), (c))
#elif defined(SYCL_FFT_DOUBLE)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complex
#define SYCL_NUFFT_REAL_T    double
#define SYCL_NUFFT_SCALAR_T  double
#define SYCL_NUFFT_LANES     1
#define SYCL_NUFFT_FMA(a, b, c) sycl::fma((a), (b), (c))
#elif defined(SYCL_FFT_DF)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexdf
#define SYCL_NUFFT_REAL_T    sycl_fft::df_t
#define SYCL_NUFFT_SCALAR_T  sycl_fft::df_t
#define SYCL_NUFFT_LANES     1
#define SYCL_NUFFT_FMA(a, b, c) sycl_fft::df_add(sycl_fft::df_mul((a), (b)), (c))
#else
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexf
#define SYCL_NUFFT_REAL_T    float
#define SYCL_NUFFT_SCALAR_T  float
#define SYCL_NUFFT_LANES     1
#define SYCL_NUFFT_FMA(a, b, c) sycl::fma((a), (b), (c))
#endif

// Helper: load d_y and multiply by d_shift for batch pair / single batch
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(load_shifted_y)(
    const SYCL_NUFFT_COMPLEX_T* d_y,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    size_t bp,
    size_t total_M_stride,
    size_t chunk_bs,
    int m_idx,
    SYCL_NUFFT_REAL_T& sy_re,
    SYCL_NUFFT_REAL_T& sy_im) {
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    const ComplexT sh = d_shift[m_idx];
#if defined(SYCL_FFT_HALF)
    using RealT = SYCL_NUFFT_REAL_T;
    const RealT sh_v = SYCL_FFT_BIT_CAST(RealT, sh);
    const SYCL_NUFFT_SCALAR_T sr_h = sycl_fft::f16x2_lo(sh_v);
    const SYCL_NUFFT_SCALAR_T si_h = sycl_fft::f16x2_hi(sh_v);
    const RealT sr_v = SYCL_FFT_F16X2_SPLAT_H(sr_h);
    const RealT si_v = SYCL_FFT_F16X2_SPLAT_H(si_h);
    const RealT neg_si_v = SYCL_FFT_BIT_CAST(
        RealT, SYCL_FFT_BIT_CAST(uint32_t, si_v) ^ 0x80008000u);
    const size_t b0 = 2 * bp;
    const size_t b1 = 2 * bp + 1;
    const bool v0 = (b0 < chunk_bs);
    const bool v1 = (b1 < chunk_bs);
    const ComplexT* y0 = d_y + b0 * total_M_stride;
    const ComplexT* y1 = v1 ? (d_y + b1 * total_M_stride) : y0;
    RealT y_re, y_im;
    sycl_fft::f16x2_load_pair(y0, y1, m_idx, v0, v1, y_re, y_im);
    sy_re = SYCL_FFT_F16X2_FMA(
        y_re, sr_v, SYCL_FFT_F16X2_FMA(y_im, neg_si_v, sycl_fft::kZero_f16x2));
    sy_im = SYCL_FFT_F16X2_FMA(
        y_re, si_v, SYCL_FFT_F16X2_FMA(y_im, sr_v, sycl_fft::kZero_f16x2));
#elif defined(SYCL_FFT_DF)
    (void)chunk_bs;
    const ComplexT y = d_y[bp * total_M_stride + static_cast<size_t>(m_idx)];
    sycl_fft::df_cmul_fused(y.re, y.im, sh.re, sh.im, sy_re, sy_im);
#else
    (void)chunk_bs;
    const ComplexT y = d_y[bp * total_M_stride + static_cast<size_t>(m_idx)];
    SYCL_FFT_CMUL(y.x(), y.y(), sh.x(), sh.y(), sy_re, sy_im);
#endif
}

// ============================================================================
// 1. Precompute L2-Resident Spreading Weights d_weights[M * W] (once per set_points)
// ============================================================================
template <int W>
inline void SYCL_FFT_SUFFIX(launch_precompute_weights_impl)(
    sycl::queue& q,
    const void* d_tbl_raw,
    const SYCL_NUFFT_SCALAR_T* d_u,
    SYCL_NUFFT_SCALAR_T* d_weights,
    int M,
    size_t wg_size) {
    using ScalarT = SYCL_NUFFT_SCALAR_T;
#if defined(SYCL_FFT_HALF)
    using TableT = std::decay_t<decltype(get_pswf_table_f16<W>(PswfMode::PSWF21))>;
#elif defined(SYCL_FFT_DOUBLE)
    using TableT = std::decay_t<decltype(get_pswf_table_f64<W>(PswfMode::PSWF21))>;
#elif defined(SYCL_FFT_DF)
    using TableT = std::decay_t<decltype(get_pswf_table_df<W>(PswfMode::PSWF21))>;
#else
    using TableT = std::decay_t<decltype(get_pswf_table_f32<W>(PswfMode::PSWF21))>;
#endif
    const TableT* d_tbl = static_cast<const TableT*>(d_tbl_raw);
    const size_t wg = std::max<size_t>(1, wg_size);
    const size_t num_wg = (static_cast<size_t>(M) + wg - 1) / wg;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * wg, wg),
            [=](sycl::nd_item<1> item) {
                const int m = static_cast<int>(item.get_global_linear_id());
                if (m >= M) return;
                const ScalarT u = d_u[m];
                ScalarT w_reg[W];
#if defined(SYCL_FFT_HALF)
                eval_pswf_spread_weights_f16<W>(u, *d_tbl, w_reg);
#elif defined(SYCL_FFT_DOUBLE)
                eval_pswf_spread_weights_f64<W>(u, *d_tbl, w_reg);
#elif defined(SYCL_FFT_DF)
                eval_pswf_spread_weights_df<W>(u, *d_tbl, w_reg);
#else
                eval_pswf_spread_weights_f32<W>(u, *d_tbl, w_reg);
#endif
                ScalarT* dst = d_weights + static_cast<size_t>(m) * static_cast<size_t>(W);
                #pragma unroll
                for (int j = 0; j < W; ++j) {
                    dst[j] = w_reg[j];
                }
            });
    });
}

// ============================================================================
// 2a. Fast Windowed SMEM Spreading with Precomputed d_weights[M * W]
//     - ZERO PSWF polynomial evaluation & ZERO l_w SMEM (ultra-lean PTX / fast JIT!)
//     - l_sy[k * (b_per_wg + 1) + lane_b]: 0 bank conflicts in Phase 1b & Phase 2
//     - Supports MultiPass (t_j = prev_pow2(W) < W) for w in {5, 9, 17, 18}
// ============================================================================
template <int W, bool MultiPass>
inline void SYCL_FFT_SUFFIX(launch_spread_smem_win_precomp_impl)(
    sycl::queue& q,
    const SYCL_NUFFT_SCALAR_T* d_weights,
    const int* d_first,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    const int* d_bucket_off,
    const int* d_bucket_m,
    const SYCL_NUFFT_COMPLEX_T* d_y,
    SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int M,
    int nfft,
    size_t total_M_stride,
    size_t chunk_bs,
    const SpreadConfig& cfg) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    using RealT    = SYCL_NUFFT_REAL_T;

    constexpr int kLanes = SYCL_NUFFT_LANES;
    const int wg_size     = cfg.wg_size;
    const int t_j         = cfg.t_j;
    const int log2_tj     = int_log2_pow2(t_j);
    const int tj_mask     = t_j - 1;
    const int b_per_wg    = cfg.b_per_wg;
    const int sy_stride   = b_per_wg + 1; // Odd when b_per_wg is pow2 >= 2 -> 0 bank conflicts
    const int win_elems   = cfg.win_elems;
    const int num_wins    = cfg.num_wins;
    const int log2_nw     = int_log2_pow2(num_wins);
    const int nw_mask     = num_wins - 1;
    const int grid_stride = cfg.grid_stride;
    const int tile_m      = cfg.tile_m;
    const int log2_tile_m = int_log2_pow2(tile_m);
    const int tile_m_mask = tile_m - 1;
    const int nfft_mask   = nfft - 1;
    const int smem_grid_elems = b_per_wg * grid_stride;
    const int smem_sy_elems   = tile_m * sy_stride;
    const int smem_w_elems    = tile_m * W;

    const size_t bs_packed    = (chunk_bs + kLanes - 1) / kLanes;
    const size_t num_b_groups = (bs_packed + static_cast<size_t>(b_per_wg) - 1) / static_cast<size_t>(b_per_wg);
    const size_t total_wgs    = num_b_groups * static_cast<size_t>(num_wins);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<RealT, 1>   l_re(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_im(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<int, 1>     l_first(sycl::range<1>(static_cast<size_t>(tile_m)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_re(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_im(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
        sycl::local_accessor<ScalarT, 1> l_w(sycl::range<1>(static_cast<size_t>(smem_w_elems)), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(total_wgs * static_cast<size_t>(wg_size), static_cast<size_t>(wg_size)),
            [=](sycl::nd_item<1> item) {
                const auto grp = item.get_group();
                const int tid  = static_cast<int>(item.get_local_linear_id());
                const size_t wg_id   = item.get_group_linear_id();
                const size_t b_group = wg_id >> log2_nw;
                const int win        = static_cast<int>(wg_id & static_cast<size_t>(nw_mask));
                const int win_lo     = win * win_elems;

                const int lane_b     = tid >> log2_tj;
                const int lane_t     = tid & tj_mask;
                const size_t bp_base = b_group * static_cast<size_t>(b_per_wg);
                const size_t my_bp   = bp_base + static_cast<size_t>(lane_b);
                const int grid_base  = lane_b * grid_stride;

                for (int idx = tid; idx < smem_grid_elems; idx += wg_size) {
                    l_re[idx] = SYCL_FFT_ZERO;
                    l_im[idx] = SYCL_FFT_ZERO;
                }
                sycl::group_barrier(grp);

                const int p_start = (num_wins == 1) ? 0 : d_bucket_off[win];
                const int p_end   = (num_wins == 1) ? M : d_bucket_off[win + 1];

                for (int p_base = p_start; p_base < p_end; p_base += tile_m) {
                    const int cur_tile = sycl::min(tile_m, p_end - p_base);

                    // Phase 1a: Load first bins and cooperatively stage precomputed weights
                    // d_weights[m * W + j] -> l_w[k * W + j] ONCE per tile (zero PSWF eval,
                    // coalesced across j, and 1-cycle bank-conflict-free LDS in Phase 2!).
                    if (tid < cur_tile) {
                        const int m_idx = (num_wins == 1) ? (p_base + tid) : d_bucket_m[p_base + tid];
                        l_first[tid] = d_first[m_idx];
                    }
                    const int total_w = cur_tile * W;
                    for (int idx = tid; idx < total_w; idx += wg_size) {
                        const int k = idx / W;
                        const int j = idx - k * W;
                        const int m_idx = (num_wins == 1) ? (p_base + k) : d_bucket_m[p_base + k];
                        l_w[idx] = d_weights[static_cast<size_t>(m_idx) * static_cast<size_t>(W) + static_cast<size_t>(j)];
                    }

                    // Phase 1b: All lanes cooperatively load d_y (coalesced across k),
                    // apply phase shift, and store bank-conflict-free into l_sy[k * sy_stride + b_sub]
                    const int total_sy = b_per_wg * tile_m;
                    for (int idx = tid; idx < total_sy; idx += wg_size) {
                        const int b_sub = idx >> log2_tile_m;
                        const int k     = idx & tile_m_mask;
                        if (k < cur_tile) {
                            const size_t bp = bp_base + static_cast<size_t>(b_sub);
                            if (bp < bs_packed) {
                                const int m_idx = (num_wins == 1) ? (p_base + k) : d_bucket_m[p_base + k];
                                RealT sy_re, sy_im;
                                SYCL_FFT_SUFFIX(load_shifted_y)(
                                    d_y, d_shift, bp, total_M_stride, chunk_bs, m_idx, sy_re, sy_im);
                                const int sy_slot = k * sy_stride + b_sub;
                                l_sy_re[sy_slot] = sy_re;
                                l_sy_im[sy_slot] = sy_im;
                            }
                        }
                    }

                    sycl::group_barrier(grp);

                    // Phase 2: Congruence-partitioned, zero-atomic, zero-LDG, bank-conflict-free accumulation
                    if (my_bp < bs_packed) {
                        for (int k = 0; k < cur_tile; ++k) {
                            const int first = l_first[k];
                            const int j0 = (lane_t - first) & tj_mask;
                            const RealT sy_re = l_sy_re[k * sy_stride + lane_b];
                            const RealT sy_im = l_sy_im[k * sy_stride + lane_b];
                            const int w_base = k * W;

                            auto accum_bin = [&](int j_val) SYCL_FFT_LAMBDA_INLINE {
                                const int bin = ((first + j_val) & nfft_mask) - win_lo;
                                if (num_wins == 1 ||
                                    static_cast<unsigned>(bin) < static_cast<unsigned>(win_elems)) {
                                    const int g_idx = grid_base + bin;
                                    const ScalarT ws = l_w[w_base + j_val];
#if defined(SYCL_FFT_HALF)
                                    const RealT wv = SYCL_FFT_F16X2_SPLAT_H(ws);
#else
                                    const RealT wv = ws;
#endif
                                    l_re[g_idx] = SYCL_NUFFT_FMA(sy_re, wv, l_re[g_idx]);
                                    l_im[g_idx] = SYCL_NUFFT_FMA(sy_im, wv, l_im[g_idx]);
                                }
                            };

                            if constexpr (MultiPass) {
                                // t_j = prev_pow2(W) < W -> j0 is always < W; only lanes with j0 + t_j < W do pass 1
                                accum_bin(j0);
                                const int j1 = j0 + t_j;
                                if (j1 < W) {
                                    accum_bin(j1);
                                }
                            } else {
                                if (j0 < W) {
                                    accum_bin(j0);
                                }
                            }
                        }
                    }

                    sycl::group_barrier(grp);
                }

                // Coalesced writeback of local-memory window directly to batch-major d_fft_buf
                for (int b_sub = 0; b_sub < b_per_wg; ++b_sub) {
                    const size_t bp_out = bp_base + static_cast<size_t>(b_sub);
                    if (bp_out >= bs_packed) break;
                    const int g_off = b_sub * grid_stride;
#if defined(SYCL_FFT_HALF)
                    const size_t b0_out = 2 * bp_out;
                    const size_t b1_out = 2 * bp_out + 1;
                    const bool v0 = (b0_out < chunk_bs);
                    const bool v1 = (b1_out < chunk_bs);
                    ComplexT* dst0 = d_fft_buf + b0_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    ComplexT* dst1 = v1 ? (d_fft_buf + b1_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo)) : dst0;
                    for (int n = tid; n < win_elems; n += wg_size) {
                        sycl_fft::f16x2_store_pair(dst0, dst1, n, v0, v1, l_re[g_off + n], l_im[g_off + n]);
                    }
#elif defined(SYCL_FFT_DF)
                    ComplexT* dst = d_fft_buf + bp_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    for (int n = tid; n < win_elems; n += wg_size) {
                        dst[n] = ComplexT{l_re[g_off + n], l_im[g_off + n]};
                    }
#else
                    ComplexT* dst = d_fft_buf + bp_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    for (int n = tid; n < win_elems; n += wg_size) {
                        dst[n] = ComplexT(l_re[g_off + n], l_im[g_off + n]);
                    }
#endif
                }
            });
    });
}

// ============================================================================
// 2b. On-the-Fly Windowed SMEM Spreading (huge-M fallback when d_weights == nullptr)
//     - Bank-conflict-free l_sy[k * (b_per_wg + 1) + lane_b] and l_w[j * (tile_m + 1) + k]
// ============================================================================
template <int W, bool MultiPass>
inline void SYCL_FFT_SUFFIX(launch_spread_smem_win_otf_impl)(
    sycl::queue& q,
    const void* d_tbl_raw,
    const int* d_first,
    const SYCL_NUFFT_SCALAR_T* d_u,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    const int* d_bucket_off,
    const int* d_bucket_m,
    const SYCL_NUFFT_COMPLEX_T* d_y,
    SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int M,
    int nfft,
    size_t total_M_stride,
    size_t chunk_bs,
    const SpreadConfig& cfg) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    using RealT    = SYCL_NUFFT_REAL_T;

    constexpr int kLanes = SYCL_NUFFT_LANES;
    const int wg_size     = cfg.wg_size;
    const int t_j         = cfg.t_j;
    const int log2_tj     = int_log2_pow2(t_j);
    const int tj_mask     = t_j - 1;
    const int b_per_wg    = cfg.b_per_wg;
    const int sy_stride   = b_per_wg + 1;
    const int win_elems   = cfg.win_elems;
    const int num_wins    = cfg.num_wins;
    const int log2_nw     = int_log2_pow2(num_wins);
    const int nw_mask     = num_wins - 1;
    const int grid_stride = cfg.grid_stride;
    const int tile_m      = cfg.tile_m;
    const int w_stride    = tile_m + 1; // Odd -> 0 bank conflicts for SoA l_w[j * w_stride + k]
    const int log2_tile_m = int_log2_pow2(tile_m);
    const int tile_m_mask = tile_m - 1;
    const int nfft_mask   = nfft - 1;
    const int smem_grid_elems = b_per_wg * grid_stride;
    const int smem_sy_elems   = tile_m * sy_stride;

    const size_t bs_packed    = (chunk_bs + kLanes - 1) / kLanes;
    const size_t num_b_groups = (bs_packed + static_cast<size_t>(b_per_wg) - 1) / static_cast<size_t>(b_per_wg);
    const size_t total_wgs    = num_b_groups * static_cast<size_t>(num_wins);

#if defined(SYCL_FFT_HALF)
    using TableT = std::decay_t<decltype(get_pswf_table_f16<W>(PswfMode::PSWF21))>;
#elif defined(SYCL_FFT_DOUBLE)
    using TableT = std::decay_t<decltype(get_pswf_table_f64<W>(PswfMode::PSWF21))>;
#elif defined(SYCL_FFT_DF)
    using TableT = std::decay_t<decltype(get_pswf_table_df<W>(PswfMode::PSWF21))>;
#else
    using TableT = std::decay_t<decltype(get_pswf_table_f32<W>(PswfMode::PSWF21))>;
#endif
    const TableT* d_tbl = static_cast<const TableT*>(d_tbl_raw);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<RealT, 1>   l_re(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_im(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<int, 1>     l_first(sycl::range<1>(static_cast<size_t>(tile_m)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_re(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_im(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
        sycl::local_accessor<ScalarT, 1> l_w(sycl::range<1>(static_cast<size_t>(W * w_stride)), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(total_wgs * static_cast<size_t>(wg_size), static_cast<size_t>(wg_size)),
            [=](sycl::nd_item<1> item) {
                const auto grp = item.get_group();
                const int tid  = static_cast<int>(item.get_local_linear_id());
                const size_t wg_id   = item.get_group_linear_id();
                const size_t b_group = wg_id >> log2_nw;
                const int win        = static_cast<int>(wg_id & static_cast<size_t>(nw_mask));
                const int win_lo     = win * win_elems;

                const int lane_b     = tid >> log2_tj;
                const int lane_t     = tid & tj_mask;
                const size_t bp_base = b_group * static_cast<size_t>(b_per_wg);
                const size_t my_bp   = bp_base + static_cast<size_t>(lane_b);
                const int grid_base  = lane_b * grid_stride;

                for (int idx = tid; idx < smem_grid_elems; idx += wg_size) {
                    l_re[idx] = SYCL_FFT_ZERO;
                    l_im[idx] = SYCL_FFT_ZERO;
                }
                sycl::group_barrier(grp);

                const int p_start = (num_wins == 1) ? 0 : d_bucket_off[win];
                const int p_end   = (num_wins == 1) ? M : d_bucket_off[win + 1];

                for (int p_base = p_start; p_base < p_end; p_base += tile_m) {
                    const int cur_tile = sycl::min(tile_m, p_end - p_base);

                    if (tid < cur_tile) {
                        const int m_idx = (num_wins == 1) ? (p_base + tid) : d_bucket_m[p_base + tid];
                        l_first[tid] = d_first[m_idx];
                        const ScalarT u = d_u[m_idx];

                        ScalarT w_reg[W];
#if defined(SYCL_FFT_HALF)
                        eval_pswf_spread_weights_f16<W>(u, *d_tbl, w_reg);
#elif defined(SYCL_FFT_DOUBLE)
                        eval_pswf_spread_weights_f64<W>(u, *d_tbl, w_reg);
#elif defined(SYCL_FFT_DF)
                        eval_pswf_spread_weights_df<W>(u, *d_tbl, w_reg);
#else
                        eval_pswf_spread_weights_f32<W>(u, *d_tbl, w_reg);
#endif
                        #pragma unroll
                        for (int j = 0; j < W; ++j) {
                            l_w[j * w_stride + tid] = w_reg[j];
                        }
                    }

                    const int total_sy = b_per_wg * tile_m;
                    for (int idx = tid; idx < total_sy; idx += wg_size) {
                        const int b_sub = idx >> log2_tile_m;
                        const int k     = idx & tile_m_mask;
                        if (k < cur_tile) {
                            const size_t bp = bp_base + static_cast<size_t>(b_sub);
                            if (bp < bs_packed) {
                                const int m_idx = (num_wins == 1) ? (p_base + k) : d_bucket_m[p_base + k];
                                RealT sy_re, sy_im;
                                SYCL_FFT_SUFFIX(load_shifted_y)(
                                    d_y, d_shift, bp, total_M_stride, chunk_bs, m_idx, sy_re, sy_im);
                                const int sy_slot = k * sy_stride + b_sub;
                                l_sy_re[sy_slot] = sy_re;
                                l_sy_im[sy_slot] = sy_im;
                            }
                        }
                    }

                    sycl::group_barrier(grp);

                    if (my_bp < bs_packed) {
                        for (int k = 0; k < cur_tile; ++k) {
                            const int first = l_first[k];
                            const int j0 = (lane_t - first) & tj_mask;
                            const RealT sy_re = l_sy_re[k * sy_stride + lane_b];
                            const RealT sy_im = l_sy_im[k * sy_stride + lane_b];

                            auto accum_bin = [&](int j_val) SYCL_FFT_LAMBDA_INLINE {
                                const int bin = ((first + j_val) & nfft_mask) - win_lo;
                                if (num_wins == 1 ||
                                    static_cast<unsigned>(bin) < static_cast<unsigned>(win_elems)) {
                                    const int g_idx = grid_base + bin;
                                    const ScalarT ws = l_w[j_val * w_stride + k];
#if defined(SYCL_FFT_HALF)
                                    const RealT wv = SYCL_FFT_F16X2_SPLAT_H(ws);
#else
                                    const RealT wv = ws;
#endif
                                    l_re[g_idx] = SYCL_NUFFT_FMA(sy_re, wv, l_re[g_idx]);
                                    l_im[g_idx] = SYCL_NUFFT_FMA(sy_im, wv, l_im[g_idx]);
                                }
                            };

                            if constexpr (MultiPass) {
                                accum_bin(j0);
                                const int j1 = j0 + t_j;
                                if (j1 < W) {
                                    accum_bin(j1);
                                }
                            } else {
                                if (j0 < W) {
                                    accum_bin(j0);
                                }
                            }
                        }
                    }

                    sycl::group_barrier(grp);
                }

                for (int b_sub = 0; b_sub < b_per_wg; ++b_sub) {
                    const size_t bp_out = bp_base + static_cast<size_t>(b_sub);
                    if (bp_out >= bs_packed) break;
                    const int g_off = b_sub * grid_stride;
#if defined(SYCL_FFT_HALF)
                    const size_t b0_out = 2 * bp_out;
                    const size_t b1_out = 2 * bp_out + 1;
                    const bool v0 = (b0_out < chunk_bs);
                    const bool v1 = (b1_out < chunk_bs);
                    ComplexT* dst0 = d_fft_buf + b0_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    ComplexT* dst1 = v1 ? (d_fft_buf + b1_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo)) : dst0;
                    for (int n = tid; n < win_elems; n += wg_size) {
                        sycl_fft::f16x2_store_pair(dst0, dst1, n, v0, v1, l_re[g_off + n], l_im[g_off + n]);
                    }
#elif defined(SYCL_FFT_DF)
                    ComplexT* dst = d_fft_buf + bp_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    for (int n = tid; n < win_elems; n += wg_size) {
                        dst[n] = ComplexT{l_re[g_off + n], l_im[g_off + n]};
                    }
#else
                    ComplexT* dst = d_fft_buf + bp_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    for (int n = tid; n < win_elems; n += wg_size) {
                        dst[n] = ComplexT(l_re[g_off + n], l_im[g_off + n]);
                    }
#endif
                }
            });
    });
}

template <int W>
inline void SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)(
    sycl::queue& q,
    const void* d_tbl_raw,
    const SYCL_NUFFT_SCALAR_T* d_weights,
    const int* d_first,
    const SYCL_NUFFT_SCALAR_T* d_u,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    const int* d_bucket_off,
    const int* d_bucket_m,
    const SYCL_NUFFT_COMPLEX_T* d_y,
    SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int M,
    int nfft,
    size_t total_M_stride,
    size_t chunk_bs,
    const SpreadConfig& cfg) {
    constexpr bool kCanMultiPass = (W == 5 || W == 9 || W == 17 || W == 18);
    if (d_weights) {
        if constexpr (kCanMultiPass) {
            if (cfg.t_j < W) {
                SYCL_FFT_SUFFIX(launch_spread_smem_win_precomp_impl)<W, true>(
                    q, d_weights, d_first, d_shift, d_bucket_off, d_bucket_m,
                    d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, cfg);
                return;
            }
        }
        SYCL_FFT_SUFFIX(launch_spread_smem_win_precomp_impl)<W, false>(
            q, d_weights, d_first, d_shift, d_bucket_off, d_bucket_m,
            d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, cfg);
    } else {
        if constexpr (kCanMultiPass) {
            if (cfg.t_j < W) {
                SYCL_FFT_SUFFIX(launch_spread_smem_win_otf_impl)<W, true>(
                    q, d_tbl_raw, d_first, d_u, d_shift, d_bucket_off, d_bucket_m,
                    d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, cfg);
                return;
            }
        }
        SYCL_FFT_SUFFIX(launch_spread_smem_win_otf_impl)<W, false>(
            q, d_tbl_raw, d_first, d_u, d_shift, d_bucket_off, d_bucket_m,
            d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, cfg);
    }
}

// ============================================================================
// 3. Collaborative Zero-Atomic Batch-Inner Spreading (forced fallback / GPU alt)
// ============================================================================
template <int W>
inline void SYCL_FFT_SUFFIX(launch_spread_collab_impl)(
    sycl::queue& q,
    const void* d_tbl_raw,
    const SYCL_NUFFT_SCALAR_T* d_weights,
    const int* d_first,
    const SYCL_NUFFT_SCALAR_T* d_u,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    const SYCL_NUFFT_COMPLEX_T* d_y,
    SYCL_NUFFT_REAL_T* d_grid_T_re,
    SYCL_NUFFT_REAL_T* d_grid_T_im,
    int M,
    int nfft,
    size_t total_M_stride,
    size_t chunk_bs,
    size_t packed_stride,
    const SpreadConfig& cfg) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    using RealT    = SYCL_NUFFT_REAL_T;

    constexpr int kLanes = SYCL_NUFFT_LANES;
    const int wg_size = cfg.wg_size;
    const int tile_m  = cfg.tile_m;
    const size_t bs_packed = (chunk_bs + kLanes - 1) / kLanes;
    const size_t num_b_groups = (bs_packed + static_cast<size_t>(wg_size) - 1) / static_cast<size_t>(wg_size);
    const int nfft_mask = nfft - 1;

#if defined(SYCL_FFT_HALF)
    using TableT = std::decay_t<decltype(get_pswf_table_f16<W>(PswfMode::PSWF21))>;
#elif defined(SYCL_FFT_DOUBLE)
    using TableT = std::decay_t<decltype(get_pswf_table_f64<W>(PswfMode::PSWF21))>;
#elif defined(SYCL_FFT_DF)
    using TableT = std::decay_t<decltype(get_pswf_table_df<W>(PswfMode::PSWF21))>;
#else
    using TableT = std::decay_t<decltype(get_pswf_table_f32<W>(PswfMode::PSWF21))>;
#endif
    const TableT* d_tbl = static_cast<const TableT*>(d_tbl_raw);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1>      l_first(sycl::range<1>(static_cast<size_t>(tile_m)), cgh);
        sycl::local_accessor<ComplexT, 1> l_shift(sycl::range<1>(static_cast<size_t>(tile_m)), cgh);
        sycl::local_accessor<ScalarT, 1>  l_w(sycl::range<1>(static_cast<size_t>(tile_m * W)), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_b_groups * static_cast<size_t>(wg_size), static_cast<size_t>(wg_size)),
            [=](sycl::nd_item<1> item) {
                const auto grp = item.get_group();
                const int tid = static_cast<int>(item.get_local_linear_id());
                const size_t b_group = item.get_group_linear_id();
                const size_t my_bp   = b_group * static_cast<size_t>(wg_size) + static_cast<size_t>(tid);

                for (int m_base = 0; m_base < M; m_base += tile_m) {
                    const int cur_tile = sycl::min(tile_m, M - m_base);

                    for (int k = tid; k < cur_tile; k += wg_size) {
                        const int m_idx = m_base + k;
                        l_first[k] = d_first[m_idx];
                        l_shift[k] = d_shift[m_idx];
                        if (d_weights) {
                            const ScalarT* src = d_weights + static_cast<size_t>(m_idx) * static_cast<size_t>(W);
                            #pragma unroll
                            for (int j = 0; j < W; ++j) {
                                l_w[k * W + j] = src[j];
                            }
                        } else {
                            const ScalarT u = d_u[m_idx];
                            ScalarT w_reg[W];
#if defined(SYCL_FFT_HALF)
                            eval_pswf_spread_weights_f16<W>(u, *d_tbl, w_reg);
#elif defined(SYCL_FFT_DOUBLE)
                            eval_pswf_spread_weights_f64<W>(u, *d_tbl, w_reg);
#elif defined(SYCL_FFT_DF)
                            eval_pswf_spread_weights_df<W>(u, *d_tbl, w_reg);
#else
                            eval_pswf_spread_weights_f32<W>(u, *d_tbl, w_reg);
#endif
                            #pragma unroll
                            for (int j = 0; j < W; ++j) {
                                l_w[k * W + j] = w_reg[j];
                            }
                        }
                    }

                    sycl::group_barrier(grp);

                    if (my_bp < bs_packed) {
#if defined(SYCL_FFT_HALF)
                        const size_t b0 = 2 * my_bp;
                        const size_t b1 = 2 * my_bp + 1;
                        const bool valid0 = (b0 < chunk_bs);
                        const bool valid1 = (b1 < chunk_bs);
                        const ComplexT* y_ptr0 = d_y + b0 * total_M_stride;
                        const ComplexT* y_ptr1 = valid1 ? (d_y + b1 * total_M_stride) : y_ptr0;
#else
                        const ComplexT* y_ptr = d_y + my_bp * total_M_stride;
#endif
                        #pragma nounroll
                        for (int k = 0; k < cur_tile; ++k) {
                            const int m_idx = m_base + k;
                            const ComplexT sh = l_shift[k];
                            RealT sy_re, sy_im;
#if defined(SYCL_FFT_HALF)
                            RealT y_re, y_im;
                            sycl_fft::f16x2_load_pair(
                                y_ptr0, y_ptr1, m_idx, valid0, valid1, y_re, y_im);
                            const RealT sh_v = SYCL_FFT_BIT_CAST(RealT, sh);
                            const ScalarT sr_h = sycl_fft::f16x2_lo(sh_v);
                            const ScalarT si_h = sycl_fft::f16x2_hi(sh_v);
                            const RealT sr_v = SYCL_FFT_F16X2_SPLAT_H(sr_h);
                            const RealT si_v = SYCL_FFT_F16X2_SPLAT_H(si_h);
                            const RealT neg_si_v = SYCL_FFT_BIT_CAST(
                                RealT, SYCL_FFT_BIT_CAST(uint32_t, si_v) ^ 0x80008000u);
                            sy_re = SYCL_FFT_F16X2_FMA(
                                y_re, sr_v, SYCL_FFT_F16X2_FMA(y_im, neg_si_v, sycl_fft::kZero_f16x2));
                            sy_im = SYCL_FFT_F16X2_FMA(
                                y_re, si_v, SYCL_FFT_F16X2_FMA(y_im, sr_v, sycl_fft::kZero_f16x2));
#elif defined(SYCL_FFT_DF)
                            const ComplexT y = y_ptr[m_idx];
                            sycl_fft::df_cmul_fused(y.re, y.im, sh.re, sh.im, sy_re, sy_im);
#else
                            const ComplexT y = y_ptr[m_idx];
                            SYCL_FFT_CMUL(y.x(), y.y(), sh.x(), sh.y(), sy_re, sy_im);
#endif
                            const int first = l_first[k];

#if defined(SYCL_FFT_DF)
                            #pragma nounroll
#else
                            #pragma unroll
#endif
                            for (int j = 0; j < W; ++j) {
                                const int idx = (first + j) & nfft_mask;
                                const size_t g_off =
                                    static_cast<size_t>(idx) * packed_stride + my_bp;
                                const ScalarT w_s = l_w[k * W + j];
#if defined(SYCL_FFT_HALF)
                                const RealT w_v = SYCL_FFT_F16X2_SPLAT_H(w_s);
#else
                                const RealT w_v = w_s;
#endif
                                d_grid_T_re[g_off] =
                                    SYCL_NUFFT_FMA(sy_re, w_v, d_grid_T_re[g_off]);
                                d_grid_T_im[g_off] =
                                    SYCL_NUFFT_FMA(sy_im, w_v, d_grid_T_im[g_off]);
                            }
                        }
                    }

                    sycl::group_barrier(grp);
                }
            });
    });
}

// ============================================================================
// 4. Barrier-Free Direct CPU Spreading (1 thread per transform / FP16 pair)
//    Zero local_accessor, zero group_barrier, writes directly to d_fft_buf!
// ============================================================================
template <int W>
inline void SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)(
    sycl::queue& q,
    const void* d_tbl_raw,
    const SYCL_NUFFT_SCALAR_T* d_weights,
    const int* d_first,
    const SYCL_NUFFT_SCALAR_T* d_u,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    const SYCL_NUFFT_COMPLEX_T* d_y,
    SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int M,
    int nfft,
    size_t total_M_stride,
    size_t chunk_bs) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    using RealT    = SYCL_NUFFT_REAL_T;

    constexpr int kLanes = SYCL_NUFFT_LANES;
    const size_t bs_packed = (chunk_bs + kLanes - 1) / kLanes;
    const int nfft_mask = nfft - 1;

#if defined(SYCL_FFT_HALF)
    using TableT = std::decay_t<decltype(get_pswf_table_f16<W>(PswfMode::PSWF21))>;
#elif defined(SYCL_FFT_DOUBLE)
    using TableT = std::decay_t<decltype(get_pswf_table_f64<W>(PswfMode::PSWF21))>;
#elif defined(SYCL_FFT_DF)
    using TableT = std::decay_t<decltype(get_pswf_table_df<W>(PswfMode::PSWF21))>;
#else
    using TableT = std::decay_t<decltype(get_pswf_table_f32<W>(PswfMode::PSWF21))>;
#endif
    const TableT* d_tbl = static_cast<const TableT*>(d_tbl_raw);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>(bs_packed),
            [=](sycl::id<1> idx_id) {
                const size_t bp = idx_id[0];
#if defined(SYCL_FFT_HALF)
                const size_t b0 = 2 * bp;
                const size_t b1 = 2 * bp + 1;
                const bool v0 = (b0 < chunk_bs);
                const bool v1 = (b1 < chunk_bs);
                ComplexT* dst0 = d_fft_buf + b0 * static_cast<size_t>(nfft);
                ComplexT* dst1 = v1 ? (d_fft_buf + b1 * static_cast<size_t>(nfft)) : dst0;
                for (int n = 0; n < nfft; ++n) {
                    sycl_fft::f16x2_store_pair(dst0, dst1, n, v0, v1, kZero_f16x2, kZero_f16x2);
                }
#else
                ComplexT* dst = d_fft_buf + bp * static_cast<size_t>(nfft);
                for (int n = 0; n < nfft; ++n) {
#if defined(SYCL_FFT_DF)
                    dst[n] = ComplexT{SYCL_FFT_ZERO, SYCL_FFT_ZERO};
#else
                    dst[n] = ComplexT(SYCL_FFT_ZERO, SYCL_FFT_ZERO);
#endif
                }
#endif
                for (int m = 0; m < M; ++m) {
                    RealT sy_re, sy_im;
                    SYCL_FFT_SUFFIX(load_shifted_y)(
                        d_y, d_shift, bp, total_M_stride, chunk_bs, m, sy_re, sy_im);
                    const int first = d_first[m];

                    ScalarT w_local[W];
                    const ScalarT* w_ptr = d_weights
                        ? (d_weights + static_cast<size_t>(m) * static_cast<size_t>(W))
                        : w_local;
                    if (!d_weights) {
                        const ScalarT u = d_u[m];
#if defined(SYCL_FFT_HALF)
                        eval_pswf_spread_weights_f16<W>(u, *d_tbl, w_local);
#elif defined(SYCL_FFT_DOUBLE)
                        eval_pswf_spread_weights_f64<W>(u, *d_tbl, w_local);
#elif defined(SYCL_FFT_DF)
                        eval_pswf_spread_weights_df<W>(u, *d_tbl, w_local);
#else
                        eval_pswf_spread_weights_f32<W>(u, *d_tbl, w_local);
#endif
                    }

                    #pragma unroll
                    for (int j = 0; j < W; ++j) {
                        const int bin = (first + j) & nfft_mask;
#if defined(SYCL_FFT_HALF)
                        const RealT wv = SYCL_FFT_F16X2_SPLAT_H(w_ptr[j]);
                        RealT cur_re, cur_im;
                        sycl_fft::f16x2_load_pair(dst0, dst1, bin, v0, v1, cur_re, cur_im);
                        cur_re = SYCL_NUFFT_FMA(sy_re, wv, cur_re);
                        cur_im = SYCL_NUFFT_FMA(sy_im, wv, cur_im);
                        sycl_fft::f16x2_store_pair(dst0, dst1, bin, v0, v1, cur_re, cur_im);
#elif defined(SYCL_FFT_DF)
                        const RealT wv = w_ptr[j];
                        dst[bin].re = SYCL_NUFFT_FMA(sy_re, wv, dst[bin].re);
                        dst[bin].im = SYCL_NUFFT_FMA(sy_im, wv, dst[bin].im);
#else
                        const RealT wv = w_ptr[j];
                        dst[bin] = ComplexT(
                            SYCL_NUFFT_FMA(sy_re, wv, dst[bin].x()),
                            SYCL_NUFFT_FMA(sy_im, wv, dst[bin].y()));
#endif
                    }
                }
            });
    });
}

// ============================================================================
// Precision-Specific PSWF Table Upload & External Entrypoints
// ============================================================================
template <typename TableT>
static uint8_t* upload_table_impl(sycl::queue& q, const TableT& tbl) {
    uint8_t* ptr = sycl::malloc_device<uint8_t>(sizeof(TableT), q);
    if (!ptr) return nullptr;
    q.memcpy(ptr, &tbl, sizeof(TableT)).wait();
    return ptr;
}

uint8_t* SYCL_FFT_SUFFIX(upload_pswf_table)(sycl::queue& q, int w, PswfMode mode) {
#if defined(SYCL_FFT_HALF)
    // FP16 retains subnormal-safe 2^e spreading kernel scaling
    if (w == 4) return upload_table_impl(q, get_pswf_table_f16<4>(mode));
    return upload_table_impl(q, get_pswf_table_f16<5>(mode));
#elif defined(SYCL_FFT_DOUBLE)
    // FP64 rescales kernel to exact geometric-mean-1: sqrt(phi_hat(0) * phi_hat(edge)) == 1
    if (w == 16) return upload_table_impl(q, make_geo_scaled_pswf_table(get_pswf_table_f64<16>(mode), mode));
    if (w == 17) return upload_table_impl(q, make_geo_scaled_pswf_table(get_pswf_table_f64<17>(mode), mode));
    return upload_table_impl(q, make_geo_scaled_pswf_table(get_pswf_table_f64<18>(mode), mode));
#elif defined(SYCL_FFT_DF)
    // DF rescales kernel to exact geometric-mean-1: sqrt(phi_hat(0) * phi_hat(edge)) == 1
    if (w == 14) return upload_table_impl(q, make_geo_scaled_pswf_table(get_pswf_table_df<14>(mode), mode));
    if (w == 15) return upload_table_impl(q, make_geo_scaled_pswf_table(get_pswf_table_df<15>(mode), mode));
    return upload_table_impl(q, make_geo_scaled_pswf_table(get_pswf_table_df<16>(mode), mode));
#else
    // FP32 rescales kernel to exact geometric-mean-1: sqrt(phi_hat(0) * phi_hat(edge)) == 1
    if (w == 8) return upload_table_impl(q, make_geo_scaled_pswf_table(get_pswf_table_f32<8>(mode), mode));
    return upload_table_impl(q, make_geo_scaled_pswf_table(get_pswf_table_f32<9>(mode), mode));
#endif
}

void SYCL_FFT_SUFFIX(launch_precompute_weights)(
    sycl::queue& q, int w, const void* d_tbl,
    const SYCL_NUFFT_SCALAR_T* d_u, SYCL_NUFFT_SCALAR_T* d_weights,
    int M, size_t wg_size) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<4>(q, d_tbl, d_u, d_weights, M, wg_size);
    else SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<5>(q, d_tbl, d_u, d_weights, M, wg_size);
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<16>(q, d_tbl, d_u, d_weights, M, wg_size);
    else if (w == 17) SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<17>(q, d_tbl, d_u, d_weights, M, wg_size);
    else SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<18>(q, d_tbl, d_u, d_weights, M, wg_size);
#elif defined(SYCL_FFT_DF)
    if (w == 14) SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<14>(q, d_tbl, d_u, d_weights, M, wg_size);
    else if (w == 15) SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<15>(q, d_tbl, d_u, d_weights, M, wg_size);
    else SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<16>(q, d_tbl, d_u, d_weights, M, wg_size);
#else
    if (w == 8) SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<8>(q, d_tbl, d_u, d_weights, M, wg_size);
    else SYCL_FFT_SUFFIX(launch_precompute_weights_impl)<9>(q, d_tbl, d_u, d_weights, M, wg_size);
#endif
}

void SYCL_FFT_SUFFIX(launch_spread_smem_win)(
    sycl::queue& q, int w, const void* d_tbl, const SYCL_NUFFT_SCALAR_T* d_weights,
    const int* d_first, const SYCL_NUFFT_SCALAR_T* d_u, const SYCL_NUFFT_COMPLEX_T* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const SYCL_NUFFT_COMPLEX_T* d_y, SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg) {
    (void)mode;
#if defined(SYCL_FFT_HALF)
    if (w == 4) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<4>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<5>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    }
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<16>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    } else if (w == 17) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<17>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<18>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    }
#elif defined(SYCL_FFT_DF)
    if (w == 14) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<14>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    } else if (w == 15) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<15>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<16>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    }
#else
    if (w == 8) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<8>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_dispatch_w)<9>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, cfg);
    }
#endif
}

void SYCL_FFT_SUFFIX(launch_spread_collab)(
    sycl::queue& q, int w, const void* d_tbl, const SYCL_NUFFT_SCALAR_T* d_weights,
    const int* d_first, const SYCL_NUFFT_SCALAR_T* d_u, const SYCL_NUFFT_COMPLEX_T* d_shift,
    const SYCL_NUFFT_COMPLEX_T* d_y,
    SYCL_NUFFT_REAL_T* d_grid_T_re, SYCL_NUFFT_REAL_T* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg) {
    (void)mode;
#if defined(SYCL_FFT_HALF)
    if (w == 4) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<4>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<5>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    }
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<16>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    } else if (w == 17) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<17>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<18>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    }
#elif defined(SYCL_FFT_DF)
    if (w == 14) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<14>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    } else if (w == 15) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<15>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<16>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    }
#else
    if (w == 8) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<8>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<9>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, cfg);
    }
#endif
}

void SYCL_FFT_SUFFIX(launch_spread_cpu_direct)(
    sycl::queue& q, int w, const void* d_tbl, const SYCL_NUFFT_SCALAR_T* d_weights,
    const int* d_first, const SYCL_NUFFT_SCALAR_T* d_u, const SYCL_NUFFT_COMPLEX_T* d_shift,
    const SYCL_NUFFT_COMPLEX_T* d_y, SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<4>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<5>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    }
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<16>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else if (w == 17) {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<17>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<18>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    }
#elif defined(SYCL_FFT_DF)
    if (w == 14) {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<14>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else if (w == 15) {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<15>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<16>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    }
#else
    if (w == 8) {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<8>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_cpu_direct_impl)<9>(
            q, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    }
#endif
}

#undef SYCL_NUFFT_COMPLEX_T
#undef SYCL_NUFFT_REAL_T
#undef SYCL_NUFFT_SCALAR_T
#undef SYCL_NUFFT_LANES
#undef SYCL_NUFFT_FMA

} // namespace detail
} // namespace sycl_nufft
