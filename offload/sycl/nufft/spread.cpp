// Decoupled SYCL Type-1 NuFFT Spreading Kernel Module.
// Compiled once per precision with -D flags so each precision's spreading kernels
// live in their own lean HCF module:
//   -DSYCL_FFT_HALF | -DSYCL_FFT_FLOAT | -DSYCL_FFT_DOUBLE | -DSYCL_FFT_DF
// Defaults to SYCL_FFT_DF (Double-Float) when no half/float/double is selected.
// ZERO hardcoded GPU/Blackwell performance-scaling constants: all launch geometry
// (wg_size, t_j, b_per_wg, win_elems, num_wins, grid_stride, tile_m) is supplied at
// runtime via SpreadConfig derived from DeviceFftConfig + BankModel + AutotuneCache.

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

// Hardware-generalized multi-warp windowed congruence-partitioned SMEM spreading:
//   - t_j = next_pow2(W) >= W always -> each lane updates at most 1 grid bin per point
//     (eliminates warp-divergent second pass / width cliffs at FP32 w=9 and FP64 w=17,18).
//   - Multi-warp work-group (wg_size = b_per_wg * t_j) boosts SM/CU occupancy and amortizes
//     on-the-fly hardcoded piecewise polynomial evaluations across b_per_wg packed transforms.
//   - grid_stride = win_elems + bank_pad (from BankModel) eliminates inter-transform SMEM bank conflicts.
//   - Works for arbitrary nfft and local_mem_size via O(M)-bucketed grid windows (num_wins >= 1).
template <int W>
inline void SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)(
    sycl::queue& q,
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
    PswfMode mode,
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

    const size_t bs_packed    = (chunk_bs + kLanes - 1) / kLanes;
    const size_t num_b_groups = (bs_packed + static_cast<size_t>(b_per_wg) - 1) / static_cast<size_t>(b_per_wg);
    const size_t total_wgs    = num_b_groups * static_cast<size_t>(num_wins);

#if defined(SYCL_FFT_HALF)
    const auto tbl = get_pswf_table_f16<W>(mode);
#elif defined(SYCL_FFT_DOUBLE)
    const auto tbl = get_pswf_table_f64<W>(mode);
#elif defined(SYCL_FFT_DF)
    const auto tbl = get_pswf_table_df<W>(mode);
#else
    const auto tbl = get_pswf_table_f32<W>(mode);
#endif

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<RealT, 1>   l_re(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_im(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<int, 1>     l_first(sycl::range<1>(static_cast<size_t>(tile_m)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_re(sycl::range<1>(static_cast<size_t>(b_per_wg * tile_m)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_im(sycl::range<1>(static_cast<size_t>(b_per_wg * tile_m)), cgh);
        sycl::local_accessor<ScalarT, 1> l_w(sycl::range<1>(static_cast<size_t>(tile_m * W)), cgh);

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

                // Zero local-memory subgrid cooperatively across all work-group lanes
                for (int idx = tid; idx < smem_grid_elems; idx += wg_size) {
                    l_re[idx] = SYCL_FFT_ZERO;
                    l_im[idx] = SYCL_FFT_ZERO;
                }
                sycl::group_barrier(grp);

                const int p_start = (num_wins == 1) ? 0 : d_bucket_off[win];
                const int p_end   = (num_wins == 1) ? M : d_bucket_off[win + 1];

                for (int p_base = p_start; p_base < p_end; p_base += tile_m) {
                    const int cur_tile = sycl::min(tile_m, p_end - p_base);

                    // Phase 1a: Evaluate piecewise minimax PSWF weights once per sample point in tile
                    if (tid < cur_tile) {
                        const int m_idx = (num_wins == 1) ? (p_base + tid) : d_bucket_m[p_base + tid];
                        l_first[tid] = d_first[m_idx];
                        const ScalarT u = d_u[m_idx];

                        ScalarT w_reg[W];
#if defined(SYCL_FFT_HALF)
                        eval_pswf_spread_weights_f16<W>(u, tbl, w_reg);
#elif defined(SYCL_FFT_DOUBLE)
                        eval_pswf_spread_weights_f64<W>(u, tbl, w_reg);
#elif defined(SYCL_FFT_DF)
                        eval_pswf_spread_weights_df<W>(u, tbl, w_reg);
#else
                        eval_pswf_spread_weights_f32<W>(u, tbl, w_reg);
#endif
                        #pragma unroll
                        for (int j = 0; j < W; ++j) {
                            l_w[tid * W + j] = w_reg[j];
                        }
                    }

                    // Phase 1b: All work-group lanes cooperatively load y and apply phase shift
                    const int total_sy = b_per_wg * tile_m;
                    for (int idx = tid; idx < total_sy; idx += wg_size) {
                        const int b_sub = idx >> log2_tile_m;
                        const int k     = idx & tile_m_mask;
                        if (k < cur_tile) {
                            const size_t bp = bp_base + static_cast<size_t>(b_sub);
                            if (bp < bs_packed) {
                                const int m_idx = (num_wins == 1) ? (p_base + k) : d_bucket_m[p_base + k];
                                const ComplexT sh = d_shift[m_idx];
                                RealT sy_re, sy_im;
#if defined(SYCL_FFT_HALF)
                                const RealT sh_v = SYCL_FFT_BIT_CAST(RealT, sh);
                                const ScalarT sr_h = sh_v[0];
                                const ScalarT si_h = sh_v[1];
                                const RealT sr_v = SYCL_FFT_F16X2_SPLAT_H(sr_h);
                                const RealT si_v = SYCL_FFT_F16X2_SPLAT_H(si_h);
                                const RealT neg_si_v = SYCL_FFT_BIT_CAST(
                                    RealT, SYCL_FFT_BIT_CAST(uint32_t, si_v) ^ 0x80008000u);
                                const size_t b0 = 2 * bp;
                                const size_t b1 = 2 * bp + 1;
                                RealT y_re, y_im;
                                sycl_fft::f16x2_load_pair(
                                    d_y + b0 * total_M_stride,
                                    d_y + b1 * total_M_stride,
                                    m_idx,
                                    b0 < chunk_bs,
                                    b1 < chunk_bs,
                                    y_re,
                                    y_im);
                                sy_re = SYCL_FFT_F16X2_FMA(
                                    y_re, sr_v, SYCL_FFT_F16X2_FMA(y_im, neg_si_v, sycl_fft::kZero_f16x2));
                                sy_im = SYCL_FFT_F16X2_FMA(
                                    y_re, si_v, SYCL_FFT_F16X2_FMA(y_im, sr_v, sycl_fft::kZero_f16x2));
#elif defined(SYCL_FFT_DF)
                                const ComplexT y = d_y[bp * total_M_stride + static_cast<size_t>(m_idx)];
                                sycl_fft::df_cmul_fused(y.re, y.im, sh.re, sh.im, sy_re, sy_im);
#else
                                const ComplexT y = d_y[bp * total_M_stride + static_cast<size_t>(m_idx)];
                                SYCL_FFT_CMUL(y.x(), y.y(), sh.x(), sh.y(), sy_re, sy_im);
#endif
                                l_sy_re[idx] = sy_re;
                                l_sy_im[idx] = sy_im;
                            }
                        }
                    }

                    sycl::group_barrier(grp);

                    // Phase 2: Congruence-partitioned zero-atomic, bank-conflict-free accumulation
                    // Because t_j >= W unconditionally, each lane hits at most 1 bin per point.
                    if (my_bp < bs_packed) {
                        const int sy_base = lane_b * tile_m;
                        for (int k = 0; k < cur_tile; ++k) {
                            const int first = l_first[k];
                            const int j0 = (lane_t - first) & tj_mask;
                            if (j0 < W) {
                                const int bin0 = ((first + j0) & nfft_mask) - win_lo;
                                if (num_wins == 1 ||
                                    static_cast<unsigned>(bin0) < static_cast<unsigned>(win_elems)) {
                                    const int idx0 = grid_base + bin0;
                                    const RealT sy_re = l_sy_re[sy_base + k];
                                    const RealT sy_im = l_sy_im[sy_base + k];
                                    const ScalarT w0_s = l_w[k * W + j0];
#if defined(SYCL_FFT_HALF)
                                    const RealT w0_v = SYCL_FFT_F16X2_SPLAT_H(w0_s);
#else
                                    const RealT w0_v = w0_s;
#endif
                                    l_re[idx0] = SYCL_NUFFT_FMA(sy_re, w0_v, l_re[idx0]);
                                    l_im[idx0] = SYCL_NUFFT_FMA(sy_im, w0_v, l_im[idx0]);
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
                    ComplexT* dst0 = d_fft_buf + b0_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    ComplexT* dst1 = d_fft_buf + b1_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    const bool v0 = (b0_out < chunk_bs);
                    const bool v1 = (b1_out < chunk_bs);
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

// Hardware-generalized collaborative zero-atomic batch-inner spreading kernel:
//   - Used automatically on CPUs / non-native local-mem devices and as an autotune contender.
//   - Runtime wg_size and tile_m derived from DeviceFftConfig (zero hardcoded constants).
template <int W>
inline void SYCL_FFT_SUFFIX(launch_spread_collab_impl)(
    sycl::queue& q,
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
    PswfMode mode,
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
    const auto tbl = get_pswf_table_f16<W>(mode);
#elif defined(SYCL_FFT_DOUBLE)
    const auto tbl = get_pswf_table_f64<W>(mode);
#elif defined(SYCL_FFT_DF)
    const auto tbl = get_pswf_table_df<W>(mode);
#else
    const auto tbl = get_pswf_table_f32<W>(mode);
#endif

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
                        const ScalarT u = d_u[m_idx];

                        ScalarT w_reg[W];
#if defined(SYCL_FFT_HALF)
                        eval_pswf_spread_weights_f16<W>(u, tbl, w_reg);
#elif defined(SYCL_FFT_DOUBLE)
                        eval_pswf_spread_weights_f64<W>(u, tbl, w_reg);
#elif defined(SYCL_FFT_DF)
                        eval_pswf_spread_weights_df<W>(u, tbl, w_reg);
#else
                        eval_pswf_spread_weights_f32<W>(u, tbl, w_reg);
#endif
                        #pragma unroll
                        for (int j = 0; j < W; ++j) {
                            l_w[k * W + j] = w_reg[j];
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
                        const ComplexT* y_ptr1 = d_y + b1 * total_M_stride;
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
                            const ScalarT sr_h = sh_v[0];
                            const ScalarT si_h = sh_v[1];
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

// External-linkage precision dispatch entrypoints called by sycl_nufft1_plan.cpp
void SYCL_FFT_SUFFIX(launch_spread_smem_win)(
    sycl::queue& q, int w,
    const int* d_first, const SYCL_NUFFT_SCALAR_T* d_u, const SYCL_NUFFT_COMPLEX_T* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const SYCL_NUFFT_COMPLEX_T* d_y, SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<4>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<5>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    }
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<16>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else if (w == 17) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<17>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<18>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    }
#elif defined(SYCL_FFT_DF)
    if (w == 14) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<14>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else if (w == 15) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<15>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<16>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    }
#else
    if (w == 8) {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<8>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_smem_win_impl)<9>(
            q, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf,
            M, nfft, total_M_stride, chunk_bs, mode, cfg);
    }
#endif
}

void SYCL_FFT_SUFFIX(launch_spread_collab)(
    sycl::queue& q, int w,
    const int* d_first, const SYCL_NUFFT_SCALAR_T* d_u, const SYCL_NUFFT_COMPLEX_T* d_shift,
    const SYCL_NUFFT_COMPLEX_T* d_y,
    SYCL_NUFFT_REAL_T* d_grid_T_re, SYCL_NUFFT_REAL_T* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<4>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<5>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    }
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<16>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else if (w == 17) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<17>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<18>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    }
#elif defined(SYCL_FFT_DF)
    if (w == 14) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<14>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else if (w == 15) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<15>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<16>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    }
#else
    if (w == 8) {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<8>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else {
        SYCL_FFT_SUFFIX(launch_spread_collab_impl)<9>(
            q, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im,
            M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
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
