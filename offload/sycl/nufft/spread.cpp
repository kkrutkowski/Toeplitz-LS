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

#if defined(SYCL_FFT_HALF) || (!defined(SYCL_FFT_DOUBLE) && !defined(SYCL_FFT_DF))
#define SYCL_NUFFT_PACK_SMEM 1
struct alignas(8) SYCL_FFT_SUFFIX(SpreadPairT) {
    SYCL_NUFFT_REAL_T re;
    SYCL_NUFFT_REAL_T im;
};
#else
#define SYCL_NUFFT_PACK_SMEM 0
#endif

// Helper: load d_y[bp, m_idx] and multiply by d_shift[shift_idx]
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(load_shifted_y)(
    const SYCL_NUFFT_COMPLEX_T* d_y,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    size_t bp,
    size_t total_M_stride,
    size_t chunk_bs,
    int m_idx,
    int shift_idx,
    SYCL_NUFFT_REAL_T& sy_re,
    SYCL_NUFFT_REAL_T& sy_im) {
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    const ComplexT sh = d_shift[shift_idx];
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
// 1b. Pack Geometry & Weights into Contiguous Bucket Order (once per set_points)
//     Eliminates d_bucket_m indirection and idx/W division in hot spreading loops.
// ============================================================================
template <int W>
inline void SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)(
    sycl::queue& q,
    int num_wins,
    int max_entries,
    const int* d_bucket_off,
    const int* d_bucket_m,
    const int* d_first,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    const SYCL_NUFFT_SCALAR_T* d_weights,
    int* d_b_first,
    SYCL_NUFFT_COMPLEX_T* d_b_shift,
    SYCL_NUFFT_SCALAR_T* d_b_weights,
    size_t wg_size) {
    if (num_wins <= 1 || max_entries <= 0) return;
    const size_t wg = std::max<size_t>(1, wg_size);
    const size_t num_wg = (static_cast<size_t>(max_entries) + wg - 1) / wg;
    const bool has_w = (d_weights != nullptr && d_b_weights != nullptr);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * wg, wg),
            [=](sycl::nd_item<1> item) {
                const int p = static_cast<int>(item.get_global_linear_id());
                const int total = d_bucket_off[num_wins];
                if (p >= total) return;
                const int m = d_bucket_m[p];
                d_b_first[p] = d_first[m];
                d_b_shift[p] = d_shift[m];
                if (has_w) {
                    const SYCL_NUFFT_SCALAR_T* src = d_weights + static_cast<size_t>(m) * static_cast<size_t>(W);
                    SYCL_NUFFT_SCALAR_T* dst = d_b_weights + static_cast<size_t>(p) * static_cast<size_t>(W);
                    #pragma unroll
                    for (int j = 0; j < W; ++j) {
                        dst[j] = src[j];
                    }
                }
            });
    });
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
// 2a. Fast Windowed SMEM Spreading with Bucket-Packed Precomputed Weights
//     - Interleaved 64-bit PairT (LDS.64/STS.64) for FP16/FP32 -> 2x fewer LSU ops
//     - Bucket-packed d_first, d_shift, d_weights -> 0 indirection & 0 idx/W div
//     - Shrunk k_pow2 Phase 1b loop -> eliminates 50%-75% partial-tile waste
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
    const int nfft_mask   = nfft - 1;
    const int smem_grid_elems = b_per_wg * grid_stride;
    const int smem_sy_elems   = tile_m * sy_stride;
    const int smem_w_elems    = tile_m * W;

    const size_t bs_packed    = (chunk_bs + kLanes - 1) / kLanes;
    const size_t num_b_groups = (bs_packed + static_cast<size_t>(b_per_wg) - 1) / static_cast<size_t>(b_per_wg);
    const size_t total_wgs    = num_b_groups * static_cast<size_t>(num_wins);

    q.submit([&](sycl::handler& cgh) {
#if SYCL_NUFFT_PACK_SMEM
        using PairT = SYCL_FFT_SUFFIX(SpreadPairT);
        sycl::local_accessor<PairT, 1>   l_grid(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<PairT, 1>   l_sy(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
#else
        sycl::local_accessor<RealT, 1>   l_re(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_im(sycl::range<1>(static_cast<size_t>(smem_grid_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_re(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_im(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
#endif
        sycl::local_accessor<int, 1>     l_first(sycl::range<1>(static_cast<size_t>(tile_m)), cgh);
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
#if SYCL_NUFFT_PACK_SMEM
                    l_grid[idx] = PairT{SYCL_FFT_ZERO, SYCL_FFT_ZERO};
#else
                    l_re[idx] = SYCL_FFT_ZERO;
                    l_im[idx] = SYCL_FFT_ZERO;
#endif
                }
                sycl::group_barrier(grp);

                const int p_start = (num_wins == 1) ? 0 : d_bucket_off[win];
                const int p_end   = (num_wins == 1) ? M : d_bucket_off[win + 1];

                for (int p_base = p_start; p_base < p_end; p_base += tile_m) {
                    const int cur_tile = sycl::min(tile_m, p_end - p_base);

                    // Phase 1a: Coalesced contiguous loads from bucket-packed d_first & d_weights
                    if (tid < cur_tile) {
                        l_first[tid] = d_first[p_base + tid];
                    }
                    const int total_w = cur_tile * W;
                    const ScalarT* w_src = d_weights + static_cast<size_t>(p_base) * static_cast<size_t>(W);
                    for (int idx = tid; idx < total_w; idx += wg_size) {
                        l_w[idx] = w_src[idx];
                    }

                    // Phase 1b: Cooperative d_y load + contiguous d_shift multiply
                    int k_pow2 = tile_m;
                    int log2_k = log2_tile_m;
                    while (k_pow2 > 1 && (k_pow2 >> 1) >= cur_tile) {
                        k_pow2 >>= 1;
                        --log2_k;
                    }
                    const int k_mask   = k_pow2 - 1;
                    const int total_sy = b_per_wg * k_pow2;
                    for (int idx = tid; idx < total_sy; idx += wg_size) {
                        const int b_sub = idx >> log2_k;
                        const int k     = idx & k_mask;
                        if (k < cur_tile) {
                            const size_t bp = bp_base + static_cast<size_t>(b_sub);
                            if (bp < bs_packed) {
                                const int p_idx = p_base + k;
                                const int m_idx = (num_wins == 1) ? p_idx : d_bucket_m[p_idx];
                                RealT sy_re, sy_im;
                                SYCL_FFT_SUFFIX(load_shifted_y)(
                                    d_y, d_shift, bp, total_M_stride, chunk_bs, m_idx, p_idx, sy_re, sy_im);
                                const int sy_slot = k * sy_stride + b_sub;
#if SYCL_NUFFT_PACK_SMEM
                                l_sy[sy_slot] = PairT{sy_re, sy_im};
#else
                                l_sy_re[sy_slot] = sy_re;
                                l_sy_im[sy_slot] = sy_im;
#endif
                            }
                        }
                    }

                    sycl::group_barrier(grp);

                    // Phase 2: Congruence-partitioned, zero-atomic, bank-conflict-free accumulation
                    if (my_bp < bs_packed) {
                        for (int k = 0; k < cur_tile; ++k) {
                            const int first = l_first[k];
                            const int j0 = (lane_t - first) & tj_mask;
#if SYCL_NUFFT_PACK_SMEM
                            const PairT sy_p = l_sy[k * sy_stride + lane_b];
                            const RealT sy_re = sy_p.re;
                            const RealT sy_im = sy_p.im;
#else
                            const RealT sy_re = l_sy_re[k * sy_stride + lane_b];
                            const RealT sy_im = l_sy_im[k * sy_stride + lane_b];
#endif
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
#if SYCL_NUFFT_PACK_SMEM
                                    PairT g_p = l_grid[g_idx];
                                    g_p.re = SYCL_NUFFT_FMA(sy_re, wv, g_p.re);
                                    g_p.im = SYCL_NUFFT_FMA(sy_im, wv, g_p.im);
                                    l_grid[g_idx] = g_p;
#else
                                    l_re[g_idx] = SYCL_NUFFT_FMA(sy_re, wv, l_re[g_idx]);
                                    l_im[g_idx] = SYCL_NUFFT_FMA(sy_im, wv, l_im[g_idx]);
#endif
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
                        const PairT g_p = l_grid[g_off + n];
                        sycl_fft::f16x2_store_pair(dst0, dst1, n, v0, v1, g_p.re, g_p.im);
                    }
#elif defined(SYCL_FFT_DF)
                    ComplexT* dst = d_fft_buf + bp_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    for (int n = tid; n < win_elems; n += wg_size) {
                        dst[n] = ComplexT{l_re[g_off + n], l_im[g_off + n]};
                    }
#elif SYCL_NUFFT_PACK_SMEM
                    ComplexT* dst = d_fft_buf + bp_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    for (int n = tid; n < win_elems; n += wg_size) {
                        const PairT g_p = l_grid[g_off + n];
                        dst[n] = ComplexT(g_p.re, g_p.im);
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
// 2a-reg. Register-Resident Windowed Spreading (widened Q in {2, 4, 8, 16, 32},
//         no hardcoded register-file byte cap: spill probe + Phase A measurement decide)
//     - ZERO grid SMEM (acc_re[Q], acc_im[Q] promoted to scalar thread registers via
//       compile-time binary-search constant-index helpers)
//     - ZERO grid SMEM zeroing, ZERO initial barrier, ZERO grid LDS/STS in inner loop
//     - Direct coalesced register-to-global writeback
// ============================================================================
template <int W, int Q, bool MultiPass>
struct SYCL_FFT_SUFFIX(SpreadRegKernelTag) {};

#if defined(SYCL_FFT_DF)
template <int Lo, int Hi>
SYCL_FFT_FORCE_INLINE void reg_acc_df_get(
    int q_idx,
    const sycl_fft::df_t* acc_re,
    const sycl_fft::df_t* acc_im,
    sycl_fft::df_t& cr,
    sycl_fft::df_t& ci) {
    if constexpr (Hi - Lo == 1) {
        cr = acc_re[Lo];
        ci = acc_im[Lo];
    } else {
        constexpr int Mid = (Lo + Hi) / 2;
        if (q_idx < Mid) {
            reg_acc_df_get<Lo, Mid>(q_idx, acc_re, acc_im, cr, ci);
        } else {
            reg_acc_df_get<Mid, Hi>(q_idx, acc_re, acc_im, cr, ci);
        }
    }
}

template <int Lo, int Hi>
SYCL_FFT_FORCE_INLINE void reg_acc_df_set(
    int q_idx,
    sycl_fft::df_t* acc_re,
    sycl_fft::df_t* acc_im,
    sycl_fft::df_t cr,
    sycl_fft::df_t ci) {
    if constexpr (Hi - Lo == 1) {
        acc_re[Lo] = cr;
        acc_im[Lo] = ci;
    } else {
        constexpr int Mid = (Lo + Hi) / 2;
        if (q_idx < Mid) {
            reg_acc_df_set<Lo, Mid>(q_idx, acc_re, acc_im, cr, ci);
        } else {
            reg_acc_df_set<Mid, Hi>(q_idx, acc_re, acc_im, cr, ci);
        }
    }
}
#else
template <int Lo, int Hi, typename RealT>
SYCL_FFT_FORCE_INLINE void reg_acc_fma_range(
    int q_idx, RealT* acc_re, RealT* acc_im, RealT sy_re, RealT sy_im, RealT wv) {
    if constexpr (Hi - Lo == 1) {
        acc_re[Lo] = SYCL_NUFFT_FMA(sy_re, wv, acc_re[Lo]);
        acc_im[Lo] = SYCL_NUFFT_FMA(sy_im, wv, acc_im[Lo]);
    } else {
        constexpr int Mid = (Lo + Hi) / 2;
        if (q_idx < Mid) {
            reg_acc_fma_range<Lo, Mid>(q_idx, acc_re, acc_im, sy_re, sy_im, wv);
        } else {
            reg_acc_fma_range<Mid, Hi>(q_idx, acc_re, acc_im, sy_re, sy_im, wv);
        }
    }
}
#endif

template <int W, int Q, bool MultiPass>
inline void SYCL_FFT_SUFFIX(launch_spread_smem_win_reg_impl)(
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
    const int sy_stride   = b_per_wg + 1;
    const int win_elems   = cfg.win_elems;
    const int num_wins    = cfg.num_wins;
    const int log2_nw     = int_log2_pow2(num_wins);
    const int nw_mask     = num_wins - 1;
    const int tile_m      = cfg.tile_m;
    const int log2_tile_m = int_log2_pow2(tile_m);
    const int nfft_mask   = nfft - 1;
    const int smem_sy_elems = tile_m * sy_stride;
    const int smem_w_elems  = tile_m * W;

    const size_t bs_packed    = (chunk_bs + kLanes - 1) / kLanes;
    const size_t num_b_groups = (bs_packed + static_cast<size_t>(b_per_wg) - 1) / static_cast<size_t>(b_per_wg);
    const size_t total_wgs    = num_b_groups * static_cast<size_t>(num_wins);

    q.submit([&](sycl::handler& cgh) {
#if SYCL_NUFFT_PACK_SMEM
        using PairT = SYCL_FFT_SUFFIX(SpreadPairT);
        sycl::local_accessor<PairT, 1>   l_sy(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
#else
        sycl::local_accessor<RealT, 1>   l_sy_re(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_im(sycl::range<1>(static_cast<size_t>(smem_sy_elems)), cgh);
#endif
        sycl::local_accessor<int, 1>     l_first(sycl::range<1>(static_cast<size_t>(tile_m)), cgh);
        sycl::local_accessor<ScalarT, 1> l_w(sycl::range<1>(static_cast<size_t>(smem_w_elems)), cgh);

        cgh.parallel_for<SYCL_FFT_SUFFIX(SpreadRegKernelTag)<W, Q, MultiPass>>(
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

                RealT acc_re[Q], acc_im[Q];
                #pragma unroll
                for (int qi = 0; qi < Q; ++qi) {
                    acc_re[qi] = SYCL_FFT_ZERO;
                    acc_im[qi] = SYCL_FFT_ZERO;
                }

                const int p_start = (num_wins == 1) ? 0 : d_bucket_off[win];
                const int p_end   = (num_wins == 1) ? M : d_bucket_off[win + 1];

                for (int p_base = p_start; p_base < p_end; p_base += tile_m) {
                    const int cur_tile = sycl::min(tile_m, p_end - p_base);

                    if (tid < cur_tile) {
                        l_first[tid] = d_first[p_base + tid];
                    }
                    const int total_w = cur_tile * W;
                    const ScalarT* w_src = d_weights + static_cast<size_t>(p_base) * static_cast<size_t>(W);
                    for (int idx = tid; idx < total_w; idx += wg_size) {
                        l_w[idx] = w_src[idx];
                    }

                    int k_pow2 = tile_m;
                    int log2_k = log2_tile_m;
                    while (k_pow2 > 1 && (k_pow2 >> 1) >= cur_tile) {
                        k_pow2 >>= 1;
                        --log2_k;
                    }
                    const int k_mask   = k_pow2 - 1;
                    const int total_sy = b_per_wg * k_pow2;
                    for (int idx = tid; idx < total_sy; idx += wg_size) {
                        const int b_sub = idx >> log2_k;
                        const int k     = idx & k_mask;
                        if (k < cur_tile) {
                            const size_t bp = bp_base + static_cast<size_t>(b_sub);
                            if (bp < bs_packed) {
                                const int p_idx = p_base + k;
                                const int m_idx = (num_wins == 1) ? p_idx : d_bucket_m[p_idx];
                                RealT sy_re, sy_im;
                                SYCL_FFT_SUFFIX(load_shifted_y)(
                                    d_y, d_shift, bp, total_M_stride, chunk_bs, m_idx, p_idx, sy_re, sy_im);
                                const int sy_slot = k * sy_stride + b_sub;
#if SYCL_NUFFT_PACK_SMEM
                                l_sy[sy_slot] = PairT{sy_re, sy_im};
#else
                                l_sy_re[sy_slot] = sy_re;
                                l_sy_im[sy_slot] = sy_im;
#endif
                            }
                        }
                    }

                    sycl::group_barrier(grp);

                    if (my_bp < bs_packed) {
                        for (int k = 0; k < cur_tile; ++k) {
                            const int first = l_first[k];
                            const int j0 = (lane_t - first) & tj_mask;
#if SYCL_NUFFT_PACK_SMEM
                            const PairT sy_p = l_sy[k * sy_stride + lane_b];
                            const RealT sy_re = sy_p.re;
                            const RealT sy_im = sy_p.im;
#else
                            const RealT sy_re = l_sy_re[k * sy_stride + lane_b];
                            const RealT sy_im = l_sy_im[k * sy_stride + lane_b];
#endif
                            const int w_base = k * W;

                            auto accum_bin = [&](int j_val) SYCL_FFT_LAMBDA_INLINE {
                                const int bin = ((first + j_val) & nfft_mask) - win_lo;
                                if (num_wins == 1 ||
                                    static_cast<unsigned>(bin) < static_cast<unsigned>(win_elems)) {
                                    const int q_idx = bin >> log2_tj;
                                    const ScalarT ws = l_w[w_base + j_val];
#if defined(SYCL_FFT_HALF)
                                    const RealT wv = SYCL_FFT_F16X2_SPLAT_H(ws);
#else
                                    const RealT wv = ws;
#endif
#if defined(SYCL_FFT_DF)
                                    const RealT pr = sycl_fft::df_mul(sy_re, wv);
                                    const RealT pi = sycl_fft::df_mul(sy_im, wv);
                                    RealT cr, ci;
                                    reg_acc_df_get<0, Q>(q_idx, acc_re, acc_im, cr, ci);
                                    cr = sycl_fft::df_add(cr, pr);
                                    ci = sycl_fft::df_add(ci, pi);
                                    reg_acc_df_set<0, Q>(q_idx, acc_re, acc_im, cr, ci);
#else
                                    reg_acc_fma_range<0, Q>(q_idx, acc_re, acc_im, sy_re, sy_im, wv);
#endif
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

                    if (p_base + tile_m < p_end) {
                        sycl::group_barrier(grp);
                    }
                }

                // Direct coalesced register-to-global writeback (0 SMEM round-trip!)
                if (my_bp < bs_packed) {
#if defined(SYCL_FFT_HALF)
                    const size_t b0_out = 2 * my_bp;
                    const size_t b1_out = 2 * my_bp + 1;
                    const bool v0 = (b0_out < chunk_bs);
                    const bool v1 = (b1_out < chunk_bs);
                    ComplexT* dst0 = d_fft_buf + b0_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    ComplexT* dst1 = v1 ? (d_fft_buf + b1_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo)) : dst0;
                    #pragma unroll
                    for (int qi = 0; qi < Q; ++qi) {
                        sycl_fft::f16x2_store_pair(dst0, dst1, qi * t_j + lane_t, v0, v1, acc_re[qi], acc_im[qi]);
                    }
#elif defined(SYCL_FFT_DF)
                    ComplexT* dst = d_fft_buf + my_bp * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    #pragma unroll
                    for (int qi = 0; qi < Q; ++qi) {
                        dst[qi * t_j + lane_t] = ComplexT{acc_re[qi], acc_im[qi]};
                    }
#else
                    ComplexT* dst = d_fft_buf + my_bp * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                    #pragma unroll
                    for (int qi = 0; qi < Q; ++qi) {
                        dst[qi * t_j + lane_t] = ComplexT(acc_re[qi], acc_im[qi]);
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
                                    d_y, d_shift, bp, total_M_stride, chunk_bs, m_idx, m_idx, sy_re, sy_im);
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
        if (cfg.use_reg_acc) {
            const int q_val = cfg.win_elems / cfg.t_j;
            auto launch_reg_q = [&](auto q_ic) {
                constexpr int Q = q_ic.value;
                if constexpr (kCanMultiPass) {
                    if (cfg.t_j < W) {
                        SYCL_FFT_SUFFIX(launch_spread_smem_win_reg_impl)<W, Q, true>(
                            q, d_weights, d_first, d_shift, d_bucket_off, d_bucket_m,
                            d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, cfg);
                        return;
                    }
                }
                SYCL_FFT_SUFFIX(launch_spread_smem_win_reg_impl)<W, Q, false>(
                    q, d_weights, d_first, d_shift, d_bucket_off, d_bucket_m,
                    d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, cfg);
            };
            if constexpr (W > 16) {
                if (q_val == 2) { launch_reg_q(std::integral_constant<int, 2>{}); return; }
            }
            if (q_val == 4)  { launch_reg_q(std::integral_constant<int, 4>{});  return; }
            if (q_val == 8)  { launch_reg_q(std::integral_constant<int, 8>{});  return; }
            if (q_val == 16) { launch_reg_q(std::integral_constant<int, 16>{}); return; }
            if (q_val == 32) { launch_reg_q(std::integral_constant<int, 32>{}); return; }
        }
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
                        d_y, d_shift, bp, total_M_stride, chunk_bs, m, m, sy_re, sy_im);
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

void SYCL_FFT_SUFFIX(launch_pack_bucket_geom)(
    sycl::queue& q, int w, int num_wins, int max_entries,
    const int* d_bucket_off, const int* d_bucket_m,
    const int* d_first, const SYCL_NUFFT_COMPLEX_T* d_shift, const SYCL_NUFFT_SCALAR_T* d_weights,
    int* d_b_first, SYCL_NUFFT_COMPLEX_T* d_b_shift, SYCL_NUFFT_SCALAR_T* d_b_weights,
    size_t wg_size) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<4>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    else SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<5>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<16>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    else if (w == 17) SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<17>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    else SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<18>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
#elif defined(SYCL_FFT_DF)
    if (w == 14) SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<14>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    else if (w == 15) SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<15>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    else SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<16>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
#else
    if (w == 8) SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<8>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    else SYCL_FFT_SUFFIX(launch_pack_bucket_geom_impl)<9>(q, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
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

template <int W>
inline sycl_fft::KernelProbeInfo SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)(
    sycl::queue& q, const sycl::device& dev, const SpreadConfig& cfg) {
    if (!cfg.use_reg_acc || cfg.t_j <= 0) return sycl_fft::KernelProbeInfo{};
    constexpr bool kCanMultiPass = (W == 5 || W == 9 || W == 17 || W == 18);
    const int q_val = cfg.win_elems / cfg.t_j;
    auto probe_q = [&](auto q_ic) -> sycl_fft::KernelProbeInfo {
        constexpr int Q = q_ic.value;
        if constexpr (kCanMultiPass) {
            if (cfg.t_j < W) {
                return sycl_fft::probe_kernel_bundle<SYCL_FFT_SUFFIX(SpreadRegKernelTag)<W, Q, true>>(q, dev);
            }
        }
        return sycl_fft::probe_kernel_bundle<SYCL_FFT_SUFFIX(SpreadRegKernelTag)<W, Q, false>>(q, dev);
    };
    if constexpr (W > 16) {
        if (q_val == 2) return probe_q(std::integral_constant<int, 2>{});
    }
    if (q_val == 4)  return probe_q(std::integral_constant<int, 4>{});
    if (q_val == 8)  return probe_q(std::integral_constant<int, 8>{});
    if (q_val == 16) return probe_q(std::integral_constant<int, 16>{});
    if (q_val == 32) return probe_q(std::integral_constant<int, 32>{});
    return sycl_fft::KernelProbeInfo{};
}

sycl_fft::KernelProbeInfo SYCL_FFT_SUFFIX(probe_spread_reg_kernel)(
    sycl::queue& q, const sycl::device& dev, int w, const SpreadConfig& cfg) {
#if defined(SYCL_FFT_HALF)
    if (w == 4) return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<4>(q, dev, cfg);
    return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<5>(q, dev, cfg);
#elif defined(SYCL_FFT_DOUBLE)
    if (w == 16) return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<16>(q, dev, cfg);
    if (w == 17) return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<17>(q, dev, cfg);
    return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<18>(q, dev, cfg);
#elif defined(SYCL_FFT_DF)
    if (w == 14) return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<14>(q, dev, cfg);
    if (w == 15) return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<15>(q, dev, cfg);
    return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<16>(q, dev, cfg);
#else
    if (w == 8) return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<8>(q, dev, cfg);
    return SYCL_FFT_SUFFIX(probe_spread_reg_dispatch_w)<9>(q, dev, cfg);
#endif
}

#undef SYCL_NUFFT_COMPLEX_T
#undef SYCL_NUFFT_REAL_T
#undef SYCL_NUFFT_SCALAR_T
#undef SYCL_NUFFT_LANES
#undef SYCL_NUFFT_FMA
#undef SYCL_NUFFT_PACK_SMEM

} // namespace detail
} // namespace sycl_nufft
