// Auxiliary NuFFT-1 Execution Module (separate HCF, zero PSWF polynomial tables):
//   1. High-precision x-axis coordinate gridding (FP64 for FP64; Dekker df_t for DF/FP32/FP16)
//   2. O(M) GPU window-bucket builder for multi-window shared-memory spreading
//   3. Raw-size-deduplicated grid transpose + L2 zeroing (4B FP32, shared 8B FP64/DF, 2xFP16 pair)
#include "nufft/nufft1.hpp"

namespace sycl_nufft {
namespace detail {

namespace {

template <typename ScalarT, typename ComplexT, bool IsHalf, bool IsDf>
void launch_precompute_points_df_x(
    sycl::queue& q,
    const sycl_fft::df_t* d_x,
    sycl_fft::df_t df_scaled,
    int M,
    int nfft,
    int w,
    int output_shift,
    bool mode43,
    int* d_first,
    ScalarT* d_u,
    ComplexT* d_shift,
    size_t wg_size) {
    const size_t wg = std::max<size_t>(1, wg_size);
    const size_t num_wg = (static_cast<size_t>(M) + wg - 1) / wg;
    const int phase_shift = mode43 ? (output_shift / 3) : output_shift;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * wg, wg),
            [=](sycl::nd_item<1> item) {
                using namespace sycl_fft;
                const int m = static_cast<int>(item.get_global_linear_id());
                if (m >= M) return;

                const df_t xm = df_mul(d_x[m], df_scaled);
                const float nfft_f = static_cast<float>(nfft);
                const float inv_nfft_f = 1.0f / nfft_f;
                const df_t x_n = df_scale_pow2(xm, nfft_f);
                const df_t half_w_df = df_make(0.5f * static_cast<float>(w), 0.0f);
                const df_t m_left_df = df_ceil(df_sub(x_n, half_w_df));

                const df_t dist0 = df_sub(m_left_df, x_n);
                df_t u_df = df_add(df_scale_pow2(dist0, 2.0f),
                                   df_make(static_cast<float>(w - 1), 0.0f));
                if (u_df.hi < -1.0f || (u_df.hi == -1.0f && u_df.lo < 0.0f)) {
                    u_df = df_make(-1.0f, 0.0f);
                }
                if (u_df.hi > 1.0f || (u_df.hi == 1.0f && u_df.lo > 0.0f)) {
                    u_df = df_make(1.0f, 0.0f);
                }

                // Exact power-of-two modulo in df_t (m_left_df.hi and m_left_df.lo are exact integers)
                const float r_hi = m_left_df.hi - sycl::floor(m_left_df.hi * inv_nfft_f) * nfft_f;
                const float r_lo = m_left_df.lo - sycl::floor(m_left_df.lo * inv_nfft_f) * nfft_f;
                const int first = (static_cast<int>(r_hi) + static_cast<int>(r_lo)) & (nfft - 1);

                const df_t phase_df = df_mul_f(xm, static_cast<float>(phase_shift));
                const df_t frac_df = df_sub(phase_df, df_floor(phase_df));

                d_first[m] = first;
                if constexpr (IsDf) {
                    df_t c, s;
                    sincos2pidf_branchless<+1>(frac_df, c, s);
                    df_t sr = c, si = s;
                    if (mode43) {
                        const df_t c2 = df_sqr(c);
                        const df_t s2 = df_sqr(s);
                        sr = df_mul(c, df_sub(df_scale_pow2(c2, 4.0f), df_make(3.0f, 0.0f)));
                        si = df_mul(s, df_sub(df_make(3.0f, 0.0f), df_scale_pow2(s2, 4.0f)));
                    }
                    d_u[m]     = u_df;
                    d_shift[m] = Complexdf{sr, si};
                } else if constexpr (IsHalf) {
                    const float frac_f = frac_df.hi + frac_df.lo;
                    float c, s;
                    sincos2pif_branchless<+1>(frac_f, c, s);
                    float sr = c, si = s;
                    if (mode43) {
                        sr = c * sycl::fma(4.0f, c * c, -3.0f);
                        si = s * sycl::fma(-4.0f, s * s, 3.0f);
                    }
                    const float u_f = u_df.hi + u_df.lo;
                    const f16_scalar_t sr_h = static_cast<f16_scalar_t>(sr);
                    const f16_scalar_t si_h = static_cast<f16_scalar_t>(si);
                    d_u[m]     = static_cast<f16_scalar_t>(u_f);
                    d_shift[m] = SYCL_FFT_BIT_CAST(Complexh, (f16x2_t{sr_h, si_h}));
                } else {
                    const float frac_f = frac_df.hi + frac_df.lo;
                    float c, s;
                    sincos2pif_branchless<+1>(frac_f, c, s);
                    float sr = c, si = s;
                    if (mode43) {
                        sr = c * sycl::fma(4.0f, c * c, -3.0f);
                        si = s * sycl::fma(-4.0f, s * s, 3.0f);
                    }
                    d_u[m]     = u_df.hi + u_df.lo;
                    d_shift[m] = Complexf(sr, si);
                }
            });
    });
}

template <typename RawRealT, typename RawComplexT>
void launch_transpose_and_zero_raw(
    sycl::queue& q,
    RawRealT* d_grid_T_re,
    RawRealT* d_grid_T_im,
    RawComplexT* d_fft_buf,
    int nfft,
    size_t chunk_bs,
    size_t packed_stride,
    int tile,
    int stride) {
    const size_t groups_n = (static_cast<size_t>(nfft) + static_cast<size_t>(tile) - 1) / static_cast<size_t>(tile);
    const size_t groups_b = (chunk_bs + static_cast<size_t>(tile) - 1) / static_cast<size_t>(tile);
    const size_t smem_elems = static_cast<size_t>(tile) * static_cast<size_t>(stride);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<RawRealT, 1> l_re(sycl::range<1>(smem_elems), cgh);
        sycl::local_accessor<RawRealT, 1> l_im(sycl::range<1>(smem_elems), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n * static_cast<size_t>(tile),
                                             groups_b * static_cast<size_t>(tile)),
                              sycl::range<2>(static_cast<size_t>(tile),
                                             static_cast<size_t>(tile))),
            [=](sycl::nd_item<2> item) {
                const auto grp = item.get_group();
                const int ly = static_cast<int>(item.get_local_id(0));
                const int lx = static_cast<int>(item.get_local_id(1));
                const size_t n_base = item.get_group(0) * static_cast<size_t>(tile);
                const size_t b_base = item.get_group(1) * static_cast<size_t>(tile);

                const size_t n_in  = n_base + static_cast<size_t>(ly);
                const size_t bp_in = b_base + static_cast<size_t>(lx);
                RawRealT r_val = 0;
                RawRealT i_val = 0;
                if (n_in < static_cast<size_t>(nfft) && bp_in < chunk_bs) {
                    const size_t g_off = n_in * packed_stride + bp_in;
                    r_val = d_grid_T_re[g_off];
                    i_val = d_grid_T_im[g_off];
                    d_grid_T_re[g_off] = 0;
                    d_grid_T_im[g_off] = 0;
                }
                l_re[lx * stride + ly] = r_val;
                l_im[lx * stride + ly] = i_val;

                sycl::group_barrier(grp);

                const size_t bp_out = b_base + static_cast<size_t>(ly);
                const size_t n_out  = n_base + static_cast<size_t>(lx);
                if (bp_out < chunk_bs && n_out < static_cast<size_t>(nfft)) {
                    const RawRealT tr = l_re[ly * stride + lx];
                    const RawRealT ti = l_im[ly * stride + lx];
                    if constexpr (sizeof(RawRealT) == 4) {
                        const uint64_t packed = static_cast<uint64_t>(tr) | (static_cast<uint64_t>(ti) << 32);
                        d_fft_buf[bp_out * static_cast<size_t>(nfft) + n_out] = packed;
                    } else {
                        d_fft_buf[bp_out * static_cast<size_t>(nfft) + n_out] = sycl::ulong2(tr, ti);
                    }
                }
            });
    });
}

} // namespace

void launch_precompute_pointsh(
    sycl::queue& q, const sycl_fft::df_t* d_x, sycl_fft::df_t df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43,
    int* d_first, sycl_fft::f16_scalar_t* d_u, sycl_fft::Complexh* d_shift,
    size_t wg_size) {
    launch_precompute_points_df_x<sycl_fft::f16_scalar_t, sycl_fft::Complexh, true, false>(
        q, d_x, df_scaled, M, nfft, w, output_shift, mode43, d_first, d_u, d_shift, wg_size);
}

void launch_precompute_pointsf(
    sycl::queue& q, const sycl_fft::df_t* d_x, sycl_fft::df_t df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43,
    int* d_first, float* d_u, sycl_fft::Complexf* d_shift,
    size_t wg_size) {
    launch_precompute_points_df_x<float, sycl_fft::Complexf, false, false>(
        q, d_x, df_scaled, M, nfft, w, output_shift, mode43, d_first, d_u, d_shift, wg_size);
}

void launch_precompute_pointsdf(
    sycl::queue& q, const sycl_fft::df_t* d_x, sycl_fft::df_t df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43,
    int* d_first, sycl_fft::df_t* d_u, sycl_fft::Complexdf* d_shift,
    size_t wg_size) {
    launch_precompute_points_df_x<sycl_fft::df_t, sycl_fft::Complexdf, false, true>(
        q, d_x, df_scaled, M, nfft, w, output_shift, mode43, d_first, d_u, d_shift, wg_size);
}

void launch_precompute_points(
    sycl::queue& q, const double* d_x, double df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43,
    int* d_first, double* d_u, sycl_fft::Complex* d_shift,
    size_t wg_size) {
    const size_t wg = std::max<size_t>(1, wg_size);
    const size_t num_wg = (static_cast<size_t>(M) + wg - 1) / wg;
    const int phase_shift = mode43 ? (output_shift / 3) : output_shift;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * wg, wg),
            [=](sycl::nd_item<1> item) {
                const int m = static_cast<int>(item.get_global_linear_id());
                if (m >= M) return;

                const double xm = d_x[m] * df_scaled;
                const double nfft_d = static_cast<double>(nfft);
                const double inv_nfft_d = 1.0 / nfft_d;
                const double x_n = xm * nfft_d;
                const double m_left_d = sycl::ceil(x_n - 0.5 * static_cast<double>(w));

                double u = 2.0 * (m_left_d - x_n) + static_cast<double>(w - 1);
                if (u < -1.0) u = -1.0;
                if (u >  1.0) u =  1.0;

                const double rem_d = m_left_d - sycl::floor(m_left_d * inv_nfft_d) * nfft_d;
                const int first = static_cast<int>(rem_d) & (nfft - 1);

                const double phase = xm * static_cast<double>(phase_shift);
                double c, s;
                sycl_fft::sincos2pi_branchless<+1>(phase, c, s);
                double sr = c, si = s;
                if (mode43) {
                    sr = c * sycl::fma(4.0, c * c, -3.0);
                    si = s * sycl::fma(-4.0, s * s, 3.0);
                }

                d_first[m] = first;
                d_u[m]     = u;
                d_shift[m] = sycl_fft::Complex(sr, si);
            });
    });
}

void launch_build_window_buckets(
    sycl::queue& q, const int* d_first, int M, int nfft, int w,
    int win_elems, int num_wins,
    int* d_bucket_counts, int* d_bucket_off, int* d_bucket_m,
    size_t wg_size) {
    (void)nfft;
    if (num_wins <= 1 || M <= 0) return;

    const int log2_win = int_log2_pow2(win_elems);
    const int win_mask = win_elems - 1;
    const int nw_mask  = num_wins - 1;
    const size_t wg = std::max<size_t>(1, wg_size);
    const size_t num_wg = (static_cast<size_t>(M) + wg - 1) / wg;

    q.memset(d_bucket_counts, 0, static_cast<size_t>(num_wins) * sizeof(int));

    // Pass 1: Histogram point counts per grid window (straddling points enter both adjacent windows)
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * wg, wg),
            [=](sycl::nd_item<1> item) {
                const int m = static_cast<int>(item.get_global_linear_id());
                if (m >= M) return;
                const int first = d_first[m];
                const int w0 = first >> log2_win;
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(d_bucket_counts[w0]).fetch_add(1);
                if ((first & win_mask) + w > win_elems) {
                    const int w1 = (w0 + 1) & nw_mask;
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(d_bucket_counts[w1]).fetch_add(1);
                }
            });
    });

    // Pass 2: Exclusive prefix sum over num_wins (single work-item, num_wins is small <= 1024)
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>(1),
            [=](sycl::id<1>) {
                int sum = 0;
                for (int wi = 0; wi < num_wins; ++wi) {
                    const int cnt = d_bucket_counts[wi];
                    d_bucket_off[wi] = sum;
                    d_bucket_counts[wi] = sum; // reuse as atomic write cursor for Pass 3
                    sum += cnt;
                }
                d_bucket_off[num_wins] = sum;
            });
    });

    // Pass 3: Scatter original point indices into contiguous per-window buckets
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * wg, wg),
            [=](sycl::nd_item<1> item) {
                const int m = static_cast<int>(item.get_global_linear_id());
                if (m >= M) return;
                const int first = d_first[m];
                const int w0 = first >> log2_win;
                const int p0 = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>(d_bucket_counts[w0]).fetch_add(1);
                d_bucket_m[p0] = m;
                if ((first & win_mask) + w > win_elems) {
                    const int w1 = (w0 + 1) & nw_mask;
                    const int p1 = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>(d_bucket_counts[w1]).fetch_add(1);
                    d_bucket_m[p1] = m;
                }
            });
    });
}

void launch_transpose_and_zero_4b(
    sycl::queue& q, uint32_t* d_grid_T_re, uint32_t* d_grid_T_im,
    uint64_t* d_fft_buf, int nfft, size_t chunk_bs, size_t packed_stride,
    int tile, int stride) {
    launch_transpose_and_zero_raw<uint32_t, uint64_t>(
        q, d_grid_T_re, d_grid_T_im, d_fft_buf, nfft, chunk_bs, packed_stride, tile, stride);
}

void launch_transpose_and_zero_8b(
    sycl::queue& q, uint64_t* d_grid_T_re, uint64_t* d_grid_T_im,
    sycl::ulong2* d_fft_buf, int nfft, size_t chunk_bs, size_t packed_stride,
    int tile, int stride) {
    launch_transpose_and_zero_raw<uint64_t, sycl::ulong2>(
        q, d_grid_T_re, d_grid_T_im, d_fft_buf, nfft, chunk_bs, packed_stride, tile, stride);
}

void launch_transpose_and_zero_f16pair(
    sycl::queue& q, sycl_fft::f16x2_t* d_grid_T_re, sycl_fft::f16x2_t* d_grid_T_im,
    sycl_fft::Complexh* d_fft_buf, int nfft, size_t chunk_bs, size_t packed_stride,
    int tile, int stride) {
    const size_t bs_packed = (chunk_bs + 1) / 2;
    const size_t groups_n = (static_cast<size_t>(nfft) + static_cast<size_t>(tile) - 1) / static_cast<size_t>(tile);
    const size_t groups_b = (bs_packed + static_cast<size_t>(tile) - 1) / static_cast<size_t>(tile);
    const size_t smem_elems = static_cast<size_t>(tile) * static_cast<size_t>(stride);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl_fft::f16x2_t, 1> l_re(sycl::range<1>(smem_elems), cgh);
        sycl::local_accessor<sycl_fft::f16x2_t, 1> l_im(sycl::range<1>(smem_elems), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n * static_cast<size_t>(tile),
                                             groups_b * static_cast<size_t>(tile)),
                              sycl::range<2>(static_cast<size_t>(tile),
                                             static_cast<size_t>(tile))),
            [=](sycl::nd_item<2> item) {
                const auto grp = item.get_group();
                const int ly = static_cast<int>(item.get_local_id(0));
                const int lx = static_cast<int>(item.get_local_id(1));
                const size_t n_base = item.get_group(0) * static_cast<size_t>(tile);
                const size_t b_base = item.get_group(1) * static_cast<size_t>(tile);

                const size_t n_in  = n_base + static_cast<size_t>(ly);
                const size_t bp_in = b_base + static_cast<size_t>(lx);
                sycl_fft::f16x2_t r_val = sycl_fft::kZero_f16x2;
                sycl_fft::f16x2_t i_val = sycl_fft::kZero_f16x2;
                if (n_in < static_cast<size_t>(nfft) && bp_in < bs_packed) {
                    const size_t g_off = n_in * packed_stride + bp_in;
                    r_val = d_grid_T_re[g_off];
                    i_val = d_grid_T_im[g_off];
                    d_grid_T_re[g_off] = sycl_fft::kZero_f16x2;
                    d_grid_T_im[g_off] = sycl_fft::kZero_f16x2;
                }
                l_re[lx * stride + ly] = r_val;
                l_im[lx * stride + ly] = i_val;

                sycl::group_barrier(grp);

                const size_t bp_out = b_base + static_cast<size_t>(ly);
                const size_t n_out  = n_base + static_cast<size_t>(lx);
                if (bp_out < bs_packed && n_out < static_cast<size_t>(nfft)) {
                    const sycl_fft::f16x2_t tr = l_re[ly * stride + lx];
                    const sycl_fft::f16x2_t ti = l_im[ly * stride + lx];
                    const size_t b0 = 2 * bp_out;
                    const size_t b1 = 2 * bp_out + 1;
                    sycl_fft::f16x2_store_pair(
                        d_fft_buf + b0 * static_cast<size_t>(nfft),
                        d_fft_buf + b1 * static_cast<size_t>(nfft),
                        static_cast<int>(n_out),
                        b0 < chunk_bs,
                        b1 < chunk_bs,
                        tr,
                        ti);
                }
            });
    });
}

} // namespace detail
} // namespace sycl_nufft
