// Multi-includable Quadruple-Precision SYCL Type-1 NuFFT (NuFFT-1) Kernels & Plan.
// Intentionally has no #pragma once; included 4x by sycl_nufft1.hpp with
// sycl_fft_precision.hpp configured for FP16 (h), FP32 (f), FP64 (raw), and DF (df).

namespace sycl_nufft {

using namespace sycl_fft;

#if defined(SYCL_FFT_HALF)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexh
#define SYCL_NUFFT_REAL_T    sycl_fft::f16x2_t
#define SYCL_NUFFT_SCALAR_T  sycl_fft::f16_scalar_t
#define SYCL_NUFFT_XCOORD_T  sycl_fft::df_t
#define SYCL_NUFFT_LANES     2
#define SYCL_NUFFT_SMEM_CAP  4096
#define SYCL_NUFFT_FMA(a, b, c) SYCL_FFT_F16X2_FMA((a), (b), (c))
#elif defined(SYCL_FFT_DOUBLE)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complex
#define SYCL_NUFFT_REAL_T    double
#define SYCL_NUFFT_SCALAR_T  double
#define SYCL_NUFFT_XCOORD_T  double
#define SYCL_NUFFT_LANES     1
#define SYCL_NUFFT_SMEM_CAP  2048
#define SYCL_NUFFT_FMA(a, b, c) sycl::fma((a), (b), (c))
#elif defined(SYCL_FFT_DF)
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexdf
#define SYCL_NUFFT_REAL_T    sycl_fft::df_t
#define SYCL_NUFFT_SCALAR_T  sycl_fft::df_t
#define SYCL_NUFFT_XCOORD_T  sycl_fft::df_t
#define SYCL_NUFFT_LANES     1
#define SYCL_NUFFT_SMEM_CAP  2048
#define SYCL_NUFFT_FMA(a, b, c) sycl_fft::df_add(sycl_fft::df_mul((a), (b)), (c))
#else
#define SYCL_NUFFT_COMPLEX_T sycl_fft::Complexf
#define SYCL_NUFFT_REAL_T    float
#define SYCL_NUFFT_SCALAR_T  float
#define SYCL_NUFFT_XCOORD_T  sycl_fft::df_t
#define SYCL_NUFFT_LANES     1
#define SYCL_NUFFT_SMEM_CAP  4096
#define SYCL_NUFFT_FMA(a, b, c) sycl::fma((a), (b), (c))
#endif

#if defined(SYCL_FFT_USE_EXTERN_TEMPLATES)

class SYCL_FFT_SUFFIX(Nufft1Plan) {
public:
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using RealT    = SYCL_NUFFT_REAL_T;
    using XCoordT  = SYCL_NUFFT_XCOORD_T;
    using FftPlanT = sycl_fft::SYCL_FFT_SUFFIX(DifFftPlan)<+1>;

    SYCL_FFT_SUFFIX(Nufft1Plan)(
        sycl::queue& q,
        int M,
        int N,
        int w,
        PswfMode mode,
        size_t max_bs,
        const sycl_fft::FftLaunchConfig& fft_cfg = {});

    ~SYCL_FFT_SUFFIX(Nufft1Plan)();

    SYCL_FFT_SUFFIX(Nufft1Plan)(const SYCL_FFT_SUFFIX(Nufft1Plan)&) = delete;
    SYCL_FFT_SUFFIX(Nufft1Plan)& operator=(const SYCL_FFT_SUFFIX(Nufft1Plan)&) = delete;

    // Precompute x-axis gridding (first_idx, sub-panel coordinate u, center phase shift)
    // in FP64 (for FP64) or Dekker double-float df_t (for DF, FP32, FP16).
    void set_points(const XCoordT* d_x, XCoordT df, int freq_factor = 1);

    // Execute NuFFT-1 for a batch of `bs` signals `d_y` [bs][M] -> `d_out` [bs][N_out].
    // Default output order is SlotBitReversed (natural-to-bit-reversed fast path).
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

private:
    sycl::queue& q_;
    int          M_;
    int          N_;
    int          w_;
    PswfMode     mode_;
    int          nfft_;
    int          log_nfft_;
    int          nout_;
    int          output_shift_;
    size_t       max_bs_;
    size_t       chunk_bs_;
    size_t       packed_stride_;

    FftPlanT     fft_plan_;

    int*         d_first_       = nullptr;
    ScalarT*     d_u_           = nullptr;
    ComplexT*    d_shift_       = nullptr;
    RealT*       d_grid_T_re_   = nullptr;
    RealT*       d_grid_T_im_   = nullptr;
    ComplexT*    d_fft_buf_     = nullptr;
};

#endif // defined(SYCL_FFT_USE_EXTERN_TEMPLATES)

#if !defined(SYCL_FFT_USE_EXTERN_TEMPLATES) || defined(SYCL_NUFFT_IMPLEMENTATION_TU)

namespace detail {

// Validate supported widths per precision:
//   FP16: {4, 5}, FP32: {8, 9}, DF: {14, 15, 16}, FP64: {16, 17, 18}
inline bool SYCL_FFT_SUFFIX(is_valid_pswf_width)(int w) {
#if defined(SYCL_FFT_HALF)
    return (w == 4 || w == 5);
#elif defined(SYCL_FFT_DOUBLE)
    return (w == 16 || w == 17 || w == 18);
#elif defined(SYCL_FFT_DF)
    return (w == 14 || w == 15 || w == 16);
#else
    return (w == 8 || w == 9);
#endif
}

// High-precision x-axis gridding kernel:
//   - Uses FP64 (double) for the FP64 implementation
//   - Uses Dekker double-float (df_t = 2xFP32) for DF, FP32, and FP16 implementations
//   - Outputs wrapped start index first_m in [0, N_fft), sub-panel coordinate u_m in [-1, 1],
//     and center phase rotation shift_m = exp(+2*pi*i * x_m * df * q * output_shift).
inline void SYCL_FFT_SUFFIX(launch_precompute_points)(
    sycl::queue& q,
    const SYCL_NUFFT_XCOORD_T* d_x,
    SYCL_NUFFT_XCOORD_T df_scaled,
    int M,
    int nfft,
    int w,
    int output_shift,
    bool mode43,
    int* d_first,
    SYCL_NUFFT_SCALAR_T* d_u,
    SYCL_NUFFT_COMPLEX_T* d_shift) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    (void)sizeof(ScalarT);

    constexpr size_t kWg = 256;
    const size_t num_wg = (static_cast<size_t>(M) + kWg - 1) / kWg;
    const int phase_shift = mode43 ? (output_shift / 3) : output_shift;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * kWg, kWg),
            [=](sycl::nd_item<1> item) {
                const int m = static_cast<int>(item.get_global_linear_id());
                if (m >= M) return;

#if defined(SYCL_FFT_DOUBLE)
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
                d_shift[m] = ComplexT(sr, si);
#else
                using namespace sycl_fft;
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

#if defined(SYCL_FFT_DF)
                df_t c, s;
                sincos2pidf_branchless<+1>(frac_df, c, s);
                df_t sr = c, si = s;
                if (mode43) {
                    const df_t c2 = df_sqr(c);
                    const df_t s2 = df_sqr(s);
                    sr = df_mul(c, df_sub(df_scale_pow2(c2, 4.0f), df_make(3.0f, 0.0f)));
                    si = df_mul(s, df_sub(df_make(3.0f, 0.0f), df_scale_pow2(s2, 4.0f)));
                }
                d_first[m] = first;
                d_u[m]     = u_df;
                d_shift[m] = ComplexT{sr, si};
#elif defined(SYCL_FFT_HALF)
                const float frac_f = frac_df.hi + frac_df.lo;
                float c, s;
                sincos2pif_branchless<+1>(frac_f, c, s);
                float sr = c, si = s;
                if (mode43) {
                    sr = c * sycl::fma(4.0f, c * c, -3.0f);
                    si = s * sycl::fma(-4.0f, s * s, 3.0f);
                }
                const float u_f = u_df.hi + u_df.lo;
                const ScalarT sr_h = static_cast<ScalarT>(sr);
                const ScalarT si_h = static_cast<ScalarT>(si);
                d_first[m] = first;
                d_u[m]     = static_cast<ScalarT>(u_f);
                d_shift[m] = SYCL_FFT_BIT_CAST(ComplexT, (sycl_fft::f16x2_t{sr_h, si_h}));
#else
                const float frac_f = frac_df.hi + frac_df.lo;
                float c, s;
                sincos2pif_branchless<+1>(frac_f, c, s);
                float sr = c, si = s;
                if (mode43) {
                    sr = c * sycl::fma(4.0f, c * c, -3.0f);
                    si = s * sycl::fma(-4.0f, s * s, 3.0f);
                }
                const float u_f = u_df.hi + u_df.lo;
                d_first[m] = first;
                d_u[m]     = u_f;
                d_shift[m] = ComplexT(sr, si);
#endif
#endif
            });
    });
}

// Congruence-partitioned zero-atomic shared-memory spreading kernel:
//   - Accumulates `b_per_wg` packed transforms directly in on-chip shared memory
//     (`l_re`, `l_im`) and writes batch-major output directly to `d_fft_buf`
//     (zero global RMWs, zero transpose pass, zero atomics).
//   - For nfft > SYCL_NUFFT_SMEM_CAP (e.g. nfft = 4096 in FP64/DF), tiles the frequency grid
//     into `num_wins = nfft / SYCL_NUFFT_SMEM_CAP` shared-memory windows of 2048 bins (32 KiB).
//   - Parallelizes Phase-1 piecewise minimax evaluation across BOTH points `pt_k` and half-panels
//     `panel_j` across all 64 threads (2 warps) in the work-group.
template <int W>
inline void SYCL_FFT_SUFFIX(launch_spread_smem)(
    sycl::queue& q,
    const int* d_first,
    const SYCL_NUFFT_SCALAR_T* d_u,
    const SYCL_NUFFT_COMPLEX_T* d_shift,
    const SYCL_NUFFT_COMPLEX_T* d_y,
    SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int M,
    int nfft,
    size_t total_M_stride,
    size_t chunk_bs,
    PswfMode mode) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    using RealT    = SYCL_NUFFT_REAL_T;

    constexpr int kWgSize  = 32; // 1 warp per work-group -> single-barrier lockstep SIMT
    constexpr int kTileM   = 32;
    constexpr int kLanes   = SYCL_NUFFT_LANES;
    constexpr int kSmemCap = SYCL_NUFFT_SMEM_CAP;

#if defined(SYCL_FFT_DOUBLE) || defined(SYCL_FFT_DF)
    const int win_elems = std::min(nfft, kSmemCap);
    const int num_wins  = nfft / win_elems;
#else
    const int win_elems = nfft;
    constexpr int num_wins = 1;
#endif
    const int b_per_wg  = (num_wins > 1) ? 1 : std::max(1, std::min(4, kSmemCap / nfft));
    const int t_j       = kWgSize / b_per_wg; // 8, 16, or 32
    const int log2_tj   = (t_j == 8) ? 3 : ((t_j == 16) ? 4 : 5);
    const int tj_mask   = t_j - 1;
    const int smem_grid_elems = b_per_wg * win_elems;
    const int nfft_mask = nfft - 1;

    const size_t bs_packed = (chunk_bs + kLanes - 1) / kLanes;
    const size_t num_groups = (bs_packed + static_cast<size_t>(b_per_wg) - 1) / static_cast<size_t>(b_per_wg);

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
        sycl::local_accessor<RealT, 1>   l_re(sycl::range<1>(smem_grid_elems), cgh);
        sycl::local_accessor<RealT, 1>   l_im(sycl::range<1>(smem_grid_elems), cgh);
        sycl::local_accessor<int, 1>     l_first(sycl::range<1>(kTileM), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_re(sycl::range<1>(4 * kTileM), cgh);
        sycl::local_accessor<RealT, 1>   l_sy_im(sycl::range<1>(4 * kTileM), cgh);
        sycl::local_accessor<ScalarT, 1> l_w(sycl::range<1>(kTileM * W), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const auto grp = item.get_group();
                const int tid  = static_cast<int>(item.get_local_linear_id());
                const int lane_b = tid >> log2_tj;
                const int lane_t = tid & tj_mask;
                const size_t bp_base = item.get_group_linear_id() * static_cast<size_t>(b_per_wg);
                const size_t my_bp   = bp_base + static_cast<size_t>(lane_b);
                const int grid_base  = lane_b * win_elems;

                for (int win = 0; win < num_wins; ++win) {
                    const int win_lo = win * win_elems;

                    // Zero local-memory subgrid
                    for (int idx = tid; idx < smem_grid_elems; idx += kWgSize) {
                        l_re[idx] = SYCL_FFT_ZERO;
                        l_im[idx] = SYCL_FFT_ZERO;
                    }

                    for (int m_base = 0; m_base < M; m_base += kTileM) {
                        const int tile_m = sycl::min(kTileM, M - m_base);
                        if (tid < tile_m) {
                            const int m_idx = m_base + tid;
                            l_first[tid] = d_first[m_idx];
                            const ScalarT u   = d_u[m_idx];
                            const ComplexT sh = d_shift[m_idx];

                            ScalarT w_reg[W];
#if defined(SYCL_FFT_HALF)
                            eval_pswf_spread_weights_f16<W>(u, tbl, w_reg);
                            const RealT sh_v = SYCL_FFT_BIT_CAST(RealT, sh);
                            const ScalarT sr_h = sh_v[0];
                            const ScalarT si_h = sh_v[1];
                            const RealT sr_v = SYCL_FFT_F16X2_SPLAT_H(sr_h);
                            const RealT si_v = SYCL_FFT_F16X2_SPLAT_H(si_h);
                            const RealT neg_si_v = SYCL_FFT_BIT_CAST(
                                RealT, SYCL_FFT_BIT_CAST(uint32_t, si_v) ^ 0x80008000u);
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

                            for (int b_sub = 0; b_sub < b_per_wg; ++b_sub) {
                                const size_t bp = bp_base + static_cast<size_t>(b_sub);
                                if (bp < bs_packed) {
                                    RealT sy_re, sy_im;
#if defined(SYCL_FFT_HALF)
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
                                    l_sy_re[b_sub * kTileM + tid] = sy_re;
                                    l_sy_im[b_sub * kTileM + tid] = sy_im;
                                }
                            }
                        }

                        sycl::group_barrier(grp);

                        if (my_bp < bs_packed) {
                            const int sy_base = lane_b * kTileM;
                            for (int k = 0; k < tile_m; ++k) {
                                const int first = l_first[k];
                                const int j0 = (lane_t - first) & tj_mask;
                                if (j0 < W) {
                                    const RealT sy_re = l_sy_re[sy_base + k];
                                    const RealT sy_im = l_sy_im[sy_base + k];
                                    const int bin0 = ((first + j0) & nfft_mask) - win_lo;
                                    if (num_wins == 1 ||
                                        static_cast<unsigned>(bin0) < static_cast<unsigned>(win_elems)) {
                                        const int idx0 = grid_base + bin0;
                                        const ScalarT w0_s = l_w[k * W + j0];
#if defined(SYCL_FFT_HALF)
                                        const RealT w0_v = SYCL_FFT_F16X2_SPLAT_H(w0_s);
#else
                                        const RealT w0_v = w0_s;
#endif
                                        l_re[idx0] = SYCL_NUFFT_FMA(sy_re, w0_v, l_re[idx0]);
                                        l_im[idx0] = SYCL_NUFFT_FMA(sy_im, w0_v, l_im[idx0]);
                                    }

                                    if constexpr (W > 8) {
                                        const int j1 = j0 + t_j;
                                        if (j1 < W) {
                                            const int bin1 = ((first + j1) & nfft_mask) - win_lo;
                                            if (num_wins == 1 ||
                                                static_cast<unsigned>(bin1) < static_cast<unsigned>(win_elems)) {
                                                const int idx1 = grid_base + bin1;
                                                const ScalarT w1_s = l_w[k * W + j1];
#if defined(SYCL_FFT_HALF)
                                                const RealT w1_v = SYCL_FFT_F16X2_SPLAT_H(w1_s);
#else
                                                const RealT w1_v = w1_s;
#endif
                                                l_re[idx1] = SYCL_NUFFT_FMA(sy_re, w1_v, l_re[idx1]);
                                                l_im[idx1] = SYCL_NUFFT_FMA(sy_im, w1_v, l_im[idx1]);
                                            }
                                        }
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
                        const int g_off = b_sub * win_elems;
#if defined(SYCL_FFT_HALF)
                        const size_t b0_out = 2 * bp_out;
                        const size_t b1_out = 2 * bp_out + 1;
                        ComplexT* dst0 = d_fft_buf + b0_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                        ComplexT* dst1 = d_fft_buf + b1_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                        const bool v0 = (b0_out < chunk_bs);
                        const bool v1 = (b1_out < chunk_bs);
                        for (int n = tid; n < win_elems; n += kWgSize) {
                            sycl_fft::f16x2_store_pair(dst0, dst1, n, v0, v1, l_re[g_off + n], l_im[g_off + n]);
                        }
#elif defined(SYCL_FFT_DF)
                        ComplexT* dst = d_fft_buf + bp_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                        for (int n = tid; n < win_elems; n += kWgSize) {
                            dst[n] = ComplexT{l_re[g_off + n], l_im[g_off + n]};
                        }
#else
                        ComplexT* dst = d_fft_buf + bp_out * static_cast<size_t>(nfft) + static_cast<size_t>(win_lo);
                        for (int n = tid; n < win_elems; n += kWgSize) {
                            dst[n] = ComplexT(l_re[g_off + n], l_im[g_off + n]);
                        }
#endif
                    }
                    if (num_wins > 1) {
                        sycl::group_barrier(grp);
                    }
                }
            });
    });
}

// Multi-warp collaborative zero-atomic batch-inner spreading kernel templated on width W:
//   - 128 threads per work-group (4 warps) own 128 packed batch items
//     (256 simultaneous transforms in FP16, 128 in FP32/FP64/DF).
//   - In tiles of 32 points:
//       1. The first 32 threads evaluate the W piecewise minimax PSWF weights once for point
//          `m_base + tid` and stage `l_first[32]`, `l_shift[32]`, and `l_w[32][W]` in shared
//          memory (128x/256x polynomial FLOP reduction, <= 5.2 KiB shared memory total).
//       2. All 128 threads (4 warps) stream through `k = 0..tile_m-1`, rotate `y * l_shift[k]`
//          in registers, and accumulate `sy * w_j` onto `d_grid_T[idx * packed_stride + my_bp]`
//          with 100% warp-coalesced memory transactions and zero atomics.
template <int W>
inline void SYCL_FFT_SUFFIX(launch_spread_collab)(
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
    PswfMode mode) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    (void)sizeof(ScalarT);
    using RealT    = SYCL_NUFFT_REAL_T;
    (void)sizeof(ComplexT);

    constexpr int kTileM  = 32;
    constexpr int kWgSize = 128; // 4 warps per work-group -> high SM occupancy
    constexpr int kLanes  = SYCL_NUFFT_LANES;
    const size_t bs_packed = (chunk_bs + kLanes - 1) / kLanes;
    const size_t num_b_groups = (bs_packed + kWgSize - 1) / kWgSize;
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
        sycl::local_accessor<int, 1>      l_first(sycl::range<1>(kTileM), cgh);
        sycl::local_accessor<ComplexT, 1> l_shift(sycl::range<1>(kTileM), cgh);
        sycl::local_accessor<ScalarT, 1>  l_w(sycl::range<1>(kTileM * W), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_b_groups * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const auto grp = item.get_group();
                const int tid = static_cast<int>(item.get_local_linear_id());
                const size_t b_group = item.get_group_linear_id();
                const size_t my_bp   = b_group * static_cast<size_t>(kWgSize) + static_cast<size_t>(tid);

                for (int m_base = 0; m_base < M; m_base += kTileM) {
                    const int tile_m = sycl::min(kTileM, M - m_base);

                    if (tid < tile_m) {
                        const int m_idx = m_base + tid;
                        l_first[tid] = d_first[m_idx];
                        l_shift[tid] = d_shift[m_idx];
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
                        for (int k = 0; k < tile_m; ++k) {
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

// 16x16 shared-memory tiled transpose from batch-inner [nfft][packed_stride]
// to batch-major [chunk_bs][nfft], simultaneously zeroing d_grid_T in L2 cache.
inline void SYCL_FFT_SUFFIX(launch_transpose_and_zero_grid)(
    sycl::queue& q,
    SYCL_NUFFT_REAL_T* d_grid_T_re,
    SYCL_NUFFT_REAL_T* d_grid_T_im,
    SYCL_NUFFT_COMPLEX_T* d_fft_buf,
    int nfft,
    size_t chunk_bs,
    size_t packed_stride) {
    using RealT    = SYCL_NUFFT_REAL_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    (void)sizeof(ComplexT);

    constexpr int kTile = 16;
    constexpr int kLanes = SYCL_NUFFT_LANES;
    const size_t bs_packed = (chunk_bs + kLanes - 1) / kLanes;
    const size_t groups_n = (static_cast<size_t>(nfft) + kTile - 1) / kTile;
    const size_t groups_b = (bs_packed + kTile - 1) / kTile;

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<RealT, 1> l_re(sycl::range<1>(kTile * (kTile + 1)), cgh);
        sycl::local_accessor<RealT, 1> l_im(sycl::range<1>(kTile * (kTile + 1)), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n * kTile, groups_b * kTile),
                              sycl::range<2>(kTile, kTile)),
            [=](sycl::nd_item<2> item) {
                const auto grp = item.get_group();
                const int ly = static_cast<int>(item.get_local_id(0));
                const int lx = static_cast<int>(item.get_local_id(1));
                const size_t n_base = item.get_group(0) * static_cast<size_t>(kTile);
                const size_t b_base = item.get_group(1) * static_cast<size_t>(kTile);

                // Coalesced read from d_grid_T[n_in, bp_in] + immediate zeroing in L2
                const size_t n_in  = n_base + static_cast<size_t>(ly);
                const size_t bp_in = b_base + static_cast<size_t>(lx);
                RealT r_val = SYCL_FFT_ZERO;
                RealT i_val = SYCL_FFT_ZERO;
                if (n_in < static_cast<size_t>(nfft) && bp_in < bs_packed) {
                    const size_t g_off = n_in * packed_stride + bp_in;
                    r_val = d_grid_T_re[g_off];
                    i_val = d_grid_T_im[g_off];
                    d_grid_T_re[g_off] = SYCL_FFT_ZERO;
                    d_grid_T_im[g_off] = SYCL_FFT_ZERO;
                }
                l_re[lx * (kTile + 1) + ly] = r_val;
                l_im[lx * (kTile + 1) + ly] = i_val;

                sycl::group_barrier(grp);

                // Coalesced write to d_fft_buf[b_out, n_out] (lx is now the contiguous n_out index)
                const size_t bp_out = b_base + static_cast<size_t>(ly);
                const size_t n_out  = n_base + static_cast<size_t>(lx);
                if (bp_out < bs_packed && n_out < static_cast<size_t>(nfft)) {
                    const RealT tr = l_re[ly * (kTile + 1) + lx];
                    const RealT ti = l_im[ly * (kTile + 1) + lx];
#if defined(SYCL_FFT_HALF)
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
#elif defined(SYCL_FFT_DF)
                    d_fft_buf[bp_out * static_cast<size_t>(nfft) + n_out] = ComplexT{tr, ti};
#else
                    d_fft_buf[bp_out * static_cast<size_t>(nfft) + n_out] = ComplexT(tr, ti);
#endif
                }
            });
    });
}

// FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution pass templated on width W:
//   - Evaluates deconv(k') = inv_prefac / phi(eta_{k'}) via a single piecewise panel Horner
//     evaluation in registers per output bin i in [0, N_out), and applies it across a tile
//     of batch items (zero global LUT reads, zero Gauss-Legendre quadrature).
//   - Supports both NufftOutOrder::SlotBitReversed (primary fast path, raw DIF slots)
//     and NufftOutOrder::Natural (verification path).
template <int W>
inline void SYCL_FFT_SUFFIX(launch_deconv_emit)(
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
    NufftOutOrder order) {
    using ScalarT  = SYCL_NUFFT_SCALAR_T;
    using ComplexT = SYCL_NUFFT_COMPLEX_T;
    (void)sizeof(ScalarT);
    using RealT    = SYCL_NUFFT_REAL_T;
    (void)sizeof(ComplexT);

    constexpr int kWgI = 64;
    constexpr int kBatchTile = 8;
    constexpr int kLanes = SYCL_NUFFT_LANES;
    const size_t bs_packed = (chunk_bs + kLanes - 1) / kLanes;
    const size_t groups_i = (static_cast<size_t>(nout) + kWgI - 1) / kWgI;
    const size_t groups_b = (bs_packed + kBatchTile - 1) / kBatchTile;

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
            sycl::nd_range<2>(sycl::range<2>(groups_b, groups_i * kWgI),
                              sycl::range<2>(1, kWgI)),
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

                const size_t bp_start = b_tile_idx * static_cast<size_t>(kBatchTile);
                const size_t bp_end   = sycl::min(bs_packed, bp_start + static_cast<size_t>(kBatchTile));

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

} // namespace detail

SYCL_FFT_SUFFIX(Nufft1Plan)::SYCL_FFT_SUFFIX(Nufft1Plan)(
    sycl::queue& q,
    int M,
    int N,
    int w,
    PswfMode mode,
    size_t max_bs,
    const sycl_fft::FftLaunchConfig& fft_cfg)
    : q_(q),
      M_(M),
      N_(N),
      w_(w),
      mode_(mode),
      nfft_(next_pow2_int(2 * N)),
      log_nfft_(int_log2_pow2(next_pow2_int(2 * N))),
      nout_(nufft1_nout(N, mode)),
      output_shift_(nufft1_output_shift(N, mode)),
      max_bs_(std::max<size_t>(1, max_bs)),
      chunk_bs_(1),
      packed_stride_(32),
      fft_plan_(q, int_log2_pow2(next_pow2_int(2 * N)), std::max<size_t>(1, max_bs), fft_cfg) {
    if (M_ <= 0 || N_ <= 0) {
        throw std::invalid_argument("Nufft1Plan: M and N must be positive");
    }
    if (mode_ == PswfMode::PSWF43 && (N_ % 4) != 0) {
        throw std::invalid_argument("Nufft1Plan: PSWF43 requires N divisible by 4");
    }
    if (!detail::SYCL_FFT_SUFFIX(is_valid_pswf_width)(w_)) {
        throw std::invalid_argument("Nufft1Plan: unsupported kernel width w for this precision");
    }

    // Size L2-resident chunk_bs so that d_grid_T and d_fft_buf stay inside L2 cache
    // Note: fft_plan_.chunk_bs() is 0 for single-pass FFTs (log_nfft <= 13).
    const size_t fft_chunk = fft_plan_.chunk_bs();
    chunk_bs_ = (fft_chunk > 0) ? std::min(max_bs_, fft_chunk) : max_bs_;
    constexpr size_t kMaxGridChunkBytes = 16ULL * 1024ULL * 1024ULL; // 16 MiB
    const size_t max_bs_by_grid = std::max<size_t>(
        64, kMaxGridChunkBytes / (static_cast<size_t>(nfft_) * sizeof(ComplexT)));
    chunk_bs_ = std::max<size_t>(1, std::min(chunk_bs_, max_bs_by_grid));

    constexpr size_t kLanes = SYCL_NUFFT_LANES;
    const size_t max_packed = (chunk_bs_ + kLanes - 1) / kLanes;
    packed_stride_ = ((max_packed + 31) / 32) * 32;

    d_first_     = sycl::malloc_device<int>(static_cast<size_t>(M_), q_);
    d_u_         = sycl::malloc_device<ScalarT>(static_cast<size_t>(M_), q_);
    d_shift_     = sycl::malloc_device<ComplexT>(static_cast<size_t>(M_), q_);
    d_grid_T_re_ = sycl::malloc_device<RealT>(static_cast<size_t>(nfft_) * packed_stride_, q_);
    d_grid_T_im_ = sycl::malloc_device<RealT>(static_cast<size_t>(nfft_) * packed_stride_, q_);
    d_fft_buf_   = sycl::malloc_device<ComplexT>(chunk_bs_ * static_cast<size_t>(nfft_), q_);

    if (!d_first_ || !d_u_ || !d_shift_ || !d_grid_T_re_ || !d_grid_T_im_ || !d_fft_buf_) {
        if (d_first_) sycl::free(d_first_, q_);
        if (d_u_) sycl::free(d_u_, q_);
        if (d_shift_) sycl::free(d_shift_, q_);
        if (d_grid_T_re_) sycl::free(d_grid_T_re_, q_);
        if (d_grid_T_im_) sycl::free(d_grid_T_im_, q_);
        if (d_fft_buf_) sycl::free(d_fft_buf_, q_);
        throw std::runtime_error("Nufft1Plan: device memory allocation failed");
    }

    // Zero d_grid_T once at construction; transpose_and_zero_grid resets it after each chunk.
    q_.memset(d_grid_T_re_, 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
    q_.memset(d_grid_T_im_, 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
    q_.wait();
}

SYCL_FFT_SUFFIX(Nufft1Plan)::~SYCL_FFT_SUFFIX(Nufft1Plan)() {
    q_.wait();
    if (d_first_) sycl::free(d_first_, q_);
    if (d_u_) sycl::free(d_u_, q_);
    if (d_shift_) sycl::free(d_shift_, q_);
    if (d_grid_T_re_) sycl::free(d_grid_T_re_, q_);
    if (d_grid_T_im_) sycl::free(d_grid_T_im_, q_);
    if (d_fft_buf_) sycl::free(d_fft_buf_, q_);
}

void SYCL_FFT_SUFFIX(Nufft1Plan)::set_points(
    const XCoordT* d_x, XCoordT df, int freq_factor) {
    if (freq_factor <= 0) {
        throw std::invalid_argument("Nufft1Plan::set_points: freq_factor must be >= 1");
    }
#if defined(SYCL_FFT_DOUBLE)
    const XCoordT df_scaled = df * static_cast<double>(freq_factor);
#else
    const XCoordT df_scaled = sycl_fft::df_mul_f(df, static_cast<float>(freq_factor));
#endif
    detail::SYCL_FFT_SUFFIX(launch_precompute_points)(
        q_,
        d_x,
        df_scaled,
        M_,
        nfft_,
        w_,
        output_shift_,
        mode_ == PswfMode::PSWF43,
        d_first_,
        d_u_,
        d_shift_);
}

void SYCL_FFT_SUFFIX(Nufft1Plan)::execute(
    const ComplexT* d_y,
    ComplexT* d_out,
    size_t bs,
    NufftOutOrder order) {
    if (bs == 0) return;

    const size_t total_M_stride    = static_cast<size_t>(M_);
    const size_t total_nout_stride = static_cast<size_t>(nout_);

    for (size_t b_off = 0; b_off < bs; b_off += chunk_bs_) {
        const size_t cur_bs = std::min(chunk_bs_, bs - b_off);
        const ComplexT* cur_y = d_y + b_off * total_M_stride;
        ComplexT* cur_out     = d_out + b_off * total_nout_stride;

        const bool use_smem_spread = (nfft_ <= 2 * SYCL_NUFFT_SMEM_CAP);

        // 1. Piecewise minimax PSWF spreading (direct shared-memory path when nfft <= SYCL_NUFFT_SMEM_CAP)
        if (use_smem_spread) {
#if defined(SYCL_FFT_HALF)
            if (w_ == 4) {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<4>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            } else {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<5>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            }
#elif defined(SYCL_FFT_DOUBLE)
            if (w_ == 16) {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<16>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            } else if (w_ == 17) {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<17>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            } else {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<18>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            }
#elif defined(SYCL_FFT_DF)
            if (w_ == 14) {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<14>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            } else if (w_ == 15) {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<15>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            } else {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<16>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            }
#else
            if (w_ == 8) {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<8>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            } else {
                detail::SYCL_FFT_SUFFIX(launch_spread_smem)<9>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_);
            }
#endif
        } else {
#if defined(SYCL_FFT_HALF)
            if (w_ == 4) {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<4>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            } else {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<5>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            }
#elif defined(SYCL_FFT_DOUBLE)
            if (w_ == 16) {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<16>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            } else if (w_ == 17) {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<17>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            } else {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<18>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            }
#elif defined(SYCL_FFT_DF)
            if (w_ == 14) {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<14>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            } else if (w_ == 15) {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<15>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            } else {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<16>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            }
#else
            if (w_ == 8) {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<8>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            } else {
                detail::SYCL_FFT_SUFFIX(launch_spread_collab)<9>(
                    q_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                    M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_);
            }
#endif

            // 2. Transpose batch-inner grid [nfft][packed_stride] -> batch-major [cur_bs][nfft] + zero in L2
            detail::SYCL_FFT_SUFFIX(launch_transpose_and_zero_grid)(
                q_, d_grid_T_re_, d_grid_T_im_, d_fft_buf_, nfft_, cur_bs, packed_stride_);
        }

        // 3. Raw DIF Radix-2 FFT (Sign = +1, NO bit-reversal pass!)
        fft_plan_.execute_raw(d_fft_buf_, d_fft_buf_, cur_bs);

        // 4. FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution pass
#if defined(SYCL_FFT_HALF)
        if (w_ == 4) {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<4>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        } else {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<5>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        }
#elif defined(SYCL_FFT_DOUBLE)
        if (w_ == 16) {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<16>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        } else if (w_ == 17) {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<17>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        } else {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<18>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        }
#elif defined(SYCL_FFT_DF)
        if (w_ == 14) {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<14>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        } else if (w_ == 15) {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<15>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        } else {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<16>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        }
#else
        if (w_ == 8) {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<8>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        } else {
            detail::SYCL_FFT_SUFFIX(launch_deconv_emit)<9>(
                q_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order);
        }
#endif
    }
}

#endif // !defined(SYCL_FFT_USE_EXTERN_TEMPLATES) || defined(SYCL_NUFFT_IMPLEMENTATION_TU)

#undef SYCL_NUFFT_COMPLEX_T
#undef SYCL_NUFFT_REAL_T
#undef SYCL_NUFFT_SCALAR_T
#undef SYCL_NUFFT_XCOORD_T
#undef SYCL_NUFFT_LANES
#undef SYCL_NUFFT_SMEM_CAP
#undef SYCL_NUFFT_FMA

} // namespace sycl_nufft
