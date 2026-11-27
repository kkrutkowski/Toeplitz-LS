// Decoupled SYCL On-the-Fly Subgrid Phase-Modulation Module.
// Compiled 4x via -D flags (isolated HCF per precision):
//   -DSYCL_FFT_HALF | -DSYCL_FFT_FLOAT | -DSYCL_FFT_DOUBLE | -DSYCL_FFT_DF
//
// Uses short strip-recurrence (K_b = 8 blocks per work-item when b_count >= 4)
// so each thread loads d_x[m], d_w[m], d_yw[m] ONCE per strip and amortizes
// expensive sincos2pi evaluations 4x across consecutive subgrids while keeping
// drift strictly below precision epsilon.

#if !defined(SYCL_FFT_HALF) && !defined(SYCL_FFT_FLOAT) && !defined(SYCL_FFT_DOUBLE) && !defined(SYCL_FFT_DF)
#    define SYCL_FFT_DF 1
#endif

#include <algorithm>

#include "common/precision.hpp"
#include "periodogram/subgrid.hpp"

namespace sycl_periodogram {
namespace detail {

using namespace sycl_fft;

#if defined(SYCL_FFT_HALF)
using SubgridXCoordT = sycl_fft::df_t;
using SubgridScalarT = sycl_fft::f16_scalar_t;
using SubgridComplexT = sycl_fft::Complexh;
#elif defined(SYCL_FFT_DOUBLE)
using SubgridXCoordT = double;
using SubgridScalarT = double;
using SubgridComplexT = sycl_fft::Complex;
#elif defined(SYCL_FFT_DF)
using SubgridXCoordT = sycl_fft::df_t;
using SubgridScalarT = sycl_fft::df_t;
using SubgridComplexT = sycl_fft::Complexdf;
#else
using SubgridXCoordT = sycl_fft::df_t;
using SubgridScalarT = float;
using SubgridComplexT = sycl_fft::Complexf;
#endif

void SYCL_FFT_SUFFIX(launch_build_subgrid_sources)(sycl::queue& q, const SubgridXCoordT* d_x, const SubgridScalarT* d_w, const SubgridScalarT* d_yw,
                                                   SubgridComplexT* d_src, int M, int b_start, int b_count, double q_f0, double q_block_df, size_t wg_size) {
    if (M <= 0 || b_count <= 0) return;

    const size_t wg_m = std::max<size_t>(1, std::min<size_t>(wg_size, 256));
    const size_t groups_m = (static_cast<size_t>(M) + wg_m - 1) / wg_m;
    const bool has_yw = (d_yw != nullptr);

    // Strip up to 8 consecutive blocks per work-item when b_count >= 4
    // (2 sincos2pi calls + 7 complex multiplies instead of 8 sincos2pi calls).
    const int strip_b = (b_count >= 4) ? 8 : 1;
    const size_t groups_b = static_cast<size_t>((b_count + strip_b - 1) / strip_b);

#if defined(SYCL_FFT_DOUBLE)
    const double f0_base = q_f0 + static_cast<double>(b_start) * q_block_df;
    const double f_step = q_block_df;
#else
    const df_t f0_base_df = df_from_double(q_f0 + static_cast<double>(b_start) * q_block_df);
    const df_t f_step_df = df_from_double(q_block_df);
#endif

    const size_t M_sz = static_cast<size_t>(M);
    const size_t yw_offset = static_cast<size_t>(b_count) * M_sz;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(groups_b, groups_m * wg_m), sycl::range<2>(1, wg_m)), [=](sycl::nd_item<2> item) {
            const int b0 = static_cast<int>(item.get_global_id(0)) * strip_b;
            const int m = static_cast<int>(item.get_global_id(1));
            if (m >= M || b0 >= b_count) return;
            const int b_end = (b0 + strip_b < b_count) ? (b0 + strip_b) : b_count;

            const SubgridXCoordT xm = d_x[m];
#if defined(SYCL_FFT_DOUBLE)
            const double wm = d_w[m];
            const double ywm = has_yw ? d_yw[m] : 0.0;
            const double fb0 = sycl::fma(static_cast<double>(b0), f_step, f0_base);
            double c = 1.0, s = 0.0;
            sincos2pi_branchless<+1>(xm * fb0, c, s);

            size_t out_idx = static_cast<size_t>(b0) * M_sz + static_cast<size_t>(m);
            d_src[out_idx] = Complex(wm * c, wm * s);
            if (has_yw) {
                d_src[yw_offset + out_idx] = Complex(ywm * c, ywm * s);
            }

            if (b0 + 1 < b_end) {
                double dc = 1.0, ds = 0.0;
                sincos2pi_branchless<+1>(xm * f_step, dc, ds);
                for (int b = b0 + 1; b < b_end; ++b) {
                    const double nc = sycl::fma(c, dc, -s * ds);
                    const double ns = sycl::fma(c, ds, s * dc);
                    c = nc;
                    s = ns;
                    out_idx += M_sz;
                    d_src[out_idx] = Complex(wm * c, wm * s);
                    if (has_yw) {
                        d_src[yw_offset + out_idx] = Complex(ywm * c, ywm * s);
                    }
                }
            }
#elif defined(SYCL_FFT_DF)
                const df_t wm  = d_w[m];
                const df_t ywm = has_yw ? d_yw[m] : df_make(0.0f, 0.0f);
                const df_t fb0 = df_add(f0_base_df, df_mul_f(f_step_df, static_cast<float>(b0)));
                df_t c, s;
                sincos2pidf_branchless<+1>(df_mul(xm, fb0), c, s);

                size_t out_idx = static_cast<size_t>(b0) * M_sz + static_cast<size_t>(m);
                d_src[out_idx] = Complexdf{df_mul(wm, c), df_mul(wm, s)};
                if (has_yw) {
                    d_src[yw_offset + out_idx] = Complexdf{df_mul(ywm, c), df_mul(ywm, s)};
                }

                if (b0 + 1 < b_end) {
                    df_t dc, ds;
                    sincos2pidf_branchless<+1>(df_mul(xm, f_step_df), dc, ds);
                    for (int b = b0 + 1; b < b_end; ++b) {
                        df_t nc, ns;
                        df_cmul_fused(c, s, dc, ds, nc, ns);
                        c = nc;
                        s = ns;
                        out_idx += M_sz;
                        d_src[out_idx] = Complexdf{df_mul(wm, c), df_mul(wm, s)};
                        if (has_yw) {
                            d_src[yw_offset + out_idx] = Complexdf{df_mul(ywm, c), df_mul(ywm, s)};
                        }
                    }
                }
#elif defined(SYCL_FFT_HALF)
                const float wm  = static_cast<float>(d_w[m]);
                const float ywm = has_yw ? static_cast<float>(d_yw[m]) : 0.0f;
                const df_t fb0 = df_add(f0_base_df, df_mul_f(f_step_df, static_cast<float>(b0)));
                const df_t ph0 = df_mul(xm, fb0);
                const df_t fr0 = df_sub(ph0, df_floor(ph0));
                float c = 1.0f, s = 0.0f;
                sincos2pif_branchless<+1>(fr0.hi + fr0.lo, c, s);

                size_t out_idx = static_cast<size_t>(b0) * M_sz + static_cast<size_t>(m);
                d_src[out_idx] = SYCL_FFT_BIT_CAST(
                    Complexh,
                    f16x2_make(static_cast<f16_scalar_t>(wm * c),
                               static_cast<f16_scalar_t>(wm * s)));
                if (has_yw) {
                    d_src[yw_offset + out_idx] = SYCL_FFT_BIT_CAST(
                        Complexh,
                        f16x2_make(static_cast<f16_scalar_t>(ywm * c),
                                   static_cast<f16_scalar_t>(ywm * s)));
                }

                if (b0 + 1 < b_end) {
                    const df_t dph = df_mul(xm, f_step_df);
                    const df_t dfr = df_sub(dph, df_floor(dph));
                    float dc = 1.0f, ds = 0.0f;
                    sincos2pif_branchless<+1>(dfr.hi + dfr.lo, dc, ds);
                    for (int b = b0 + 1; b < b_end; ++b) {
                        const float nc = sycl::fma(c, dc, -s * ds);
                        const float ns = sycl::fma(c, ds,  s * dc);
                        c = nc;
                        s = ns;
                        out_idx += M_sz;
                        d_src[out_idx] = SYCL_FFT_BIT_CAST(
                            Complexh,
                            f16x2_make(static_cast<f16_scalar_t>(wm * c),
                                       static_cast<f16_scalar_t>(wm * s)));
                        if (has_yw) {
                            d_src[yw_offset + out_idx] = SYCL_FFT_BIT_CAST(
                                Complexh,
                                f16x2_make(static_cast<f16_scalar_t>(ywm * c),
                                           static_cast<f16_scalar_t>(ywm * s)));
                        }
                    }
                }
#else
                const float wm  = d_w[m];
                const float ywm = has_yw ? d_yw[m] : 0.0f;
                const df_t fb0 = df_add(f0_base_df, df_mul_f(f_step_df, static_cast<float>(b0)));
                const df_t ph0 = df_mul(xm, fb0);
                const df_t fr0 = df_sub(ph0, df_floor(ph0));
                float c = 1.0f, s = 0.0f;
                sincos2pif_branchless<+1>(fr0.hi + fr0.lo, c, s);

                size_t out_idx = static_cast<size_t>(b0) * M_sz + static_cast<size_t>(m);
                d_src[out_idx] = Complexf(wm * c, wm * s);
                if (has_yw) {
                    d_src[yw_offset + out_idx] = Complexf(ywm * c, ywm * s);
                }

                if (b0 + 1 < b_end) {
                    const df_t dph = df_mul(xm, f_step_df);
                    const df_t dfr = df_sub(dph, df_floor(dph));
                    float dc = 1.0f, ds = 0.0f;
                    sincos2pif_branchless<+1>(dfr.hi + dfr.lo, dc, ds);
                    for (int b = b0 + 1; b < b_end; ++b) {
                        const float nc = sycl::fma(c, dc, -s * ds);
                        const float ns = sycl::fma(c, ds,  s * dc);
                        c = nc;
                        s = ns;
                        out_idx += M_sz;
                        d_src[out_idx] = Complexf(wm * c, wm * s);
                        if (has_yw) {
                            d_src[yw_offset + out_idx] = Complexf(ywm * c, ywm * s);
                        }
                    }
                }
#endif
        });
    });
}

}  // namespace detail
}  // namespace sycl_periodogram
