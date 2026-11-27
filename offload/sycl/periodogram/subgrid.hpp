#pragma once

// On-the-fly GPU subgrid phase-modulation kernel declarations.
// Computes d_src[b][m] = w[m] * exp(+2*pi*i * q * (f0 + (b_start + b) * nout * df) * x[m])
// and optionally (when d_yw != nullptr) d_src[b_count + b][m] = yw[m] * same phase,
// reusing the exact same cos/sin evaluation and eliminating CPU twiddle-ladder LUTs.

#include <sycl/sycl.hpp>

#include "common/math.hpp"

namespace sycl_periodogram {
namespace detail {

void launch_build_subgrid_sourcesh(sycl::queue& q, const sycl_fft::df_t* d_x, const sycl_fft::f16_scalar_t* d_w, const sycl_fft::f16_scalar_t* d_yw,
                                   sycl_fft::Complexh* d_src, int M, int b_start, int b_count, double q_f0, double q_block_df, size_t wg_size);

void launch_build_subgrid_sourcesf(sycl::queue& q, const sycl_fft::df_t* d_x, const float* d_w, const float* d_yw, sycl_fft::Complexf* d_src, int M,
                                   int b_start, int b_count, double q_f0, double q_block_df, size_t wg_size);

void launch_build_subgrid_sources(sycl::queue& q, const double* d_x, const double* d_w, const double* d_yw, sycl_fft::Complex* d_src, int M, int b_start,
                                  int b_count, double q_f0, double q_block_df, size_t wg_size);

void launch_build_subgrid_sourcesdf(sycl::queue& q, const sycl_fft::df_t* d_x, const sycl_fft::df_t* d_w, const sycl_fft::df_t* d_yw,
                                    sycl_fft::Complexdf* d_src, int M, int b_start, int b_count, double q_f0, double q_block_df, size_t wg_size);

}  // namespace detail
}  // namespace sycl_periodogram
