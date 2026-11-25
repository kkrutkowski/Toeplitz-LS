// Decoupled FP64 NuFFT-1 Spreading Module (separate HCF)
#define SYCL_NUFFT_KERNEL_ONLY 1
#include "sycl_nufft1.hpp"
#define SYCL_FFT_DOUBLE
#include "sycl_fft_precision.hpp"
#include "sycl_nufft1_spread_impl.hpp"
#undef SYCL_FFT_DOUBLE
