// Decoupled FP16 (2xFP16) NuFFT-1 GL-Free Deconvolution Module (separate HCF)
#define SYCL_NUFFT_KERNEL_ONLY 1
#include "sycl_nufft1.hpp"
#define SYCL_FFT_HALF
#include "sycl_fft_precision.hpp"
#include "sycl_nufft1_deconv_impl.hpp"
#undef SYCL_FFT_HALF
