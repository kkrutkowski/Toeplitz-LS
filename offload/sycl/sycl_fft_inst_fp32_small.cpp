// Explicit instantiations for FP32 Small Single-Pass FFT kernels (logN = 10..12)
#define SYCL_FFT_ONLY_FLOAT
#define SYCL_FFT_EXTERN_BITREV
#define SYCL_FFT_EXTERN_DEVICE
#define SYCL_FFT_INST_SMALL_ONLY
#include "sycl_fft_dif.hpp"
