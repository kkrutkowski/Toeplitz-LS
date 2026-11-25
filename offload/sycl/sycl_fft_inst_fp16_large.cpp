// Explicit instantiations for FP16 (2xFP16) Large Single-Pass (logN = 13..14), Pass-0 & DifFftPlanh
#define SYCL_FFT_ONLY_HALF
#define SYCL_FFT_EXTERN_BITREV
#define SYCL_FFT_EXTERN_DEVICE
#define SYCL_FFT_INST_LARGE_ONLY
#include "sycl_fft_dif.hpp"

namespace sycl_fft {
#if SYCL_FFT_ENABLE_NEG_SIGN
template class DifFftPlanh<-1>;
#endif
#if SYCL_FFT_ENABLE_POS_SIGN
template class DifFftPlanh<+1>;
#endif
} // namespace sycl_fft
