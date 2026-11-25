// Explicit instantiations for FP32 Large Single-Pass (logN = 13..14), Pass-0 & DifFftPlanf
#define SYCL_FFT_ONLY_FLOAT
#define SYCL_FFT_EXTERN_BITREV
#define SYCL_FFT_EXTERN_DEVICE
#define SYCL_FFT_INST_LARGE_ONLY
#include "sycl_fft_dif.hpp"

namespace sycl_fft {
#if SYCL_FFT_ENABLE_NEG_SIGN
template class DifFftPlanf<-1>;
#endif
#if SYCL_FFT_ENABLE_POS_SIGN
template class DifFftPlanf<+1>;
#endif
} // namespace sycl_fft
