// Explicit instantiations for FP64 DifFftPlan (separate HCF module)
#define SYCL_FFT_ONLY_DOUBLE
#define SYCL_FFT_EXTERN_BITREV
#include "sycl_fft_dif.hpp"

namespace sycl_fft {
#if SYCL_FFT_ENABLE_NEG_SIGN
template class DifFftPlan<-1>;
#endif
#if SYCL_FFT_ENABLE_POS_SIGN
template class DifFftPlan<+1>;
#endif
} // namespace sycl_fft
