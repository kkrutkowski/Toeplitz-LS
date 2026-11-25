// Explicit instantiations for DIF Radix-2 FFT kernels & DifFftPlan*.
// Compiled multiple times with -D flags into isolated HCF object modules:
//   Precision (defaults to SYCL_FFT_DF when no half/float/double is selected):
//     -DSYCL_FFT_HALF | -DSYCL_FFT_FLOAT | -DSYCL_FFT_DOUBLE | -DSYCL_FFT_DF
//   Size split (optional; omit to instantiate both small & large kernels + plan):
//     -DSYCL_FFT_INST_SMALL_ONLY | -DSYCL_FFT_INST_LARGE_ONLY
#if !defined(SYCL_FFT_HALF) && !defined(SYCL_FFT_FLOAT) && \
    !defined(SYCL_FFT_DOUBLE) && !defined(SYCL_FFT_DF)
#  define SYCL_FFT_DF 1
#endif

#if defined(SYCL_FFT_HALF)
#  define SYCL_FFT_ONLY_HALF 1
#  undef SYCL_FFT_HALF
#elif defined(SYCL_FFT_FLOAT)
#  define SYCL_FFT_ONLY_FLOAT 1
#elif defined(SYCL_FFT_DOUBLE)
#  define SYCL_FFT_ONLY_DOUBLE 1
#  undef SYCL_FFT_DOUBLE
#else
#  define SYCL_FFT_ONLY_DF 1
#  undef SYCL_FFT_DF
#endif

#define SYCL_FFT_EXTERN_BITREV 1
#define SYCL_FFT_EXTERN_DEVICE 1
#include "fft/dif.hpp"

#if !defined(SYCL_FFT_INST_SMALL_ONLY)
namespace sycl_fft {
#if defined(SYCL_FFT_ONLY_HALF)
#  if SYCL_FFT_ENABLE_NEG_SIGN
template class DifFftPlanh<-1>;
#  endif
#  if SYCL_FFT_ENABLE_POS_SIGN
template class DifFftPlanh<+1>;
#  endif
#elif defined(SYCL_FFT_ONLY_FLOAT)
#  if SYCL_FFT_ENABLE_NEG_SIGN
template class DifFftPlanf<-1>;
#  endif
#  if SYCL_FFT_ENABLE_POS_SIGN
template class DifFftPlanf<+1>;
#  endif
#elif defined(SYCL_FFT_ONLY_DOUBLE)
#  if SYCL_FFT_ENABLE_NEG_SIGN
template class DifFftPlan<-1>;
#  endif
#  if SYCL_FFT_ENABLE_POS_SIGN
template class DifFftPlan<+1>;
#  endif
#else
#  if SYCL_FFT_ENABLE_NEG_SIGN
template class DifFftPlandf<-1>;
#  endif
#  if SYCL_FFT_ENABLE_POS_SIGN
template class DifFftPlandf<+1>;
#  endif
#endif
} // namespace sycl_fft
#endif
