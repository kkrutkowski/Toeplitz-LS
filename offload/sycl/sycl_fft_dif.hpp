#pragma once

// Triple-Precision SYCL DIF (Sande-Tukey) Radix-2 FFT Umbrella Header:
//   FP32:        f suffix   (Complexf,  DifFftPlanf,  sin2pif/cos2pif,   ...)
//   FP64:        no suffix  (Complex,   DifFftPlan,   sin2pi/cos2pi,     ...)
//   DoubleFloat: df suffix  (Complexdf, DifFftPlandf, sin2pidf/cos2pidf, ...)
//
// Optional selective-instantiation guards:
//   SYCL_FFT_ONLY_FLOAT / SYCL_FFT_ONLY_DOUBLE / SYCL_FFT_ONLY_DF

#include "sycl_fft_math.hpp"

#include <algorithm>
#include <new>
#include <stdexcept>

#if !defined(SYCL_FFT_ONLY_DOUBLE) && !defined(SYCL_FFT_ONLY_DF)
#undef SYCL_FFT_DOUBLE
#undef SYCL_FFT_DF
#include "sycl_fft_precision.hpp"
#include "sycl_fft_dif_impl.hpp"
#endif

#if !defined(SYCL_FFT_ONLY_FLOAT) && !defined(SYCL_FFT_ONLY_DF)
#undef SYCL_FFT_DF
#define SYCL_FFT_DOUBLE
#include "sycl_fft_precision.hpp"
#include "sycl_fft_dif_impl.hpp"
#undef SYCL_FFT_DOUBLE
#endif

#if !defined(SYCL_FFT_ONLY_FLOAT) && !defined(SYCL_FFT_ONLY_DOUBLE)
#undef SYCL_FFT_DOUBLE
#define SYCL_FFT_DF
#include "sycl_fft_precision.hpp"
#include "sycl_fft_dif_impl.hpp"
#undef SYCL_FFT_DF
#endif

// Cleanly undefine internal precision macros so caller translation units stay clean
#undef SYCL_FFT_SUFFIX
#undef SYCL_FFT_REAL
#undef SYCL_FFT_COMPLEX
#undef SYCL_FFT_SMEM_T
#undef SYCL_FFT_ANGLE_T
#undef SYCL_FFT_WIDE
#undef SYCL_FFT_STASH_8192
#undef SYCL_FFT_COMPUTE_HEAVY
#undef SYCL_FFT_ZERO
#undef SYCL_FFT_ONE
#undef SYCL_FFT_SQRT1_2
#undef SYCL_FFT_ADD
#undef SYCL_FFT_SUB
#undef SYCL_FFT_NEG
#undef SYCL_FFT_MUL
#undef SYCL_FFT_SCALE_HALF
#undef SYCL_FFT_CMUL
#undef SYCL_FFT_CSQR
#undef SYCL_FFT_CMUL_SQRT1_2
#undef SYCL_FFT_LOAD_COMPLEX
#undef SYCL_FFT_STORE_COMPLEX
#undef SYCL_FFT_TO_SMEM
#undef SYCL_FFT_FROM_SMEM
#undef SYCL_FFT_SHUFFLE_XOR
#undef SYCL_FFT_COND_NEG
#undef SYCL_FFT_TWIDDLE256_RE
#undef SYCL_FFT_TWIDDLE256_IM
#undef SYCL_FFT_TWIDDLE32_RE
#undef SYCL_FFT_TWIDDLE32_IM
#undef SYCL_FFT_SINCOS2PI_OCT1
#undef SYCL_FFT_SINCOS2PI_HALF
