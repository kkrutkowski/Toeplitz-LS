#pragma once

// Quadruple-Precision SYCL DIF (Sande-Tukey) Radix-2 FFT Umbrella Header:
//   FP16 (2x):   h suffix   (Complexh,  DifFftPlanh,  sin2pih/cos2pih,   ...)
//   FP32:        f suffix   (Complexf,  DifFftPlanf,  sin2pif/cos2pif,   ...)
//   FP64:        no suffix  (Complex,   DifFftPlan,   sin2pi/cos2pi,     ...)
//   DoubleFloat: df suffix  (Complexdf, DifFftPlandf, sin2pidf/cos2pidf, ...)
//
// Optional selective-instantiation guards:
//   SYCL_FFT_ONLY_HALF / SYCL_FFT_ONLY_FLOAT / SYCL_FFT_ONLY_DOUBLE / SYCL_FFT_ONLY_DF
//
// Sign convention compile-time controls (default: +1 only):
//   SYCL_FFT_ENABLE_POS_SIGN (0 or 1, default 1)
//   SYCL_FFT_ENABLE_NEG_SIGN (0 or 1, default 0)
//   SYCL_FFT_DEFAULT_SIGN    (+1 or -1; defaults to +1 if POS enabled, else -1)
#if !defined(SYCL_FFT_ENABLE_POS_SIGN) && !defined(SYCL_FFT_ENABLE_NEG_SIGN)
#define SYCL_FFT_ENABLE_POS_SIGN 1
#define SYCL_FFT_ENABLE_NEG_SIGN 0
#else
#ifndef SYCL_FFT_ENABLE_POS_SIGN
#define SYCL_FFT_ENABLE_POS_SIGN 0
#endif
#ifndef SYCL_FFT_ENABLE_NEG_SIGN
#define SYCL_FFT_ENABLE_NEG_SIGN 0
#endif
#endif

#if !SYCL_FFT_ENABLE_POS_SIGN && !SYCL_FFT_ENABLE_NEG_SIGN
#error "At least one of SYCL_FFT_ENABLE_POS_SIGN or SYCL_FFT_ENABLE_NEG_SIGN must be 1"
#endif

#ifndef SYCL_FFT_DEFAULT_SIGN
#if SYCL_FFT_ENABLE_POS_SIGN
#define SYCL_FFT_DEFAULT_SIGN (+1)
#else
#define SYCL_FFT_DEFAULT_SIGN (-1)
#endif
#endif
#include "sycl_fft_math.hpp"
#include "sycl_fft_layout.hpp"
#include "sycl_fft_device.hpp"
#include "sycl_fft_bitrev.hpp"
#include <algorithm>
#include <new>
#include <stdexcept>

#if !defined(SYCL_FFT_ONLY_FLOAT) && !defined(SYCL_FFT_ONLY_DOUBLE) && !defined(SYCL_FFT_ONLY_DF)
#undef SYCL_FFT_DOUBLE
#undef SYCL_FFT_DF
#define SYCL_FFT_HALF
#include "sycl_fft_precision.hpp"
#include "sycl_fft_dif_impl.hpp"
#undef SYCL_FFT_HALF
#endif

#if !defined(SYCL_FFT_ONLY_HALF) && !defined(SYCL_FFT_ONLY_DOUBLE) && !defined(SYCL_FFT_ONLY_DF)
#undef SYCL_FFT_HALF
#undef SYCL_FFT_DOUBLE
#undef SYCL_FFT_DF
#include "sycl_fft_precision.hpp"
#include "sycl_fft_dif_impl.hpp"
#endif

#if !defined(SYCL_FFT_ONLY_HALF) && !defined(SYCL_FFT_ONLY_FLOAT) && !defined(SYCL_FFT_ONLY_DF)
#undef SYCL_FFT_HALF
#undef SYCL_FFT_DF
#define SYCL_FFT_DOUBLE
#include "sycl_fft_precision.hpp"
#include "sycl_fft_dif_impl.hpp"
#undef SYCL_FFT_DOUBLE
#endif

#if !defined(SYCL_FFT_ONLY_HALF) && !defined(SYCL_FFT_ONLY_FLOAT) && !defined(SYCL_FFT_ONLY_DOUBLE)
#undef SYCL_FFT_HALF
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
#undef SYCL_FFT_PREC_TAG
#undef SYCL_FFT_IS_DOUBLE
#undef SYCL_FFT_IS_DF
#undef SYCL_FFT_IS_HALF
#undef SYCL_FFT_PACKED_BATCH
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
