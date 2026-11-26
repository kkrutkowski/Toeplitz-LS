#ifndef TLS_TUNING_H
#define TLS_TUNING_H

#include <nanofft.h>
#include <tls.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Runtime algorithm switches.
 *
 * These exist so one binary can be interleaved across algorithm variants by the
 * benchmark harness (see tests/bench.sh).  Every setter takes effect on the next
 * plan / periodogram creation, never in the middle of one, so a thread that
 * never calls a setter sees the compile-time defaults below.
 *
 * The defaults are the values a plain library build ships with; the benchmark
 * results in tests/README.md record how they were chosen.
 */
enum {
    TLS_TUNING_FFT_FUSE_AUTO = 0, /* the per-precision build default */
    TLS_TUNING_FFT_FUSE_DEFAULT = TLS_TUNING_FFT_FUSE_AUTO,
    TLS_TUNING_COBRAVO_DEFAULT = 1, /* 1 = vectorized tile bit reversal */
    TLS_TUNING_DEFER_DEFAULT = 1    /* 1 = deferred (slot-order) bit reversal */
};

/* FFT fusion depth: 0 = the per-precision default (floor(log2(VEC_LEN)) stages,
 * one less for float, 2 for double-double), or 1 (plain one-stage-at-a-time
 * DIF) up to NANOFFT_FFT_FUSE_MAX. */
TLS_API int tls_tuning_fft_fuse(void);
TLS_API void tls_tuning_set_fft_fuse(int fuse);

/* Breadth-wise FFT fusion depth: 0 = per-precision breadth default, or 1..NANOFFT_FFT_FUSE_MAX. */
TLS_API int tls_tuning_fft_bfuse(void);
TLS_API void tls_tuning_set_fft_bfuse(int fuse);

/* Fusion order: where the leftover stages of an odd stage count go.
 *   TLS_FFT_ORDER_END   = leftover pass first, then every full-depth pass
 *   TLS_FFT_ORDER_FRONT = every full-depth pass first, leftover pass last */
TLS_API int tls_tuning_fft_order(void);
TLS_API void tls_tuning_set_fft_order(int order);

enum {
    TLS_FFT_ORDER_END = 0,
    TLS_FFT_ORDER_FRONT = 1,
};

/* COBRAVO vectorized permutation on/off (0 keeps the scalar in-tile swap). */
TLS_API int tls_tuning_cobravo(void);
TLS_API void tls_tuning_set_cobravo(int on);

/* Deferred bit reversal on/off: with 0 every NuFFT returns frequency order. */
TLS_API int tls_tuning_defer(void);
TLS_API void tls_tuning_set_defer(int on);

/* IHS deconvolution placement when deferred bit reversal is active:
 *   TLS_IHS_DECONV_POST (0) = default: raw slots, deconv after sum & permute
 *   TLS_IHS_DECONV_PRE  (1) = deconv during slot emit (for benchmarking) */
enum {
    TLS_IHS_DECONV_POST = 0,
    TLS_IHS_DECONV_PRE = 1,
};
TLS_API int tls_tuning_ihs_deconv(void);
TLS_API void tls_tuning_set_ihs_deconv(int mode);

/* Bit-reversal strategy: 0 = automatic (the measured crossover), 1 = scalar
 * table shuffle, 2 = precomputed swap pairs, 3 = Cobravo tiled reversal.
 * Exposed so the benchmark harness can interleave strategies in one process. */
TLS_API int tls_tuning_permute(void);
TLS_API void tls_tuning_set_permute(int mode);

enum {
    TLS_PERMUTE_AUTO = 0,
    TLS_PERMUTE_TABLE = 1,
    TLS_PERMUTE_PAIRS = 2,
    TLS_PERMUTE_COBRAVO = 3,
};

/* PSWF spreading algorithm default when using the non-_spread C entry points:
 * TLS_SPREAD_AUTO (0) = cost-model choice, TLS_SPREAD_MINMAX (1) = precomputed
 * minmax polynomial weights, TLS_SPREAD_PIECEWISE (2) = on-the-fly vectorized
 * piecewise polynomial evaluation. */
TLS_API int tls_tuning_spread(void);
TLS_API void tls_tuning_set_spread(int mode);

/* Vectorization default when using the non-_opts C entry points:
 * TLS_VEC_AUTO (0) = cost-model choice, TLS_VEC_DEPTH (1) = depth-wise SIMD
 * within one block, TLS_VEC_BREADTH (2) = breadth-wise SIMD across V blocks. */
TLS_API int tls_tuning_vectorize(void);
TLS_API void tls_tuning_set_vectorize(int mode);

/* Bytes of periodogram work arrays allocated by the most recent fastchi2 call. */
TLS_API unsigned long long tls_tuning_work_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* TLS_TUNING_H */
