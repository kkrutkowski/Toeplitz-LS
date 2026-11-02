#ifndef TLS_TUNING_H
#define TLS_TUNING_H

#include <nanofft.h>

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
    TLS_TUNING_FFT_FUSE_DEFAULT = 3, /* DIF stages fused per FFT pass */
    TLS_TUNING_COBRAVO_DEFAULT = 1,  /* 1 = vectorized tile bit reversal */
    TLS_TUNING_DEFER_DEFAULT = 1     /* 1 = deferred (slot-order) bit reversal */
};

/* FFT fusion depth: 1 (plain one-stage-at-a-time DIF) up to NANOFFT_FFT_FUSE_MAX. */
int tls_tuning_fft_fuse(void);
void tls_tuning_set_fft_fuse(int fuse);

/* COBRAVO vectorized permutation on/off (0 keeps the scalar in-tile swap). */
int tls_tuning_cobravo(void);
void tls_tuning_set_cobravo(int on);

/* Deferred bit reversal on/off: with 0 every NuFFT returns frequency order. */
int tls_tuning_defer(void);
void tls_tuning_set_defer(int on);

/* Bytes of periodogram work arrays allocated by the most recent fastchi2 call. */
unsigned long long tls_tuning_work_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* TLS_TUNING_H */
