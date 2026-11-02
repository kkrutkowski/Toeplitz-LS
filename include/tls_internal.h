#ifndef TLS_INTERNAL_H
#define TLS_INTERNAL_H

/*
 * Internal interfaces shared between the library translation units.
 *
 * This header is not part of the installed API; it only exists so chi2per.c
 * can drive the slot-order (deferred bit reversal) NuFFT output without
 * putting those entry points in the public nufft1.h.
 */

#include <nanofft.h>
#include <nufft1.h>

#ifdef __cplusplus
extern "C" {
#endif

/* How a PSWF NuFFT hands one block back to the caller. */
enum {
    TLS_NUFFT_OUT_NATURAL = 0, /* frequency order, Nout entries (classic contract) */
    TLS_NUFFT_OUT_SLOTS = 1    /* kept FFT positions only, also Nout entries */
};

/*
 * SLOTS writes the deconvolved spectrum of the kept FFT positions in slot
 * order (position p holds bin reverse_bits(p)); TLS_*_nufft_pswf_finish_slots()
 * turns such a block back into frequency order.  SLOTS is only honoured when
 * the plan found the periodic kept-bin pattern, which
 * TLS_*_nufft_pswf_slots_available() reports.
 */
#define TLS_INTERNAL_PSWF_DECL(prefix, type, plan_type)                                                                               \
    void prefix##_nufft_pswf_execute_mode(plan_type *plan, const type *y_real, const type *y_imag, type *out_real, type *out_imag,      \
                                          int freq_factor, int mode);                                                                  \
    int prefix##_nufft_pswf_slots_available(plan_type *plan);                                                                    \
    void prefix##_nufft_pswf_finish_slots(plan_type *plan, const type *compact, type *out, int count);

TLS_INTERNAL_PSWF_DECL(tlsf, float, tlsf_nufft_pswf_plan)
TLS_INTERNAL_PSWF_DECL(tls, double, tls_nufft_pswf_plan)
TLS_INTERNAL_PSWF_DECL(tlsdd, dd_t, tlsdd_nufft_pswf_plan)

#undef TLS_INTERNAL_PSWF_DECL

/* chi2per reports the size of the work arrays it allocated (benchmark stats). */
void tls_tuning_report_work_bytes(unsigned long long bytes);

#ifdef __cplusplus
}
#endif

#endif /* TLS_INTERNAL_H */
