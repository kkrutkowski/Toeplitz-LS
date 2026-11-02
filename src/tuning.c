/* tuning.c - process-global algorithm switches used by the benchmark harness.
 *
 * Deliberately not thread-local: a plan reads these once at creation time and
 * a fastchi2() call reads the defer switch once at entry, so a setter racing
 * with a running call can only change which variant the *next* call uses.
 * Plain int stores are atomic enough for that contract.
 */

#include <nanofft.h>
#include <tls_tuning.h>

#ifndef NANOFFT_FFT_FUSE_MAX
#    define NANOFFT_FFT_FUSE_MAX 4
#endif

static int tuning_fft_fuse = TLS_TUNING_FFT_FUSE_DEFAULT;
static int tuning_cobravo = TLS_TUNING_COBRAVO_DEFAULT;
static int tuning_defer = TLS_TUNING_DEFER_DEFAULT;
static int tuning_permute = TLS_PERMUTE_AUTO;
static unsigned long long tuning_work_bytes = 0;

int tls_tuning_fft_fuse(void) { return tuning_fft_fuse; }

void tls_tuning_set_fft_fuse(int fuse) {
    if (fuse < 1) fuse = 1;
    if (fuse > NANOFFT_FFT_FUSE_MAX) fuse = NANOFFT_FFT_FUSE_MAX;
    tuning_fft_fuse = fuse;
}

int tls_tuning_cobravo(void) { return tuning_cobravo; }

void tls_tuning_set_cobravo(int on) { tuning_cobravo = on ? 1 : 0; }

int tls_tuning_defer(void) { return tuning_defer; }

void tls_tuning_set_defer(int on) { tuning_defer = on ? 1 : 0; }

int tls_tuning_permute(void) { return tuning_permute; }

void tls_tuning_set_permute(int mode) {
    if (mode < TLS_PERMUTE_AUTO) mode = TLS_PERMUTE_AUTO;
    if (mode > TLS_PERMUTE_COBRAVO) mode = TLS_PERMUTE_COBRAVO;
    tuning_permute = mode;
}

unsigned long long tls_tuning_work_bytes(void) { return tuning_work_bytes; }

/* Internal: chi2per reports the size of the work arrays it just allocated. */
void tls_tuning_report_work_bytes(unsigned long long bytes) { tuning_work_bytes = bytes; }
