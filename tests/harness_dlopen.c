/*
 * Minimal macOS dlopen harness for the packaged dylib.
 *
 * The wheel ships Mach-O dylibs rather than a Unix .so to link against, so this
 * harness loads the baseline dylib at run time and computes one small
 * periodogram.  Run it from `darling shell`; see dispatch/README.md.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>

typedef int (*fastchi2_fn)(const double *, const double *, const double *, int, double, double, int, int, int, int, int, double *, double *);

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <libtls.dylib>\n", argv[0]);
        return 2;
    }
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    fastchi2_fn fastchi2 = (fastchi2_fn)dlsym(handle, "tls_fastchi2");
    if (!fastchi2) {
        fprintf(stderr, "dlsym: %s\n", dlerror());
        return 1;
    }

    double t[64], y[64], dy[64], power[128], cond[128];
    for (int i = 0; i < 64; ++i) {
        t[i] = 0.137 * (double)i;
        y[i] = 1.0 + 0.3 * sin(2.0 * M_PI * 0.7 * t[i]) + 0.1 * cos(2.0 * M_PI * 1.4 * t[i]);
        dy[i] = 1.0;
    }
    int status = fastchi2(t, y, dy, 64, 0.01, 0.01, 128, 3, 1, 1, 0, power, cond);
    if (status < 0) {
        fprintf(stderr, "tls_fastchi2 failed: %d\n", status);
        return 1;
    }

    double peak = 0.0, sum = 0.0;
    for (int i = 0; i < 128; ++i) {
        if (isnan(power[i])) continue;
        if (power[i] > peak) peak = power[i];
        sum += power[i];
    }
    printf("CHECKSUM %.17g %.17g\n", peak, sum);
    return 0;
}
