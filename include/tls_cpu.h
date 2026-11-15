#ifndef TLS_CPU_H
#define TLS_CPU_H

#include <tls_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * SVE vector length of the running CPU, in bytes (16, 32, 48, 64, ...), or 0
 * when the architecture has no SVE at all.
 *
 * SVE registers are variable-length, and the library's kernels are built for
 * one compile-time vector width (`-msve-vector-bits=N`), so the dispatcher has
 * to know the hardware length exactly before it may load an SVE library.  On
 * Linux that comes from prctl(PR_SVE_GET_VL); Windows exposes only an "SVE
 * present" feature bit, which is why this probe exists: it reads the length
 * with a single `cntb` instruction and is safe to call *only* after the OS has
 * reported SVE support.  Everywhere else it returns 0 and is never called.
 *
 * The function lives in the baseline library, so the dispatcher can ask the
 * already-loaded, always-safe library before choosing anything wider.
 */
TLS_API unsigned long tls_probe_sve_vector_length(void);

/*
 * What this CPU can actually execute, as a bitmask.  Filled in by the library
 * itself with CPUID (x86) or HWCAP / OS feature queries (AArch64), so the
 * Python dispatcher never has to guess from an OS summary: every bit set here
 * is a feature the instruction set really has *and* that the OS has enabled
 * state for (XCR0 for AVX, YMM/ZMM state).
 *
 * The order of these bits is mirrored by _FEATURE_BITS in
 * toeplitz_ls/_dispatch.py - change both together.
 */
#define TLS_CPU_SSE4_1 (1ull << 0)
#define TLS_CPU_SSE4_2 (1ull << 1)
#define TLS_CPU_SSSE3 (1ull << 2)
#define TLS_CPU_POPCNT (1ull << 3)
#define TLS_CPU_CX16 (1ull << 4)
#define TLS_CPU_LAHF_LM (1ull << 5)
#define TLS_CPU_AVX (1ull << 6)
#define TLS_CPU_XSAVE (1ull << 7)
#define TLS_CPU_AVX2 (1ull << 8)
#define TLS_CPU_FMA (1ull << 9)
#define TLS_CPU_BMI1 (1ull << 10)
#define TLS_CPU_BMI2 (1ull << 11)
#define TLS_CPU_F16C (1ull << 12)
#define TLS_CPU_LZCNT (1ull << 13)
#define TLS_CPU_MOVBE (1ull << 14)
#define TLS_CPU_AVX512F (1ull << 15)
#define TLS_CPU_AVX512BW (1ull << 16)
#define TLS_CPU_AVX512CD (1ull << 17)
#define TLS_CPU_AVX512DQ (1ull << 18)
#define TLS_CPU_AVX512VL (1ull << 19)
#define TLS_CPU_ASIMD (1ull << 20)
#define TLS_CPU_SVE (1ull << 21)
#define TLS_CPU_DOTPROD (1ull << 22)
#define TLS_CPU_I8MM (1ull << 23)
#define TLS_CPU_BF16 (1ull << 24)

TLS_API unsigned long long tls_cpu_features(void);

#ifdef __cplusplus
}
#endif

#endif /* TLS_CPU_H */
