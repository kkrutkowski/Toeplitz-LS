/*
 * probe_arm64.c - runtime SVE vector length probe.
 *
 * The library ships one shared object per compile-time vector width, because
 * the GNU vector extensions it is written with need a constant lane count.
 * Picking the right one therefore needs the *hardware* SVE vector length,
 * which Linux reports through prctl(PR_SVE_GET_VL) but Windows does not: it
 * only publishes a "SVE present" bit.  `cntb` returns VL/8, i.e. the length in
 * bytes, and is the portable way to ask the CPU directly.
 *
 * This translation unit is part of every aarch64 baseline library, so the
 * dispatcher can call it through the library it has already loaded safely.  It
 * must only be called once the operating system has confirmed SVE support:
 * executing `cntb` on a CPU without SVE is an illegal instruction.  Non-aarch64
 * targets compile the stub that returns 0.
 */

#include <stdint.h>
#include <tls_cpu.h>

#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(__APPLE__)

TLS_API unsigned long tls_probe_sve_vector_length(void) {
    /* 64-bit on purpose: the operand is an x-register, while unsigned long is
     * only 32 bits wide on Windows. */
    uint64_t bytes = 0;
    /* clang accepts GNU inline asm on aarch64 even in MSVC mode. */
    __asm__ volatile(".arch_extension sve\n\tcntb %0" : "=r"(bytes));
    return (unsigned long)bytes;
}

#else

TLS_API unsigned long tls_probe_sve_vector_length(void) { return 0; }

#endif
