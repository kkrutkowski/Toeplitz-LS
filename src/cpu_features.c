/*
 * cpu_features.c - ask the CPU itself which instruction sets are usable.
 *
 * The Python dispatcher must never pick a library from an OS summary alone:
 * Windows exposes four PF_* bits, macOS encodes flags as strings, and both are
 * incomplete for the microarchitecture levels we ship.  Instead the baseline
 * library - always present, always safe to load - answers with a bitmask built
 * from CPUID (x86) or HWCAP / OS feature queries (AArch64).  A bit means "this
 * CPU has the instruction set AND the OS has enabled the state for it", so a
 * variant is selected only when it can really execute.
 *
 * Kept free of precision defines: one object, not one per precision.
 */

#include <stdint.h>
#include <tls_cpu.h>

#if defined(__x86_64__) || defined(_M_X64)
#    define TLS_CPU_IS_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#    define TLS_CPU_IS_ARM 1
#endif

#if defined(TLS_CPU_IS_X86)

#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <cpuid.h>
#    endif

static void tls_cpuid(unsigned leaf, unsigned sub, unsigned out[4]) {
#    if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)sub);
    out[0] = (unsigned)regs[0];
    out[1] = (unsigned)regs[1];
    out[2] = (unsigned)regs[2];
    out[3] = (unsigned)regs[3];
#    else
    __cpuid_count(leaf, sub, out[0], out[1], out[2], out[3]);
#    endif
}

/* XCR0 tells us whether the OS enabled YMM/ZMM state; without that the
 * AVX/AVX-512 instructions fault even though CPUID advertises them.  The
 * instruction is emitted as bytes so old assemblers cannot choke on it. */
static unsigned long long tls_xgetbv0(void) {
#    if defined(_MSC_VER)
    return (unsigned long long)_xgetbv(0);
#    else
    uint32_t lo, hi;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0u));
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
#    endif
}

static unsigned long long tls_x86_features(void) {
    unsigned long long f = 0;
    unsigned regs[4];
    unsigned max_leaf;
    unsigned max_ext;
    int ymm_state = 0;
    int zmm_state = 0;

    tls_cpuid(0, 0, regs);
    max_leaf = regs[0];

    if (max_leaf >= 1) {
        unsigned ecx;
        tls_cpuid(1, 0, regs);
        ecx = regs[2];

        if (ecx & (1u << 9)) f |= TLS_CPU_SSSE3;
        if (ecx & (1u << 12)) f |= TLS_CPU_FMA;
        if (ecx & (1u << 13)) f |= TLS_CPU_CX16;
        if (ecx & (1u << 19)) f |= TLS_CPU_SSE4_1;
        if (ecx & (1u << 20)) f |= TLS_CPU_SSE4_2;
        if (ecx & (1u << 22)) f |= TLS_CPU_MOVBE;
        if (ecx & (1u << 23)) f |= TLS_CPU_POPCNT;
        if (ecx & (1u << 26)) f |= TLS_CPU_XSAVE;
        if (ecx & (1u << 29)) f |= TLS_CPU_F16C;

        if ((ecx & (1u << 27)) && (ecx & (1u << 28))) { /* OSXSAVE && AVX */
            unsigned long long xcr0 = tls_xgetbv0();
            if ((xcr0 & 0x6ull) == 0x6ull) {
                f |= TLS_CPU_AVX;
                ymm_state = 1;
                if ((xcr0 & 0xe0ull) == 0xe0ull) {
                    zmm_state = 1;
                }
            }
        }
    }

    if (max_leaf >= 7) {
        unsigned ebx, ecx;
        tls_cpuid(7, 0, regs);
        ebx = regs[1];
        ecx = regs[2];

        if (ymm_state) {
            if (ebx & (1u << 3)) f |= TLS_CPU_BMI1;
            if (ebx & (1u << 5)) f |= TLS_CPU_AVX2;
            if (ebx & (1u << 8)) f |= TLS_CPU_BMI2;
            if (zmm_state) {
                if (ebx & (1u << 16)) f |= TLS_CPU_AVX512F;
                if (ebx & (1u << 17)) f |= TLS_CPU_AVX512DQ;
                if (ebx & (1u << 28)) f |= TLS_CPU_AVX512CD;
                if (ebx & (1u << 30)) f |= TLS_CPU_AVX512BW;
                if (ebx & (1u << 31)) f |= TLS_CPU_AVX512VL;
            }
        }
        (void)ecx;
    }

    /* LAHF/SAHF in long mode and LZCNT/ABM live in the extended leaves. */
    tls_cpuid(0x80000000u, 0, regs);
    max_ext = regs[0];
    if (max_ext >= 0x80000001u) {
        unsigned ecx;
        tls_cpuid(0x80000001u, 0, regs);
        ecx = regs[2];
        if (ecx & (1u << 0)) f |= TLS_CPU_LAHF_LM;
        if (ecx & (1u << 5)) f |= TLS_CPU_LZCNT;
    }

    return f;
}

#endif /* TLS_CPU_IS_X86 */

#if defined(TLS_CPU_IS_ARM)

#    if defined(_WIN32)
#        include <windows.h>
#        ifndef PF_ARM_NEON_INSTRUCTIONS_AVAILABLE
#            define PF_ARM_NEON_INSTRUCTIONS_AVAILABLE 19
#        endif
#        ifndef PF_ARM_SVE_INSTRUCTIONS_AVAILABLE
#            define PF_ARM_SVE_INSTRUCTIONS_AVAILABLE 46
#        endif
#    elif !defined(__APPLE__)
#        include <sys/auxv.h>
#        ifndef HWCAP_ASIMD
#            define HWCAP_ASIMD (1ul << 1)
#        endif
#        ifndef HWCAP_SVE
#            define HWCAP_SVE (1ul << 22)
#        endif
#        ifndef HWCAP2_DOTPROD
#            define HWCAP2_DOTPROD (1ul << 20)
#        endif
#        ifndef HWCAP2_I8MM
#            define HWCAP2_I8MM (1ul << 13)
#        endif
#    endif

static unsigned long long tls_arm_features(void) {
    unsigned long long f = 0;

#    if defined(_WIN32)
    /* AArch64 always has NEON; SVE presence is all Windows publishes, so the
     * vector *length* has to come from tls_probe_sve_vector_length(). */
    if (IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE)) f |= TLS_CPU_ASIMD;
    if (IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)) f |= TLS_CPU_SVE;
#    elif defined(__APPLE__)
    /* Apple silicon: NEON always, no SVE, no SVE2. */
    f |= TLS_CPU_ASIMD;
#    else
    {
        unsigned long hwcap = getauxval(AT_HWCAP);
        if (hwcap & HWCAP_ASIMD) f |= TLS_CPU_ASIMD;
        if (hwcap & HWCAP_SVE) f |= TLS_CPU_SVE;
        {
            unsigned long hwcap2 = getauxval(AT_HWCAP2);
            if (hwcap2 & HWCAP2_DOTPROD) f |= TLS_CPU_DOTPROD;
            if (hwcap2 & HWCAP2_I8MM) f |= TLS_CPU_I8MM;
        }
    }
#    endif

    return f;
}

#endif /* TLS_CPU_IS_ARM */

TLS_API unsigned long long tls_cpu_features(void) {
#if defined(TLS_CPU_IS_X86)
    return tls_x86_features();
#elif defined(TLS_CPU_IS_ARM)
    return tls_arm_features();
#else
    return 0ull;
#endif
}
