#ifndef TLS_PLATFORM_H
#define TLS_PLATFORM_H

/*
 * Platform and compiler abstraction used by the library translation units.
 *
 * Everything here is either a compile-time macro or a static inline helper, so
 * including it costs nothing at run time.  Keep it free of any dependency
 * outside the C standard library: the release builds target glibc, musl, the
 * MSVC/UCRT ABI (through clang-cl) and macOS libSystem with the same sources.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <tls_export.h>

#if defined(_MSC_VER)
#    include <malloc.h> /* _aligned_malloc / _aligned_free */
#endif

/* -------------------------------------------------------------------------
 * restrict
 * ------------------------------------------------------------------------- */
#if defined(_MSC_VER)
#    define NANOFFT_RESTRICT __restrict
#else
#    define NANOFFT_RESTRICT __restrict__
#endif

/* -------------------------------------------------------------------------
 * Aligned allocation
 *
 * aligned_alloc() is C11 but its contract (size a multiple of the alignment)
 * is easy to violate and it is missing from older musl and from the MSVC
 * runtime, which offers _aligned_malloc() instead.  posix_memalign() is
 * available on every POSIX target and has no size constraint, so it is the
 * single POSIX path; MSVC keeps its own pair.
 *
 * Both helpers return NULL on failure or on a zero size, and a pointer that
 * must be released with nanofft_aligned_free() only.
 * ------------------------------------------------------------------------- */
static inline void *nanofft_aligned_alloc(size_t align, size_t bytes) {
    if (align < sizeof(void *)) align = sizeof(void *);
    if (bytes == 0) return NULL;
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, align);
#else
    {
        void *block = NULL;
        if (posix_memalign(&block, align, bytes) != 0) return NULL;
        return block;
    }
#endif
}

static inline void nanofft_aligned_free(void *block) {
#if defined(_MSC_VER)
    _aligned_free(block);
#else
    free(block);
#endif
}

/* Zero-filled allocation with the same alignment and the same free helper. */
static inline void *nanofft_calloc_aligned(size_t align, size_t count, size_t size) {
    void *block;
    if (count != 0 && size > (size_t)-1 / count) return NULL;
    block = nanofft_aligned_alloc(align, count * size);
    if (block != NULL) memset(block, 0, count * size);
    return block;
}

/*
 * Every allocation inside the library uses this pair: NANOFFT_ALIGN alignment
 * is what the vector loads and stores assume, and the matching free is what
 * keeps the MSVC path (which needs _aligned_free) correct.  On POSIX both are
 * plain malloc/free underneath.  NANOFFT_ALIGN comes from nanofft.h, so these
 * macros are expanded at the call site and tls_platform.h itself never needs
 * to see it.
 */
#define nanofft_malloc(bytes) nanofft_aligned_alloc(NANOFFT_ALIGN, (bytes))
#define nanofft_calloc(count, size) nanofft_calloc_aligned(NANOFFT_ALIGN, (count), (size))

/*
 * NaN / infinity sentinels.  Under -ffast-math the compiler is allowed to
 * assume neither occurs, so clang warns about the macros.  The expressions are
 * intentional (degenerate-input sentinels, and the double-double code that
 * relies on them is built without fast-math), so silence exactly those spots.
 */
#if defined(__clang__)
#    define NANOFFT_DIAG_PUSH_NAN _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wnan-infinity-disabled\"")
#    define NANOFFT_DIAG_POP _Pragma("clang diagnostic pop")
#else
#    define NANOFFT_DIAG_PUSH_NAN
#    define NANOFFT_DIAG_POP
#endif

/* Helpers that only some precision builds use. */
#if defined(__GNUC__) || defined(__clang__)
#    define NANOFFT_UNUSED __attribute__((unused))
#else
#    define NANOFFT_UNUSED
#endif

/* -------------------------------------------------------------------------
 * Exact floating-point ordering
 *
 * The compensated-arithmetic helpers (Kahan sums, error-free transforms) only
 * work when the compiler does not reassociate them.  GCC has a per-function
 * optimization attribute for that.  clang ignores that attribute, so it gets
 * the FP pragma instead, plus noinline so a fast-math caller cannot absorb the
 * body and undo the guard.
 * ------------------------------------------------------------------------- */
#if defined(__clang__)
#    define NANOFFT_EXACT_FP __attribute__((noinline))
#    define NANOFFT_EXACT_FP_PRAGMA _Pragma("clang fp reassociate(off)") _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#    define NANOFFT_EXACT_FP __attribute__((optimize("no-fast-math", "no-associative-math", "no-reciprocal-math", "no-unsafe-math-optimizations")))
#    define NANOFFT_EXACT_FP_PRAGMA
#else
#    define NANOFFT_EXACT_FP
#    define NANOFFT_EXACT_FP_PRAGMA
#endif

/* -------------------------------------------------------------------------
 * Vector shuffles
 *
 * GCC and clang agree on __builtin_shufflevector() (two input vectors whose
 * lanes are indexed 0..N-1 then N..2N-1), and clang has nothing else.
 * __builtin_shuffle() is the older GCC spelling and the only one that accepts
 * a mask held in a variable, which is why every call site below passes its
 * mask as a literal token list instead.
 *
 * The `vt` argument is the integer vector type matching the shuffle's lane
 * count; it is unused on the clang path but keeps one macro shape for both
 * compilers.
 * ------------------------------------------------------------------------- */
#if defined(__clang__)
#    define NANOFFT_SHUFFLE2(vt, a, b, ...) __builtin_shufflevector((a), (b), __VA_ARGS__)
#    define NANOFFT_SHUFFLE1(vt, v, ...) __builtin_shufflevector((v), (v), __VA_ARGS__)
#elif defined(__GNUC__)
#    define NANOFFT_SHUFFLE2(vt, a, b, ...) __builtin_shuffle((a), (b), (vt){__VA_ARGS__})
#    define NANOFFT_SHUFFLE1(vt, v, ...) __builtin_shuffle((v), (vt){__VA_ARGS__})
#else
#    error "this compiler provides neither __builtin_shufflevector nor __builtin_shuffle"
#endif

/* -------------------------------------------------------------------------
 * Mathematical constants
 *
 * glibc exposes the M_* family from <math.h> by default, musl only under
 * _XOPEN_SOURCE/_GNU_SOURCE, and the MSVC CRT not at all, so the library
 * carries its own copies.  Values are the correctly rounded binary64
 * constants (enough for both precisions; the double-double paths use their
 * own split literals).
 * ------------------------------------------------------------------------- */
#define NANOFFT_PI 3.14159265358979323846264338327950288
#define NANOFFT_TWO_PI 6.28318530717958647692528676655900577
#define NANOFFT_SQRT1_2 0.70710678118654752440084436210484904
#define NANOFFT_LN2 0.69314718055994530941723212145817657
#define NANOFFT_INV_LN2 1.44269504088896340735992468100189214

#endif /* TLS_PLATFORM_H */
