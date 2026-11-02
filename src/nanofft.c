#include <math.h>
#include <nanofft.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tls_tuning.h>

/* =========================================================================
 * Triple-Precision Macro Engine
 * ========================================================================= */
#define NANOFFT_NEEDS_INTERNAL_VEC
#include "nanofft_precision.h"
#undef NANOFFT_NEEDS_INTERNAL_VEC

#define PLAN_T PREFIX(plan)
#define FALLBACK_PLAN_T FALLBACK_PREFIX(plan)

/* Build-time depth of this precision pass: the runtime switch can only pick
 * up to it, and only those fused kernels are compiled. */
#if defined(DOUBLE_DOUBLE)
#    define NANOFFT_FFT_FUSE_BUILD NANOFFT_FFT_FUSE_DD
#elif defined(DOUBLE)
#    define NANOFFT_FFT_FUSE_BUILD NANOFFT_FFT_FUSE_DOUBLE
#else
#    define NANOFFT_FFT_FUSE_BUILD NANOFFT_FFT_FUSE_FLOAT
#endif

#define NANOFFT_EFFECTIVE_FUSE(fuse)                                                                                                                 ((fuse) < 1 ? 1 : ((fuse) > NANOFFT_FFT_FUSE_BUILD ? NANOFFT_FFT_FUSE_BUILD : (fuse)))

typedef struct {
    uint32_t i;
    uint32_t j;
} rev_pair;

struct PLAN_T {
    uint32_t N;
    int fuse;     /* radix-2 DIF stages fused per pass (1 = classically staged) */
    int cobravo;  /* 1 = vectorized tile bit reversal, 0 = scalar in-tile swap */
    FLOAT *twiddle_real;
    FLOAT *twiddle_imag;
    FLOAT *cobra_buffer_real; /* two tiles: A at offset 0, B at BLOCK_ELEMS */
    FLOAT *cobra_buffer_imag;
    int permute;        /* effective bit-reversal strategy for this plan */
    rev_pair *pairs;    /* lazily built swap-pair list, NULL until first use */
    uint32_t npairs;
};

/* =========================================================================
 * Internal scalar helpers
 * ========================================================================= */

static inline bool is_power_of_two(int N) { return (N > 0) && ((N & (N - 1)) == 0); }
static inline uint32_t intmin(uint32_t a, uint32_t b) { return (a < b) ? a : b; }
static inline uint32_t intmax(int32_t a, int32_t b) { return (a > b) ? a : b; }

#define LOG_BLOCK_WIDTH 6
#define BLOCK_WIDTH (1u << LOG_BLOCK_WIDTH)
#define BLOCK_ELEMS (BLOCK_WIDTH * BLOCK_WIDTH)

/* The classic COBRA 64x64 tiling needs log2(N) >= 12.  A tile width of
 * min(log2(N)/2, 6) runs the same algorithm down to log2(N) = 8, where the
 * scalar in-place permutation stops paying off. */
#define COBRAVO_MIN_LOG 8
/* Measured crossover of the precomputed swap-pair walk (tests/bench_permute.c):
 * below PAIRS_MIN_LOG the plain table shuffle wins, from PAIRS_MIN_LOG to
 * PAIRS_MAX_LOG the pair list does, and above it the 4N-byte pair table leaves
 * L1 and the tiled reversal takes over again. */
#define PAIRS_MIN_LOG 6
/* The pair table is 4N bytes whatever the element size, so the crossover only
 * depends on where it leaves L1: 2^13 entries for 4-byte floats, 2^12 once the
 * elements are 8 bytes wide (measured; see tests/bench_permute.c). */
#if defined(DOUBLE) || defined(DOUBLE_DOUBLE)
#    define PAIRS_MAX_LOG 12
#else
#    define PAIRS_MAX_LOG 13
#endif
#define COBRAVO_W4_LEN (1u << 8)  /* in-tile indices of a 16x16 tile */
#define COBRAVO_W5_LEN (1u << 10) /* 32x32 */
#define COBRAVO_W6_LEN (1u << 12) /* 64x64 */
#define COBRAVO_IDX_LEN (COBRAVO_W4_LEN + COBRAVO_W5_LEN + COBRAVO_W6_LEN)

/* Gather-emitted tiles for the real types; double-double keeps the scalar
 * in-tile swap (its 16-byte elements have no gather). */
#if !defined(DOUBLE_DOUBLE) && (defined(__AVX512F__) || defined(__AVX2__))
#    define COBRAVO_GATHER 1
#    include <immintrin.h>
#else
#    define COBRAVO_GATHER 0
#endif

static inline uint32_t intlog2(uint32_t input) {
    uint32_t output;
    frexp(input >> 1, (int *)&output);
    return output;
}

/* =========================================================================
 * Bit-reverse permutation
 * ========================================================================= */

static const uint8_t bit_reverse_table[256] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98,
    0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC,
    0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2,
    0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA, 0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6,
    0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE, 0x01, 0x81,
    0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9,
    0x39, 0xB9, 0x79, 0xF9, 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD,
    0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD, 0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
    0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB, 0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97,
    0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

static inline uint32_t reverse_bits(uint32_t num, uint32_t bits) {
    uint32_t result = 0;
    unsigned int bytes = (bits + 7) >> 3;

    for (unsigned int i = 0; i < bytes; ++i) {
        result <<= 8;
        result |= bit_reverse_table[num & 0xFF];
        num >>= 8;
    }

    result >>= (8 * bytes - bits);
    return result;
}

static inline void table_shuffle(FLOAT *real, FLOAT *imag, uint32_t log_n) {
    for (int i = 0; i < (1 << log_n); ++i) {
        int j = reverse_bits(i, log_n);
        if (j > i) {
            FLOAT temp_real = real[i];
            FLOAT temp_imag = imag[i];

            real[i] = real[j];
            imag[i] = imag[j];

            real[j] = temp_real;
            imag[j] = temp_imag;
        }
    }
}

static inline void table_shuffle_1d(FLOAT *values, uint32_t log_n) {
    for (uint32_t i = 0; i < (1u << log_n); ++i) {
        uint32_t j = reverse_bits(i, log_n);
        if (j > i) {
            FLOAT temp = values[i];
            values[i] = values[j];
            values[j] = temp;
        }
    }
}

/* Swap pairs of the bit reversal, precomputed once per plan: unlike
 * table_shuffle this walks only the N/2 - 1 pairs that actually exchange two
 * elements, needs no bit arithmetic per element and has no per-element branch.
 * The table is 8 bytes per pair (4N bytes total), so it is only built for the
 * grid sizes where the microbenchmark shows it winning. */

static inline void pair_shuffle(FLOAT *real, FLOAT *imag, const rev_pair *pairs, uint32_t npairs) {
    for (uint32_t k = 0; k < npairs; ++k) {
        const uint32_t i = pairs[k].i;
        const uint32_t j = pairs[k].j;
        const FLOAT tr = real[i];
        const FLOAT ti = imag[i];
        real[i] = real[j];
        imag[i] = imag[j];
        real[j] = tr;
        imag[j] = ti;
    }
}

static inline void pair_shuffle_1d(FLOAT *values, const rev_pair *pairs, uint32_t npairs) {
    for (uint32_t k = 0; k < npairs; ++k) {
        const uint32_t i = pairs[k].i;
        const uint32_t j = pairs[k].j;
        const FLOAT t = values[i];
        values[i] = values[j];
        values[j] = t;
    }
}

/* -------------------------------------------------------------------------
 * COBRAVO tile bit reversal
 *
 * The index is split as (a | b | c), where a and c are the top and bottom w
 * bits.  A 2w-bit reversal maps a <-> c, so every 2^w x 2^w tile is reversed
 * inside a contiguous staging buffer and written to the mirrored b offset.
 * Each row of a tile is at least one SIMD vector wide, so the staging buffer
 * is loaded and stored with row-sized memcpy / vector moves.
 * ------------------------------------------------------------------------- */

#if COBRAVO_GATHER
/* In-tile reversal indices of every supported tile width, packed back to back.
 * Pure data derived from the fixed bit reversal, built by a constructor so no
 * transform can observe a half-initialised table. */
static int cobravo_gather_idx[COBRAVO_IDX_LEN] __attribute__((aligned(64)));

static inline int *cobravo_idx_table(uint32_t w) {
    if (w > 5u) return cobravo_gather_idx + COBRAVO_W4_LEN + COBRAVO_W5_LEN;
    if (w == 5u) return cobravo_gather_idx + COBRAVO_W4_LEN;
    return cobravo_gather_idx;
}

__attribute__((constructor)) static void cobravo_build_idx_table(void) {
    for (uint32_t w = 4u; w <= (uint32_t)LOG_BLOCK_WIDTH; ++w) {
        int *tab = cobravo_idx_table(w);
        const uint32_t tile = 1u << (2u * w);
        for (uint32_t i = 0; i < tile; ++i) tab[i] = (int)reverse_bits(i, 2u * w);
    }
}
#else
static inline int *cobravo_idx_table(uint32_t w) {
    (void)w;
    return NULL;
}
#endif

static inline void cobravo_load_tile(const FLOAT *src, uint32_t b_shifted, uint32_t shift_top, uint32_t w, FLOAT *dst) {
    const uint32_t width = 1u << w;
    for (uint32_t a = 0; a < width; ++a) memcpy(dst + (size_t)a * width, src + ((size_t)a << shift_top) + b_shifted, (size_t)width * sizeof(FLOAT));
}

static inline void cobravo_store_tile(const FLOAT *src, FLOAT *dst, uint32_t b_shifted, uint32_t shift_top, uint32_t w) {
    const uint32_t width = 1u << w;
    for (uint32_t a = 0; a < width; ++a) memcpy(dst + ((size_t)a << shift_top) + b_shifted, src + (size_t)a * width, (size_t)width * sizeof(FLOAT));
}

#if COBRAVO_GATHER
/* Destination positions are enumerated in order; the source index is the
 * in-tile reversal, so one gather plus one contiguous store per vector. */
static inline void cobravo_emit_tile_gather(const FLOAT *src, FLOAT *dst, uint32_t b_shifted, uint32_t shift_top, uint32_t w) {
    const uint32_t tile = 1u << (2u * w);
    const uint32_t row_mask = (1u << w) - 1u;
    const int *idx = cobravo_idx_table(w);
#    if defined(DOUBLE)
#        if defined(__AVX512F__)
    for (uint32_t v = 0; v < tile; v += 8u) {
        const __m256i ix = _mm256_load_si256((const __m256i *)(const void *)(idx + v));
        const __m512d x = _mm512_i32gather_pd(ix, src, 8);
        const uint32_t off = ((v >> w) << shift_top) | b_shifted | (v & row_mask);
        _mm512_storeu_pd(dst + off, x);
    }
#        else
    for (uint32_t v = 0; v < tile; v += 4u) {
        const __m128i ix = _mm_load_si128((const __m128i *)(const void *)(idx + v));
        const __m256d x = _mm256_i32gather_pd(src, ix, 8);
        const uint32_t off = ((v >> w) << shift_top) | b_shifted | (v & row_mask);
        _mm256_storeu_pd(dst + off, x);
    }
#        endif
#    else
#        if defined(__AVX512F__)
    for (uint32_t v = 0; v < tile; v += 16u) {
        const __m512i ix = _mm512_load_si512((const void *)(idx + v));
        const __m512 x = _mm512_i32gather_ps(ix, src, 4);
        const uint32_t off = ((v >> w) << shift_top) | b_shifted | (v & row_mask);
        _mm512_storeu_ps(dst + off, x);
    }
#        else
    for (uint32_t v = 0; v < tile; v += 8u) {
        const __m256i ix = _mm256_load_si256((const __m256i *)(const void *)(idx + v));
        const __m256 x = _mm256_i32gather_ps(src, ix, 4);
        const uint32_t off = ((v >> w) << shift_top) | b_shifted | (v & row_mask);
        _mm256_storeu_ps(dst + off, x);
    }
#        endif
#    endif
}
#endif /* COBRAVO_GATHER */

static inline void cobravo_reverse_tile(FLOAT *tile_values, uint32_t w) {
    const uint32_t tile = 1u << (2u * w);
    for (uint32_t i = 0; i < tile; ++i) {
        const uint32_t j = reverse_bits(i, 2u * w);
        if (j > i) {
            const FLOAT temp = tile_values[i];
            tile_values[i] = tile_values[j];
            tile_values[j] = temp;
        }
    }
}

static inline void cobravo_emit_tile(FLOAT *src, FLOAT *dst, uint32_t b_shifted, uint32_t shift_top, uint32_t w, int vectorized) {
#if COBRAVO_GATHER
    /* The gather writes one contiguous vector per step and derives its offset
     * from the first element, so it is only valid when a tile row holds a whole
     * vector; narrower tiles must use the scalar swap below. */
    if (vectorized && w >= 4u) {
        cobravo_emit_tile_gather(src, dst, b_shifted, shift_top, w);
        return;
    }
#else
    (void)vectorized;
#endif
    cobravo_reverse_tile(src, w);
    cobravo_store_tile(src, dst, b_shifted, shift_top, w);
}

static inline void cobravo_shuffle(FLOAT *real, FLOAT *imag, uint32_t log_n, FLOAT *bufA_real, FLOAT *bufA_imag, FLOAT *bufB_real, FLOAT *bufB_imag,
                                   int vectorized) {
    uint32_t w = log_n >> 1;
    if (w > (uint32_t)LOG_BLOCK_WIDTH) w = (uint32_t)LOG_BLOCK_WIDTH;
    const uint32_t width = 1u << w;
    const uint32_t num_b_bits = log_n - 2u * w;
    const uint32_t b_size = 1u << num_b_bits;
    const uint32_t shift_top = num_b_bits + w;

    for (uint32_t b = 0; b < b_size; ++b) {
        const uint32_t b_rev = reverse_bits(b, num_b_bits);
        if (b > b_rev) continue;

        const uint32_t b_shifted = b * width;
        const uint32_t b_rev_shifted = b_rev * width;

        cobravo_load_tile(real, b_shifted, shift_top, w, bufA_real);
        cobravo_load_tile(imag, b_shifted, shift_top, w, bufA_imag);
        if (b != b_rev) {
            cobravo_load_tile(real, b_rev_shifted, shift_top, w, bufB_real);
            cobravo_load_tile(imag, b_rev_shifted, shift_top, w, bufB_imag);
        }

        cobravo_emit_tile(bufA_real, real, b_rev_shifted, shift_top, w, vectorized);
        cobravo_emit_tile(bufA_imag, imag, b_rev_shifted, shift_top, w, vectorized);
        if (b != b_rev) {
            cobravo_emit_tile(bufB_real, real, b_shifted, shift_top, w, vectorized);
            cobravo_emit_tile(bufB_imag, imag, b_shifted, shift_top, w, vectorized);
        }
    }
}

static inline void cobravo_shuffle_1d(FLOAT *values, uint32_t log_n, FLOAT *bufA, FLOAT *bufB, int vectorized) {
    uint32_t w = log_n >> 1;
    if (w > (uint32_t)LOG_BLOCK_WIDTH) w = (uint32_t)LOG_BLOCK_WIDTH;
    const uint32_t width = 1u << w;
    const uint32_t num_b_bits = log_n - 2u * w;
    const uint32_t b_size = 1u << num_b_bits;
    const uint32_t shift_top = num_b_bits + w;

    for (uint32_t b = 0; b < b_size; ++b) {
        const uint32_t b_rev = reverse_bits(b, num_b_bits);
        if (b > b_rev) continue;

        const uint32_t b_shifted = b * width;
        const uint32_t b_rev_shifted = b_rev * width;

        cobravo_load_tile(values, b_shifted, shift_top, w, bufA);
        if (b != b_rev) cobravo_load_tile(values, b_rev_shifted, shift_top, w, bufB);

        cobravo_emit_tile(bufA, values, b_rev_shifted, shift_top, w, vectorized);
        if (b != b_rev) cobravo_emit_tile(bufB, values, b_shifted, shift_top, w, vectorized);
    }
}

/* Build the swap-pair list on first use: one reverse_bits() sweep, then every
 * later permutation of this plan is a branch-free walk over the pairs. */
static void ensure_pairs(PLAN_T *plan) {
    if (plan->pairs != NULL) return;
    const uint32_t order = intlog2(plan->N);
    const uint32_t capacity = plan->N / 2u;
    rev_pair *pairs = (rev_pair *)malloc((size_t)capacity * sizeof(rev_pair));
    if (pairs == NULL) return;
    uint32_t count = 0;
    for (uint32_t i = 1; i < plan->N; ++i) {
        const uint32_t j = reverse_bits(i, order);
        if (j > i) {
            pairs[count].i = i;
            pairs[count].j = j;
            ++count;
        }
    }
    plan->pairs = pairs;
    plan->npairs = count;
}

inline static void bit_reverse_permutation(FLOAT *real, FLOAT *imag, uint32_t N, PLAN_T *plan) {
    const uint32_t order = intlog2(N);
    if (plan->permute == TLS_PERMUTE_PAIRS && plan->pairs == NULL) ensure_pairs(plan);
    if (plan->permute == TLS_PERMUTE_PAIRS && plan->pairs != NULL) {
        pair_shuffle(real, imag, plan->pairs, plan->npairs);
        return;
    }
    if (plan->permute != TLS_PERMUTE_COBRAVO) {
        table_shuffle(real, imag, order);
        return;
    }
    cobravo_shuffle(real, imag, order, plan->cobra_buffer_real, plan->cobra_buffer_imag, plan->cobra_buffer_real + BLOCK_ELEMS,
                    plan->cobra_buffer_imag + BLOCK_ELEMS, plan->cobravo);
}

inline static void bit_reverse_permutation_1d(FLOAT *values, uint32_t N, PLAN_T *plan) {
    const uint32_t order = intlog2(N);
    if (plan->permute == TLS_PERMUTE_PAIRS && plan->pairs == NULL) ensure_pairs(plan);
    if (plan->permute == TLS_PERMUTE_PAIRS && plan->pairs != NULL) {
        pair_shuffle_1d(values, plan->pairs, plan->npairs);
        return;
    }
    if (plan->permute != TLS_PERMUTE_COBRAVO) {
        table_shuffle_1d(values, order);
        return;
    }
    cobravo_shuffle_1d(values, order, plan->cobra_buffer_real, plan->cobra_buffer_real + BLOCK_ELEMS, plan->cobravo);
}

/* =========================================================================
 * Intra-vector butterfly pass
 * ========================================================================= */

#if INTERNAL_VEC_LEN == 16 || INTERNAL_VEC_LEN == 8 || INTERNAL_VEC_LEN == 4 || INTERNAL_VEC_LEN == 2
#    define HAS_INTRA_VEC_PASS

#    if INTERNAL_VEC_LEN == 16
#        define W16_C1 ((FLOAT)0.9238795325112867)
#        define W16_S1 ((FLOAT)0.3826834323650898)

static const INTERNAL_VEC_INT permutations[4] __attribute__((aligned(64))) = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                                                                              {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15},
                                                                              {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15},
                                                                              {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}};

static const INTERNAL_VEC_INT inv_permutations[4] __attribute__((aligned(64))) = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                                                                                  {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15},
                                                                                  {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15},
                                                                                  {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}};

static const INTERNAL_VEC real_twiddles[4]
    __attribute__((aligned(64))) = {{(FLOAT)1.0, W16_C1, (FLOAT)M_SQRT1_2, W16_S1, (FLOAT)0.0, -W16_S1, (FLOAT)-M_SQRT1_2, -W16_C1, (FLOAT)1.0, W16_C1,
                                     (FLOAT)M_SQRT1_2, W16_S1, (FLOAT)0.0, -W16_S1, (FLOAT)-M_SQRT1_2, -W16_C1},
                                    {(FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)-M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)-M_SQRT1_2,
                                     (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)-M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)-M_SQRT1_2},
                                    {(FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0,
                                     (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0},
                                    {(FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0,
                                     (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0}};

static const INTERNAL_VEC imag_twiddles[4]
    __attribute__((aligned(64))) = {{(FLOAT)0.0, W16_S1, (FLOAT)M_SQRT1_2, W16_C1, (FLOAT)1.0, W16_C1, (FLOAT)M_SQRT1_2, W16_S1, (FLOAT)0.0, W16_S1,
                                     (FLOAT)M_SQRT1_2, W16_C1, (FLOAT)1.0, W16_C1, (FLOAT)M_SQRT1_2, W16_S1},
                                    {(FLOAT)0.0, (FLOAT)M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2,
                                     (FLOAT)0.0, (FLOAT)M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2},
                                    {(FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0,
                                     (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0},
                                    {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0,
                                     (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0}};

static inline void nanofft_vec_shuffle(INTERNAL_VEC *a, INTERNAL_VEC *b) {
    INTERNAL_VEC tmp = __builtin_shuffle(*a, *b, (INTERNAL_VEC_INT){0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23});
    *b = __builtin_shuffle(*a, *b, (INTERNAL_VEC_INT){8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31});
    *a = tmp;
}

#    elif INTERNAL_VEC_LEN == 8
static const INTERNAL_VEC_INT permutations[3] __attribute__((aligned(64))) = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 4, 5, 2, 3, 6, 7}, {0, 2, 4, 6, 1, 3, 5, 7}};
static const INTERNAL_VEC_INT inv_permutations[3] __attribute__((aligned(64))) = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 4, 5, 2, 3, 6, 7}, {0, 4, 1, 5, 2, 6, 3, 7}};

static const INTERNAL_VEC real_twiddles[3]
    __attribute__((aligned(64))) = {{(FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)-M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)-M_SQRT1_2},
                                    {(FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0},
                                    {(FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0}};

static const INTERNAL_VEC imag_twiddles[3]
    __attribute__((aligned(64))) = {{(FLOAT)0.0, (FLOAT)M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2, (FLOAT)0.0, (FLOAT)M_SQRT1_2, (FLOAT)1.0, (FLOAT)M_SQRT1_2},
                                    {(FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0},
                                    {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0}};

static inline void nanofft_vec_shuffle(INTERNAL_VEC *a, INTERNAL_VEC *b) {
    INTERNAL_VEC tmp = __builtin_shuffle(*a, *b, (INTERNAL_VEC_INT){0, 1, 2, 3, 8, 9, 10, 11});
    *b = __builtin_shuffle(*a, *b, (INTERNAL_VEC_INT){4, 5, 6, 7, 12, 13, 14, 15});
    *a = tmp;
}

#    elif INTERNAL_VEC_LEN == 4
static const INTERNAL_VEC_INT permutations[2] __attribute__((aligned(64))) = {{0, 1, 2, 3}, {0, 2, 1, 3}};
static const INTERNAL_VEC_INT inv_permutations[2] __attribute__((aligned(64))) = {{0, 1, 2, 3}, {0, 2, 1, 3}};

static const INTERNAL_VEC real_twiddles[2]
    __attribute__((aligned(64))) = {{(FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0}, {(FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0, (FLOAT)1.0}};
static const INTERNAL_VEC imag_twiddles[2]
    __attribute__((aligned(64))) = {{(FLOAT)0.0, (FLOAT)1.0, (FLOAT)0.0, (FLOAT)1.0}, {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0}};

static inline void nanofft_vec_shuffle(INTERNAL_VEC *a, INTERNAL_VEC *b) {
    INTERNAL_VEC tmp = __builtin_shuffle(*a, *b, (INTERNAL_VEC_INT){0, 1, 4, 5});
    *b = __builtin_shuffle(*a, *b, (INTERNAL_VEC_INT){2, 3, 6, 7});
    *a = tmp;
}

#    elif INTERNAL_VEC_LEN == 2
static const INTERNAL_VEC_INT permutations[1] __attribute__((aligned(64))) = {{0, 1}};
static const INTERNAL_VEC_INT inv_permutations[1] __attribute__((aligned(64))) = {{0, 1}};
static const INTERNAL_VEC real_twiddles[1] __attribute__((aligned(64))) = {{(FLOAT)1.0, (FLOAT)1.0}};
static const INTERNAL_VEC imag_twiddles[1] __attribute__((aligned(64))) = {{(FLOAT)0.0, (FLOAT)0.0}};

static inline void nanofft_vec_shuffle(INTERNAL_VEC *a, INTERNAL_VEC *b) {
    INTERNAL_VEC tmp = __builtin_shuffle(*a, *b, (INTERNAL_VEC_INT){0, 2});
    *b = __builtin_shuffle(*a, *b, (INTERNAL_VEC_INT){1, 3});
    *a = tmp;
}
#    endif

static inline void nanofft_vec_perm(INTERNAL_VEC *a, INTERNAL_VEC *b, uint32_t idx) {
    *a = __builtin_shuffle(*a, permutations[idx]);
    *b = __builtin_shuffle(*b, permutations[idx]);
}

static inline void nanofft_vec_inv_perm(INTERNAL_VEC *a, INTERNAL_VEC *b, uint32_t idx) {
    *a = __builtin_shuffle(*a, inv_permutations[idx]);
    *b = __builtin_shuffle(*b, inv_permutations[idx]);
}
#endif

/* =========================================================================
 * Core FFT kernels
 * ========================================================================= */

/*
 * Radix-2^L DIF pass.
 *
 * A pass fusing L Sande-Tukey DIF stages with initial step `step` acts on the
 * 2^L points at i + j + m*q (q = step >> L).  Stage t (0-based) pairs m with
 * m + 2^(L-1-t) and applies the twiddle exp(-2 pi i * 2^t * (j + m*q) / step),
 * which is exactly what the plain stage-at-a-time loop applies.  The plan's
 * twiddle table stores, per pass and per stage, those factors with m as the
 * outer and j the inner index, so a fused pass streams (2^L - 1) vectors per
 * j-block instead of reloading the buffer once per stage.
 */
#define NANOFFT_FUSED_PASS_BODY(L)                                                                                    \
    {                                                                                                                 \
        const uint32_t q = step >> (L);                                                                                \
        uint32_t base[1 << (L)];                                                                                      \
        {                                                                                                             \
            uint32_t acc = shift;                                                                                     \
            for (uint32_t t = 0; t < (L); ++t) {                                                                      \
                base[t] = acc;                                                                                        \
                acc += step >> (t + 1);                                                                               \
            }                                                                                                         \
        }                                                                                                             \
        for (uint32_t i = 0; i < n; i += step) {                                                                      \
            for (uint32_t j = 0; j < q; j += INTERNAL_VEC_LEN) {                                                      \
                INTERNAL_VEC vr[1 << (L)], vi[1 << (L)];                                                              \
                _Pragma("GCC unroll 16") for (uint32_t m = 0; m < (1u << (L)); ++m) {                                 \
                    vr[m] = LOAD_VEC(&real[i + j + m * q]);                                                           \
                    vi[m] = LOAD_VEC(&imag[i + j + m * q]);                                                           \
                }                                                                                                     \
                _Pragma("GCC unroll 4") for (uint32_t t = 0; t < (L); ++t) {                                          \
                    const uint32_t h = 1u << ((L) - 1u - t);                                                          \
                    _Pragma("GCC unroll 16") for (uint32_t lo = 0; lo < (1u << (L)); lo += 2u * h) {                  \
                        _Pragma("GCC unroll 8") for (uint32_t m = 0; m < h; ++m) {                                    \
                            const uint32_t a = lo + m, b = a + h;                                                     \
                            const INTERNAL_VEC wr = LOAD_VEC(&tw_r[base[t] + m * q + j]);                             \
                            const INTERNAL_VEC wi = LOAD_VEC(&tw_i[base[t] + m * q + j]);                             \
                            const INTERNAL_VEC sr = ADD(vr[a], vr[b]);                                                \
                            const INTERNAL_VEC si = ADD(vi[a], vi[b]);                                                \
                            const INTERNAL_VEC dr = SUB(vr[a], vr[b]);                                                \
                            const INTERNAL_VEC di = SUB(vi[a], vi[b]);                                                \
                            vr[a] = sr;                                                                               \
                            vi[a] = si;                                                                               \
                            vr[b] = SUB(MUL(dr, wr), MUL(di, wi));                                                    \
                            vi[b] = ADD(MUL(dr, wi), MUL(di, wr));                                                    \
                        }                                                                                             \
                    }                                                                                                 \
                }                                                                                                     \
                _Pragma("GCC unroll 16") for (uint32_t m = 0; m < (1u << (L)); ++m) {                                 \
                    STORE_VEC(&real[i + j + m * q], vr[m]);                                                           \
                    STORE_VEC(&imag[i + j + m * q], vi[m]);                                                           \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

static inline void sande_tukey_fused_pass(FLOAT *real, FLOAT *imag, const FLOAT *tw_r, const FLOAT *tw_i, uint32_t n, uint32_t step, uint32_t shift,
                                         int layers) {
    switch (layers) {
        case 1:
            NANOFFT_FUSED_PASS_BODY(1)
            break;
#if NANOFFT_FFT_FUSE_BUILD >= 2
        case 2:
            NANOFFT_FUSED_PASS_BODY(2)
            break;
#endif
#if NANOFFT_FFT_FUSE_BUILD >= 3
        case 3:
            NANOFFT_FUSED_PASS_BODY(3)
            break;
#endif
#if NANOFFT_FFT_FUSE_BUILD >= 4
        case 4:
            NANOFFT_FUSED_PASS_BODY(4)
            break;
#endif
        default:
            break;
    }
}

/* Plain stage-at-a-time DIF (fuse == 1).  Kept separate from the fused kernel
 * so the un-fused build behaves exactly like it always did. */
static void sande_tukey_stages(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, uint32_t N, uint32_t shift) {
    for (uint32_t step = N; step > INTERNAL_VEC_LEN; step >>= 1) {
        uint32_t half_step = step >> 1;
        for (uint32_t i = 0; i < N; i += step) {
            for (uint32_t j = 0; j < half_step; j += INTERNAL_VEC_LEN) {
                INTERNAL_VEC r_even = LOAD_VEC(&real_signal[i + j]);
                INTERNAL_VEC i_even = LOAD_VEC(&imag_signal[i + j]);
                INTERNAL_VEC r_odd = LOAD_VEC(&real_signal[i + j + half_step]);
                INTERNAL_VEC i_odd = LOAD_VEC(&imag_signal[i + j + half_step]);

                STORE_VEC(&real_signal[i + j], ADD(r_even, r_odd));
                STORE_VEC(&imag_signal[i + j], ADD(i_even, i_odd));

                INTERNAL_VEC r_tmp = SUB(r_even, r_odd);
                INTERNAL_VEC i_tmp = SUB(i_even, i_odd);
                INTERNAL_VEC b_real = LOAD_VEC(&real_buffer[shift + j]);
                INTERNAL_VEC b_imag = LOAD_VEC(&imag_buffer[shift + j]);

                STORE_VEC(&real_signal[i + j + half_step], SUB(MUL(r_tmp, b_real), MUL(i_tmp, b_imag)));
                STORE_VEC(&imag_signal[i + j + half_step], ADD(MUL(r_tmp, b_imag), MUL(i_tmp, b_real)));
            }
        }
        shift += half_step;
    }
}

static void sande_tukey_in_place(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, uint32_t N, int fuse) {
    const uint32_t log_n = intlog2(N);
    const uint32_t log_v = (INTERNAL_VEC_LEN > 1) ? intlog2((uint32_t)INTERNAL_VEC_LEN) : 0;
    const uint32_t stages = (log_n > log_v) ? log_n - log_v : 0;
    const uint32_t layers = (uint32_t)fuse;
    uint32_t shift = 0;
    uint32_t step = N;

    if (layers == 1) {
        /* The plain transform leaves the same state the intra-vector pass
         * expects; the scalar tail below is only reachable with one lane. */
        sande_tukey_stages(real_signal, imag_signal, real_buffer, imag_buffer, N, 0);
        goto intra_vector;
    }

    for (uint32_t pass = 0; pass < stages / layers; ++pass) {
        sande_tukey_fused_pass(real_signal, imag_signal, real_buffer, imag_buffer, N, step, shift, (int)layers);
        shift += step - (step >> layers);
        step >>= layers;
    }

    const uint32_t remainder = stages % layers;
    if (remainder) {
        sande_tukey_fused_pass(real_signal, imag_signal, real_buffer, imag_buffer, N, step, shift, (int)remainder);
        shift += step - (step >> remainder);
        step >>= remainder;
    }

intra_vector:
#ifdef HAS_INTRA_VEC_PASS
    for (uint32_t i = intmax(0, (int32_t)intlog2(INTERNAL_VEC_LEN) - (int32_t)log_n); i < (uint32_t)intlog2(INTERNAL_VEC_LEN); i++) {
        for (uint32_t j = 0; j < N; j += INTERNAL_VEC_LEN * 2) {
            INTERNAL_VEC r_even = LOAD_VEC(&real_signal[j]);
            INTERNAL_VEC r_odd = LOAD_VEC(&real_signal[j + INTERNAL_VEC_LEN]);
            nanofft_vec_perm(&r_even, &r_odd, i);
            nanofft_vec_shuffle(&r_even, &r_odd);

            INTERNAL_VEC i_even = LOAD_VEC(&imag_signal[j]);
            INTERNAL_VEC i_odd = LOAD_VEC(&imag_signal[j + INTERNAL_VEC_LEN]);
            nanofft_vec_perm(&i_even, &i_odd, i);
            nanofft_vec_shuffle(&i_even, &i_odd);

            INTERNAL_VEC r_tmp = SUB(r_even, r_odd);
            INTERNAL_VEC i_tmp = SUB(i_even, i_odd);

            r_even = ADD(r_even, r_odd);
            i_even = ADD(i_even, i_odd);

            r_odd = SUB(MUL(r_tmp, real_twiddles[i]), MUL(i_tmp, imag_twiddles[i]));
            i_odd = ADD(MUL(r_tmp, imag_twiddles[i]), MUL(i_tmp, real_twiddles[i]));

            nanofft_vec_shuffle(&r_even, &r_odd);
            nanofft_vec_inv_perm(&r_even, &r_odd, i);
            STORE_VEC(&real_signal[j], r_even);
            STORE_VEC(&real_signal[j + INTERNAL_VEC_LEN], r_odd);

            nanofft_vec_shuffle(&i_even, &i_odd);
            nanofft_vec_inv_perm(&i_even, &i_odd, i);
            STORE_VEC(&imag_signal[j], i_even);
            STORE_VEC(&imag_signal[j + INTERNAL_VEC_LEN], i_odd);
        }
    }
#else
    for (uint32_t tail = intmin(INTERNAL_VEC_LEN, N); tail > 1; tail >>= 1) {
        uint32_t half_step = tail >> 1;
        for (uint32_t i = 0; i < N; i += tail) {
            for (uint32_t j = 0; j < half_step; j++) {
                FLOAT re = real_signal[i + j], ie = imag_signal[i + j];
                FLOAT ro = real_signal[i + j + half_step], io = imag_signal[i + j + half_step];

                real_signal[i + j] = ADD(re, ro);
                imag_signal[i + j] = ADD(ie, io);

                FLOAT rt = SUB(re, ro), it = SUB(ie, io);
                real_signal[i + j + half_step] = SUB(MUL(rt, real_buffer[shift + j]), MUL(it, imag_buffer[shift + j]));
                imag_signal[i + j + half_step] = ADD(MUL(rt, imag_buffer[shift + j]), MUL(it, real_buffer[shift + j]));
            }
        }
        shift += half_step;
    }
#endif
}

/* Twiddles of one fused pass: per stage t the (m, j) grid described above. */
static void generate_fused_pass(uint32_t step, uint32_t shift, uint32_t layers, FLOAT *real_buffer, FLOAT *imag_buffer) {
    const uint32_t q = step >> layers;
    uint32_t base = shift;
    for (uint32_t t = 0; t < layers; ++t) {
        const uint32_t half = step >> (t + 1);
        const uint32_t groups = half / q; /* 2^(layers-1-t) */
        const uint32_t mult = 1u << t;
        for (uint32_t m = 0; m < groups; ++m) {
            for (uint32_t j = 0; j < q; ++j) {
#if defined(DOUBLE_DOUBLE)
                dd_t angle = dd_div(dd_make(-(double)(mult * (j + m * q)), 0.0), dd_make((double)step, 0.0));
#else
                double angle = -1.0 * (double)(mult * (j + m * q)) / (double)step;
#endif
                real_buffer[base + m * q + j] = M_COS2PI(angle);
                imag_buffer[base + m * q + j] = NEG(M_SIN2PI(angle));
            }
        }
        base += half;
    }
}

static void generate_buffer(uint32_t N, FLOAT *real_buffer, FLOAT *imag_buffer, int fuse) {
    const uint32_t log_n = intlog2(N);
    const uint32_t log_v = (INTERNAL_VEC_LEN > 1) ? intlog2((uint32_t)INTERNAL_VEC_LEN) : 0;
    const uint32_t stages = (log_n > log_v) ? log_n - log_v : 0;
    const uint32_t layers = (uint32_t)fuse;
    uint32_t shift = 0;
    uint32_t step = N;

    for (uint32_t pass = 0; pass < stages / layers; ++pass) {
        generate_fused_pass(step, shift, layers, real_buffer, imag_buffer);
        shift += step - (step >> layers);
        step >>= layers;
    }

    const uint32_t remainder = stages % layers;
    if (remainder) generate_fused_pass(step, shift, remainder, real_buffer, imag_buffer);

    /* The remaining stages live inside one vector and use compile-time
     * twiddles; nothing else is read from the table. */
}

/* =========================================================================
 * Public API Implementation
 * ========================================================================= */

PLAN_T *PREFIX(make_plan)(uint32_t N) {
    if (!is_power_of_two(N)) {
        fprintf(stderr, "Signal length must be a power of 2\n");
        exit(EXIT_FAILURE);
    }

    PLAN_T *plan = (PLAN_T *)malloc(sizeof(PLAN_T));
    plan->N = N;
    plan->fuse = NANOFFT_EFFECTIVE_FUSE(tls_tuning_fft_fuse());
    plan->cobravo = tls_tuning_cobravo();
    plan->pairs = NULL;
    plan->npairs = 0;
    {
        const uint32_t order = intlog2(N);
        int mode = tls_tuning_permute();
        if (mode == TLS_PERMUTE_AUTO) {
            mode = (order < PAIRS_MIN_LOG) ? TLS_PERMUTE_TABLE
                                           : ((order <= PAIRS_MAX_LOG) ? TLS_PERMUTE_PAIRS : TLS_PERMUTE_COBRAVO);
        }
        /* The pair table costs 4N bytes: keep a forced request from allocating
         * it for a whole monolithic grid, and keep a forced tiled reversal away
         * from tile widths the vectorized emitter cannot address. */
        if (mode == TLS_PERMUTE_PAIRS && order > PAIRS_MAX_LOG + 3u) mode = TLS_PERMUTE_COBRAVO;
        if (mode == TLS_PERMUTE_COBRAVO && order < COBRAVO_MIN_LOG) mode = TLS_PERMUTE_TABLE;
        plan->permute = mode;
    }

    plan->twiddle_real = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));
    plan->twiddle_imag = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));
    generate_buffer(N, plan->twiddle_real, plan->twiddle_imag, plan->fuse);

    size_t cobra_size = 2u * BLOCK_ELEMS * sizeof(FLOAT);
    plan->cobra_buffer_real = (FLOAT *)aligned_alloc(64, cobra_size);
    plan->cobra_buffer_imag = (FLOAT *)aligned_alloc(64, cobra_size);

    return plan;
}

void PREFIX(destroy_plan)(PLAN_T *plan) {
    if (plan) {
        free(plan->twiddle_real);
        free(plan->twiddle_imag);
        free(plan->cobra_buffer_real);
        free(plan->cobra_buffer_imag);
        free(plan->pairs);
        free(plan);
    }
}

void PREFIX(execute_raw)(PLAN_T *plan, FLOAT *real_signal, FLOAT *imag_signal) {
    sande_tukey_in_place(real_signal, imag_signal, plan->twiddle_real, plan->twiddle_imag, plan->N, plan->fuse);
}

void PREFIX(permute)(PLAN_T *plan, FLOAT *real_signal, FLOAT *imag_signal) { bit_reverse_permutation(real_signal, imag_signal, plan->N, plan); }

void PREFIX(permute_1d)(PLAN_T *plan, FLOAT *values) { bit_reverse_permutation_1d(values, plan->N, plan); }

void PREFIX(execute)(PLAN_T *plan, FLOAT *real_signal, FLOAT *imag_signal) {
    PREFIX(execute_raw)(plan, real_signal, imag_signal);
    PREFIX(permute)(plan, real_signal, imag_signal);
}

/* =========================================================================
 * Weak Fallback Mechanisms
 * ========================================================================= */

// Macro stringification helpers
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

__attribute__((weak)) FALLBACK_PLAN_T *FALLBACK_PREFIX(make_plan)(uint32_t N) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "[nanofft warning] %s is missing. Falling back to %s.\n", STR(FALLBACK_PREFIX(make_plan)), STR(PREFIX(make_plan)));
        warned = true;
    }
    return (FALLBACK_PLAN_T *)PREFIX(make_plan)(N);
}

__attribute__((weak)) void FALLBACK_PREFIX(destroy_plan)(FALLBACK_PLAN_T *plan) { PREFIX(destroy_plan)((PLAN_T *)plan); }

__attribute__((weak)) void FALLBACK_PREFIX(execute_raw)(FALLBACK_PLAN_T *plan, FALLBACK_FLOAT *real, FALLBACK_FLOAT *imag) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr,
                "[nanofft warning] %s is missing. Falling back to %s. "
                "Expect severe performance degradation due to array casting!\n",
                STR(FALLBACK_PREFIX(execute_raw)), STR(PREFIX(execute_raw)));
        warned = true;
    }

    uint32_t N = ((PLAN_T *)plan)->N;

    FLOAT *temp_r = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));
    FLOAT *temp_i = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));

    for (uint32_t i = 0; i < N; ++i) {
        temp_r[i] = FROM_FALLBACK(real[i]);
        temp_i[i] = FROM_FALLBACK(imag[i]);
    }

    PREFIX(execute_raw)((PLAN_T *)plan, temp_r, temp_i);

    for (uint32_t i = 0; i < N; ++i) {
        real[i] = TO_FALLBACK(temp_r[i]);
        imag[i] = TO_FALLBACK(temp_i[i]);
    }

    free(temp_r);
    free(temp_i);
}

__attribute__((weak)) void FALLBACK_PREFIX(execute)(FALLBACK_PLAN_T *plan, FALLBACK_FLOAT *real, FALLBACK_FLOAT *imag) {
    FALLBACK_PREFIX(execute_raw)(plan, real, imag);
    FALLBACK_PREFIX(permute)(plan, real, imag);
}

__attribute__((weak)) void FALLBACK_PREFIX(permute)(FALLBACK_PLAN_T *plan, FALLBACK_FLOAT *real, FALLBACK_FLOAT *imag) {
    uint32_t N = ((PLAN_T *)plan)->N;
    FLOAT *temp_r = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));
    FLOAT *temp_i = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));

    for (uint32_t i = 0; i < N; ++i) {
        temp_r[i] = FROM_FALLBACK(real[i]);
        temp_i[i] = FROM_FALLBACK(imag[i]);
    }

    PREFIX(permute)((PLAN_T *)plan, temp_r, temp_i);

    for (uint32_t i = 0; i < N; ++i) {
        real[i] = TO_FALLBACK(temp_r[i]);
        imag[i] = TO_FALLBACK(temp_i[i]);
    }

    free(temp_r);
    free(temp_i);
}

__attribute__((weak)) void FALLBACK_PREFIX(permute_1d)(FALLBACK_PLAN_T *plan, FALLBACK_FLOAT *values) {
    uint32_t N = ((PLAN_T *)plan)->N;
    FLOAT *temp = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));

    for (uint32_t i = 0; i < N; ++i) temp[i] = FROM_FALLBACK(values[i]);
    PREFIX(permute_1d)((PLAN_T *)plan, temp);
    for (uint32_t i = 0; i < N; ++i) values[i] = TO_FALLBACK(temp[i]);

    free(temp);
}
