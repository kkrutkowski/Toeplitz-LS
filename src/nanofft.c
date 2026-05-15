#include <math.h>
#include <nanofft.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Triple-Precision Macro Engine
 * ========================================================================= */
#define NANOFFT_NEEDS_INTERNAL_VEC
#include "nanofft_precision.h"
#undef NANOFFT_NEEDS_INTERNAL_VEC

#define PLAN_T PREFIX(plan)
#define FALLBACK_PLAN_T FALLBACK_PREFIX(plan)

struct PLAN_T {
    uint32_t N;
    FLOAT *twiddle_real;
    FLOAT *twiddle_imag;
    FLOAT *cobra_buffer_real;
    FLOAT *cobra_buffer_imag;
};

/* =========================================================================
 * Internal scalar helpers
 * ========================================================================= */

static inline bool is_power_of_two(int N) { return (N > 0) && ((N & (N - 1)) == 0); }
static inline uint32_t intmin(uint32_t a, uint32_t b) { return (a < b) ? a : b; }
static inline uint32_t intmax(int32_t a, int32_t b) { return (a > b) ? a : b; }

#define LOG_BLOCK_WIDTH 6
#define BLOCK_WIDTH (1 << LOG_BLOCK_WIDTH)

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

static inline void cobra_shuffle(FLOAT *real, FLOAT *imag, uint32_t log_n, FLOAT *buffer_real, FLOAT *buffer_imag) {
    uint32_t num_b_bits = log_n - 2 * LOG_BLOCK_WIDTH;
    uint32_t b_size = 1 << num_b_bits;
    uint32_t shift_top = num_b_bits + LOG_BLOCK_WIDTH;

    uint32_t rev6[BLOCK_WIDTH];
    for (uint32_t i = 0; i < BLOCK_WIDTH; ++i) {
        rev6[i] = bit_reverse_table[i] >> 2;
    }

    for (uint32_t b = 0; b < b_size; ++b) {
        uint32_t b_rev = reverse_bits(b, num_b_bits);
        uint32_t b_shifted = b << LOG_BLOCK_WIDTH;
        uint32_t b_rev_shifted = b_rev << LOG_BLOCK_WIDTH;

        for (uint32_t a = 0; a < BLOCK_WIDTH; ++a) {
            uint32_t a_rev = rev6[a];
            uint32_t a_shifted = a << shift_top;
            uint32_t a_rev_shifted = a_rev << LOG_BLOCK_WIDTH;

            for (uint32_t c = 0; c < BLOCK_WIDTH; ++c) {
                uint32_t idx = a_shifted | b_shifted | c;
                uint32_t buffer_idx = a_rev_shifted | c;

                buffer_real[buffer_idx] = real[idx];
                buffer_imag[buffer_idx] = imag[idx];
            }
        }

        for (uint32_t c = 0; c < BLOCK_WIDTH; ++c) {
            uint32_t c_rev = rev6[c];
            uint32_t c_rev_shifted = c_rev << shift_top;

            for (uint32_t a_rev = 0; a_rev < BLOCK_WIDTH; ++a_rev) {
                uint32_t a = rev6[a_rev];
                bool index_less_than_reverse = (a < c_rev) || (a == c_rev && b < b_rev) || (a == c_rev && b == b_rev && a_rev < c);

                if (index_less_than_reverse) {
                    uint32_t v_idx = c_rev_shifted | b_rev_shifted | a_rev;
                    uint32_t b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;

                    FLOAT temp_real = real[v_idx];
                    FLOAT temp_imag = imag[v_idx];

                    real[v_idx] = buffer_real[b_idx];
                    imag[v_idx] = buffer_imag[b_idx];

                    buffer_real[b_idx] = temp_real;
                    buffer_imag[b_idx] = temp_imag;
                }
            }
        }

        for (uint32_t a = 0; a < BLOCK_WIDTH; ++a) {
            uint32_t a_rev = rev6[a];
            uint32_t a_shifted = a << shift_top;

            for (uint32_t c = 0; c < BLOCK_WIDTH; ++c) {
                uint32_t c_rev = rev6[c];
                bool index_less_than_reverse = (a < c_rev) || (a == c_rev && b < b_rev) || (a == c_rev && b == b_rev && a_rev < c);

                if (index_less_than_reverse) {
                    uint32_t v_idx = a_shifted | b_shifted | c;
                    uint32_t b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;

                    FLOAT temp_real = real[v_idx];
                    FLOAT temp_imag = imag[v_idx];

                    real[v_idx] = buffer_real[b_idx];
                    imag[v_idx] = buffer_imag[b_idx];

                    buffer_real[b_idx] = temp_real;
                    buffer_imag[b_idx] = temp_imag;
                }
            }
        }
    }
}

inline static void bit_reverse_permutation(FLOAT *real, FLOAT *imag, uint32_t N, FLOAT *buffer_real, FLOAT *buffer_imag) {
    uint32_t order = intlog2(N);
    if (order <= 2 * LOG_BLOCK_WIDTH) {
        table_shuffle(real, imag, order);
    } else {
        cobra_shuffle(real, imag, order, buffer_real, buffer_imag);
    }
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

static void sande_tukey_in_place(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, uint32_t N) {
    uint32_t shift = 0;

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

#ifdef HAS_INTRA_VEC_PASS
    for (uint32_t i = intmax(0, (int32_t)intlog2(INTERNAL_VEC_LEN) - (int32_t)intlog2(N)); i < (uint32_t)intlog2(INTERNAL_VEC_LEN); i++) {
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
    for (uint32_t step = intmin(INTERNAL_VEC_LEN, N); step > 1; step >>= 1) {
        uint32_t half_step = step >> 1;
        for (uint32_t i = 0; i < N; i += step) {
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

static void generate_buffer(uint32_t N, FLOAT *real_buffer, FLOAT *imag_buffer) {
    uint32_t shift = 0;
    for (uint32_t step = N; step > 1; step >>= 1) {
        uint32_t half_step = step >> 1;
        for (uint32_t j = 0; j < half_step; j++) {
#if defined(DOUBLE_DOUBLE)
            dd_t angle = dd_div(dd_make(-(double)j, 0.0), dd_make((double)step, 0.0));
#else
            double angle = -1.0 * j / step;
#endif
            real_buffer[shift + j] = M_COS2PI(angle);
            imag_buffer[shift + j] = NEG(M_SIN2PI(angle));
        }
        shift += half_step;
    }
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

    plan->twiddle_real = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));
    plan->twiddle_imag = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));
    generate_buffer(N, plan->twiddle_real, plan->twiddle_imag);

    size_t cobra_size = BLOCK_WIDTH * BLOCK_WIDTH * sizeof(FLOAT);
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
        free(plan);
    }
}

void PREFIX(execute)(PLAN_T *plan, FLOAT *real_signal, FLOAT *imag_signal) {
    sande_tukey_in_place(real_signal, imag_signal, plan->twiddle_real, plan->twiddle_imag, plan->N);
    bit_reverse_permutation(real_signal, imag_signal, plan->N, plan->cobra_buffer_real, plan->cobra_buffer_imag);
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

__attribute__((weak)) void FALLBACK_PREFIX(execute)(FALLBACK_PLAN_T *plan, FALLBACK_FLOAT *real, FALLBACK_FLOAT *imag) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr,
                "[nanofft warning] %s is missing. Falling back to %s. "
                "Expect severe performance degradation due to array casting!\n",
                STR(FALLBACK_PREFIX(execute)), STR(PREFIX(execute)));
        warned = true;
    }

    uint32_t N = ((PLAN_T *)plan)->N;

    FLOAT *temp_r = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));
    FLOAT *temp_i = (FLOAT *)aligned_alloc(64, N * sizeof(FLOAT));

    for (uint32_t i = 0; i < N; ++i) {
        temp_r[i] = FROM_FALLBACK(real[i]);
        temp_i[i] = FROM_FALLBACK(imag[i]);
    }

    PREFIX(execute)((PLAN_T *)plan, temp_r, temp_i);

    for (uint32_t i = 0; i < N; ++i) {
        real[i] = TO_FALLBACK(temp_r[i]);
        imag[i] = TO_FALLBACK(temp_i[i]);
    }

    free(temp_r);
    free(temp_i);
}
