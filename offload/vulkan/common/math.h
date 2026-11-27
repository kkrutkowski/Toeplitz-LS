#ifndef TLS_VK_MATH_H
#define TLS_VK_MATH_H

/* Host-side plain C11 math types and binary16 / Dekker double-float converters. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

static inline uint32_t vk_bit_reverse(uint32_t x, int bits) {
    if (bits <= 0) return 0u;
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    x = (x << 16) | (x >> 16);
    return x >> (32 - bits);
}

static inline uint32_t vk_f32_to_u32(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static inline float vk_u32_to_f32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t vk_f32_to_f16_bits(float f) {
#if defined(__FLT16_MAX__) && !defined(_MSC_VER)
    _Float16 h = (_Float16)f;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
#else
    uint32_t u = vk_f32_to_u32(f);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t exp = (u >> 23) & 0xFFu;
    uint32_t mant = u & 0x007FFFFFu;
    if (exp == 0u) return (uint16_t)sign;
    if (exp == 255u) return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    int e = (int)exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        uint32_t m = mant | 0x00800000u;
        int shift = 14 - e;
        uint32_t round_bit = 1u << (shift - 1);
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t h = m >> shift;
        if (rem > round_bit || (rem == round_bit && (h & 1u))) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t rem = mant & 0x1FFFu;
    uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t)(sign | h);
#endif
}

static inline float vk_f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x03FFu;
    uint32_t u = 0;
    if (exp == 0u) {
        if (mant == 0u) {
            u = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 127u - 14u;
            while ((m & 0x0400u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            u = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    return vk_u32_to_f32(u);
}

typedef struct VkComplexh {
    uint16_t re;
    uint16_t im;
} VkComplexh;

typedef struct VkComplexf {
    float re;
    float im;
} VkComplexf;

typedef struct VkComplex {
    double re;
    double im;
} VkComplex;

typedef struct VkDf {
    float hi;
    float lo;
} VkDf;

typedef struct VkComplexdf {
    VkDf re;
    VkDf im;
} VkComplexdf;

_Static_assert(sizeof(VkComplexh) == 4, "VkComplexh must be 4 bytes");
_Static_assert(sizeof(VkComplexf) == 8, "VkComplexf must be 8 bytes");
_Static_assert(sizeof(VkComplex) == 16, "VkComplex must be 16 bytes");
_Static_assert(sizeof(VkDf) == 8, "VkDf must be 8 bytes");
_Static_assert(sizeof(VkComplexdf) == 16, "VkComplexdf must be 16 bytes");

static inline VkDf vk_df_make(float hi, float lo) {
    VkDf v = {hi, lo};
    return v;
}

static inline VkDf vk_df_from_double(double d) {
    float hi = (float)d;
    float lo = (float)(d - (double)hi);
    VkDf v = {hi, lo};
    return v;
}

static inline double vk_df_to_double(VkDf a) { return (double)a.hi + (double)a.lo; }

static inline VkComplexh vk_complexh_from_doubles(double r, double i) {
    VkComplexh c = {vk_f32_to_f16_bits((float)r), vk_f32_to_f16_bits((float)i)};
    return c;
}

#endif /* TLS_VK_MATH_H */
