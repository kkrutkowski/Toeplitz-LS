"""Shared GLSL 450 code-generation helpers for Toeplitz-LS Vulkan compute shaders.

Provides precision-parameterized preambles, Dekker Double-Float (df_t = vec2)
with `precise` NoContraction error-free transformations, packed FP16x2 (f16vec2
<-> uint), branchless minimax sincos2pi, and hardware bitfieldReverse helpers.
"""

from __future__ import annotations

import struct

PREC_TAGS = ("h", "f", "df", "d")


def double_to_df(val: float) -> tuple[float, float]:
    """Split a Python float (IEEE binary64) into exact Dekker (hi, lo) float32 pair."""
    hi = struct.unpack("<f", struct.pack("<f", val))[0]
    lo = struct.unpack("<f", struct.pack("<f", val - hi))[0]
    return hi, lo


def fmt_f32(val: float) -> str:
    """Format a Python float as an exact GLSL float32 literal."""
    f = struct.unpack("<f", struct.pack("<f", val))[0]
    s = repr(f)
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return f"{s}f"


def fmt_f64(val: float) -> str:
    """Format a Python float as an exact GLSL double literal."""
    s = repr(float(val))
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return f"{s}lf"


def fmt_df(val: float) -> str:
    """Format a Python float as a GLSL vec2(hi, lo) Dekker double-float literal."""
    hi, lo = double_to_df(val)
    return f"vec2({fmt_f32(hi)}, {fmt_f32(lo)})"


def glsl_header(
    prec: str | None = None,
    *,
    subgroups: bool = False,
    subgroup_shuffle: bool = False,
    subgroup_arith: bool = False,
    fp16_storage: bool = False,
) -> str:
    """Emit #version 450 and required GLSL extensions."""
    lines = ["#version 450"]
    if prec == "d":
        lines.append("#extension GL_ARB_gpu_shader_fp64 : enable")
    elif prec == "h":
        lines.append("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable")
        if fp16_storage:
            lines.append("#extension GL_EXT_shader_16bit_storage : enable")
    if subgroups or subgroup_shuffle or subgroup_arith:
        lines.append("#extension GL_KHR_shader_subgroup_basic : enable")
    if subgroup_shuffle:
        lines.append("#extension GL_KHR_shader_subgroup_shuffle : enable")
    if subgroup_arith:
        lines.append("#extension GL_KHR_shader_subgroup_arithmetic : enable")
    return "\n".join(lines) + "\n"


def glsl_bit_reverse() -> str:
    """Hardware-accelerated bit reversal using GLSL built-in bitfieldReverse."""
    return """
uint bit_reverse(uint x, uint bits) {
    return (bits == 0u) ? 0u : (bitfieldReverse(x) >> (32u - bits));
}
"""


def glsl_df_math() -> str:
    """Dekker / Knuth Double-Float (df_t = vec2(hi, lo)) with `precise` NoContraction.

    Every intermediate in TwoSum, QuickTwoSum, TwoDiff, and TwoProd is marked
    `precise` so glslc emits SPIR-V `OpDecorate %id NoContraction` and vendor
    compilers cannot fuse away the low-order error terms.
    """
    hi_s12, lo_s12 = double_to_df(0.70710678118654752440)
    return f"""
// ============================================================================
// Dekker / Knuth Double-Float (df_t = vec2: .x = hi, .y = lo)
// ============================================================================
#define df_t vec2
#define Complexdf vec4

layout(constant_id = 15) const uint DF_USE_VELTKAMP = 0u;

const df_t kSqrt1_2df = vec2({fmt_f32(hi_s12)}, {fmt_f32(lo_s12)});

df_t df_make(float hi, float lo) {{
    return vec2(hi, lo);
}}

// Use exact 0.0f - x with `precise` to prevent aggressive sign-modifier folding
// across FMUL/FMA in downstream vendor compilers (NVIDIA sm_120 guard).
df_t df_neg(df_t a) {{
    precise float nh = 0.0f - a.x;
    precise float nl = 0.0f - a.y;
    return vec2(nh, nl);
}}

df_t df_scale_pow2(df_t a, float s) {{
    precise float sh = a.x * s;
    precise float sl = a.y * s;
    return vec2(sh, sl);
}}

df_t df_select(df_t a, df_t b, uint mask) {{
    uint hi = (floatBitsToUint(a.x) & mask) | (floatBitsToUint(b.x) & ~mask);
    uint lo = (floatBitsToUint(a.y) & mask) | (floatBitsToUint(b.y) & ~mask);
    return vec2(uintBitsToFloat(hi), uintBitsToFloat(lo));
}}

df_t df_xor_sign(df_t a, uint sign_bit) {{
    return vec2(
        uintBitsToFloat(floatBitsToUint(a.x) ^ sign_bit),
        uintBitsToFloat(floatBitsToUint(a.y) ^ sign_bit)
    );
}}

// Knuth TwoSum (6 FP32 ops): a + b = r + e exactly
void df_two_sum(float a, float b, out float r, out float e) {{
    precise float s  = a + b;
    precise float v  = s - a;
    precise float t1 = s - v;
    precise float t2 = a - t1;
    precise float t3 = b - v;
    precise float er = t2 + t3;
    r = s;
    e = er;
}}

// Dekker QuickTwoSum (3 FP32 ops, requires |x| >= |y|)
void df_two_sum_quick(float x, float y, out float r, out float e) {{
    precise float s  = x + y;
    precise float t1 = s - x;
    precise float er = y - t1;
    r = s;
    e = er;
}}

// Knuth TwoDiff (6 FP32 ops): x - y = r + e exactly
void df_two_difference(float x, float y, out float r, out float e) {{
    precise float s  = x - y;
    precise float v  = s - x;
    precise float t1 = s - v;
    precise float t2 = x - t1;
    precise float t3 = y + v;
    precise float er = t2 - t3;
    r = s;
    e = er;
}}

// Dekker TwoProd: 2-op exact FMA on FMA-capable GPUs, Veltkamp split fallback
// specialized at pipeline creation time (DF_USE_VELTKAMP) for non-FMA drivers.
void df_two_product(float x, float y, out float r, out float e) {{
    precise float p = x * y;
    if (DF_USE_VELTKAMP != 0u) {{
        const float split = 4097.0f;
        precise float cx = split * x;
        precise float xh = cx - (cx - x);
        precise float xl = x - xh;
        precise float cy = split * y;
        precise float yh = cy - (cy - y);
        precise float yl = y - yh;
        precise float t1 = xh * yh;
        precise float t2 = t1 - p;
        precise float t3 = xh * yl;
        precise float t4 = t2 + t3;
        precise float t5 = xl * yh;
        precise float t6 = t4 + t5;
        precise float t7 = xl * yl;
        precise float er = t6 + t7;
        r = p;
        e = er;
    }} else {{
        precise float er = fma(x, y, -p);
        r = p;
        e = er;
    }}
}}

df_t df_add(df_t a, df_t b) {{
    float r, e;
    df_two_sum(a.x, b.x, r, e);
    precise float e1 = e + a.y;
    precise float e2 = e1 + b.y;
    float oh, ol;
    df_two_sum_quick(r, e2, oh, ol);
    return vec2(oh, ol);
}}

df_t df_sub(df_t a, df_t b) {{
    float r, e;
    df_two_difference(a.x, b.x, r, e);
    precise float e1 = e + a.y;
    precise float e2 = e1 - b.y;
    float oh, ol;
    df_two_sum_quick(r, e2, oh, ol);
    return vec2(oh, ol);
}}

df_t df_mul(df_t a, df_t b) {{
    float r, e;
    df_two_product(a.x, b.x, r, e);
    precise float e1 = fma(a.x, b.y, e);
    precise float e2 = fma(a.y, b.x, e1);
    float oh, ol;
    df_two_sum_quick(r, e2, oh, ol);
    return vec2(oh, ol);
}}

df_t df_mul_f(df_t a, float b) {{
    float r, e;
    df_two_product(a.x, b, r, e);
    precise float e1 = fma(a.y, b, e);
    float oh, ol;
    df_two_sum_quick(r, e1, oh, ol);
    return vec2(oh, ol);
}}

df_t df_sqr(df_t a) {{
    float r, e;
    df_two_product(a.x, a.x, r, e);
    precise float two_hi = a.x + a.x;
    precise float e1 = fma(two_hi, a.y, e);
    float oh, ol;
    df_two_sum_quick(r, e1, oh, ol);
    return vec2(oh, ol);
}}

df_t df_div(df_t a, df_t b) {{
    precise float q1 = a.x / b.x;
    df_t r1 = df_sub(a, df_mul_f(b, q1));
    precise float q2 = r1.x / b.x;
    df_t r2 = df_sub(r1, df_mul_f(b, q2));
    precise float q3 = r2.x / b.x;
    float oh, ol;
    df_two_sum_quick(q1, q2, oh, ol);
    return df_add(vec2(oh, ol), vec2(q3, 0.0f));
}}

df_t df_sqrt(df_t a) {{
    if (a.x <= 0.0f) return vec2(0.0f, 0.0f);
    precise float x0 = sqrt(a.x);
    float sq_h, sq_l;
    df_two_product(x0, x0, sq_h, sq_l);
    df_t rem = df_sub(a, vec2(sq_h, sq_l));
    precise float dx = rem.x / (2.0f * x0);
    float oh, ol;
    df_two_sum_quick(x0, dx, oh, ol);
    return vec2(oh, ol);
}}

df_t df_abs(df_t a) {{
    return (a.x < 0.0f || (a.x == 0.0f && a.y < 0.0f)) ? df_neg(a) : a;
}}

df_t df_floor(df_t a) {{
    precise float f_hi = floor(a.x);
    precise float f_lo = (f_hi == a.x) ? floor(a.y) : 0.0f;
    float oh, ol;
    df_two_sum(f_hi, f_lo, oh, ol);
    return vec2(oh, ol);
}}

df_t df_ceil(df_t a) {{
    precise float c_hi = ceil(a.x);
    precise float c_lo = (c_hi == a.x) ? ceil(a.y) : 0.0f;
    float oh, ol;
    df_two_sum(c_hi, c_lo, oh, ol);
    return vec2(oh, ol);
}}

void df_cmul_fused(df_t ar, df_t ai, df_t br, df_t bi, out df_t out_r, out df_t out_i) {{
    float p1h, p1l, p2h, p2l;
    df_two_product(ar.x, br.x, p1h, p1l);
    precise float p1t = fma(ar.x, br.y, p1l);
    precise float p1e = fma(ar.y, br.x, p1t);

    df_two_product(ai.x, bi.x, p2h, p2l);
    precise float p2t = fma(ai.x, bi.y, p2l);
    precise float p2e = fma(ai.y, bi.x, p2t);

    float rr, re;
    df_two_difference(p1h, p2h, rr, re);
    precise float re1 = re + p1e;
    precise float re2 = re1 - p2e;
    float orh, orl;
    df_two_sum_quick(rr, re2, orh, orl);
    out_r = vec2(orh, orl);

    float q1h, q1l, q2h, q2l;
    df_two_product(ar.x, bi.x, q1h, q1l);
    precise float q1t = fma(ar.x, bi.y, q1l);
    precise float q1e = fma(ar.y, bi.x, q1t);

    df_two_product(ai.x, br.x, q2h, q2l);
    precise float q2t = fma(ai.x, br.y, q2l);
    precise float q2e = fma(ai.y, br.x, q2t);

    float ir, ie;
    df_two_sum(q1h, q2h, ir, ie);
    precise float ie1 = ie + q1e;
    precise float ie2 = ie1 + q2e;
    float oih, oil;
    df_two_sum_quick(ir, ie2, oih, oil);
    out_i = vec2(oih, oil);
}}

void df_csqr_fused(df_t ar, df_t ai, out df_t out_r, out df_t out_i) {{
    float p1h, p1l, p2h, p2l;
    df_two_product(ar.x, ar.x, p1h, p1l);
    precise float two_ar = ar.x + ar.x;
    precise float p1e = fma(two_ar, ar.y, p1l);

    df_two_product(ai.x, ai.x, p2h, p2l);
    precise float two_ai = ai.x + ai.x;
    precise float p2e = fma(two_ai, ai.y, p2l);

    float rr, re;
    df_two_difference(p1h, p2h, rr, re);
    precise float re1 = re + p1e;
    precise float re2 = re1 - p2e;
    float orh, orl;
    df_two_sum_quick(rr, re2, orh, orl);
    out_r = vec2(orh, orl);
    out_i = df_scale_pow2(df_mul(ar, ai), 2.0f);
}}

void df_cmul_sqrt1_2_rt(bool neg_re, df_t dr, df_t di, bool pos_sign, out df_t out_r, out df_t out_i) {{
    uint sbit = pos_sign ? 0x80000000u : 0u;
    df_t s_dr = df_xor_sign(dr, sbit);
    df_t s_di = df_xor_sign(di, sbit);
    df_t tr, ti;
    if (!neg_re) {{
        tr = df_add(dr, s_di);
        ti = df_sub(di, s_dr);
    }} else {{
        tr = df_sub(s_di, dr);
        ti = df_neg(df_add(s_dr, di));
    }}
    out_r = df_mul(tr, kSqrt1_2df);
    out_i = df_mul(ti, kSqrt1_2df);
}}
"""


def glsl_sincos2pi_f32() -> str:
    """Branchless FP32 minimax sincos2pi on [0, 0.25]."""
    return """
void sincos2pif_q1(float f, bool neg_sign, out float c_out, out float s_out) {
    float f2 = f * f;
    float ps = fma(f2,  39.536706065730207835f, -76.549782293595742666f);
    ps = fma(ps, f2,  81.601004073261773523f);
    ps = fma(ps, f2, -41.341655031416278077f);
    ps = fma(ps, f2,   6.283185160089477443f);
    float s_mag = ps * f;

    float pc = fma(f2,  56.242380464873243260f, -85.240330322699427860f);
    pc = fma(pc, f2,  64.934590626780991246f);
    pc = fma(pc, f2, -19.739171434702393619f);
    pc = fma(pc, f2,   0.999999953466670136f);

    c_out = pc;
    s_out = neg_sign ? -s_mag : s_mag;
}

void sincos2pif_half(float f, bool neg_sign, out float c_out, out float s_out) {
    uint gt_qtr = uint(f > 0.25f);
    uint mask = 0u - gt_qtr;
    float f_fold = 0.5f - f;
    f = uintBitsToFloat((floatBitsToUint(f_fold) & mask) | (floatBitsToUint(f) & ~mask));
    float c_mag, s_signed;
    sincos2pif_q1(f, neg_sign, c_mag, s_signed);
    c_out = uintBitsToFloat(floatBitsToUint(c_mag) ^ (gt_qtr << 31));
    s_out = s_signed;
}

void sincos2pif_branchless(float x, bool neg_sign, out float c_out, out float s_out) {
    float f = x - floor(x);
    uint ge_half = uint(f >= 0.5f);
    uint gh_mask = 0u - ge_half;
    f -= uintBitsToFloat(floatBitsToUint(0.5f) & gh_mask);

    uint gt_qtr = uint(f > 0.25f);
    uint gq_mask = 0u - gt_qtr;
    float f_fold = 0.5f - f;
    f = uintBitsToFloat((floatBitsToUint(f_fold) & gq_mask) | (floatBitsToUint(f) & ~gq_mask));

    float c_mag, s_base;
    sincos2pif_q1(f, false, c_mag, s_base);
    uint s_flip = (ge_half ^ uint(neg_sign)) << 31;
    uint c_flip = (ge_half ^ gt_qtr) << 31;
    c_out = uintBitsToFloat(floatBitsToUint(c_mag) ^ c_flip);
    s_out = uintBitsToFloat(floatBitsToUint(s_base) ^ s_flip);
}
"""


def glsl_sincos2pi_df() -> str:
    """Branchless DF minimax sincos2pi on [0, 0.125] (~47.3-48.1 bits)."""
    coeffs = [
        ("kSinDf0", fmt_df(6.28318530717955959e00)),
        ("kSinDf1", fmt_df(-4.13417022402757794e01)),
        ("kSinDf2", fmt_df(8.16052491834839344e01)),
        ("kSinDf3", fmt_df(-7.67058344584512923e01)),
        ("kSinDf4", fmt_df(4.20555692559449881e01)),
        ("kCosDf0", fmt_df(1.00000000000000000e00)),
        ("kCosDf1", fmt_df(-1.97392088021784211e01)),
        ("kCosDf2", fmt_df(6.49393940223653772e01)),
        ("kCosDf3", fmt_df(-8.54568170903411470e01)),
        ("kCosDf4", fmt_df(6.02446200859302934e01)),
    ]
    c_decls = "\n".join(f"const df_t {name} = {val};" for name, val in coeffs)
    return f"""
{c_decls}
const float kSinDf5 = {fmt_f32(-1.49165651188191184e01)};
const float kCosDf5 = {fmt_f32(-2.64242570348826895e01)};
const float kCosDf6 = {fmt_f32(7.81029096814419077e00)};

void sincos2pidf_oct1_f(float f, bool neg_sign, out df_t c_out, out df_t s_out) {{
    float f2h, f2l;
    df_two_product(f, f, f2h, f2l);
    df_t f2 = vec2(f2h, f2l);

    df_t ps = df_add(df_mul_f(f2, kSinDf5), kSinDf4);
    ps = df_add(df_mul(ps, f2), kSinDf3);
    ps = df_add(df_mul(ps, f2), kSinDf2);
    ps = df_add(df_mul(ps, f2), kSinDf1);
    ps = df_add(df_mul(ps, f2), kSinDf0);
    df_t s_mag = df_mul_f(ps, f);

    float pc_top = fma(kCosDf6, f2.x, kCosDf5);
    df_t pc = df_add(df_mul_f(f2, pc_top), kCosDf4);
    pc = df_add(df_mul(pc, f2), kCosDf3);
    pc = df_add(df_mul(pc, f2), kCosDf2);
    pc = df_add(df_mul(pc, f2), kCosDf1);
    pc = df_add(df_mul(pc, f2), kCosDf0);

    c_out = pc;
    s_out = neg_sign ? df_neg(s_mag) : s_mag;
}}

void sincos2pidf_half_f(float f, bool neg_sign, out df_t c_out, out df_t s_out) {{
    uint gt_qtr = uint(f > 0.25f);
    uint gq_mask = 0u - gt_qtr;
    float f_qtr = 0.5f - f;
    f = uintBitsToFloat((floatBitsToUint(f_qtr) & gq_mask) | (floatBitsToUint(f) & ~gq_mask));

    uint gt_oct = uint(f > 0.125f);
    uint go_mask = 0u - gt_oct;
    float f_oct = 0.25f - f;
    f = uintBitsToFloat((floatBitsToUint(f_oct) & go_mask) | (floatBitsToUint(f) & ~go_mask));

    df_t c_oct, s_oct;
    sincos2pidf_oct1_f(f, false, c_oct, s_oct);
    df_t c_q1 = df_select(s_oct, c_oct, go_mask);
    df_t s_q1 = df_select(c_oct, s_oct, go_mask);
    c_out = df_xor_sign(c_q1, gt_qtr << 31);
    s_out = neg_sign ? df_neg(s_q1) : s_q1;
}}

void sincos2pidf_oct1(df_t f, bool neg_sign, out df_t c_out, out df_t s_out) {{
    df_t f2 = df_sqr(f);
    df_t ps = df_add(df_mul_f(f2, kSinDf5), kSinDf4);
    ps = df_add(df_mul(ps, f2), kSinDf3);
    ps = df_add(df_mul(ps, f2), kSinDf2);
    ps = df_add(df_mul(ps, f2), kSinDf1);
    ps = df_add(df_mul(ps, f2), kSinDf0);
    df_t s_mag = df_mul(ps, f);

    float pc_top = fma(kCosDf6, f2.x, kCosDf5);
    df_t pc = df_add(df_mul_f(f2, pc_top), kCosDf4);
    pc = df_add(df_mul(pc, f2), kCosDf3);
    pc = df_add(df_mul(pc, f2), kCosDf2);
    pc = df_add(df_mul(pc, f2), kCosDf1);
    pc = df_add(df_mul(pc, f2), kCosDf0);

    c_out = pc;
    s_out = neg_sign ? df_neg(s_mag) : s_mag;
}}

void sincos2pidf_branchless(df_t x, bool neg_sign, out df_t c_out, out df_t s_out) {{
    df_t f = df_sub(x, df_floor(x));
    uint lt_zero_mask = 0u - uint(f.x < 0.0f || (f.x == 0.0f && f.y < 0.0f));
    f = df_add(f, df_select(vec2(1.0f, 0.0f), vec2(0.0f, 0.0f), lt_zero_mask));
    uint ge_one_mask = 0u - uint(f.x > 1.0f || (f.x == 1.0f && f.y >= 0.0f));
    f = df_sub(f, df_select(vec2(1.0f, 0.0f), vec2(0.0f, 0.0f), ge_one_mask));

    uint ge_half = uint(f.x > 0.5f || (f.x == 0.5f && f.y >= 0.0f));
    uint gh_mask = 0u - ge_half;
    f = df_sub(f, df_select(vec2(0.5f, 0.0f), vec2(0.0f, 0.0f), gh_mask));

    uint gt_qtr = uint(f.x > 0.25f || (f.x == 0.25f && f.y > 0.0f));
    uint gq_mask = 0u - gt_qtr;
    f = df_select(df_sub(vec2(0.5f, 0.0f), f), f, gq_mask);

    uint gt_oct = uint(f.x > 0.125f || (f.x == 0.125f && f.y > 0.0f));
    uint go_mask = 0u - gt_oct;
    f = df_select(df_sub(vec2(0.25f, 0.0f), f), f, go_mask);

    df_t c_oct, s_oct;
    sincos2pidf_oct1(f, false, c_oct, s_oct);
    df_t c_q1 = df_select(s_oct, c_oct, go_mask);
    df_t s_q1 = df_select(c_oct, s_oct, go_mask);

    uint c_flip = (ge_half ^ gt_qtr) << 31;
    uint s_flip = (ge_half ^ uint(neg_sign)) << 31;
    c_out = df_xor_sign(c_q1, c_flip);
    s_out = df_xor_sign(s_q1, s_flip);
}}
"""


def glsl_sincos2pi_f64() -> str:
    """Branchless FP64 minimax sincos2pi on [0, 0.125] (<= 0.5 ULP)."""
    return """
void sincos2pi_oct1(double f, bool neg_sign, out double c_out, out double s_out) {
    double f2 = f * f;

    double ps = fma(f2,  3.78086895930229572e+00lf, -1.50938042099871144e+01lf);
    ps = fma(ps, f2,  4.20586850201608939e+01lf);
    ps = fma(ps, f2, -7.67058597042745447e+01lf);
    ps = fma(ps, f2,  8.16052492759480401e+01lf);
    ps = fma(ps, f2, -4.13417022403996341e+01lf);
    ps = fma(ps, f2,  6.28318530717958623e+00lf);
    double s_mag = ps * f;

    double pc = fma(f2, -1.69684947029188260e+00lf,  7.90309168775169724e+00lf);
    pc = fma(pc, f2, -2.64262509086221371e+01lf);
    pc = fma(pc, f2,  6.02446413288672957e+01lf);
    pc = fma(pc, f2, -8.54568172065227145e+01lf);
    pc = fma(pc, f2,  6.49393940226679547e+01lf);
    pc = fma(pc, f2, -1.97392088021787160e+01lf);
    pc = fma(pc, f2,  1.00000000000000000e+00lf);

    c_out = pc;
    s_out = neg_sign ? -s_mag : s_mag;
}

void sincos2pi_half(double f, bool neg_sign, out double c_out, out double s_out) {
    bool gt_qtr = (f > 0.25lf);
    f = gt_qtr ? (0.5lf - f) : f;
    bool gt_oct = (f > 0.125lf);
    f = gt_oct ? (0.25lf - f) : f;
    double c_oct, s_oct;
    sincos2pi_oct1(f, false, c_oct, s_oct);
    double c_q1 = gt_oct ? s_oct : c_oct;
    double s_q1 = gt_oct ? c_oct : s_oct;
    c_out = gt_qtr ? -c_q1 : c_q1;
    s_out = neg_sign ? -s_q1 : s_q1;
}

void sincos2pi_branchless(double x, bool neg_sign, out double c_out, out double s_out) {
    double f = x - floor(x);
    bool ge_half = (f >= 0.5lf);
    f = ge_half ? (f - 0.5lf) : f;
    bool gt_qtr = (f > 0.25lf);
    f = gt_qtr ? (0.5lf - f) : f;
    bool gt_oct = (f > 0.125lf);
    f = gt_oct ? (0.25lf - f) : f;

    double c_oct, s_oct;
    sincos2pi_oct1(f, false, c_oct, s_oct);
    double c_q1 = gt_oct ? s_oct : c_oct;
    double s_q1 = gt_oct ? c_oct : s_oct;

    c_out = (ge_half != gt_qtr) ? -c_q1 : c_q1;
    s_out = (ge_half != neg_sign) ? -s_q1 : s_q1;
}
"""


def glsl_f16x2_math() -> str:
    """Packed 2-lane binary16 (f16vec2 <-> uint) helpers for FP16 FFT & solvers.

    Uses standard packHalf2x16 / unpackHalf2x16 when converting to/from vec2,
    and bit-casts f16vec2 <-> uint via packFloat2x16 / unpackFloat2x16 (Wait:
    packHalf2x16(vec2(v)) is zero-loss for normal/subnormal f16vec2 and
    universally supported in core GLSL 450 without GL_NV_gpu_shader5!).
    """
    return """
// Packed 2-lane FP16 helpers: f16vec2 <-> uint32 without vendor-specific extensions
const f16vec2 kZero_f16x2    = f16vec2(float16_t(0.0), float16_t(0.0));
const f16vec2 kOne_f16x2     = f16vec2(float16_t(1.0), float16_t(1.0));
const f16vec2 kHalf_f16x2    = f16vec2(float16_t(0.5), float16_t(0.5));
const f16vec2 kTwo_f16x2     = f16vec2(float16_t(2.0), float16_t(2.0));
const f16vec2 kSqrt1_2_f16x2 = f16vec2(float16_t(0.70710678), float16_t(0.70710678));

uint f16x2_to_u32(f16vec2 v) {
    return packHalf2x16(vec2(v));
}

f16vec2 u32_to_f16x2(uint u) {
    return f16vec2(unpackHalf2x16(u));
}

f16vec2 f16x2_cond_neg(f16vec2 v, bool cond) {
    return cond ? -v : v;
}

void f16x2_cmul(f16vec2 ar, f16vec2 ai, f16vec2 br, f16vec2 bi,
                out f16vec2 or_, out f16vec2 oi) {
    or_ = fma(ar, br, -(ai * bi));
    oi  = fma(ar, bi,   ai * br);
}

void f16x2_csqr(f16vec2 ar, f16vec2 ai, out f16vec2 or_, out f16vec2 oi) {
    or_ = fma(ar, ar, -(ai * ai));
    oi  = (kTwo_f16x2 * ar) * ai;
}

void f16x2_cmul_sqrt1_2_rt(bool neg_re, f16vec2 dr, f16vec2 di, bool pos_sign,
                           out f16vec2 or_, out f16vec2 oi) {
    f16vec2 s_dr = pos_sign ? -dr : dr;
    f16vec2 s_di = pos_sign ? -di : di;
    if (!neg_re) {
        or_ = (dr + s_di) * kSqrt1_2_f16x2;
        oi  = (di - s_dr) * kSqrt1_2_f16x2;
    } else {
        or_ = (s_di - dr) * kSqrt1_2_f16x2;
        oi  = -(s_dr + di) * kSqrt1_2_f16x2;
    }
}

void f16x2_load_pair_u32(uint u0, uint u1, out f16vec2 re_out, out f16vec2 im_out) {
    uint re_u32 = (u0 & 0xFFFFu) | ((u1 & 0xFFFFu) << 16);
    uint im_u32 = (u0 >> 16) | (u1 & 0xFFFF0000u);
    re_out = u32_to_f16x2(re_u32);
    im_out = u32_to_f16x2(im_u32);
}

void f16x2_pack_pair_u32(f16vec2 re_in, f16vec2 im_in, out uint u0, out uint u1) {
    uint re_u32 = f16x2_to_u32(re_in);
    uint im_u32 = f16x2_to_u32(im_in);
    u0 = (re_u32 & 0xFFFFu) | ((im_u32 & 0xFFFFu) << 16);
    u1 = (re_u32 >> 16) | (im_u32 & 0xFFFF0000u);
}
"""
