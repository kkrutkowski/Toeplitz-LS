"""GLSL compute shader codegen for Quadruple-Precision DIF (Sande-Tukey) Radix-2 FFT.

Generates all proven SYCL DIF FFT schedules across 4 precisions (h, f, df, d):
  - 44 SinglePass kernels (logN 10..14 narrow h/f; logN 10..13 wide df/d)
    with SPIR-V specialization constants for BankModel strides, SMEM size,
    sign (+1/-1), compact-slot epilogue (full / PSWF21 / PSWF43), deconv,
    subgroup tile sync, and DF Veltkamp fallback.
  - 12 Pass-0 / MultiStageGlobal kernels (4 precisions x radices 8, 16, 32).
"""

from __future__ import annotations

import math
from common_glsl import (
    double_to_df,
    fmt_df,
    fmt_f32,
    fmt_f64,
    glsl_df_math,
    glsl_f16x2_math,
    glsl_header,
    glsl_sincos2pi_df,
    glsl_sincos2pi_f32,
    glsl_sincos2pi_f64,
)

# Exact W_32^k = exp(-i 2pi k / 32) for k = 0..15
K_COS32 = [math.cos(2.0 * math.pi * k / 32.0) for k in range(16)]
K_SIN32_NEG = [-math.sin(2.0 * math.pi * k / 32.0) for k in range(16)]

# Exact W_256^k = exp(-i 2pi k / 256) for k = 0..15
K_W256_RE = [math.cos(2.0 * math.pi * k / 256.0) for k in range(16)]
K_W256_IM_NEG = [-math.sin(2.0 * math.pi * k / 256.0) for k in range(16)]


# SinglePass schedule specifications:
#   (variant_id, logN, logE, logChunks, B, staging, t1_layout, phase3, stash_chunk1)
NARROW_SPECS = [
    ("split_10_e3_b2_aff",   10, 3, 0, 2, "split", "aff", "shuf",  False),
    ("split_10_e3_b1_aff",   10, 3, 0, 1, "split", "aff", "shuf",  False),
    ("split_10_e4_b1_aff",   10, 4, 0, 1, "split", "aff", "shuf",  False),
    ("seq_10_e4_b1_2tstr",   10, 4, 0, 1, "seq",   "aff", "2tstr", False),
    ("split_11_e3_b1_aff",   11, 3, 0, 1, "split", "aff", "shuf",  False),
    ("split_11_e4_b1_aff",   11, 4, 0, 1, "split", "aff", "shuf",  False),
    ("split_11_e4_b1_2tstr", 11, 4, 0, 1, "split", "aff", "2tstr", False),
    ("split_12_e4_b1_aff",   12, 4, 0, 1, "split", "aff", "shuf",  False),
    ("seq_12_e4_b1_sg",      12, 4, 0, 1, "seq",   "aff", "shuf",  False),
    ("seq_12_e4_b1_2txor",   12, 4, 0, 1, "seq",   "aff", "2txor", False),
    ("seq_13_e4_b1_sg",      13, 4, 0, 1, "seq",   "aff", "shuf",  False),
    ("seq_13_e5_c2_sg",      13, 5, 1, 1, "seq",   "aff", "shuf",  False),
    ("seq_13_e5_c2_2txor",   13, 5, 1, 1, "seq",   "aff", "2txor", False),
    ("seq_14_e5_c2_sg",      14, 5, 1, 1, "seq",   "aff", "shuf",  False),
]

WIDE_SPECS = [
    ("seq_10_e4_b1_2tstr",      10, 4, 0, 1, "seq",   "aff", "2tstr", False),
    ("split_10_e4_b1_xor",      10, 4, 0, 1, "split", "xor", "shuf",  False),
    ("wide_11_e4_b1_2tstr",     11, 4, 0, 1, "split", "aff", "2tstr", False),
    ("seq_11_e4_b1_sg",         11, 4, 0, 1, "seq",   "aff", "shuf",  False),
    ("wide_12_e4_b1_2txor",     12, 4, 0, 1, "seq",   "aff", "2txor", False),
    ("seq_12_e4_b1_sg",         12, 4, 0, 1, "seq",   "aff", "shuf",  False),
    ("wide_13_e5_c2_xor_stash", 13, 5, 1, 1, "seq",   "aff", "2txor", True),
    ("wide_13_e5_c2_xor_nostash", 13, 5, 1, 1, "seq", "aff", "2txor", False),
]


def _prec_helpers(prec: str, need_shuf: bool) -> str:
    """Emit precision-specific types, arithmetic macros/functions, and W_32/W_256 tables."""
    if prec == "f":
        c32 = ", ".join(fmt_f32(v) for v in K_COS32)
        s32 = ", ".join(fmt_f32(v) for v in K_SIN32_NEG)
        w256r = ", ".join(fmt_f32(v) for v in K_W256_RE)
        w256i = ", ".join(fmt_f32(v) for v in K_W256_IM_NEG)
        shuf = """
float fft_shuf_xor(float v, uint mask) { return subgroupShuffleXor(v, mask); }
""" if need_shuf else ""
        return (
            glsl_sincos2pi_f32()
            + f"""
#define REAL float
#define SMEM_T float
#define BUF_T vec2
#define DECONV_T float
#define ZERO_R 0.0f
#define ONE_R  1.0f
#define PACKED_BATCH 1u

const float kSqrt1_2f = {fmt_f32(math.sqrt(0.5))};
const float kCos32[16] = float[16]({c32});
const float kSin32Neg[16] = float[16]({s32});
const float kW256Re[16] = float[16]({w256r});
const float kW256ImNeg[16] = float[16]({w256i});

float fft_add(float a, float b) {{ return a + b; }}
float fft_sub(float a, float b) {{ return a - b; }}
float fft_mul(float a, float b) {{ return a * b; }}
float fft_half(float a) {{ return a * 0.5f; }}
float fft_cond_neg(float v, bool cond) {{
    return uintBitsToFloat(floatBitsToUint(v) ^ (cond ? 0x80000000u : 0u));
}}
void fft_cmul(float ar, float ai, float br, float bi, out float or_, out float oi) {{
    or_ = fma(ar, br, -(ai * bi));
    oi  = fma(ar, bi,   ai * br);
}}
void fft_csqr(float ar, float ai, out float or_, out float oi) {{
    or_ = fma(ar, ar, -(ai * ai));
    oi  = (2.0f * ar) * ai;
}}
void fft_cmul_sqrt1_2(bool neg_re, bool pos_sign, float dr, float di, out float or_, out float oi) {{
    uint sb = pos_sign ? 0x80000000u : 0u;
    float sdr = uintBitsToFloat(floatBitsToUint(dr) ^ sb);
    float sdi = uintBitsToFloat(floatBitsToUint(di) ^ sb);
    if (!neg_re) {{
        or_ = (dr + sdi) * kSqrt1_2f;
        oi  = (di - sdr) * kSqrt1_2f;
    }} else {{
        or_ = (sdi - dr) * kSqrt1_2f;
        oi  = -(sdr + di) * kSqrt1_2f;
    }}
}}
float to_smem(float r) {{ return r; }}
float from_smem(float s) {{ return s; }}
{shuf}
"""
        )

    if prec == "d":
        c32 = ", ".join(fmt_f64(v) for v in K_COS32)
        s32 = ", ".join(fmt_f64(v) for v in K_SIN32_NEG)
        w256r = ", ".join(fmt_f64(v) for v in K_W256_RE)
        w256i = ", ".join(fmt_f64(v) for v in K_W256_IM_NEG)
        shuf = """
double fft_shuf_xor(double v, uint mask) {
    uvec2 u = unpackDouble2x32(v);
    return packDouble2x32(uvec2(subgroupShuffleXor(u.x, mask), subgroupShuffleXor(u.y, mask)));
}
""" if need_shuf else ""
        return (
            glsl_sincos2pi_f64()
            + f"""
#define REAL double
#define SMEM_T double
#define BUF_T dvec2
#define DECONV_T double
#define ZERO_R 0.0lf
#define ONE_R  1.0lf
#define PACKED_BATCH 1u

const double kSqrt1_2 = {fmt_f64(math.sqrt(0.5))};
const double kCos32[16] = double[16]({c32});
const double kSin32Neg[16] = double[16]({s32});
const double kW256Re[16] = double[16]({w256r});
const double kW256ImNeg[16] = double[16]({w256i});

double fft_add(double a, double b) {{ return a + b; }}
double fft_sub(double a, double b) {{ return a - b; }}
double fft_mul(double a, double b) {{ return a * b; }}
double fft_half(double a) {{ return a * 0.5lf; }}
double fft_cond_neg(double v, bool cond) {{ return cond ? -v : v; }}
void fft_cmul(double ar, double ai, double br, double bi, out double or_, out double oi) {{
    or_ = fma(ar, br, -(ai * bi));
    oi  = fma(ar, bi,   ai * br);
}}
void fft_csqr(double ar, double ai, out double or_, out double oi) {{
    or_ = fma(ar, ar, -(ai * ai));
    oi  = (2.0lf * ar) * ai;
}}
void fft_cmul_sqrt1_2(bool neg_re, bool pos_sign, double dr, double di, out double or_, out double oi) {{
    double sdr = pos_sign ? -dr : dr;
    double sdi = pos_sign ? -di : di;
    if (!neg_re) {{
        or_ = (dr + sdi) * kSqrt1_2;
        oi  = (di - sdr) * kSqrt1_2;
    }} else {{
        or_ = (sdi - dr) * kSqrt1_2;
        oi  = -(sdr + di) * kSqrt1_2;
    }}
}}
double to_smem(double r) {{ return r; }}
double from_smem(double s) {{ return s; }}
{shuf}
"""
        )

    if prec == "df":
        c32 = ", ".join(fmt_df(v) for v in K_COS32)
        s32 = ", ".join(fmt_df(v) for v in K_SIN32_NEG)
        w256r = ", ".join(fmt_df(v) for v in K_W256_RE)
        w256i = ", ".join(fmt_df(v) for v in K_W256_IM_NEG)
        shuf = """
df_t fft_shuf_xor(df_t v, uint mask) {
    return vec2(subgroupShuffleXor(v.x, mask), subgroupShuffleXor(v.y, mask));
}
""" if need_shuf else ""
        return (
            glsl_df_math()
            + glsl_sincos2pi_df()
            + f"""
#define REAL df_t
#define SMEM_T df_t
#define BUF_T vec4
#define DECONV_T df_t
#define ZERO_R vec2(0.0f, 0.0f)
#define ONE_R  vec2(1.0f, 0.0f)
#define PACKED_BATCH 1u

const df_t kCos32[16] = df_t[16]({c32});
const df_t kSin32Neg[16] = df_t[16]({s32});
const df_t kW256Re[16] = df_t[16]({w256r});
const df_t kW256ImNeg[16] = df_t[16]({w256i});

df_t fft_add(df_t a, df_t b) {{ return df_add(a, b); }}
df_t fft_sub(df_t a, df_t b) {{ return df_sub(a, b); }}
df_t fft_mul(df_t a, df_t b) {{ return df_mul(a, b); }}
df_t fft_half(df_t a) {{ return df_scale_pow2(a, 0.5f); }}
df_t fft_cond_neg(df_t v, bool cond) {{ return df_xor_sign(v, cond ? 0x80000000u : 0u); }}
void fft_cmul(df_t ar, df_t ai, df_t br, df_t bi, out df_t or_, out df_t oi) {{ df_cmul_fused(ar, ai, br, bi, or_, oi); }}
void fft_csqr(df_t ar, df_t ai, out df_t or_, out df_t oi) {{ df_csqr_fused(ar, ai, or_, oi); }}
void fft_cmul_sqrt1_2(bool neg_re, bool pos_sign, df_t dr, df_t di, out df_t or_, out df_t oi) {{
    df_cmul_sqrt1_2_rt(neg_re, dr, di, pos_sign, or_, oi);
}}
df_t to_smem(df_t r) {{ return r; }}
df_t from_smem(df_t s) {{ return s; }}
{shuf}
"""
        )

    # prec == "h" (packed 2-lane f16vec2, uint32 SMEM & shuffle, float32 twiddle ladder)
    c32h = ", ".join(f"f16vec2(float16_t({fmt_f32(v)}), float16_t({fmt_f32(v)}))" for v in K_COS32)
    s32h = ", ".join(f"f16vec2(float16_t({fmt_f32(v)}), float16_t({fmt_f32(v)}))" for v in K_SIN32_NEG)
    w256r = ", ".join(fmt_f32(v) for v in K_W256_RE)
    w256i = ", ".join(fmt_f32(v) for v in K_W256_IM_NEG)
    shuf = """
f16vec2 fft_shuf_xor(f16vec2 v, uint mask) {
    return u32_to_f16x2(subgroupShuffleXor(f16x2_to_u32(v), mask));
}
""" if need_shuf else ""
    return (
        glsl_sincos2pi_f32()
        + glsl_f16x2_math()
        + f"""
#define REAL f16vec2
#define SMEM_T uint
#define BUF_T uint
#define DECONV_T float16_t
#define ZERO_R kZero_f16x2
#define ONE_R  kOne_f16x2
#define PACKED_BATCH 2u

const f16vec2 kCos32[16] = f16vec2[16]({c32h});
const f16vec2 kSin32Neg[16] = f16vec2[16]({s32h});
const float kW256ReF[16] = float[16]({w256r});
const float kW256ImNegF[16] = float[16]({w256i});

f16vec2 fft_add(f16vec2 a, f16vec2 b) {{ return a + b; }}
f16vec2 fft_sub(f16vec2 a, f16vec2 b) {{ return a - b; }}
f16vec2 fft_mul(f16vec2 a, f16vec2 b) {{ return a * b; }}
f16vec2 fft_half(f16vec2 a) {{ return a * kHalf_f16x2; }}
f16vec2 fft_cond_neg(f16vec2 v, bool cond) {{ return f16x2_cond_neg(v, cond); }}
void fft_cmul(f16vec2 ar, f16vec2 ai, f16vec2 br, f16vec2 bi, out f16vec2 or_, out f16vec2 oi) {{
    f16x2_cmul(ar, ai, br, bi, or_, oi);
}}
void fft_csqr(f16vec2 ar, f16vec2 ai, out f16vec2 or_, out f16vec2 oi) {{
    f16x2_csqr(ar, ai, or_, oi);
}}
void fft_cmul_sqrt1_2(bool neg_re, bool pos_sign, f16vec2 dr, f16vec2 di, out f16vec2 or_, out f16vec2 oi) {{
    f16x2_cmul_sqrt1_2_rt(neg_re, dr, di, pos_sign, or_, oi);
}}
uint to_smem(f16vec2 r) {{ return f16x2_to_u32(r); }}
f16vec2 from_smem(uint s) {{ return u32_to_f16x2(s); }}
{shuf}
"""
    )


def _emit_reg_base(log_e: int, arr_re: str = "r_re", arr_im: str = "r_im", fn_name: str | None = None) -> str:
    """Emit fully unrolled dif_radix2_reg_base<log_e> in GLSL."""
    e = 1 << log_e
    name = fn_name or f"dif_radix2_reg_base_{log_e}"
    lines = [f"void {name}(inout REAL {arr_re}[{e}], inout REAL {arr_im}[{e}], bool pos_sign) {{"]
    for s in range(log_e):
        h = e >> (s + 1)
        step = h << 1
        groups = e // step
        for m in range(h):
            idx32 = m * (32 // step)
            for g in range(groups):
                a = g * step + m
                b = a + h
                lines.append(f"    {{")
                lines.append(f"        REAL ar = fft_half({arr_re}[{a}]), ai = fft_half({arr_im}[{a}]);")
                lines.append(f"        REAL br = fft_half({arr_re}[{b}]), bi = fft_half({arr_im}[{b}]);")
                lines.append(f"        REAL dr = fft_sub(ar, br), di = fft_sub(ai, bi);")
                lines.append(f"        {arr_re}[{a}] = fft_add(ar, br); {arr_im}[{a}] = fft_add(ai, bi);")
                if m == 0:
                    lines.append(f"        {arr_re}[{b}] = dr; {arr_im}[{b}] = di;")
                elif idx32 == 8:
                    lines.append(f"        {arr_re}[{b}] = fft_cond_neg(di, pos_sign);")
                    lines.append(f"        {arr_im}[{b}] = fft_cond_neg(dr, !pos_sign);")
                elif idx32 == 4:
                    lines.append(f"        fft_cmul_sqrt1_2(false, pos_sign, dr, di, {arr_re}[{b}], {arr_im}[{b}]);")
                elif idx32 == 12:
                    lines.append(f"        fft_cmul_sqrt1_2(true, pos_sign, dr, di, {arr_re}[{b}], {arr_im}[{b}]);")
                else:
                    lines.append(f"        REAL cr = kCos32[{idx32}];")
                    lines.append(f"        REAL ci = fft_cond_neg(kSin32Neg[{idx32}], pos_sign);")
                    lines.append(f"        fft_cmul(dr, di, cr, ci, {arr_re}[{b}], {arr_im}[{b}]);")
                lines.append(f"    }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_twiddle_block(prec: str, log_e: int, log_t: int | None, arr_re: str, arr_im: str,
                        t_expr: str, inv_m_expr: str, pos_expr: str,
                        on_elem_cb) -> str:
    """Emit factored bit-reversed twiddle ladder on E = 2^log_e elements.

    on_elem_cb(p, re_var, im_var) returns a list of GLSL lines for element p.
    """
    e = 1 << log_e
    lines = ["{"]
    if prec == "h":
        lines.append("    float w1r, w1i;")
        if log_t is not None and log_e >= 4 and (log_e + log_t) <= 8:
            scale = 256 // (e * (1 << log_t))
            lines.append(f"    uint tw_idx = ({t_expr}) * {scale}u;")
            lines.append("    w1r = kW256ReF[tw_idx]; w1i = kW256ImNegF[tw_idx];")
        elif log_e >= 3:
            lines.append(f"    sincos2pif_q1(float({t_expr}) * ({inv_m_expr}), true, w1r, w1i);")
        else:
            lines.append(f"    sincos2pif_half(float({t_expr}) * ({inv_m_expr}), true, w1r, w1i);")
        lines.append(f"    w1i = uintBitsToFloat(floatBitsToUint(w1i) ^ (({pos_expr}) ? 0x80000000u : 0u));")
        lines.append("    float w2r = fma(w1r, w1r, -(w1i * w1i)), w2i = (2.0f * w1r) * w1i;")
        lines.append("    float w3r = fma(w2r, w1r, -(w2i * w1i)), w3i = fma(w2r, w1i, w2i * w1r);")
        lines.append("    float w4r = fma(w2r, w2r, -(w2i * w2i)), w4i = (2.0f * w2r) * w2i;")

        lines.extend(on_elem_cb(0, f"{arr_re}[0]", f"{arr_im}[0]"))
        if log_e == 3:
            lines.append("    float lo_r[4] = float[4](1.0f, w2r, w1r, w3r);")
            lines.append("    float lo_i[4] = float[4](0.0f, w2i, w1i, w3i);")
            for p in range(1, 8):
                a, b = p >> 1, p & 1
                if a == 0:
                    lines.append("    { float wr = w4r, wi = w4i;")
                elif b == 0:
                    lines.append(f"    {{ float wr = lo_r[{a}], wi = lo_i[{a}];")
                else:
                    lines.append(f"    {{ float wr = fma(lo_r[{a}], w4r, -(lo_i[{a}] * w4i));")
                    lines.append(f"      float wi = fma(lo_r[{a}], w4i, lo_i[{a}] * w4r);")
                lines.append("      f16vec2 twr = f16vec2(float16_t(wr)), twi = f16vec2(float16_t(wi));")
                lines.append(f"      REAL vr, vi; fft_cmul({arr_re}[{p}], {arr_im}[{p}], twr, twi, vr, vi);")
                lines.extend(on_elem_cb(p, "vr", "vi"))
                lines.append("    }")
        elif log_e == 4:
            lines.append("    float w8r = fma(w4r, w4r, -(w4i * w4i)), w8i = (2.0f * w4r) * w4i;")
            lines.append("    float w12r = fma(w8r, w4r, -(w8i * w4i)), w12i = fma(w8r, w4i, w8i * w4r);")
            lines.append("    float lo_r[4] = float[4](1.0f, w2r, w1r, w3r);")
            lines.append("    float lo_i[4] = float[4](0.0f, w2i, w1i, w3i);")
            lines.append("    float hi_r[4] = float[4](1.0f, w8r, w4r, w12r);")
            lines.append("    float hi_i[4] = float[4](0.0f, w8i, w4i, w12i);")
            for p in range(1, 16):
                a, b = p >> 2, p & 3
                if a == 0:
                    lines.append(f"    {{ float wr = hi_r[{b}], wi = hi_i[{b}];")
                elif b == 0:
                    lines.append(f"    {{ float wr = lo_r[{a}], wi = lo_i[{a}];")
                else:
                    lines.append(f"    {{ float wr = fma(lo_r[{a}], hi_r[{b}], -(lo_i[{a}] * hi_i[{b}]));")
                    lines.append(f"      float wi = fma(lo_r[{a}], hi_i[{b}], lo_i[{a}] * hi_r[{b}]);")
                lines.append("      f16vec2 twr = f16vec2(float16_t(wr)), twi = f16vec2(float16_t(wi));")
                lines.append(f"      REAL vr, vi; fft_cmul({arr_re}[{p}], {arr_im}[{p}], twr, twi, vr, vi);")
                lines.extend(on_elem_cb(p, "vr", "vi"))
                lines.append("    }")
        elif log_e == 5:
            lines.append("    float w5r = fma(w4r, w1r, -(w4i * w1i)), w5i = fma(w4r, w1i, w4i * w1r);")
            lines.append("    float w6r = fma(w4r, w2r, -(w4i * w2i)), w6i = fma(w4r, w2i, w4i * w2r);")
            lines.append("    float w7r = fma(w4r, w3r, -(w4i * w3i)), w7i = fma(w4r, w3i, w4i * w3r);")
            lines.append("    float w8r = fma(w4r, w4r, -(w4i * w4i)), w8i = (2.0f * w4r) * w4i;")
            lines.append("    float w16r = fma(w8r, w8r, -(w8i * w8i)), w16i = (2.0f * w8r) * w8i;")
            lines.append("    float w24r = fma(w16r, w8r, -(w16i * w8i)), w24i = fma(w16r, w8i, w16i * w8r);")
            lines.append("    float lo_r[8] = float[8](1.0f, w4r, w2r, w6r, w1r, w5r, w3r, w7r);")
            lines.append("    float lo_i[8] = float[8](0.0f, w4i, w2i, w6i, w1i, w5i, w3i, w7i);")
            lines.append("    float hi_r[4] = float[4](1.0f, w16r, w8r, w24r);")
            lines.append("    float hi_i[4] = float[4](0.0f, w16i, w8i, w24i);")
            for p in range(1, 32):
                a, b = p >> 2, p & 3
                if a == 0:
                    lines.append(f"    {{ float wr = hi_r[{b}], wi = hi_i[{b}];")
                elif b == 0:
                    lines.append(f"    {{ float wr = lo_r[{a}], wi = lo_i[{a}];")
                else:
                    lines.append(f"    {{ float wr = fma(lo_r[{a}], hi_r[{b}], -(lo_i[{a}] * hi_i[{b}]));")
                    lines.append(f"      float wi = fma(lo_r[{a}], hi_i[{b}], lo_i[{a}] * hi_r[{b}]);")
                lines.append("      f16vec2 twr = f16vec2(float16_t(wr)), twi = f16vec2(float16_t(wi));")
                lines.append(f"      REAL vr, vi; fft_cmul({arr_re}[{p}], {arr_im}[{p}], twr, twi, vr, vi);")
                lines.extend(on_elem_cb(p, "vr", "vi"))
                lines.append("    }")
    else:
        lines.append("    REAL w1r, w1i;")
        if log_t is not None and log_e >= 4 and (log_e + log_t) <= 8:
            scale = 256 // (e * (1 << log_t))
            lines.append(f"    uint tw_idx = ({t_expr}) * {scale}u;")
            lines.append("    w1r = kW256Re[tw_idx]; w1i = kW256ImNeg[tw_idx];")
        else:
            ang_cast = "double" if prec == "d" else "float"
            if prec == "f":
                fn = "sincos2pif_q1" if log_e >= 3 else "sincos2pif_half"
            elif prec == "df":
                fn = "sincos2pidf_oct1_f" if log_e >= 3 else "sincos2pidf_half_f"
            else:
                fn = "sincos2pi_oct1" if log_e >= 3 else "sincos2pi_half"
            lines.append(f"    {fn}({ang_cast}({t_expr}) * ({inv_m_expr}), true, w1r, w1i);")
        lines.append(f"    w1i = fft_cond_neg(w1i, ({pos_expr}));")
        lines.append("    REAL w2r, w2i, w3r, w3i, w4r, w4i;")
        lines.append("    fft_csqr(w1r, w1i, w2r, w2i);")
        lines.append("    fft_cmul(w2r, w2i, w1r, w1i, w3r, w3i);")
        lines.append("    fft_csqr(w2r, w2i, w4r, w4i);")

        lines.extend(on_elem_cb(0, f"{arr_re}[0]", f"{arr_im}[0]"))
        if log_e == 3:
            lines.append("    REAL lo_r[4] = REAL[4](ONE_R, w2r, w1r, w3r);")
            lines.append("    REAL lo_i[4] = REAL[4](ZERO_R, w2i, w1i, w3i);")
            for p in range(1, 8):
                a, b = p >> 1, p & 1
                lines.append("    { REAL wr, wi, vr, vi;")
                if a == 0:
                    lines.append("      wr = w4r; wi = w4i;")
                elif b == 0:
                    lines.append(f"      wr = lo_r[{a}]; wi = lo_i[{a}];")
                else:
                    lines.append(f"      fft_cmul(lo_r[{a}], lo_i[{a}], w4r, w4i, wr, wi);")
                lines.append(f"      fft_cmul({arr_re}[{p}], {arr_im}[{p}], wr, wi, vr, vi);")
                lines.extend(on_elem_cb(p, "vr", "vi"))
                lines.append("    }")
        elif log_e == 4:
            lines.append("    REAL w8r, w8i, w12r, w12i;")
            lines.append("    fft_csqr(w4r, w4i, w8r, w8i);")
            lines.append("    fft_cmul(w8r, w8i, w4r, w4i, w12r, w12i);")
            lines.append("    REAL lo_r[4] = REAL[4](ONE_R, w2r, w1r, w3r);")
            lines.append("    REAL lo_i[4] = REAL[4](ZERO_R, w2i, w1i, w3i);")
            lines.append("    REAL hi_r[4] = REAL[4](ONE_R, w8r, w4r, w12r);")
            lines.append("    REAL hi_i[4] = REAL[4](ZERO_R, w8i, w4i, w12i);")
            for p in range(1, 16):
                a, b = p >> 2, p & 3
                lines.append("    { REAL wr, wi, vr, vi;")
                if a == 0:
                    lines.append(f"      wr = hi_r[{b}]; wi = hi_i[{b}];")
                elif b == 0:
                    lines.append(f"      wr = lo_r[{a}]; wi = lo_i[{a}];")
                else:
                    lines.append(f"      fft_cmul(lo_r[{a}], lo_i[{a}], hi_r[{b}], hi_i[{b}], wr, wi);")
                lines.append(f"      fft_cmul({arr_re}[{p}], {arr_im}[{p}], wr, wi, vr, vi);")
                lines.extend(on_elem_cb(p, "vr", "vi"))
                lines.append("    }")
        elif log_e == 5:
            lines.append("    REAL w8r, w8i, w16r, w16i, w24r, w24i;")
            lines.append("    fft_csqr(w4r, w4i, w8r, w8i);")
            lines.append("    fft_csqr(w8r, w8i, w16r, w16i);")
            lines.append("    fft_cmul(w16r, w16i, w8r, w8i, w24r, w24i);")
            lines.append("    REAL hi_r[4] = REAL[4](ONE_R, w16r, w8r, w24r);")
            lines.append("    REAL hi_i[4] = REAL[4](ZERO_R, w16i, w8i, w24i);")
            if prec in ("df", "d"):
                lines.append("    REAL base_r[4] = REAL[4](ONE_R, w2r, w1r, w3r);")
                lines.append("    REAL base_i[4] = REAL[4](ZERO_R, w2i, w1i, w3i);")
                for a in range(8):
                    ahi, a0 = a >> 1, a & 1
                    lines.append("    { REAL lor, loi;")
                    if a == 0:
                        lines.append("      lor = ONE_R; loi = ZERO_R;")
                    elif ahi == 0:
                        lines.append("      lor = w4r; loi = w4i;")
                    elif a0 == 0:
                        lines.append(f"      lor = base_r[{ahi}]; loi = base_i[{ahi}];")
                    else:
                        lines.append(f"      fft_cmul(base_r[{ahi}], base_i[{ahi}], w4r, w4i, lor, loi);")
                    for b in range(4):
                        p = (a << 2) | b
                        if p == 0:
                            continue
                        lines.append("      { REAL wr, wi, vr, vi;")
                        if a == 0:
                            lines.append(f"        wr = hi_r[{b}]; wi = hi_i[{b}];")
                        elif b == 0:
                            lines.append("        wr = lor; wi = loi;")
                        else:
                            lines.append(f"        fft_cmul(lor, loi, hi_r[{b}], hi_i[{b}], wr, wi);")
                        lines.append(f"        fft_cmul({arr_re}[{p}], {arr_im}[{p}], wr, wi, vr, vi);")
                        lines.extend(on_elem_cb(p, "vr", "vi"))
                        lines.append("      }")
                    lines.append("    }")
            else:
                lines.append("    REAL w5r, w5i, w6r, w6i, w7r, w7i;")
                lines.append("    fft_cmul(w4r, w4i, w1r, w1i, w5r, w5i);")
                lines.append("    fft_cmul(w4r, w4i, w2r, w2i, w6r, w6i);")
                lines.append("    fft_cmul(w4r, w4i, w3r, w3i, w7r, w7i);")
                lines.append("    REAL lo_r[8] = REAL[8](ONE_R, w4r, w2r, w6r, w1r, w5r, w3r, w7r);")
                lines.append("    REAL lo_i[8] = REAL[8](ZERO_R, w4i, w2i, w6i, w1i, w5i, w3i, w7i);")
                for p in range(1, 32):
                    a, b = p >> 2, p & 3
                    lines.append("    { REAL wr, wi, vr, vi;")
                    if a == 0:
                        lines.append(f"      wr = hi_r[{b}]; wi = hi_i[{b}];")
                    elif b == 0:
                        lines.append(f"      wr = lo_r[{a}]; wi = lo_i[{a}];")
                    else:
                        lines.append(f"      fft_cmul(lo_r[{a}], lo_i[{a}], hi_r[{b}], hi_i[{b}], wr, wi);")
                    lines.append(f"      fft_cmul({arr_re}[{p}], {arr_im}[{p}], wr, wi, vr, vi);")
                    lines.extend(on_elem_cb(p, "vr", "vi"))
                    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_load_store_helpers(prec: str, writable_in: bool) -> str:
    in_qual = "" if writable_in else "readonly "
    if prec == "h":
        stash = """
void load_stash(uint base0, uint base1, uint idx, bool v0, bool v1, out REAL re_o, out REAL im_o) {
    uint u0 = 0u, u1 = 0u;
    if (SPEC_COMPACT_MODE == 0u) {
        u0 = v0 ? out_buf[base0 + idx] : 0u;
        u1 = v1 ? out_buf[base1 + idx] : 0u;
    } else {
        u0 = v0 ? in_buf[base0 + idx] : 0u;
        u1 = v1 ? in_buf[base1 + idx] : 0u;
    }
    f16x2_load_pair_u32(u0, u1, re_o, im_o);
}
void store_stash(uint base0, uint base1, uint idx, bool v0, bool v1, REAL re_i, REAL im_i) {
    uint u0, u1;
    f16x2_pack_pair_u32(re_i, im_i, u0, u1);
    if (SPEC_COMPACT_MODE == 0u) {
        if (v0) out_buf[base0 + idx] = u0;
        if (v1) out_buf[base1 + idx] = u1;
    } else {
        if (v0) in_buf[base0 + idx] = u0;
        if (v1) in_buf[base1 + idx] = u1;
    }
}
""" if writable_in else ""
        return f"""
layout(set = 0, binding = 0) {in_qual}buffer InBuf  {{ uint in_buf[]; }};
layout(set = 0, binding = 1) buffer OutBuf          {{ uint out_buf[]; }};
layout(set = 0, binding = 2) readonly buffer DecBuf {{ float16_t dec_buf[]; }};

void load_elem(uint base0, uint base1, uint idx, bool v0, bool v1, out REAL re_o, out REAL im_o) {{
    uint u0 = v0 ? in_buf[base0 + idx] : 0u;
    uint u1 = v1 ? in_buf[base1 + idx] : 0u;
    f16x2_load_pair_u32(u0, u1, re_o, im_o);
}}
void store_elem(uint base0, uint base1, uint idx, bool v0, bool v1, REAL re_i, REAL im_i) {{
    uint u0, u1;
    f16x2_pack_pair_u32(re_i, im_i, u0, u1);
    if (v0) out_buf[base0 + idx] = u0;
    if (v1) out_buf[base1 + idx] = u1;
}}
{stash}
REAL load_deconv_val(uint d0, uint d1, uint ci, uint log_n1) {{
    return (log_n1 == 0u) ? f16vec2(dec_buf[d0 + ci]) : f16vec2(dec_buf[d0 + ci], dec_buf[d1 + ci]);
}}
"""
    if prec == "df":
        stash = """
void load_stash(uint base0, uint base1, uint idx, bool v0, bool v1, out REAL re_o, out REAL im_o) {
    if (v0) {
        vec4 v = (SPEC_COMPACT_MODE == 0u) ? out_buf[base0 + idx] : in_buf[base0 + idx];
        re_o = v.xy; im_o = v.zw;
    } else { re_o = ZERO_R; im_o = ZERO_R; }
}
void store_stash(uint base0, uint base1, uint idx, bool v0, bool v1, REAL re_i, REAL im_i) {
    if (v0) {
        if (SPEC_COMPACT_MODE == 0u) out_buf[base0 + idx] = vec4(re_i, im_i);
        else in_buf[base0 + idx] = vec4(re_i, im_i);
    }
}
""" if writable_in else ""
        return f"""
layout(set = 0, binding = 0) {in_qual}buffer InBuf  {{ vec4 in_buf[]; }};
layout(set = 0, binding = 1) buffer OutBuf          {{ vec4 out_buf[]; }};
layout(set = 0, binding = 2) readonly buffer DecBuf {{ vec2 dec_buf[]; }};

void load_elem(uint base0, uint base1, uint idx, bool v0, bool v1, out REAL re_o, out REAL im_o) {{
    if (v0) {{ vec4 v = in_buf[base0 + idx]; re_o = v.xy; im_o = v.zw; }}
    else {{ re_o = ZERO_R; im_o = ZERO_R; }}
}}
void store_elem(uint base0, uint base1, uint idx, bool v0, bool v1, REAL re_i, REAL im_i) {{
    if (v0) out_buf[base0 + idx] = vec4(re_i, im_i);
}}
{stash}
REAL load_deconv_val(uint d0, uint d1, uint ci, uint log_n1) {{ return dec_buf[d0 + ci]; }}
"""
    # f or d
    btype = "vec2" if prec == "f" else "dvec2"
    dtype = "float" if prec == "f" else "double"
    stash = f"""
void load_stash(uint base0, uint base1, uint idx, bool v0, bool v1, out REAL re_o, out REAL im_o) {{
    if (v0) {{
        {btype} v = (SPEC_COMPACT_MODE == 0u) ? out_buf[base0 + idx] : in_buf[base0 + idx];
        re_o = v.x; im_o = v.y;
    }} else {{ re_o = ZERO_R; im_o = ZERO_R; }}
}}
void store_stash(uint base0, uint base1, uint idx, bool v0, bool v1, REAL re_i, REAL im_i) {{
    if (v0) {{
        if (SPEC_COMPACT_MODE == 0u) out_buf[base0 + idx] = {btype}(re_i, im_i);
        else in_buf[base0 + idx] = {btype}(re_i, im_i);
    }}
}}
""" if writable_in else ""
    return f"""
layout(set = 0, binding = 0) {in_qual}buffer InBuf  {{ {btype} in_buf[]; }};
layout(set = 0, binding = 1) buffer OutBuf          {{ {btype} out_buf[]; }};
layout(set = 0, binding = 2) readonly buffer DecBuf {{ {dtype} dec_buf[]; }};

void load_elem(uint base0, uint base1, uint idx, bool v0, bool v1, out REAL re_o, out REAL im_o) {{
    if (v0) {{ {btype} v = in_buf[base0 + idx]; re_o = v.x; im_o = v.y; }}
    else {{ re_o = ZERO_R; im_o = ZERO_R; }}
}}
void store_elem(uint base0, uint base1, uint idx, bool v0, bool v1, REAL re_i, REAL im_i) {{
    if (v0) out_buf[base0 + idx] = {btype}(re_i, im_i);
}}
{stash}
REAL load_deconv_val(uint d0, uint d1, uint ci, uint log_n1) {{ return dec_buf[d0 + ci]; }}
"""


def _emit_epilogue_block(chunk: int, w: int, get_val_fn) -> str:
    """Emit compact-slot filtering + optional deconvolution epilogue."""
    lines = ["if (valid0) {"]
    lines.append("  if (SPEC_COMPACT_MODE == 0u) {")
    for m in range(chunk):
        vr, vi = get_val_fn(m)
        lines.append(f"    store_elem(g_out0, g_out1, out_base + {m * w}u, valid0, valid1, {vr}, {vi});")
    lines.append("  } else {")
    if w >= 8:
        lines.append("    bool keep = false; uint ci0 = 0u; uint ci_stride = 0u;")
        lines.append("    if (SPEC_COMPACT_MODE == 1u) {")
        lines.append("      uint r = l & 3u; keep = (r == 0u || r == 3u);")
        lines.append("      ci0 = ((out_base >> 2u) << 1u) + ((r == 3u) ? 1u : 0u);")
        lines.append(f"      ci_stride = {w >> 1}u;")
        lines.append("    } else {")
        lines.append("      uint r = l & 7u; keep = (r != 1u && r != 6u);")
        lines.append("      uint off = r - ((r >= 2u) ? 1u : 0u) - ((r >= 7u) ? 1u : 0u);")
        lines.append("      ci0 = (out_base >> 3u) * 6u + off;")
        lines.append(f"      ci_stride = {(w >> 3) * 6}u;")
        lines.append("    }")
        lines.append("    if (keep) {")
        lines.append("      if (SPEC_APPLY_DECONV != 0u) {")
        for m in range(chunk):
            vr, vi = get_val_fn(m)
            lines.append(f"        {{ uint ci = ci0 + {m}u * ci_stride; REAL dv = load_deconv_val(d_dec0, d_dec1, ci, pc.log_n1);")
            lines.append(f"          store_elem(g_out0, g_out1, ci, valid0, valid1, fft_mul({vr}, dv), fft_mul({vi}, dv)); }}")
        lines.append("      } else {")
        for m in range(chunk):
            vr, vi = get_val_fn(m)
            lines.append(f"        store_elem(g_out0, g_out1, ci0 + {m}u * ci_stride, valid0, valid1, {vr}, {vi});")
        lines.append("      }")
        lines.append("    }")
    else:
        # w == 4
        lines.append("    if (SPEC_COMPACT_MODE == 1u) {")
        lines.append("      if (l == 0u || l == 3u) {")
        lines.append("        uint ci0 = ((out_base >> 2u) << 1u) + ((l == 3u) ? 1u : 0u);")
        lines.append("        if (SPEC_APPLY_DECONV != 0u) {")
        for m in range(chunk):
            vr, vi = get_val_fn(m)
            lines.append(f"          {{ uint ci = ci0 + {m * 2}u; REAL dv = load_deconv_val(d_dec0, d_dec1, ci, pc.log_n1);")
            lines.append(f"            store_elem(g_out0, g_out1, ci, valid0, valid1, fft_mul({vr}, dv), fft_mul({vi}, dv)); }}")
        lines.append("        } else {")
        for m in range(chunk):
            vr, vi = get_val_fn(m)
            lines.append(f"          store_elem(g_out0, g_out1, ci0 + {m * 2}u, valid0, valid1, {vr}, {vi});")
        lines.append("        }")
        lines.append("      }")
        lines.append("    } else {")
        lines.append("      uint blk0 = (out_base >> 3u) * 6u;")
        lines.append("      if (l != 1u) {")
        lines.append("        uint off_e = (l >= 2u) ? (l - 1u) : l;")
        lines.append("        if (SPEC_APPLY_DECONV != 0u) {")
        for k in range(chunk // 2):
            vr, vi = get_val_fn(2 * k)
            lines.append(f"          {{ uint ci = blk0 + {k * 6}u + off_e; REAL dv = load_deconv_val(d_dec0, d_dec1, ci, pc.log_n1);")
            lines.append(f"            store_elem(g_out0, g_out1, ci, valid0, valid1, fft_mul({vr}, dv), fft_mul({vi}, dv)); }}")
        lines.append("        } else {")
        for k in range(chunk // 2):
            vr, vi = get_val_fn(2 * k)
            lines.append(f"          store_elem(g_out0, g_out1, blk0 + {k * 6}u + off_e, valid0, valid1, {vr}, {vi});")
        lines.append("        }")
        lines.append("      }")
        lines.append("      if (l != 2u) {")
        lines.append("        uint off_o = (l == 3u) ? 5u : (l + 3u);")
        lines.append("        if (SPEC_APPLY_DECONV != 0u) {")
        for k in range(chunk // 2):
            vr, vi = get_val_fn(2 * k + 1)
            lines.append(f"          {{ uint ci = blk0 + {k * 6}u + off_o; REAL dv = load_deconv_val(d_dec0, d_dec1, ci, pc.log_n1);")
            lines.append(f"            store_elem(g_out0, g_out1, ci, valid0, valid1, fft_mul({vr}, dv), fft_mul({vi}, dv)); }}")
        lines.append("        } else {")
        for k in range(chunk // 2):
            vr, vi = get_val_fn(2 * k + 1)
            lines.append(f"          store_elem(g_out0, g_out1, blk0 + {k * 6}u + off_o, valid0, valid1, {vr}, {vi});")
        lines.append("        }")
        lines.append("      }")
        lines.append("    }")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _gen_single_pass(prec: str, spec: tuple) -> dict:
    vid, logN, logE, logChunks, B, staging, t1_layout, phase3, stash1 = spec
    N = 1 << logN
    E = 1 << logE
    logT = logN - logE
    T = 1 << logT
    logChunk = logE - logChunks
    Chunk = 1 << logChunk
    logW = logT - logChunk
    W = 1 << logW
    WG = B * T

    need_shuf = (phase3 == "shuf")
    need_sg = need_shuf

    hdr = glsl_header(
        prec,
        subgroups=need_sg,
        subgroup_shuffle=need_shuf,
        fp16_storage=(prec == "h"),
    )

    smem_decl = (
        "shared SMEM_T smem_re[SPEC_SMEM_ELEMS];\nshared SMEM_T smem_im[SPEC_SMEM_ELEMS];\n"
        if staging == "split"
        else "shared SMEM_T smem_buf[SPEC_SMEM_ELEMS];\n"
    )

    # Emit reg_base functions needed (deduplicated)
    reg_bases = set()
    reg_bases.add(logE)
    reg_bases.add(logChunk)
    if phase3 != "shuf":
        reg_bases.add(logW)
    reg_base_code = "\n".join(_emit_reg_base(r) for r in sorted(reg_bases))

    # Emit finish_chunk body
    fc_lines = [
        f"void finish_chunk(inout REAL c_re[{Chunk}], inout REAL c_im[{Chunk}],",
        "                  uint g_out0, uint g_out1, uint d_dec0, uint d_dec1,",
        "                  uint chunk_idx, uint smem_base, uint t, uint sub, uint l,",
        "                  bool valid0, bool valid1, bool pos_sign) {",
        "    uint row_base = smem_base + sub * SPEC_ROW_STRIDE;",
    ]

    # Transpose 1
    def t1_w(m: int) -> str:
        return (
            f"smem_base + {m * T}u + ((sub ^ {m}u) << {logW}u) + l"
            if t1_layout == "xor"
            else f"smem_base + {m}u * SPEC_ROW_STRIDE + t"
        )

    def t1_r(m: int) -> str:
        return (
            f"smem_base + sub * {T}u + (({m}u ^ sub) << {logW}u) + l"
            if t1_layout == "xor"
            else f"row_base + {m * W}u + l"
        )

    if staging == "split":
        for m in range(Chunk):
            fc_lines.append(f"    {{ uint wi = {t1_w(m)}; smem_re[wi] = to_smem(c_re[{m}]); smem_im[wi] = to_smem(c_im[{m}]); }}")
        fc_lines.append("    barrier();")
        for m in range(Chunk):
            fc_lines.append(f"    {{ uint ri = {t1_r(m)}; c_re[{m}] = from_smem(smem_re[ri]); c_im[{m}] = from_smem(smem_im[ri]); }}")
    else:
        for m in range(Chunk):
            fc_lines.append(f"    smem_buf[{t1_w(m)}] = to_smem(c_re[{m}]);")
        fc_lines.append("    barrier();")
        for m in range(Chunk):
            fc_lines.append(f"    c_re[{m}] = from_smem(smem_buf[{t1_r(m)}]);")
        fc_lines.append("    barrier();")
        for m in range(Chunk):
            fc_lines.append(f"    smem_buf[{t1_w(m)}] = to_smem(c_im[{m}]);")
        fc_lines.append("    barrier();")
        for m in range(Chunk):
            fc_lines.append(f"    c_im[{m}] = from_smem(smem_buf[{t1_r(m)}]);")

    # Phase 2: Register DIF across Chunk + inter-lane twiddles (LogChunk, LogW)
    fc_lines.append(f"    dif_radix2_reg_base_{logChunk}(c_re, c_im, pos_sign);")
    if logW > 0:
        inv_m_p2 = fmt_f64(1.0 / (Chunk * W)) if prec == "d" else fmt_f32(1.0 / (Chunk * W))
        fc_lines.append(
            _emit_twiddle_block(
                prec, logChunk, logW, "c_re", "c_im", "l", inv_m_p2, "pos_sign",
                lambda p, vr, vi: [] if p == 0 else [f"      c_re[{p}] = {vr}; c_im[{p}] = {vi};"],
            )
        )

    out_base_expr = f"(sub * {T}u + l)" if logChunks == 0 else f"((chunk_idx * {Chunk}u + sub) * {T}u + l)"
    fc_lines.append(f"    uint out_base = {out_base_expr};")

    # Phase 3 & Coalesced Store
    if phase3 == "shuf":
        for s in range(logW):
            h = W >> (s + 1)
            fc_lines.append(f"    {{ bool is_hi = (l & {h}u) != 0u; uint k = l & {h - 1}u;")
            if h > 2:
                fc_lines.append(f"      uint idx32 = k * {16 // h}u;")
                fc_lines.append("      REAL tw_re = kCos32[idx32];")
                fc_lines.append("      REAL tw_im = fft_cond_neg(kSin32Neg[idx32], pos_sign);")
            for m in range(Chunk):
                fc_lines.append(f"      {{ REAL my_re = fft_half(c_re[{m}]), my_im = fft_half(c_im[{m}]);")
                fc_lines.append(f"        REAL p_re = fft_shuf_xor(my_re, {h}u), p_im = fft_shuf_xor(my_im, {h}u);")
                fc_lines.append("        REAL u_re = fft_add(p_re, fft_cond_neg(my_re, is_hi));")
                fc_lines.append("        REAL u_im = fft_add(p_im, fft_cond_neg(my_im, is_hi));")
                if h == 1:
                    fc_lines.append(f"        c_re[{m}] = u_re; c_im[{m}] = u_im;")
                elif h == 2:
                    fc_lines.append(f"        if (is_hi && k != 0u) {{ c_re[{m}] = fft_cond_neg(u_im, pos_sign); c_im[{m}] = fft_cond_neg(u_re, !pos_sign); }}")
                    fc_lines.append(f"        else {{ c_re[{m}] = u_re; c_im[{m}] = u_im; }}")
                else:
                    fc_lines.append(f"        if (is_hi) {{ fft_cmul(u_re, u_im, tw_re, tw_im, c_re[{m}], c_im[{m}]); }}")
                    fc_lines.append(f"        else {{ c_re[{m}] = u_re; c_im[{m}] = u_im; }}")
                fc_lines.append("      }")
            fc_lines.append("    }")
        fc_lines.append(_emit_epilogue_block(Chunk, W, lambda m: (f"c_re[{m}]", f"c_im[{m}]")))

    elif phase3 == "2txor":
        fc_lines.append("    barrier();")
        if staging == "split":
            for m in range(Chunk):
                fc_lines.append(f"    {{ uint idx = row_base + {m * W}u + (l ^ {m}u); smem_re[idx] = to_smem(c_re[{m}]); smem_im[idx] = to_smem(c_im[{m}]); }}")
            fc_lines.append("    barrier();")
            for k in range(Chunk):
                fc_lines.append(f"    {{ uint idx = row_base + l * {W}u + ({k}u ^ l); c_re[{k}] = from_smem(smem_re[idx]); c_im[{k}] = from_smem(smem_im[idx]); }}")
        else:
            for m in range(Chunk):
                fc_lines.append(f"    smem_buf[row_base + {m * W}u + (l ^ {m}u)] = to_smem(c_re[{m}]);")
            fc_lines.append("    barrier();")
            for k in range(Chunk):
                fc_lines.append(f"    c_re[{k}] = from_smem(smem_buf[row_base + l * {W}u + ({k}u ^ l)]);")
            fc_lines.append("    barrier();")
            for m in range(Chunk):
                fc_lines.append(f"    smem_buf[row_base + {m * W}u + (l ^ {m}u)] = to_smem(c_im[{m}]);")
            fc_lines.append("    barrier();")
            for k in range(Chunk):
                fc_lines.append(f"    c_im[{k}] = from_smem(smem_buf[row_base + l * {W}u + ({k}u ^ l)]);")

        fc_lines.append(f"    dif_radix2_reg_base_{logW}(c_re, c_im, pos_sign);")
        fc_lines.append("    barrier();")

        if staging == "split":
            for k in range(Chunk):
                fc_lines.append(f"    {{ uint idx = row_base + l * {W}u + ({k}u ^ l); smem_re[idx] = to_smem(c_re[{k}]); smem_im[idx] = to_smem(c_im[{k}]); }}")
            fc_lines.append("    barrier();")
            fc_lines.append(_emit_epilogue_block(
                Chunk, W,
                lambda m: (f"from_smem(smem_re[row_base + {m * W}u + (l ^ {m}u)])",
                           f"from_smem(smem_im[row_base + {m * W}u + (l ^ {m}u)])"),
            ))
        else:
            for k in range(Chunk):
                fc_lines.append(f"    smem_buf[row_base + l * {W}u + ({k}u ^ l)] = to_smem(c_re[{k}]);")
            fc_lines.append("    barrier();")
            for m in range(Chunk):
                fc_lines.append(f"    c_re[{m}] = from_smem(smem_buf[row_base + {m * W}u + (l ^ {m}u)]);")
            fc_lines.append("    barrier();")
            for k in range(Chunk):
                fc_lines.append(f"    smem_buf[row_base + l * {W}u + ({k}u ^ l)] = to_smem(c_im[{k}]);")
            fc_lines.append("    barrier();")
            for m in range(Chunk):
                fc_lines.append(f"    c_im[{m}] = from_smem(smem_buf[row_base + {m * W}u + (l ^ {m}u)]);")
            fc_lines.append(_emit_epilogue_block(Chunk, W, lambda m: (f"c_re[{m}]", f"c_im[{m}]")))

    else:
        # phase3 == "2tstr"
        groups = Chunk // W
        for g in range(groups):
            fc_lines.append(f"    REAL blk_re_{g}[{W}], blk_im_{g}[{W}];")
        fc_lines.append("    barrier();")
        if staging == "split":
            for m in range(Chunk):
                fc_lines.append(f"    {{ uint idx = row_base + {m}u * SPEC_STRIDE2 + l; smem_re[idx] = to_smem(c_re[{m}]); smem_im[idx] = to_smem(c_im[{m}]); }}")
            fc_lines.append("    barrier();")
            for g in range(groups):
                fc_lines.append(f"    {{ uint off = row_base + (l + {g * W}u) * SPEC_STRIDE2;")
                for k in range(W):
                    fc_lines.append(f"      blk_re_{g}[{k}] = from_smem(smem_re[off + {k}u]); blk_im_{g}[{k}] = from_smem(smem_im[off + {k}u]);")
                fc_lines.append("    }")
        else:
            for m in range(Chunk):
                fc_lines.append(f"    smem_buf[row_base + {m}u * SPEC_STRIDE2 + l] = to_smem(c_re[{m}]);")
            fc_lines.append("    barrier();")
            for g in range(groups):
                fc_lines.append(f"    {{ uint off = row_base + (l + {g * W}u) * SPEC_STRIDE2;")
                for k in range(W):
                    fc_lines.append(f"      blk_re_{g}[{k}] = from_smem(smem_buf[off + {k}u]);")
                fc_lines.append("    }")
            fc_lines.append("    barrier();")
            for m in range(Chunk):
                fc_lines.append(f"    smem_buf[row_base + {m}u * SPEC_STRIDE2 + l] = to_smem(c_im[{m}]);")
            fc_lines.append("    barrier();")
            for g in range(groups):
                fc_lines.append(f"    {{ uint off = row_base + (l + {g * W}u) * SPEC_STRIDE2;")
                for k in range(W):
                    fc_lines.append(f"      blk_im_{g}[{k}] = from_smem(smem_buf[off + {k}u]);")
                fc_lines.append("    }")

        for g in range(groups):
            fc_lines.append(f"    dif_radix2_reg_base_{logW}(blk_re_{g}, blk_im_{g}, pos_sign);")

        fc_lines.append("    barrier();")
        if staging == "split":
            for g in range(groups):
                fc_lines.append(f"    {{ uint off = row_base + (l + {g * W}u) * SPEC_STRIDE2;")
                for k in range(W):
                    fc_lines.append(f"      smem_re[off + {k}u] = to_smem(blk_re_{g}[{k}]); smem_im[off + {k}u] = to_smem(blk_im_{g}[{k}]);")
                fc_lines.append("    }")
            fc_lines.append("    barrier();")
            fc_lines.append(_emit_epilogue_block(
                Chunk, W,
                lambda m: (f"from_smem(smem_re[row_base + {m}u * SPEC_STRIDE2 + l])",
                           f"from_smem(smem_im[row_base + {m}u * SPEC_STRIDE2 + l])"),
            ))
        else:
            for g in range(groups):
                fc_lines.append(f"    {{ uint off = row_base + (l + {g * W}u) * SPEC_STRIDE2;")
                for k in range(W):
                    fc_lines.append(f"      smem_buf[off + {k}u] = to_smem(blk_re_{g}[{k}]);")
                fc_lines.append("    }")
            fc_lines.append("    barrier();")
            for m in range(Chunk):
                fc_lines.append(f"    c_re[{m}] = from_smem(smem_buf[row_base + {m}u * SPEC_STRIDE2 + l]);")
            fc_lines.append("    barrier();")
            for g in range(groups):
                fc_lines.append(f"    {{ uint off = row_base + (l + {g * W}u) * SPEC_STRIDE2;")
                for k in range(W):
                    fc_lines.append(f"      smem_buf[off + {k}u] = to_smem(blk_im_{g}[{k}]);")
                fc_lines.append("    }")
            fc_lines.append("    barrier();")
            for m in range(Chunk):
                fc_lines.append(f"    c_im[{m}] = from_smem(smem_buf[row_base + {m}u * SPEC_STRIDE2 + l]);")
            fc_lines.append(_emit_epilogue_block(Chunk, W, lambda m: (f"c_re[{m}]", f"c_im[{m}]")))

    fc_lines.append("}")

    # Emit main()
    inv_m_p1 = fmt_f64(1.0 / N) if prec == "d" else fmt_f32(1.0 / N)
    main_lines = [
        "void main() {",
        "    uint wg_id = gl_WorkGroupID.x;",
        "    uint tid = gl_LocalInvocationID.x;",
        f"    uint b_local = {(f'tid >> {logT}u') if B > 1 else '0u'};",
        f"    uint t = {(f'tid & {T - 1}u') if B > 1 else 'tid'};",
        f"    uint pair_idx = {(f'wg_id * {B}u + b_local') if B > 1 else 'wg_id'};",
        "    uint batch0 = pair_idx * PACKED_BATCH;",
        "    bool valid0 = batch0 < pc.bs;",
        "    bool valid1 = (PACKED_BATCH == 2u) && (batch0 + 1u < pc.bs);",
        f"    uint sub_nout = (SPEC_COMPACT_MODE == 0u) ? {N}u : ((SPEC_COMPACT_MODE == 1u) ? {N >> 1}u : {(N >> 2) * 3}u);",
        "    uint n1_mask = (1u << pc.log_n1) - 1u;",
        "    uint p0_0 = batch0 & n1_mask;",
        "    uint p0_1 = (batch0 + 1u) & n1_mask;",
        f"    uint g_in0  = pc.in_offset  + (valid0 ? (batch0 * {N}u) : 0u);",
        f"    uint g_in1  = pc.in_offset  + (valid1 ? ((batch0 + 1u) * {N}u) : (valid0 ? (batch0 * {N}u) : 0u));",
        "    uint g_out0 = pc.out_offset + (valid0 ? (batch0 * sub_nout) : 0u);",
        "    uint g_out1 = pc.out_offset + (valid1 ? ((batch0 + 1u) * sub_nout) : (valid0 ? (batch0 * sub_nout) : 0u));",
        "    uint d_dec0 = (SPEC_APPLY_DECONV != 0u) ? (p0_0 * sub_nout) : 0u;",
        "    uint d_dec1 = (SPEC_APPLY_DECONV != 0u) ? (p0_1 * sub_nout) : 0u;",
        f"    uint smem_base = {'0u' if B == 1 else f'b_local * ({Chunk}u * SPEC_ROW_STRIDE)'};",
        f"    uint sub = t >> {logW}u;",
        f"    uint l = t & {W - 1}u;",
        "    bool pos_sign = (SPEC_POS_SIGN != 0u);",
    ]

    if logChunks == 0:
        main_lines.append(f"    REAL r_re[{E}], r_im[{E}];")
        main_lines.append("    if (valid0) {")
        for m in range(E):
            main_lines.append(f"        load_elem(g_in0, g_in1, t + {m * T}u, valid0, valid1, r_re[{m}], r_im[{m}]);")
        main_lines.append("    } else {")
        for m in range(E):
            main_lines.append(f"        r_re[{m}] = ZERO_R; r_im[{m}] = ZERO_R;")
        main_lines.append("    }")
        main_lines.append(f"    dif_radix2_reg_base_{logE}(r_re, r_im, pos_sign);")
        main_lines.append(
            _emit_twiddle_block(
                prec, logE, logT, "r_re", "r_im", "t", inv_m_p1, "pos_sign",
                lambda p, vr, vi: [] if p == 0 else [f"      r_re[{p}] = {vr}; r_im[{p}] = {vi};"],
            )
        )
        main_lines.append("    finish_chunk(r_re, r_im, g_out0, g_out1, d_dec0, d_dec1, 0u, smem_base, t, sub, l, valid0, valid1, pos_sign);")
    elif stash1:
        main_lines.append(f"    REAL c_re[{Chunk}], c_im[{Chunk}];")
        main_lines.append("    uint g_st0 = (SPEC_COMPACT_MODE == 0u) ? g_out0 : g_in0;")
        main_lines.append("    uint g_st1 = (SPEC_COMPACT_MODE == 0u) ? g_out1 : g_in1;")
        main_lines.append("    {")
        main_lines.append(f"        REAL r_re[{E}], r_im[{E}];")
        for m in range(E):
            main_lines.append(f"        load_elem(g_in0, g_in1, t + {m * T}u, valid0, valid1, r_re[{m}], r_im[{m}]);")
        main_lines.append(f"        dif_radix2_reg_base_{logE}(r_re, r_im, pos_sign);")
        main_lines.append(
            _emit_twiddle_block(
                prec, logE, logT, "r_re", "r_im", "t", inv_m_p1, "pos_sign",
                lambda p, vr, vi: (
                    [f"      c_re[{p}] = {vr}; c_im[{p}] = {vi};"]
                    if p < Chunk
                    else [f"      store_stash(g_st0, g_st1, {p * T}u + t, valid0, valid1, {vr}, {vi});"]
                ),
            )
        )
        main_lines.append("    }")
        main_lines.append("    finish_chunk(c_re, c_im, g_out0, g_out1, d_dec0, d_dec1, 0u, smem_base, t, sub, l, valid0, valid1, pos_sign);")
        main_lines.append("    barrier();")
        for m in range(Chunk):
            main_lines.append(f"    load_stash(g_st0, g_st1, {(Chunk + m) * T}u + t, valid0, valid1, c_re[{m}], c_im[{m}]);")
        main_lines.append("    finish_chunk(c_re, c_im, g_out0, g_out1, d_dec0, d_dec1, 1u, smem_base, t, sub, l, valid0, valid1, pos_sign);")
    else:
        main_lines.append(f"    REAL r_re[{E}], r_im[{E}];")
        for m in range(E):
            main_lines.append(f"    load_elem(g_in0, g_in1, t + {m * T}u, valid0, valid1, r_re[{m}], r_im[{m}]);")
        main_lines.append(f"    dif_radix2_reg_base_{logE}(r_re, r_im, pos_sign);")
        main_lines.append(
            _emit_twiddle_block(
                prec, logE, logT, "r_re", "r_im", "t", inv_m_p1, "pos_sign",
                lambda p, vr, vi: [] if p == 0 else [f"      r_re[{p}] = {vr}; r_im[{p}] = {vi};"],
            )
        )
        main_lines.append(f"    REAL c_re[{Chunk}], c_im[{Chunk}];")
        for m in range(Chunk):
            main_lines.append(f"    c_re[{m}] = r_re[{m}]; c_im[{m}] = r_im[{m}];")
        main_lines.append("    finish_chunk(c_re, c_im, g_out0, g_out1, d_dec0, d_dec1, 0u, smem_base, t, sub, l, valid0, valid1, pos_sign);")
        main_lines.append("    barrier();")
        for m in range(Chunk):
            main_lines.append(f"    c_re[{m}] = r_re[{Chunk + m}]; c_im[{m}] = r_im[{Chunk + m}];")
        main_lines.append("    finish_chunk(c_re, c_im, g_out0, g_out1, d_dec0, d_dec1, 1u, smem_base, t, sub, l, valid0, valid1, pos_sign);")
    main_lines.append("}")

    src = (
        hdr
        + f"layout(local_size_x = {WG}, local_size_y = 1, local_size_z = 1) in;\n"
        + f"layout(constant_id = 0) const uint SPEC_ROW_STRIDE   = {T}u;\n"
        + f"layout(constant_id = 1) const uint SPEC_STRIDE2      = {W}u;\n"
        + f"layout(constant_id = 2) const uint SPEC_SMEM_ELEMS   = {B * Chunk * T}u;\n"
        + "layout(constant_id = 3) const uint SPEC_POS_SIGN     = 1u;\n"
        + "layout(constant_id = 4) const uint SPEC_COMPACT_MODE = 0u;\n"
        + "layout(constant_id = 5) const uint SPEC_APPLY_DECONV = 0u;\n"
        + _prec_helpers(prec, need_shuf)
        + smem_decl
        + _emit_load_store_helpers(prec, writable_in=stash1)
        + """
layout(push_constant) uniform PC {
    uint bs;
    uint log_n1;
    uint in_offset;
    uint out_offset;
} pc;
"""
        + reg_base_code
        + "\n".join(fc_lines) + "\n"
        + "\n".join(main_lines) + "\n"
    )

    return {
        "target_env": "vulkan1.2" if prec == "h" else "vulkan1.1",
        "check_nocontract": (prec == "df"),
        "source": src,
    }


def _gen_pass0(prec: str, log_n1: int) -> dict:
    """Generate Pass-0 / MultiStageGlobal register-only DIF stage (LogN1 in 3..5)."""
    n1 = 1 << log_n1
    hdr = glsl_header(prec, fp16_storage=(prec == "h"))
    ang_type = "double" if prec == "d" else "float"

    if prec == "h":
        buf_decl = """
layout(set = 0, binding = 0) readonly buffer InBuf { uint in_buf[]; };
layout(set = 0, binding = 1) buffer OutBuf         { uint out_buf[]; };
void load_elem(uint b0, uint b1, uint idx, bool v0, bool v1, out REAL ro, out REAL io) {
    f16x2_load_pair_u32(v0 ? in_buf[b0 + idx] : 0u, v1 ? in_buf[b1 + idx] : 0u, ro, io);
}
void store_elem(uint b0, uint b1, uint idx, bool v0, bool v1, REAL ri, REAL ii) {
    uint u0, u1; f16x2_pack_pair_u32(ri, ii, u0, u1);
    if (v0) out_buf[b0 + idx] = u0;
    if (v1) out_buf[b1 + idx] = u1;
}
"""
    elif prec == "df":
        buf_decl = """
layout(set = 0, binding = 0) readonly buffer InBuf { vec4 in_buf[]; };
layout(set = 0, binding = 1) buffer OutBuf         { vec4 out_buf[]; };
void load_elem(uint b0, uint b1, uint idx, bool v0, bool v1, out REAL ro, out REAL io) {
    if (v0) { vec4 v = in_buf[b0 + idx]; ro = v.xy; io = v.zw; } else { ro = ZERO_R; io = ZERO_R; }
}
void store_elem(uint b0, uint b1, uint idx, bool v0, bool v1, REAL ri, REAL ii) {
    if (v0) out_buf[b0 + idx] = vec4(ri, ii);
}
"""
    else:
        bt = "vec2" if prec == "f" else "dvec2"
        buf_decl = f"""
layout(set = 0, binding = 0) readonly buffer InBuf {{ {bt} in_buf[]; }};
layout(set = 0, binding = 1) buffer OutBuf         {{ {bt} out_buf[]; }};
void load_elem(uint b0, uint b1, uint idx, bool v0, bool v1, out REAL ro, out REAL io) {{
    if (v0) {{ {bt} v = in_buf[b0 + idx]; ro = v.x; io = v.y; }} else {{ ro = ZERO_R; io = ZERO_R; }}
}}
void store_elem(uint b0, uint b1, uint idx, bool v0, bool v1, REAL ri, REAL ii) {{
    if (v0) out_buf[b0 + idx] = {bt}(ri, ii);
}}
"""

    tw_store = _emit_twiddle_block(
        prec, log_n1, None, "r_re", "r_im", "j", "pc.inv_m", "pos_sign",
        lambda p, vr, vi: [f"      store_elem(g_out0, g_out1, {p}u * n2, valid0, valid1, {vr}, {vi});"],
    )

    load_lines = "\n".join(
        f"    load_elem(g_in0, g_in1, {m}u * n2, valid0, valid1, r_re[{m}], r_im[{m}]);"
        for m in range(n1)
    )
    notw_lines = "\n".join(
        f"        store_elem(g_out0, g_out1, {m}u, valid0, valid1, r_re[{m}], r_im[{m}]);"
        for m in range(n1)
    )

    src = (
        hdr
        + "layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;\n"
        + "layout(constant_id = 1) const uint SPEC_HAS_TWIDDLES = 1u;\n"
        + "layout(constant_id = 2) const uint SPEC_POS_SIGN     = 1u;\n"
        + _prec_helpers(prec, False)
        + buf_decl
        + f"""
layout(push_constant) uniform PC {{
    uint chunk_bs;
    uint log_n2;
    uint in_offset;
    uint out_offset;
    {ang_type} inv_m;
}} pc;
"""
        + _emit_reg_base(log_n1)
        + f"""
void main() {{
    uint n2 = (SPEC_HAS_TWIDDLES != 0u) ? (1u << pc.log_n2) : 1u;
    uint total_threads = ((pc.chunk_bs + PACKED_BATCH - 1u) / PACKED_BATCH) * n2;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total_threads) return;

    uint n = {n1}u << ((SPEC_HAS_TWIDDLES != 0u) ? pc.log_n2 : 0u);
    uint pair_idx = (SPEC_HAS_TWIDDLES != 0u) ? (gid >> pc.log_n2) : gid;
    uint batch0 = pair_idx * PACKED_BATCH;
    bool valid0 = batch0 < pc.chunk_bs;
    bool valid1 = (PACKED_BATCH == 2u) && (batch0 + 1u < pc.chunk_bs);
    uint j = (SPEC_HAS_TWIDDLES != 0u) ? (gid & (n2 - 1u)) : 0u;

    uint g_in0  = pc.in_offset  + (valid0 ? (batch0 * n + j) : 0u);
    uint g_in1  = pc.in_offset  + (valid1 ? ((batch0 + 1u) * n + j) : (valid0 ? (batch0 * n + j) : 0u));
    uint g_out0 = pc.out_offset + (valid0 ? (batch0 * n + j) : 0u);
    uint g_out1 = pc.out_offset + (valid1 ? ((batch0 + 1u) * n + j) : (valid0 ? (batch0 * n + j) : 0u));
    bool pos_sign = (SPEC_POS_SIGN != 0u);

    REAL r_re[{n1}], r_im[{n1}];
{load_lines}
    dif_radix2_reg_base_{log_n1}(r_re, r_im, pos_sign);
    if (SPEC_HAS_TWIDDLES != 0u) {{
{tw_store}
    }} else {{
{notw_lines}
    }}
}}
"""
    )

    return {
        "target_env": "vulkan1.2" if prec == "h" else "vulkan1.1",
        "check_nocontract": (prec == "df"),
        "source": src,
    }


def generate_dif_fft_shaders() -> dict[str, dict]:
    """Generate all 56 DIF radix-2 FFT shaders (44 SinglePass + 12 Pass0/MultiStage)."""
    shaders: dict[str, dict] = {}
    for prec in ("h", "f"):
        for spec in NARROW_SPECS:
            shaders[f"dif_{prec}_{spec[0]}"] = _gen_single_pass(prec, spec)
    for prec in ("df", "d"):
        for spec in WIDE_SPECS:
            shaders[f"dif_{prec}_{spec[0]}"] = _gen_single_pass(prec, spec)
    for prec in ("h", "f", "df", "d"):
        for r in (3, 4, 5):
            shaders[f"dif_{prec}_pass0_r{r}"] = _gen_pass0(prec, r)
    return shaders
