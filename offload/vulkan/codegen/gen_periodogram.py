"""GLSL compute shader codegen for Periodogram Subgrid & Solvers (FastChi2 & IHS).

Generates 40 shaders across 4 precisions (h, f, df, d):
  - 4  subgrid_{h,f,df,d} (K_b = 8 strip-recurrence, DF phase for h/f/df; FP64 for d)
  - 4  fastchi2_gls_{h,f,df,d} (d = 1 GLS, all 4 normalizations + natural-order write)
  - 16 fastchi2_lev_d{2,3,4,5}_{h,f,df,d} (fully unrolled register-resident Levinson)
  - 4  fastchi2_lev_large_{h,f,df,d} (d >= 6 warp-tiled contiguous paired scratch Levinson)
  - 4  ihs_single_{h,f,df,d} (fused d = 1 IHS)
  - 4  ihs_accum_{h,f,df,d} (multi-harmonic IHS accumulator)
  - 4  ihs_finalize_{h,f,df,d} (multi-harmonic IHS finalizer + natural-order write)
"""

from __future__ import annotations

from common_glsl import (
    fmt_df,
    glsl_bit_reverse,
    glsl_df_math,
    glsl_f16x2_math,
    glsl_header,
    glsl_sincos2pi_df,
    glsl_sincos2pi_f32,
    glsl_sincos2pi_f64,
)


def _glsl_compact_slot_helpers() -> str:
    return (
        glsl_bit_reverse()
        + """
int nufft1_compact_idx_to_slot(int i, int mode43) {
    if (mode43 == 0) {
        return ((i >> 1) << 2) + (((i & 1) != 0) ? 3 : 0);
    } else {
        int q = i / 6;
        int r = i - 6 * q;
        int off = r + ((r >= 1) ? 1 : 0) + ((r >= 5) ? 1 : 0);
        return (q << 3) + off;
    }
}

int nufft1_slot_to_natural_idx(int slot_p, int log_nfft, int nfft, int output_shift) {
    int bin = int(bit_reverse(uint(slot_p), uint(log_nfft)));
    int first_bin = nfft - output_shift;
    int kp = (bin >= first_bin) ? (bin - nfft) : bin;
    return kp + output_shift;
}
"""
    )


def _glsl_fp64_transcendentals() -> str:
    """53-bit IEEE-754 double log, log1p, exp, pow (GLSL 450 lacks double log/exp/pow)."""
    return """
double d_nan() { return packDouble2x32(uvec2(0u, 0x7ff80000u)); }
double d_inf() { return packDouble2x32(uvec2(0u, 0x7ff00000u)); }
bool d_is_finite(double v) { return (unpackDouble2x32(v).y & 0x7ff00000u) != 0x7ff00000u; }

double d_log(double x) {
    if (!(x > 0.0lf) || !d_is_finite(x)) return 0.0lf;
    uvec2 u = unpackDouble2x32(x);
    int e = int((u.y >> 20) & 0x7ffu) - 1023;
    u.y = (u.y & 0x800fffffu) | (1023u << 20);
    double m = packDouble2x32(u);
    if (m > 1.4142135623730950488lf) {
        m *= 0.5lf;
        ++e;
    }
    double t = (m - 1.0lf) / (m + 1.0lf);
    double t2 = t * t;
    double p = 1.0lf / 19.0lf;
    p = fma(p, t2, 1.0lf / 17.0lf);
    p = fma(p, t2, 1.0lf / 15.0lf);
    p = fma(p, t2, 1.0lf / 13.0lf);
    p = fma(p, t2, 1.0lf / 11.0lf);
    p = fma(p, t2, 1.0lf /  9.0lf);
    p = fma(p, t2, 1.0lf /  7.0lf);
    p = fma(p, t2, 1.0lf /  5.0lf);
    p = fma(p, t2, 1.0lf /  3.0lf);
    p = fma(p, t2, 1.0lf);
    return fma(double(e), 0.693147180559945309417232121458lf, 2.0lf * t * p);
}

double d_log1p(double x) {
    if (abs(x) < 0.15lf) {
        double t = x / (2.0lf + x);
        double t2 = t * t;
        double p = 1.0lf / 13.0lf;
        p = fma(p, t2, 1.0lf / 11.0lf);
        p = fma(p, t2, 1.0lf /  9.0lf);
        p = fma(p, t2, 1.0lf /  7.0lf);
        p = fma(p, t2, 1.0lf /  5.0lf);
        p = fma(p, t2, 1.0lf /  3.0lf);
        p = fma(p, t2, 1.0lf);
        return 2.0lf * t * p;
    }
    return d_log(1.0lf + x);
}

double d_exp(double x) {
    if (x < -709.0lf) return 0.0lf;
    if (x >  709.0lf) return d_inf();
    double kf = floor(x * 1.44269504088896340736lf + 0.5lf);
    int k = int(kf);
    double r = fma(-kf, 0.693147180559945309417232121458lf, x);
    double p = 1.0lf / 6227020800.0lf;
    p = fma(p, r, 1.0lf / 479001600.0lf);
    p = fma(p, r, 1.0lf / 39916800.0lf);
    p = fma(p, r, 1.0lf / 3628800.0lf);
    p = fma(p, r, 1.0lf / 362880.0lf);
    p = fma(p, r, 1.0lf / 40320.0lf);
    p = fma(p, r, 1.0lf / 5040.0lf);
    p = fma(p, r, 1.0lf / 720.0lf);
    p = fma(p, r, 1.0lf / 120.0lf);
    p = fma(p, r, 1.0lf / 24.0lf);
    p = fma(p, r, 1.0lf / 6.0lf);
    p = fma(p, r, 0.5lf);
    p = fma(p, r, 1.0lf);
    p = fma(p, r, 1.0lf);
    if (k < -1022) {
        uvec2 s1 = uvec2(0u, uint(1023 - 512) << 20);
        uvec2 s2 = uvec2(0u, uint(1023 + k + 512) << 20);
        return p * packDouble2x32(s1) * packDouble2x32(s2);
    }
    uvec2 sc = uvec2(0u, uint(1023 + k) << 20);
    return p * packDouble2x32(sc);
}

double d_pow(double x, double y) {
    if (x <= 0.0lf) return 0.0lf;
    return d_exp(y * d_log(x));
}
"""


def _glsl_df_transcendentals() -> str:
    """~47-bit Dekker DF log & exp (zero FP64 ops, matches levinson.cpp)."""
    return f"""
float f_nan() {{ return uintBitsToFloat(0x7fc00000u); }}
float f_inf() {{ return uintBitsToFloat(0x7f800000u); }}
bool f_is_finite(float v) {{ return (floatBitsToUint(v) & 0x7f800000u) != 0x7f800000u; }}

const df_t kLn2Df = {fmt_df(0.693147180559945309417232121458)};

df_t df_log(df_t x) {{
    if (x.x <= 0.0f) return vec2(0.0f, 0.0f);
    uint u = floatBitsToUint(x.x);
    int e = int((u >> 23) & 0xFFu) - 127;
    uint scale_u = uint(127 - e) << 23;
    df_t m = df_scale_pow2(x, uintBitsToFloat(scale_u));
    if (m.x > 1.41421356f) {{
        m = df_scale_pow2(m, 0.5f);
        ++e;
    }}
    df_t t = df_div(df_sub(m, vec2(1.0f, 0.0f)), df_add(m, vec2(1.0f, 0.0f)));
    df_t t2 = df_sqr(t);
    df_t poly = vec2(1.0f / 15.0f, 0.0f);
    poly = df_add(df_mul(poly, t2), {fmt_df(1.0 / 13.0)});
    poly = df_add(df_mul(poly, t2), {fmt_df(1.0 / 11.0)});
    poly = df_add(df_mul(poly, t2), {fmt_df(1.0 /  9.0)});
    poly = df_add(df_mul(poly, t2), {fmt_df(1.0 /  7.0)});
    poly = df_add(df_mul(poly, t2), {fmt_df(1.0 /  5.0)});
    poly = df_add(df_mul(poly, t2), {fmt_df(1.0 /  3.0)});
    poly = df_add(df_mul(poly, t2), vec2(1.0f, 0.0f));
    df_t ln_m = df_scale_pow2(df_mul(t, poly), 2.0f);
    return df_add(ln_m, df_mul_f(kLn2Df, float(e)));
}}

df_t df_log1p(df_t x) {{
    if (abs(x.x) < 0.15f) {{
        df_t t = df_div(x, df_add(vec2(2.0f, 0.0f), x));
        df_t t2 = df_sqr(t);
        df_t poly = {fmt_df(1.0 / 11.0)};
        poly = df_add(df_mul(poly, t2), {fmt_df(1.0 / 9.0)});
        poly = df_add(df_mul(poly, t2), {fmt_df(1.0 / 7.0)});
        poly = df_add(df_mul(poly, t2), {fmt_df(1.0 / 5.0)});
        poly = df_add(df_mul(poly, t2), {fmt_df(1.0 / 3.0)});
        poly = df_add(df_mul(poly, t2), vec2(1.0f, 0.0f));
        return df_scale_pow2(df_mul(t, poly), 2.0f);
    }}
    return df_log(df_add(vec2(1.0f, 0.0f), x));
}}

df_t df_exp(df_t x) {{
    float xf = x.x + x.y;
    if (xf < -87.0f) return vec2(0.0f, 0.0f);
    if (xf >  88.0f) return vec2(3.4028234e38f, 0.0f);
    float kf = floor(xf * 1.4426950408889634f + 0.5f);
    int k = int(kf);
    df_t r = df_sub(x, df_mul_f(kLn2Df, kf));
    df_t p = {fmt_df(1.0 / 39916800.0)};
    p = df_add(df_mul(p, r), {fmt_df(1.0 / 3628800.0)});
    p = df_add(df_mul(p, r), {fmt_df(1.0 / 362880.0)});
    p = df_add(df_mul(p, r), {fmt_df(1.0 / 40320.0)});
    p = df_add(df_mul(p, r), {fmt_df(1.0 / 5040.0)});
    p = df_add(df_mul(p, r), {fmt_df(1.0 / 720.0)});
    p = df_add(df_mul(p, r), {fmt_df(1.0 / 120.0)});
    p = df_add(df_mul(p, r), {fmt_df(1.0 / 24.0)});
    p = df_add(df_mul(p, r), {fmt_df(1.0 / 6.0)});
    p = df_add(df_mul(p, r), vec2(0.5f, 0.0f));
    p = df_add(df_mul(p, r), vec2(1.0f, 0.0f));
    p = df_add(df_mul(p, r), vec2(1.0f, 0.0f));
    uint scale_u = uint(127 + k) << 23;
    return df_scale_pow2(p, uintBitsToFloat(scale_u));
}}
"""


def _glsl_f32_helpers(singularity_thresh: str) -> str:
    return f"""
float f_nan() {{ return uintBitsToFloat(0x7fc00000u); }}
float f_inf() {{ return uintBitsToFloat(0x7f800000u); }}
bool f_is_finite(float v) {{ return (floatBitsToUint(v) & 0x7f800000u) != 0x7f800000u; }}
bool cond_is_singular_f(float cb) {{ return !f_is_finite(cb) || cb < 0.0f || cb > {singularity_thresh}; }}

float f_log1p(float x) {{
    if (abs(x) < 0.15f) {{
        float t = x / (2.0f + x);
        float t2 = t * t;
        float p = fma(t2, fma(t2, fma(t2, 1.0f / 7.0f, 1.0f / 5.0f), 1.0f / 3.0f), 1.0f);
        return 2.0f * t * p;
    }}
    return log(1.0f + x);
}}
"""


# ============================================================================
# 1. Subgrid Phase-Modulation Shaders
# ============================================================================

def _gen_subgrid(prec: str) -> dict:
    if prec == "d":
        src = (
            glsl_header("d")
            + glsl_sincos2pi_f64()
            + """
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer XBuf   { double d_x[]; };
layout(std430, binding = 1) readonly  buffer WBuf   { double d_w[]; };
layout(std430, binding = 2) readonly  buffer YwBuf  { double d_yw[]; };
layout(std430, binding = 3) writeonly buffer SrcBuf { dvec2  d_src[]; };

layout(push_constant) uniform Push {
    double f0_base;
    double f_step;
    int    M;
    int    b_count;
    int    strip_b;
    int    has_yw;
    uint   x_off;
    uint   w_off;
    uint   yw_off;
} pc;

void main() {
    int m  = int(gl_GlobalInvocationID.x);
    int b0 = int(gl_WorkGroupID.y) * pc.strip_b;
    if (m >= pc.M || b0 >= pc.b_count) return;
    int b_end = min(b0 + pc.strip_b, pc.b_count);

    double xm  = d_x[pc.x_off + uint(m)];
    double wm  = d_w[pc.w_off + uint(m)];
    double ywm = (pc.has_yw != 0) ? d_yw[pc.yw_off + uint(m)] : 0.0lf;

    double fb0 = fma(double(b0), pc.f_step, pc.f0_base);
    double c, s;
    sincos2pi_branchless(xm * fb0, false, c, s);

    uint M_u = uint(pc.M);
    uint yw_stride = uint(pc.b_count) * M_u;
    uint out_idx = uint(b0) * M_u + uint(m);
    d_src[out_idx] = dvec2(wm * c, wm * s);
    if (pc.has_yw != 0) d_src[yw_stride + out_idx] = dvec2(ywm * c, ywm * s);

    if (b0 + 1 < b_end) {
        double dc, ds;
        sincos2pi_branchless(xm * pc.f_step, false, dc, ds);
        for (int b = b0 + 1; b < b_end; ++b) {
            double nc = fma(c, dc, -(s * ds));
            double ns = fma(c, ds,   s * dc);
            c = nc; s = ns;
            out_idx += M_u;
            d_src[out_idx] = dvec2(wm * c, wm * s);
            if (pc.has_yw != 0) d_src[yw_stride + out_idx] = dvec2(ywm * c, ywm * s);
        }
    }
}
"""
        )
        return {"source": src, "check_nocontract": False}

    if prec == "df":
        hdr = glsl_header("df") + glsl_df_math() + glsl_sincos2pi_df()
        sc_t, buf_t = "vec2", "vec4"
        body = """
    df_t xm  = d_x[pc.x_off + uint(m)];
    df_t wm  = d_w[pc.w_off + uint(m)];
    df_t ywm = (pc.has_yw != 0) ? d_yw[pc.yw_off + uint(m)] : vec2(0.0f, 0.0f);

    df_t fb0 = df_add(pc.f0_base, df_mul_f(pc.f_step, float(b0)));
    df_t c, s;
    sincos2pidf_branchless(df_mul(xm, fb0), false, c, s);

    uint M_u = uint(pc.M);
    uint yw_stride = uint(pc.b_count) * M_u;
    uint out_idx = uint(b0) * M_u + uint(m);
    df_t wc = df_mul(wm, c), ws = df_mul(wm, s);
    d_src[out_idx] = vec4(wc.x, wc.y, ws.x, ws.y);
    if (pc.has_yw != 0) {
        df_t ywc = df_mul(ywm, c), yws = df_mul(ywm, s);
        d_src[yw_stride + out_idx] = vec4(ywc.x, ywc.y, yws.x, yws.y);
    }

    if (b0 + 1 < b_end) {
        df_t dc, ds;
        sincos2pidf_branchless(df_mul(xm, pc.f_step), false, dc, ds);
        for (int b = b0 + 1; b < b_end; ++b) {
            df_t nc, ns;
            df_cmul_fused(c, s, dc, ds, nc, ns);
            c = nc; s = ns;
            out_idx += M_u;
            df_t wc2 = df_mul(wm, c), ws2 = df_mul(wm, s);
            d_src[out_idx] = vec4(wc2.x, wc2.y, ws2.x, ws2.y);
            if (pc.has_yw != 0) {
                df_t ywc2 = df_mul(ywm, c), yws2 = df_mul(ywm, s);
                d_src[yw_stride + out_idx] = vec4(ywc2.x, ywc2.y, yws2.x, yws2.y);
            }
        }
    }"""
    elif prec == "h":
        hdr = glsl_header("h", fp16_storage=True) + glsl_df_math() + glsl_sincos2pi_f32()
        sc_t, buf_t = "float16_t", "uint"
        body = """
    df_t xm   = d_x[pc.x_off + uint(m)];
    float wm  = float(d_w[pc.w_off + uint(m)]);
    float ywm = (pc.has_yw != 0) ? float(d_yw[pc.yw_off + uint(m)]) : 0.0f;

    df_t fb0 = df_add(pc.f0_base, df_mul_f(pc.f_step, float(b0)));
    df_t ph0 = df_mul(xm, fb0);
    df_t fr0 = df_sub(ph0, df_floor(ph0));
    float c, s;
    sincos2pif_branchless(fr0.x + fr0.y, false, c, s);

    uint M_u = uint(pc.M);
    uint yw_stride = uint(pc.b_count) * M_u;
    uint out_idx = uint(b0) * M_u + uint(m);
    d_src[out_idx] = packHalf2x16(vec2(wm * c, wm * s));
    if (pc.has_yw != 0) d_src[yw_stride + out_idx] = packHalf2x16(vec2(ywm * c, ywm * s));

    if (b0 + 1 < b_end) {
        df_t dph = df_mul(xm, pc.f_step);
        df_t dfr = df_sub(dph, df_floor(dph));
        float dc, ds;
        sincos2pif_branchless(dfr.x + dfr.y, false, dc, ds);
        for (int b = b0 + 1; b < b_end; ++b) {
            float nc = fma(c, dc, -(s * ds));
            float ns = fma(c, ds,   s * dc);
            c = nc; s = ns;
            out_idx += M_u;
            d_src[out_idx] = packHalf2x16(vec2(wm * c, wm * s));
            if (pc.has_yw != 0) d_src[yw_stride + out_idx] = packHalf2x16(vec2(ywm * c, ywm * s));
        }
    }"""
    else:  # f
        hdr = glsl_header("f") + glsl_df_math() + glsl_sincos2pi_f32()
        sc_t, buf_t = "float", "vec2"
        body = """
    df_t xm   = d_x[pc.x_off + uint(m)];
    float wm  = d_w[pc.w_off + uint(m)];
    float ywm = (pc.has_yw != 0) ? d_yw[pc.yw_off + uint(m)] : 0.0f;

    df_t fb0 = df_add(pc.f0_base, df_mul_f(pc.f_step, float(b0)));
    df_t ph0 = df_mul(xm, fb0);
    df_t fr0 = df_sub(ph0, df_floor(ph0));
    float c, s;
    sincos2pif_branchless(fr0.x + fr0.y, false, c, s);

    uint M_u = uint(pc.M);
    uint yw_stride = uint(pc.b_count) * M_u;
    uint out_idx = uint(b0) * M_u + uint(m);
    d_src[out_idx] = vec2(wm * c, wm * s);
    if (pc.has_yw != 0) d_src[yw_stride + out_idx] = vec2(ywm * c, ywm * s);

    if (b0 + 1 < b_end) {
        df_t dph = df_mul(xm, pc.f_step);
        df_t dfr = df_sub(dph, df_floor(dph));
        float dc, ds;
        sincos2pif_branchless(dfr.x + dfr.y, false, dc, ds);
        for (int b = b0 + 1; b < b_end; ++b) {
            float nc = fma(c, dc, -(s * ds));
            float ns = fma(c, ds,   s * dc);
            c = nc; s = ns;
            out_idx += M_u;
            d_src[out_idx] = vec2(wm * c, wm * s);
            if (pc.has_yw != 0) d_src[yw_stride + out_idx] = vec2(ywm * c, ywm * s);
        }
    }"""

    src = (
        hdr
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer XBuf   {{ vec2   d_x[]; }};
layout(std430, binding = 1) readonly  buffer WBuf   {{ {sc_t} d_w[]; }};
layout(std430, binding = 2) readonly  buffer YwBuf  {{ {sc_t} d_yw[]; }};
layout(std430, binding = 3) writeonly buffer SrcBuf {{ {buf_t} d_src[]; }};

layout(push_constant) uniform Push {{
    vec2 f0_base;
    vec2 f_step;
    int  M;
    int  b_count;
    int  strip_b;
    int  has_yw;
    uint x_off;
    uint w_off;
    uint yw_off;
}} pc;

void main() {{
    int m  = int(gl_GlobalInvocationID.x);
    int b0 = int(gl_WorkGroupID.y) * pc.strip_b;
    if (m >= pc.M || b0 >= pc.b_count) return;
    int b_end = min(b0 + pc.strip_b, pc.b_count);
{body}
}}
"""
    )
    return {"source": src, "check_nocontract": True}


# ============================================================================
# 2. Normalization & Precision Helpers for FastChi2 Solvers
# ============================================================================

def _glsl_fastchi2_norm_helpers(prec: str) -> str:
    if prec == "d":
        return (
            _glsl_fp64_transcendentals()
            + """
bool cond_is_singular(double cb) { return !d_is_finite(cb) || cb < 0.0lf || cb > 1.0e13lf; }

double cybenko_kappa_to_S(double kappa) {
    if (kappa <= 1.0lf || !d_is_finite(kappa)) return 0.0lf;
    if (kappa > 5000.0lf) kappa = 5000.0lf;
    double pow_k = d_pow(kappa, pc.cyb_inv_2d);
    double k_val = (pow_k - 1.0lf) / (pow_k + 1.0lf);
    return pc.cyb_sum_N_eff * (k_val * k_val);
}

double nll_exact_pochhammer(int d, double r2, double b) {
    if (!d_is_finite(r2) || r2 <= 0.0lf) return 0.0lf;
    if (r2 > 0.9999999999999lf) r2 = 0.9999999999999lf;
    double term0 = -b * d_log1p(-r2);
    if (d <= 1) return (term0 > 0.0lf && d_is_finite(term0)) ? term0 : 0.0lf;
    double c = 1.0lf, s_minus_1 = 0.0lf;
    for (int k = 1; k < d; ++k) {
        c *= ((b + double(k - 1)) / double(k)) * r2;
        s_minus_1 += c;
    }
    double nll = term0 - d_log1p(s_minus_1);
    return (nll > 0.0lf && d_is_finite(nll)) ? nll : 0.0lf;
}

double nll_cybenko(double kappa, int d) {
    double s = cybenko_kappa_to_S(kappa);
    if (s <= 0.0lf || !d_is_finite(s)) return 0.0lf;
    double k_eff = pc.cyb_k_eff, th_eff = pc.cyb_theta_eff, lg_k = pc.cyb_lgamma_k;
    if (d <= 2) {
        double z = s / th_eff;
        int order = 2 * d;
        double delta = k_eff - double(order);
        double c = 1.0lf, poly = 1.0lf, d_poly = 0.0lf;
        for (int k = 1; k < order; ++k) {
            c *= (z / double(k));
            poly += c;
            if (abs(delta) > 1.0e-5lf) {
                double H_k = 0.0lf;
                for (int i = 1; i <= k; ++i) H_k += 1.0lf / double(i);
                d_poly -= c * H_k;
            }
        }
        double nll = z - d_log(poly);
        if (abs(delta) > 1.0e-5lf) nll -= delta * (d_log(z) + d_poly / poly);
        return (nll > 0.0lf && d_is_finite(nll)) ? nll : 0.0lf;
    } else {
        double a = k_eff, z = s / th_eff;
        if (z < a + 3.0lf) {
            double c = 1.0lf, sum = 1.0lf;
            for (int k = 1; k < 60; ++k) {
                c *= z / (a + double(k - 1));
                sum += c;
                if (c < 1.0e-14lf * sum) break;
            }
            double lg_inc = a * d_log(z) - z - d_log(a) + d_log(sum);
            double nll = lg_k - lg_inc;
            return (nll > 0.0lf && d_is_finite(nll)) ? nll : 0.0lf;
        } else {
            double inv_z = 1.0lf / z;
            double t1 = (a - 1.0lf) * inv_z;
            double t2 = t1 * ((a - 2.0lf) * inv_z);
            double t3 = t2 * ((a - 3.0lf) * inv_z);
            double t4 = t3 * ((a - 4.0lf) * inv_z);
            double poly = 1.0lf + t1 + t2 + t3 + t4;
            double nll = z - (a - 1.0lf) * d_log(z) + lg_k - d_log(poly);
            return (nll > 0.0lf && d_is_finite(nll)) ? nll : 0.0lf;
        }
    }
}

double compute_bayes_laplace(double delta) {
    if (delta >= 0.0lf) return delta + 0.693147180559945309417lf;
    double inner = 1.0lf - 0.5lf * d_exp(delta);
    if (inner <= 0.0lf) return 0.0lf;
    return -d_log(inner);
}

double normalize_power(double dot, double cond_bound, int norm, int degree) {
    if (!d_is_finite(dot)) return d_nan();
    if (dot <= 0.0lf) return 0.0lf;
    double r2 = dot * pc.inv_chi2_ref;
    if (!d_is_finite(r2) || r2 >= 1.0lf) return d_nan();
    if (norm == 0) return r2;
    if (norm == 1) {
        double d1 = 2.0lf * double(degree) + 1.0lf;
        double d2 = double(pc.M) - 1.0lf - d1;
        double residual = max(1.0lf - r2, 1e-32lf);
        return (d2 * r2) / (d1 * residual);
    }
    if (norm == 2) return nll_exact_pochhammer(degree, r2, pc.b_param);
    double nll_r2 = nll_exact_pochhammer(degree, r2, pc.b_param);
    double kappa = cond_bound / double(2 * degree + 1);
    double nll_cond = nll_cybenko(kappa, degree);
    return compute_bayes_laplace(nll_r2 - nll_cond);
}
"""
        )

    if prec == "df":
        return (
            _glsl_df_transcendentals()
            + f"""
bool cond_is_singular_f(float cb) {{ return !f_is_finite(cb) || cb < 0.0f || cb > 1.0e13f; }}

df_t cybenko_kappa_to_S(df_t kappa) {{
    if (!f_is_finite(kappa.x) || kappa.x < 1.0f || (kappa.x == 1.0f && kappa.y <= 0.0f)) return vec2(0.0f, 0.0f);
    if (kappa.x > 5000.0f) kappa = vec2(5000.0f, 0.0f);
    df_t pow_k = df_exp(df_mul(df_log(kappa), pc.cyb_inv_2d));
    df_t one = vec2(1.0f, 0.0f);
    df_t k_val = df_div(df_sub(pow_k, one), df_add(pow_k, one));
    return df_mul(pc.cyb_sum_N_eff, df_sqr(k_val));
}}

df_t nll_exact_pochhammer(int d, df_t r2, df_t b) {{
    if (!f_is_finite(r2.x) || r2.x <= 0.0f) return vec2(0.0f, 0.0f);
    df_t kMaxR2 = {fmt_df(0.999999999999)};
    if (r2.x > kMaxR2.x || (r2.x == kMaxR2.x && r2.y > kMaxR2.y)) r2 = kMaxR2;
    df_t one = vec2(1.0f, 0.0f);
    df_t term0 = df_mul(df_neg(b), df_log1p(df_neg(r2)));
    if (d <= 1) return (term0.x > 0.0f && f_is_finite(term0.x)) ? term0 : vec2(0.0f, 0.0f);
    df_t c = one, s_minus_1 = vec2(0.0f, 0.0f);
    for (int k = 1; k < d; ++k) {{
        df_t fac = df_div(df_add(b, vec2(float(k - 1), 0.0f)), vec2(float(k), 0.0f));
        c = df_mul(df_mul(c, fac), r2);
        s_minus_1 = df_add(s_minus_1, c);
    }}
    df_t nll = df_sub(term0, df_log1p(s_minus_1));
    return (nll.x > 0.0f && f_is_finite(nll.x)) ? nll : vec2(0.0f, 0.0f);
}}

df_t nll_cybenko(df_t kappa, int d) {{
    df_t s = cybenko_kappa_to_S(kappa);
    if (!f_is_finite(s.x) || s.x <= 0.0f) return vec2(0.0f, 0.0f);
    df_t k_eff = pc.cyb_k_eff, th_eff = pc.cyb_theta_eff, lg_k = pc.cyb_lgamma_k;
    df_t z = df_div(s, th_eff);
    df_t one = vec2(1.0f, 0.0f);
    if (d <= 2) {{
        int order = 2 * d;
        df_t delta = df_sub(k_eff, vec2(float(order), 0.0f));
        bool need_delta = (abs(delta.x) > 1.0e-5f);
        df_t c = one, poly = one, d_poly = vec2(0.0f, 0.0f);
        for (int k = 1; k < order; ++k) {{
            c = df_mul(c, df_div(z, vec2(float(k), 0.0f)));
            poly = df_add(poly, c);
            if (need_delta) {{
                df_t H_k = vec2(0.0f, 0.0f);
                for (int i = 1; i <= k; ++i) H_k = df_add(H_k, df_div(one, vec2(float(i), 0.0f)));
                d_poly = df_sub(d_poly, df_mul(c, H_k));
            }}
        }}
        df_t nll = df_sub(z, df_log(poly));
        if (need_delta) nll = df_sub(nll, df_mul(delta, df_add(df_log(z), df_div(d_poly, poly))));
        return (nll.x > 0.0f && f_is_finite(nll.x)) ? nll : vec2(0.0f, 0.0f);
    }} else {{
        df_t a = k_eff;
        if (z.x < a.x + 3.0f) {{
            df_t c = one, sum = one;
            for (int k = 1; k < 60; ++k) {{
                c = df_mul(c, df_div(z, df_add(a, vec2(float(k - 1), 0.0f))));
                sum = df_add(sum, c);
                if (c.x < 1.0e-14f * sum.x) break;
            }}
            df_t lg_inc = df_add(df_sub(df_sub(df_mul(a, df_log(z)), z), df_log(a)), df_log(sum));
            df_t nll = df_sub(lg_k, lg_inc);
            return (nll.x > 0.0f && f_is_finite(nll.x)) ? nll : vec2(0.0f, 0.0f);
        }} else {{
            df_t inv_z = df_div(one, z);
            df_t t1 = df_mul(df_sub(a, one), inv_z);
            df_t t2 = df_mul(t1, df_mul(df_sub(a, vec2(2.0f, 0.0f)), inv_z));
            df_t t3 = df_mul(t2, df_mul(df_sub(a, vec2(3.0f, 0.0f)), inv_z));
            df_t t4 = df_mul(t3, df_mul(df_sub(a, vec2(4.0f, 0.0f)), inv_z));
            df_t poly = df_add(df_add(one, t1), df_add(df_add(t2, t3), t4));
            df_t nll = df_sub(df_add(df_sub(z, df_mul(df_sub(a, one), df_log(z))), lg_k), df_log(poly));
            return (nll.x > 0.0f && f_is_finite(nll.x)) ? nll : vec2(0.0f, 0.0f);
        }}
    }}
}}

df_t compute_bayes_laplace(df_t delta) {{
    if (delta.x > 0.0f || (delta.x == 0.0f && delta.y >= 0.0f)) return df_add(delta, kLn2Df);
    df_t inner = df_sub(vec2(1.0f, 0.0f), df_scale_pow2(df_exp(delta), 0.5f));
    if (inner.x <= 0.0f) return vec2(0.0f, 0.0f);
    return df_neg(df_log(inner));
}}

df_t normalize_power(df_t dot, df_t cond_bound, int norm, int degree) {{
    if (!f_is_finite(dot.x) || !f_is_finite(dot.y)) return vec2(f_nan(), f_nan());
    if (dot.x < 0.0f || (dot.x == 0.0f && dot.y <= 0.0f)) return vec2(0.0f, 0.0f);
    df_t r2 = df_mul(dot, pc.inv_chi2_ref);
    if (!f_is_finite(r2.x) || !f_is_finite(r2.y) || r2.x > 1.0f || (r2.x == 1.0f && r2.y >= 0.0f)) return vec2(f_nan(), f_nan());
    if (norm == 0) return r2;
    if (norm == 1) {{
        df_t d1 = vec2(float(2 * degree + 1), 0.0f);
        df_t d2 = vec2(float(pc.M - 1 - (2 * degree + 1)), 0.0f);
        df_t residual = df_sub(vec2(1.0f, 0.0f), r2);
        if (residual.x < 1e-32f) residual = vec2(1e-32f, 0.0f);
        return df_div(df_mul(d2, r2), df_mul(d1, residual));
    }}
    if (norm == 2) return nll_exact_pochhammer(degree, r2, pc.b_param);
    df_t nll_r2 = nll_exact_pochhammer(degree, r2, pc.b_param);
    df_t kappa = df_div(cond_bound, vec2(float(2 * degree + 1), 0.0f));
    df_t nll_cond = nll_cybenko(kappa, degree);
    return compute_bayes_laplace(df_sub(nll_r2, nll_cond));
}}
"""
        )

    # h & f (float normalization)
    sthresh = "500.0f" if prec == "h" else "5.0e5f"
    return (
        _glsl_f32_helpers(sthresh)
        + """
float cybenko_kappa_to_S(float kappa) {
    if (kappa <= 1.0f || !f_is_finite(kappa)) return 0.0f;
    if (kappa > 5000.0f) kappa = 5000.0f;
    float pow_k = pow(kappa, pc.cyb_inv_2d);
    float k_val = (pow_k - 1.0f) / (pow_k + 1.0f);
    return pc.cyb_sum_N_eff * (k_val * k_val);
}

float nll_exact_pochhammer(int d, float r2, float b) {
    if (!f_is_finite(r2) || r2 <= 0.0f) return 0.0f;
    if (r2 > 0.9999999f) r2 = 0.9999999f;
    float term0 = -b * f_log1p(-r2);
    if (d <= 1) return (term0 > 0.0f && f_is_finite(term0)) ? term0 : 0.0f;
    float c = 1.0f, s_minus_1 = 0.0f;
    for (int k = 1; k < d; ++k) {
        c *= ((b + float(k - 1)) / float(k)) * r2;
        s_minus_1 += c;
    }
    float nll = term0 - f_log1p(s_minus_1);
    return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
}

float nll_cybenko(float kappa, int d) {
    float s = cybenko_kappa_to_S(kappa);
    if (s <= 0.0f || !f_is_finite(s)) return 0.0f;
    float k_eff = pc.cyb_k_eff, th_eff = pc.cyb_theta_eff, lg_k = pc.cyb_lgamma_k;
    if (d <= 2) {
        float z = s / th_eff;
        int order = 2 * d;
        float delta = k_eff - float(order);
        float c = 1.0f, poly = 1.0f, d_poly = 0.0f;
        for (int k = 1; k < order; ++k) {
            c *= (z / float(k));
            poly += c;
            if (abs(delta) > 1.0e-5f) {
                float H_k = 0.0f;
                for (int i = 1; i <= k; ++i) H_k += 1.0f / float(i);
                d_poly -= c * H_k;
            }
        }
        float nll = z - log(poly);
        if (abs(delta) > 1.0e-5f) nll -= delta * (log(z) + d_poly / poly);
        return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
    } else {
        float a = k_eff, z = s / th_eff;
        if (z < a + 3.0f) {
            float c = 1.0f, sum = 1.0f;
            for (int k = 1; k < 60; ++k) {
                c *= z / (a + float(k - 1));
                sum += c;
                if (c < 1.0e-7f * sum) break;
            }
            float lg_inc = a * log(z) - z - log(a) + log(sum);
            float nll = lg_k - lg_inc;
            return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
        } else {
            float inv_z = 1.0f / z;
            float t1 = (a - 1.0f) * inv_z;
            float t2 = t1 * ((a - 2.0f) * inv_z);
            float t3 = t2 * ((a - 3.0f) * inv_z);
            float t4 = t3 * ((a - 4.0f) * inv_z);
            float poly = 1.0f + t1 + t2 + t3 + t4;
            float nll = z - (a - 1.0f) * log(z) + lg_k - log(poly);
            return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
        }
    }
}

float compute_bayes_laplace(float delta) {
    if (delta >= 0.0f) return delta + 0.6931471805599453f;
    float inner = 1.0f - 0.5f * exp(delta);
    if (inner <= 0.0f) return 0.0f;
    return -log(inner);
}

float normalize_power(float dot, float cond_bound, int norm, int degree) {
    if (!f_is_finite(dot)) return f_nan();
    if (dot <= 0.0f) return 0.0f;
    float r2 = dot * pc.inv_chi2_ref;
    if (!f_is_finite(r2) || r2 >= 1.0f) return f_nan();
    if (norm == 0) return r2;
    if (norm == 1) {
        float d1 = 2.0f * float(degree) + 1.0f;
        float d2 = float(pc.M) - 1.0f - d1;
        float residual = max(1.0f - r2, 1e-32f);
        return (d2 * r2) / (d1 * residual);
    }
    if (norm == 2) return nll_exact_pochhammer(degree, r2, pc.b_param);
    float nll_r2 = nll_exact_pochhammer(degree, r2, pc.b_param);
    float kappa = cond_bound / float(2 * degree + 1);
    float nll_cond = nll_cybenko(kappa, degree);
    return compute_bayes_laplace(nll_r2 - nll_cond);
}
"""
    )


def _solve_prec_macros(prec: str) -> dict:
    if prec == "h":
        return {
            "hdr": glsl_header("h", fp16_storage=True) + glsl_f16x2_math(),
            "buf_t": "uint",
            "sc_t": "float16_t",
            "real_t": "f16vec2",
            "out_t": "float",
            "push_sc_t": "float",
            "scr_t": "uint",
            "scr_pair_t": "uvec2",
            "check_nc": False,
            "ops": """
#define S_REAL f16vec2
#define S_ZERO kZero_f16x2
#define S_ONE  kOne_f16x2
#define S_HALF kHalf_f16x2
#define S_TWO  kTwo_f16x2
S_REAL s_add(S_REAL a, S_REAL b) { return a + b; }
S_REAL s_sub(S_REAL a, S_REAL b) { return a - b; }
S_REAL s_mul(S_REAL a, S_REAL b) { return a * b; }
S_REAL s_neg(S_REAL a) { return -a; }
S_REAL s_div(S_REAL a, S_REAL b) { return f16vec2(vec2(a) / vec2(b)); }
S_REAL s_sqrt(S_REAL a) { return f16vec2(sqrt(vec2(a))); }
S_REAL s_fma(S_REAL a, S_REAL b, S_REAL c) { return fma(a, b, c); }
S_REAL s_abs2(S_REAL r, S_REAL i) { return fma(r, r, i * i); }
void s_cmac(S_REAL ar, S_REAL ai, S_REAL br, S_REAL bi, inout S_REAL acc_r, inout S_REAL acc_i) {
    acc_r = fma(ar, br, fma(-ai, bi, acc_r));
    acc_i = fma(ar, bi, fma( ai, br, acc_i));
}
void s_cmac_conj(S_REAL gr, S_REAL gi, S_REAL ar, S_REAL ai,
                 S_REAL base_r, S_REAL base_i, out S_REAL nr, out S_REAL ni) {
    nr = fma(gr, ar, fma( gi, ai, base_r));
    ni = fma(gi, ar, fma(-gr, ai, base_i));
}
uint s_pack(S_REAL v) { return f16x2_to_u32(v); }
S_REAL s_unpack(uint u) { return u32_to_f16x2(u); }
uvec2 s_pack_pair(S_REAL ar, S_REAL ai) { return uvec2(f16x2_to_u32(ar), f16x2_to_u32(ai)); }
void s_unpack_pair(uvec2 p, out S_REAL ar, out S_REAL ai) { ar = u32_to_f16x2(p.x); ai = u32_to_f16x2(p.y); }
S_REAL s_splat(float v) { float16_t h = float16_t(v); return f16vec2(h, h); }
void load_trig_slot(uint row_off, uint bin_idx, out S_REAL re_o, out S_REAL im_o) {
    uint u0 = d_trig[row_off + bin_idx];
    uint u1 = d_trig[row_off + bin_idx + 1u];
    f16x2_load_pair_u32(u0, u1, re_o, im_o);
}
S_REAL load_slot_phi(int slot_i) { return f16vec2(d_slot_phi_hat[slot_i], d_slot_phi_hat[slot_i + 1]); }
S_REAL load_slot_dec(int slot_i) { return f16vec2(d_slot_deconv[slot_i], d_slot_deconv[slot_i + 1]); }
""",
        }
    if prec == "f":
        return {
            "hdr": glsl_header("f"),
            "buf_t": "vec2",
            "sc_t": "float",
            "real_t": "float",
            "out_t": "float",
            "push_sc_t": "float",
            "scr_t": "float",
            "scr_pair_t": "vec2",
            "check_nc": False,
            "ops": """
#define S_REAL float
#define S_ZERO 0.0f
#define S_ONE  1.0f
#define S_HALF 0.5f
#define S_TWO  2.0f
S_REAL s_add(S_REAL a, S_REAL b) { return a + b; }
S_REAL s_sub(S_REAL a, S_REAL b) { return a - b; }
S_REAL s_mul(S_REAL a, S_REAL b) { return a * b; }
S_REAL s_neg(S_REAL a) { return -a; }
S_REAL s_div(S_REAL a, S_REAL b) { return a / b; }
S_REAL s_sqrt(S_REAL a) { return sqrt(a); }
S_REAL s_fma(S_REAL a, S_REAL b, S_REAL c) { return fma(a, b, c); }
S_REAL s_abs2(S_REAL r, S_REAL i) { return fma(r, r, i * i); }
void s_cmac(S_REAL ar, S_REAL ai, S_REAL br, S_REAL bi, inout S_REAL acc_r, inout S_REAL acc_i) {
    acc_r = fma(ar, br, fma(-ai, bi, acc_r));
    acc_i = fma(ar, bi, fma( ai, br, acc_i));
}
void s_cmac_conj(S_REAL gr, S_REAL gi, S_REAL ar, S_REAL ai,
                 S_REAL base_r, S_REAL base_i, out S_REAL nr, out S_REAL ni) {
    nr = fma(gr, ar, fma( gi, ai, base_r));
    ni = fma(gi, ar, fma(-gr, ai, base_i));
}
float s_pack(S_REAL v) { return v; }
S_REAL s_unpack(float u) { return u; }
vec2 s_pack_pair(S_REAL ar, S_REAL ai) { return vec2(ar, ai); }
void s_unpack_pair(vec2 p, out S_REAL ar, out S_REAL ai) { ar = p.x; ai = p.y; }
S_REAL s_splat(float v) { return v; }
void load_trig_slot(uint row_off, uint bin_idx, out S_REAL re_o, out S_REAL im_o) {
    vec2 v = d_trig[row_off + bin_idx];
    re_o = v.x; im_o = v.y;
}
S_REAL load_slot_phi(int slot_i) { return d_slot_phi_hat[slot_i]; }
S_REAL load_slot_dec(int slot_i) { return d_slot_deconv[slot_i]; }
""",
        }
    if prec == "df":
        return {
            "hdr": glsl_header("df") + glsl_df_math(),
            "buf_t": "vec4",
            "sc_t": "vec2",
            "real_t": "df_t",
            "out_t": "vec2",
            "push_sc_t": "vec2",
            "scr_t": "vec2",
            "scr_pair_t": "vec4",
            "check_nc": True,
            "ops": """
#define S_REAL df_t
#define S_ZERO vec2(0.0f, 0.0f)
#define S_ONE  vec2(1.0f, 0.0f)
#define S_HALF vec2(0.5f, 0.0f)
#define S_TWO  vec2(2.0f, 0.0f)
S_REAL s_add(S_REAL a, S_REAL b) { return df_add(a, b); }
S_REAL s_sub(S_REAL a, S_REAL b) { return df_sub(a, b); }
S_REAL s_mul(S_REAL a, S_REAL b) { return df_mul(a, b); }
S_REAL s_neg(S_REAL a) { return df_neg(a); }
S_REAL s_div(S_REAL a, S_REAL b) { return df_div(a, b); }
S_REAL s_sqrt(S_REAL a) { return df_sqrt(a); }
S_REAL s_fma(S_REAL a, S_REAL b, S_REAL c) { return df_add(df_mul(a, b), c); }
S_REAL s_abs2(S_REAL r, S_REAL i) { return df_add(df_sqr(r), df_sqr(i)); }
void s_cmac(S_REAL ar, S_REAL ai, S_REAL br, S_REAL bi, inout S_REAL acc_r, inout S_REAL acc_i) {
    df_t pr, pi;
    df_cmul_fused(ar, ai, br, bi, pr, pi);
    acc_r = df_add(acc_r, pr);
    acc_i = df_add(acc_i, pi);
}
void s_cmac_conj(S_REAL gr, S_REAL gi, S_REAL ar, S_REAL ai,
                 S_REAL base_r, S_REAL base_i, out S_REAL nr, out S_REAL ni) {
    df_t pr, pi;
    df_cmul_fused(gr, gi, ar, df_neg(ai), pr, pi);
    nr = df_add(base_r, pr);
    ni = df_add(base_i, pi);
}
vec2 s_pack(S_REAL v) { return v; }
S_REAL s_unpack(vec2 u) { return u; }
vec4 s_pack_pair(S_REAL ar, S_REAL ai) { return vec4(ar.x, ar.y, ai.x, ai.y); }
void s_unpack_pair(vec4 p, out S_REAL ar, out S_REAL ai) { ar = p.xy; ai = p.zw; }
S_REAL s_splat(vec2 v) { return v; }
void load_trig_slot(uint row_off, uint bin_idx, out S_REAL re_o, out S_REAL im_o) {
    vec4 v = d_trig[row_off + bin_idx];
    re_o = v.xy; im_o = v.zw;
}
S_REAL load_slot_phi(int slot_i) { return d_slot_phi_hat[slot_i]; }
S_REAL load_slot_dec(int slot_i) { return d_slot_deconv[slot_i]; }
""",
        }
    return {
        "hdr": glsl_header("d"),
        "buf_t": "dvec2",
        "sc_t": "double",
        "real_t": "double",
        "out_t": "double",
        "push_sc_t": "double",
        "scr_t": "double",
        "scr_pair_t": "dvec2",
        "check_nc": False,
        "ops": """
#define S_REAL double
#define S_ZERO 0.0lf
#define S_ONE  1.0lf
#define S_HALF 0.5lf
#define S_TWO  2.0lf
S_REAL s_add(S_REAL a, S_REAL b) { return a + b; }
S_REAL s_sub(S_REAL a, S_REAL b) { return a - b; }
S_REAL s_mul(S_REAL a, S_REAL b) { return a * b; }
S_REAL s_neg(S_REAL a) { return -a; }
S_REAL s_div(S_REAL a, S_REAL b) { return a / b; }
S_REAL s_sqrt(S_REAL a) { return sqrt(a); }
S_REAL s_fma(S_REAL a, S_REAL b, S_REAL c) { return fma(a, b, c); }
S_REAL s_abs2(S_REAL r, S_REAL i) { return fma(r, r, i * i); }
void s_cmac(S_REAL ar, S_REAL ai, S_REAL br, S_REAL bi, inout S_REAL acc_r, inout S_REAL acc_i) {
    acc_r = fma(ar, br, fma(-ai, bi, acc_r));
    acc_i = fma(ar, bi, fma( ai, br, acc_i));
}
void s_cmac_conj(S_REAL gr, S_REAL gi, S_REAL ar, S_REAL ai,
                 S_REAL base_r, S_REAL base_i, out S_REAL nr, out S_REAL ni) {
    nr = fma(gr, ar, fma( gi, ai, base_r));
    ni = fma(gi, ar, fma(-gr, ai, base_i));
}
double s_pack(S_REAL v) { return v; }
S_REAL s_unpack(double u) { return u; }
dvec2 s_pack_pair(S_REAL ar, S_REAL ai) { return dvec2(ar, ai); }
void s_unpack_pair(dvec2 p, out S_REAL ar, out S_REAL ai) { ar = p.x; ai = p.y; }
S_REAL s_splat(double v) { return v; }
void load_trig_slot(uint row_off, uint bin_idx, out S_REAL re_o, out S_REAL im_o) {
    dvec2 v = d_trig[row_off + bin_idx];
    re_o = v.x; im_o = v.y;
}
S_REAL load_slot_phi(int slot_i) { return d_slot_phi_hat[slot_i]; }
S_REAL load_slot_dec(int slot_i) { return d_slot_deconv[slot_i]; }
""",
    }


def _fastchi2_push_block(push_sc_t: str) -> str:
    return f"""
layout(push_constant) uniform Push {{
    {push_sc_t} ws_fft;
    {push_sc_t} inv_ws_fft;
    {push_sc_t} yws_fft;
    {push_sc_t} inv_chi2_ref;
    {push_sc_t} b_param;
    {push_sc_t} cyb_sum_N_eff;
    {push_sc_t} cyb_inv_2d;
    {push_sc_t} cyb_k_eff;
    {push_sc_t} cyb_theta_eff;
    {push_sc_t} cyb_lgamma_k;
    int  degree;
    int  M;
    int  Nf;
    int  nfft;
    int  log_nfft;
    int  nout;
    int  output_shift;
    int  b_start;
    int  b_count;
    uint flags;
}} pc;
"""


# ============================================================================
# 3. FastChi2 GLS (d == 1) Shaders
# ============================================================================

def _gen_fastchi2_gls(prec: str) -> dict:
    p = _solve_prec_macros(prec)

    if prec == "h":
        items_expr = "pc.nout >> 1"
        slot_expr = "local_i << 1"
        trig_helper = """
void compute_gls_trig_f(float num_l, float den_l, float rad_l,
                        out float S2w_l, out float C2w_l, out float Ctau_l, out float Stau_l) {
    bool nz = (rad_l != 0.0f);
    bool den_neg = (den_l < 0.0f);
    float den_abs = den_neg ? -den_l : den_l;
    S2w_l = nz ? ((den_neg ? -num_l : num_l) / rad_l) : 0.0f;
    C2w_l = nz ? (den_abs / rad_l) : 1.0f;
    float op = max(1.0f + C2w_l, 0.0f);
    float om = max(1.0f - C2w_l, 0.0f);
    Ctau_l = 0.7071067811865475f * sqrt(op);
    float st_mag = 0.7071067811865475f * sqrt(om);
    Stau_l = (S2w_l > 0.0f) ? st_mag : ((S2w_l < 0.0f) ? -st_mag : 0.0f);
}
float gls_cond_lane_f(float c2ps2, float nr, float ni, float den_k) {
    float lim = 0.999f;
    float k1_a2 = (!(c2ps2 >= 0.0f) || c2ps2 > lim) ? lim : c2ps2;
    float k2_a2 = lim;
    if (den_k > 0.0f) {
        float raw = (nr * nr + ni * ni) / (den_k * den_k);
        k2_a2 = (!(raw >= 0.0f) || raw > lim) ? lim : raw;
    }
    float r1 = sqrt(k1_a2);
    float r2 = sqrt(k2_a2);
    return 3.0f * ((1.0f + r1) / (1.0f - r1)) * ((1.0f + r2) / (1.0f - r2));
}
"""
        trig_call = """
    float s2w0, c2w0, ct0, st0, s2w1, c2w1, ct1, st1;
    compute_gls_trig_f(float(num.x), float(den.x), float(radius.x), s2w0, c2w0, ct0, st0);
    compute_gls_trig_f(float(num.y), float(den.y), float(radius.y), s2w1, c2w1, ct1, st1);
    f16vec2 S2w  = f16vec2(float16_t(s2w0), float16_t(s2w1));
    f16vec2 C2w  = f16vec2(float16_t(c2w0), float16_t(c2w1));
    f16vec2 Ctau = f16vec2(float16_t(ct0),  float16_t(ct1));
    f16vec2 Stau = f16vec2(float16_t(st0),  float16_t(st1));"""
        cond_call = """
        cond_bound = f16vec2(
            float16_t(gls_cond_lane_f(float(c2_plus_s2.x), float(k2_nr.x), float(k2_ni.x), float(k2_den.x))),
            float16_t(gls_cond_lane_f(float(c2_plus_s2.y), float(k2_nr.y), float(k2_ni.y), float(k2_den.y)))
        );"""
        emit_out = """
    int blk_base = (pc.b_start + b) * pc.nout;
    for (int lane = 0; lane < 2; ++lane) {
        int s_i = slot_i + lane;
        int slot_p = nufft1_compact_idx_to_slot(s_i, mode43);
        int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
        int gf = blk_base + nat_k;
        if (gf < pc.Nf) {
            float dot_l = (lane == 0) ? float(dot.x) : float(dot.y);
            float cb_l  = (lane == 0) ? float(cond_bound.x) : float(cond_bound.y);
            if (has_cond) d_cond[gf] = cb_l;
            d_power[gf] = normalize_power(dot_l, cb_l, norm, 1);
        }
    }"""
    elif prec == "df":
        items_expr = "pc.nout"
        slot_expr = "local_i"
        trig_helper = f"""
void compute_gls_trig(df_t num_l, df_t den_l, df_t rad_l,
                      out df_t S2w_l, out df_t C2w_l, out df_t Ctau_l, out df_t Stau_l) {{
    bool nz = (rad_l.x != 0.0f || rad_l.y != 0.0f);
    bool den_neg = (den_l.x < 0.0f || (den_l.x == 0.0f && den_l.y < 0.0f));
    df_t den_abs = den_neg ? df_neg(den_l) : den_l;
    df_t inv_rad = nz ? df_div(vec2(1.0f, 0.0f), rad_l) : vec2(0.0f, 0.0f);
    S2w_l = nz ? (den_neg ? df_neg(df_mul(num_l, inv_rad)) : df_mul(num_l, inv_rad)) : vec2(0.0f, 0.0f);
    C2w_l = nz ? df_mul(den_abs, inv_rad) : vec2(1.0f, 0.0f);
    df_t op = df_add(vec2(1.0f, 0.0f), C2w_l);
    df_t om = df_sub(vec2(1.0f, 0.0f), C2w_l);
    if (op.x < 0.0f) op = vec2(0.0f, 0.0f);
    if (om.x < 0.0f) om = vec2(0.0f, 0.0f);
    Ctau_l = df_mul(kSqrt1_2df, df_sqrt(op));
    df_t st_mag = df_mul(kSqrt1_2df, df_sqrt(om));
    bool s2_pos = (S2w_l.x > 0.0f || (S2w_l.x == 0.0f && S2w_l.y > 0.0f));
    bool s2_neg = (S2w_l.x < 0.0f || (S2w_l.x == 0.0f && S2w_l.y < 0.0f));
    Stau_l = s2_pos ? st_mag : (s2_neg ? df_neg(st_mag) : vec2(0.0f, 0.0f));
}}
df_t gls_cond_lane(df_t c2ps2, df_t nr, df_t ni, df_t den_k) {{
    df_t lim = {fmt_df(0.999999999999)};
    df_t one = vec2(1.0f, 0.0f);
    bool c1_ok = (c2ps2.x > 0.0f || (c2ps2.x == 0.0f && c2ps2.y >= 0.0f));
    bool c1_gt = (c2ps2.x > lim.x || (c2ps2.x == lim.x && c2ps2.y > lim.y));
    df_t k1_a2 = (!c1_ok || c1_gt) ? lim : c2ps2;
    df_t k2_a2 = lim;
    if (den_k.x > 0.0f || (den_k.x == 0.0f && den_k.y > 0.0f)) {{
        df_t raw = df_div(df_add(df_sqr(nr), df_sqr(ni)), df_sqr(den_k));
        bool r_ok = (raw.x > 0.0f || (raw.x == 0.0f && raw.y >= 0.0f));
        bool r_gt = (raw.x > lim.x || (raw.x == lim.x && raw.y > lim.y));
        k2_a2 = (!r_ok || r_gt) ? lim : raw;
    }}
    df_t r1 = df_sqrt(k1_a2);
    df_t r2 = df_sqrt(k2_a2);
    df_t f1 = df_div(df_add(one, r1), df_sub(one, r1));
    df_t f2 = df_div(df_add(one, r2), df_sub(one, r2));
    return df_mul_f(df_mul(f1, f2), 3.0f);
}}
"""
        trig_call = "    df_t S2w, C2w, Ctau, Stau;\n    compute_gls_trig(num, den, radius, S2w, C2w, Ctau, Stau);"
        cond_call = "        cond_bound = gls_cond_lane(c2_plus_s2, k2_nr, k2_ni, k2_den);"
        emit_out = """
    int slot_p = nufft1_compact_idx_to_slot(slot_i, mode43);
    int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    int gf = (pc.b_start + b) * pc.nout + nat_k;
    if (gf < pc.Nf) {
        if (has_cond) d_cond[gf] = cond_bound;
        d_power[gf] = normalize_power(dot, cond_bound, norm, 1);
    }"""
    else:  # f or d
        t = "float" if prec == "f" else "double"
        s0 = "0.0f" if prec == "f" else "0.0lf"
        s1 = "1.0f" if prec == "f" else "1.0lf"
        s3 = "3.0f" if prec == "f" else "3.0lf"
        sq = "0.7071067811865475f" if prec == "f" else "0.7071067811865475244lf"
        lim = "0.9999999f" if prec == "f" else "0.999999999999999lf"
        items_expr = "pc.nout"
        slot_expr = "local_i"
        trig_helper = f"""
void compute_gls_trig({t} num_l, {t} den_l, {t} rad_l,
                      out {t} S2w_l, out {t} C2w_l, out {t} Ctau_l, out {t} Stau_l) {{
    bool nz = (rad_l != {s0});
    bool den_neg = (den_l < {s0});
    {t} den_abs = den_neg ? -den_l : den_l;
    S2w_l = nz ? ((den_neg ? -num_l : num_l) / rad_l) : {s0};
    C2w_l = nz ? (den_abs / rad_l) : {s1};
    {t} op = max({s1} + C2w_l, {s0});
    {t} om = max({s1} - C2w_l, {s0});
    Ctau_l = {sq} * sqrt(op);
    {t} st_mag = {sq} * sqrt(om);
    Stau_l = (S2w_l > {s0}) ? st_mag : ((S2w_l < {s0}) ? -st_mag : {s0});
}}
{t} gls_cond_lane({t} c2ps2, {t} nr, {t} ni, {t} den_k) {{
    {t} lim = {lim};
    {t} k1_a2 = (!(c2ps2 >= {s0}) || c2ps2 > lim) ? lim : c2ps2;
    {t} k2_a2 = lim;
    if (den_k > {s0}) {{
        {t} raw = (nr * nr + ni * ni) / (den_k * den_k);
        k2_a2 = (!(raw >= {s0}) || raw > lim) ? lim : raw;
    }}
    {t} r1 = sqrt(k1_a2);
    {t} r2 = sqrt(k2_a2);
    return {s3} * (({s1} + r1) / ({s1} - r1)) * (({s1} + r2) / ({s1} - r2));
}}
"""
        trig_call = f"    {t} S2w, C2w, Ctau, Stau;\n    compute_gls_trig(num, den, radius, S2w, C2w, Ctau, Stau);"
        cond_call = "        cond_bound = gls_cond_lane(c2_plus_s2, k2_nr, k2_ni, k2_den);"
        emit_out = """
    int slot_p = nufft1_compact_idx_to_slot(slot_i, mode43);
    int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    int gf = (pc.b_start + b) * pc.nout + nat_k;
    if (gf < pc.Nf) {
        if (has_cond) d_cond[gf] = cond_bound;
        d_power[gf] = normalize_power(dot, cond_bound, norm, 1);
    }"""

    src = (
        p["hdr"]
        + _glsl_compact_slot_helpers()
        + f"""
layout(constant_id = 0) const uint WG_SIZE       = 64u;
layout(constant_id = 1) const uint SPEC_USE_1D   = 0u;
layout(constant_id = 2) const uint SPEC_NORM     = 0u;
layout(constant_id = 3) const uint SPEC_HAS_COND = 0u;
layout(constant_id = 4) const uint SPEC_USE_RAW  = 0u;
layout(constant_id = 5) const uint SPEC_MODE43   = 1u;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer TrigBuf   {{ {p['buf_t']} d_trig[]; }};
layout(std430, binding = 1) readonly  buffer PhiBuf    {{ {p['sc_t']}  d_slot_phi_hat[]; }};
layout(std430, binding = 2) readonly  buffer DecBuf    {{ {p['sc_t']}  d_slot_deconv[]; }};
layout(std430, binding = 3) writeonly buffer PowBuf    {{ {p['out_t']} d_power[]; }};
layout(std430, binding = 4) writeonly buffer CondBuf   {{ {p['out_t']} d_cond[]; }};
{_fastchi2_push_block(p['push_sc_t'])}
{p['ops']}
{_glsl_fastchi2_norm_helpers(prec)}
{trig_helper}
void main() {{
    int items_per_blk = {items_expr};
    int b, local_i;
    if (SPEC_USE_1D != 0u) {{
        uint gid = gl_GlobalInvocationID.x;
        uint total_items = uint(pc.b_count) * uint(items_per_blk);
        if (gid >= total_items) return;
        b = int(gid / uint(items_per_blk));
        local_i = int(gid - uint(b) * uint(items_per_blk));
    }} else {{
        local_i = int(gl_GlobalInvocationID.x);
        b = int(gl_WorkGroupID.y);
        if (local_i >= items_per_blk || b >= pc.b_count) return;
    }}

    int mode43     = int(SPEC_MODE43);
    int norm       = int(SPEC_NORM);
    bool has_cond  = (SPEC_HAS_COND != 0u);
    bool use_raw   = (SPEC_USE_RAW != 0u);
    bool need_cond = (norm == 3) || has_cond;

    int slot_i = {slot_expr};
    uint bin_idx = uint(b) * uint(pc.nout) + uint(slot_i);
    uint chunk_bins = uint(pc.b_count) * uint(pc.nout);

    S_REAL C, S, C2, S2, Ch, Sh;
    load_trig_slot(0u,                bin_idx, C,  S);
    load_trig_slot(chunk_bins,        bin_idx, Ch, Sh);
    load_trig_slot(2u * chunk_bins,   bin_idx, C2, S2);

    S_REAL ws_base = s_splat(pc.ws_fft);
    S_REAL yws_base = s_splat(pc.yws_fft);
    S_REAL inv_ws_base = s_splat(pc.inv_ws_fft);

    S_REAL ws_k = ws_base;
    S_REAL yws_k = yws_base;
    S_REAL inv_ws = inv_ws_base;
    S_REAL dscale = S_ONE;
    if (use_raw) {{
        S_REAL pk = load_slot_phi(slot_i);
        dscale = load_slot_dec(slot_i);
        ws_k   = s_mul(ws_base, pk);
        yws_k  = s_mul(yws_base, pk);
        inv_ws = s_mul(inv_ws_base, dscale);
    }}

    S_REAL num = s_sub(S2, s_mul(s_mul(S_TWO, S), s_mul(C, inv_ws)));
    S_REAL den = s_sub(C2, s_mul(s_sub(s_mul(C, C), s_mul(S, S)), inv_ws));
    S_REAL radius = s_sqrt(s_add(s_mul(num, num), s_mul(den, den)));
{trig_call}

    S_REAL YC = s_add(s_mul(Ch, Ctau), s_mul(Sh, Stau));
    S_REAL YS = s_sub(s_mul(Sh, Ctau), s_mul(Ch, Stau));
    S_REAL XC = s_add(s_mul(C, Ctau),  s_mul(S, Stau));
    S_REAL XS = s_sub(s_mul(S, Ctau),  s_mul(C, Stau));

    S_REAL c2_rot = s_add(s_mul(C2, C2w), s_mul(S2, S2w));
    S_REAL CC = s_sub(s_mul(S_HALF, s_add(ws_k, c2_rot)), s_mul(s_mul(XC, XC), inv_ws));
    S_REAL SS = s_sub(s_mul(S_HALF, s_sub(ws_k, c2_rot)), s_mul(s_mul(XS, XS), inv_ws));
    YC = s_sub(YC, s_mul(s_mul(yws_k, XC), inv_ws));
    YS = s_sub(YS, s_mul(s_mul(yws_k, XS), inv_ws));

    S_REAL dot = s_add(s_div(s_mul(YC, YC), CC), s_div(s_mul(YS, YS), SS));
    if (use_raw) dot = s_mul(dot, dscale);

    S_REAL cond_bound = S_ZERO;
    if (need_cond) {{
        S_REAL c  = s_mul(C,  inv_ws);
        S_REAL s  = s_mul(S,  inv_ws);
        S_REAL c2 = s_mul(C2, inv_ws);
        S_REAL s2 = s_mul(S2, inv_ws);
        S_REAL c2_plus_s2 = s_add(s_mul(c, c), s_mul(s, s));
        S_REAL k2_nr = s_sub(s_add(c2, s_mul(s, s)), s_mul(c, c));
        S_REAL k2_ni = s_sub(s2, s_mul(S_TWO, s_mul(c, s)));
        S_REAL k2_den = s_sub(S_ONE, c2_plus_s2);
{cond_call}
    }}
{emit_out}
}}
"""
    )
    return {"source": src, "check_nocontract": p["check_nc"]}


# ============================================================================
# 4. FastChi2 Levinson-Durbin (d >= 2) Shaders
# ============================================================================

def _gen_fastchi2_lev(prec: str, exact_deg: int, mode: str) -> dict:
    """mode: 'unrolled' (exact_deg 2..5 in registers), 'smem' (d>=6 bank-conflict-free SoA SMEM), 'large' (global scratch)."""
    p = _solve_prec_macros(prec)
    items_expr = "pc.nout >> 1" if prec == "h" else "pc.nout"
    slot_expr = "local_i << 1" if prec == "h" else "local_i"

    if mode == "unrolled":
        norder_expr = f"{2 * exact_deg + 1}"
        deg_expr = f"{exact_deg}"
    else:
        norder_expr = "(2 * pc.degree + 1)"
        deg_expr = "pc.degree"

    if prec == "h":
        norder_splat = "s_splat(float(norder))"
        emit_out = """
    int blk_base = (pc.b_start + b) * pc.nout;
    for (int lane = 0; lane < 2; ++lane) {
        int s_i = slot_i + lane;
        int slot_p = nufft1_compact_idx_to_slot(s_i, mode43);
        int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
        int gf = blk_base + nat_k;
        if (gf < pc.Nf) {
            float cd_l = (lane == 0) ? float(cond_den.x) : float(cond_den.y);
            float cb_l = (lane == 0) ? float(cond_bound.x) : float(cond_bound.y);
            if (!(cd_l > 0.0f) || !f_is_finite(cb_l) || cb_l < 1.0f) cb_l = f_inf();
            float dot_l = (lane == 0) ? float(dot.x) : float(dot.y);
            if (has_cond) {
                d_cond[gf] = cb_l;
                d_power[gf] = normalize_power(dot_l, cb_l, norm, degree);
            } else if (cond_is_singular_f(cb_l)) {
                d_power[gf] = f_nan();
            } else {
                d_power[gf] = normalize_power(dot_l, cb_l, norm, degree);
            }
        }
    }"""
    elif prec == "df":
        norder_splat = "vec2(float(norder), 0.0f)"
        emit_out = """
    int slot_p = nufft1_compact_idx_to_slot(slot_i, mode43);
    int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    int gf = (pc.b_start + b) * pc.nout + nat_k;
    if (gf < pc.Nf) {
        float cd_f = cond_den.x + cond_den.y;
        float cb_f_raw = cond_bound.x + cond_bound.y;
        bool bad_cb = (!(cd_f > 0.0f) || !f_is_finite(cb_f_raw) || cb_f_raw < 1.0f);
        df_t cb_clean = bad_cb ? vec2(f_inf(), 0.0f) : cond_bound;
        float cb_f = bad_cb ? f_inf() : cb_f_raw;
        if (has_cond) {
            d_cond[gf] = cb_clean;
            d_power[gf] = normalize_power(dot, cb_clean, norm, degree);
        } else if (cond_is_singular_f(cb_f)) {
            d_power[gf] = vec2(f_nan(), f_nan());
        } else {
            d_power[gf] = normalize_power(dot, cb_clean, norm, degree);
        }
    }"""
    elif prec == "f":
        norder_splat = "float(norder)"
        emit_out = """
    int slot_p = nufft1_compact_idx_to_slot(slot_i, mode43);
    int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    int gf = (pc.b_start + b) * pc.nout + nat_k;
    if (gf < pc.Nf) {
        float cb_clean = (!(cond_den > 0.0f) || !f_is_finite(cond_bound) || cond_bound < 1.0f) ? f_inf() : cond_bound;
        if (has_cond) {
            d_cond[gf] = cb_clean;
            d_power[gf] = normalize_power(dot, cb_clean, norm, degree);
        } else if (cond_is_singular_f(cb_clean)) {
            d_power[gf] = f_nan();
        } else {
            d_power[gf] = normalize_power(dot, cb_clean, norm, degree);
        }
    }"""
    else:  # d
        norder_splat = "double(norder)"
        emit_out = """
    int slot_p = nufft1_compact_idx_to_slot(slot_i, mode43);
    int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    int gf = (pc.b_start + b) * pc.nout + nat_k;
    if (gf < pc.Nf) {
        double cb_clean = (!(cond_den > 0.0lf) || !d_is_finite(cond_bound) || cond_bound < 1.0lf) ? d_inf() : cond_bound;
        if (has_cond) {
            d_cond[gf] = cb_clean;
            d_power[gf] = normalize_power(dot, cb_clean, norm, degree);
        } else if (cond_is_singular(cb_clean)) {
            d_power[gf] = d_nan();
        } else {
            d_power[gf] = normalize_power(dot, cb_clean, norm, degree);
        }
    }"""

    extra_decls = ""
    extra_helpers = ""
    scratch_binding = ""
    pre_y0_block = ""
    y0_load = "    get_Y(0, degree, chunk_bins, bin_idx, yws_k, y0r, y0i);"

    if mode == "unrolled":
        norder = 2 * exact_deg + 1
        # Preload all 2*d R entries and d Y entries once into registers for d<=3 (all precisions)
        # and d<=6 (narrow h/f), eliminating O(d^2) redundant strided global loads from d_trig.
        preload_ry = (exact_deg <= 3) or (prec in ("h", "f") and exact_deg <= 6)
        if preload_ry:
            preload_lines = [
                f"    S_REAL a_r[{norder - 1}];",
                f"    S_REAL a_i[{norder - 1}];",
                f"    S_REAL R_r[{2 * exact_deg}]; S_REAL R_i[{2 * exact_deg}];",
                f"    S_REAL Y_r[{exact_deg}]; S_REAL Y_i[{exact_deg}];",
            ]
            for q in range(1, 2 * exact_deg + 1):
                wrow = (2 * (q - 1)) if q <= exact_deg else (exact_deg + q - 1)
                preload_lines.append(
                    f"    load_trig_slot({wrow}u * chunk_bins, bin_idx, R_r[{q - 1}], R_i[{q - 1}]); "
                    f"R_i[{q - 1}] = s_neg(R_i[{q - 1}]);"
                )
            for q in range(1, exact_deg + 1):
                yrow = 2 * (q - 1) + 1
                preload_lines.append(
                    f"    load_trig_slot({yrow}u * chunk_bins, bin_idx, Y_r[{q - 1}], Y_i[{q - 1}]);"
                )
            pre_y0_block = "\n".join(preload_lines)
            y0_load = f"    y0r = Y_r[{exact_deg - 1}]; y0i = Y_i[{exact_deg - 1}];"
            pred_decl = ""
        else:
            pred_decl = f"    S_REAL a_r[{norder - 1}];\n    S_REAL a_i[{norder - 1}];"

        def emit_get_r(k_idx: int, vr: str, vi: str) -> str:
            if preload_ry:
                if k_idx == 0:
                    return f"{vr} = ws_k; {vi} = S_ZERO;"
                return f"{vr} = R_r[{k_idx - 1}]; {vi} = R_i[{k_idx - 1}];"
            return f"get_R({k_idx}, degree, chunk_bins, bin_idx, ws_k, {vr}, {vi});"

        def emit_get_y(k_idx: int, vr: str, vi: str) -> str:
            if preload_ry:
                h_idx = k_idx - exact_deg
                if h_idx == 0:
                    return f"{vr} = yws_k; {vi} = S_ZERO;"
                if h_idx > 0:
                    return f"{vr} = Y_r[{h_idx - 1}]; {vi} = s_neg(Y_i[{h_idx - 1}]);"
                return f"{vr} = Y_r[{-h_idx - 1}]; {vi} = Y_i[{-h_idx - 1}];"
            return f"get_Y({k_idx}, degree, chunk_bins, bin_idx, yws_k, {vr}, {vi});"

        step_lines = []
        for k in range(1, norder):
            save_a = (k + 1) < norder
            step_lines.append(f"    {{ // k = {k}")
            step_lines.append("        S_REAL lam_r, lam_i;")
            step_lines.append(f"        {emit_get_r(k, 'lam_r', 'lam_i')}")
            for i in range(1, k):
                step_lines.append(f"        {{ S_REAL rr, ri; {emit_get_r(k - i, 'rr', 'ri')}")
                step_lines.append(f"          s_cmac(rr, ri, a_r[{i}], a_i[{i}], lam_r, lam_i); }}")
            step_lines.append("        S_REAL gamma_r = s_neg(s_mul(lam_r, inv_E));")
            step_lines.append("        S_REAL gamma_i = s_neg(s_mul(lam_i, inv_E));")
            step_lines.append("        S_REAL abs2_g = s_abs2(gamma_r, gamma_i);")
            step_lines.append("        S_REAL abs_g  = s_sqrt(abs2_g);")
            step_lines.append("        E = s_mul(E, s_sub(S_ONE, abs2_g));")
            step_lines.append("        inv_E = s_div(S_ONE, E);")
            if save_a:
                step_lines.append(f"        a_r[{k}] = gamma_r; a_i[{k}] = gamma_i;")
            step_lines.append(f"        S_REAL xr, xi; {emit_get_y(k, 'xr', 'xi')}")
            step_lines.append("        s_cmac(gamma_r, gamma_i, y0r, y0i, xr, xi);")
            for lo in range(1, (k - 1) // 2 + 1):
                hi = k - lo
                step_lines.append(f"        {{ S_REAL alr = a_r[{lo}], ali = a_i[{lo}], ahr = a_r[{hi}], ahi = a_i[{hi}];")
                step_lines.append("          S_REAL nlr, nli, nhr, nhi;")
                step_lines.append("          s_cmac_conj(gamma_r, gamma_i, ahr, ahi, alr, ali, nlr, nli);")
                step_lines.append("          s_cmac_conj(gamma_r, gamma_i, alr, ali, ahr, ahi, nhr, nhi);")
                if save_a:
                    step_lines.append(f"          a_r[{lo}] = nlr; a_i[{lo}] = nli; a_r[{hi}] = nhr; a_i[{hi}] = nhi;")
                step_lines.append(f"          S_REAL yhr, yhi, ylr, yli; {emit_get_y(hi, 'yhr', 'yhi')} {emit_get_y(lo, 'ylr', 'yli')}")
                step_lines.append("          s_cmac(nlr, nli, yhr, yhi, xr, xi);")
                step_lines.append("          s_cmac(nhr, nhi, ylr, yli, xr, xi); }")
            if (k & 1) == 0:
                mid = k // 2
                step_lines.append(f"        {{ S_REAL alr = a_r[{mid}], ali = a_i[{mid}], nlr, nli;")
                step_lines.append("          s_cmac_conj(gamma_r, gamma_i, alr, ali, alr, ali, nlr, nli);")
                if save_a:
                    step_lines.append(f"          a_r[{mid}] = nlr; a_i[{mid}] = nli;")
                step_lines.append(f"          S_REAL ylr, yli; {emit_get_y(mid, 'ylr', 'yli')}")
                step_lines.append("          s_cmac(nlr, nli, ylr, yli, xr, xi); }")
            step_lines.append("        cond_num = s_mul(cond_num, s_add(S_ONE, abs_g));")
            step_lines.append("        cond_den = s_mul(cond_den, s_sub(S_ONE, abs_g));")
            step_lines.append("        S_REAL abs2_xi = s_abs2(xr, xi);")
            step_lines.append("        r2 = s_fma(abs2_xi, inv_E, r2);")
            step_lines.append("    }")
        lev_body = "\n".join(step_lines)
    elif mode == "smem":  # bank-conflict-free SoA shared memory (a_r[], a_i[] in LDS/SMEM; zero global scratch)
        extra_decls = (
            f"layout(constant_id = 6) const uint SMEM_ELEMS = 1024u;\n"
            f"shared {p['scr_t']} l_ar[SMEM_ELEMS];\n"
            f"shared {p['scr_t']} l_ai[SMEM_ELEMS];\n"
        )
        pred_decl = "    uint tid = gl_LocalInvocationID.x;"
        lev_body = """
    for (int k = 1; k < norder; ++k) {
        S_REAL lam_r, lam_i;
        get_R(k, degree, chunk_bins, bin_idx, ws_k, lam_r, lam_i);
        for (int i = 1; i < k; ++i) {
            S_REAL rr, ri;
            get_R(k - i, degree, chunk_bins, bin_idx, ws_k, rr, ri);
            uint si = uint(i - 1) * WG_SIZE + tid;
            S_REAL ar = s_unpack(l_ar[si]);
            S_REAL ai = s_unpack(l_ai[si]);
            s_cmac(rr, ri, ar, ai, lam_r, lam_i);
        }
        S_REAL gamma_r = s_neg(s_mul(lam_r, inv_E));
        S_REAL gamma_i = s_neg(s_mul(lam_i, inv_E));
        S_REAL abs2_g = s_abs2(gamma_r, gamma_i);
        S_REAL abs_g  = s_sqrt(abs2_g);
        E = s_mul(E, s_sub(S_ONE, abs2_g));
        inv_E = s_div(S_ONE, E);
        bool save_a = (k + 1 < norder);
        if (save_a) {
            uint sk = uint(k - 1) * WG_SIZE + tid;
            l_ar[sk] = s_pack(gamma_r);
            l_ai[sk] = s_pack(gamma_i);
        }

        S_REAL xr, xi;
        get_Y(k, degree, chunk_bins, bin_idx, yws_k, xr, xi);
        s_cmac(gamma_r, gamma_i, y0r, y0i, xr, xi);

        int lo = 1, hi = k - 1;
        for (; lo < hi; ++lo, --hi) {
            uint slo = uint(lo - 1) * WG_SIZE + tid;
            uint shi = uint(hi - 1) * WG_SIZE + tid;
            S_REAL alr = s_unpack(l_ar[slo]), ali = s_unpack(l_ai[slo]);
            S_REAL ahr = s_unpack(l_ar[shi]), ahi = s_unpack(l_ai[shi]);
            S_REAL nlr, nli, nhr, nhi;
            s_cmac_conj(gamma_r, gamma_i, ahr, ahi, alr, ali, nlr, nli);
            s_cmac_conj(gamma_r, gamma_i, alr, ali, ahr, ahi, nhr, nhi);
            if (save_a) {
                l_ar[slo] = s_pack(nlr); l_ai[slo] = s_pack(nli);
                l_ar[shi] = s_pack(nhr); l_ai[shi] = s_pack(nhi);
            }
            S_REAL yhr, yhi, ylr, yli;
            get_Y(hi, degree, chunk_bins, bin_idx, yws_k, yhr, yhi);
            get_Y(lo, degree, chunk_bins, bin_idx, yws_k, ylr, yli);
            s_cmac(nlr, nli, yhr, yhi, xr, xi);
            s_cmac(nhr, nhi, ylr, yli, xr, xi);
        }
        if (lo == hi) {
            uint slo = uint(lo - 1) * WG_SIZE + tid;
            S_REAL alr = s_unpack(l_ar[slo]), ali = s_unpack(l_ai[slo]);
            S_REAL nlr, nli;
            s_cmac_conj(gamma_r, gamma_i, alr, ali, alr, ali, nlr, nli);
            if (save_a) {
                l_ar[slo] = s_pack(nlr); l_ai[slo] = s_pack(nli);
            }
            S_REAL ylr, yli;
            get_Y(lo, degree, chunk_bins, bin_idx, yws_k, ylr, yli);
            s_cmac(nlr, nli, ylr, yli, xr, xi);
        }
        cond_num = s_mul(cond_num, s_add(S_ONE, abs_g));
        cond_den = s_mul(cond_den, s_sub(S_ONE, abs_g));
        S_REAL abs2_xi = s_abs2(xr, xi);
        r2 = s_fma(abs2_xi, inv_E, r2);
    }"""
    else:  # large (warp-tiled contiguous paired scratch parameterized by TILE_LANES)
        extra_decls = (
            "layout(constant_id = 6) const uint TILE_LANES      = 32u;\n"
            "layout(constant_id = 7) const uint LOG2_TILE_LANES = 5u;\n"
        )
        scratch_binding = f"layout(std430, binding = 5) buffer ScratchBuf {{ {p['scr_pair_t']} d_a_scratch[]; }};"
        pred_decl = (
            "    uint item_stride = (uint(items_per_blk) + TILE_LANES - 1u) & ~(TILE_LANES - 1u);\n"
            "    uint scr_gid = uint(b) * item_stride + uint(local_i);\n"
            "    uint scr_base = (((scr_gid >> LOG2_TILE_LANES) * uint(norder - 2)) << LOG2_TILE_LANES) | (scr_gid & (TILE_LANES - 1u));"
        )
        lev_body = """
    for (int k = 1; k < norder; ++k) {
        S_REAL lam_r, lam_i;
        get_R(k, degree, chunk_bins, bin_idx, ws_k, lam_r, lam_i);
        for (int i = 1; i < k; ++i) {
            S_REAL rr, ri, ar, ai;
            get_R(k - i, degree, chunk_bins, bin_idx, ws_k, rr, ri);
            s_unpack_pair(d_a_scratch[scr_base + (uint(i - 1) << LOG2_TILE_LANES)], ar, ai);
            s_cmac(rr, ri, ar, ai, lam_r, lam_i);
        }
        S_REAL gamma_r = s_neg(s_mul(lam_r, inv_E));
        S_REAL gamma_i = s_neg(s_mul(lam_i, inv_E));
        S_REAL abs2_g = s_abs2(gamma_r, gamma_i);
        S_REAL abs_g  = s_sqrt(abs2_g);
        E = s_mul(E, s_sub(S_ONE, abs2_g));
        inv_E = s_div(S_ONE, E);
        bool save_a = (k + 1 < norder);
        if (save_a) {
            d_a_scratch[scr_base + (uint(k - 1) << LOG2_TILE_LANES)] = s_pack_pair(gamma_r, gamma_i);
        }

        S_REAL xr, xi;
        get_Y(k, degree, chunk_bins, bin_idx, yws_k, xr, xi);
        s_cmac(gamma_r, gamma_i, y0r, y0i, xr, xi);

        int lo = 1, hi = k - 1;
        for (; lo < hi; ++lo, --hi) {
            uint blo = scr_base + (uint(lo - 1) << LOG2_TILE_LANES);
            uint bhi = scr_base + (uint(hi - 1) << LOG2_TILE_LANES);
            S_REAL alr, ali, ahr, ahi;
            s_unpack_pair(d_a_scratch[blo], alr, ali);
            s_unpack_pair(d_a_scratch[bhi], ahr, ahi);
            S_REAL nlr, nli, nhr, nhi;
            s_cmac_conj(gamma_r, gamma_i, ahr, ahi, alr, ali, nlr, nli);
            s_cmac_conj(gamma_r, gamma_i, alr, ali, ahr, ahi, nhr, nhi);
            if (save_a) {
                d_a_scratch[blo] = s_pack_pair(nlr, nli);
                d_a_scratch[bhi] = s_pack_pair(nhr, nhi);
            }
            S_REAL yhr, yhi, ylr, yli;
            get_Y(hi, degree, chunk_bins, bin_idx, yws_k, yhr, yhi);
            get_Y(lo, degree, chunk_bins, bin_idx, yws_k, ylr, yli);
            s_cmac(nlr, nli, yhr, yhi, xr, xi);
            s_cmac(nhr, nhi, ylr, yli, xr, xi);
        }
        if (lo == hi) {
            uint blo = scr_base + (uint(lo - 1) << LOG2_TILE_LANES);
            S_REAL alr, ali, nlr, nli;
            s_unpack_pair(d_a_scratch[blo], alr, ali);
            s_cmac_conj(gamma_r, gamma_i, alr, ali, alr, ali, nlr, nli);
            if (save_a) {
                d_a_scratch[blo] = s_pack_pair(nlr, nli);
            }
            S_REAL ylr, yli;
            get_Y(lo, degree, chunk_bins, bin_idx, yws_k, ylr, yli);
            s_cmac(nlr, nli, ylr, yli, xr, xi);
        }
        cond_num = s_mul(cond_num, s_add(S_ONE, abs_g));
        cond_den = s_mul(cond_den, s_sub(S_ONE, abs_g));
        S_REAL abs2_xi = s_abs2(xr, xi);
        r2 = s_fma(abs2_xi, inv_E, r2);
    }"""

    src = (
        p["hdr"]
        + _glsl_compact_slot_helpers()
        + f"""
layout(constant_id = 0) const uint WG_SIZE       = 64u;
layout(constant_id = 1) const uint SPEC_USE_1D   = 0u;
layout(constant_id = 2) const uint SPEC_NORM     = 0u;
layout(constant_id = 3) const uint SPEC_HAS_COND = 0u;
layout(constant_id = 4) const uint SPEC_USE_RAW  = 0u;
layout(constant_id = 5) const uint SPEC_MODE43   = 1u;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
{extra_decls}
layout(std430, binding = 0) readonly  buffer TrigBuf {{ {p['buf_t']} d_trig[]; }};
layout(std430, binding = 1) readonly  buffer PhiBuf  {{ {p['sc_t']}  d_slot_phi_hat[]; }};
layout(std430, binding = 2) readonly  buffer DecBuf  {{ {p['sc_t']}  d_slot_deconv[]; }};
layout(std430, binding = 3) writeonly buffer PowBuf  {{ {p['out_t']} d_power[]; }};
layout(std430, binding = 4) writeonly buffer CondBuf {{ {p['out_t']} d_cond[]; }};
{scratch_binding}
{_fastchi2_push_block(p['push_sc_t'])}
{p['ops']}
{_glsl_fastchi2_norm_helpers(prec)}

uint trig_w_row(int q, int degree) {{
    return (q <= degree) ? uint(2 * (q - 1)) : uint(degree + q - 1);
}}
uint trig_yw_row(int q) {{ return uint(2 * (q - 1) + 1); }}

void get_R(int k, int degree, uint chunk_bins, uint bin_idx, S_REAL ws_k, out S_REAL rr, out S_REAL ri) {{
    if (k == 0) {{
        rr = ws_k; ri = S_ZERO;
    }} else {{
        load_trig_slot(trig_w_row(k, degree) * chunk_bins, bin_idx, rr, ri);
        ri = s_neg(ri);
    }}
}}

void get_Y(int k, int degree, uint chunk_bins, uint bin_idx, S_REAL yws_k, out S_REAL yr, out S_REAL yi) {{
    int h = k - degree;
    if (h == 0) {{
        yr = yws_k; yi = S_ZERO;
    }} else if (h > 0) {{
        load_trig_slot(trig_yw_row(h) * chunk_bins, bin_idx, yr, yi);
        yi = s_neg(yi);
    }} else {{
        load_trig_slot(trig_yw_row(-h) * chunk_bins, bin_idx, yr, yi);
    }}
}}
{extra_helpers}
void main() {{
    int items_per_blk = {items_expr};
    int b, local_i;
    if (SPEC_USE_1D != 0u) {{
        uint gid = gl_GlobalInvocationID.x;
        uint total_items = uint(pc.b_count) * uint(items_per_blk);
        if (gid >= total_items) return;
        b = int(gid / uint(items_per_blk));
        local_i = int(gid - uint(b) * uint(items_per_blk));
    }} else {{
        local_i = int(gl_GlobalInvocationID.x);
        b = int(gl_WorkGroupID.y);
        if (local_i >= items_per_blk || b >= pc.b_count) return;
    }}

    int degree     = {deg_expr};
    int norder     = {norder_expr};
    int mode43     = int(SPEC_MODE43);
    int norm       = int(SPEC_NORM);
    bool has_cond  = (SPEC_HAS_COND != 0u);
    bool use_raw   = (SPEC_USE_RAW != 0u);

    int slot_i = {slot_expr};
    uint bin_idx = uint(b) * uint(pc.nout) + uint(slot_i);
    uint chunk_bins = uint(pc.b_count) * uint(pc.nout);

    S_REAL ws_k = s_splat(pc.ws_fft);
    S_REAL yws_k = s_splat(pc.yws_fft);
    S_REAL inv_E = s_splat(pc.inv_ws_fft);
    S_REAL dscale = S_ONE;
    if (use_raw) {{
        S_REAL pk = load_slot_phi(slot_i);
        dscale = load_slot_dec(slot_i);
        ws_k  = s_mul(ws_k, pk);
        yws_k = s_mul(yws_k, pk);
        inv_E = s_mul(inv_E, dscale);
    }}

{pre_y0_block}
    S_REAL E = ws_k;
    S_REAL y0r, y0i;
{y0_load}
    S_REAL r2 = s_mul(s_abs2(y0r, y0i), inv_E);
    S_REAL cond_num = {norder_splat};
    S_REAL cond_den = S_ONE;
{pred_decl}
{lev_body}

    S_REAL cond_bound = s_div(cond_num, cond_den);
    S_REAL dot = r2;
    if (use_raw) dot = s_mul(dot, dscale);
{emit_out}
}}
"""
    )
    return {"source": src, "check_nocontract": p["check_nc"]}


# ============================================================================
# 5. IHS Shaders (single, accum, finalize)
# ============================================================================

def _glsl_ihs_math(prec: str) -> str:
    if prec == "d":
        return (
            _glsl_fp64_transcendentals()
            + """
double ihs_wilkie(double z, double inv_n, double inv_n2_288) {
    if (!(inv_n > 0.0lf)) return z;
    if (!(z > 0.0lf) || !d_is_finite(z)) return 0.0lf;
    double p2 = fma(-9.0lf, z, 76.0lf);
    p2 = fma(p2, z, -132.0lf);
    p2 = fma(p2, z, 24.0lf);
    double delta = fma(z * (2.0lf - z), 0.25lf * inv_n, -(z * p2) * inv_n2_288);
    if (!(delta > -1.0lf) || !d_is_finite(delta)) return z;
    double corr = z - d_log1p(delta);
    return (corr > 0.0lf && d_is_finite(corr)) ? corr : z;
}

double ihs_erlang(double s, int degree) {
    if (!(s > 0.0lf) || !d_is_finite(s)) return 0.0lf;
    double log_sum = 1.0lf, term = 1.0lf;
    for (int i = 1; i < degree; ++i) {
        term *= s / double(i);
        log_sum += term;
    }
    if (!(log_sum > 0.0lf) || !d_is_finite(log_sum)) return 0.0lf;
    double nll = s - d_log(log_sum);
    return (nll > 0.0lf && d_is_finite(nll)) ? nll : 0.0lf;
}
"""
        )
    if prec == "df":
        return (
            _glsl_df_transcendentals()
            + """
df_t ihs_wilkie(df_t z, df_t inv_n, df_t inv_n2_288) {
    if (!(inv_n.x > 0.0f)) return z;
    if (!(z.x > 0.0f) || !f_is_finite(z.x)) return vec2(0.0f, 0.0f);
    df_t p2 = df_add(df_mul_f(z, -9.0f), vec2(76.0f, 0.0f));
    p2 = df_add(df_mul(p2, z), vec2(-132.0f, 0.0f));
    p2 = df_add(df_mul(p2, z), vec2(24.0f, 0.0f));
    df_t t1 = df_mul(df_mul(z, df_sub(vec2(2.0f, 0.0f), z)), df_scale_pow2(inv_n, 0.25f));
    df_t delta = df_sub(t1, df_mul(df_mul(z, p2), inv_n2_288));
    if (!(delta.x > -1.0f) || !f_is_finite(delta.x)) return z;
    df_t corr = df_sub(z, df_log1p(delta));
    return (corr.x > 0.0f && f_is_finite(corr.x)) ? corr : z;
}

df_t ihs_erlang(df_t s, int degree) {
    if (!(s.x > 0.0f) || !f_is_finite(s.x)) return vec2(0.0f, 0.0f);
    df_t log_sum = vec2(1.0f, 0.0f), term = vec2(1.0f, 0.0f);
    for (int i = 1; i < degree; ++i) {
        term = df_mul(term, df_div(s, vec2(float(i), 0.0f)));
        log_sum = df_add(log_sum, term);
    }
    if (!(log_sum.x > 0.0f) || !f_is_finite(log_sum.x)) return vec2(0.0f, 0.0f);
    df_t nll = df_sub(s, df_log(log_sum));
    return (nll.x > 0.0f && f_is_finite(nll.x)) ? nll : vec2(0.0f, 0.0f);
}
"""
        )
    return (
        _glsl_f32_helpers("5.0e5f")
        + """
float ihs_wilkie(float z, float inv_n, float inv_n2_288) {
    if (!(inv_n > 0.0f)) return z;
    if (!(z > 0.0f) || !f_is_finite(z)) return 0.0f;
    float p2 = fma(-9.0f, z, 76.0f);
    p2 = fma(p2, z, -132.0f);
    p2 = fma(p2, z, 24.0f);
    float delta = fma(z * (2.0f - z), 0.25f * inv_n, -(z * p2) * inv_n2_288);
    if (!(delta > -1.0f) || !f_is_finite(delta)) return z;
    float corr = z - f_log1p(delta);
    return (corr > 0.0f && f_is_finite(corr)) ? corr : z;
}

float ihs_erlang(float s, int degree) {
    if (!(s > 0.0f) || !f_is_finite(s)) return 0.0f;
    float log_sum = 1.0f, term = 1.0f;
    for (int i = 1; i < degree; ++i) {
        term *= s / float(i);
        log_sum += term;
    }
    if (!(log_sum > 0.0f) || !f_is_finite(log_sum)) return 0.0f;
    float nll = s - log(log_sum);
    return (nll > 0.0f && f_is_finite(nll)) ? nll : 0.0f;
}
"""
    )


def _ihs_push_block(out_t: str) -> str:
    return f"""
layout(push_constant) uniform Push {{
    {out_t} base_scale;
    {out_t} inv_n_eff;
    {out_t} inv_n2_288;
    int degree;
    int Nf;
    int nfft;
    int log_nfft;
    int nout;
    int output_shift;
    int b_start;
    int b_count;
    int mode43;
    int norm;
    int first_harmonic;
}} pc;
"""


def _gen_ihs_shaders(prec: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if prec == "h":
        hdr = glsl_header("f")  # Note: d_z is uint (unpackHalf2x16), d_slot_deconv & d_power are float! Zero FP16 ext needed!
        z_t, out_t, check_nc = "uint", "float", False
        load_abs2 = "vec2 z = unpackHalf2x16(d_z[idx]); float val = fma(z.x, z.x, z.y * z.y);"
    elif prec == "f":
        hdr = glsl_header("f")
        z_t, out_t, check_nc = "vec2", "float", False
        load_abs2 = "vec2 z = d_z[idx]; float val = fma(z.x, z.x, z.y * z.y);"
    elif prec == "df":
        hdr = glsl_header("df") + glsl_df_math()
        z_t, out_t, check_nc = "vec4", "vec2", True
        load_abs2 = "vec4 z = d_z[idx]; df_t val = df_add(df_sqr(z.xy), df_sqr(z.zw));"
    else:
        hdr = glsl_header("d")
        z_t, out_t, check_nc = "dvec2", "double", False
        load_abs2 = "dvec2 z = d_z[idx]; double val = fma(z.x, z.x, z.y * z.y);"

    math_blk = _glsl_ihs_math(prec)
    push_blk = _ihs_push_block(out_t)
    slot_blk = _glsl_compact_slot_helpers()

    if prec == "df":
        single_body = """
    df_t d = d_slot_deconv[slot_i];
    val = df_mul(val, df_mul(pc.base_scale, df_sqr(d)));
    if (SPEC_NORM == 0u) {
        bool ge_one = (val.x > 1.0f || (val.x == 1.0f && val.y >= 0.0f));
        d_power[gf] = (!f_is_finite(val.x) || ge_one) ? vec2(f_nan(), f_nan()) : val;
    } else if (SPEC_NORM == 2u) {
        df_t v = ihs_wilkie(val, pc.inv_n_eff, pc.inv_n2_288);
        d_power[gf] = (f_is_finite(v.x) && (v.x > 0.0f || (v.x == 0.0f && v.y > 0.0f))) ? v : vec2(0.0f, 0.0f);
    } else {
        d_power[gf] = (f_is_finite(val.x) && (val.x > 0.0f || (val.x == 0.0f && val.y > 0.0f))) ? val : vec2(0.0f, 0.0f);
    }"""
        accum_body = """
    if (SPEC_NORM == 2u && SPEC_MULTI_HARM != 0u) {
        df_t d = d_slot_deconv[slot_i];
        val = ihs_wilkie(df_mul(val, df_mul(pc.base_scale, df_sqr(d))), pc.inv_n_eff, pc.inv_n2_288);
    }
    d_acc[idx] = (pc.first_harmonic != 0) ? val : df_add(d_acc[idx], val);"""
        fin_body = """
    df_t val = d_acc[idx];
    if (!(SPEC_NORM == 2u && SPEC_MULTI_HARM != 0u)) {
        df_t d = d_slot_deconv[slot_i];
        val = df_mul(val, df_mul(pc.base_scale, df_sqr(d)));
    }
    if (SPEC_NORM == 0u) {
        bool ge_one = (val.x > 1.0f || (val.x == 1.0f && val.y >= 0.0f));
        d_power[gf] = (SPEC_MULTI_HARM == 0u && (!f_is_finite(val.x) || ge_one)) ? vec2(f_nan(), f_nan()) : val;
    } else if (SPEC_NORM == 2u && SPEC_MULTI_HARM == 0u) {
        df_t v = ihs_wilkie(val, pc.inv_n_eff, pc.inv_n2_288);
        d_power[gf] = (f_is_finite(v.x) && (v.x > 0.0f || (v.x == 0.0f && v.y > 0.0f))) ? v : vec2(0.0f, 0.0f);
    } else {
        df_t v = ihs_erlang(val, pc.degree);
        d_power[gf] = (v.x > 0.0f || (v.x == 0.0f && v.y > 0.0f)) ? v : vec2(0.0f, 0.0f);
    }"""
    elif prec == "d":
        single_body = """
    double d = d_slot_deconv[slot_i];
    val *= pc.base_scale * d * d;
    if (SPEC_NORM == 0u) {
        d_power[gf] = (!d_is_finite(val) || val >= 1.0lf) ? d_nan() : val;
    } else if (SPEC_NORM == 2u) {
        double v = ihs_wilkie(val, pc.inv_n_eff, pc.inv_n2_288);
        d_power[gf] = (d_is_finite(v) && v > 0.0lf) ? v : 0.0lf;
    } else {
        d_power[gf] = (d_is_finite(val) && val > 0.0lf) ? val : 0.0lf;
    }"""
        accum_body = """
    if (SPEC_NORM == 2u && SPEC_MULTI_HARM != 0u) {
        double d = d_slot_deconv[slot_i];
        val = ihs_wilkie(val * (pc.base_scale * d * d), pc.inv_n_eff, pc.inv_n2_288);
    }
    d_acc[idx] = (pc.first_harmonic != 0) ? val : (d_acc[idx] + val);"""
        fin_body = """
    double val = d_acc[idx];
    if (!(SPEC_NORM == 2u && SPEC_MULTI_HARM != 0u)) {
        double d = d_slot_deconv[slot_i];
        val *= pc.base_scale * d * d;
    }
    if (SPEC_NORM == 0u) {
        d_power[gf] = (SPEC_MULTI_HARM == 0u && (!d_is_finite(val) || val >= 1.0lf)) ? d_nan() : val;
    } else if (SPEC_NORM == 2u && SPEC_MULTI_HARM == 0u) {
        double v = ihs_wilkie(val, pc.inv_n_eff, pc.inv_n2_288);
        d_power[gf] = (d_is_finite(v) && v > 0.0lf) ? v : 0.0lf;
    } else {
        double v = ihs_erlang(val, pc.degree);
        d_power[gf] = (v >= 0.0lf) ? v : 0.0lf;
    }"""
    else:  # h, f
        single_body = """
    float d = d_slot_deconv[slot_i];
    val *= pc.base_scale * d * d;
    if (SPEC_NORM == 0u) {
        d_power[gf] = (!f_is_finite(val) || val >= 1.0f) ? f_nan() : val;
    } else if (SPEC_NORM == 2u) {
        float v = ihs_wilkie(val, pc.inv_n_eff, pc.inv_n2_288);
        d_power[gf] = (f_is_finite(v) && v > 0.0f) ? v : 0.0f;
    } else {
        d_power[gf] = (f_is_finite(val) && val > 0.0f) ? val : 0.0f;
    }"""
        accum_body = """
    if (SPEC_NORM == 2u && SPEC_MULTI_HARM != 0u) {
        float d = d_slot_deconv[slot_i];
        val = ihs_wilkie(val * (pc.base_scale * d * d), pc.inv_n_eff, pc.inv_n2_288);
    }
    d_acc[idx] = (pc.first_harmonic != 0) ? val : (d_acc[idx] + val);"""
        fin_body = """
    float val = d_acc[idx];
    if (!(SPEC_NORM == 2u && SPEC_MULTI_HARM != 0u)) {
        float d = d_slot_deconv[slot_i];
        val *= pc.base_scale * d * d;
    }
    if (SPEC_NORM == 0u) {
        d_power[gf] = (SPEC_MULTI_HARM == 0u && (!f_is_finite(val) || val >= 1.0f)) ? f_nan() : val;
    } else if (SPEC_NORM == 2u && SPEC_MULTI_HARM == 0u) {
        float v = ihs_wilkie(val, pc.inv_n_eff, pc.inv_n2_288);
        d_power[gf] = (f_is_finite(v) && v > 0.0f) ? v : 0.0f;
    } else {
        float v = ihs_erlang(val, pc.degree);
        d_power[gf] = (v >= 0.0f) ? v : 0.0f;
    }"""

    ihs_specs = """
layout(constant_id = 0) const uint WG_SIZE         = 64u;
layout(constant_id = 1) const uint SPEC_USE_1D     = 0u;
layout(constant_id = 2) const uint SPEC_NORM       = 0u;
layout(constant_id = 3) const uint SPEC_MODE43     = 1u;
layout(constant_id = 4) const uint SPEC_MULTI_HARM = 0u;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
"""
    ihs_coord = """
    int b, slot_i;
    uint idx;
    if (SPEC_USE_1D != 0u) {
        idx = gl_GlobalInvocationID.x;
        uint total_bins = uint(pc.b_count) * uint(pc.nout);
        if (idx >= total_bins) return;
        b = int(idx / uint(pc.nout));
        slot_i = int(idx - uint(b) * uint(pc.nout));
    } else {
        slot_i = int(gl_GlobalInvocationID.x);
        b = int(gl_WorkGroupID.y);
        if (slot_i >= pc.nout || b >= pc.b_count) return;
        idx = uint(b) * uint(pc.nout) + uint(slot_i);
    }"""

    out[f"ihs_single_{prec}"] = {
        "source": hdr + slot_blk + ihs_specs + f"""
layout(std430, binding = 0) readonly  buffer ZBuf   {{ {z_t}   d_z[]; }};
layout(std430, binding = 1) readonly  buffer DecBuf {{ {out_t} d_slot_deconv[]; }};
layout(std430, binding = 2) writeonly buffer PowBuf {{ {out_t} d_power[]; }};
{push_blk}
{math_blk}
void main() {{{ihs_coord}
    int slot_p = nufft1_compact_idx_to_slot(slot_i, int(SPEC_MODE43));
    int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    int gf = (pc.b_start + b) * pc.nout + nat_k;
    if (gf >= pc.Nf) return;
    {load_abs2}
{single_body}
}}
""",
        "check_nocontract": check_nc,
    }

    out[f"ihs_accum_{prec}"] = {
        "source": hdr + ihs_specs + f"""
layout(std430, binding = 0) readonly buffer ZBuf   {{ {z_t}   d_z[]; }};
layout(std430, binding = 1) readonly buffer DecBuf {{ {out_t} d_slot_deconv[]; }};
layout(std430, binding = 2) buffer          AccBuf {{ {out_t} d_acc[]; }};
{push_blk}
{math_blk}
void main() {{{ihs_coord}
    {load_abs2}
{accum_body}
}}
""",
        "check_nocontract": check_nc,
    }

    out[f"ihs_finalize_{prec}"] = {
        "source": hdr + slot_blk + ihs_specs + f"""
layout(std430, binding = 0) readonly  buffer AccBuf {{ {out_t} d_acc[]; }};
layout(std430, binding = 1) readonly  buffer DecBuf {{ {out_t} d_slot_deconv[]; }};
layout(std430, binding = 2) writeonly buffer PowBuf {{ {out_t} d_power[]; }};
{push_blk}
{math_blk}
void main() {{{ihs_coord}
    int slot_p = nufft1_compact_idx_to_slot(slot_i, int(SPEC_MODE43));
    int nat_k = nufft1_slot_to_natural_idx(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    int gf = (pc.b_start + b) * pc.nout + nat_k;
    if (gf >= pc.Nf) return;
{fin_body}
}}
""",
        "check_nocontract": check_nc,
    }

    return out


def generate_periodogram_shaders() -> dict[str, dict]:
    shaders: dict[str, dict] = {}
    for prec in ("h", "f", "df", "d"):
        shaders[f"subgrid_{prec}"] = _gen_subgrid(prec)
        shaders[f"fastchi2_gls_{prec}"] = _gen_fastchi2_gls(prec)
        unroll_degs = (2, 3, 4, 5, 6, 7, 8) if prec in ("h", "f") else (2, 3, 4, 5)
        for deg in unroll_degs:
            shaders[f"fastchi2_lev_d{deg}_{prec}"] = _gen_fastchi2_lev(prec, exact_deg=deg, mode="unrolled")
        shaders[f"fastchi2_lev_smem_{prec}"] = _gen_fastchi2_lev(prec, exact_deg=0, mode="smem")
        shaders[f"fastchi2_lev_large_{prec}"] = _gen_fastchi2_lev(prec, exact_deg=0, mode="large")
        shaders.update(_gen_ihs_shaders(prec))
    return shaders
