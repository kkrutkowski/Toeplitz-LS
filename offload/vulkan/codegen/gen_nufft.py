"""GLSL compute shader codegen for Quadruple-Precision Type-1 NuFFT (NuFFT-1).

Generates all Type-1 NuFFT shaders across 4 precisions (h, f, df, d):
  - 4  precompute_points_{h,f,df,d} (DF x-axis gridding for h/f/df; FP64 for d)
  - 4  window bucket builders: bucket_{count,scan,scatter,sort}
  - 10 precompute_weights_{h,f,df,d}_w{W}
  - 4  pack_bucket_geom_{h,f,df,d}
  - 4  spread_smem_{h,f,df,d} (spec-constant parameterized windowed SMEM spreading)
  - 14 spread_reg_{h,f,df,d}_q{Q} (register-resident accumulator spreading)
  - 4  spread_cpu_direct_{h,f,df,d} (barrier-free, zero-SMEM CPU/llvmpipe fallback)
  - 12 deconv_table_{h,f,df,d}_w{W} + deconv_table_h_f32_w{4,5}
  - 10 deconv_emit_{h,f,df,d}_w{W} (Natural order & MultiStageGlobal fallback)
Total: 66 shaders.
"""

from __future__ import annotations

from common_glsl import (
    glsl_bit_reverse,
    glsl_df_math,
    glsl_f16x2_math,
    glsl_header,
    glsl_sincos2pi_df,
    glsl_sincos2pi_f32,
    glsl_sincos2pi_f64,
)

# (prec_tag, W, NEven, NOdd)
PSWF_WIDTH_SPECS = [
    ("h", 4, 4, 3),
    ("h", 5, 4, 3),
    ("f", 8, 5, 4),
    ("f", 9, 5, 4),
    ("df", 14, 7, 6),
    ("df", 15, 7, 6),
    ("df", 16, 7, 6),
    ("d", 16, 8, 7),
    ("d", 17, 8, 7),
    ("d", 18, 8, 7),
]

REG_Q_BY_PREC = {
    "h": (8, 16, 32),
    "f": (4, 8, 16, 32),
    "df": (4, 8, 16),
    "d": (2, 4, 8, 16),
}


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

int nufft1_slot_to_k_prime(int slot_p, int log_nfft, int nfft, int output_shift) {
    int bin = int(bit_reverse(uint(slot_p), uint(log_nfft)));
    int first_bin = nfft - output_shift;
    return (bin >= first_bin) ? (bin - nfft) : bin;
}
"""
    )


def _emit_weight_eval_body(prec: str, w: int, n_even: int, n_odd: int) -> list[str]:
    """Emit Horner weight evaluation writing d_weights[m*w .. (m+1)*w-1] from local register `u`."""
    n_left = w // 2
    odd_base = 4 + ((w + 1) // 2) * n_even
    if prec == "h":
        lines = [
            "    f16vec2 u_v = f16vec2(u, u);",
            "    f16vec2 t_v = u_v * u_v;",
            f"    uint base = uint(m) * {w}u;",
        ]
        for j in range(n_left):
            lines.append(
                f"    f16vec2 eo_{j} = f16vec2(float16_t(d_tbl[{4 + j * n_even + (n_even - 1)}]), float16_t(d_tbl[{odd_base + j * n_odd + (n_odd - 1)}]));"
            )
            max_terms = max(n_even, n_odd)
            for k in range(max_terms - 2, -1, -1):
                e_idx = 4 + j * n_even + k
                o_idx = odd_base + j * n_odd + k
                if k < n_odd:
                    lines.append(f"    eo_{j} = fma(eo_{j}, t_v, f16vec2(float16_t(d_tbl[{e_idx}]), float16_t(d_tbl[{o_idx}])));")
                else:
                    lines.append(f"    eo_{j}.x = fma(eo_{j}.x, t_v.x, float16_t(d_tbl[{e_idx}]));")
            lines.append(f"    float16_t uo_{j} = u * eo_{j}.y;")
            lines.append(f"    d_weights[base + {j}u] = eo_{j}.x + uo_{j};")
            lines.append(f"    d_weights[base + {w - 1 - j}u] = eo_{j}.x - uo_{j};")
        if w & 1:
            j_mid = w // 2
            lines.append(f"    float16_t em_acc = float16_t(d_tbl[{4 + j_mid * n_even + (n_even - 1)}]);")
            for k in range(n_even - 2, -1, -1):
                lines.append(f"    em_acc = fma(em_acc, t_v.x, float16_t(d_tbl[{4 + j_mid * n_even + k}]));")
            lines.append(f"    d_weights[base + {j_mid}u] = em_acc;")
        return lines

    if prec in ("f", "d"):
        t = "float" if prec == "f" else "double"
        lines = [
            f"    {t} t = u * u;",
            f"    uint base = uint(m) * {w}u;",
        ]
        for j in range(n_left):
            lines.append("    {")
            lines.append(f"        {t} e_acc = d_tbl[{4 + j * n_even + (n_even - 1)}];")
            for k in range(n_even - 2, -1, -1):
                lines.append(f"        e_acc = fma(e_acc, t, d_tbl[{4 + j * n_even + k}]);")
            lines.append(f"        {t} o_acc = d_tbl[{odd_base + j * n_odd + (n_odd - 1)}];")
            for k in range(n_odd - 2, -1, -1):
                lines.append(f"        o_acc = fma(o_acc, t, d_tbl[{odd_base + j * n_odd + k}]);")
            lines.append(f"        {t} uo = u * o_acc;")
            lines.append(f"        d_weights[base + {j}u] = e_acc + uo;")
            lines.append(f"        d_weights[base + {w - 1 - j}u] = e_acc - uo;")
            lines.append("    }")
        if w & 1:
            j_mid = w // 2
            lines.append("    {")
            lines.append(f"        {t} e_acc = d_tbl[{4 + j_mid * n_even + (n_even - 1)}];")
            for k in range(n_even - 2, -1, -1):
                lines.append(f"        e_acc = fma(e_acc, t, d_tbl[{4 + j_mid * n_even + k}]);")
            lines.append(f"        d_weights[base + {j_mid}u] = e_acc;")
            lines.append("    }")
        return lines

    # DF with mixed-precision float tail
    e_tail = 4 if n_even > 5 else n_even
    o_tail = 4 if n_odd > 5 else n_odd
    lines = [
        "    df_t t = df_sqr(u);",
        "    float tf = t.x;",
        f"    uint base = uint(m) * {w}u;",
    ]

    def emit_horner_df(acc_name: str, row_off: int, n_terms: int, tail: int) -> list[str]:
        out_l = []
        if tail < n_terms:
            out_l.append(f"        float {acc_name}_f = d_tbl[{row_off + n_terms - 1}].x;")
            for k in range(n_terms - 2, tail - 1, -1):
                out_l.append(f"        {acc_name}_f = fma({acc_name}_f, tf, d_tbl[{row_off + k}].x);")
            out_l.append(f"        df_t {acc_name} = vec2({acc_name}_f, 0.0f);")
            for k in range(tail - 1, -1, -1):
                out_l.append(f"        {acc_name} = df_add(df_mul({acc_name}, t), d_tbl[{row_off + k}]);")
        else:
            out_l.append(f"        df_t {acc_name} = d_tbl[{row_off + n_terms - 1}];")
            for k in range(n_terms - 2, -1, -1):
                out_l.append(f"        {acc_name} = df_add(df_mul({acc_name}, t), d_tbl[{row_off + k}]);")
        return out_l

    for j in range(n_left):
        lines.append("    {")
        lines.extend(emit_horner_df("e_acc", 4 + j * n_even, n_even, e_tail))
        lines.extend(emit_horner_df("o_acc", odd_base + j * n_odd, n_odd, o_tail))
        lines.append("        df_t uo = df_mul(u, o_acc);")
        lines.append(f"        d_weights[base + {j}u] = df_add(e_acc, uo);")
        lines.append(f"        d_weights[base + {w - 1 - j}u] = df_sub(e_acc, uo);")
        lines.append("    }")
    if w & 1:
        j_mid = w // 2
        lines.append("    {")
        lines.extend(emit_horner_df("e_acc", 4 + j_mid * n_even, n_even, e_tail))
        lines.append(f"        d_weights[base + {j_mid}u] = e_acc;")
        lines.append("    }")
    return lines


def _gen_precompute_points_weights(prec: str, w: int, n_even: int, n_odd: int) -> dict:
    """Fused x-axis gridding + phase shift + PSWF weight evaluation (keeps u in register; removes d_u)."""
    w_body = "\n".join(_emit_weight_eval_body(prec, w, n_even, n_odd))
    if prec == "d":
        src = (
            glsl_header("d")
            + glsl_sincos2pi_f64()
            + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer XBuf     {{ double d_x[]; }};
layout(std430, binding = 1) readonly  buffer TblBuf   {{ double d_tbl[]; }};
layout(std430, binding = 2) writeonly buffer FirstBuf {{ int    d_first[]; }};
layout(std430, binding = 3) writeonly buffer ShiftBuf {{ dvec2  d_shift[]; }};
layout(std430, binding = 4) writeonly buffer WBuf     {{ double d_weights[]; }};

layout(push_constant) uniform Push {{
    double df_scaled;
    int    M;
    int    nfft;
    int    w;
    int    phase_shift;
    uint   flags;
}} pc;

void main() {{
    int m = int(gl_GlobalInvocationID.x);
    if (m >= pc.M) return;

    bool mode43   = (pc.flags & 1u) != 0u;
    bool pos_sign = (pc.flags & 2u) != 0u;

    double xm = d_x[m] * pc.df_scaled;
    double nfft_d = double(pc.nfft);
    double inv_nfft_d = 1.0lf / nfft_d;
    double x_n = xm * nfft_d;
    double m_left_d = ceil(x_n - 0.5lf * double(pc.w));

    double u = 2.0lf * (m_left_d - x_n) + double(pc.w - 1);
    u = clamp(u, -1.0lf, 1.0lf);

    double rem_d = m_left_d - floor(m_left_d * inv_nfft_d) * nfft_d;
    int first = int(rem_d) & (pc.nfft - 1);

    double phase = xm * double(pc.phase_shift);
    double c, s;
    sincos2pi_branchless(phase, false, c, s);
    double sr = c, si = s;
    if (mode43) {{
        sr = c * fma(4.0lf, c * c, -3.0lf);
        si = s * fma(-4.0lf, s * s, 3.0lf);
    }}
    if (!pos_sign) si = -si;

    d_first[m] = first;
    d_shift[m] = dvec2(sr, si);
{w_body}
}}
"""
        )
        return {"source": src, "check_nocontract": False}

    if prec == "h":
        hdr = glsl_header("h", fp16_storage=True) + glsl_df_math() + glsl_sincos2pi_f32() + glsl_f16x2_math()
        tbl_t, w_t, shift_t = "float", "float16_t", "uint"
        emit_tail = f"""
    float frac_f = frac_df.x + frac_df.y;
    float c, s;
    sincos2pif_branchless(frac_f, false, c, s);
    float sr = c, si = s;
    if (mode43) {{
        sr = c * fma(4.0f, c * c, -3.0f);
        si = s * fma(-4.0f, s * s, 3.0f);
    }}
    if (!pos_sign) si = -si;
    d_shift[m] = packHalf2x16(vec2(sr, si));
    float16_t u = float16_t(u_df.x + u_df.y);
{w_body}
"""
    elif prec == "f":
        hdr = glsl_header("f") + glsl_df_math() + glsl_sincos2pi_f32()
        tbl_t, w_t, shift_t = "float", "float", "vec2"
        emit_tail = f"""
    float frac_f = frac_df.x + frac_df.y;
    float c, s;
    sincos2pif_branchless(frac_f, false, c, s);
    float sr = c, si = s;
    if (mode43) {{
        sr = c * fma(4.0f, c * c, -3.0f);
        si = s * fma(-4.0f, s * s, 3.0f);
    }}
    if (!pos_sign) si = -si;
    d_shift[m] = vec2(sr, si);
    float u = u_df.x + u_df.y;
{w_body}
"""
    else:  # df
        hdr = glsl_header("df") + glsl_df_math() + glsl_sincos2pi_df()
        tbl_t, w_t, shift_t = "vec2", "vec2", "vec4"
        emit_tail = f"""
    df_t c, s;
    sincos2pidf_branchless(frac_df, false, c, s);
    df_t sr = c, si = s;
    if (mode43) {{
        df_t c2 = df_sqr(c);
        df_t s2 = df_sqr(s);
        sr = df_mul(c, df_sub(df_scale_pow2(c2, 4.0f), vec2(3.0f, 0.0f)));
        si = df_mul(s, df_sub(vec2(3.0f, 0.0f), df_scale_pow2(s2, 4.0f)));
    }}
    if (!pos_sign) si = df_neg(si);
    d_shift[m] = vec4(sr.x, sr.y, si.x, si.y);
    df_t u = u_df;
{w_body}
"""

    src = (
        hdr
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer XBuf     {{ vec2       d_x[]; }};
layout(std430, binding = 1) readonly  buffer TblBuf   {{ {tbl_t}    d_tbl[]; }};
layout(std430, binding = 2) writeonly buffer FirstBuf {{ int        d_first[]; }};
layout(std430, binding = 3) writeonly buffer ShiftBuf {{ {shift_t}  d_shift[]; }};
layout(std430, binding = 4) writeonly buffer WBuf     {{ {w_t}      d_weights[]; }};

layout(push_constant) uniform Push {{
    vec2 df_scaled;
    int  M;
    int  nfft;
    int  w;
    int  phase_shift;
    uint flags;
}} pc;

void main() {{
    int m = int(gl_GlobalInvocationID.x);
    if (m >= pc.M) return;

    bool mode43   = (pc.flags & 1u) != 0u;
    bool pos_sign = (pc.flags & 2u) != 0u;

    df_t xm = df_mul(d_x[m], pc.df_scaled);
    float nfft_f = float(pc.nfft);
    float inv_nfft_f = 1.0f / nfft_f;
    df_t x_n = df_scale_pow2(xm, nfft_f);
    df_t half_w_df = vec2(0.5f * float(pc.w), 0.0f);
    df_t m_left_df = df_ceil(df_sub(x_n, half_w_df));

    df_t dist0 = df_sub(m_left_df, x_n);
    df_t u_df = df_add(df_scale_pow2(dist0, 2.0f), vec2(float(pc.w - 1), 0.0f));
    if (u_df.x < -1.0f || (u_df.x == -1.0f && u_df.y < 0.0f)) {{
        u_df = vec2(-1.0f, 0.0f);
    }}
    if (u_df.x > 1.0f || (u_df.x == 1.0f && u_df.y > 0.0f)) {{
        u_df = vec2(1.0f, 0.0f);
    }}

    float r_hi = m_left_df.x - floor(m_left_df.x * inv_nfft_f) * nfft_f;
    float r_lo = m_left_df.y - floor(m_left_df.y * inv_nfft_f) * nfft_f;
    int first = (int(r_hi) + int(r_lo)) & (pc.nfft - 1);

    df_t phase_df = df_mul_f(xm, float(pc.phase_shift));
    df_t frac_df = df_sub(phase_df, df_floor(phase_df));

    d_first[m] = first;
{emit_tail}
}}
"""
    )
    return {"source": src, "check_nocontract": True}


def _gen_bucket_shaders() -> dict[str, dict]:
    out: dict[str, dict] = {}

    out["nufft_bucket_count"] = {
        "source": glsl_header()
        + """
layout(constant_id = 0) const uint WG_SIZE       = 128u;
layout(constant_id = 1) const uint USE_SMEM_HIST = 0u;
layout(constant_id = 2) const uint HIST_BINS     = 1u;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer FirstBuf { int d_first[]; };
layout(std430, binding = 1) buffer CountBuf { int d_bucket_counts[]; };

layout(push_constant) uniform Push {
    int M;
    int w;
    int log2_win;
    int win_mask;
    int nw_mask;
} pc;

shared int l_hist[HIST_BINS];

void main() {
    int m = int(gl_GlobalInvocationID.x);
    if (USE_SMEM_HIST != 0u) {
        int tid = int(gl_LocalInvocationID.x);
        for (int i = tid; i < int(HIST_BINS); i += int(WG_SIZE)) {
            l_hist[i] = 0;
        }
        barrier();
        if (m < pc.M) {
            int first = d_first[m];
            int w0 = first >> pc.log2_win;
            atomicAdd(l_hist[w0], 1);
            if ((first & pc.win_mask) + pc.w > (pc.win_mask + 1)) {
                int w1 = (w0 + 1) & pc.nw_mask;
                atomicAdd(l_hist[w1], 1);
            }
        }
        barrier();
        for (int i = tid; i < int(HIST_BINS); i += int(WG_SIZE)) {
            int c = l_hist[i];
            if (c != 0) atomicAdd(d_bucket_counts[i], c);
        }
    } else {
        if (m >= pc.M) return;
        int first = d_first[m];
        int w0 = first >> pc.log2_win;
        atomicAdd(d_bucket_counts[w0], 1);
        if ((first & pc.win_mask) + pc.w > (pc.win_mask + 1)) {
            int w1 = (w0 + 1) & pc.nw_mask;
            atomicAdd(d_bucket_counts[w1], 1);
        }
    }
}
""",
        "check_nocontract": False,
    }

    out["nufft_bucket_scan"] = {
        "source": glsl_header()
        + """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer CountBuf { int d_bucket_counts[]; };
layout(std430, binding = 1) writeonly buffer OffBuf { int d_bucket_off[]; };

layout(push_constant) uniform Push {
    int num_wins;
} pc;

shared int l_scan[256];
shared int l_running;

void main() {
    int tid = int(gl_LocalInvocationID.x);
    if (pc.num_wins <= 256) {
        int val = (tid < pc.num_wins) ? d_bucket_counts[tid] : 0;
        l_scan[tid] = val;
        barrier();
        for (int off = 1; off < 256; off <<= 1) {
            int add = (tid >= off) ? l_scan[tid - off] : 0;
            barrier();
            l_scan[tid] += add;
            barrier();
        }
        int incl = l_scan[tid];
        int pref = incl - val;
        if (tid < pc.num_wins) {
            d_bucket_off[tid] = pref;
            d_bucket_counts[tid] = pref;
        }
        if (tid == pc.num_wins - 1) {
            d_bucket_off[pc.num_wins] = incl;
        }
    } else {
        if (tid == 0) l_running = 0;
        barrier();
        for (int base = 0; base < pc.num_wins; base += 256) {
            int wi = base + tid;
            int val = (wi < pc.num_wins) ? d_bucket_counts[wi] : 0;
            l_scan[tid] = val;
            barrier();
            for (int off = 1; off < 256; off <<= 1) {
                int add = (tid >= off) ? l_scan[tid - off] : 0;
                barrier();
                l_scan[tid] += add;
                barrier();
            }
            int incl = l_scan[tid];
            int pref = incl - val;
            int run = l_running;
            if (wi < pc.num_wins) {
                d_bucket_off[wi] = run + pref;
                d_bucket_counts[wi] = run + pref;
            }
            int last_tid = min(256, pc.num_wins - base) - 1;
            barrier();
            if (tid == last_tid) {
                l_running = run + incl;
            }
            barrier();
        }
        if (tid == 0) {
            d_bucket_off[pc.num_wins] = l_running;
        }
    }
}
""",
        "check_nocontract": False,
    }

    out["nufft_bucket_scatter"] = {
        "source": glsl_header()
        + """
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer FirstBuf { int d_first[]; };
layout(std430, binding = 1) buffer CountBuf { int d_bucket_counts[]; };
layout(std430, binding = 2) writeonly buffer MBuf { int d_bucket_m[]; };

layout(push_constant) uniform Push {
    int M;
    int w;
    int log2_win;
    int win_mask;
    int nw_mask;
} pc;

void main() {
    int m = int(gl_GlobalInvocationID.x);
    if (m >= pc.M) return;
    int first = d_first[m];
    int w0 = first >> pc.log2_win;
    int p0 = atomicAdd(d_bucket_counts[w0], 1);
    d_bucket_m[p0] = m;
    if ((first & pc.win_mask) + pc.w > (pc.win_mask + 1)) {
        int w1 = (w0 + 1) & pc.nw_mask;
        int p1 = atomicAdd(d_bucket_counts[w1], 1);
        d_bucket_m[p1] = m;
    }
}
""",
        "check_nocontract": False,
    }

    out["nufft_bucket_sort"] = {
        "source": glsl_header()
        + """
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer OffBuf { int d_bucket_off[]; };
layout(std430, binding = 1) buffer MBuf { int d_bucket_m[]; };

shared int l_idx[512];

void main() {
    int win  = int(gl_WorkGroupID.x);
    int tid  = int(gl_LocalInvocationID.x);
    int wg_n = int(gl_WorkGroupSize.x);
    int off  = d_bucket_off[win];
    int cnt  = d_bucket_off[win + 1] - off;
    if (cnt <= 1 || cnt > 512) return;

    int p2 = 1;
    while (p2 < cnt) p2 <<= 1;

    for (int i = tid; i < p2; i += wg_n) {
        l_idx[i] = (i < cnt) ? d_bucket_m[off + i] : 0x7fffffff;
    }
    barrier();

    for (int k = 2; k <= p2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < p2; i += wg_n) {
                int ixj = i ^ j;
                if (ixj > i) {
                    int a = l_idx[i];
                    int b = l_idx[ixj];
                    bool asc = ((i & k) == 0);
                    if ((asc && a > b) || (!asc && a < b)) {
                        l_idx[i] = b;
                        l_idx[ixj] = a;
                    }
                }
            }
            barrier();
        }
    }

    for (int i = tid; i < cnt; i += wg_n) {
        d_bucket_m[off + i] = l_idx[i];
    }
}
""",
        "check_nocontract": False,
    }

    return out


def _gen_pack_bucket_geom(prec: str) -> dict:
    if prec == "h":
        hdr = glsl_header("h", fp16_storage=True)
        buf_t, sc_t = "uint", "float16_t"
    elif prec == "f":
        hdr = glsl_header("f")
        buf_t, sc_t = "vec2", "float"
    elif prec == "df":
        hdr = glsl_header("df")
        buf_t, sc_t = "vec4", "vec2"
    else:
        hdr = glsl_header("d")
        buf_t, sc_t = "dvec2", "double"

    src = (
        hdr
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer OffBuf    {{ int    d_bucket_off[]; }};
layout(std430, binding = 1) readonly  buffer MBuf      {{ int    d_bucket_m[]; }};
layout(std430, binding = 2) readonly  buffer FirstBuf  {{ int    d_first[]; }};
layout(std430, binding = 3) readonly  buffer ShiftBuf  {{ {buf_t} d_shift[]; }};
layout(std430, binding = 4) readonly  buffer WBuf      {{ {sc_t}  d_weights[]; }};
layout(std430, binding = 5) writeonly buffer BFirstBuf {{ int    d_b_first[]; }};
layout(std430, binding = 6) writeonly buffer BShiftBuf {{ {buf_t} d_b_shift[]; }};
layout(std430, binding = 7) writeonly buffer BWBuf     {{ {sc_t}  d_b_weights[]; }};

layout(push_constant) uniform Push {{
    int w;
    int num_wins;
    int has_w;
}} pc;

void main() {{
    int p = int(gl_GlobalInvocationID.x);
    int total = d_bucket_off[pc.num_wins];
    if (p >= total) return;
    int m = d_bucket_m[p];
    d_b_first[p] = d_first[m];
    d_b_shift[p] = d_shift[m];
    if (pc.has_w != 0) {{
        uint src_off = uint(m) * uint(pc.w);
        uint dst_off = uint(p) * uint(pc.w);
        for (int j = 0; j < pc.w; ++j) {{
            d_b_weights[dst_off + uint(j)] = d_weights[src_off + uint(j)];
        }}
    }}
}}
"""
    )
    return {"source": src, "check_nocontract": False}


def _spread_prec_defs(prec: str) -> dict:
    if prec == "h":
        return {
            "hdr": glsl_header("h", fp16_storage=True) + glsl_f16x2_math(),
            "buf_t": "uint",
            "sc_t": "float16_t",
            "real_t": "f16vec2",
            "lanes": 2,
            "pack_smem": True,
            "smem_pair_t": "uvec2",
            "zero_expr": "kZero_f16x2",
            "zero_pair": "uvec2(0u, 0u)",
            "check_nc": False,
        }
    if prec == "f":
        return {
            "hdr": glsl_header("f"),
            "buf_t": "vec2",
            "sc_t": "float",
            "real_t": "float",
            "lanes": 1,
            "pack_smem": True,
            "smem_pair_t": "vec2",
            "zero_expr": "0.0f",
            "zero_pair": "vec2(0.0f, 0.0f)",
            "check_nc": False,
        }
    if prec == "df":
        return {
            "hdr": glsl_header("df") + glsl_df_math(),
            "buf_t": "vec4",
            "sc_t": "vec2",
            "real_t": "df_t",
            "lanes": 1,
            "pack_smem": False,
            "smem_pair_t": "",
            "zero_expr": "vec2(0.0f, 0.0f)",
            "zero_pair": "",
            "check_nc": True,
        }
    return {
        "hdr": glsl_header("d"),
        "buf_t": "dvec2",
        "sc_t": "double",
        "real_t": "double",
        "lanes": 1,
        "pack_smem": False,
        "smem_pair_t": "",
        "zero_expr": "0.0lf",
        "zero_pair": "",
        "check_nc": False,
    }


def _emit_load_shifted_y(prec: str) -> str:
    if prec == "h":
        return """
void load_shifted_y(uint bp, uint total_M_stride, uint chunk_bs, uint y_off, int m_idx, int shift_idx,
                    out f16vec2 sy_re, out f16vec2 sy_im) {
    f16vec2 sh = u32_to_f16x2(d_shift[shift_idx]);
    f16vec2 sr_v = f16vec2(sh.x, sh.x);
    f16vec2 si_v = f16vec2(sh.y, sh.y);
    uint b0 = 2u * bp;
    uint b1 = b0 + 1u;
    bool v0 = (b0 < chunk_bs);
    bool v1 = (b1 < chunk_bs);
    uint u0 = v0 ? d_y[y_off + b0 * total_M_stride + uint(m_idx)] : 0u;
    uint u1 = v1 ? d_y[y_off + b1 * total_M_stride + uint(m_idx)] : 0u;
    f16vec2 y_re, y_im;
    f16x2_load_pair_u32(u0, u1, y_re, y_im);
    f16x2_cmul(y_re, y_im, sr_v, si_v, sy_re, sy_im);
}
"""
    if prec == "f":
        return """
void load_shifted_y(uint bp, uint total_M_stride, uint chunk_bs, uint y_off, int m_idx, int shift_idx,
                    out float sy_re, out float sy_im) {
    vec2 sh = d_shift[shift_idx];
    vec2 y  = d_y[y_off + bp * total_M_stride + uint(m_idx)];
    sy_re = fma(y.x, sh.x, -(y.y * sh.y));
    sy_im = fma(y.x, sh.y,   y.y * sh.x);
}
"""
    if prec == "df":
        return """
void load_shifted_y(uint bp, uint total_M_stride, uint chunk_bs, uint y_off, int m_idx, int shift_idx,
                    out df_t sy_re, out df_t sy_im) {
    vec4 sh = d_shift[shift_idx];
    vec4 y  = d_y[y_off + bp * total_M_stride + uint(m_idx)];
    df_cmul_fused(y.xy, y.zw, sh.xy, sh.zw, sy_re, sy_im);
}
"""
    return """
void load_shifted_y(uint bp, uint total_M_stride, uint chunk_bs, uint y_off, int m_idx, int shift_idx,
                    out double sy_re, out double sy_im) {
    dvec2 sh = d_shift[shift_idx];
    dvec2 y  = d_y[y_off + bp * total_M_stride + uint(m_idx)];
    sy_re = fma(y.x, sh.x, -(y.y * sh.y));
    sy_im = fma(y.x, sh.y,   y.y * sh.x);
}
"""


def _gen_spread_smem(prec: str) -> dict:
    d = _spread_prec_defs(prec)

    if d["pack_smem"]:
        smem_decl = f"""
shared {d["smem_pair_t"]} l_grid[SMEM_GRID_ELEMS];
shared {d["smem_pair_t"]} l_sy[SMEM_SY_ELEMS];
"""
        zero_body = f"        l_grid[idx] = {d['zero_pair']};"
        if prec == "h":
            store_sy = "                    l_sy[sy_slot] = uvec2(f16x2_to_u32(sy_re), f16x2_to_u32(sy_im));"
            load_sy = """                uvec2 sy_p = l_sy[k * sy_stride + eff_lane_b];
                f16vec2 sy_re = u32_to_f16x2(sy_p.x);
                f16vec2 sy_im = u32_to_f16x2(sy_p.y);"""
            accum_code = """
                    int g_idx = grid_base + bin;
                    float16_t ws = l_w[w_base + j_val];
                    f16vec2 wv = f16vec2(ws, ws);
                    uvec2 gp = l_grid[g_idx];
                    f16vec2 gr = fma(sy_re, wv, u32_to_f16x2(gp.x));
                    f16vec2 gi = fma(sy_im, wv, u32_to_f16x2(gp.y));
                    l_grid[g_idx] = uvec2(f16x2_to_u32(gr), f16x2_to_u32(gi));"""
            red_pair_code = """
                        uvec2 ga = l_grid[dst_off + n];
                        uvec2 gb = l_grid[src_off + n];
                        f16vec2 sr = u32_to_f16x2(ga.x) + u32_to_f16x2(gb.x);
                        f16vec2 si = u32_to_f16x2(ga.y) + u32_to_f16x2(gb.y);
                        l_grid[dst_off + n] = uvec2(f16x2_to_u32(sr), f16x2_to_u32(si));"""
            wb_code = """
        uint b0_out = 2u * bp_out;
        uint b1_out = b0_out + 1u;
        bool v0 = (b0_out < pc.chunk_bs);
        bool v1 = (b1_out < pc.chunk_bs);
        uint dst0 = km_off + b0_out * uint(pc.nfft) + uint(win_lo);
        uint dst1 = km_off + b1_out * uint(pc.nfft) + uint(win_lo);
        for (int n = tid; n < pc.win_elems; n += int(WG_SIZE)) {
            uvec2 gp = l_grid[g_off + n];
            uint u0, u1;
            f16x2_pack_pair_u32(u32_to_f16x2(gp.x), u32_to_f16x2(gp.y), u0, u1);
            if (v0) d_fft_buf[dst0 + uint(n)] = u0;
            if (v1) d_fft_buf[dst1 + uint(n)] = u1;
        }"""
        else:  # f
            store_sy = "                    l_sy[sy_slot] = vec2(sy_re, sy_im);"
            load_sy = """                vec2 sy_p = l_sy[k * sy_stride + eff_lane_b];
                float sy_re = sy_p.x;
                float sy_im = sy_p.y;"""
            accum_code = """
                    int g_idx = grid_base + bin;
                    float wv = l_w[w_base + j_val];
                    vec2 gp = l_grid[g_idx];
                    gp.x = fma(sy_re, wv, gp.x);
                    gp.y = fma(sy_im, wv, gp.y);
                    l_grid[g_idx] = gp;"""
            red_pair_code = """
                        l_grid[dst_off + n] += l_grid[src_off + n];"""
            wb_code = """
        uint dst = km_off + bp_out * uint(pc.nfft) + uint(win_lo);
        for (int n = tid; n < pc.win_elems; n += int(WG_SIZE)) {
            d_fft_buf[dst + uint(n)] = l_grid[g_off + n];
        }"""
    else:
        smem_decl = f"""
shared {d["real_t"]} l_re[SMEM_GRID_ELEMS];
shared {d["real_t"]} l_im[SMEM_GRID_ELEMS];
shared {d["real_t"]} l_sy_re[SMEM_SY_ELEMS];
shared {d["real_t"]} l_sy_im[SMEM_SY_ELEMS];
"""
        zero_body = f"        l_re[idx] = {d['zero_expr']};\n        l_im[idx] = {d['zero_expr']};"
        store_sy = "                    l_sy_re[sy_slot] = sy_re;\n                    l_sy_im[sy_slot] = sy_im;"
        load_sy = f"""                {d["real_t"]} sy_re = l_sy_re[k * sy_stride + eff_lane_b];
                {d["real_t"]} sy_im = l_sy_im[k * sy_stride + eff_lane_b];"""
        if prec == "df":
            accum_code = """
                    int g_idx = grid_base + bin;
                    df_t wv = l_w[w_base + j_val];
                    l_re[g_idx] = df_add(df_mul(sy_re, wv), l_re[g_idx]);
                    l_im[g_idx] = df_add(df_mul(sy_im, wv), l_im[g_idx]);"""
            red_pair_code = """
                        l_re[dst_off + n] = df_add(l_re[dst_off + n], l_re[src_off + n]);
                        l_im[dst_off + n] = df_add(l_im[dst_off + n], l_im[src_off + n]);"""
            wb_code = """
        uint dst = km_off + bp_out * uint(pc.nfft) + uint(win_lo);
        for (int n = tid; n < pc.win_elems; n += int(WG_SIZE)) {
            df_t gr = l_re[g_off + n];
            df_t gi = l_im[g_off + n];
            d_fft_buf[dst + uint(n)] = vec4(gr.x, gr.y, gi.x, gi.y);
        }"""
        else:  # d
            accum_code = """
                    int g_idx = grid_base + bin;
                    double wv = l_w[w_base + j_val];
                    l_re[g_idx] = fma(sy_re, wv, l_re[g_idx]);
                    l_im[g_idx] = fma(sy_im, wv, l_im[g_idx]);"""
            red_pair_code = """
                        l_re[dst_off + n] += l_re[src_off + n];
                        l_im[dst_off + n] += l_im[src_off + n];"""
            wb_code = """
        uint dst = km_off + bp_out * uint(pc.nfft) + uint(win_lo);
        for (int n = tid; n < pc.win_elems; n += int(WG_SIZE)) {
            d_fft_buf[dst + uint(n)] = dvec2(l_re[g_off + n], l_im[g_off + n]);
        }"""

    src = (
        d["hdr"]
        + f"""
layout(constant_id = 0)  const uint WG_SIZE         = 64u;
layout(constant_id = 1)  const uint W_CONST         = 8u;
layout(constant_id = 2)  const uint T_J             = 8u;
layout(constant_id = 3)  const uint LOG2_TJ         = 3u;
layout(constant_id = 4)  const uint B_PER_WG        = 8u;
layout(constant_id = 5)  const uint TILE_M          = 32u;
layout(constant_id = 6)  const uint LOG2_TILE_M     = 5u;
layout(constant_id = 7)  const uint GRID_STRIDE     = 256u;
layout(constant_id = 8)  const uint SMEM_GRID_ELEMS = 2048u;
layout(constant_id = 9)  const uint SMEM_SY_ELEMS   = 288u;
layout(constant_id = 10) const uint SMEM_W_ELEMS    = 256u;
layout(constant_id = 11) const uint MULTIPASS       = 0u;
layout(constant_id = 12) const uint SPLIT_M         = 1u;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer WBuf     {{ {d["sc_t"]}  d_weights[]; }};
layout(std430, binding = 1) readonly  buffer FirstBuf {{ int          d_first[]; }};
layout(std430, binding = 2) readonly  buffer ShiftBuf {{ {d["buf_t"]} d_shift[]; }};
layout(std430, binding = 3) readonly  buffer OffBuf   {{ int          d_bucket_off[]; }};
layout(std430, binding = 4) readonly  buffer MBuf     {{ int          d_bucket_m[]; }};
layout(std430, binding = 5) readonly  buffer YBuf     {{ {d["buf_t"]} d_y[]; }};
layout(std430, binding = 6) writeonly buffer FftBuf   {{ {d["buf_t"]} d_fft_buf[]; }};

layout(push_constant) uniform Push {{
    int  M;
    int  nfft;
    int  win_elems;
    int  num_wins;
    int  log2_nw;
    uint total_M_stride;
    uint chunk_bs;
    uint y_offset_elems;
    uint k_m_splits;
    uint dst_km_stride;
}} pc;
{smem_decl}
shared int l_first[TILE_M];
shared {d["sc_t"]} l_w[SMEM_W_ELEMS];
{_emit_load_shifted_y(prec)}
void main() {{
    int tid = int(gl_LocalInvocationID.x);
    uint wg_id = gl_WorkGroupID.x;
    uint nw_mask = uint(pc.num_wins - 1);
    uint b_group = wg_id >> uint(pc.log2_nw);
    int win = int(wg_id & nw_mask);
    int win_lo = win * pc.win_elems;

    int lane_b = tid >> int(LOG2_TJ);
    int lane_t = tid & int(T_J - 1u);
    int eff_b_slots = int(B_PER_WG / SPLIT_M);
    int eff_lane_b  = (SPLIT_M > 1u) ? (lane_b & (eff_b_slots - 1)) : lane_b;
    int m_slice     = (SPLIT_M > 1u) ? (lane_b / eff_b_slots) : 0;
    int sy_stride = int(B_PER_WG) + 1;
    uint bp_base = b_group * B_PER_WG;
    uint my_bp   = bp_base + uint(eff_lane_b);
    int grid_base = lane_b * int(GRID_STRIDE);
    int nfft_mask = pc.nfft - 1;
    uint bs_packed = (pc.chunk_bs + {d["lanes"] - 1}u) / {d["lanes"]}u;

    for (int idx = tid; idx < int(SMEM_GRID_ELEMS); idx += int(WG_SIZE)) {{
{zero_body}
    }}
    barrier();

    int p_start = (pc.num_wins == 1) ? 0 : d_bucket_off[win];
    int p_end   = (pc.num_wins == 1) ? pc.M : d_bucket_off[win + 1];
    uint km = gl_WorkGroupID.y;
    if (pc.k_m_splits > 1u) {{
        int total_p = p_end - p_start;
        int chunk_p = (total_p + int(pc.k_m_splits) - 1) / int(pc.k_m_splits);
        int my_lo = p_start + int(km) * chunk_p;
        p_end = min(p_end, my_lo + chunk_p);
        p_start = min(my_lo, p_end);
    }}
    uint km_off = km * pc.dst_km_stride;

    for (int p_base = p_start; p_base < p_end; p_base += int(TILE_M)) {{
        int cur_tile = min(int(TILE_M), p_end - p_base);

        if (tid < cur_tile) {{
            l_first[tid] = d_first[p_base + tid];
        }}
        int total_w = cur_tile * int(W_CONST);
        uint w_src_off = uint(p_base) * W_CONST;
        for (int idx = tid; idx < total_w; idx += int(WG_SIZE)) {{
            l_w[idx] = d_weights[w_src_off + uint(idx)];
        }}

        int k_pow2 = int(TILE_M);
        int log2_k = int(LOG2_TILE_M);
        while (k_pow2 > 1 && (k_pow2 >> 1) >= cur_tile) {{
            k_pow2 >>= 1;
            --log2_k;
        }}
        int k_mask = k_pow2 - 1;
        int total_sy = eff_b_slots * k_pow2;
        for (int idx = tid; idx < total_sy; idx += int(WG_SIZE)) {{
            int b_sub = idx >> log2_k;
            int k = idx & k_mask;
            if (k < cur_tile) {{
                uint bp = bp_base + uint(b_sub);
                if (bp < bs_packed) {{
                    int p_idx = p_base + k;
                    int m_idx = (pc.num_wins == 1) ? p_idx : d_bucket_m[p_idx];
                    {d["real_t"]} sy_re, sy_im;
                    load_shifted_y(bp, pc.total_M_stride, pc.chunk_bs, pc.y_offset_elems, m_idx, p_idx, sy_re, sy_im);
                    int sy_slot = k * sy_stride + b_sub;
{store_sy}
                }}
            }}
        }}

        barrier();

        if (my_bp < bs_packed) {{
            for (int k = m_slice; k < cur_tile; k += int(SPLIT_M)) {{
                int first = l_first[k];
                int j0 = (lane_t - first) & int(T_J - 1u);
{load_sy}
                int w_base = k * int(W_CONST);

                if (MULTIPASS != 0u) {{
                    int j_val = j0;
                    int bin = ((first + j_val) & nfft_mask) - win_lo;
                    if (pc.num_wins == 1 || uint(bin) < uint(pc.win_elems)) {{
{accum_code}
                    }}
                    int j1 = j0 + int(T_J);
                    if (j1 < int(W_CONST)) {{
                        j_val = j1;
                        bin = ((first + j_val) & nfft_mask) - win_lo;
                        if (pc.num_wins == 1 || uint(bin) < uint(pc.win_elems)) {{
{accum_code}
                        }}
                    }}
                }} else {{
                    if (j0 < int(W_CONST)) {{
                        int j_val = j0;
                        int bin = ((first + j_val) & nfft_mask) - win_lo;
                        if (pc.num_wins == 1 || uint(bin) < uint(pc.win_elems)) {{
{accum_code}
                        }}
                    }}
                }}
            }}
        }}

        barrier();
    }}

    if (SPLIT_M > 1u) {{
        for (int s = int(SPLIT_M >> 1); s >= 1; s >>= 1) {{
            for (int b_sub = 0; b_sub < eff_b_slots; ++b_sub) {{
                uint bp_out = bp_base + uint(b_sub);
                if (bp_out >= bs_packed) break;
                for (int ms = 0; ms < s; ++ms) {{
                    int dst_off = (ms * eff_b_slots + b_sub) * int(GRID_STRIDE);
                    int src_off = ((ms + s) * eff_b_slots + b_sub) * int(GRID_STRIDE);
                    for (int n = tid; n < pc.win_elems; n += int(WG_SIZE)) {{{red_pair_code}
                    }}
                }}
            }}
            barrier();
        }}
    }}

    for (int b_sub = 0; b_sub < eff_b_slots; ++b_sub) {{
        uint bp_out = bp_base + uint(b_sub);
        if (bp_out >= bs_packed) break;
        int g_off = b_sub * int(GRID_STRIDE);
{wb_code}
    }}
}}
"""
    )
    return {"source": src, "check_nocontract": d["check_nc"]}


def _emit_bsearch_tree(lo: int, hi: int, indent: str, leaf_fn) -> list[str]:
    if hi - lo == 1:
        return [indent + line for line in leaf_fn(lo)]
    mid = (lo + hi) // 2
    lines = [f"{indent}if (q_idx < {mid}) {{"]
    lines.extend(_emit_bsearch_tree(lo, mid, indent + "    ", leaf_fn))
    lines.append(f"{indent}}} else {{")
    lines.extend(_emit_bsearch_tree(mid, hi, indent + "    ", leaf_fn))
    lines.append(f"{indent}}}")
    return lines


def _gen_spread_reg(prec: str, q_val: int) -> dict:
    d = _spread_prec_defs(prec)

    if d["pack_smem"]:
        smem_decl = f"shared {d['smem_pair_t']} l_sy[SMEM_SY_ELEMS];\nshared {d['smem_pair_t']} l_red[SMEM_RED_ELEMS];"
        if prec == "h":
            store_sy = "                    l_sy[sy_slot] = uvec2(f16x2_to_u32(sy_re), f16x2_to_u32(sy_im));"
            load_sy = """                uvec2 sy_p = l_sy[k * sy_stride + eff_lane_b];
                f16vec2 sy_re = u32_to_f16x2(sy_p.x);
                f16vec2 sy_im = u32_to_f16x2(sy_p.y);"""

            def red_store(qi):
                return f"        l_red[tid] = uvec2(f16x2_to_u32(acc_re[{qi}]), f16x2_to_u32(acc_im[{qi}]));"

            def red_add(qi):
                return f"            {{ uvec2 rp = l_red[ridx]; acc_re[{qi}] += u32_to_f16x2(rp.x); acc_im[{qi}] += u32_to_f16x2(rp.y); }}"
        else:
            store_sy = "                    l_sy[sy_slot] = vec2(sy_re, sy_im);"
            load_sy = """                vec2 sy_p = l_sy[k * sy_stride + eff_lane_b];
                float sy_re = sy_p.x;
                float sy_im = sy_p.y;"""

            def red_store(qi):
                return f"        l_red[tid] = vec2(acc_re[{qi}], acc_im[{qi}]);"

            def red_add(qi):
                return f"            {{ vec2 rp = l_red[ridx]; acc_re[{qi}] += rp.x; acc_im[{qi}] += rp.y; }}"
    else:
        smem_decl = f"""shared {d["real_t"]} l_sy_re[SMEM_SY_ELEMS];
shared {d["real_t"]} l_sy_im[SMEM_SY_ELEMS];
shared {d["real_t"]} l_red_re[SMEM_RED_ELEMS];
shared {d["real_t"]} l_red_im[SMEM_RED_ELEMS];"""
        store_sy = "                    l_sy_re[sy_slot] = sy_re;\n                    l_sy_im[sy_slot] = sy_im;"
        load_sy = f"""                {d["real_t"]} sy_re = l_sy_re[k * sy_stride + eff_lane_b];
                {d["real_t"]} sy_im = l_sy_im[k * sy_stride + eff_lane_b];"""

        def red_store(qi):
            return f"        l_red_re[tid] = acc_re[{qi}]; l_red_im[tid] = acc_im[{qi}];"

        if prec == "df":

            def red_add(qi):
                return f"            acc_re[{qi}] = df_add(acc_re[{qi}], l_red_re[ridx]); acc_im[{qi}] = df_add(acc_im[{qi}], l_red_im[ridx]);"
        else:

            def red_add(qi):
                return f"            acc_re[{qi}] += l_red_re[ridx]; acc_im[{qi}] += l_red_im[ridx];"

    # Register accumulator update helper using compile-time binary search (0 local spills)
    if prec == "df":
        get_tree = "\n".join(_emit_bsearch_tree(0, q_val, "    ", lambda i: [f"cr = acc_re[{i}]; ci = acc_im[{i}];"]))
        set_tree = "\n".join(_emit_bsearch_tree(0, q_val, "    ", lambda i: [f"acc_re[{i}] = cr; acc_im[{i}] = ci;"]))
        reg_helper = f"""
void reg_acc_update(int q_idx, inout df_t acc_re[{q_val}], inout df_t acc_im[{q_val}], df_t sy_re, df_t sy_im, df_t wv) {{
    df_t pr = df_mul(sy_re, wv);
    df_t pi = df_mul(sy_im, wv);
    df_t cr, ci;
{get_tree}
    cr = df_add(cr, pr);
    ci = df_add(ci, pi);
{set_tree}
}}
"""
    else:
        fma_tree = "\n".join(
            _emit_bsearch_tree(
                0,
                q_val,
                "    ",
                lambda i: [
                    f"acc_re[{i}] = fma(sy_re, wv, acc_re[{i}]);",
                    f"acc_im[{i}] = fma(sy_im, wv, acc_im[{i}]);",
                ],
            )
        )
        reg_helper = f"""
void reg_acc_update(int q_idx, inout {d["real_t"]} acc_re[{q_val}], inout {d["real_t"]} acc_im[{q_val}],
                    {d["real_t"]} sy_re, {d["real_t"]} sy_im, {d["real_t"]} wv) {{
{fma_tree}
}}
"""

    wv_expr = "f16vec2(ws, ws)" if prec == "h" else "ws"
    accum_call = f"""
                    int q_idx = bin >> int(LOG2_TJ);
                    {d["sc_t"]} ws = l_w[w_base + j_val];
                    {d["real_t"]} wv = {wv_expr};
                    reg_acc_update(q_idx, acc_re, acc_im, sy_re, sy_im, wv);"""

    # Unrolled direct register-to-global writeback (and optional SPLIT_M SMEM reduction)
    wb_lines = []
    red_wb_lines = []
    if prec == "h":
        hdr_wb = [
            "        uint b0_out = 2u * my_bp;",
            "        uint b1_out = b0_out + 1u;",
            "        bool v0 = (b0_out < pc.chunk_bs);",
            "        bool v1 = (b1_out < pc.chunk_bs);",
            "        uint dst0 = km_off + b0_out * uint(pc.nfft) + uint(win_lo);",
            "        uint dst1 = km_off + b1_out * uint(pc.nfft) + uint(win_lo);",
        ]
        wb_lines.extend(hdr_wb)
        for qi in range(q_val):
            wb_lines.extend(
                [
                    f"        {{ uint u0, u1; f16x2_pack_pair_u32(acc_re[{qi}], acc_im[{qi}], u0, u1);",
                    f"          uint off_n = uint({qi}) * T_J + uint(lane_t);",
                    "          if (v0) d_fft_buf[dst0 + off_n] = u0;",
                    "          if (v1) d_fft_buf[dst1 + off_n] = u1; }",
                ]
            )
        for qi in range(q_val):
            red_wb_lines.append(red_store(qi))
            red_wb_lines.append("        barrier();")
            red_wb_lines.append("        if (m_slice == 0 && my_bp < bs_packed) {")
            red_wb_lines.append("            for (int ms = 1; ms < int(SPLIT_M); ++ms) {")
            red_wb_lines.append("                int ridx = (ms * eff_b_slots + eff_lane_b) * int(T_J) + lane_t;")
            red_wb_lines.append(red_add(qi))
            red_wb_lines.append("            }")
            red_wb_lines.append("            uint b0_out = 2u * my_bp; uint b1_out = b0_out + 1u;")
            red_wb_lines.append("            uint u0, u1;")
            red_wb_lines.append(f"            f16x2_pack_pair_u32(acc_re[{qi}], acc_im[{qi}], u0, u1);")
            red_wb_lines.append(f"            uint off_n = uint({qi}) * T_J + uint(lane_t);")
            red_wb_lines.append("            if (b0_out < pc.chunk_bs) d_fft_buf[km_off + b0_out * uint(pc.nfft) + uint(win_lo) + off_n] = u0;")
            red_wb_lines.append("            if (b1_out < pc.chunk_bs) d_fft_buf[km_off + b1_out * uint(pc.nfft) + uint(win_lo) + off_n] = u1;")
            red_wb_lines.append("        }")
            if qi + 1 < q_val:
                red_wb_lines.append("        barrier();")
    else:
        if prec == "df":

            def pack_expr(qi):
                return f"vec4(acc_re[{qi}].x, acc_re[{qi}].y, acc_im[{qi}].x, acc_im[{qi}].y)"
        elif prec == "f":

            def pack_expr(qi):
                return f"vec2(acc_re[{qi}], acc_im[{qi}])"
        else:

            def pack_expr(qi):
                return f"dvec2(acc_re[{qi}], acc_im[{qi}])"

        wb_lines.append("        uint dst = km_off + my_bp * uint(pc.nfft) + uint(win_lo);")
        for qi in range(q_val):
            wb_lines.append(f"        d_fft_buf[dst + uint({qi}) * T_J + uint(lane_t)] = {pack_expr(qi)};")
        for qi in range(q_val):
            red_wb_lines.append(red_store(qi))
            red_wb_lines.append("        barrier();")
            red_wb_lines.append("        if (m_slice == 0 && my_bp < bs_packed) {")
            red_wb_lines.append("            for (int ms = 1; ms < int(SPLIT_M); ++ms) {")
            red_wb_lines.append("                int ridx = (ms * eff_b_slots + eff_lane_b) * int(T_J) + lane_t;")
            red_wb_lines.append(red_add(qi))
            red_wb_lines.append("            }")
            red_wb_lines.append("            uint dst = km_off + my_bp * uint(pc.nfft) + uint(win_lo);")
            red_wb_lines.append(f"            d_fft_buf[dst + uint({qi}) * T_J + uint(lane_t)] = {pack_expr(qi)};")
            red_wb_lines.append("        }")
            if qi + 1 < q_val:
                red_wb_lines.append("        barrier();")

    init_acc = "\n".join(f"    acc_re[{qi}] = {d['zero_expr']}; acc_im[{qi}] = {d['zero_expr']};" for qi in range(q_val))

    src = (
        d["hdr"]
        + f"""
layout(constant_id = 0)  const uint WG_SIZE        = 64u;
layout(constant_id = 1)  const uint W_CONST        = 8u;
layout(constant_id = 2)  const uint T_J            = 8u;
layout(constant_id = 3)  const uint LOG2_TJ        = 3u;
layout(constant_id = 4)  const uint B_PER_WG       = 8u;
layout(constant_id = 5)  const uint TILE_M         = 32u;
layout(constant_id = 6)  const uint LOG2_TILE_M    = 5u;
layout(constant_id = 9)  const uint SMEM_SY_ELEMS  = 288u;
layout(constant_id = 10) const uint SMEM_W_ELEMS   = 256u;
layout(constant_id = 11) const uint MULTIPASS      = 0u;
layout(constant_id = 12) const uint SPLIT_M        = 1u;
layout(constant_id = 13) const uint SMEM_RED_ELEMS = 1u;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer WBuf     {{ {d["sc_t"]}  d_weights[]; }};
layout(std430, binding = 1) readonly  buffer FirstBuf {{ int          d_first[]; }};
layout(std430, binding = 2) readonly  buffer ShiftBuf {{ {d["buf_t"]} d_shift[]; }};
layout(std430, binding = 3) readonly  buffer OffBuf   {{ int          d_bucket_off[]; }};
layout(std430, binding = 4) readonly  buffer MBuf     {{ int          d_bucket_m[]; }};
layout(std430, binding = 5) readonly  buffer YBuf     {{ {d["buf_t"]} d_y[]; }};
layout(std430, binding = 6) writeonly buffer FftBuf   {{ {d["buf_t"]} d_fft_buf[]; }};

layout(push_constant) uniform Push {{
    int  M;
    int  nfft;
    int  win_elems;
    int  num_wins;
    int  log2_nw;
    uint total_M_stride;
    uint chunk_bs;
    uint y_offset_elems;
    uint k_m_splits;
    uint dst_km_stride;
}} pc;

{smem_decl}
shared int l_first[TILE_M];
shared {d["sc_t"]} l_w[SMEM_W_ELEMS];
{_emit_load_shifted_y(prec)}
{reg_helper}
void main() {{
    int tid = int(gl_LocalInvocationID.x);
    uint wg_id = gl_WorkGroupID.x;
    uint nw_mask = uint(pc.num_wins - 1);
    uint b_group = wg_id >> uint(pc.log2_nw);
    int win = int(wg_id & nw_mask);
    int win_lo = win * pc.win_elems;

    int lane_b = tid >> int(LOG2_TJ);
    int lane_t = tid & int(T_J - 1u);
    int eff_b_slots = int(B_PER_WG / SPLIT_M);
    int eff_lane_b  = (SPLIT_M > 1u) ? (lane_b & (eff_b_slots - 1)) : lane_b;
    int m_slice     = (SPLIT_M > 1u) ? (lane_b / eff_b_slots) : 0;
    int sy_stride = int(B_PER_WG) + 1;
    uint bp_base = b_group * B_PER_WG;
    uint my_bp   = bp_base + uint(eff_lane_b);
    int nfft_mask = pc.nfft - 1;
    uint bs_packed = (pc.chunk_bs + {d["lanes"] - 1}u) / {d["lanes"]}u;

    {d["real_t"]} acc_re[{q_val}];
    {d["real_t"]} acc_im[{q_val}];
{init_acc}

    int p_start = (pc.num_wins == 1) ? 0 : d_bucket_off[win];
    int p_end   = (pc.num_wins == 1) ? pc.M : d_bucket_off[win + 1];
    uint km = gl_WorkGroupID.y;
    if (pc.k_m_splits > 1u) {{
        int total_p = p_end - p_start;
        int chunk_p = (total_p + int(pc.k_m_splits) - 1) / int(pc.k_m_splits);
        int my_lo = p_start + int(km) * chunk_p;
        p_end = min(p_end, my_lo + chunk_p);
        p_start = min(my_lo, p_end);
    }}
    uint km_off = km * pc.dst_km_stride;

    for (int p_base = p_start; p_base < p_end; p_base += int(TILE_M)) {{
        int cur_tile = min(int(TILE_M), p_end - p_base);

        if (tid < cur_tile) {{
            l_first[tid] = d_first[p_base + tid];
        }}
        int total_w = cur_tile * int(W_CONST);
        uint w_src_off = uint(p_base) * W_CONST;
        for (int idx = tid; idx < total_w; idx += int(WG_SIZE)) {{
            l_w[idx] = d_weights[w_src_off + uint(idx)];
        }}

        int k_pow2 = int(TILE_M);
        int log2_k = int(LOG2_TILE_M);
        while (k_pow2 > 1 && (k_pow2 >> 1) >= cur_tile) {{
            k_pow2 >>= 1;
            --log2_k;
        }}
        int k_mask = k_pow2 - 1;
        int total_sy = eff_b_slots * k_pow2;
        for (int idx = tid; idx < total_sy; idx += int(WG_SIZE)) {{
            int b_sub = idx >> log2_k;
            int k = idx & k_mask;
            if (k < cur_tile) {{
                uint bp = bp_base + uint(b_sub);
                if (bp < bs_packed) {{
                    int p_idx = p_base + k;
                    int m_idx = (pc.num_wins == 1) ? p_idx : d_bucket_m[p_idx];
                    {d["real_t"]} sy_re, sy_im;
                    load_shifted_y(bp, pc.total_M_stride, pc.chunk_bs, pc.y_offset_elems, m_idx, p_idx, sy_re, sy_im);
                    int sy_slot = k * sy_stride + b_sub;
{store_sy}
                }}
            }}
        }}

        barrier();

        if (my_bp < bs_packed) {{
            for (int k = m_slice; k < cur_tile; k += int(SPLIT_M)) {{
                int first = l_first[k];
                int j0 = (lane_t - first) & int(T_J - 1u);
{load_sy}
                int w_base = k * int(W_CONST);

                if (MULTIPASS != 0u) {{
                    int j_val = j0;
                    int bin = ((first + j_val) & nfft_mask) - win_lo;
                    if (pc.num_wins == 1 || uint(bin) < uint(pc.win_elems)) {{
{accum_call}
                    }}
                    int j1 = j0 + int(T_J);
                    if (j1 < int(W_CONST)) {{
                        j_val = j1;
                        bin = ((first + j_val) & nfft_mask) - win_lo;
                        if (pc.num_wins == 1 || uint(bin) < uint(pc.win_elems)) {{
{accum_call}
                        }}
                    }}
                }} else {{
                    if (j0 < int(W_CONST)) {{
                        int j_val = j0;
                        int bin = ((first + j_val) & nfft_mask) - win_lo;
                        if (pc.num_wins == 1 || uint(bin) < uint(pc.win_elems)) {{
{accum_call}
                        }}
                    }}
                }}
            }}
        }}

        if (p_base + int(TILE_M) < p_end) {{
            barrier();
        }}
    }}

    if (SPLIT_M > 1u) {{
{chr(10).join(red_wb_lines)}
    }} else if (my_bp < bs_packed) {{
{chr(10).join(wb_lines)}
    }}
}}
"""
    )
    return {"source": src, "check_nocontract": d["check_nc"]}


def _gen_spread_cpu_direct(prec: str) -> dict:
    d = _spread_prec_defs(prec)

    if prec == "h":
        zero_loop = """
    uint b0 = 2u * bp;
    uint b1 = b0 + 1u;
    bool v0 = (b0 < pc.chunk_bs);
    bool v1 = (b1 < pc.chunk_bs);
    uint dst0 = b0 * uint(pc.nfft);
    uint dst1 = b1 * uint(pc.nfft);
    for (int n = 0; n < pc.nfft; ++n) {
        if (v0) d_fft_buf[dst0 + uint(n)] = 0u;
        if (v1) d_fft_buf[dst1 + uint(n)] = 0u;
    }"""
        inner_accum = """
            float16_t ws = d_weights[w_off + uint(j)];
            f16vec2 wv = f16vec2(ws, ws);
            uint u0 = v0 ? d_fft_buf[dst0 + uint(bin)] : 0u;
            uint u1 = v1 ? d_fft_buf[dst1 + uint(bin)] : 0u;
            f16vec2 cur_re, cur_im;
            f16x2_load_pair_u32(u0, u1, cur_re, cur_im);
            cur_re = fma(sy_re, wv, cur_re);
            cur_im = fma(sy_im, wv, cur_im);
            f16x2_pack_pair_u32(cur_re, cur_im, u0, u1);
            if (v0) d_fft_buf[dst0 + uint(bin)] = u0;
            if (v1) d_fft_buf[dst1 + uint(bin)] = u1;"""
    elif prec == "f":
        zero_loop = """
    uint dst = bp * uint(pc.nfft);
    for (int n = 0; n < pc.nfft; ++n) {
        d_fft_buf[dst + uint(n)] = vec2(0.0f, 0.0f);
    }"""
        inner_accum = """
            float wv = d_weights[w_off + uint(j)];
            vec2 cur = d_fft_buf[dst + uint(bin)];
            d_fft_buf[dst + uint(bin)] = vec2(fma(sy_re, wv, cur.x), fma(sy_im, wv, cur.y));"""
    elif prec == "df":
        zero_loop = """
    uint dst = bp * uint(pc.nfft);
    for (int n = 0; n < pc.nfft; ++n) {
        d_fft_buf[dst + uint(n)] = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }"""
        inner_accum = """
            df_t wv = d_weights[w_off + uint(j)];
            vec4 cur = d_fft_buf[dst + uint(bin)];
            df_t nr = df_add(df_mul(sy_re, wv), cur.xy);
            df_t ni = df_add(df_mul(sy_im, wv), cur.zw);
            d_fft_buf[dst + uint(bin)] = vec4(nr.x, nr.y, ni.x, ni.y);"""
    else:
        zero_loop = """
    uint dst = bp * uint(pc.nfft);
    for (int n = 0; n < pc.nfft; ++n) {
        d_fft_buf[dst + uint(n)] = dvec2(0.0lf, 0.0lf);
    }"""
        inner_accum = """
            double wv = d_weights[w_off + uint(j)];
            dvec2 cur = d_fft_buf[dst + uint(bin)];
            d_fft_buf[dst + uint(bin)] = dvec2(fma(sy_re, wv, cur.x), fma(sy_im, wv, cur.y));"""

    src = (
        d["hdr"]
        + f"""
layout(constant_id = 0) const uint WG_SIZE = 1u;
layout(constant_id = 1) const uint W_CONST = 8u;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer WBuf     {{ {d["sc_t"]}  d_weights[]; }};
layout(std430, binding = 1) readonly buffer FirstBuf {{ int          d_first[]; }};
layout(std430, binding = 2) readonly buffer ShiftBuf {{ {d["buf_t"]} d_shift[]; }};
layout(std430, binding = 3) readonly buffer YBuf     {{ {d["buf_t"]} d_y[]; }};
layout(std430, binding = 4) buffer          FftBuf   {{ {d["buf_t"]} d_fft_buf[]; }};

layout(push_constant) uniform Push {{
    int  M;
    int  nfft;
    uint total_M_stride;
    uint chunk_bs;
    uint y_offset_elems;
}} pc;
{_emit_load_shifted_y(prec)}
void main() {{
    uint bp = gl_GlobalInvocationID.x;
    uint bs_packed = (pc.chunk_bs + {d["lanes"] - 1}u) / {d["lanes"]}u;
    if (bp >= bs_packed) return;
    int nfft_mask = pc.nfft - 1;
{zero_loop}
    for (int m = 0; m < pc.M; ++m) {{
        {d["real_t"]} sy_re, sy_im;
        load_shifted_y(bp, pc.total_M_stride, pc.chunk_bs, pc.y_offset_elems, m, m, sy_re, sy_im);
        int first = d_first[m];
        uint w_off = uint(m) * W_CONST;
        for (int j = 0; j < int(W_CONST); ++j) {{
            int bin = (first + j) & nfft_mask;
{inner_accum}
        }}
    }}
}}
"""
    )
    return {"source": src, "check_nocontract": d["check_nc"]}


def _emit_eval_pswf_phi_fn(prec: str, w: int, n_even: int, n_odd: int) -> str:
    """Emit eval_pswf_phi(int abs_k_prime, ...) matching pswf_coeffs.hpp (0 local spills)."""
    n_half = (w + 1) // 2
    n_left = w // 2
    odd_base = 4 + n_half * n_even

    if prec in ("h", "f"):
        lines = [
            "float eval_pswf_phi(int abs_k_prime, float inv_nfft) {",
            "    float eta = (d_tbl[3] * inv_nfft) * float(abs_k_prime);",
            f"    float s = {0.5 * w:.1f}f * (1.0f - eta);",
            "    int j = int(s);",
            f"    if (j >= {n_half}) j = {n_half - 1};",
            "    if (j < 0) j = 0;",
            "    float u = 2.0f * (s - float(j)) - 1.0f;",
            "    float t = u * u;",
            f"    float mask[{n_half}];",
        ]
        for jj in range(n_half):
            lines.append(f"    mask[{jj}] = (j == {jj}) ? 1.0f : 0.0f;")
        lines.append("    float e_acc = 0.0f;")
        for m in range(n_even - 1, -1, -1):
            lines.append(f"    {{ float c_e = mask[0] * d_tbl[{4 + 0 * n_even + m}];")
            for jj in range(1, n_half):
                lines.append(f"      c_e = fma(mask[{jj}], d_tbl[{4 + jj * n_even + m}], c_e);")
            lines.append("      e_acc = c_e; }" if m == n_even - 1 else "      e_acc = fma(e_acc, t, c_e); }")
        lines.append("    float o_acc = 0.0f;")
        for m in range(n_odd - 1, -1, -1):
            lines.append(f"    {{ float c_o = mask[0] * d_tbl[{odd_base + 0 * n_odd + m}];")
            for jj in range(1, n_left):
                lines.append(f"      c_o = fma(mask[{jj}], d_tbl[{odd_base + jj * n_odd + m}], c_o);")
            lines.append("      o_acc = c_o; }" if m == n_odd - 1 else "      o_acc = fma(o_acc, t, c_o); }")
        lines.append("    return fma(u, o_acc, e_acc);")
        lines.append("}")
        return "\n".join(lines) + "\n"

    if prec == "d":
        lines = [
            "double eval_pswf_phi(int abs_k_prime, double inv_nfft) {",
            "    double eta = (d_tbl[3] * inv_nfft) * double(abs_k_prime);",
            f"    double s = {0.5 * w:.1f}lf * (1.0lf - eta);",
            "    int j = int(s);",
            f"    if (j >= {n_half}) j = {n_half - 1};",
            "    if (j < 0) j = 0;",
            "    double u = 2.0lf * (s - double(j)) - 1.0lf;",
            "    double t = u * u;",
            f"    double mask[{n_half}];",
        ]
        for jj in range(n_half):
            lines.append(f"    mask[{jj}] = (j == {jj}) ? 1.0lf : 0.0lf;")
        lines.append("    double e_acc = 0.0lf;")
        for m in range(n_even - 1, -1, -1):
            lines.append(f"    {{ double c_e = mask[0] * d_tbl[{4 + 0 * n_even + m}];")
            for jj in range(1, n_half):
                lines.append(f"      c_e = fma(mask[{jj}], d_tbl[{4 + jj * n_even + m}], c_e);")
            lines.append("      e_acc = c_e; }" if m == n_even - 1 else "      e_acc = fma(e_acc, t, c_e); }")
        lines.append("    double o_acc = 0.0lf;")
        for m in range(n_odd - 1, -1, -1):
            lines.append(f"    {{ double c_o = mask[0] * d_tbl[{odd_base + 0 * n_odd + m}];")
            for jj in range(1, n_left):
                lines.append(f"      c_o = fma(mask[{jj}], d_tbl[{odd_base + jj * n_odd + m}], c_o);")
            lines.append("      o_acc = c_o; }" if m == n_odd - 1 else "      o_acc = fma(o_acc, t, c_o); }")
        lines.append("    return fma(u, o_acc, e_acc);")
        lines.append("}")
        return "\n".join(lines) + "\n"

    # DF
    e_tail = 5 if n_even > 5 else n_even
    o_tail = 4 if n_odd > 5 else n_odd
    lines = [
        "df_t eval_pswf_phi(int abs_k_prime, float inv_nfft) {",
        "    df_t eta = df_scale_pow2(df_mul_f(d_tbl[3], float(abs_k_prime)), inv_nfft);",
        "    df_t one_minus_eta = df_sub(vec2(1.0f, 0.0f), eta);",
        f"    df_t s = df_scale_pow2(df_mul_f(one_minus_eta, {float(w):.1f}f), 0.5f);",
        "    int j = int(s.x);",
        "    if (float(j) > s.x || (float(j) == s.x && s.y < 0.0f)) --j;",
        f"    if (j >= {n_half}) j = {n_half - 1};",
        "    if (j < 0) j = 0;",
        "    df_t s_minus_j = df_sub(s, vec2(float(j), 0.0f));",
        "    df_t u = df_sub(df_scale_pow2(s_minus_j, 2.0f), vec2(1.0f, 0.0f));",
        "    df_t t = df_sqr(u);",
        "    float tf = t.x;",
        f"    float mask[{n_half}];",
    ]
    for jj in range(n_half):
        lines.append(f"    mask[{jj}] = (j == {jj}) ? 1.0f : 0.0f;")
    lines.append("    df_t e_acc = vec2(0.0f, 0.0f);")
    if e_tail < n_even:
        lines.append("    { float ef = 0.0f;")
        for m in range(n_even - 1, e_tail - 1, -1):
            lines.append(f"      {{ float ce_hi = mask[0] * d_tbl[{4 + 0 * n_even + m}].x;")
            for jj in range(1, n_half):
                lines.append(f"        ce_hi = fma(mask[{jj}], d_tbl[{4 + jj * n_even + m}].x, ce_hi);")
            lines.append("        ef = ce_hi; }" if m == n_even - 1 else "        ef = fma(ef, tf, ce_hi); }")
        lines.append("      e_acc = vec2(ef, 0.0f); }")
    for m in range(e_tail - 1, -1, -1):
        lines.append(f"    {{ float ce_hi = mask[0] * d_tbl[{4 + 0 * n_even + m}].x;")
        lines.append(f"      float ce_lo = mask[0] * d_tbl[{4 + 0 * n_even + m}].y;")
        for jj in range(1, n_half):
            lines.append(f"      ce_hi = fma(mask[{jj}], d_tbl[{4 + jj * n_even + m}].x, ce_hi);")
            lines.append(f"      ce_lo = fma(mask[{jj}], d_tbl[{4 + jj * n_even + m}].y, ce_lo);")
        lines.append("      df_t c_e = vec2(ce_hi, ce_lo);")
        lines.append("      e_acc = c_e; }" if m == n_even - 1 else "      e_acc = df_add(df_mul(e_acc, t), c_e); }")

    lines.append("    df_t o_acc = vec2(0.0f, 0.0f);")
    if o_tail < n_odd:
        lines.append("    { float of = 0.0f;")
        for m in range(n_odd - 1, o_tail - 1, -1):
            lines.append(f"      {{ float co_hi = mask[0] * d_tbl[{odd_base + 0 * n_odd + m}].x;")
            for jj in range(1, n_left):
                lines.append(f"        co_hi = fma(mask[{jj}], d_tbl[{odd_base + jj * n_odd + m}].x, co_hi);")
            lines.append("        of = co_hi; }" if m == n_odd - 1 else "        of = fma(of, tf, co_hi); }")
        lines.append("      o_acc = vec2(of, 0.0f); }")
    for m in range(o_tail - 1, -1, -1):
        lines.append(f"    {{ float co_hi = mask[0] * d_tbl[{odd_base + 0 * n_odd + m}].x;")
        lines.append(f"      float co_lo = mask[0] * d_tbl[{odd_base + 0 * n_odd + m}].y;")
        for jj in range(1, n_left):
            lines.append(f"      co_hi = fma(mask[{jj}], d_tbl[{odd_base + jj * n_odd + m}].x, co_hi);")
            lines.append(f"      co_lo = fma(mask[{jj}], d_tbl[{odd_base + jj * n_odd + m}].y, co_lo);")
        lines.append("      df_t c_o = vec2(co_hi, co_lo);")
        lines.append("      o_acc = c_o; }" if m == n_odd - 1 else "      o_acc = df_add(df_mul(o_acc, t), c_o); }")
    lines.append("    return df_add(e_acc, df_mul(u, o_acc));")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _gen_deconv_table(prec: str, w: int, n_even: int, n_odd: int, *, f32_out: bool = False) -> dict:
    if prec == "h":
        hdr = glsl_header("f" if f32_out else "h", fp16_storage=(not f32_out))
        tbl_t = "float"
        out_t = "float" if f32_out else "float16_t"
        scale_t = "float"
        check_nc = False
        eval_body = f"""
    float inv_nfft = 1.0f / float(pc.nfft);
    float phi = eval_pswf_phi(abs_kp, inv_nfft);
    float val_f = (pc.table_kind == 0) ? (d_tbl[1] * phi) : ((pc.output_scale * d_tbl[2]) / phi);
    d_out[i] = {out_t}(val_f);"""
    elif prec == "f":
        hdr = glsl_header("f")
        tbl_t, out_t, scale_t = "float", "float", "float"
        check_nc = False
        eval_body = """
    float inv_nfft = 1.0f / float(pc.nfft);
    float phi = eval_pswf_phi(abs_kp, inv_nfft);
    d_out[i] = (pc.table_kind == 0) ? (d_tbl[1] * phi) : ((pc.output_scale * d_tbl[2]) / phi);"""
    elif prec == "df":
        hdr = glsl_header("df") + glsl_df_math()
        tbl_t, out_t, scale_t = "vec2", "vec2", "vec2"
        check_nc = True
        eval_body = """
    float inv_nfft = 1.0f / float(pc.nfft);
    df_t phi = eval_pswf_phi(abs_kp, inv_nfft);
    if (pc.table_kind == 0) {
        d_out[i] = df_mul(d_tbl[1], phi);
    } else {
        df_t dec = df_div(d_tbl[2], phi);
        if (pc.output_scale.x != 1.0f || pc.output_scale.y != 0.0f) {
            dec = df_mul(dec, pc.output_scale);
        }
        d_out[i] = dec;
    }"""
    else:
        hdr = glsl_header("d")
        tbl_t, out_t, scale_t = "double", "double", "double"
        check_nc = False
        eval_body = """
    double inv_nfft = 1.0lf / double(pc.nfft);
    double phi = eval_pswf_phi(abs_kp, inv_nfft);
    d_out[i] = (pc.table_kind == 0) ? (d_tbl[1] * phi) : ((pc.output_scale * d_tbl[2]) / phi);"""

    src = (
        hdr
        + _glsl_compact_slot_helpers()
        + f"""
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer TblBuf {{ {tbl_t} d_tbl[]; }};
layout(std430, binding = 1) writeonly buffer OutBuf {{ {out_t} d_out[]; }};

layout(push_constant) uniform Push {{
    {scale_t} output_scale;
    int nfft;
    int log_nfft;
    int nout;
    int output_shift;
    int mode43;
    int table_kind;
}} pc;
{_emit_eval_pswf_phi_fn(prec, w, n_even, n_odd)}
void main() {{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.nout) return;
    int k_prime;
    if (pc.table_kind == 2) {{
        k_prime = i - pc.output_shift;
    }} else {{
        int slot_p = nufft1_compact_idx_to_slot(i, pc.mode43);
        k_prime = nufft1_slot_to_k_prime(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    }}
    int abs_kp = abs(k_prime);
{eval_body}
}}
"""
    )
    return {"source": src, "check_nocontract": check_nc}


def _gen_deconv_emit(prec: str, w: int, n_even: int, n_odd: int) -> dict:
    if prec == "h":
        hdr = glsl_header("h", fp16_storage=True) + glsl_f16x2_math()
        tbl_t, buf_t, scale_t = "float", "uint", "float"
        lanes = 2
        check_nc = False
        deconv_decl = """
    f16vec2 deconv_v = kOne_f16x2;
    if (!raw_mode) {
        float inv_nfft = 1.0f / float(pc.nfft);
        float phi = eval_pswf_phi(abs_kp, inv_nfft);
        float16_t ds = float16_t((pc.output_scale * d_tbl[2]) / phi);
        deconv_v = f16vec2(ds, ds);
    }"""
        loop_body = """
        uint b0 = 2u * bp;
        uint b1 = b0 + 1u;
        bool v0 = (b0 < pc.chunk_bs);
        bool v1 = (b1 < pc.chunk_bs);
        uint u0 = v0 ? d_fft_buf[b0 * uint(pc.nfft) + uint(slot_p)] : 0u;
        uint u1 = v1 ? d_fft_buf[b1 * uint(pc.nfft) + uint(slot_p)] : 0u;
        f16vec2 fr, fi;
        f16x2_load_pair_u32(u0, u1, fr, fi);
        if (!raw_mode) {
            fr *= deconv_v;
            fi *= deconv_v;
        }
        f16x2_pack_pair_u32(fr, fi, u0, u1);
        if (v0) d_out[pc.out_offset_elems + b0 * pc.total_nout_stride + uint(i)] = u0;
        if (v1) d_out[pc.out_offset_elems + b1 * pc.total_nout_stride + uint(i)] = u1;"""
    elif prec == "f":
        hdr = glsl_header("f")
        tbl_t, buf_t, scale_t = "float", "vec2", "float"
        lanes = 1
        check_nc = False
        deconv_decl = """
    float deconv_v = 1.0f;
    if (!raw_mode) {
        float inv_nfft = 1.0f / float(pc.nfft);
        float phi = eval_pswf_phi(abs_kp, inv_nfft);
        deconv_v = (pc.output_scale * d_tbl[2]) / phi;
    }"""
        loop_body = """
        vec2 fval = d_fft_buf[bp * uint(pc.nfft) + uint(slot_p)];
        d_out[pc.out_offset_elems + bp * pc.total_nout_stride + uint(i)] = raw_mode ? fval : (fval * deconv_v);"""
    elif prec == "df":
        hdr = glsl_header("df") + glsl_df_math()
        tbl_t, buf_t, scale_t = "vec2", "vec4", "vec2"
        lanes = 1
        check_nc = True
        deconv_decl = """
    df_t deconv_v = vec2(1.0f, 0.0f);
    if (!raw_mode) {
        float inv_nfft = 1.0f / float(pc.nfft);
        df_t phi = eval_pswf_phi(abs_kp, inv_nfft);
        deconv_v = df_div(d_tbl[2], phi);
        if (pc.output_scale.x != 1.0f || pc.output_scale.y != 0.0f) {
            deconv_v = df_mul(deconv_v, pc.output_scale);
        }
    }"""
        loop_body = """
        vec4 fval = d_fft_buf[bp * uint(pc.nfft) + uint(slot_p)];
        if (raw_mode) {
            d_out[pc.out_offset_elems + bp * pc.total_nout_stride + uint(i)] = fval;
        } else {
            df_t or_ = df_mul(fval.xy, deconv_v);
            df_t oi_ = df_mul(fval.zw, deconv_v);
            d_out[pc.out_offset_elems + bp * pc.total_nout_stride + uint(i)] = vec4(or_.x, or_.y, oi_.x, oi_.y);
        }"""
    else:
        hdr = glsl_header("d")
        tbl_t, buf_t, scale_t = "double", "dvec2", "double"
        lanes = 1
        check_nc = False
        deconv_decl = """
    double deconv_v = 1.0lf;
    if (!raw_mode) {
        double inv_nfft = 1.0lf / double(pc.nfft);
        double phi = eval_pswf_phi(abs_kp, inv_nfft);
        deconv_v = (pc.output_scale * d_tbl[2]) / phi;
    }"""
        loop_body = """
        dvec2 fval = d_fft_buf[bp * uint(pc.nfft) + uint(slot_p)];
        d_out[pc.out_offset_elems + bp * pc.total_nout_stride + uint(i)] = raw_mode ? fval : (fval * deconv_v);"""

    src = (
        hdr
        + _glsl_compact_slot_helpers()
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer TblBuf {{ {tbl_t} d_tbl[]; }};
layout(std430, binding = 1) readonly  buffer FftBuf {{ {buf_t} d_fft_buf[]; }};
layout(std430, binding = 2) writeonly buffer OutBuf {{ {buf_t} d_out[]; }};

layout(push_constant) uniform Push {{
    {scale_t} output_scale;
    int  nfft;
    int  log_nfft;
    int  nout;
    int  output_shift;
    int  mode43;
    int  order_mode;
    int  batch_tile;
    uint chunk_bs;
    uint total_nout_stride;
    uint out_offset_elems;
}} pc;
{_emit_eval_pswf_phi_fn(prec, w, n_even, n_odd)}
void main() {{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.nout) return;
    uint b_tile_idx = gl_WorkGroupID.y;
    bool raw_mode = (pc.order_mode == 2);

    int slot_p;
    int k_prime;
    if (pc.order_mode == 0 || raw_mode) {{
        slot_p = nufft1_compact_idx_to_slot(i, pc.mode43);
        k_prime = nufft1_slot_to_k_prime(slot_p, pc.log_nfft, pc.nfft, pc.output_shift);
    }} else {{
        k_prime = i - pc.output_shift;
        int bin = (k_prime >= 0) ? k_prime : (pc.nfft + k_prime);
        slot_p = int(bit_reverse(uint(bin), uint(pc.log_nfft)));
    }}
    int abs_kp = abs(k_prime);
{deconv_decl}

    uint bs_packed = (pc.chunk_bs + {lanes - 1}u) / {lanes}u;
    uint bp_start = b_tile_idx * uint(pc.batch_tile);
    uint bp_end   = min(bs_packed, bp_start + uint(pc.batch_tile));
    for (uint bp = bp_start; bp < bp_end; ++bp) {{
{loop_body}
    }}
}}
"""
    )
    return {"source": src, "check_nocontract": check_nc}


def _gen_reduce_grid(prec: str) -> dict:
    """Sum K_m inter-WG partial spreading grids [K_m][chunk_bs * nfft] -> d_fft_buf[chunk_bs * nfft]."""
    if prec == "h":
        hdr = glsl_header("f")
        buf_t, check_nc = "uint", False
        body = """
    vec2 acc = unpackHalf2x16(d_partials[idx]);
    for (uint k = 1u; k < pc.k_m_splits; ++k) {
        acc += unpackHalf2x16(d_partials[k * pc.total_elems + idx]);
    }
    d_out[idx] = packHalf2x16(acc);"""
    elif prec == "f":
        hdr = glsl_header("f")
        buf_t, check_nc = "vec2", False
        body = """
    vec2 acc = d_partials[idx];
    for (uint k = 1u; k < pc.k_m_splits; ++k) {
        acc += d_partials[k * pc.total_elems + idx];
    }
    d_out[idx] = acc;"""
    elif prec == "df":
        hdr = glsl_header("df") + glsl_df_math()
        buf_t, check_nc = "vec4", True
        body = """
    vec4 first = d_partials[idx];
    df_t ar = first.xy, ai = first.zw;
    for (uint k = 1u; k < pc.k_m_splits; ++k) {
        vec4 v = d_partials[k * pc.total_elems + idx];
        ar = df_add(ar, v.xy);
        ai = df_add(ai, v.zw);
    }
    d_out[idx] = vec4(ar.x, ar.y, ai.x, ai.y);"""
    else:
        hdr = glsl_header("d")
        buf_t, check_nc = "dvec2", False
        body = """
    dvec2 acc = d_partials[idx];
    for (uint k = 1u; k < pc.k_m_splits; ++k) {
        acc += d_partials[k * pc.total_elems + idx];
    }
    d_out[idx] = acc;"""

    src = (
        hdr
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly  buffer PartBuf {{ {buf_t} d_partials[]; }};
layout(std430, binding = 1) writeonly buffer OutBuf  {{ {buf_t} d_out[]; }};

layout(push_constant) uniform Push {{
    uint total_elems;
    uint k_m_splits;
}} pc;

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.total_elems) return;
{body}
}}
"""
    )
    return {"source": src, "check_nocontract": check_nc}


def generate_nufft_shaders() -> dict[str, dict]:
    shaders: dict[str, dict] = {}

    for prec in ("h", "f", "df", "d"):
        shaders[f"nufft_pack_bucket_geom_{prec}"] = _gen_pack_bucket_geom(prec)
        shaders[f"nufft_spread_smem_{prec}"] = _gen_spread_smem(prec)
        shaders[f"nufft_spread_cpu_direct_{prec}"] = _gen_spread_cpu_direct(prec)
        shaders[f"nufft_reduce_grid_{prec}"] = _gen_reduce_grid(prec)
        for q_val in REG_Q_BY_PREC[prec]:
            shaders[f"nufft_spread_reg_{prec}_q{q_val}"] = _gen_spread_reg(prec, q_val)

    shaders.update(_gen_bucket_shaders())

    for prec, w, n_even, n_odd in PSWF_WIDTH_SPECS:
        shaders[f"nufft_precompute_points_weights_{prec}_w{w}"] = _gen_precompute_points_weights(prec, w, n_even, n_odd)
        shaders[f"nufft_deconv_table_{prec}_w{w}"] = _gen_deconv_table(prec, w, n_even, n_odd, f32_out=False)
        shaders[f"nufft_deconv_emit_{prec}_w{w}"] = _gen_deconv_emit(prec, w, n_even, n_odd)
        if prec == "h":
            shaders[f"nufft_deconv_table_h_f32_w{w}"] = _gen_deconv_table("h", w, n_even, n_odd, f32_out=True)

    return shaders
