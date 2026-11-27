"""Hardware probe compute shaders for Vulkan device validation.

Generates:
  - probe_subgroup : checks linear subgroup lane mapping & subgroupShuffleXor
  - probe_df_kat   : Known-Answer Test for Dekker TwoSum/TwoDiff/TwoProd/df_mul/sincos2pidf
  - probe_fp16_kat : Known-Answer Test for native f16vec2 FMA & u32 pack/unpack
  - probe_fp64_ratio_f32 / probe_fp64_ratio_f64 : lazy FP64/FP32 throughput ratio probe
"""

from __future__ import annotations

from common_glsl import (
    glsl_df_math,
    glsl_f16x2_math,
    glsl_header,
    glsl_sincos2pi_df,
)


def generate_probe_shaders() -> dict[str, dict]:
    """Return mapping shader_name -> {source, target_env, check_nocontract}."""
    shaders: dict[str, dict] = {}

    # 1. Subgroup topology & XOR shuffle probe
    shaders["probe_subgroup"] = {
        "target_env": "vulkan1.1",
        "check_nocontract": False,
        "source": (
            glsl_header(None, subgroups=True, subgroup_shuffle=True)
            + """
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer OutBuf { uint out_data[]; };

layout(push_constant) uniform PC {
    uint probe_wg;
} pc;

void main() {
    uint t = gl_LocalInvocationIndex;
    uint sg_sz = gl_SubgroupSize;
    if (t == 0u) {
        out_data[0] = sg_sz;
    }
    bool pow2 = (sg_sz > 0u) && ((sg_sz & (sg_sz - 1u)) == 0u);
    if (!pow2 || gl_SubgroupInvocationID != (t & (sg_sz - 1u))) {
        out_data[1] = 0u;
    }
    uint max_h = min(sg_sz, pc.probe_wg);
    if (max_h >= 2u  && subgroupShuffleXor(t, 1u)  != (t ^ 1u))  out_data[2] = 0u;
    if (max_h >= 4u  && subgroupShuffleXor(t, 2u)  != (t ^ 2u))  out_data[2] = 0u;
    if (max_h >= 8u  && subgroupShuffleXor(t, 4u)  != (t ^ 4u))  out_data[2] = 0u;
    if (max_h >= 16u && subgroupShuffleXor(t, 8u)  != (t ^ 8u))  out_data[2] = 0u;
    if (max_h >= 32u && subgroupShuffleXor(t, 16u) != (t ^ 16u)) out_data[2] = 0u;
}
"""
        ),
    }

    # 2. Dekker Double-Float Known-Answer Test (verifies NoContraction end-to-end on vendor driver)
    shaders["probe_df_kat"] = {
        "target_env": "vulkan1.1",
        "check_nocontract": True,
        "source": (
            glsl_header("df")
            + glsl_df_math()
            + glsl_sincos2pi_df()
            + """
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float in_data[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf { float out_data[]; };

void main() {
    float r, e;
    // 0: TwoSum(1.0f, 3*2^-24)
    df_two_sum(in_data[0], in_data[1], r, e);
    out_data[0] = r; out_data[1] = e;
    // 1: TwoDiff(1.5f, 3*2^-24)
    df_two_difference(in_data[2], in_data[3], r, e);
    out_data[2] = r; out_data[3] = e;
    // 2: TwoProd(1 + 2^-12, 1 + 2^-13)
    df_two_product(in_data[4], in_data[5], r, e);
    out_data[4] = r; out_data[5] = e;
    // 3: df_mul + df_neg guard (CUDA sm_120 sign-fold regression test)
    df_t a = vec2(in_data[4], in_data[6]);
    df_t b = df_neg(vec2(in_data[5], in_data[7]));
    df_t prod = df_mul(a, b);
    out_data[6] = prod.x; out_data[7] = prod.y;
    // 4: sincos2pidf_branchless(1/7)
    df_t c, s;
    sincos2pidf_branchless(vec2(in_data[8], in_data[9]), false, c, s);
    out_data[8]  = c.x; out_data[9]  = c.y;
    out_data[10] = s.x; out_data[11] = s.y;
}
"""
        ),
    }

    # 3. Native FP16 Known-Answer Test
    shaders["probe_fp16_kat"] = {
        "target_env": "vulkan1.2",
        "check_nocontract": False,
        "source": (
            glsl_header("h", fp16_storage=True)
            + glsl_f16x2_math()
            + """
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float16_t in_h[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf { float out_f[]; };

void main() {
    f16vec2 a = f16vec2(in_h[0], in_h[1]);
    f16vec2 b = f16vec2(in_h[2], in_h[3]);
    f16vec2 c = f16vec2(in_h[4], in_h[5]);
    f16vec2 r = fma(a, b, c);
    uint packed = f16x2_to_u32(r);
    f16vec2 unp = u32_to_f16x2(packed);
    out_f[0] = float(unp.x);
    out_f[1] = float(unp.y);
}
"""
        ),
    }

    # 4. FP32 vs FP64 FMA throughput microbenchmark shaders
    shaders["probe_fp64_ratio_f32"] = {
        "target_env": "vulkan1.1",
        "check_nocontract": False,
        "source": (
            glsl_header("f")
            + """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) writeonly buffer OutBuf { float out_f[]; };
layout(push_constant) uniform PC { uint iters; uint total; } pc;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    float a = 1.0001f + float(gid & 255u) * 1e-4f;
    float b = 0.9999f;
    float x = 0.5f, y = -0.25f, z = 0.75f, w = -0.5f;
    for (uint i = 0u; i < pc.iters; ++i) {
        x = fma(x, a, b); y = fma(y, b, a);
        z = fma(z, a, x); w = fma(w, b, y);
        x = fma(x, b, -z); y = fma(y, a, -w);
        z = fma(z, b, x); w = fma(w, a, y);
    }
    out_f[gid] = x + y + z + w;
}
"""
        ),
    }

    shaders["probe_fp64_ratio_f64"] = {
        "target_env": "vulkan1.1",
        "check_nocontract": False,
        "source": (
            glsl_header("d")
            + """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) writeonly buffer OutBuf { double out_d[]; };
layout(push_constant) uniform PC { uint iters; uint total; } pc;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    double a = 1.0001lf + double(gid & 255u) * 1e-4lf;
    double b = 0.9999lf;
    double x = 0.5lf, y = -0.25lf, z = 0.75lf, w = -0.5lf;
    for (uint i = 0u; i < pc.iters; ++i) {
        x = fma(x, a, b); y = fma(y, b, a);
        z = fma(z, a, x); w = fma(w, b, y);
        x = fma(x, b, -z); y = fma(y, a, -w);
        z = fma(z, b, x); w = fma(w, a, y);
    }
    out_d[gid] = x + y + z + w;
}
"""
        ),
    }

    return shaders
