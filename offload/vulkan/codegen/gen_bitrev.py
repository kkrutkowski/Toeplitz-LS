"""GLSL compute shader codegen for the Multi-Strategy Bit-Reversal Engine.

Deduplicated by raw element size:
  - 4b  : uint  (Complexh = 4B)
  - 8b  : uvec2 (Complexf = 8B)
  - 16b : uvec4 (Complex / Complexdf = 16B)

All LUT buffers pack 16-bit entries into uint32 words when logN <= 16 so no
16-bit storage extension is required by any bit-reversal shader (Vulkan 1.1
universal).

Strategies (mirroring offload/sycl/bitrev/bitrev.cpp):
  1. bitrev_naive_{4,8,16}b      (spec: WG_SIZE, IN_PLACE)
  2. bitrev_lut_pairs_{4,8,16}b  (spec: WG_SIZE, PACKED_U16)
  3. bitrev_lut_idx_{4,8,16}b    (spec: WG_SIZE, PACKED_U16)
  4. bitrev_whole_smem_{4,8,16}b (spec: WG_SIZE, SMEM_ELEMS)
  5. bitrev_cobra_w4_{4,8,16}b   (spec: WG_SIZE, IN_PLACE, USE_LUT, SMEM_ELEMS)
  6. bitrev_cobra_w5_{4,8,16}b   (spec: WG_SIZE, IN_PLACE, USE_LUT, SMEM_ELEMS)
  7. bitrev_scalar_cobra_{4,8,16}b (spec: WG_SIZE, IN_PLACE)
"""

from __future__ import annotations

from common_glsl import glsl_bit_reverse, glsl_header

RAW_TYPES = (
    ("4b", "uint"),
    ("8b", "uvec2"),
    ("16b", "uvec4"),
)


def _gen_naive(tag: str, raw_t: str) -> str:
    return (
        glsl_header(None)
        + glsl_bit_reverse()
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint IN_PLACE = 1u;

layout(set = 0, binding = 0) readonly buffer SrcBuf {{ {raw_t} src_buf[]; }};
layout(set = 0, binding = 1) buffer DstBuf          {{ {raw_t} dst_buf[]; }};

layout(push_constant) uniform PC {{
    uint total;
    uint logN;
    uint batch_offset;
}} pc;

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    uint mask = (1u << pc.logN) - 1u;
    uint batch_off = (gid & ~mask) + pc.batch_offset;
    uint idx = gid & mask;
    uint rev = bit_reverse(idx, pc.logN);
    if (IN_PLACE != 0u) {{
        if (idx < rev) {{
            {raw_t} a = dst_buf[batch_off + idx];
            {raw_t} b = dst_buf[batch_off + rev];
            dst_buf[batch_off + idx] = b;
            dst_buf[batch_off + rev] = a;
        }}
    }} else {{
        dst_buf[batch_off + idx] = src_buf[batch_off + rev];
    }}
}}
"""
    )


def _gen_lut_pairs(tag: str, raw_t: str) -> str:
    return (
        glsl_header(None)
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint PACKED_U16 = 1u;

layout(set = 0, binding = 0) buffer DstBuf          {{ {raw_t} dst_buf[]; }};
layout(set = 0, binding = 1) readonly buffer LutBuf {{ uint pairs[]; }};

layout(push_constant) uniform PC {{
    uint npairs;
    uint total_pairs;
    uint logN;
    uint batch_offset;
}} pc;

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.total_pairs) return;
    uint batch = gid / pc.npairs;
    uint p = gid - batch * pc.npairs;
    uint base = (batch << pc.logN) + pc.batch_offset;
    uint i, j;
    if (PACKED_U16 != 0u) {{
        uint packed = pairs[p];
        i = packed & 0xFFFFu;
        j = packed >> 16;
    }} else {{
        i = pairs[2u * p];
        j = pairs[2u * p + 1u];
    }}
    {raw_t} a = dst_buf[base + i];
    {raw_t} b = dst_buf[base + j];
    dst_buf[base + i] = b;
    dst_buf[base + j] = a;
}}
"""
    )


def _gen_lut_idx(tag: str, raw_t: str) -> str:
    return (
        glsl_header(None)
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint PACKED_U16 = 1u;

layout(set = 0, binding = 0) readonly buffer SrcBuf  {{ {raw_t} src_buf[]; }};
layout(set = 0, binding = 1) writeonly buffer DstBuf {{ {raw_t} dst_buf[]; }};
layout(set = 0, binding = 2) readonly buffer LutBuf  {{ uint rev_idx[]; }};

layout(push_constant) uniform PC {{
    uint total;
    uint logN;
    uint batch_offset;
}} pc;

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    uint mask = (1u << pc.logN) - 1u;
    uint base = (gid & ~mask) + pc.batch_offset;
    uint i = gid & mask;
    uint j;
    if (PACKED_U16 != 0u) {{
        uint word = rev_idx[i >> 1];
        j = ((i & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
    }} else {{
        j = rev_idx[i];
    }}
    dst_buf[base + i] = src_buf[base + j];
}}
"""
    )


def _gen_whole_smem(tag: str, raw_t: str) -> str:
    return (
        glsl_header(None)
        + glsl_bit_reverse()
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint SMEM_ELEMS = 1024u;
layout(constant_id = 2) const uint IN_PLACE = 1u;

shared {raw_t} smem[SMEM_ELEMS];

layout(set = 0, binding = 0) readonly buffer SrcBuf {{ {raw_t} src_buf[]; }};
layout(set = 0, binding = 1) buffer DstBuf          {{ {raw_t} dst_buf[]; }};

layout(push_constant) uniform PC {{
    uint logN;
    uint xor_k;
    uint batch_offset;
}} pc;

void main() {{
    uint batch = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint wg = gl_WorkGroupSize.x;
    uint n = 1u << pc.logN;
    uint base = (batch << pc.logN) + pc.batch_offset;
    uint shift = pc.logN - pc.xor_k;
    uint mask = (pc.xor_k > 0u) ? ((1u << pc.xor_k) - 1u) : 0u;

    for (uint idx = tid; idx < n; idx += wg) {{
        uint slot = idx ^ ((idx >> shift) & mask);
        smem[slot] = (IN_PLACE != 0u) ? dst_buf[base + idx] : src_buf[base + idx];
    }}
    barrier();

    for (uint idx = tid; idx < n; idx += wg) {{
        uint r = bit_reverse(idx, pc.logN);
        uint slot = r ^ ((r >> shift) & mask);
        dst_buf[base + idx] = smem[slot];
    }}
}}
"""
    )


def _gen_cobra_smem(tag: str, raw_t: str, wbits: int) -> str:
    w = 1 << wbits
    mask = w - 1
    tile_elems = w * w
    if wbits == 4:
        xor_expr = "(a << 4u) + (c ^ a ^ (a >> 3u) ^ ((a >> 1u) & 2u))"
    else:
        xor_expr = "(a << 5u) + (c ^ a ^ (a >> 4u) ^ ((a >> 2u) & 2u))"

    return (
        glsl_header(None)
        + glsl_bit_reverse()
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint IN_PLACE = 1u;
layout(constant_id = 2) const uint USE_LUT  = 0u;
layout(constant_id = 3) const uint SMEM_ELEMS = {tile_elems * 2}u;

const uint kWBits = {wbits}u;
const uint kMask  = {mask}u;
const uint kTileElems = {tile_elems}u;

shared {raw_t} smem[SMEM_ELEMS];

layout(set = 0, binding = 0) readonly buffer SrcBuf  {{ {raw_t} src_buf[]; }};
layout(set = 0, binding = 1) buffer DstBuf           {{ {raw_t} dst_buf[]; }};
layout(set = 0, binding = 2) readonly buffer RevWBuf {{ uint rev_w_packed[]; }};
layout(set = 0, binding = 3) readonly buffer BLutBuf {{ uint b_lut_packed[]; }};

layout(push_constant) uniform PC {{
    uint work_per_batch;
    uint logN;
    uint batch_offset;
}} pc;

uint cobra_xor_slot(uint a, uint c) {{
    return {xor_expr};
}}

uint read_u16_rev_w(uint idx) {{
    uint word = rev_w_packed[idx >> 1];
    return ((idx & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
}}

void main() {{
    uint wg_id = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint wg = gl_WorkGroupSize.x;
    uint batch = wg_id / pc.work_per_batch;
    uint b_idx = wg_id - batch * pc.work_per_batch;
    uint m = pc.logN - 2u * kWBits;
    uint a_shift = m + kWBits;

    uint b = 0u;
    uint b_rev = 0u;
    if (USE_LUT != 0u) {{
        if (IN_PLACE != 0u) {{
            uint packed = b_lut_packed[b_idx];
            b = packed & 0xFFFFu;
            b_rev = packed >> 16;
        }} else {{
            b = b_idx;
            uint word = b_lut_packed[b_idx >> 1];
            b_rev = ((b_idx & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
        }}
    }} else {{
        b = b_idx;
        b_rev = (m == 0u) ? 0u : bit_reverse(b, m);
        if (IN_PLACE != 0u && b > b_rev) return;
    }}

    uint base = (batch << pc.logN) + pc.batch_offset;
    uint off_b = base + (b << kWBits);
    uint off_brev = base + (b_rev << kWBits);

    for (uint lin = tid; lin < kTileElems; lin += wg) {{
        uint a = lin >> kWBits;
        uint c = lin & kMask;
        uint gidx = off_b + (a << a_shift) + c;
        smem[cobra_xor_slot(a, c)] = (IN_PLACE != 0u) ? dst_buf[gidx] : src_buf[gidx];
    }}

    if (IN_PLACE != 0u && b != b_rev) {{
        for (uint lin = tid; lin < kTileElems; lin += wg) {{
            uint a = lin >> kWBits;
            uint c = lin & kMask;
            uint gidx = off_brev + (a << a_shift) + c;
            smem[kTileElems + cobra_xor_slot(a, c)] = dst_buf[gidx];
        }}
    }}

    barrier();

    for (uint lin = tid; lin < kTileElems; lin += wg) {{
        uint a_out = lin >> kWBits;
        uint c_out = lin & kMask;
        uint a_src = (USE_LUT != 0u) ? read_u16_rev_w(c_out) : bit_reverse(c_out, kWBits);
        uint c_src = (USE_LUT != 0u) ? read_u16_rev_w(a_out) : bit_reverse(a_out, kWBits);
        dst_buf[off_brev + (a_out << a_shift) + c_out] = smem[cobra_xor_slot(a_src, c_src)];
    }}

    if (IN_PLACE != 0u && b != b_rev) {{
        for (uint lin = tid; lin < kTileElems; lin += wg) {{
            uint a_out = lin >> kWBits;
            uint c_out = lin & kMask;
            uint a_src = (USE_LUT != 0u) ? read_u16_rev_w(c_out) : bit_reverse(c_out, kWBits);
            uint c_src = (USE_LUT != 0u) ? read_u16_rev_w(a_out) : bit_reverse(a_out, kWBits);
            dst_buf[off_b + (a_out << a_shift) + c_out] =
                smem[kTileElems + cobra_xor_slot(a_src, c_src)];
        }}
    }}
}}
"""
    )


def _gen_scalar_cobra(tag: str, raw_t: str) -> str:
    return (
        glsl_header(None)
        + f"""
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint IN_PLACE = 1u;

const uint kWBits = 4u;
const uint kW = 16u;

layout(set = 0, binding = 0) readonly buffer SrcBuf  {{ {raw_t} src_buf[]; }};
layout(set = 0, binding = 1) buffer DstBuf           {{ {raw_t} dst_buf[]; }};
layout(set = 0, binding = 2) readonly buffer RevWBuf {{ uint rev_w_packed[]; }};
layout(set = 0, binding = 3) readonly buffer BLutBuf {{ uint b_lut_packed[]; }};

layout(push_constant) uniform PC {{
    uint work_per_batch;
    uint total_work;
    uint logN;
    uint batch_offset;
}} pc;

uint read_u16_rev_w(uint idx) {{
    uint word = rev_w_packed[idx >> 1];
    return ((idx & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
}}

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.total_work) return;

    uint batch = gid / pc.work_per_batch;
    uint b_idx = gid - batch * pc.work_per_batch;
    uint m = pc.logN - 8u;
    uint a_shift = m + 4u;

    uint b, b_rev;
    if (IN_PLACE != 0u) {{
        uint packed = b_lut_packed[b_idx];
        b = packed & 0xFFFFu;
        b_rev = packed >> 16;
    }} else {{
        b = b_idx;
        uint word = b_lut_packed[b_idx >> 1];
        b_rev = ((b_idx & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
    }}

    uint base = (batch << pc.logN) + pc.batch_offset;
    uint off_b = base + (b << 4u);
    uint off_brev = base + (b_rev << 4u);

    {raw_t} tileA[256];
    for (uint a = 0u; a < kW; ++a) {{
        uint row_off = off_b + (a << a_shift);
        for (uint c = 0u; c < kW; ++c) {{
            tileA[a * kW + c] = (IN_PLACE != 0u) ? dst_buf[row_off + c] : src_buf[row_off + c];
        }}
    }}

    if (IN_PLACE != 0u && b != b_rev) {{
        {raw_t} tileB[256];
        for (uint a = 0u; a < kW; ++a) {{
            uint row_off = off_brev + (a << a_shift);
            for (uint c = 0u; c < kW; ++c) {{
                tileB[a * kW + c] = dst_buf[row_off + c];
            }}
        }}
        for (uint a_out = 0u; a_out < kW; ++a_out) {{
            uint c_src = read_u16_rev_w(a_out);
            uint row_rev = off_brev + (a_out << a_shift);
            uint row_b   = off_b    + (a_out << a_shift);
            for (uint c_out = 0u; c_out < kW; ++c_out) {{
                uint a_src = read_u16_rev_w(c_out);
                dst_buf[row_rev + c_out] = tileA[a_src * kW + c_src];
                dst_buf[row_b   + c_out] = tileB[a_src * kW + c_src];
            }}
        }}
        return;
    }}

    for (uint a_out = 0u; a_out < kW; ++a_out) {{
        uint c_src = read_u16_rev_w(a_out);
        uint row_rev = off_brev + (a_out << a_shift);
        for (uint c_out = 0u; c_out < kW; ++c_out) {{
            uint a_src = read_u16_rev_w(c_out);
            dst_buf[row_rev + c_out] = tileA[a_src * kW + c_src];
        }}
    }}
}}
"""
    )


def generate_bitrev_shaders() -> dict[str, dict]:
    """Generate all bit-reversal compute shaders deduplicated by raw byte width."""
    shaders: dict[str, dict] = {}
    for tag, raw_t in RAW_TYPES:
        shaders[f"bitrev_naive_{tag}"] = {
            "target_env": "vulkan1.1",
            "check_nocontract": False,
            "source": _gen_naive(tag, raw_t),
        }
        shaders[f"bitrev_lut_pairs_{tag}"] = {
            "target_env": "vulkan1.1",
            "check_nocontract": False,
            "source": _gen_lut_pairs(tag, raw_t),
        }
        shaders[f"bitrev_lut_idx_{tag}"] = {
            "target_env": "vulkan1.1",
            "check_nocontract": False,
            "source": _gen_lut_idx(tag, raw_t),
        }
        shaders[f"bitrev_whole_smem_{tag}"] = {
            "target_env": "vulkan1.1",
            "check_nocontract": False,
            "source": _gen_whole_smem(tag, raw_t),
        }
        shaders[f"bitrev_cobra_w4_{tag}"] = {
            "target_env": "vulkan1.1",
            "check_nocontract": False,
            "source": _gen_cobra_smem(tag, raw_t, 4),
        }
        shaders[f"bitrev_cobra_w5_{tag}"] = {
            "target_env": "vulkan1.1",
            "check_nocontract": False,
            "source": _gen_cobra_smem(tag, raw_t, 5),
        }
        shaders[f"bitrev_scalar_cobra_{tag}"] = {
            "target_env": "vulkan1.1",
            "check_nocontract": False,
            "source": _gen_scalar_cobra(tag, raw_t),
        }
    return shaders
