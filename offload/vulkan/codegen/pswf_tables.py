#!/usr/bin/env python3
"""Parse exact PSWF piecewise minimax tables from offload/sycl/nufft/pswf_coeffs.hpp
and generate plain C23 offload/vulkan/nufft/pswf_coeffs.h (single source of truth).
"""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path

SYCL_PSWF_HPP = Path(__file__).resolve().parents[2] / "sycl" / "nufft" / "pswf_coeffs.hpp"

# Metadata for every supported (prec_key, w) table: (n_even, n_odd)
TABLE_SPECS = [
    ("fp16", 4, 4, 3),
    ("fp16", 5, 4, 3),
    ("fp32", 8, 5, 4),
    ("fp32", 9, 5, 4),
    ("df",  14, 7, 6),
    ("df",  15, 7, 6),
    ("df",  16, 7, 6),
    ("fp64", 16, 8, 7),
    ("fp64", 17, 8, 7),
    ("fp64", 18, 8, 7),
]


def _parse_numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?", text)


def _extract_array_block(src: str, array_name: str) -> str:
    idx = src.find(array_name)
    if idx < 0:
        raise RuntimeError(f"Could not find {array_name} in {SYCL_PSWF_HPP}")
    start = src.find("{", idx)
    depth = 0
    for pos in range(start, len(src)):
        ch = src[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start : pos + 1]
    raise RuntimeError(f"Unclosed brace block for {array_name}")


def _split_top_level_braces(block: str) -> list[str]:
    """Given '{ a, b }', return ['{...}', '{...}'] top-level brace children."""
    assert block[0] == "{" and block[-1] == "}"
    out: list[str] = []
    depth = 0
    start = -1
    for i in range(1, len(block) - 1):
        ch = block[i]
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(block[start : i + 1])
    return out


def parse_all_tables() -> dict[tuple[str, int], list[dict]]:
    """Return dict[(prec_key, w)] -> [mode21_dict, mode43_dict]."""
    src = SYCL_PSWF_HPP.read_text(encoding="utf-8")
    tables: dict[tuple[str, int], list[dict]] = {}

    for prec_key, w, n_even, n_odd in TABLE_SPECS:
        arr_name = f"kPswfTable_{prec_key}_w{w}"
        block = _extract_array_block(src, arr_name)
        modes = _split_top_level_braces(block)
        assert len(modes) == 2, f"{arr_name}: expected 2 modes, got {len(modes)}"
        n_half = (w + 1) // 2
        parsed_modes = []

        for m_blk in modes:
            # Strip comments inside mode block
            clean = re.sub(r"//[^\n]*", "", m_blk)
            nums = _parse_numbers(clean)
            if prec_key == "df":
                # 1 scale_exp + 4*(hi,lo) + n_half*n_even*(hi,lo) + n_half*n_odd*(hi,lo)
                expected = 1 + 2 * (4 + n_half * (n_even + n_odd))
                assert len(nums) == expected, f"{arr_name}: got {len(nums)} nums, expected {expected}"
                pos = 0
                scale_exp = int(nums[pos]); pos += 1
                def take_df() -> tuple[float, float]:
                    nonlocal pos
                    hi = struct.unpack("<f", struct.pack("<f", float(nums[pos].rstrip("fF"))))[0]
                    lo = struct.unpack("<f", struct.pack("<f", float(nums[pos + 1].rstrip("fF"))))[0]
                    pos += 2
                    return (hi, lo)
                c = take_df()
                prefac = take_df()
                inv_prefac = take_df()
                eta_scale = take_df()
                even = [[take_df() for _ in range(n_even)] for _ in range(n_half)]
                odd  = [[take_df() for _ in range(n_odd)]  for _ in range(n_half)]
            else:
                expected = 1 + 4 + n_half * (n_even + n_odd)
                assert len(nums) == expected, f"{arr_name}: got {len(nums)} nums, expected {expected}"
                pos = 0
                scale_exp = int(nums[pos]); pos += 1
                def take_f() -> float:
                    nonlocal pos
                    raw = nums[pos]
                    v = float(raw.rstrip("fF"))
                    if raw.endswith(("f", "F")):
                        v = struct.unpack("<f", struct.pack("<f", v))[0]
                    pos += 1
                    return v
                c = take_f()
                prefac = take_f()
                inv_prefac = take_f()
                eta_scale = take_f()
                even = [[take_f() for _ in range(n_even)] for _ in range(n_half)]
                odd  = [[take_f() for _ in range(n_odd)]  for _ in range(n_half)]

            parsed_modes.append({
                "scale_exp": scale_exp,
                "c": c,
                "prefac": prefac,
                "inv_prefac": inv_prefac,
                "eta_scale": eta_scale,
                "even": even,
                "odd": odd,
            })
        tables[(prec_key, w)] = parsed_modes
    return tables


def _val_to_float(v) -> float:
    return (v[0] + v[1]) if isinstance(v, tuple) else float(v)


def _eval_psi(tbl: dict, w: int, n_even: int, n_odd: int, eta: float) -> float:
    n_half = (w + 1) // 2
    n_left = w // 2
    s = 0.5 * w * (1.0 - eta)
    j = int(math.floor(s))
    if j >= n_half:
        j = n_half - 1
    if j < 0:
        j = 0
    u = 2.0 * (s - j) - 1.0
    t = u * u
    e_acc = _val_to_float(tbl["even"][j][n_even - 1])
    for m in range(n_even - 2, -1, -1):
        e_acc = e_acc * t + _val_to_float(tbl["even"][j][m])
    o_acc = 0.0
    if j < n_left:
        o_acc = _val_to_float(tbl["odd"][j][n_odd - 1])
        for m in range(n_odd - 2, -1, -1):
            o_acc = o_acc * t + _val_to_float(tbl["odd"][j][m])
    return e_acc + u * o_acc


def _flatten_table(prec_key: str, w: int, n_even: int, n_odd: int, tbl: dict, mode_idx: int, geo_scale: bool):
    """Return flat list of scalar values: [c, prefac, inv_prefac, eta_scale, even..., odd...]."""
    r = 1.0
    if geo_scale:
        edge_frac = 0.375 if mode_idx == 1 else 0.25
        eta_max = _val_to_float(tbl["eta_scale"]) * edge_frac
        prefac_d = _val_to_float(tbl["prefac"])
        phi0 = prefac_d * _eval_psi(tbl, w, n_even, n_odd, 0.0)
        phi_edge = prefac_d * _eval_psi(tbl, w, n_even, n_odd, eta_max)
        r = 1.0 / math.sqrt(phi0 * phi_edge)

    n_half = (w + 1) // 2
    if prec_key == "df":
        import struct
        def to_df(val: float) -> tuple[float, float]:
            hi = struct.unpack("<f", struct.pack("<f", val))[0]
            lo = struct.unpack("<f", struct.pack("<f", val - hi))[0]
            return (hi, lo)

        flat = [tbl["c"], tbl["prefac"], tbl["inv_prefac"], tbl["eta_scale"]]
        for j in range(n_half):
            for m in range(n_even):
                flat.append(to_df(_val_to_float(tbl["even"][j][m]) * r))
        for j in range(n_half):
            for m in range(n_odd):
                flat.append(to_df(_val_to_float(tbl["odd"][j][m]) * r))
        return flat
    else:
        flat = [tbl["c"], tbl["prefac"], tbl["inv_prefac"], tbl["eta_scale"]]
        for j in range(n_half):
            for m in range(n_even):
                flat.append(tbl["even"][j][m] * r)
        for j in range(n_half):
            for m in range(n_odd):
                flat.append(tbl["odd"][j][m] * r)
        return flat


def generate_c23_header(out_path: Path) -> None:
    """Emit plain C23 pswf_coeffs.h with flat SSBO-ready tables (raw & geo-scaled)."""
    tables = parse_all_tables()
    lines = [
        "#ifndef TLS_VK_PSWF_COEFFS_H",
        "#define TLS_VK_PSWF_COEFFS_H",
        "/* AUTO-GENERATED by offload/vulkan/codegen/pswf_tables.py from",
        " * offload/sycl/nufft/pswf_coeffs.hpp (single source of truth). DO NOT EDIT. */",
        "",
        "#include \"common/math.h\"",
        "#include \"common/vk_device.h\"",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "typedef enum VkPswfMode {",
        "    VK_PSWF21 = 0, /* sigma = 2.0, N_out = N,    output_shift = N/2,  N_fft = 2N */",
        "    VK_PSWF43 = 1  /* sigma = 4/3, N_out = 3N/2, output_shift = 3N/4, N_fft = 2N */",
        "} VkPswfMode;",
        "",
    ]

    def fmt_c_f32(v: float) -> str:
        return f"{v:.9e}f"

    def fmt_c_f64(v: float) -> str:
        return f"{v:.17e}"

    for prec_key, w, n_even, n_odd in TABLE_SPECS:
        n_half = (w + 1) // 2
        count = 4 + n_half * (n_even + n_odd)
        modes = tables[(prec_key, w)]

        if prec_key == "fp16":
            # Emit both raw 2^e (default) and geo-scaled (FP16 IHS)
            for suffix, geo in (("", False), ("_geo", True)):
                lines.append(f"static constexpr float kVkPswfFlat_{prec_key}_w{w}{suffix}[2][{count}] = {{")
                for m_idx in range(2):
                    flat = _flatten_table(prec_key, w, n_even, n_odd, modes[m_idx], m_idx, geo)
                    vals = ", ".join(fmt_c_f32(v) for v in flat)
                    lines.append(f"    {{{vals}}},")
                lines.append("};")
        elif prec_key == "fp32":
            lines.append(f"static constexpr float kVkPswfFlat_{prec_key}_w{w}[2][{count}] = {{")
            for m_idx in range(2):
                flat = _flatten_table(prec_key, w, n_even, n_odd, modes[m_idx], m_idx, True)
                vals = ", ".join(fmt_c_f32(v) for v in flat)
                lines.append(f"    {{{vals}}},")
            lines.append("};")
        elif prec_key == "df":
            lines.append(f"static constexpr VkDf kVkPswfFlat_{prec_key}_w{w}[2][{count}] = {{")
            for m_idx in range(2):
                flat = _flatten_table(prec_key, w, n_even, n_odd, modes[m_idx], m_idx, True)
                vals = ", ".join(f"{{{fmt_c_f32(hi)}, {fmt_c_f32(lo)}}}" for hi, lo in flat)
                lines.append(f"    {{{vals}}},")
            lines.append("};")
        elif prec_key == "fp64":
            lines.append(f"static constexpr double kVkPswfFlat_{prec_key}_w{w}[2][{count}] = {{")
            for m_idx in range(2):
                flat = _flatten_table(prec_key, w, n_even, n_odd, modes[m_idx], m_idx, True)
                vals = ", ".join(fmt_c_f64(v) for v in flat)
                lines.append(f"    {{{vals}}},")
            lines.append("};")

    lines.extend([
        "",
        "static inline const void* vk_pswf_get_flat_table(",
        "    VkPrecKind prec, int w, VkPswfMode mode, bool geo_scale_fp16, size_t* out_bytes) {",
        "    const int m = (mode == VK_PSWF43) ? 1 : 0;",
        "    switch (prec) {",
        "        case VK_PREC_HALF:",
        "            if (w == 4) {",
        "                *out_bytes = sizeof(kVkPswfFlat_fp16_w4[m]);",
        "                return geo_scale_fp16 ? (const void*)kVkPswfFlat_fp16_w4_geo[m] : (const void*)kVkPswfFlat_fp16_w4[m];",
        "            }",
        "            if (w == 5) {",
        "                *out_bytes = sizeof(kVkPswfFlat_fp16_w5[m]);",
        "                return geo_scale_fp16 ? (const void*)kVkPswfFlat_fp16_w5_geo[m] : (const void*)kVkPswfFlat_fp16_w5[m];",
        "            }",
        "            break;",
        "        case VK_PREC_FLOAT:",
        "            if (w == 8) { *out_bytes = sizeof(kVkPswfFlat_fp32_w8[m]); return (const void*)kVkPswfFlat_fp32_w8[m]; }",
        "            if (w == 9) { *out_bytes = sizeof(kVkPswfFlat_fp32_w9[m]); return (const void*)kVkPswfFlat_fp32_w9[m]; }",
        "            break;",
        "        case VK_PREC_DF:",
        "            if (w == 14) { *out_bytes = sizeof(kVkPswfFlat_df_w14[m]); return (const void*)kVkPswfFlat_df_w14[m]; }",
        "            if (w == 15) { *out_bytes = sizeof(kVkPswfFlat_df_w15[m]); return (const void*)kVkPswfFlat_df_w15[m]; }",
        "            if (w == 16) { *out_bytes = sizeof(kVkPswfFlat_df_w16[m]); return (const void*)kVkPswfFlat_df_w16[m]; }",
        "            break;",
        "        case VK_PREC_DOUBLE:",
        "            if (w == 16) { *out_bytes = sizeof(kVkPswfFlat_fp64_w16[m]); return (const void*)kVkPswfFlat_fp64_w16[m]; }",
        "            if (w == 17) { *out_bytes = sizeof(kVkPswfFlat_fp64_w17[m]); return (const void*)kVkPswfFlat_fp64_w17[m]; }",
        "            if (w == 18) { *out_bytes = sizeof(kVkPswfFlat_fp64_w18[m]); return (const void*)kVkPswfFlat_fp64_w18[m]; }",
        "            break;",
        "    }",
        "    *out_bytes = 0;",
        "    return nullptr;",
        "}",
        "",
        "#endif /* TLS_VK_PSWF_COEFFS_H */",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if not out_path.is_file() or out_path.read_text(encoding="utf-8") != content:
        out_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    dst = Path(__file__).resolve().parents[1] / "nufft" / "pswf_coeffs.h"
    generate_c23_header(dst)
    print(f"[pswf_tables] wrote {dst}")
