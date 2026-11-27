#!/usr/bin/env python3
"""Build-time GLSL -> SPIR-V codegen, validator, NoContraction auditor, and C++ embedder.

Generates all Vulkan compute shaders, compiles them in parallel with glslc,
validates with spirv-val, optimizes with spirv-opt -O, audits every DF shader
via spirv-dis to ensure SPIR-V NoContraction decorations survive optimization,
and writes:
  - <out_dir>/shaders_spv.hpp
  - <out_dir>/shaders_spv.cpp
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gen_bitrev import generate_bitrev_shaders
from gen_probes import generate_probe_shaders


def collect_all_shaders() -> dict[str, dict]:
    """Collect shaders from all registered generator modules."""
    shaders: dict[str, dict] = {}
    shaders.update(generate_probe_shaders())
    shaders.update(generate_bitrev_shaders())

    # Optional milestone modules loaded automatically as they are added
    for mod_name, fn_name in (
        ("gen_dif_fft", "generate_dif_fft_shaders"),
        ("gen_nufft", "generate_nufft_shaders"),
        ("gen_periodogram", "generate_periodogram_shaders"),
    ):
        if (SCRIPT_DIR / f"{mod_name}.py").is_file():
            mod = __import__(mod_name)
            shaders.update(getattr(mod, fn_name)())

    return shaders


def _compile_one(task: tuple[str, dict, str, str, str, str, str, str, str]) -> tuple[str, bytes, int]:
    name, spec, glsl_dir, spv_dir, glslc, spirv_val, spirv_opt, spirv_dis, spirv_as = task
    glsl_path = Path(glsl_dir) / f"{name}.comp"
    raw_spv = Path(spv_dir) / f"{name}.raw.spv"
    patched_spv = Path(spv_dir) / f"{name}.patched.spv"
    opt_spv = Path(spv_dir) / f"{name}.spv"

    src = spec["source"]
    target_env = spec.get("target_env", "vulkan1.1")
    check_nocontract = bool(spec.get("check_nocontract", False))

    # Avoid unnecessary rewrites if source & compiled spv are up-to-date
    src_hash_path = Path(spv_dir) / f"{name}.sha256"
    src_digest = hashlib.sha256(("v3_noinline_df_strip\n" + target_env + "\n" + src).encode("utf-8")).hexdigest()
    if (
        opt_spv.is_file()
        and src_hash_path.is_file()
        and src_hash_path.read_text(encoding="utf-8").strip() == src_digest
    ):
        return name, opt_spv.read_bytes(), 1

    glsl_path.write_text(src, encoding="utf-8")

    # 1. Compile GLSL -> SPIR-V (-O0; spirv-opt runs after Fma NoContraction patching)
    cmd_glslc = [
        glslc,
        f"--target-env={target_env}",
        "-fshader-stage=compute",
        "-O0",
        str(glsl_path),
        "-o",
        str(raw_spv),
    ]
    res = subprocess.run(cmd_glslc, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"glslc failed for {name}:\n{res.stderr}\n{res.stdout}")

    # 2. Validate raw SPIR-V
    res = subprocess.run([spirv_val, str(raw_spv)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"spirv-val (raw) failed for {name}:\n{res.stderr}")

    opt_in = raw_spv
    # 2b. For DF shaders, attach OpDecorate NoContraction to every GLSL.std.450 Fma
    #     instruction (glslang omits NoContraction on OpExtInst Fma, which lets Mesa
    #     NIR/RADV treat Fma as exact=false and fold away Dekker error terms).
    if check_nocontract:
        res = subprocess.run([spirv_dis, str(raw_spv)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"spirv-dis (raw) failed for {name}:\n{res.stderr}")
        asm = res.stdout
        fma_ids = re.findall(r"(%\w+)\s*=\s*OpExtInst\s+%float\s+%\w+\s+Fma\b", asm)
        if fma_ids:
            existing = set(re.findall(r"OpDecorate\s+(%\w+)\s+NoContraction\b", asm))
            add_lines = [f"               OpDecorate {fid} NoContraction" for fid in fma_ids if fid not in existing]
            if add_lines:
                anchor = "OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize"
                if anchor in asm:
                    asm = asm.replace(anchor, anchor + "\n" + "\n".join(add_lines), 1)
                else:
                    # Insert right before first OpTypeVoid
                    asm = asm.replace("OpTypeVoid", "\n".join(add_lines) + "\n               OpTypeVoid", 1)
                spvasm_path = Path(spv_dir) / f"{name}.spvasm"
                spvasm_path.write_text(asm, encoding="utf-8")
                res = subprocess.run([spirv_as, str(spvasm_path), "-o", str(patched_spv)], capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"spirv-as failed for {name}:\n{res.stderr}")
                opt_in = patched_spv

    # 3. Optimize SPIR-V & strip debug/nonsemantic metadata.
    #    For DF shaders (check_nocontract=True), avoid exhaustive function inlining
    #    before Vulkan pipeline specialization: df_two_product holds both FMA and
    #    Veltkamp branches guarded by spec-constant DF_USE_VELTKAMP, so inlining
    #    every df_add/df_mul/df_log/df_exp call into main() bloats SPIR-V 5x-7x
    #    (13.5 MiB -> 2.5 MiB) and slows cold vkCreateComputePipelines without
    #    helping runtime (drivers specialize DF_USE_VELTKAMP first, then inline).
    if check_nocontract:
        opt_args = [
            spirv_opt,
            "--strip-debug",
            "--strip-nonsemantic",
            "--eliminate-dead-branches",
            "--merge-return",
            "--eliminate-dead-code-aggressive",
            "--private-to-local",
            "--scalar-replacement=100",
            "--ssa-rewrite",
            "--ccp",
            "--simplify-instructions",
            "--redundancy-elimination",
            "--eliminate-local-single-block",
            "--eliminate-local-single-store",
            "--eliminate-dead-code-aggressive",
            "--cfg-cleanup",
            "--compact-ids",
            str(opt_in),
            "-o",
            str(opt_spv),
        ]
    else:
        opt_args = [
            spirv_opt,
            "-O",
            "--strip-debug",
            "--strip-nonsemantic",
            "--compact-ids",
            str(opt_in),
            "-o",
            str(opt_spv),
        ]
    res = subprocess.run(opt_args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"spirv-opt failed for {name}:\n{res.stderr}")

    # 4. Re-validate optimized SPIR-V
    res = subprocess.run([spirv_val, str(opt_spv)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"spirv-val (opt) failed for {name}:\n{res.stderr}")

    # 5. Audit NoContraction on Dekker DF shaders (including every surviving Fma)
    if check_nocontract:
        res = subprocess.run([spirv_dis, str(opt_spv)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"spirv-dis failed for {name}:\n{res.stderr}")
        nc_ids = set(re.findall(r"OpDecorate\s+(%\w+)\s+NoContraction\b", res.stdout))
        if len(nc_ids) < 4:
            raise RuntimeError(
                f"NoContraction audit FAILED for DF shader '{name}': found only {len(nc_ids)} "
                f"OpDecorate NoContraction annotations after spirv-opt -O!"
            )
        opt_fma_ids = re.findall(r"(%\w+)\s*=\s*OpExtInst\s+%float\s+%\w+\s+Fma\b", res.stdout)
        missing_fma = [fid for fid in opt_fma_ids if fid not in nc_ids]
        if missing_fma:
            raise RuntimeError(
                f"NoContraction Fma audit FAILED for DF shader '{name}': {len(missing_fma)} Fma ops lost NoContraction!"
            )

    src_hash_path.write_text(src_digest + "\n", encoding="utf-8")
    return name, opt_spv.read_bytes(), 0


def emit_c_registry(
    out_dir: Path,
    compiled: list[tuple[str, bytes]],
    code_hash: str,
) -> None:
    """Emit plain C11 shaders_spv.h and shaders_spv.c."""
    h_path = out_dir / "shaders_spv.h"
    c_path = out_dir / "shaders_spv.c"

    enum_lines = []
    for i, (name, _) in enumerate(compiled):
        enum_lines.append(f"    VK_SHADER_{name} = {i},")

    header = f"""#ifndef TLS_VK_SHADERS_SPV_H
#define TLS_VK_SHADERS_SPV_H
/* AUTO-GENERATED by offload/vulkan/codegen/compile_shaders.py — DO NOT EDIT. */

#include <stddef.h>
#include <stdint.h>

typedef enum VkShaderId {{
{chr(10).join(enum_lines)}
    VK_SHADER_COUNT = {len(compiled)}
}} VkShaderId;

typedef struct VkEmbeddedSpirv {{
    const char*     name;
    const uint32_t* words;
    size_t          word_count;
}} VkEmbeddedSpirv;

const VkEmbeddedSpirv* vk_get_embedded_spirv(VkShaderId id);
const VkEmbeddedSpirv* vk_find_embedded_spirv(const char* name);
const char*            vk_shader_bundle_hash(void);

#endif /* TLS_VK_SHADERS_SPV_H */
"""

    c_parts = [
        '/* AUTO-GENERATED by offload/vulkan/codegen/compile_shaders.py — DO NOT EDIT. */',
        '#include "shaders_spv.h"',
        '#include <string.h>',
        '',
    ]

    for name, raw_bytes in compiled:
        if len(raw_bytes) % 4 != 0:
            raise RuntimeError(f"SPIR-V for {name} has non-multiple-of-4 size {len(raw_bytes)}")
        nwords = len(raw_bytes) // 4
        words = [int.from_bytes(raw_bytes[i * 4 : (i + 1) * 4], "little") for i in range(nwords)]
        c_parts.append(f"static const uint32_t kSpv_{name}[{nwords}] = {{")
        for row_start in range(0, nwords, 8):
            chunk = ", ".join(f"0x{w:08x}u" for w in words[row_start : row_start + 8])
            c_parts.append(f"    {chunk},")
        c_parts.append("};")

    c_parts.append("")
    c_parts.append(f"static const VkEmbeddedSpirv kAllSpirv[{len(compiled)}] = {{")
    for name, raw_bytes in compiled:
        nwords = len(raw_bytes) // 4
        c_parts.append(f'    {{"{name}", kSpv_{name}, {nwords}u}},')
    c_parts.append("};")
    c_parts.append("")
    c_parts.append("const VkEmbeddedSpirv* vk_get_embedded_spirv(VkShaderId id) {")
    c_parts.append(f"    if ((unsigned)id >= {len(compiled)}u) return NULL;")
    c_parts.append("    return &kAllSpirv[(unsigned)id];")
    c_parts.append("}")
    c_parts.append("")
    c_parts.append("const VkEmbeddedSpirv* vk_find_embedded_spirv(const char* name) {")
    c_parts.append("    if (!name) return NULL;")
    c_parts.append(f"    for (size_t i = 0; i < {len(compiled)}u; ++i) {{")
    c_parts.append("        if (strcmp(kAllSpirv[i].name, name) == 0) return &kAllSpirv[i];")
    c_parts.append("    }")
    c_parts.append("    return NULL;")
    c_parts.append("}")
    c_parts.append("")
    c_parts.append("const char* vk_shader_bundle_hash(void) {")
    c_parts.append(f'    return "{code_hash}";')
    c_parts.append("}")
    c_parts.append("")
    c_src = "\n".join(c_parts)

    if not h_path.is_file() or h_path.read_text(encoding="utf-8") != header:
        h_path.write_text(header, encoding="utf-8")
    if not c_path.is_file() or c_path.read_text(encoding="utf-8") != c_src:
        c_path.write_text(c_src, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile & embed Toeplitz-LS Vulkan GLSL shaders.")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated files")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="Parallel compile workers")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    glsl_dir = out_dir / "glsl"
    spv_dir = out_dir / "spv"
    glsl_dir.mkdir(parents=True, exist_ok=True)
    spv_dir.mkdir(parents=True, exist_ok=True)

    glslc = shutil.which("glslc")
    spirv_val = shutil.which("spirv-val")
    spirv_opt = shutil.which("spirv-opt")
    spirv_dis = shutil.which("spirv-dis")
    spirv_as = shutil.which("spirv-as")
    for tool_name, path in (
        ("glslc", glslc), ("spirv-val", spirv_val),
        ("spirv-opt", spirv_opt), ("spirv-dis", spirv_dis), ("spirv-as", spirv_as),
    ):
        if not path:
            raise RuntimeError(f"Required Vulkan SDK tool '{tool_name}' not found on PATH")

    shaders = collect_all_shaders()
    names_sorted = sorted(shaders.keys())
    tasks = [
        (name, shaders[name], str(glsl_dir), str(spv_dir), glslc, spirv_val, spirv_opt, spirv_dis, spirv_as)
        for name in names_sorted
    ]

    compiled_map: dict[str, bytes] = {}
    cached_cnt = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for name, data, was_cached in pool.map(_compile_one, tasks):
            compiled_map[name] = data
            cached_cnt += was_cached

    compiled = [(name, compiled_map[name]) for name in names_sorted]
    hasher = hashlib.sha256()
    total_bytes = 0
    for name, data in compiled:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(data)
        total_bytes += len(data)
    code_hash = hasher.hexdigest()[:16]

    emit_c_registry(out_dir, compiled, code_hash)
    print(
        f"[vk-codegen] {len(compiled)} shaders ({len(compiled) - cached_cnt} compiled, "
        f"{cached_cnt} cached), {total_bytes / 1024.0:.1f} KiB SPIR-V, hash={code_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
