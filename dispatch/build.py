"""Build the per-variant shared libraries for one wheel target.

One target = one platform (OS + architecture + libc).  For each
microarchitecture variant of that target the builder produces

    .build/<target>/<variant>/libtls-<variant>.a       static archive
    .build/<target>/<variant>/libtls-<variant>.<ext>   the shared object

and a ``manifest.json`` describing what the dispatcher may load.  The wheel
builder then packages the shared objects and the manifest into
``toeplitz_ls/_lib/``.

Two generated inputs come from measurement and follow the same rule: run the
generator when the variant can execute on this host, otherwise borrow the data
from a variant with the same vector width (which is the property both the
scaling cost model and the fusion-depth table depend on).  Borrowing is
recorded in the generated files themselves, and a cache under
``.build/tune-cache/vl<N>/`` keeps the release build from re-measuring the same
vector width once per target.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import toolchains as T
from . import variants as V

ROOT = V.repo_root()
BUILD = ROOT / ".build"
CACHE = BUILD / "tune-cache"
INCLUDE = ROOT / "include"
SRC = ROOT / "src"

PRECISION_SOURCES = ("nanofft", "nufft1", "linalg", "chi2per", "utils")
PRECISIONS = (("", "f"), ("DOUBLE", "d"), ("DOUBLE_DOUBLE", "dd"))

BASE_DEFINES = ("MAX_TWIDDLE_REUSE=16",)

# Flags shared by every gcc/clang family build.  -O3 + -ffast-math is the
# maximum optimisation level every compiler in the matrix supports; the
# double-double translation units opt out of fast-math separately.
GNU_CFLAGS = ("-std=gnu11", "-O3", "-ffast-math", "-fPIC", "-Wall", "-Wextra", "-Werror", "-fvisibility=hidden", "-DTLS_BUILD_SHARED")
GNU_DD_CFLAGS = ("-fno-fast-math",)

# clang-cl spelling of the same policy.  /O2 is clang's -O3 in MSVC mode;
# /fp:fast is its fast-math, /fp:precise is the double-double opt-out.
CL_CFLAGS = ("/O2", "/fp:fast", "/MT", "/W4", "/DTLS_BUILD_SHARED")
CL_DD_CFLAGS = ("/fp:precise",)


class BuildError(RuntimeError):
    pass


@dataclass
class VariantArtifacts:
    target: str
    variant: V.Variant
    static_lib: Path
    shared_lib: Path
    scaling: str  # "measured" | "borrowed:<variant>" | "generic"
    depth: str  # "measured" | "borrowed:<variant>" | "default"


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
def _is_cl(tc: T.Toolchain) -> bool:
    return tc.family == "clang-cl"


def compile_command(tc, *, src, obj, variant, define, include_dirs, dd, depth_include=None):
    if _is_cl(tc):
        cmd = [*tc.cc, *tc.cflags, "/c", str(src), "/Fo" + str(obj), *CL_CFLAGS]
        if dd:
            cmd += list(CL_DD_CFLAGS)
        if define:
            cmd.append("/D" + define)
        for name in BASE_DEFINES:
            cmd.append("/D" + name)
        cmd += list(variant.clang_cl_flags or variant.cflags)
        if depth_include is not None:
            cmd += ["/DNANOFFT_DEPTH_TABLE", "/I" + str(depth_include)]
        for inc in tc.msvc_include + [str(d) for d in include_dirs]:
            cmd.append("-imsvc" + inc)
        return cmd

    cmd = [*tc.cc, *tc.cflags, *variant.cflags, *GNU_CFLAGS]
    if tc.sysroot:
        cmd.append("--sysroot=" + tc.sysroot)
    if dd:
        cmd += list(GNU_DD_CFLAGS)
    if define:
        cmd.append("-D" + define)
    for name in BASE_DEFINES:
        cmd.append("-D" + name)
    if "zig" in tc.name:
        cmd.append("-Wno-unused-function")
    else:
        cmd.append("-Wno-nan-infinity-disabled")
    if depth_include is not None:
        cmd += ["-DNANOFFT_DEPTH_TABLE", "-I" + str(depth_include)]
    for inc in include_dirs:
        cmd.append("-I" + str(inc))
    cmd += ["-c", str(src), "-o", str(obj)]
    return cmd


def link_command(tc, *, objs, out, variant, musl_tmp=None):
    if _is_cl(tc):
        cmd = [*tc.cc, *tc.cflags, "/LD", "/MT", "/Fe" + str(out), *[str(o) for o in objs], "/link", "/NOLOGO"]
        for path in tc.msvc_lib:
            cmd.append("/LIBPATH:" + path)
        return cmd

    cmd = [*tc.cc, *tc.cflags, *variant.cflags]
    if tc.sysroot:
        cmd.append("--sysroot=" + tc.sysroot)
    cmd += ["-L" + d for d in tc.lib_dirs]
    cmd += ["-shared", "-o", str(out), *[str(o) for o in objs]]
    if tc.musl_soname:
        # musl's own shared object has no usable SONAME on every distro, and
        # musllinux wheels must record the canonical one.  Linking against a
        # copy named exactly like that fixes the NEEDED entry without patching.
        cmd += ["-nodefaultlibs", "-L" + str(musl_tmp), "-l:" + tc.musl_soname, "-lgcc"]
    else:
        cmd.append("-lm")
    return cmd


def _run_quietly(cmd, cwd=None, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    result = subprocess.run([str(c) for c in cmd], cwd=cwd, text=True, capture_output=True, env=full_env)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise BuildError("command failed: " + " ".join(str(c) for c in cmd))
    return result


# ---------------------------------------------------------------------------
# Generated inputs: scaling.h and nanofft_depth_table.h
# ---------------------------------------------------------------------------
def _empty_depth_table(path: Path, note: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "/* No tuned table for this variant: every entry is 0, so the plan uses\n"
        " * the per-precision fusion-depth default.  source: %s */\n"
        "#ifndef NANOFFT_DEPTH_TABLE_H\n#define NANOFFT_DEPTH_TABLE_H\n"
        "static const unsigned char nanofft_depth_float[32] = {0};\n"
        "static const unsigned char nanofft_depth_double[32] = {0};\n"
        "static const unsigned char nanofft_depth_dd[32] = {0};\n"
        "#endif\n" % note
    )
    return path


def _measure_inputs(tc, variant, workdir: Path, *, note: str) -> tuple[Path, Path]:
    """Compile and run the scaling generator and the depth tuner for `variant`."""
    workdir.mkdir(parents=True, exist_ok=True)
    scaling_dir = workdir / "scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)
    header = scaling_dir / "scaling.h"
    header.write_text("")

    # Objects for the generator/tuner: no PIC needed, but the same flags.
    objs = {}
    for src_name in ("nanofft", "nufft1"):
        for define, tag in PRECISIONS:
            obj = workdir / f"{src_name}-{tag}.o"
            _run_quietly(
                env=tc.env,
                cmd=compile_command(
                    tc,
                    src=SRC / f"{src_name}.c",
                    obj=obj,
                    variant=variant,
                    define=define,
                    include_dirs=[INCLUDE, scaling_dir],
                    dd=(define == "DOUBLE_DOUBLE"),
                ),
            )
            objs[(src_name, tag)] = obj
    tuning_obj = workdir / "tuning.o"
    _run_quietly(
        compile_command(tc, src=SRC / "tuning.c", obj=tuning_obj, variant=variant, define="", include_dirs=[INCLUDE, scaling_dir], dd=False), env=tc.env
    )

    # scaling.h: the generator appends one precision's constants per run.
    for (define, tag), precision in zip(PRECISIONS, ("Single", "Double", "Double-Double")):
        gen = workdir / f"scaling-{tag}"
        cmd = [
            *tc.cc,
            *tc.cflags,
            *variant.cflags,
            "-O3",
            "-ffast-math",
            "-DSAVE",
            f"-D{define}" if define else "-D_NONE",
            "-I" + str(INCLUDE),
            "-I" + str(scaling_dir),
        ]
        if tc.sysroot:
            cmd.append("--sysroot=" + tc.sysroot)
        cmd += ["-L" + d for d in tc.lib_dirs]
        cmd += [str(SRC / "scaling.c"), str(objs[("nanofft", tag)]), str(objs[("nufft1", tag)]), str(tuning_obj), "-o", str(gen), "-lm"]
        _run_quietly(cmd)
        env = dict(os.environ)
        env.update(tc.env)
        subprocess.run([str(gen)], cwd=scaling_dir, check=True, stdout=subprocess.DEVNULL, env=env)

    # Depth table: the tuner measures execute_raw, so it must not itself be
    # built against a table.
    tuner = workdir / "bench_fuse"
    tuner_src = ROOT / "tests" / "bench_fuse.c"
    cmd = [*tc.cc, *tc.cflags, *variant.cflags, "-std=gnu11", "-O3", "-ffast-math", "-I" + str(INCLUDE), "-I" + str(scaling_dir)]
    if tc.sysroot:
        cmd.append("--sysroot=" + tc.sysroot)
    cmd += ["-L" + d for d in tc.lib_dirs]
    cmd += [str(tuner_src), str(objs[("nanofft", "f")]), str(objs[("nanofft", "d")]), str(objs[("nanofft", "dd")]), str(tuning_obj), "-o", str(tuner), "-lm"]
    _run_quietly(cmd)

    table = workdir / "nanofft_depth_table.h"
    budget = ["--layouts", "16", "--reps", "8", "--minlog", "6", "--maxlog", "18"]
    env = dict(os.environ)
    env.update(tc.env)
    subprocess.run([str(tuner), "--tune", *budget, "--emit", str(table), "--note", note], cwd=workdir, check=True, env=env)
    return header, table


def _cached(vl_arch: str, vl: int) -> tuple[Path, Path] | None:
    base = CACHE / f"{vl_arch}-vl{vl}"
    scaling, table = base / "scaling.h", base / "nanofft_depth_table.h"
    if scaling.exists() and table.exists():
        return scaling, table
    return None


def _cache_store(vl_arch: str, vl: int, scaling: Path, table: Path) -> None:
    base = CACHE / f"{vl_arch}-vl{vl}"
    base.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scaling, base / "scaling.h")
    shutil.copyfile(table, base / "nanofft_depth_table.h")


def _reference_inputs(vl: int, note: str) -> tuple[Path, Path, str] | None:
    """Borrow scaling + depth data for vector width `vl` from a host build.

    This is the "copy the scaling parameters compiled on the host machine with
    equivalent vector length" rule: the cost model and the depth table both
    depend on the vector width, so a build for a variant that cannot execute
    here reuses the measurements of the host variant with the same width.
    """
    if not T.have("gcc"):
        return None
    host_variant = {
        16: "x86-64",
        32: "x86-64-v3",
        64: "x86-64-v4",
    }.get(vl)
    if host_variant is None:
        return None
    cached = _cached("x86_64", vl)
    if cached:
        return cached[0], cached[1], f"host x86-64 {host_variant} (vector width {vl})"

    host = V.TARGETS["linux-x86_64-glibc"]
    variant = next((v for v in host.variants if v.name == host_variant), None)
    if variant is None:
        return None
    sampler = _host_supports(variant)
    if not sampler:
        return None
    print("[tune] measuring vector width %d on %s (one-off, cached)" % (vl, host_variant), flush=True)
    tc = T.native_gcc()
    work = BUILD / "host-reference" / f"vl{vl}"
    scaling, table = _measure_inputs(tc, variant, work, note="host reference %s" % host_variant)
    _cache_store("x86_64", vl, scaling, table)
    return scaling, table, f"host x86-64 {host_variant} (vector width {vl})"


_CANONICAL = {16: "x86-64", 32: "x86-64-v3", 64: "x86-64-v4"}


def _is_canonical(variant: V.Variant) -> bool:
    """Whether this variant is the one measured for its vector width."""
    vl = V.vector_bytes(variant)
    if variant.arch == "aarch64":
        return variant.sve_bytes == 0  # the NEON baseline, width 16
    return _CANONICAL.get(vl) == variant.name


_HOST_FLAGS: dict[str, bool] = {}


def _host_supports(variant: V.Variant) -> bool:
    """Whether this host CPU can execute `variant` (needed to measure it)."""
    if variant.name in _HOST_FLAGS:
        return _HOST_FLAGS[variant.name]
    ok = True
    try:
        flags = set()
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                key, sep, value = line.partition(":")
                if sep and key.strip().lower() == "flags":
                    flags.update(value.split())
                    break
        for group in variant.requires:
            if not any(alt in flags for alt in group):
                ok = False
                break
    except OSError:
        ok = False
    _HOST_FLAGS[variant.name] = ok
    return ok


# ---------------------------------------------------------------------------
# One variant
# ---------------------------------------------------------------------------
def build_variant(target, variant, tc: T.Toolchain, *, run: bool, reference: tuple[Path, Path, str] | None) -> VariantArtifacts:
    work = BUILD / target.name / variant.name
    work.mkdir(parents=True, exist_ok=True)

    gen_dir = work / "generated"
    gen_dir.mkdir(exist_ok=True)
    scaling_path = gen_dir / "scaling.h"
    table_path = gen_dir / "nanofft_depth_table.h"

    if run:
        scaling, table = _measure_inputs(tc, variant, work / "measure", note="%s %s" % (target.name, variant.name))
        shutil.copyfile(scaling, scaling_path)
        shutil.copyfile(table, table_path)
        scaling_state = depth_state = "measured"
        _cache_store("x86_64" if target.arch == "x86_64" else "aarch64", V.vector_bytes(variant), scaling_path, table_path)
    elif reference is not None:
        shutil.copyfile(reference[0], scaling_path)
        shutil.copyfile(reference[1], table_path)
        scaling_state = depth_state = "borrowed: " + reference[2]
    else:
        shutil.copyfile(INCLUDE / "scaling_generic.h", scaling_path)
        _empty_depth_table(table_path, "%s %s" % (target.name, variant.name))
        scaling_state = "generic"
        depth_state = "default"

    # Both generated headers are picked up through -I<gen_dir>.
    depth_dir = gen_dir
    include_dirs = [INCLUDE, gen_dir]

    objs = []
    for src_name in PRECISION_SOURCES:
        for define, tag in PRECISIONS:
            obj = work / f"{src_name}-{tag}.o"
            _run_quietly(
                env=tc.env,
                cmd=compile_command(
                    tc,
                    src=SRC / f"{src_name}.c",
                    obj=obj,
                    variant=variant,
                    define=define,
                    include_dirs=include_dirs,
                    dd=(define == "DOUBLE_DOUBLE"),
                    depth_include=depth_dir,
                ),
            )
            objs.append(obj)
    tuning_obj = work / "tuning.o"
    _run_quietly(
        compile_command(tc, src=SRC / "tuning.c", obj=tuning_obj, variant=variant, define="", include_dirs=include_dirs, dd=False, depth_include=depth_dir),
        env=tc.env,
    )
    objs.append(tuning_obj)
    probe_obj = work / "probe_arm64.o"
    _run_quietly(
        compile_command(tc, src=SRC / "probe_arm64.c", obj=probe_obj, variant=variant, define="", include_dirs=include_dirs, dd=False, depth_include=depth_dir),
        env=tc.env,
    )
    objs.append(probe_obj)
    cpu_obj = work / "cpu_features.o"
    _run_quietly(
        compile_command(tc, src=SRC / "cpu_features.c", obj=cpu_obj, variant=variant, define="", include_dirs=include_dirs, dd=False, depth_include=depth_dir),
        env=tc.env,
    )
    objs.append(cpu_obj)

    static_lib = work / f"{variant.library_stem}.a"
    _run_quietly([*tc.ar, "rcs", str(static_lib), *[str(o) for o in objs]], env=tc.env)

    musl_tmp = None
    if tc.musl_soname:
        musl_tmp = work / "musl-link"
        musl_tmp.mkdir(exist_ok=True)
        shutil.copyfile(Path(tc.musl_libdir) / "libc.so", musl_tmp / tc.musl_soname)
    out = work / f"{variant.library_stem}{target.extension}"
    _run_quietly(link_command(tc, objs=objs, out=out, variant=variant, musl_tmp=musl_tmp), env=tc.env)
    if not _is_cl(tc):
        _quiet_strip(out)

    return VariantArtifacts(
        target=target.name,
        variant=variant,
        static_lib=static_lib,
        shared_lib=out,
        scaling=scaling_state,
        depth=depth_state,
    )


def _quiet_strip(path: Path) -> None:
    strip = T.have("llvm-strip") or T.have("strip")
    if not strip:
        return
    # Do not fail the build over an unstripped library.
    subprocess.run([strip, "--strip-unneeded", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------------
def build_target(name: str, *, only=None, keep_going=False) -> tuple[list[VariantArtifacts], Path]:
    target = V.TARGETS[name]
    tc = T.for_target(target)
    print("\n=== %s (%s) ===" % (name, tc.name), flush=True)

    selected = [v for v in target.variants if only is None or v.name in only]
    if not selected:
        raise BuildError("no variants selected for %s" % name)

    artifacts: list[VariantArtifacts] = []
    for variant in selected:
        # Measure only the canonical variant of each vector width; the rest of
        # that width borrows its calibration (both the cost model and the depth
        # table are functions of the vector width).
        arch_key = "x86_64" if target.arch == "x86_64" else "aarch64"
        already_measured = _cached(arch_key, V.vector_bytes(variant)) is not None
        runnable = tc.can_run and _host_supports(variant) and target.os == "linux" and _is_canonical(variant) and not already_measured
        reference = None
        if not runnable:
            reference = _reference_inputs(V.vector_bytes(variant), "%s %s" % (target.name, variant.name))
            if reference is None:
                # Last resort: a sibling of this target that could be measured.
                sibling = next((a for a in artifacts if V.vector_bytes(a.variant) == V.vector_bytes(variant) and a.depth == "measured"), None)
                if sibling is not None:
                    reference = (
                        BUILD / name / sibling.variant.name / "generated" / "scaling.h",
                        BUILD / name / sibling.variant.name / "generated" / "nanofft_depth_table.h",
                        sibling.variant.name,
                    )
        print("[build] %s/%s%s" % (target.name, variant.name, "" if runnable else "  (borrowed calibration)"), flush=True)
        artifacts.append(build_variant(target, variant, tc, run=runnable, reference=reference))

    manifest = write_manifest(target, artifacts)
    return artifacts, manifest


def write_manifest(target, artifacts: list[VariantArtifacts]) -> Path:
    staging = BUILD / target.name / "_lib"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    entries = []
    for art in artifacts:
        shutil.copyfile(art.shared_lib, staging / art.shared_lib.name)
        entries.append(
            {
                "name": art.variant.name,
                "file": art.shared_lib.name,
                "arch": art.variant.arch,
                "vector_bytes": V.vector_bytes(art.variant),
                "requires": [list(group) for group in art.variant.requires],
            }
        )
    manifest = {
        "schema": 1,
        "library": "tls",
        "target": {
            "name": target.name,
            "os": target.os,
            "arch": target.arch,
            "libc": target.libc,
            "platform_tag": target.platform_tag,
        },
        "baseline": 0,
        "variants": entries,
    }
    path = staging / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("[manifest] %s (%d variants)" % (path, len(entries)), flush=True)
    return path


def build_targets(names, *, only=None) -> list[tuple[list[VariantArtifacts], Path]]:
    out = []
    for name in names:
        out.append(build_target(name, only=only))
    return out


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Build the per-variant Toeplitz-LS libraries for one or more wheel targets.")
    parser.add_argument("--target", action="append", dest="targets", choices=sorted(V.TARGETS), help="wheel target (repeatable)")
    parser.add_argument("--all", action="store_true", help="build every target this host can build")
    parser.add_argument("--variant", action="append", dest="variants", help="restrict to these variant names (repeatable)")
    parser.add_argument("--list", action="store_true", help="print the target/variant matrix and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, target in V.TARGETS.items():
            print("%-22s %-26s %s" % (name, target.platform_tag, ", ".join(v.name for v in target.variants)))
        return 0

    names = args.targets or []
    if args.all and not names:
        names = [
            "linux-x86_64-glibc",
            "linux-x86_64-musl",
            "linux-aarch64-glibc",
            "linux-aarch64-musl",
            "win-amd64",
            "win-arm64",
            "macos-x86_64",
            "macos-arm64",
        ]
    if not names:
        parser.error("give --target, --all or --list")

    for name in names:
        artifacts, manifest = build_target(name, only=args.variants)
        print("[done] %s: %d libraries -> %s" % (name, len(artifacts), manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
