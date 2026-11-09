"""Microarchitecture variants of the Toeplitz-LS native library.

Each variant is one complete build of the library with its own ``-march``
flags, linked into its own shared object.  The wheel ships all of them and
``toeplitz_ls._dispatch`` loads exactly one, chosen from the CPU it runs on,
so there is no run-time dispatch indirection inside the library at all.

``requires`` describes what the CPU must have for a variant to be loadable.
It is a tuple of groups: every group must be satisfied, and a group is
satisfied by *any* of its entries.  That mirrors the reference implementation
in ihsnpeaks (``dispatch/variants.py``) and lets one variant accept ISA
aliases (Linux reports ``abm`` where the ISA says ``lzcnt``, Windows reports
``sse4_2`` where the level name is ``x86-64-v2``).

The requirement sets are per platform on purpose: Linux can assert the full
x86-64-vN feature list from /proc/cpuinfo, while Windows only exposes a
handful of IsProcessorFeaturePresent bits and macOS only a few sysctl
strings, so those wheels ask for the strongest feature each OS can actually
confirm and leave the rest to the baseline library.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Variant:
    """One compiled microarchitecture level."""

    name: str
    arch: str  # "x86_64" or "aarch64"
    cflags: tuple[str, ...]
    requires: tuple[tuple[str, ...], ...] = ()
    # SVE vector length in bytes this build was compiled for (0 = no SVE).
    sve_bytes: int = 0
    # GCC/clang spelling of the same level for the Windows/MSVC driver.
    clang_cl_flags: tuple[str, ...] = ()

    @property
    def symbol(self) -> str:
        return self.name.replace("-", "_").replace(".", "_").replace("+", "_")

    @property
    def library_stem(self) -> str:
        return f"libtls-{self.name}"


X86_VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="x86-64",
        arch="x86_64",
        cflags=("-march=x86-64", "-mtune=generic"),
        # Everything below is optional for the baseline.
        requires=(),
        clang_cl_flags=("-march=x86-64",),
    ),
    Variant(
        name="x86-64-v2",
        arch="x86_64",
        cflags=("-march=x86-64-v2", "-mtune=generic"),
        requires=(("sse4_2",), ("popcnt",), ("cx16",)),
        clang_cl_flags=("-march=x86-64-v2",),
    ),
    Variant(
        name="x86-64-v2+avx",
        arch="x86_64",
        cflags=("-march=x86-64-v2", "-mavx", "-mxsave", "-mtune=generic"),
        requires=(("sse4_2",), ("popcnt",), ("cx16",), ("avx",), ("xsave",)),
        clang_cl_flags=("-march=x86-64-v2", "-mavx", "-mxsave"),
    ),
    Variant(
        name="x86-64-v3",
        arch="x86_64",
        cflags=("-march=x86-64-v3", "-mtune=generic"),
        requires=(("sse4_2",), ("popcnt",), ("cx16",), ("avx",), ("xsave",), ("avx2",), ("fma",), ("bmi1",), ("bmi2",), ("f16c",), ("lzcnt",), ("movbe",)),
        clang_cl_flags=("-march=x86-64-v3",),
    ),
    Variant(
        name="x86-64-v4",
        arch="x86_64",
        cflags=("-march=x86-64-v4", "-mtune=generic"),
        requires=(
            ("sse4_2",),
            ("popcnt",),
            ("cx16",),
            ("avx",),
            ("xsave",),
            ("avx2",),
            ("fma",),
            ("bmi1",),
            ("bmi2",),
            ("f16c",),
            ("lzcnt",),
            ("movbe",),
            ("avx512f",),
            ("avx512bw",),
            ("avx512cd",),
            ("avx512dq",),
            ("avx512vl",),
        ),
        clang_cl_flags=("-march=x86-64-v4",),
    ),
)

# Windows and macOS expose far fewer queryable features; the requirement sets
# below only ask for what each OS can actually confirm.
X86_VARIANTS_WINDOWS: tuple[Variant, ...] = (
    Variant(name="x86-64", arch="x86_64", cflags=("-march=x86-64",), requires=(), clang_cl_flags=("-march=x86-64",)),
    Variant(name="x86-64-v2", arch="x86_64", cflags=("-march=x86-64-v2",), requires=(("sse4_2",),), clang_cl_flags=("-march=x86-64-v2",)),
    Variant(
        name="x86-64-v2+avx",
        arch="x86_64",
        cflags=("-march=x86-64-v2", "-mavx"),
        requires=(("sse4_2",), ("avx",)),
        clang_cl_flags=("-march=x86-64-v2", "-mavx"),
    ),
    Variant(name="x86-64-v3", arch="x86_64", cflags=("-march=x86-64-v3",), requires=(("avx2",),), clang_cl_flags=("-march=x86-64-v3",)),
    Variant(name="x86-64-v4", arch="x86_64", cflags=("-march=x86-64-v4",), requires=(("avx512f",),), clang_cl_flags=("-march=x86-64-v4",)),
)

# zig spells the microarchitecture levels with LLVM CPU names; `westmere` is
# the closest match to "x86-64-v2 plus AVX" (AVX, no AVX2).
X86_VARIANTS_MACOS: tuple[Variant, ...] = (
    Variant(name="x86-64", arch="x86_64", cflags=("-mcpu=x86_64",), requires=()),
    Variant(name="x86-64-v2", arch="x86_64", cflags=("-mcpu=x86_64_v2",), requires=(("sse4_2",),)),
    Variant(name="x86-64-v2+avx", arch="x86_64", cflags=("-mcpu=westmere",), requires=(("sse4_2",), ("avx",))),
    Variant(name="x86-64-v3", arch="x86_64", cflags=("-mcpu=x86_64_v3",), requires=(("avx2",),)),
    Variant(name="x86-64-v4", arch="x86_64", cflags=("-mcpu=x86_64_v4",), requires=(("avx512f",),)),
)

ARM_VARIANTS: tuple[Variant, ...] = (
    Variant(name="armv8-a", arch="aarch64", cflags=("-march=armv8-a+simd",), requires=(), sve_bytes=0),
    Variant(
        name="armv8.2-a+sve-256",
        arch="aarch64",
        cflags=("-march=armv8.2-a+sve", "-msve-vector-bits=256"),
        # SVE libraries are only usable when the hardware vector length is the
        # one they were compiled for; "sve" alone is not enough.
        requires=(("sve",), ("sve_vl:32",)),
        sve_bytes=32,
    ),
    Variant(
        name="armv8.2-a+sve-512",
        arch="aarch64",
        cflags=("-march=armv8.2-a+sve", "-msve-vector-bits=512"),
        requires=(("sve",), ("sve_vl:64",)),
        sve_bytes=64,
    ),
)

ARM_VARIANTS_WINDOWS: tuple[Variant, ...] = ARM_VARIANTS

# Apple silicon implements no SVE and macOS exposes no SVE state, so the
# complete arm64 macOS set is the baseline NEON library.
ARM_VARIANTS_MACOS: tuple[Variant, ...] = (Variant(name="armv8-a", arch="aarch64", cflags=(), requires=(), sve_bytes=0),)


@dataclass(frozen=True)
class Target:
    """One wheel platform: OS, architecture, libc and toolchain identity."""

    name: str  # e.g. "linux-x86_64-glibc"
    os: str  # linux / windows / macos
    arch: str
    libc: str = ""  # glibc / musl / "" (windows, macos)
    platform_tag: str = ""
    variants: tuple[Variant, ...] = field(default_factory=tuple)
    extension: str = ".so"

    @property
    def is_cross(self) -> bool:
        return self.arch != platform.machine().replace("amd64", "x86_64").replace("arm64", "aarch64")


TARGETS: dict[str, Target] = {
    "linux-x86_64-glibc": Target(
        name="linux-x86_64-glibc",
        os="linux",
        arch="x86_64",
        libc="glibc",
        platform_tag="manylinux_2_17_x86_64",
        variants=X86_VARIANTS,
    ),
    "linux-x86_64-musl": Target(
        name="linux-x86_64-musl",
        os="linux",
        arch="x86_64",
        libc="musl",
        platform_tag="musllinux_1_2_x86_64",
        variants=X86_VARIANTS,
    ),
    "linux-aarch64-glibc": Target(
        name="linux-aarch64-glibc",
        os="linux",
        arch="aarch64",
        libc="glibc",
        platform_tag="manylinux_2_17_aarch64",
        variants=ARM_VARIANTS,
    ),
    "linux-aarch64-musl": Target(
        name="linux-aarch64-musl",
        os="linux",
        arch="aarch64",
        libc="musl",
        platform_tag="musllinux_1_2_aarch64",
        variants=ARM_VARIANTS,
    ),
    "win-amd64": Target(
        name="win-amd64",
        os="windows",
        arch="x86_64",
        platform_tag="win_amd64",
        variants=X86_VARIANTS_WINDOWS,
        extension=".dll",
    ),
    "win-arm64": Target(
        name="win-arm64",
        os="windows",
        arch="aarch64",
        platform_tag="win_arm64",
        variants=ARM_VARIANTS_WINDOWS,
        extension=".dll",
    ),
    "macos-x86_64": Target(
        name="macos-x86_64",
        os="macos",
        arch="x86_64",
        platform_tag="macosx_11_0_x86_64",
        variants=X86_VARIANTS_MACOS,
    ),
    "macos-arm64": Target(
        name="macos-arm64",
        os="macos",
        arch="aarch64",
        platform_tag="macosx_11_0_arm64",
        variants=ARM_VARIANTS_MACOS,
    ),
}

# The variant order in each tuple is baseline-first; the dispatcher walks it
# from the back (widest first) and falls back to the first entry.
BASELINE = 0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def target_names() -> Iterable[str]:
    return TARGETS.keys()


def vector_bytes(variant: Variant) -> int:
    """Vector width the variant is compiled with, in bytes.

    This is the quantity the scaling calibration and the fusion-depth table
    depend on, so it is also the key used to borrow those files for a variant
    that cannot be executed on the build host.
    """
    if variant.arch == "aarch64":
        return variant.sve_bytes or 16
    if "v4" in variant.name:
        return 64
    if "v3" in variant.name or "avx" in variant.name:
        return 32
    return 16
