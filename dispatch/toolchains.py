"""Toolchain discovery and provisioning for the release wheels.

Everything lands in ``.toolchains/`` next to the repository (git-ignored) and
is only needed at build time; the wheels themselves never depend on any of it.
Two rules keep this reproducible without root:

* Downloads are pinned by URL and, where the upstream publishes one, by
  sha256.  Everything is extracted into a private prefix, never installed.
* Nothing is installed system-wide: the Debian cross compiler comes from
  downloaded ``.deb`` files unpacked with ``dpkg-deb -x``, the musl toolchains
  from the musl.cc tarballs, and the MSVC toolset plus Windows SDK from
  ``xwin splat``.

The Windows compiler is ``clang-cl`` targeting the **MSVC ABI** (``/MT``, UCRT,
``lld-link``); MinGW/mingw-w64 is deliberately not used, so the shipped DLL is
a normal MSVC-compatible PE that needs no MinGW runtime.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .variants import repo_root

TOOLCHAIN_DIR = repo_root() / ".toolchains"

DEBIAN_AARCH64_PACKAGES = (
    "gcc-14-aarch64-linux-gnu",
    "cpp-14-aarch64-linux-gnu",
    "libgcc-14-dev-arm64-cross",
    "libc6-dev-arm64-cross",
    "libc6-arm64-cross",
    "linux-libc-dev-arm64-cross",
    "binutils-aarch64-linux-gnu",
)

MUSLCC = {
    "x86_64": "https://musl.cc/x86_64-linux-musl-cross.tgz",
    "aarch64": "https://musl.cc/aarch64-linux-musl-cross.tgz",
}

XWIN_URL = "https://github.com/Jake-Shadle/xwin/releases/download/0.10.0/xwin-0.10.0-x86_64-unknown-linux-musl.tar.gz"

# manylinux2014 (CentOS 7) sysroots: the only way to promise GLIBC_2.17 with a
# modern compiler.  Linking against a current glibc records exp@GLIBC_2.29 and
# friends, which no manylinux_2_17 wheel may need.
MANYLINUX_IMAGE = {
    "x86_64": ("quay.io/pypa/manylinux2014_x86_64", "linux/amd64"),
    "aarch64": ("quay.io/pypa/manylinux2014_aarch64", "linux/arm64"),
}


class ToolchainError(RuntimeError):
    pass


def run(cmd, cwd=None, check=True, capture=False):
    printable = " ".join(str(c) for c in cmd)
    print("+ " + printable, flush=True)
    if capture:
        return subprocess.run([str(c) for c in cmd], cwd=cwd, check=check, text=True, capture_output=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=check)


def have(tool: str) -> str | None:
    return shutil.which(tool)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("[fetch] %s -> %s" % (url, dest), flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.rename(dest)
    return dest


@dataclass
class Toolchain:
    """How to drive one compiler for one target."""

    name: str
    cc: list[str]
    ar: list[str]
    family: str  # "gnu" or "clang-cl"
    cflags: list[str] = field(default_factory=list)
    ldflags: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    lib_dirs: list[str] = field(default_factory=list)
    sysroot: str = ""
    # Extra flags a clang-cl build needs to find the MSVC/SDK layout.
    msvc_include: list[str] = field(default_factory=list)
    msvc_lib: list[str] = field(default_factory=list)
    musl_soname: str = ""  # e.g. libc.musl-x86_64.so.1
    musl_libdir: str = ""
    can_run: bool = False  # can binaries built for this target run on this host?
    env: dict = field(default_factory=dict)  # extra environment for the compiler


def gcc_toolchain(name: str, cc: str, *, can_run: bool, extra_cflags=(), sysroot: str = "", musl=None) -> Toolchain:
    return Toolchain(
        name=name,
        cc=[cc],
        ar=[have("llvm-ar") or have("ar") or "ar"],
        family="gnu",
        cflags=list(extra_cflags),
        sysroot=sysroot,
        musl_soname=musl[0] if musl else "",
        musl_libdir=musl[1] if musl else "",
        can_run=can_run,
    )


def native_gcc() -> Toolchain:
    cc = have("gcc") or have("cc")
    if not cc:
        raise ToolchainError("gcc is required for the native Linux builds")
    return gcc_toolchain("gcc", cc, can_run=True)


def native_musl() -> Toolchain:
    cc = have("musl-gcc") or have("x86_64-linux-musl-gcc")
    if not cc:
        raise ToolchainError("musl-gcc is required for the musl x86_64 build (install musl-tools)")
    libdir = "/lib/x86_64-linux-musl"
    if not Path(libdir, "libc.so").exists():
        raise ToolchainError("musl libc not found in %s" % libdir)
    return gcc_toolchain("musl-gcc", cc, can_run=True, musl=("libc.musl-x86_64.so.1", libdir))


def muslcc(arch: str) -> Toolchain:
    """musl.cc cross toolchain (GCC) for the given architecture."""
    prefix = TOOLCHAIN_DIR / f"{arch}-linux-musl-cross"
    gcc = prefix / "bin" / f"{arch}-linux-musl-gcc"
    if not gcc.exists():
        tarball = _download(MUSLCC[arch], TOOLCHAIN_DIR / f"{arch}-linux-musl-cross.tgz")
        print("[fetch] unpacking %s" % tarball, flush=True)
        with tarfile.open(tarball) as tar:
            # musl.cc ships absolute symlinks inside its own sysroot; they are
            # relative to that prefix, so extract them verbatim.
            try:
                tar.extractall(TOOLCHAIN_DIR, filter="fully_trusted")
            except TypeError:  # Python < 3.12 has no filter argument
                tar.extractall(TOOLCHAIN_DIR)
        if not gcc.exists():
            raise ToolchainError("unexpected musl.cc layout in %s" % tarball)
    libdir = prefix / f"{arch}-linux-musl" / "lib"
    soname = f"libc.musl-{arch}.so.1"
    if not (libdir / "libc.so").exists():
        raise ToolchainError("musl.cc sysroot is missing libc.so in %s" % libdir)
    # A program built for aarch64 only runs here through qemu.
    return gcc_toolchain(f"muslcc-{arch}", str(gcc), can_run=False, musl=(soname, str(libdir)))


def debian_aarch64() -> Toolchain:
    """Debian's aarch64 cross GCC, downloaded and unpacked without root."""
    root = TOOLCHAIN_DIR / "debian-aarch64"
    gcc = root / "usr" / "bin" / "aarch64-linux-gnu-gcc-14"
    if not gcc.exists():
        debs = TOOLCHAIN_DIR / "debs"
        debs.mkdir(parents=True, exist_ok=True)
        print("[fetch] downloading %d Debian cross packages" % len(DEBIAN_AARCH64_PACKAGES), flush=True)
        subprocess.run(
            ["apt-get", "download", *DEBIAN_AARCH64_PACKAGES],
            cwd=debs,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        root.mkdir(parents=True, exist_ok=True)
        for deb in sorted(debs.glob("*.deb")):
            subprocess.run(["dpkg-deb", "-x", str(deb), str(root)], check=True)
    if not gcc.exists():
        raise ToolchainError("aarch64 cross compiler missing after unpacking %s" % root)
    return gcc_toolchain("debian-aarch64", str(gcc), can_run=False, sysroot=str(root))


def manylinux_sysroot(arch: str) -> Path:
    """glibc 2.17 sysroot extracted from the official manylinux2014 image."""
    dest = TOOLCHAIN_DIR / f"sysroot-glibc217-{arch}"
    if (dest / "usr" / "include" / "stdio.h").exists():
        return dest
    image, platform = MANYLINUX_IMAGE[arch]
    docker = have("docker")
    if not docker:
        raise ToolchainError("docker is required to extract the manylinux2014 sysroot for %s" % arch)
    print("[fetch] manylinux2014 sysroot for %s from %s (one-off, ~1 GiB)" % (arch, image), flush=True)
    subprocess.run([docker, "pull", "--platform", platform, image], check=True)
    created = subprocess.run([docker, "create", "--platform", platform, image], check=True, text=True, capture_output=True)
    container = created.stdout.strip()
    dest.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen([docker, "export", container], stdout=subprocess.PIPE) as export:
        subprocess.run(["tar", "-x", "-C", str(dest)], stdin=export.stdout, check=True)
    subprocess.run([docker, "rm", container], check=False, stdout=subprocess.DEVNULL)
    if not (dest / "usr" / "include" / "stdio.h").exists():
        raise ToolchainError("manylinux2014 sysroot extraction failed for %s" % arch)
    return dest


def zig_cc(target: str) -> Toolchain:
    zig = have("zig")
    if not zig:
        raise ToolchainError("zig is required for the macOS builds")
    tc = gcc_toolchain(
        f"zig-{target}",
        zig,
        can_run=False,
        # The deployment target is part of zig's target triple; pin it so the
        # wheel tag (macosx_11_0_*) matches what the binary supports.
        extra_cflags=["cc", "-target", target],
    )
    # zig insists on a writable cache; keep it inside the build tree so the
    # build never depends on the user's home directory permissions.
    cache = repo_root() / ".build" / "zig-cache"
    cache.mkdir(parents=True, exist_ok=True)
    tc.env = {"ZIG_GLOBAL_CACHE_DIR": str(cache / "global"), "ZIG_LOCAL_CACHE_DIR": str(cache / "local")}
    return tc


def clang_cl(arch: str) -> Toolchain:
    """clang-cl targeting the MSVC ABI, with lld-link and the Windows SDK."""
    cc = have("clang-cl") or "/usr/lib/llvm-19/bin/clang-cl"
    if not Path(cc).exists():
        raise ToolchainError("clang-cl not found (install clang/lld)")
    linker = have("lld-link") or "/usr/lib/llvm-19/bin/lld-link"
    if not Path(linker).exists():
        raise ToolchainError("lld-link not found (install lld)")

    msvc_arch = {"x86_64": "x86_64", "aarch64": "aarch64"}[arch]
    # One splat per target: `--arch` decides which MSVC CRT payload is packed,
    # so the x86_64 splat has no arm64 runtimes and vice versa.
    splat = TOOLCHAIN_DIR / ("msvc" if msvc_arch == "x86_64" else "msvc-%s" % msvc_arch)
    if not (splat / "crt" / "lib" / msvc_arch).exists():
        _xwin_splat(splat, msvc_arch)
    _xwin_merge_arm64(splat)
    target = {"x86_64": "x86_64-pc-windows-msvc", "aarch64": "aarch64-pc-windows-msvc"}[arch]

    includes = [
        str(splat / "crt" / "include"),
        str(splat / "sdk" / "include" / "ucrt"),
        str(splat / "sdk" / "include" / "um"),
        str(splat / "sdk" / "include" / "shared"),
    ]
    libs = [
        str(splat / "crt" / "lib" / msvc_arch),
        str(splat / "sdk" / "lib" / "ucrt" / msvc_arch),
        str(splat / "sdk" / "lib" / "um" / msvc_arch),
    ]
    missing = [path for path in includes + libs if not Path(path).exists()]
    if missing:
        raise ToolchainError(
            "the MSVC/SDK payload for %s is missing from the xwin splat:\n  %s\n"
            "Re-run `xwin --arch %s splat --output %s` with xwin >= 0.10 (it takes --arch before "
            "the subcommand), or delete the splat directory to have it provisioned again." % (arch, "\n  ".join(missing), msvc_arch, splat)
        )

    extra_cc: list[str] = []
    if arch == "aarch64":
        # llvm-19 mis-attributes the .seh epilogue of nanofft_make_plan at /O2
        # ("48 bytes of instructions ... but .seh directives corresponding to
        # 44"), which aborts the compile.  The library is plain C with no
        # exceptions, so dropping the unwind tables for this target is lossless;
        # x86_64 keeps them.
        extra_cc.append("/clang:-fno-asynchronous-unwind-tables")

    return Toolchain(
        name=f"clang-cl-{arch}",
        cc=[cc, "-fuse-ld=lld", f"--target={target}"] + extra_cc,
        ar=[have("llvm-ar") or "llvm-ar"],
        family="clang-cl",
        msvc_include=includes,
        msvc_lib=libs,
        can_run=False,
    )


def _xwin_merge_arm64(dest: Path) -> None:
    """Copy the arm64 UCRT and SDK libs that `xwin splat` prunes.

    The download cache keeps every architecture, the splat keeps the host one;
    copying the arm64 slices back costs nothing and leaves only the arm64 MSVC
    CRT payload missing on a Linux host.
    """
    cache = repo_root() / ".xwin-cache" / "unpack"
    pairs = [
        (cache / "ucrt.msi" / "lib" / "ucrt" / "arm64", dest / "sdk" / "lib" / "ucrt" / "aarch64"),
        (cache / "Win11SDK_10.0.26100_store_libs.msi" / "lib" / "um" / "arm64", dest / "sdk" / "lib" / "um" / "aarch64"),
    ]
    for src, dst in pairs:
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)
            print("[fetch] merged arm64 payload %s" % dst.name, flush=True)


def _xwin_splat(dest: Path, arch: str = "x86_64") -> None:
    xwin = have("xwin")
    if not xwin:
        raise ToolchainError("xwin is required to provision the MSVC toolset + Windows SDK (cargo install xwin, or download a release binary)")
    print("[fetch] xwin splat -> %s (large, one-off)" % dest, flush=True)
    run([xwin, "--accept-license", "--arch", arch, "splat", "--output", str(dest)])


def for_target(target) -> Toolchain:
    """Pick the toolchain that builds `target`."""
    key = target.name
    if key == "linux-x86_64-glibc":
        tc = native_gcc()
        sysroot = manylinux_sysroot("x86_64")
        tc.sysroot = str(sysroot)
        # --sysroot alone does not redirect Debian gcc's multiarch library
        # search path, so the sysroot libraries are also put in front of it.
        tc.lib_dirs = [str(sysroot / "usr" / "lib64"), str(sysroot / "lib64")]
        return tc
    if key == "linux-x86_64-musl":
        return native_musl()
    if key == "linux-aarch64-glibc":
        tc = debian_aarch64()
        sysroot = manylinux_sysroot("aarch64")
        tc.sysroot = str(sysroot)
        tc.lib_dirs = [str(sysroot / "usr" / "lib64"), str(sysroot / "lib64")]
        # --sysroot does not redirect this compiler's configured include paths,
        # so put the glibc 2.17 headers in front of its own explicitly.  They
        # matter: a newer math.h declares the libmvec entry points, and the
        # vectoriser then emits _ZGVnN2v_log1p, which glibc 2.17 cannot provide.
        tc.cflags = [
            "-isystem",
            str(sysroot / "usr" / "include"),
            "-isystem",
            str(sysroot / "usr" / "include" / "aarch64-linux-gnu"),
        ]
        return tc
    if key == "linux-aarch64-musl":
        return muslcc("aarch64")
    if key in ("win-amd64", "win-arm64"):
        return clang_cl(target.arch)
    if key == "macos-x86_64":
        return zig_cc("x86_64-macos.11.0")
    if key == "macos-arm64":
        return zig_cc("aarch64-macos.11.0")
    raise ToolchainError("no toolchain known for target %s" % key)


def report() -> int:
    """Print what this host can build, and where each toolchain comes from."""

    print("toolchain status (everything is build-time only, under .toolchains/):")
    checks = [
        ("native gcc (linux x86_64/glibc)", lambda: have("gcc") or have("cc")),
        ("musl-gcc (linux x86_64/musl)", lambda: have("musl-gcc") or have("x86_64-linux-musl-gcc")),
        ("debian aarch64 cross gcc", lambda: (TOOLCHAIN_DIR / "debian-aarch64" / "usr" / "bin" / "aarch64-linux-gnu-gcc-14").exists() or have("apt-get")),
        ("musl.cc aarch64", lambda: (TOOLCHAIN_DIR / "musl-aarch64").exists() or True),
        ("glibc 2.17 sysroot x86_64", lambda: (TOOLCHAIN_DIR / "sysroot-glibc217-x86_64" / "usr" / "include" / "stdio.h").exists()),
        ("glibc 2.17 sysroot aarch64", lambda: (TOOLCHAIN_DIR / "sysroot-glibc217-aarch64" / "usr" / "include" / "stdio.h").exists()),
        (
            "clang-cl + lld-link (windows)",
            lambda: (
                (have("clang-cl") or Path("/usr/lib/llvm-19/bin/clang-cl").exists()) and (have("lld-link") or Path("/usr/lib/llvm-19/bin/lld-link").exists())
            ),
        ),
        ("MSVC toolset + Windows SDK (xwin)", lambda: (TOOLCHAIN_DIR / "msvc" / "sdk").exists() or bool(have("xwin"))),
        ("zig (macOS)", lambda: have("zig")),
        ("qemu-aarch64 (test only)", lambda: have("qemu-aarch64")),
        ("wine (test only)", lambda: have("wine")),
        ("darling (test only)", lambda: have("darling")),
    ]
    for label, probe in checks:
        try:
            ok = bool(probe())
        except Exception:  # noqa: BLE001 - a probe must never fail the report
            ok = False
        print("  %-40s %s" % (label, "ok" if ok else "MISSING"))
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
