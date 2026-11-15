"""Verification of the built wheels.

Three levels, in increasing cost:

``python``   unzip the wheel into a temporary directory, put it on PYTHONPATH
             and import it in the current interpreter, printing the selected
             variant and a checksum of a small periodogram.  Repeating it with
             ``TOEPLITZ_LS_DISPATCH=<variant>`` exercises every shipped library
             and proves they agree.
``wine``     for a Windows wheel: run a small C harness under wine that
             LoadLibrary()s the packaged DLL, resolves the entry points and
             computes the same periodogram.  (A full Windows CPython is not
             required to prove the DLL is loadable and correct.)
``qemu``     for an aarch64 wheel: run the same harness (statically linked with
             the target toolchain) under qemu-aarch64.

Every level returns a checksum, so the same numbers can be compared across
architectures: they are computed from a fixed input and printed with enough
digits to catch a miscompiled kernel.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from . import variants as V

ROOT = V.repo_root()

# A deterministic periodogram: 64 observations, 3 harmonics, 128 frequencies.
HARNESS = r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same layout the Python wrapper passes: t (double), y/dy (T), M, f0, df, N. */
int tlsf_fastchi2(const double *, const float *, const float *, int, double, double, int, int, int, int, int, float *, float *);
int tls_fastchi2(const double *, const double *, const double *, int, double, double, int, int, int, int, int, double *, double *);

static void fill(double *t, double *y, int m) {
    for (int i = 0; i < m; ++i) {
        t[i] = 0.137 * i;
        y[i] = 1.0 + 0.3 * sin(2.0 * 3.141592653589793 * 0.7 * t[i]) + 0.1 * cos(2.0 * 3.141592653589793 * 1.4 * t[i]);
    }
}

int main(void) {
    enum { M = 64, N = 128, DEGREE = 3 };
    double t[M], y[M], dy[M];
    float yf[M], dyf[M], powerf[N], condf[N];
    double yd[M], powerd[N], condd[N];
    fill(t, y, M);
    for (int i = 0; i < M; ++i) { dy[i] = 0.1; yf[i] = (float)y[i]; yd[i] = y[i]; dyf[i] = 0.1f; }

    int stf = tlsf_fastchi2(t, yf, dyf, M, 0.1, 0.0125, N, DEGREE, 1, 1, 0, powerf, condf);
    int std = tls_fastchi2(t, yd, dy, M, 0.1, 0.0125, N, DEGREE, 1, 1, 0, powerd, condd);
    if (stf != 0 || std != 0) { printf("STATUS %d %d\n", stf, std); return 1; }

    double bestf = 0, bestd = 0, sumd = 0;
    for (int i = 0; i < N; ++i) {
        if (powerf[i] > bestf) bestf = powerf[i];
        if (powerd[i] > bestd) bestd = powerd[i];
        sumd += powerd[i];
    }
    printf("CHECKSUM %.17g %.17g %.17g\n", bestf, bestd, sumd);
    return 0;
}
"""


# Windows variant: load the packaged DLL explicitly, so no import library (and
# no knowledge of the export names at link time) is needed.
HARNESS_WIN = r"""
#include <math.h>
#include <stdio.h>
#include <windows.h>

typedef int (*fastchi2_f)(const double *, const float *, const float *, int, double, double, int, int, int, int, int, float *, float *);
typedef int (*fastchi2_d)(const double *, const double *, const double *, int, double, double, int, int, int, int, int, double *, double *);

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: harness <dll>\n"); return 2; }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { printf("LoadLibrary failed: %lu\n", GetLastError()); return 1; }
    fastchi2_f ff = (fastchi2_f)GetProcAddress(dll, "tlsf_fastchi2");
    fastchi2_d fd = (fastchi2_d)GetProcAddress(dll, "tls_fastchi2");
    if (!ff || !fd) { printf("GetProcAddress failed: %lu\n", GetLastError()); return 1; }

    enum { M = 64, N = 128, DEGREE = 3 };
    double t[M], y[M], dy[M];
    float yf[M], dyf[M], powerf[N], condf[N];
    double yd[M], powerd[N], condd[N];
    for (int i = 0; i < M; ++i) {
        t[i] = 0.137 * i;
        y[i] = 1.0 + 0.3 * sin(2.0 * 3.141592653589793 * 0.7 * t[i]) + 0.1 * cos(2.0 * 3.141592653589793 * 1.4 * t[i]);
        dy[i] = 0.1;
        yf[i] = (float)y[i];
        dyf[i] = 0.1f;
        yd[i] = y[i];
    }
    if (ff(t, yf, dyf, M, 0.1, 0.0125, N, DEGREE, 1, 1, 0, powerf, condf) != 0) { printf("float call failed\n"); return 1; }
    if (fd(t, yd, dy, M, 0.1, 0.0125, N, DEGREE, 1, 1, 0, powerd, condd) != 0) { printf("double call failed\n"); return 1; }
    double bestf = 0, bestd = 0, sumd = 0;
    for (int i = 0; i < N; ++i) {
        if (powerf[i] > bestf) bestf = powerf[i];
        if (powerd[i] > bestd) bestd = powerd[i];
        sumd += powerd[i];
    }
    printf("CHECKSUM %.17g %.17g %.17g\n", bestf, bestd, sumd);
    return 0;
}
"""


class CheckError(RuntimeError):
    pass


def _runner(runner, default):
    """The emulator command: a string is split so it can carry arguments."""
    if runner is None:
        return [default] if default else []
    if isinstance(runner, str):
        return shlex.split(runner)
    return list(runner)


def _build_harness(cc, flags, work: Path, *, static=False, libs=()):
    src = work / "harness.c"
    src.write_text(HARNESS)
    out = work / ("harness.exe" if "windows" in " ".join(cc) else "harness")
    # libs go after the source: Debian's gcc links with --as-needed, so a
    # -ltls that appeared before the object file would be dropped.
    cmd = [*cc, *flags, "-O2", str(src), "-o", str(out), *libs, "-lm"]
    if static:
        cmd.insert(1, "-static")
    subprocess.run([str(c) for c in cmd], check=True, text=True, capture_output=True)
    return out


def _parse(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("CHECKSUM"):
            return line.strip()
    raise CheckError("harness produced no checksum:\n" + output)


# ---------------------------------------------------------------------------
# Python-level check
# ---------------------------------------------------------------------------
def python_check(wheel: Path, *, variant=None, python=None) -> tuple[str, str]:
    python = python or sys.executable
    with tempfile.TemporaryDirectory(prefix="tls-wheel-") as tmp:
        site = Path(tmp)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(site)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(site)
        if variant:
            env["TOEPLITZ_LS_DISPATCH"] = variant
        else:
            env.pop("TOEPLITZ_LS_DISPATCH", None)
        script = (
            "import numpy as np, toeplitz_ls as tls\n"
            "print('VARIANT', tls.__dispatch__['name'], tls.__dispatch__['file'])\n"
            "t = 0.137*np.arange(64.0); y = 1 + 0.3*np.sin(2*np.pi*0.7*t) + 0.1*np.cos(2*np.pi*1.4*t)\n"
            "f, p = tls.tls.autopower(t, y, nterms=3, fmax=8)\n"
            "f2, p2 = tls.tlsf.autopower(t, y.astype(np.float32), nterms=3, fmax=8)\n"
            "print('CHECKSUM %.17g %.17g %.17g' % (np.nanmax(p), np.nansum(p), np.nanmax(p2)))\n"
        )
        result = subprocess.run([python, "-c", script], env=env, text=True, capture_output=True, cwd=site)
        if result.returncode != 0:
            raise CheckError("python import failed:\n" + result.stdout + result.stderr)
        name = ""
        for line in result.stdout.splitlines():
            if line.startswith("VARIANT"):
                name = line.split(None, 1)[1]
        return name, _parse(result.stdout)


# ---------------------------------------------------------------------------
# Native / emulated C harness
# ---------------------------------------------------------------------------
def library_harness(wheel: Path, target_name: str, *, runner=None) -> str:
    """Load the packaged library from a C harness and return its checksum."""
    target = V.TARGETS[target_name]
    from . import toolchains as T

    tc = T.for_target(target)
    with tempfile.TemporaryDirectory(prefix="tls-harness-") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(work / "site")
        libdir = work / "site" / "toeplitz_ls" / "_lib"

        manifest = json.loads((libdir / "manifest.json").read_text())
        baseline = manifest["variants"][manifest.get("baseline", 0)]["file"]

        if target.os == "windows":
            src = work / "harness.c"
            src.write_text(HARNESS_WIN)
            exe = work / "harness.exe"
            subprocess.run(
                [
                    *tc.cc,
                    *[f"-imsvc{p}" for p in tc.msvc_include],
                    str(src),
                    "/Fe" + str(exe),
                    "/MT",
                    "/link",
                    *[f"/LIBPATH:{p}" for p in tc.msvc_lib],
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            command = _runner(runner, "wine") + [str(exe), str(libdir / baseline)]
        else:
            flags = [f"-I{V.repo_root() / 'include'}"]
            # Cross targets must link against their own sysroot: the wheel's
            # glibc 2.17 floor comes from it, and without it the cross linker
            # resolves libm's linker script to a path that does not exist here.
            if tc.sysroot:
                flags.append("--sysroot=" + str(tc.sysroot))
                flags += [f"-L{path}" for path in tc.lib_dirs]
            # The harness is linked the way a client would: against the
            # baseline variant, found through a stable libtls.so name, and
            # loaded from the packaged copy at run time.
            shutil.copyfile(libdir / baseline, work / "libtls.so")
            flags += [f"-L{work}", f"-L{libdir}", "-Wl,-rpath," + str(libdir)]
            exe = _build_harness(tc.cc + list(tc.cflags), flags, work, static=False, libs=("-ltls",))
            command = _runner(runner, None) + [str(exe)]
            # Point the harness at the packaged library.
            for lib in libdir.glob("libtls-*.*"):
                subprocess.run(["cp", str(lib), str(work / lib.name)], check=True)
        env = dict(os.environ)
        # `work` carries the stable libtls.so name the harness was linked
        # against; `libdir` carries the packaged files under their own names.
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(work), str(libdir)])
        if target.os == "windows":
            # Keep the wine prefix inside the build tree: it is created on the
            # first run and must not depend on the user's home directory.
            prefix = ROOT / ".build" / "wine-prefix"
            prefix.mkdir(parents=True, exist_ok=True)
            env["WINEPREFIX"] = str(prefix)
            env["WINEDEBUG"] = "-all"
            # wine wants a writable runtime dir; /run/user is not writable in
            # a restricted sandbox, so keep its sockets in the prefix.
            env["XDG_RUNTIME_DIR"] = str(prefix / "runtime")
            (prefix / "runtime").mkdir(parents=True, exist_ok=True)
        result = subprocess.run([str(c) for c in command], text=True, capture_output=True, cwd=work, env=env)
        if result.returncode != 0:
            raise CheckError("harness failed (%s):\n%s\n%s" % (" ".join(map(str, command)), result.stdout, result.stderr))
        return _parse(result.stdout)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check built wheels.")
    parser.add_argument("--wheel")
    parser.add_argument("--target", choices=sorted(V.TARGETS))
    parser.add_argument("--all", action="store_true", help="check every wheel built into dist/ on this host")
    parser.add_argument("--all-variants", action="store_true", help="run the wheel once per shipped variant and compare")
    parser.add_argument("--harness", action="store_true", help="use the C harness instead of Python")
    parser.add_argument(
        "--runner",
        help="emulator for --harness, e.g. 'qemu-aarch64 -L .toolchains/sysroot-glibc217-aarch64'; "
        "defaults to wine for Windows and to running the harness directly otherwise",
    )
    args = parser.parse_args(argv)

    if args.all:
        return _check_all()
    if not args.wheel or not args.target:
        parser.error("--wheel and --target are required (or --all)")
    wheel = Path(args.wheel)
    target = V.TARGETS[args.target]
    failures = 0

    if args.harness:
        print(library_harness(wheel, args.target, runner=args.runner))
        return 0

    names = [v.name for v in target.variants] if args.all_variants else [None]
    baseline = None
    for name in names:
        try:
            selected, checksum = python_check(wheel, variant=name)
        except CheckError as exc:
            print("FAIL %-22s %s" % (name or "auto", exc))
            failures += 1
            continue
        if baseline is None:
            baseline = checksum
            print("ok   %-22s -> %s" % (name or "auto", selected))
        else:
            same = _compare(baseline, checksum)
            print("%s %-22s -> %s" % ("ok  " if same else "FAIL", name, selected))
            failures += 0 if same else 1
    return 1 if failures else 0


def _check_all() -> int:
    """Run the import check for every wheel in dist/ that this host can load."""
    from . import wheel as W

    version = W.read_version()
    failures = 0
    for name, target in V.TARGETS.items():
        path = W.DIST / ("toeplitz_ls-%s-py3-none-%s.whl" % (version, target.platform_tag))
        if not path.exists():
            print("skip %-22s (no wheel)" % name)
            continue
        if not _host_can_run(target):
            print("skip %-22s (no runner on this host; ABI checks are in dispatch.wheel --check)" % name)
            continue
        try:
            selected, checksum = python_check(path)
            print("ok   %-22s -> %s  %s" % (name, selected, checksum.split()[1] if len(checksum.split()) > 1 else ""))
        except CheckError as exc:
            print("FAIL %-22s %s" % (name, exc))
            failures += 1
    return 1 if failures else 0


def _host_can_run(target) -> bool:
    import platform as _p

    machine = _p.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64")
    system = {"linux": "linux", "windows": "windows", "darwin": "macos"}.get(_p.system().lower(), "")
    if system != target.os or machine != target.arch:
        return False
    return target.os != "linux" or target.libc == "glibc"


def _compare(a: str, b: str, rel=1e-6) -> bool:
    try:
        left = [float(x) for x in a.split()[1:]]
        right = [float(x) for x in b.split()[1:]]
    except ValueError:
        return False
    return all(abs(x - y) <= rel * max(1.0, abs(x)) for x, y in zip(left, right))


if __name__ == "__main__":
    raise SystemExit(main())
