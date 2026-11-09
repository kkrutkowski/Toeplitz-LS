# Release wheels: multiple dispatch across microarchitectures

The 1.0 wheels ship **one shared object per microarchitecture level** and let
the Python package pick the widest one the CPU can execute, once, at import.
There is no run-time dispatch inside the library: the chosen `.so` contains
only the code that will run, built with exactly its own `-march`.

```
toeplitz_ls/
    _dispatch.py          probes the CPU, reads _lib/manifest.json, loads one .so
    _lib/
        manifest.json     {"target": ..., "variants": [{"name", "file", "requires"}]}
        libtls-x86-64.so
        libtls-x86-64-v2.so
        ...
```

`TOEPLITZ_LS_DISPATCH=<variant>` forces a library (it must be supported by the
CPU; the import fails with an explanation otherwise) and
`TOEPLITZ_LS_DISPATCH=list` prints what the wheel contains and what this CPU
supports.

## The matrix

| wheel | toolchain | variants |
|---|---|---|
| `manylinux_2_17_x86_64`, `musllinux_1_2_x86_64` | gcc, glibc 2.17 sysroot / musl-gcc | `x86-64`, `x86-64-v2`, `x86-64-v2+avx`, `x86-64-v3`, `x86-64-v4` |
| `manylinux_2_17_aarch64`, `musllinux_1_2_aarch64` | Debian cross gcc / musl.cc gcc | `armv8-a`, `armv8.2-a+sve-256`, `armv8.2-a+sve-512` |
| `win_amd64` | clang-cl (MSVC ABI) + lld-link | `x86-64`, `x86-64-v2`, `x86-64-v2+avx`, `x86-64-v3`, `x86-64-v4` |
| `win_arm64` | clang-cl (MSVC ABI) + lld-link | `armv8-a`, `armv8.2-a+sve-256`, `armv8.2-a+sve-512` |
| `macosx_11_0_x86_64` | zig cc | `x86-64`, `x86-64-v2`, `x86-64-v2+avx`, `x86-64-v3`, `x86-64-v4` |
| `macosx_11_0_arm64` | zig cc | `armv8-a` |

The dispatch axis is the **vector width** (`VEC_BYTES`), because that is what
changes the FFT kernel's shape and therefore its calibration.  That is also why
macOS arm64 ships a single library: Apple silicon implements neither SVE nor
SVE2 and macOS exposes no SVE state, so there is nothing to dispatch to.  For
the same reason `dotprod`/`i8mm`/`bf16` are not separate variants on ARM — they
do not change the vector width.

SVE is special: a library compiled with `-msve-vector-bits=N` is only valid when
the hardware vector length is exactly N, so the dispatcher needs the length, not
just "SVE present".

Feature detection is *not* left to the OS summary.  The baseline library exports
`tls_cpu_features()`, which reads CPUID leaves 1/7/0x80000001 (accepting AVX
only when XCR0 enables the YMM state, and AVX-512 only when it enables the ZMM
state) or `getauxval(HWCAP)` on AArch64, and returns a bitmask the dispatcher
turns into feature names before applying the manifest.  Windows'
`IsProcessorFeaturePresent` and macOS' sysctl strings are still consulted, but
only as extra hints — an unverifiable feature stays off, which degrades to the
baseline instead of risking an illegal instruction.  Each variant's `requires`
is a conjunction of single features (Linux checks the full level, Windows and
macOS check what they can see), so `x86-64-v4` is only picked where AVX-512F,
BW, CD, DQ and VL are all present.  Linux reports it through `prctl(PR_SVE_GET_VL)`; Windows
only publishes a presence bit, so the dispatcher then calls the baseline
library's `tls_probe_sve_vector_length()` (a single `cntb`) to get the length
before choosing.

## Building

```sh
make toolchains     # show what this host can build, provision what is missing
make wheels         # build + ABI-check + package every target
make wheel-linux-x86_64-glibc        # or one target at a time
make wheels-check   # import every wheel this host can load
make sdist          # stage the source distribution (dist/toeplitz_ls-<v>.tar.gz)
```

Intermediate artifacts -- compiled objects, the staged wheel payloads, the
`.toolchains/` cache and the sdist staging tree -- live under `.build/`, which
is git-ignored.  The dot matters: a plain `build/` in the checkout shadows the
`build` package, so `python -m build` would stop working.

Everything the build needs lands in `.toolchains/` (git-ignored) and is only
needed at build time:

| dependency | used for | how it is provisioned |
|---|---|---|
| `gcc` | linux x86_64 glibc | system compiler |
| `musl-gcc` (musl-tools) | linux x86_64 musl | system package |
| Debian `gcc-14-aarch64-linux-gnu` | linux aarch64 glibc | `apt-get download` + `dpkg-deb -x`, no root |
| musl.cc cross toolchains | linux aarch64 musl | pinned tarball |
| manylinux2014 sysroot | GLIBC_2.17 for both Linux arches | extracted from the pypa image (docker, no run needed) |
| `clang-cl` + `lld-link` + xwin splat | windows | pinned xwin release binary, one splat per target arch (`xwin --arch <arch> splat`) |
| `zig` | macOS | system package |

The manylinux sysroot matters: a library linked against a *current* glibc
records `exp@GLIBC_2.29` and `lgamma@GLIBC_2.23`, which no `manylinux_2_17`
wheel may require.  Building against the CentOS 7 sysroot gives
`glibc <= 2.15`, and `dispatch.wheel` re-checks the maximum symbol version of
every shipped object before writing the wheel.

Windows is a genuine MSVC-ABI build: `clang-cl --target=<arch>-pc-windows-msvc`
with the MSVC toolset and Windows SDK from `xwin`, linked by `lld-link` as a
`/DLL` with `/MT` (static UCRT).  The resulting DLL imports nothing but
`KERNEL32.dll`, so it needs no redistributable and no MinGW runtime.  That is
only safe because the library never hands a heap pointer back to the caller and
never frees caller memory; keep that invariant.

Two sharp edges on the arm64 Windows target, both handled here:

* `xwin` selects the MSVC CRT payload with `--arch`, which is a *global* option
  (`xwin --arch aarch64 splat ...`), not a `splat` option.  The x86_64 splat
  therefore contains no arm64 runtimes; each target gets its own
  `.toolchains/msvc-<arch>`.
* llvm-19 mis-attributes the `.seh` epilogue of `nanofft_make_plan` at `/O2`
  ("48 bytes of instructions in range, but .seh directives corresponding to
  44") and aborts.  The arm64 TUs are compiled with
  `/clang:-fno-asynchronous-unwind-tables`; the library is C with no
  exceptions, so nothing is lost, and x86_64 keeps its unwind tables.

## Release check: is the wheel PyPI-clean?

The wheel writer derives its platform tag from the binaries, and the same
question can be asked independently with the tools PyPI users will run:

```sh
pip install auditwheel twine
auditwheel show dist/*manylinux*.whl     # "is consistent with manylinux_2_17_x86_64"
auditwheel show dist/*musllinux*.whl     # musl policy, needs a musl host to be exact
twine check dist/*.whl                   # metadata/long-description sanity
python -m dispatch.check --wheel dist/<wheel> --target <target>  # import + own PE/Mach-O/ELF checks
python -m dispatch.wheel --target <target> --check                 # ABI checks on the staged _lib only
```

`auditwheel` answers exactly the question a tag must answer: the highest
versioned symbol the wheel needs from libc/libm.  On this tree it reports
`GLIBC_2.15` on top of `GLIBC_2.2.5`/`GLIBC_2.14`, hence `manylinux_2_17`.
If you would rather not depend on a host sysroot, build the Linux wheels inside
the matching `quay.io/pypa/manylinux_2_28` / `quay.io/pypa/musllinux_1_2`
container (or on GitHub Actions runners, which is what `make wheels` is
designed to fit); the checks above are the same either way.

## The source distribution

`make sdist` writes `dist/toeplitz_ls-<version>.tar.gz`.  It is not a copy of
the repository: a staged tree is assembled under `.build/sdist/`, carrying the
Python package, `src/`, `include/`, `PKG-INFO` and a small PEP 517 backend
(`dispatch/sdist_backend.py`, shipped as `tls_source_build.py`).  All metadata
in it -- and in every wheel -- comes from `dispatch/project.py`, which reads
`toeplitz-ls/pyproject.toml`; there is no second copy of the version, the
license or the dependency list to fall out of sync.

The backend compiles the library at install time with the host `cc`, mirroring
the `generic` target of the Makefile but without `-march=native`, and produces
the layout `_dispatch.load()` already knows: a single `toeplitz_ls/tls.so`.
Its output is bit-identical to `make generic` (same flags, same sources).

This is the fallback, not the supported path: every platform the project
targets has a wheel, and pip only reaches for the sdist on something else
(ppc64le, riscv64, i686, ...) or when asked to `--no-binary`.  Windows source
builds are refused explicitly by the backend rather than half-supported.

## Generated inputs: scaling and the fusion-depth table

Two headers are *measured* rather than written by hand:

* `scaling.h` — the cost-model calibration used by `chi2per` to size its work
  arrays;
* `nanofft_depth_table.h` — the per-grid FFT fusion depth (see the top-level
  README).

Both depend on the vector width, so each is measured once per width on the
build host and copied to the variants that cannot execute there:

* a variant that can run here (native x86_64, and the canonical variant of each
  width) is measured directly;
* every other variant borrows the measurement of the same vector width, and the
  generated file records exactly which variant it came from;
* with nothing to borrow, `scaling.h` falls back to `scaling_generic.h` and the
  depth table to "all zero" (per-precision defaults).

## Verifying

```sh
python3 -m dispatch.wheel --target <target> --check      # ABI only
python3 -m dispatch.check  --wheel dist/<wheel> --target <target> --all-variants
python3 -m dispatch.check  --all                          # every runnable wheel
```

`--all-variants` loads the wheel once per shipped variant and compares a small
periodogram across them, which is the strongest check that the variants really
are interchangeable.

`--all` only covers wheels this host can load natively.  The other Linux
targets execute through the C harness and an emulator, which needs the target
sysroot the wheel was built against:

```sh
# glibc aarch64: its own loader lives in the manylinux sysroot
python3 -m dispatch.check --wheel dist/*manylinux_2_17_aarch64.whl \
    --target linux-aarch64-glibc --harness \
    --runner "qemu-aarch64 -L $PWD/.toolchains/sysroot-glibc217-aarch64"

# musl aarch64: the toolchain's ld-musl-aarch64.so.1 is an absolute symlink to
# the host's /lib/libc.so, so name the real libc as the loader instead
python3 -m dispatch.check --wheel dist/*musllinux_1_2_aarch64.whl \
    --target linux-aarch64-musl --harness \
    --runner "qemu-aarch64 $PWD/.toolchains/aarch64-linux-musl-cross/aarch64-linux-musl/lib/libc.so"

# musl x86_64 runs natively, like glibc x86_64
python3 -m dispatch.check --wheel dist/*musllinux_1_2_x86_64.whl \
    --target linux-x86_64-musl --harness
```

Windows goes through wine (`--target win-amd64 --harness`, no `--runner`).  Wine
puts its server socket under `/run/user/<uid>/wine` regardless of
`$XDG_RUNTIME_DIR`, so a sandbox that mounts `/run/user` read-only cannot run
it; nothing else is needed.

macOS runs under [Darling](https://www.darlinghq.org) with a `dlopen`-based
harness, since the packaged object is a dylib rather than something to link
against:

```sh
zig cc -target x86_64-macos.11.0 -O2 tests/harness_dlopen.c -o .build/mac-harness/harness
darling shell
# Inside the Darling shell, with /path/to/Toeplitz-LS the absolute host path:
/Volumes/SystemRoot/path/to/Toeplitz-LS/.build/mac-harness/harness \
    "/Volumes/SystemRoot/path/to/Toeplitz-LS/.build/mac-harness/site/toeplitz_ls/_lib/libtls-x86-64.so"
```

Use `darling shell` for every Darling test.  Bare `darling` and `darling exec`
are much slower and can appear to hang; do not use them.  The executable must
be named by its *host* path, while arguments are resolved inside the container,
where the host root appears at `/Volumes/SystemRoot`.  Darling is x86_64 only,
so the arm64 macOS wheel and both arm64 Windows DLLs can only be ABI-checked
here.

### What has actually been executed

Each library below is loaded through the C harness and computes the same
periodogram from the same input (64 observations, 128 frequencies, three
harmonics).  The exact float32 peak and float64 sum are printed by
`dispatch.check`; the release gate is cross-variant agreement, not a hard-coded
number, because the standard normalization now zeros invalid R^2 bins.

| wheel | variants run | runner |
|---|---|---|
| manylinux x86_64 | all 5 | native |
| musllinux x86_64 | all 5 | native |
| manylinux aarch64 | all 3 (SVE at VL 32/64) | qemu-aarch64 |
| musllinux aarch64 | all 3 (SVE at VL 32/64) | qemu-aarch64 |
| win_amd64 | all 5 | wine 10 |
| macos x86_64 | all 5 | Darling |
| win_arm64 | none | -- |
| macos arm64 | none | -- |

26 of the 30 shipped libraries can be executed here.  The spread is the
expected one: float32 peaks agree to ~1e-6 across vector widths and compilers,
float64 sums to ~1e-15, and glibc/musl builds of the same variant are
bit-identical, which is what a libc-independent, vector-width-driven kernel
should look like.

The four that were only ABI-checked are the arm64 Windows DLLs and the arm64
macOS dylib: neither wine nor Darling can run those ABIs.  Their *code* is
covered -- the armv8-a variant is the same object as the Linux armv8-a build
executed above, and the SVE-256/512 variants likewise -- so what remains
unverified for them is the platform layer (PE/Mach-O loading on arm64, ad-hoc
code signing) rather than the numerics.

## Generated inputs: scaling and the fusion-depth table

Two headers are *measured* rather than written by hand:

* `scaling.h` — the cost-model calibration used by `chi2per` to size its work
  arrays;
* `nanofft_depth_table.h` — the per-grid FFT fusion depth (see the top-level
  README).

Both depend on the vector width, so each is measured once per width on the
build host and copied to the variants that cannot execute there:

* a variant that can run here (native x86_64, and the canonical variant of each
  width) is measured directly;
* every other variant borrows the measurement of the same vector width, and the
  generated file records exactly which variant it came from;
* with nothing to borrow, `scaling.h` falls back to `scaling_generic.h` and the
  depth table to "all zero" (per-precision defaults).

## Verifying

```sh
python3 -m dispatch.wheel --target <target> --check      # ABI only
python3 -m dispatch.check  --wheel dist/<wheel> --target <target> --all-variants
python3 -m dispatch.check  --all                          # every runnable wheel
```

`--all-variants` loads the wheel once per shipped variant and compares a small
periodogram across them, which is the strongest check that the variants really
are interchangeable.

`--all` only covers wheels this host can load natively.  The other Linux
targets execute through the C harness and an emulator, which needs the target
sysroot the wheel was built against:

```sh
# glibc aarch64: its own loader lives in the manylinux sysroot
python3 -m dispatch.check --wheel dist/*manylinux_2_17_aarch64.whl \
    --target linux-aarch64-glibc --harness \
    --runner "qemu-aarch64 -L $PWD/.toolchains/sysroot-glibc217-aarch64"

# musl aarch64: the toolchain's ld-musl-aarch64.so.1 is an absolute symlink to
# the host's /lib/libc.so, so name the real libc as the loader instead
python3 -m dispatch.check --wheel dist/*musllinux_1_2_aarch64.whl \
    --target linux-aarch64-musl --harness \
    --runner "qemu-aarch64 $PWD/.toolchains/aarch64-linux-musl-cross/aarch64-linux-musl/lib/libc.so"

# musl x86_64 runs natively, like glibc x86_64
python3 -m dispatch.check --wheel dist/*musllinux_1_2_x86_64.whl \
    --target linux-x86_64-musl --harness
```

Windows goes through wine (`--target win-amd64 --harness`, no `--runner`).  Wine
puts its server socket under `/run/user/<uid>/wine` regardless of
`$XDG_RUNTIME_DIR`, so a sandbox that mounts `/run/user` read-only cannot run
it; nothing else is needed.

macOS runs under [Darling](https://www.darlinghq.org) with a `dlopen`-based
harness, since the packaged object is a dylib rather than something to link
against:

```sh
zig cc -target x86_64-macos.11.0 -O2 tests/harness_dlopen.c -o .build/mac-harness/harness
darling shell
# Inside the Darling shell, with /path/to/Toeplitz-LS the absolute host path:
/Volumes/SystemRoot/path/to/Toeplitz-LS/.build/mac-harness/harness \
    "/Volumes/SystemRoot/path/to/Toeplitz-LS/.build/mac-harness/site/toeplitz_ls/_lib/libtls-x86-64.so"
```

Use `darling shell` for every Darling test.  Bare `darling` and `darling exec`
are much slower and can appear to hang; do not use them.  The executable must
be named by its *host* path, while arguments are resolved inside the container,
where the host root appears at `/Volumes/SystemRoot`.  Darling is x86_64 only,
so the arm64 macOS wheel and both arm64 Windows DLLs can only be ABI-checked
here.

### What has actually been executed

Every library below was loaded through the C harness and produced the same
periodogram from the same input.  The exact float32 peak and float64 sum are
printed by `dispatch.check`; the release gate is cross-variant agreement, not a
hard-coded number, because the standard normalization now zeros invalid R^2
bins.  Not executed on any machine so far: the arm64 Windows DLLs and the arm64
macOS dylib (no emulator runs those ABIs), which is why their ABI checks matter
more.

The ABI checks are what earn the platform tag:

* glibc: maximum `GLIBC_*` symbol version ≤ 2.17, `NEEDED` only `libc.so.6` and
  `libm.so.6`;
* musl: `NEEDED` exactly `libc.musl-<arch>.so.1` (this is why the link line
  passes an explicitly named copy of the toolchain's libc);
* Windows: PE machine type, undecorated exports, imports restricted to
  `KERNEL32`/`api-ms-win-core-*`;
* macOS: `minos >= 11.0`, an ad-hoc code signature on arm64, and only
  `/usr/lib/libSystem.B.dylib`.

Emulated runs (aarch64 under `qemu-aarch64`, Windows DLLs under `wine`) exercise
the same sources and the same variant flags as the wheel; a target with no
runner on the build host is still shipped, gated by the ABI checks plus the
per-architecture execution of the identical sources elsewhere.

## Adding a variant

1. Add a `Variant(...)` to the right tuple in `dispatch/variants.py` (name,
   `-march` flags, and the feature groups the dispatcher must check).
2. That is all: the builder compiles it, the manifest lists it, the wheel ships
   it, and `--all-variants` starts checking it.  If the new variant has a
   different vector width, add its canonical name to `_CANONICAL` in
   `dispatch/build.py` so that width gets measured rather than borrowed.
