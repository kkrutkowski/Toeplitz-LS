# Handoff: Hardware-Generalized Modular SYCL Type-1 NuFFT (`PSWF21` & `PSWF43`) + DIF Radix-2 FFT (AdaptiveCpp 25.10)

Date: 2026-11-25
Working dir: `/home/krutkowski/Pulpit/Toeplitz-LS/Toeplitz-LS`
Scope: `src/miscellaneous/` + `offload/sycl/`

**STATE: BUILD GREEN, REFACTORED, AUDITED & FULLY VERIFIED.** All hardcoded NVIDIA/Blackwell performance-scaling constants outside the FFT have been eliminated (while keeping all 20 6-stage `float128` minimax piecewise `PSWF0` polynomial tables hardcoded), the pipeline outside the FFT has been redesigned for **2.3×–4.1× higher NuFFT-1 throughput**, and the build has been refactored into **21 modular translation units** isolating hardware probes, lazy bit-reversal, small/large FFTs, NuFFT-1 aux/spread/deconv execution modules, and a zero-kernel pure-host NuFFT-1 planner.

---

## 1. Architectural Highlights

### 1.1 Zero Hardcoded Performance-Scaling Constants Outside the FFT
- Every runtime execution parameter in NuFFT-1 is now derived dynamically from `sycl_fft::DeviceFftConfig`, `sycl_fft::BankModel`, and persistent shortlist autotuning in `sycl_fft::AutotuneCache` (honoring `AutotuneMode::Auto | Off | Force` and explicit overrides via `Nufft1LaunchConfig`):
  - **Spreading (`SpreadConfig`)**: `t_j = next_pow2(w)` (always $\ge w$, eliminating the $w > t_j$ cliff at FP32 $w=9$ and FP64 $w=17,18$), multi-warp `wg_size` from `effective_sg_size` & `max_work_group_size`, `win_elems` from `local_mem_size` & `has_native_local_mem`, bank-conflict-free `grid_stride = win_elems + grid_pad` from `BankModel`, `tile_m` from local-memory budget, and `transpose_tile` / `transpose_stride` from `find_min_conflict_stride2(...)`.
  - **Deconvolution (`DeconvConfig`)**: `wg_i` from `sub_group_size` & `max_work_group_size`, and `batch_tile` derived from `max_compute_units * target_waves` vs `nout` (amortizing `eval_pswf_deconv` across `16..64+` packed transforms with zero global LUT reads).
  - **Unified L2-Aware FFT $\leftrightarrow$ NuFFT Transition**: `chunk_bs_` is derived from `dev_cfg.compute_l2_chunk_bytes(buf_mult * nfft * sizeof(ComplexT), ...)` coordinated with `fft_plan_.chunk_bs()`; `packed_stride_` aligns to `max(sub_group_size, cache_line_size / sizeof(RealT))`; `d_grid_T_*` is allocated **only** when collaborative fallback is selected (zero dead VRAM on GPU SMEM paths); `BitRevPlan` inside `DifFftPlan` is constructed lazily so `Nufft1Plan` / `execute_raw()` pays zero bit-reversal setup cost.

### 1.2 NuFFT-1 Pipeline Optimizations
- **Windowed Bucketed Multi-Warp SMEM Spreading (`launch_spread_smem_win<W>`)**:
  - Multi-warp work-groups (`wg_size = 64..256`, `b_per_wg = 4..32` packed transforms) eliminate the previous 1-warp/SM (2% occupancy) bottleneck and amortize on-the-fly piecewise polynomial evaluation 4×–16× across transforms.
  - One-time $O(M)$ GPU window bucketing (`launch_build_window_buckets` inside `set_points()`) ensures each grid window scans only its `~M / num_wins` points and scales cleanly to arbitrary $N_{\text{fft}}$ (`1024..16384+`, including Two-Pass FFT sizes) with zero `d_grid_T` global transpose buffer.
  - All work-group lanes cooperatively load `d_y` and apply phase shift `d_shift` in Phase 1 with a single barrier before Phase 2; `BankModel`-derived `grid_stride` removes inter-transform SMEM bank conflicts.
- **DF Mixed-Precision Horner Tail (`nufft/pswf_coeffs.hpp`)**:
  - Rapidly decaying high-degree terms (`m >= 4` in spreading, `m >= 5` even / `m >= 4` odd in deconvolution, $|c_m|/\text{peak} \le 1.3 \times 10^{-6}$) execute in FP32 hardware FMA before promoting to `df_t`, cutting DF spread FMAs from `278/296/314` $\to$ `230/244/260` and DF deconv FMAs from `208/222/234` $\to$ `180/192/202` with zero accuracy loss.

---

## 2. Modular Directory Hierarchy, 21-TU Layout, ELF/HCF Isolation & JIT Times

### 2.1 Submodule Tree & Build System (`Makefile` & `tests/Makefile`)
`offload/sycl/` is organized as a self-contained offload submodule ready for `toeplitz_ls.offload.sycl` (compiled into `build/tls_sycl.so` with `-fPIC -fvisibility=hidden`, exposing only `TLS_SYCL_API` `extern "C"` symbols from `impl.cpp`):

```text
offload/sycl/
├── Makefile                 # Release build -> build/tls_sycl.so (objects in build/obj/; excludes tests/)
├── impl.cpp                 # Public extern "C" ABI entrypoints (TLS_SYCL_API stub; future Python/C UI)
├── common/                  # Shared SYCL math, precision macros, layouts, device probes & AutotuneCache
│   ├── math.hpp             # Shared device-inline arithmetic/trig primitives
│   ├── precision.hpp        # Re-includable precision macro engine (used by kernel .cpp TUs)
│   ├── layout.hpp           # Shared constexpr shared-memory bank-conflict solver
│   ├── device.hpp           # Declarations only (harmless to #include with zero -D flags)
│   └── device.cpp           # 1 TU: DeviceFftConfig, AutotuneCache & 10 probe kernels (44.3 KiB HCF)
├── bitrev/                  # Deduplicated 4B/8B/16B bit-reversal permutation (lazily constructed)
│   ├── bitrev.hpp           # Declarations only (opaque BitRevLutState, BitRevEngine/BitRevPlan)
│   └── bitrev.cpp           # 1 TU: BitRevLutState, 6 kernel strategies & BitRevEngine (175.4 KiB HCF)
├── fft/                     # Quadruple-precision DIF radix-2 FFT (FP16 / FP32 / FP64 / DF)
│   ├── dif.hpp              # Declarations only (DifFftPlanh/f//df<Sign> + extern templates)
│   └── dif.cpp              # Compiled 8x via -D (prec x small/large; DF default) -> 8 HCF modules
├── nufft/                   # Type-1 NuFFT (PSWF21 & PSWF43, minimax piecewise polynomials, GL-free deconv)
│   ├── pswf_coeffs.hpp      # Generated by src/miscellaneous/pswf_ppoly_remez_lut.cpp (spread/deconv only)
│   ├── nufft1.hpp           # Declarations only (Nufft1PlanBase + launch_* decls; no pswf_coeffs include)
│   ├── auxiliary.cpp        # 1 TU: x-axis gridding, O(M) window buckets, size-dedup transpose (125.1 KiB HCF)
│   ├── spread.cpp           # Compiled 4x via -D (DF default) -> 4 spreading HCF modules (111.8–432.9 KiB)
│   ├── deconv.cpp           # Compiled 4x via -D (DF default) -> 4 deconv HCF modules (43.8–207.2 KiB)
│   └── plan.cpp             # 1 TU: pure-host orchestration & autotuning (0 kernels, 2.2 KiB HCF)
├── linalg/                  # Reserved for upcoming SYCL Levinson recursion solver
│   └── .gitkeep
├── tests/                   # Standalone verification & benchmark harness (not built by release `make`)
│   ├── Makefile             # Reuses ../build/obj/ via `make -C .. objs` -> tests/build/sycl_bench
│   └── main.cpp             # 1 TU: 0 probe/FFT/NuFFT kernels
└── build/                   # Gitignored release build artifacts (build/obj/<subdir>/*.o, build/tls_sycl.so)
```

- **Release shared library build** (from `offload/sycl/`): `make -j$(nproc)` $\to$ `build/tls_sycl.so` (verifiably exports only `TLS_SYCL_API` symbols; `dlopen` tested).
- **Test & benchmark build** (from `offload/sycl/tests/`): `make -j$(nproc)` $\to$ `tests/build/sycl_bench` (reuses `../build/obj/` with zero double-compilation of SYCL TUs; supports `make run`, `make run-cpu`, `make run-gpu`).

### 2.2 Build, Binary Size & Uncontended Cold/Warm JIT Comparison

| Metric | Before (10 TUs) | After (21 TUs) | Delta / Notes |
|---|---:|---:|---|
| Clean `-j16` parallel build | `~19.33 s` (`-j8`) | **`16.41 s`** (`-j16`) / `21.69 s` (`-j8`) | Faster max-parallel build |
| Incremental rebuild (`nufft/deconv_fp32.o`) | `~6.0 s` | **`2.72 s`** | Decoupled deconv TU + link |
| Incremental rebuild (`nufft/plan.cpp`) | `~6.0 s` | **`6.27 s`** | Zero GPU kernels recompiled |
| Sum of `.o` files / Final binary | `27.6 MiB` / `18.39 MiB` | **`26.81 MiB`** / `18.66 MiB` | Smaller total `.o` size |
| NuFFT-1 Cold / Warm (`FP16`, `Nfft=1024..4096`) | `4.441 s` / `0.404 s` | **`4.372 s`** / `0.406 s` | Slightly faster cold JIT |
| NuFFT-1 Cold / Warm (`FP32`, `Nfft=1024..4096`) | `1.555 s` / `0.412 s` | **`1.543 s`** / **`0.404 s`** | Slightly faster cold & warm |
| NuFFT-1 Cold / Warm (`FP64`, `Nfft=1024..4096`) | `2.319 s` / `0.832 s` | **`2.114 s`** / **`0.568 s`** | **-9% cold, -32% warm** |
| NuFFT-1 Cold / Warm (`DF`, `Nfft=1024..4096`) | `5.616 s` / `0.583 s` | **`5.383 s`** / **`0.432 s`** | **-4% cold, -26% warm** |

### 2.3 NVPTX PTX Audit (`sm_120`, all 62 cached NuFFT-1 kernels)
- Every single refactored NuFFT-1 kernel (`launch_spread_smem_win_impl*`, `launch_spread_collab_impl*`, `launch_deconv_emit_impl*`, `launch_build_window_buckets`, `launch_transpose_and_zero_*`) achieves **`ld.local = 0`, `st.local = 0`, `call.uni = 0`, `cvt.f32.f16 = 0`**, with FP16 spreading executing pure packed `fma.rn.f16x2` (`14 / 17` ops, `fma.f32 = 0`).

---

## 3. Accuracy & Portability Verification (`M = 256, N = 512, bs = 5, x in [-500, 500], df = 0.0137`)
- **CUDA (`sm_120`) Windowed SMEM Path**: **100% PASS** across all 4 precisions, all widths, `PSWF21` & `PSWF43`, `freq_factor in {1, 2}`, `SlotVsNatDiff = 0.000e+00`.
- **CUDA (`sm_120`) `--force-collab-spread` Fallback**: **100% PASS**, `SlotVsNatDiff = 0.000e+00`.
- **OpenMP CPU (`ACPP_VISIBILITY_MASK="omp"`)**: Automatically selects `collab_wg64` (zero emulated-SMEM misuse), **100% PASS**, `SlotVsNatDiff = 0.000e+00`.
- **FFT + Lazy BitRev Regression (`logN = 10..14`)**: **100% PASS** across all 4 precisions.

---

## 4. Warm CUDA Natural-to-Bit-Reversed (`SlotBitReversed`) Throughput (`M = 512, ntest = 20, NO BitRev`)

| Precision | Mode & $w$ | $N=512$ (`bs=2048`) Before $\to$ After | $N=2048$ (`bs=512`) Before $\to$ After | Peak Eff GFLOP/s (Before $\to$ After) | Speedup |
|---|---|---:|---:|---:|---:|
| **FP16 (2×FP16)** | `PSWF21 w=4..5` | `6.83M/s` $\to$ **`16.95M/s`** | `2.02M/s` $\to$ **`5.93M/s`** | `532.8` $\to$ **`1,561.5`** | **2.48× – 2.93×** |
| **FP16 (2×FP16)** | `PSWF43 w=4..5` | `6.78M/s` $\to$ **`16.93M/s`** | `1.99M/s` $\to$ **`5.81M/s`** | `528.4` $\to$ **`1,541.0`** | **2.50× – 2.92×** |
| **FP32** | `PSWF21 w=8` | `4.51M/s` $\to$ **`10.42M/s`** | `1.13M/s` $\to$ **`3.91M/s`** | `323.6` $\to$ **`1,051.9`** | **2.31× – 3.45×** |
| **FP32** | `PSWF21 w=9` | `2.87M/s` $\to$ **`8.47M/s`** | `0.92M/s` $\to$ **`3.62M/s`** | `248.5` $\to$ **`982.9`** | **2.95× – 3.96×** |
| **FP32** | `PSWF43 w=8..9` | `2.68..4.15M/s` $\to$ **`8.26..10.19M/s`** | `0.90..1.11M/s` $\to$ **`3.62..3.85M/s`** | `300.3` $\to$ **`1,043.4`** | **2.46× – 4.01×** |
| **DF (2×FP32)** | `PSWF21 w=14..16` | `1.05..1.12M/s` $\to$ **`3.27..3.30M/s`** | `0.32..0.33M/s` $\to$ **`1.33..1.37M/s`** | `98.4` $\to$ **`386.5`** | **2.93× – 4.25×** |
| **DF (2×FP32)** | `PSWF43 w=14..16` | `1.04..1.10M/s` $\to$ **`3.17..3.23M/s`** | `0.32..0.33M/s` $\to$ **`1.26..1.29M/s`** | `97.8` $\to$ **`369.1`** | **2.87× – 4.07×** |
| **FP64** | `PSWF21 w=16..18` | `0.41..0.55M/s` $\to$ **`1.05..1.28M/s`** | `0.14..0.15M/s` $\to$ **`0.35..0.38M/s`** | `48.4` $\to$ **`112.6`** | **2.33× – 2.55×** |
| **FP64** | `PSWF43 w=16..18` | `0.41..0.53M/s` $\to$ **`1.04..1.27M/s`** | `0.14..0.15M/s` $\to$ **`0.35..0.37M/s`** | `47.4` $\to$ **`112.3`** | **2.37× – 2.56×** |

- **Large Two-Pass FFT NuFFT-1 (`FP32`, `N = 4096..8192`, `Nfft = 8192..16384`)**: **`942.2 – 967.2 Eff GFLOP/s`** at $N_{\text{fft}} = 8192$ and **`896.0 – 917.4 Eff GFLOP/s`** at $N_{\text{fft}} = 16384$ (Two-Pass FFT transition, `smem_wg128/256_win128`, zero `d_grid_T` allocation).
- **DF vs. Native FP64 NuFFT-1 Speedup**: **`3.3× – 3.7×`** (up from `1.77× – 1.89×`).
