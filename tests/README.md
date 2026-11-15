# tests

Correctness gates and the interleaved benchmark harness for the fast path.

Everything here builds the library sources directly (three precision passes,
`-march=native`, the generic cost calibration from
`include/scaling_generic.h`) into `build/`, so it exercises exactly the code
that ships while staying independent of the slow `make native` calibration.

```console
make -C tests all      # build every checker and benchmark
make -C tests check    # correctness gates
make -C tests bench    # the benches on their own
make -C tests fuse     # fusion depth / order sweep (bench_fuse)
R=5 REPS=5 sh tests/bench.sh   # gates + interleaved baseline comparison
```

The 256-bit profile is built in a second tree, so both microarchitectures can be
compared without touching the default build:

```console
make -C tests BUILD=build-v3 \
     CFLAGS="-std=gnu11 -O3 -ffast-math -march=x86-64-v3 -mtune=native -Wall -Wextra" \
     build-v3/check_nanofft build-v3/bench_fuse
```

`bench.sh` builds a pristine baseline once as a `git worktree` of `main`
(default `../baseline`, override with `BASE=/path`), compiles the same
`dump_periodogram.c` / `bench_public.c` against it, and then:

1. runs `check_nanofft` and `check_periodogram`,
2. diffs the `DUMP` summaries of both builds (the numeric "did the
   restructuring change anything" gate),
3. interleaves the two binaries round by round (order rotated every round),
4. interleaves the algorithm variants inside one process,
5. runs the FFT microbenchmark over the fusion depths.

Every reported time is the best of `R` rounds of the best of `REPS` internal
repetitions, pinned with `taskset -c $CPU` (default 2). The run-to-run
comparison noise at these workload sizes is around 2-3%, so single-percent
differences are not meaningful.

## check_nanofft

- permutation matches `reverse_bits` for `2^6..2^14`, is an involution, and the
  scalar and vectorized tile paths agree bit for bit,
- `execute == execute_raw + permute`, and the 1-D permutation equals the real
  half of the 2-D one,
- every fusion depth reproduces a double-precision naive DFT (float error
  ~2e-7), and all depths agree with each other to float roundoff,
- both fusion orders reproduce the un-fused transform on every supported grid,
  and are bit-identical where the stage count divides by the depth (the two
  orders then describe the same schedule),
- `make_plan` rejects every grid below `2 * VEC_LEN`, in a child process so the
  expected `exit()` does not take the checks with it, and the smallest accepted
  grid still matches the naive DFT,
- same for double and double-double (double-double also runs the fused
  kernels).

## check_periodogram

End-to-end over `nterms = 1, 2, 3, 8`, PSWF43/PSWF21/LRA, levinson/zohar/
bareiss/ldlt/svd, every normalization, a grid that is not a multiple of the
plan block, and the three precisions:

- deferring the bit reversal changes no per-bin value (same NaN masks, same
  `cond`), which is the strong proof that it is only a permutation,
- the fused depths agree with depth 1 to float roundoff,
- IHS at `d = 1` stays in frequency order, IHS at `d >= 2` takes the deferred
  path, and both are unchanged,
- double-double agrees to double-double roundoff.

## bench_fft

In-process interleaved microbenchmark of `execute_raw` (butterflies),
`permute`, and the full transform for `2^min_log .. 2^max_log`, with the fusion
depths and the scalar vs vector COBRAVO in the rotation. Example
(`./build/bench_fft 5 7 10 16`) on a Ryzen 9 7945HX, ns per element of the full
transform (butterflies + permutation), best cell of each row in bold:

| n | fuse=1 | fuse=2 | fuse=3 | fuse=4 | scalar perm |
|---|---|---|---|---|---|
| 2^10 | 1.184 | 1.085 | **1.007** | 1.017 | 1.056 |
| 2^11 | 1.169 | 1.115 | 1.056 | **1.051** | 1.081 |
| 2^12 | 1.345 | 1.265 | 1.186 | **1.184** | 1.223 |
| 2^13 | 1.624 | **1.493** | 1.627 | 2.084 | 1.589 |
| 2^14 | 1.877 | **1.697** | 1.746 | 1.690 | 2.466 |
| 2^16 | 2.141 | **1.963** | 2.020 | 2.014 | 2.718 |

Fusion buys 10-30% over no fusion on the butterflies; between depths 2, 3 and 4
the spread is a few percent. This bench keeps one plan per variant alive for
the whole run, so the heap placement of each variant's twiddle table is fixed
and a single cell can be off by 15% (the `fuse=4` entry at 2^13 is such a
case) - use `bench_fuse` for depth and order decisions, it controls for that.
The vectorized COBRAVO permutation is 1.4-1.6x the scalar one everywhere.

## check_vecmath

Accuracy gate for the library's own vector `sqrt`/`exp`
(`include/nanofft_vecmath.h`). The vector exponential is a Remez polynomial
rather than a call into the platform's vector math library, because libmvec
exists only in glibc and the same sources would otherwise link differently on
musl, macOS and Windows. `./build/check_vecmath` compares it against libm over
the whole finite range of both precisions (1.4 million values), in ulps of the
target type:

- normal results in the range the library actually uses (`|x| <= 40` for
  double, the PSWF range `[-225, 0]` for float) are bounded relative to libm;
- denormal results are judged relatively, since rounding them correctly needs
  more internal precision than a double carries;
- saturation matches `exp` exactly: `exp(-inf) = 0`, `exp(+inf) = +inf`, NaN
  propagates, and the finite overflow/underflow ends round to inf/0;
- `vec_sqrt`/`vecf_sqrt` are checked against the hardware square root.

Fast-math reassociation is left enabled on purpose (it is what makes the
polynomial cheap); the test pins the bound rather than forbidding it. What the
statistics see is covered by `check_periodogram` and the Python suite, whose
tolerances are far above these differences.

## bench_fuse

The fusion depth and pass order, measured so that the *layout* cannot decide the
answer (`./build/bench_fuse 64 8 5 16`):

- variants are `{unfused, depth 2..MAX in both orders, the shipped default}`,
  both vectorized precisions, every power-of-two grid from the smallest
  `make_plan` accepts to 2^16,
- every variant gets the same buffers and a plan allocated from the same
  recycled heap block, and each round nudges the heap and shifts the input by
  one more 64-byte line, so a full sweep visits all 64 relative 4 KiB offsets
  between signal and twiddle table,
- the reported ns/element is the **median over those 64 layouts** (the mean
  would track the worst page, the best would track the luckiest), and every
  variant is checked against the un-fused reference,
- `self-check` measures the shipped default against the same schedule spelled
  out as an explicit depth: the same code twice, which bounds the measurement
  noise (~0.2%).

The same binary carries the tuner used by the release build and by
`make native`:

```console
./build/bench_fuse --tune --emit <dir>/nanofft_depth_table.h --note "<variant>"
```

`--tune` sweeps the depths compiled into the build (16 layouts, 8 reps by
default), picks the median-fastest depth per power-of-two grid, prefers the
shallowest depth within 2% of the best (the differences around the optimum are
at the noise floor), and writes the header that `nanofft.c` includes when
`NANOFFT_DEPTH_TABLE` is defined. The library then uses that table whenever the
caller asks for the automatic depth; `tls_tuning_set_fft_fuse()` still wins over
it, and `make TUNE=0 native` keeps the per-precision defaults.

Ryzen 9 7945HX (Zen 4), gcc 14.2, `-O3 -ffast-math`, median ns/element of
`execute_raw` over the 64 layouts:

`-march=native` (AVX-512, 32 vector registers; float default depth 3):

| grid | unfused | d2 end | d2 front | d3 end | d3 front | d4 end | d4 front |
|---|---|---|---|---|---|---|---|
| 2^10 | 0.782 | 0.685 | 0.685 | **0.655** | 0.655 | 0.655 | 0.655 |
| 2^11 | 0.841 | 0.753 | 0.778 | 0.729 | **0.705** | 0.729 | 0.709 |
| 2^12 | 0.917 | 0.785 | 0.785 | 0.749 | **0.739** | 0.749 | 0.739 |
| 2^13 | 0.994 | 0.895 | 0.892 | 0.893 | 0.894 | 0.893 | 0.894 |

`-march=x86-64-v3` (AVX2, 16 vector registers; float default depth 2):

| grid | unfused | d2 end | d2 front | d3 end | d3 front | d4 end | d4 front |
|---|---|---|---|---|---|---|---|
| 2^10 | 0.832 | 0.763 | 0.772 | **0.754** | 0.772 | 0.754 | 0.772 |
| 2^11 | 0.890 | **0.788** | 0.788 | 0.788 | 0.788 | 0.788 | 0.788 |
| 2^12 | 1.003 | **0.875** | 0.898 | 0.876 | 0.898 | 0.876 | 0.898 |
| 2^13 | 1.063 | 0.943 | 0.942 | 0.944 | **0.941** | 0.942 | 0.942 |

Reading it:

- fusion is worth 10-20% over the single-pass kernel on the butterflies, and
  that is the only large effect here;
- depths 2, 3 and 4 are within 1-2% of each other over the range the periodogram
  uses (`2 x plan_block`, i.e. `2^10 .. 2^12` for typical light curves). The
  defaults are `floor(log2(VEC_LEN))` for the double pass, one less for float,
  2 for double-double, because a fused pass keeps `2^(L+1)` data vectors plus
  ~6 twiddle/temporary registers live: the double-equivalent depth needs half
  the register file, the full-lane depth fills it, and the full-lane depth was
  never faster in these runs;
- from 2^14 upwards the shallower depths start to win on AVX-512 (2^16: depth 2
  is ~15% ahead of depth 3 in the float pass), i.e. the transform turns
  bandwidth bound. A per-grid depth would capture that; it is not implemented,
  and the knobs are there for whoever wants to try;
- the pass order moves individual grids by up to ~5% with no consistent sign
  (better at 2^11, worse at 2^12 on AVX-512, the other way round on AVX2). The
  sweep geomean over 2^10..2^12 is within 2% either way, and end to end
  (`bench_periodogram`, 13 workloads) the order is 0.999x, i.e. neutral. The
  shipped default is the requested end-first schedule, with the leftover pass
  in front; `tls_tuning_set_fft_order()` selects the other one for A/B runs.

## bench_permute

Bit-reversal strategies only (no butterflies), one process, both precisions,
order rotated every round. `./build/bench_permute 6 8 4 16` on a Ryzen 9
7945HX, ns per element, best of 8 inside 6 rotations:

| n | table | pairs | cobravo (vec) | cobravo (scalar) | auto |
|---|---|---|---|---|---|
| 2^6 | 0.94 | **0.47** | 0.94 | 0.94 | 0.47 |
| 2^8 | 0.86 | **0.39** | 0.82 | 1.68 | 0.39 |
| 2^10 | 1.03 | **0.39** | 0.69 | 1.45 | 0.40 |
| 2^11 | 1.03 | **0.40** | 0.68 | 1.49 | 0.40 |
| 2^12 | 1.05 | **0.47** | 0.69 | 1.54 | 0.56 |
| 2^13 | 1.21 | 0.64 | **0.68** | 1.51 | 0.68 |
| 2^14 | 1.62 | 1.03 | **0.68** | 1.49 | 0.67 |
| 2^16 | 5.16 | 8.02 | **0.66** | 1.41 | 0.66 |

Double precision shifts the upper crossover down by one power of two (the pair
table is 4N bytes whatever the element size, so it leaves L1 earlier):
`pairs` wins up to 2^12 and `cobravo` from 2^13.

The dispatch therefore uses `table` below 2^6 (differences there are inside the
timer resolution), `pairs` up to 2^13 (float) / 2^12 (double), and `cobravo`
above. `tls_tuning_set_permute(0..3)` forces one strategy for A/B runs. The pair
list is built lazily on first use and freed with the plan; the blocked solver
creates one plan per call and permutes it once per block, so the build cost is
amortised.

## bench_periodogram

The same interleaving inside one process over the algorithm variants, per
workload. `M` is the number of observations, `N` the number of frequencies.

```
blap_b13_d3  M=580 N=8192 d=3
  variant                        ns   vs old
  fuse=1 vec end           658718.0    0.958
  fuse=2 vec end           652937.0    0.950
  fuse=3 vec end           643148.0    0.936
  fuse=3 vec front         643158.0    0.936
  fuse=1 vec natural       687122.0    1.000
  fuse=auto vec natural    675941.0    0.983
  fuse=auto scal defer     644962.0    0.938
  fuse=1 scal natural      687412.0    1.000
```

Reading it: COBRAVO is worth ~4% on the natural-order path, deferring the bit
reversal another ~5%, and fusion ~2% on top of that for this block size; the
gain grows with the grid (see `bench_end_to_end.txt`). The two order rows are
the same number, which is what a neutral setting looks like end to end (0.999x
geomean over all 13 workloads).

## Measured result (branch vs unmodified `main`)

Best of 5 interleaved rounds, Ryzen 9 7945HX, gcc 14.2, `-O3 -ffast-math
-march=native`, fusion depth 3 with the leftover pass last (the pre-update
defaults; the shipped per-precision defaults and the end-first order were
measured within a couple of percent of these, see `bench_fuse` above):

| workload | main | branch | ratio |
|---|---|---|---|
| blap_b12_d3 (M=580, N=4096) | 523.5 us | 476.5 us | 0.910 |
| blap_b13_d3 (N=8192) | 978.3 | 684.2 | 0.699 |
| blap_b14_d3 (N=12288) | 1249.2 | 782.5 | 0.626 |
| blap_b13_d1 (nterms=1) | 257.5 | 237.0 | 0.920 |
| blap_b13_d8 (nterms=8) | 3004.7 | 1996.8 | 0.665 |
| blap_b13_d3 nll | 913.9 | 838.5 | 0.918 |
| blap_b13_d3 pswf21 | 800.2 | 674.6 | 0.843 |
| lmc_b13_d3 (M=1800) | 1426.8 | 1312.3 | 0.920 |
| rrlyr_b12_d3 (M=950) | 719.7 | 669.3 | 0.930 |
| blap_b16_d3 (N=65536) | 5843.6 | 3161.4 | 0.541 |
| blap_b17_d3 (N=131072) | 9200.3 | 5637.7 | 0.613 |
| ihs_b13_d3 | 267.5 | 254.8 | 0.952 |
| ihs_b13_d1 (unchanged path) | 125.0 | 124.5 | 0.996 |

No workload regresses. The fixed per-call cost (plan construction and the PSWF
spreading precomputation) dominates the small grids, which is why the gain
grows with `N`: the sweep is what the four changes optimise.

Files written by `bench.sh`:

- `build/bench_end_to_end.txt` - the table above,
- `build/bench_variants.txt` - the per-workload variant breakdown,
- `build/bench_fft.txt` - the FFT microbenchmark (`bench_fuse` is not a
  `bench.sh` step; run it directly, it prints its own tables),
- `build/dump_base.txt`, `build/dump_branch.txt` - the numeric gate data.
