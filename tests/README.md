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
R=5 REPS=5 sh tests/bench.sh   # gates + interleaved baseline comparison
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
(`./build/bench_fft 5 7 10 16`) on a Ryzen 9 7945HX:

| n | fuse=1 | fuse=2 | fuse=3 | fuse=4 | scalar perm |
|---|---|---|---|---|---|
| 2^10 | 1.51 ns/elem | 1.39 | 1.41 | **1.38** | 2.12 |
| 2^11 | 1.49 | 1.47 | **1.39** | 1.99 | 2.24 |
| 2^12 | 2.30 | 1.49 | **1.47** | 1.67 | 2.12 |
| 2^13 | 1.68 | **1.58** | 1.81 | 1.90 | 2.40 |
| 2^14 | 2.01 | **1.66** | 1.76 | 1.81 | 2.66 |
| 2^15 | **1.85** | 1.98 | 1.94 | 1.95 | 2.65 |

Fusion pays off from the smallest sizes up to `2^14` (up to 1.5x on the
butterflies at `2^12`); beyond that the single-pass kernel is already bandwidth
bound and depth 1..2 wins. Depth 3 is the default because it is at or near the
front over the range the periodogram actually uses (`2 x plan_block`, i.e.
`2^10 .. 2^12` for typical light curves) and never loses much; the knobs are
there for other grids. Depth 4 needs 32 vector registers and starts to spill.
The vectorized COBRAVO permutation is 1.4-1.6x the scalar one everywhere.

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
  variant                   ns   vs old
  fuse=1 vec defer      703014    0.837
  fuse=2 vec defer      689590    0.821
  fuse=3 vec defer      688578    0.820
  fuse=1 vec natural    768607    0.915
  fuse=3 vec natural    743500    0.885
  fuse=3 scal defer     695180    0.828
  fuse=1 scal natural   840071    1.000
```

Reading it: COBRAVO is worth ~9% on the natural-order path, deferring the bit
reversal another ~8%, and fusion ~2% on top of that for this block size; the
gain grows with the grid (see `bench_end_to_end.txt`).

## Measured result (branch vs unmodified `main`)

Best of 5 interleaved rounds, Ryzen 9 7945HX, gcc 14.2, `-O3 -ffast-math
-march=native`, fusion depth 3:

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
- `build/bench_fft.txt` - the FFT microbenchmark,
- `build/dump_base.txt`, `build/dump_branch.txt` - the numeric gate data.
