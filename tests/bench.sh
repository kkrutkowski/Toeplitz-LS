#!/bin/sh
# Interleaved benchmark + correctness gate for the fast-path branch.
#
# Every round runs each binary/variant once on the same workload, rotating the
# order so none is always measured on a cold/boosted clock; the reported time
# is the best of R rounds of the best of REPS internal repetitions.
#
#   R=5 REPS=5 sh tests/bench.sh
#
# Steps:
#   1. build the branch tests,
#   2. build the pristine baseline (git worktree of main) with the same flags,
#   3. numeric gate: identical DUMP summaries from both builds,
#   4. interleaved baseline-vs-branch timing (bench_public, identical source),
#   5. interleaved variant matrix inside one process (bench_periodogram),
#   6. FFT microbenchmark over the fusion depths (bench_fft).
set -u
cd "$(dirname "$0")" || exit 1
export LC_ALL=C

R=${R:-5}
REPS=${REPS:-5}
CPU=${CPU:-2}
ROOT=$(cd .. && pwd)
BUILD=$(pwd)/build
BASE=${BASE:-$(cd .. && pwd)/../baseline}
CFLAGS=${CFLAGS:--std=gnu11 -O3 -ffast-math -march=native -mtune=native -Wall -Wextra}
FUSE=${FUSE:--DNANOFFT_FFT_FUSE_FLOAT=3 -DNANOFFT_FFT_FUSE_DOUBLE=3 -DNANOFFT_FFT_FUSE_DD=3}

mkdir -p "$BUILD" || exit 1
run() { taskset -c "$CPU" "$@"; }

echo "== build branch =="
make -s all CFLAGS="$CFLAGS" FFT_FUSE_FLOAT=3 FFT_FUSE_DOUBLE=3 FFT_FUSE_DD=3 || exit 1

echo "== build baseline ($BASE) =="
if [ ! -d "$BASE/src" ]; then
    git -C "$ROOT" worktree add "$BASE" main >/dev/null 2>&1 || { echo "cannot create baseline worktree at $BASE" >&2; exit 1; }
fi
for prec in 'f:' 'd:-DDOUBLE' 'dd:-DDOUBLE_DOUBLE'; do
    suffix=${prec%%:*}
    flags=${prec#*:}
    for src in nanofft nufft1 linalg chi2per utils; do
        # shellcheck disable=SC2086
        cc $CFLAGS $flags -I"$BASE/include" -I"$BUILD" -c "$BASE/src/$src.c" -o "$BUILD/base_${src}_${suffix}.o" || exit 1
    done
done
cc $CFLAGS -I"$BASE/include" -I"$BUILD" dump_periodogram.c "$BUILD"/base_*.o -o "$BUILD/dump_baseline" -lm || exit 1
cc $CFLAGS -I"$BASE/include" -I"$BUILD" bench_public.c "$BUILD"/base_*.o -o "$BUILD/bench_baseline" -lm || exit 1

echo "== correctness gates =="
"$BUILD/check_nanofft" || exit 1
"$BUILD/check_periodogram" || exit 1

"$BUILD/dump_periodogram" >"$BUILD/dump_branch.txt"
"$BUILD/dump_baseline" >"$BUILD/dump_base.txt"
python3 - "$BUILD/dump_base.txt" "$BUILD/dump_branch.txt" <<'PY' || exit 1
import re
import sys

def load(path):
    out = {}
    for line in open(path):
        if not line.startswith("DUMP"):
            continue
        name = line.split()[1]
        out[name] = dict(re.findall(r"(\w+)=([-\d.eE+]+)", line))
    return out

base, branch = load(sys.argv[1]), load(sys.argv[2])
if set(base) != set(branch):
    print("FAIL: case sets differ")
    raise SystemExit(1)
worst = 0.0
for name, ref in base.items():
    got = branch[name]
    if ref["st"] != got["st"] or ref["nan"] != got["nan"]:
        print(f"FAIL {name}: status/NaN count changed ({ref['st']}/{got['st']}, {ref['nan']}/{got['nan']})")
        raise SystemExit(1)
    for field in ("sum", "chk", "peak"):
        a, b = float(ref[field]), float(got[field])
        rel = abs(a - b) / max(1e-30, abs(a))
        worst = max(worst, rel)
print(f"baseline-vs-branch numeric gate: worst relative difference {worst:.3g}")
if worst > 1e-4:
    print("FAIL: the restructuring changed the periodogram beyond float roundoff")
    raise SystemExit(1)
PY

echo "== interleaved baseline vs branch =="
rm -f "$BUILD"/runs_*.txt
r=1
while [ "$r" -le "$R" ]; do
    if [ $((r % 2)) -eq 1 ]; then order="base branch"; else order="branch base"; fi
    for which in $order; do
        if [ "$which" = base ]; then bin="$BUILD/bench_baseline"; else bin="$BUILD/bench_public"; fi
        run "$bin" 1 "$REPS" >>"$BUILD/runs_$which.txt" || exit 1
    done
    r=$((r + 1))
done

python3 - "$BUILD/runs_base.txt" "$BUILD/runs_branch.txt" <<'PY' | tee "$BUILD/bench_end_to_end.txt"
import re
import sys

def best(path):
    out = {}
    for line in open(path):
        if not line.startswith("RESULT"):
            continue
        parts = line.split()
        name, ns = parts[1], float(parts[2])
        if name not in out or ns < out[name]:
            out[name] = ns
    return out

base, branch = best(sys.argv[1]), best(sys.argv[2])
print(f"{'workload':22s} {'baseline us':>12s} {'branch us':>12s} {'ratio':>7s}")
worst = 0.0
for name in base:
    b, n = base[name] / 1e3, branch.get(name, float('nan')) / 1e3
    ratio = branch[name] / base[name]
    worst = max(worst, ratio)
    print(f"{name:22s} {b:12.1f} {n:12.1f} {ratio:7.3f}")
print(f"\nworst branch/baseline ratio: {worst:.3f}")
# The noise floor of a single-core run-to-run comparison at these sizes is
# around 2-3%; anything beyond 5% is a real regression.
if worst > 1.05:
    print("WARNING: at least one workload regressed by more than 5%")
PY

echo "== interleaved algorithm variants =="
run "$BUILD/bench_periodogram" "$R" "$REPS" | tee "$BUILD/bench_variants.txt"

echo "== FFT microbenchmark (fusion depth) =="
run "$BUILD/bench_fft" "$R" "$REPS" 10 16 | tee "$BUILD/bench_fft.txt"

echo "wrote $BUILD/bench_end_to_end.txt, bench_variants.txt, bench_fft.txt"
