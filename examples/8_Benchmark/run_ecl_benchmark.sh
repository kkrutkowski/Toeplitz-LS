#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/krutkowski/python-nogil/bin/python3.14"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT_DIR="${1:-out_ecl}"
FMAX="${2:-5.0}"

echo "Running ECL benchmark (d=11, fmax=${FMAX}) into ${OUT_DIR}..."
"$PYTHON" ./run_benchmark.py \
  --phot-dir ./phot \
  --index-path ./index.tsv \
  --output-dir "$OUT_DIR" \
  --methods tls,tlsf \
  --nterms 11 \
  --fmax "$FMAX" \
  --oversampling 5.0 \
  --dataset-limit 0 \
  --compute-limit 0 \
  --max-workers 15

echo "Evaluating recovery rates with compare.py:"
"$PYTHON" ./compare.py \
  --index-path ./index.tsv \
  --result-dir "$OUT_DIR"
