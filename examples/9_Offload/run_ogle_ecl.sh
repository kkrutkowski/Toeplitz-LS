#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/krutkowski/python-nogil/bin/python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PHOT_DIR="${PHOT_DIR:-/home/krutkowski/Pulpit/LMC_ECL/phot}"
OUT_DIR="${1:-out_ogle_ecl}"
DEVICE="${DEVICE:-0}"
FMAX="${FMAX:-4.0}"
WORKERS="${WORKERS:-15}"

"$PYTHON" ./run_benchmark.py \
  --phot-dir "$PHOT_DIR" \
  --output-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --methods "sycl_tlsh,sycl_tlsf,sycl_tlsdf,sycl_tls,sycl_ihsf,cpu_tlsh,cpu_tlsf,cpu_tls" \
  --nterms "1,3" \
  --fmax "$FMAX" \
  --oversampling 5.0 \
  --use-dy \
  --max-workers "$WORKERS"
