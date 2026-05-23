#!/usr/bin/env python3
"""Report period recovery rates for Benchmark 8 output files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SCALE = 2840.0
THRESHOLD = 2.0


def read_index(path: Path) -> dict[str, float]:
    result = {}
    with path.open(newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                result[row[0]] = float(row[1])
    return result


def read_result(path: Path):
    metadata = {}
    rows = []
    with path.open(encoding="utf-8", newline="") as source:
        for line in source:
            if line.startswith("# "):
                key, _, value = line[2:].rstrip("\n").partition("\t")
                metadata[key] = value
                continue
            if not line.strip():
                continue
            row = next(csv.reader([line], delimiter="\t"))
            if row[0] == "source_id":
                continue
            rows.append((row[0], float(row[1])))
    return metadata, rows


def report(path: Path, reference: dict[str, float]) -> None:
    metadata, rows = read_result(path)
    direct = 0
    with_alias = 0
    matched = 0
    for source_id, candidate in rows:
        if source_id not in reference:
            continue
        matched += 1
        target = reference[source_id]
        direct_hit = abs(candidate - target) * SCALE < THRESHOLD
        alias_hit = (
            direct_hit
            or abs(candidate - target * 0.5) * SCALE < THRESHOLD
            or abs(candidate - target * 2.0) * SCALE < THRESHOLD
        )
        direct += int(direct_hit)
        with_alias += int(alias_hit)

    method = metadata.get("method", path.stem)
    nterms = metadata.get("nterms", "?")
    status = metadata.get("status", "legacy")
    cpu_hours = metadata.get("power_user_cpu_hours", "?")
    cumulative = metadata.get("cumulative_power_user_cpu_hours", "?")
    if matched == 0:
        print(
            f"{method} nterms={nterms} status={status}: no matched result rows "
            f"(CPU-hours={cpu_hours}, cumulative={cumulative})"
        )
        return
    print(
        f"{method} nterms={nterms} status={status}: "
        f"{100.0 * direct / matched:.2f}%|{100.0 * with_alias / matched:.2f}% "
        f"on {matched} rows; CPU-hours={cpu_hours}, cumulative={cumulative}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Measure period recovery in benchmark TSV outputs.")
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--index-path", type=Path, default=Path("./index.tsv"))
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args(argv)
    files = list(args.files)
    if args.result_dir is not None:
        files.extend(sorted(args.result_dir.glob("*.tsv")))
    if not files:
        parser.error("provide result files or --result-dir")
    args.files = files
    return args


def main(argv=None):
    args = parse_args(argv)
    reference = read_index(args.index_path)
    for path in args.files:
        report(path, reference)


if __name__ == "__main__":
    main()
