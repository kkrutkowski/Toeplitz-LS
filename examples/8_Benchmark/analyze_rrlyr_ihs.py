#!/usr/bin/env python3
"""Summarize the RR Lyrae IHS/FastChi2 benchmark results.

The output files are produced by ``run_benchmark.py`` and contain the peak
frequency and peak statistic for each source.  Period recovery is measured
with the same tolerance as ``compare.py`` (``abs(delta_f) * 2840 < 2``),
including the half/double-frequency aliases.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

SCALE = 2840.0
THRESHOLD = 2.0


def read_index(path: Path):
    result = {}
    with path.open(newline="") as source:
        header = source.readline()
        if not header:
            return result
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0]:
                result[parts[0]] = (float(parts[1]), float(parts[2]))
    return result


def read_result(path: Path):
    metadata = {}
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("# "):
                key, _, value = line[2:].rstrip("\n").partition("\t")
                metadata[key] = value
                continue
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "source_id":
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}: expected source_id, frequency, and power")
            rows[parts[0]] = (float(parts[1]), float(parts[2]))
    return metadata, rows


def recovery(rows, reference):
    matched = 0
    direct = 0
    alias = 0
    alias_counts = {"direct": 0, "2f": 0, "f/2": 0}
    frequency_errors = []
    period_errors = []
    for source_id, (candidate, _power) in rows.items():
        target = reference.get(source_id)
        if target is None:
            continue
        target_frequency, target_period = target
        matched += 1
        direct_hit = abs(candidate - target_frequency) * SCALE < THRESHOLD
        double_hit = abs(candidate - 2.0 * target_frequency) * SCALE < THRESHOLD
        half_hit = abs(candidate - 0.5 * target_frequency) * SCALE < THRESHOLD
        if direct_hit:
            alias_counts["direct"] += 1
        elif double_hit:
            alias_counts["2f"] += 1
        elif half_hit:
            alias_counts["f/2"] += 1
        if direct_hit:
            direct += 1
        if direct_hit or double_hit or half_hit:
            alias += 1
        if direct_hit and candidate > 0.0:
            frequency_errors.append(abs(candidate - target_frequency))
            period_errors.append(abs(1.0 / candidate - target_period))
    return {
        "matched": matched,
        "direct": direct,
        "alias": alias,
        "alias_counts": alias_counts,
        "frequency_error_median": float(np.median(frequency_errors))
        if frequency_errors
        else math.nan,
        "period_error_median": float(np.median(period_errors))
        if period_errors
        else math.nan,
        "period_error_p90": float(np.percentile(period_errors, 90))
        if period_errors
        else math.nan,
    }


def power_stats(rows):
    values = np.asarray([power for _frequency, power in rows.values() if np.isfinite(power)])
    if values.size == 0:
        return {
            "count": 0,
            "median": math.nan,
            "mean": math.nan,
            "p10": math.nan,
            "p90": math.nan,
        }
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _rank(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def pairwise(rows_a, rows_b):
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return None
    frequency_a = np.asarray([rows_a[key][0] for key in common])
    frequency_b = np.asarray([rows_b[key][0] for key in common])
    power_a = np.asarray([rows_a[key][1] for key in common])
    power_b = np.asarray([rows_b[key][1] for key in common])
    finite = np.isfinite(power_a) & np.isfinite(power_b)
    if not np.any(finite):
        return None
    a = power_a[finite]
    b = power_b[finite]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = b / a
    positive = np.isfinite(ratio) & (a > 0.0)
    same_frequency = np.isclose(
        frequency_a[finite], frequency_b[finite], rtol=0.0, atol=1e-12
    )
    same_count = int(np.count_nonzero(same_frequency))
    if same_count:
        same_a = a[same_frequency]
        same_b = b[same_frequency]
        with np.errstate(divide="ignore", invalid="ignore"):
            same_ratio = same_b / same_a
        same_positive = np.isfinite(same_ratio) & (same_a > 0.0)
        same_median_ratio = (
            float(np.median(same_ratio[same_positive]))
            if np.any(same_positive)
            else math.nan
        )
        same_median_delta = float(np.median(same_b - same_a))
        same_pearson = (
            float(np.corrcoef(same_a, same_b)[0, 1]) if same_count > 1 else math.nan
        )
    else:
        same_median_ratio = math.nan
        same_median_delta = math.nan
        same_pearson = math.nan
    return {
        "common": len(common),
        "finite": int(np.count_nonzero(finite)),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(np.corrcoef(_rank(a), _rank(b))[0, 1]),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "median_delta_b_minus_a": float(np.median(b - a)),
        "median_ratio_b_over_a": float(np.median(ratio[positive]))
        if np.any(positive)
        else math.nan,
        "same_best_frequency": float(np.mean(same_frequency)),
        "same_count": same_count,
        "same_median_ratio": same_median_ratio,
        "same_median_delta": same_median_delta,
        "same_pearson": same_pearson,
    }


def _fmt(value, decimals=6):
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{decimals}g}"
    return str(value)


def summarize(result_dir: Path, index_path: Path):
    reference = read_index(index_path)
    results = []
    for path in sorted(result_dir.glob("*_nterms*.tsv")):
        metadata, rows = read_result(path)
        nterms = int(metadata.get("nterms", "-1"))
        method = metadata.get("method", path.stem)
        stats = recovery(rows, reference)
        powers = power_stats(rows)
        completed = int(metadata.get("completed_files", len(rows)))
        cpu_hours = float(metadata.get("power_user_cpu_hours", "nan"))
        results.append(
            {
                "path": path,
                "nterms": nterms,
                "method": method,
                "metadata": metadata,
                "rows": rows,
                "stats": stats,
                "powers": powers,
                "cpu_hours": cpu_hours,
                "completed": completed,
            }
        )
    results.sort(key=lambda item: (item["nterms"], item["method"]))
    by_key = {(item["nterms"], item["method"]): item for item in results}

    lines = ["# RR Lyrae IHS/FastChi2 benchmark", ""]
    lines.append(
        f"Period recovery uses the `compare.py` criterion: "
        f"`abs(f - F_1) * {SCALE:g} < {THRESHOLD:g}`, with half/double aliases."
    )
    lines.append(
        "Peak-statistic columns are method-dependent: `standard` rows are "
        "coefficient-of-determination-like values, while `nll` and "
        "`regularized` rows are NLL-like values and are not on the same scale."
    )
    lines.append(
        "Setup: 39,931 eligible RR Lyrae curves, `fmax=5.0`, "
        "`oversampling=5.0`, unweighted (`dy=None`), PSWF43 backend; "
        "`tlsf` uses the Levinson solver for d=3. The automatic grid is "
        "denser for d=3 because `df` scales as `1/nterms`."
    )
    lines.append(
        "For d=1 the standard and NLL peak bins can coincide because the "
        "Wilkie correction is monotonic in the single harmonic power, even "
        "though the NLL values differ. For d>1 the per-harmonic correction "
        "can move the peak."
    )
    lines.append("")
    lines.append(
        "| d | method | normalization | matched | direct | with alias | "
        "median peak | mean peak | CPU h | ms/curve | med. period err (d) |"
    )
    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for item in results:
        stats = item["stats"]
        powers = item["powers"]
        normalization = _normalization(item["method"])
        per_curve_ms = (
            item["cpu_hours"] * 3.6e6 / item["completed"]
            if item["completed"] > 0
            else math.nan
        )
        lines.append(
            f"| {item['nterms']} | `{item['method']}` | `{normalization}` | "
            f"{stats['matched']} | {100.0 * stats['direct'] / max(stats['matched'], 1):.2f}% | "
            f"{100.0 * stats['alias'] / max(stats['matched'], 1):.2f}% | "
            f"{_fmt(powers['median'])} | {_fmt(powers['mean'])} | "
            f"{_fmt(item['cpu_hours'], 4)} | {_fmt(per_curve_ms, 4)} | "
            f"{_fmt(stats['period_error_median'], 4)} |"
        )

    lines.extend(["", "## Peak-frequency agreement", ""])
    lines.append(
        "| d | comparison | common | same best freq | "
        "recovery overlap | median delta | median ratio | Pearson | Spearman |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    pairs = [
        ("tlsf_standard", "ihsf_standard"),
        ("tlsf_regularized", "ihsf_nll"),
        ("tlsf_standard", "tlsf_regularized"),
        ("ihsf_standard", "ihsf_nll"),
    ]
    pair_stats = []
    for nterms in sorted({item["nterms"] for item in results}):
        for method_a, method_b in pairs:
            item_a = by_key.get((nterms, method_a))
            item_b = by_key.get((nterms, method_b))
            if item_a is None or item_b is None:
                continue
            stats = pairwise(item_a["rows"], item_b["rows"])
            if stats is None:
                continue
            recovery_overlap = _recovery_overlap(
                item_a["rows"], item_b["rows"], reference
            )
            pair_stats.append((nterms, method_a, method_b, stats))
            lines.append(
                f"| {nterms} | `{method_a}` vs `{method_b}` | {stats['common']} | "
                f"{100.0 * stats['same_best_frequency']:.2f}% | "
                f"{100.0 * recovery_overlap:.2f}% | "
                f"{_fmt(stats['median_delta_b_minus_a'])} | "
                f"{_fmt(stats['median_ratio_b_over_a'])} | "
                f"{_fmt(stats['pearson'], 4)} | {_fmt(stats['spearman'], 4)} |"
            )

    lines.extend(["", "## Same-frequency peak comparison", ""])
    lines.append(
        "Restricted to curves where both methods select the same best "
        "frequency; NLL/regularized values still have different scales."
    )
    lines.append("")
    lines.append(
        "| d | comparison | n same | median ratio (b/a) | median delta (b-a) | Pearson |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|")
    for nterms, method_a, method_b, stats in pair_stats:
        lines.append(
            f"| {nterms} | `{method_a}` vs `{method_b}` | {stats['same_count']} | "
            f"{_fmt(stats['same_median_ratio'])} | "
            f"{_fmt(stats['same_median_delta'])} | {_fmt(stats['same_pearson'], 4)} |"
        )

    lines.extend(["", "## Period-recovery details", ""])
    for item in results:
        stats = item["stats"]
        counts = stats["alias_counts"]
        lines.append(
            f"- d={item['nterms']} `{item['method']}`: direct {stats['direct']}, "
            f"2f {counts['2f']}, f/2 {counts['f/2']}; "
            f"median period error {_fmt(stats['period_error_median'], 4)} d, "
            f"90th percentile {_fmt(stats['period_error_p90'], 4)} d."
        )
    return "\n".join(lines) + "\n"


def _normalization(method: str) -> str:
    if method.endswith("_regularized"):
        return "regularized"
    if method.endswith("_nll"):
        return "nll"
    if method.endswith("_standard"):
        return "standard"
    return "standard"


def _recovery_overlap(rows_a, rows_b, reference):
    common = set(rows_a) & set(rows_b) & set(reference)
    if not common:
        return math.nan
    hits = 0
    for source_id in common:
        target = reference[source_id][0]
        candidate_a = rows_a[source_id][0]
        candidate_b = rows_b[source_id][0]
        hit_a = (
            abs(candidate_a - target) * SCALE < THRESHOLD
            or abs(candidate_a - 2.0 * target) * SCALE < THRESHOLD
            or abs(candidate_a - 0.5 * target) * SCALE < THRESHOLD
        )
        hit_b = (
            abs(candidate_b - target) * SCALE < THRESHOLD
            or abs(candidate_b - 2.0 * target) * SCALE < THRESHOLD
            or abs(candidate_b - 0.5 * target) * SCALE < THRESHOLD
        )
        hits += int(hit_a and hit_b)
    return hits / len(common)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("out_rrlyr_ihs"))
    parser.add_argument("--index-path", type=Path, default=Path("index_rrlyr.tsv"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = summarize(args.result_dir, args.index_path)
    if args.output is not None:
        args.output.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
