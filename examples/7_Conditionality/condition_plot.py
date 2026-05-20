import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import PercentFormatter


INDEX_PATH = Path("./index.tsv")
OUT_COND_DIR = Path("./out_cond")
OUTPUT_STEM = "condition"
SCALE = 2840.0
THRESHOLD = 2.0


def adjust_brightness(color, factor):
    rgba = to_rgba(color)
    return (rgba[0] * factor, rgba[1] * factor, rgba[2] * factor, rgba[3])


try:
    import seaborn as sns

    colors = sns.color_palette("colorblind", 4)
except ImportError:
    colors = ["#0173b2", "#de8f05", "#029e73", "#d55e00"]

adjusted_colors = [adjust_brightness(color, 1.0) for color in colors]
adjusted_colors[0] = adjust_brightness(colors[0], 0.4)
adjusted_colors[3] = adjust_brightness("red", 0.8)
adjusted_colors[2] = adjust_brightness(colors[2], 1.1)


def read_index(path):
    names = []
    freqs = []

    with path.open(newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)

        for row in reader:
            if not row:
                continue
            names.append(row[0])
            freqs.append(float(row[1]))

    return names, freqs


def read_result(path):
    names = []
    freqs = []

    with path.open(newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if not row:
                continue
            names.append(row[0])
            freqs.append(float(row[1]))

    return names, freqs


def alias_recovery_rate(index_names, index_freqs, result_path):
    result_names, result_freqs = read_result(result_path)

    shift = 0
    recovered = 0

    for i, name in enumerate(result_names):
        while index_names[i + shift] != name:
            shift += 1

        index_freq = index_freqs[i + shift]
        result_freq = result_freqs[i]

        if (
            abs(result_freq - index_freq) * SCALE < THRESHOLD
            or abs(result_freq - index_freq * 0.5) * SCALE < THRESHOLD
            or abs(result_freq - index_freq * 2.0) * SCALE < THRESHOLD
        ):
            recovered += 1

    return recovered / len(result_names)


def load_condition_results():
    index_names, index_freqs = read_index(INDEX_PATH)
    pattern = re.compile(r"^(tlsf?|tls)_d8_([0-9]+\.[0-9]{2})\.tsv$")
    series = {"tls": [], "tlsf": []}

    for path in sorted(OUT_COND_DIR.glob("*.tsv")):
        match = pattern.match(path.name)
        if not match:
            continue

        precision, log10_kappa = match.groups()
        recovery_rate = alias_recovery_rate(index_names, index_freqs, path)
        series[precision].append((float(log10_kappa), recovery_rate))

    for precision in series:
        series[precision].sort(key=lambda row: row[0])

    return series


def plot_condition_results():
    series = load_condition_results()

    plt.figure(figsize=(7.5, 5.0))

    plt.plot(
        [x for x, _rate in series["tls"]],
        [rate for _x, rate in series["tls"]],
        color=adjusted_colors[0],
        marker="^",
        markersize=0.0,
        linewidth=2.0,
        label="tls",
        linestyle="dashed",
    )
    plt.plot(
        [x for x, _rate in series["tlsf"]],
        [rate for _x, rate in series["tlsf"]],
        color=adjusted_colors[3],
        marker="o",
        markersize=0.0,
        linewidth=2.0,
        label="tlsf",
        linestyle="dashed",
    )

    plt.xlabel(r"$\log_{10}(\kappa_{max})$", fontsize=16)
    plt.ylabel("Period recovery rate", fontsize=16)
    # plt.yscale("log", base=10)
    plt.xlim(0, 20)
    plt.ylim(0.7500001, 0.95)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.legend(fontsize=12)
    plt.tick_params(axis="both", which="major", direction="in", labelsize=14)
    # plt.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.5)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_STEM}.eps", bbox_inches="tight")
    plt.savefig(f"{OUTPUT_STEM}.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    plot_condition_results()
