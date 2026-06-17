#!/usr/bin/env python3
"""Plot A7 empty_only calibration summary for the NeurIPS bundle."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ALL_RUNS = ROOT / "out" / "paper_integrated_evidence" / "tables" / "a7_empty_only_calibration_all_runs.tsv"
SUMMARY = ROOT / "out" / "paper_integrated_evidence" / "tables" / "a7_empty_only_calibration_summary.tsv"
PLOTS = ROOT / "out" / "final_paper_bundle" / "plots" / "main"
PLOTS.mkdir(parents=True, exist_ok=True)
OUT = PLOTS / "fig_a7_empty_only_calibration.png"

T95_N3 = 4.302652729911275

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.family": "serif",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.1,
    "lines.markersize": 6.0,
})


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def f(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "NA")
    if raw in {"", "NA", "None"}:
        return float("nan")
    return float(raw)


def ci95(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 3:
        return T95_N3 * std / math.sqrt(n)
    return 1.96 * std / math.sqrt(n)


def aggregate_candidate_count(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["family_slug"] != "set_dense_exact_empty_only":
            continue
        grouped[(row["w"], row["s"])].append(f(row, "candidate_count_mean"))
    return {key: sum(vals) / len(vals) for key, vals in grouped.items() if vals}


def set_rows(summary: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [r for r in summary if r["family_slug"] == "set_dense_exact_empty_only"]
    return sorted(rows, key=lambda r: f(r, "M_over_L"))


def baseline_row(summary: list[dict[str, str]]) -> dict[str, str]:
    matches = [r for r in summary if r["family_slug"] == "baseline_dense_exact"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one baseline row")
    return matches[0]


def topology_label(row: dict[str, str]) -> str:
    return f"w={row['w']}, s={row['s']}"


def label_offset(row: dict[str, str], panel: str) -> tuple[int, int]:
    w, s = int(row["w"]), int(row["s"])
    offsets = {
        "M_over_L": {
            (1, 1): (6, -13),
            (2, 1): (6, 3),
            (3, 1): (6, 13),
            (2, 2): (6, -12),
            (4, 2): (6, 7),
            (8, 4): (6, -12),
            (16, 8): (6, 6),
            (32, 16): (6, 6),
        },
        "L_over_M": {
            (1, 1): (6, -13),
            (2, 1): (6, 3),
            (3, 1): (6, 13),
            (2, 2): (6, -12),
            (4, 2): (6, 7),
            (8, 4): (6, -12),
            (16, 8): (6, 6),
            (32, 16): (6, 6),
        },
        "candidate_count": {
            (1, 1): (5, 5),
            (2, 1): (5, 5),
            (3, 1): (5, 5),
            (2, 2): (5, 5),
            (4, 2): (5, 5),
            (8, 4): (5, -13),
            (16, 8): (5, 5),
            (32, 16): (5, 5),
        },
        "vram": {
            (1, 1): (-8, 11),
            (2, 1): (-8, -1),
            (3, 1): (-8, -13),
            (2, 2): (6, -13),
            (4, 2): (6, 5),
            (8, 4): (6, -15),
            (16, 8): (6, 7),
            (32, 16): (6, -1),
        },
    }
    return offsets.get(panel, {}).get((w, s), (5, 5))


def annotate_topology(ax, row: dict[str, str], x: float, y: float, panel: str) -> None:
    dx, dy = label_offset(row, panel)
    ax.annotate(
        topology_label(row),
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.8,
        ha="right" if dx < 0 else "left",
    )


def plot_ppl_panel(ax, rows, xkey, xlabel, base, *, xscale: str | None = None, candidate_counts=None):
    color = "#1b6ca8"
    band = "#86b7d8"
    single = "#b45f06"
    xs3, ys3, ci3 = [], [], []
    xs1, ys1 = [], []
    for row in rows:
        x = candidate_counts[(row["w"], row["s"])] if xkey == "candidate_count_mean" else f(row, xkey)
        y = f(row, "mean_final_val_ppl")
        n = int(row["n"])
        if n >= 2:
            xs3.append(x)
            ys3.append(y)
            ci3.append(ci95(f(row, "std_final_val_ppl"), n))
        else:
            xs1.append(x)
            ys1.append(y)
    order = sorted(range(len(xs3)), key=lambda i: xs3[i])
    xs3 = [xs3[i] for i in order]
    ys3 = [ys3[i] for i in order]
    ci3 = [ci3[i] for i in order]

    ax.errorbar(
        xs3,
        ys3,
        yerr=ci3,
        fmt="o",
        color=color,
        ecolor=band,
        elinewidth=2.0,
        capsize=3,
        label="SetDense empty_only (3 seeds, 95% CI)",
    )
    if xs1:
        ax.scatter(
            xs1,
            ys1,
            marker="x",
            s=58,
            linewidths=2.0,
            color=single,
            label="SetDense empty_only (1 seed)",
            zorder=4,
        )

    bmean = f(base, "mean_final_val_ppl")
    bci = ci95(f(base, "std_final_val_ppl"), int(base["n"]))
    xmin, xmax = ax.get_xlim()
    ax.axhline(bmean, color="#333333", linestyle="--", linewidth=1.4, label="Dense token baseline")
    ax.axhspan(bmean - bci, bmean + bci, color="#444444", alpha=0.10, linewidth=0)
    ax.set_xlim(xmin, xmax)
    for row in rows:
        x = candidate_counts[(row["w"], row["s"])] if xkey == "candidate_count_mean" else f(row, xkey)
        y = f(row, "mean_final_val_ppl")
        annotate_topology(ax, row, x, y, xkey)
    if xscale:
        ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation perplexity")


def plot_vram_panel(ax, rows, base):
    color = "#2f7d32"
    xs3, ys3 = [], []
    xs1, ys1 = [], []
    for row in rows:
        x = f(row, "M_over_L")
        y = f(row, "mean_peak_vram_mib") / 1024.0
        if int(row["n"]) >= 2:
            xs3.append(x)
            ys3.append(y)
        else:
            xs1.append(x)
            ys1.append(y)
    order = sorted(range(len(xs3)), key=lambda i: xs3[i])
    xs3 = [xs3[i] for i in order]
    ys3 = [ys3[i] for i in order]
    ax.plot(xs3, ys3, marker="o", color=color, label="SetDense empty_only")
    if xs1:
        ax.scatter(xs1, ys1, marker="x", s=58, linewidths=2.0, color="#b45f06", zorder=4)
    ax.axhline(
        f(base, "mean_peak_vram_mib") / 1024.0,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        label="Dense token baseline",
    )
    for row in rows:
        x = f(row, "M_over_L")
        y = f(row, "mean_peak_vram_mib") / 1024.0
        annotate_topology(ax, row, x, y, "vram")
    ax.set_xlabel(r"Set-state ratio $M/L$")
    ax.set_ylabel("Peak VRAM (GiB)")


def plot_candidate_panel(ax, rows, base, candidate_counts):
    color = "#1b6ca8"
    single = "#b45f06"
    for row in rows:
        x = candidate_counts[(row["w"], row["s"])]
        y = f(row, "mean_final_val_ppl")
        n = int(row["n"])
        if n >= 2:
            yerr = ci95(f(row, "std_final_val_ppl"), n)
            ax.errorbar(
                [x],
                [y],
                yerr=[[yerr], [yerr]],
                fmt="o",
                color=color,
                ecolor="#86b7d8",
                elinewidth=2.0,
                capsize=3,
                zorder=3,
            )
            annotate_topology(ax, row, x, y, "candidate_count")
        else:
            ax.scatter([x], [y], marker="x", s=58, linewidths=2.0, color=single, zorder=4)
            annotate_topology(ax, row, x, y, "candidate_count")

    bmean = f(base, "mean_final_val_ppl")
    bci = ci95(f(base, "std_final_val_ppl"), int(base["n"]))
    ax.axhline(bmean, color="#333333", linestyle="--", linewidth=1.4, label="Dense token baseline")
    ax.axhspan(bmean - bci, bmean + bci, color="#444444", alpha=0.10, linewidth=0)
    ax.set_xlabel("Mean candidate count")
    ax.set_ylabel("Validation perplexity")


def main() -> None:
    all_rows = read_tsv(ALL_RUNS)
    summary = read_tsv(SUMMARY)
    rows = set_rows(summary)
    base = baseline_row(summary)
    candidate_counts = aggregate_candidate_count(all_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.4), constrained_layout=True)
    plot_ppl_panel(
        axes[0, 0],
        rows,
        "M_over_L",
        r"Set-state ratio $M/L$",
        base,
        candidate_counts=candidate_counts,
    )
    axes[0, 0].set_title("Quality approaches token baseline as $M/L\\to1$")

    plot_ppl_panel(
        axes[0, 1],
        rows,
        "L_over_M",
        r"Compression factor $L/M$",
        base,
        candidate_counts=candidate_counts,
    )
    axes[0, 1].set_title("Compression factor view")

    plot_candidate_panel(axes[1, 0], rows, base, candidate_counts)
    axes[1, 0].set_title("Candidate-support diagnostic")

    plot_vram_panel(axes[1, 1], rows, base)
    axes[1, 1].set_title("Memory tradeoff")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
