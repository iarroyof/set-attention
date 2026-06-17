#!/usr/bin/env python3
"""Plot A7 cross-backend empty_only calibration results."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_AUGMENTED = (
    ROOT
    / "out"
    / "paper_integrated_evidence"
    / "tables"
    / "a7_backend_family_empty_only_augmented_summary.tsv"
)
SUMMARY_BASE = ROOT / "out" / "paper_integrated_evidence" / "tables" / "a7_backend_family_empty_only_summary.tsv"
SUMMARY = SUMMARY_AUGMENTED if SUMMARY_AUGMENTED.exists() else SUMMARY_BASE
PLOTS = ROOT / "out" / "final_paper_bundle" / "plots" / "main"
PLOTS.mkdir(parents=True, exist_ok=True)
OUT = PLOTS / "fig_a7_backend_family_compression.png"

T95_N3 = 4.302652729911275
T95_BY_N = {
    2: 12.706204736432095,
    3: T95_N3,
    4: 3.182446305284263,
    5: 2.7764451051977987,
}
BACKENDS = [
    ("dense", "Dense exact"),
    ("sparse", "Sparse local-band"),
    ("linear", "Linear landmark"),
]

COLORS = {
    "dense": "#1b6ca8",
    "sparse": "#d95f02",
    "linear": "#2f7d32",
}

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.family": "serif",
    "font.size": 10.5,
    "axes.titlesize": 11.2,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.3,
    "ytick.labelsize": 9.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.7,
})


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def f(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "NA")
    if raw in {"", "NA", "None", None}:
        return float("nan")
    return float(raw)


def ci95(row: dict[str, str]) -> float:
    n = int(row["n"])
    if n <= 1:
        return 0.0
    std = f(row, "std_final_val_ppl")
    return T95_BY_N.get(n, 1.96) * std / math.sqrt(n)


def is_token(row: dict[str, str]) -> bool:
    return row["implementation"] == "baseline_token"


def topology_label(row: dict[str, str]) -> str:
    return f"w={row['w']}, s={row['s']}"


def label_offset(row: dict[str, str], xkey: str) -> tuple[int, int]:
    w, s = int(row["w"]), int(row["s"])
    common = {
        (1, 1): (5, -13),
        (2, 1): (5, -4),
        (3, 1): (5, 8),
        (2, 2): (5, -14),
        (4, 2): (5, 6),
        (8, 4): (5, -14),
        (16, 8): (5, 6),
        (32, 16): (5, 6),
    }
    if xkey == "L_over_M":
        common = {
            **common,
            (16, 8): (5, 8),
            (32, 16): (5, 0),
        }
    if xkey == "mean_candidate_count":
        common = {
            **common,
            (1, 1): (5, -13),
            (2, 1): (5, -6),
            (3, 1): (5, 7),
            (2, 2): (5, -14),
            (4, 2): (5, 6),
            (8, 4): (5, -14),
            (16, 8): (5, 7),
            (32, 16): (5, 0),
        }
    return common.get((w, s), (5, 5))


def annotate(ax, row: dict[str, str], x: float, y: float, xkey: str) -> None:
    dx, dy = label_offset(row, xkey)
    ax.annotate(
        topology_label(row),
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.4,
        ha="right" if dx < 0 else "left",
    )


def baseline_for(rows: list[dict[str, str]], backend_family: str) -> dict[str, str]:
    matches = [r for r in rows if r["backend_family"] == backend_family and is_token(r)]
    if len(matches) != 1:
        raise ValueError(f"expected one token-baseline summary row for {backend_family}, got {len(matches)}")
    return matches[0]


def set_rows_for(rows: list[dict[str, str]], backend_family: str) -> list[dict[str, str]]:
    matches = [
        r for r in rows
        if r["backend_family"] == backend_family and r["implementation"] == "set_only"
    ]
    return sorted(matches, key=lambda r: f(r, "M_over_L"))


def draw_panel(ax, rows: list[dict[str, str]], base: dict[str, str], backend_family: str, xkey: str) -> None:
    color = COLORS[backend_family]
    xs = [f(r, xkey) for r in rows]
    ys = [f(r, "mean_final_val_ppl") for r in rows]
    yerr = [ci95(r) for r in rows]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    yerr = [yerr[i] for i in order]
    sorted_rows = [rows[i] for i in order]

    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        fmt="o",
        color=color,
        ecolor=color,
        alpha=0.96,
        elinewidth=1.8,
        capsize=3,
        label="Set empty_only (95% CI; 3-5 seeds)",
    )
    for row, x, y in zip(sorted_rows, xs, ys):
        annotate(ax, row, x, y, xkey)

    bmean = f(base, "mean_final_val_ppl")
    bci = ci95(base)
    ax.axhline(bmean, color="#333333", linestyle="--", linewidth=1.35, label="Matched token baseline")
    ax.axhspan(bmean - bci, bmean + bci, color="#444444", alpha=0.10, linewidth=0)
    ax.set_ylabel("Validation perplexity")
    if xkey == "M_over_L":
        ax.set_xlabel(r"Set-state ratio $M/L$")
    elif xkey == "L_over_M":
        ax.set_xlabel(r"Compression factor $L/M$")
    elif xkey == "mean_candidate_count":
        ax.set_xlabel("Mean candidate count")
    else:
        ax.set_xlabel(xkey)


def main() -> None:
    rows = read_tsv(SUMMARY)
    fig, axes = plt.subplots(3, 3, figsize=(15.2, 10.7), constrained_layout=True)
    for col, (backend_family, title) in enumerate(BACKENDS):
        base = baseline_for(rows, backend_family)
        set_rows = set_rows_for(rows, backend_family)
        draw_panel(axes[0, col], set_rows, base, backend_family, "M_over_L")
        axes[0, col].set_title(title)
        draw_panel(axes[1, col], set_rows, base, backend_family, "L_over_M")
        draw_panel(axes[2, col], set_rows, base, backend_family, "mean_candidate_count")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
