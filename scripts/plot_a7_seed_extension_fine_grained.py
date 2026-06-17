#!/usr/bin/env python3
"""Plot the A7 quality-efficiency frontier across backend families."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "out"
    / "paper_integrated_evidence"
    / "tables"
    / "a7_backend_family_empty_only_augmented_summary.tsv"
)
PLOTS = ROOT / "out" / "final_paper_bundle" / "plots" / "main"
PLOTS.mkdir(parents=True, exist_ok=True)
OUT = PLOTS / "fig_a7_seed_extension_fine_grained.png"

T95_BY_N = {
    2: 12.706204736432095,
    3: 4.302652729911275,
    4: 3.182446305284263,
    5: 2.7764451051977987,
}
T95_BY_DF = {
    4: 2.7764451051977987,
    6: 2.4469118511449692,
    8: 2.306004135204166,
}

BACKENDS = [
    ("dense", "Dense exact", "#1b6ca8"),
    ("sparse", "Sparse local-band", "#d95f02"),
    ("linear", "Linear landmark", "#2f7d32"),
]
TOPOLOGIES = [("1", "1"), ("2", "1"), ("3", "1"), ("2", "2"), ("4", "2"), ("8", "4"), ("16", "8"), ("32", "16")]

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.family": "serif",
    "font.size": 10.5,
    "axes.titlesize": 11.2,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9.2,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
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
    return T95_BY_N.get(n, 1.96) * f(row, "std_final_val_ppl") / math.sqrt(n)


def diff_ci95(row: dict[str, str], base: dict[str, str]) -> float:
    n_set = int(row["n"])
    n_base = int(base["n"])
    se = math.sqrt(
        (f(row, "std_final_val_ppl") ** 2) / n_set
        + (f(base, "std_final_val_ppl") ** 2) / n_base
    )
    return T95_BY_DF.get(n_set + n_base - 2, 1.96) * se


def baseline_for(rows: list[dict[str, str]], backend: str) -> dict[str, str]:
    matches = [
        r for r in rows
        if r["backend_family"] == backend and r["implementation"] == "baseline_token"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one baseline row for {backend}, got {len(matches)}")
    return matches[0]


def set_row(rows: list[dict[str, str]], backend: str, w: str, s: str) -> dict[str, str]:
    matches = [
        r for r in rows
        if (
            r["backend_family"] == backend
            and r["implementation"] == "set_only"
            and r["w"] == w
            and r["s"] == s
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one set row for {backend}, w={w}, s={s}; got {len(matches)}")
    return matches[0]


def topology_label(row: dict[str, str]) -> str:
    compression = f(row, "L_over_M")
    comp_text = f"{compression:.3f}x" if compression < 1.01 else f"{compression:.2f}x"
    return (
        f"({row['w']},{row['s']})\n"
        rf"$L/M$={comp_text}"
    )


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return nondominated points for minimizing both memory and delta-PPL."""
    frontier: list[tuple[float, float]] = []
    for x, y in sorted(points):
        if not frontier or y < min(v for _, v in frontier):
            frontier.append((x, y))
    return frontier


def label_offset(row: dict[str, str]) -> tuple[int, int]:
    key = (int(row["w"]), int(row["s"]))
    offsets = {
        (1, 1): (5, -18),
        (2, 1): (6, -2),
        (3, 1): (5, 9),
        (2, 2): (5, 9),
        (4, 2): (6, -14),
        (8, 4): (6, 8),
        (16, 8): (5, -17),
        (32, 16): (5, 8),
    }
    return offsets.get(key, (5, 5))


def main() -> None:
    rows = read_tsv(SUMMARY)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.9), constrained_layout=True)

    for col, (backend, title, color) in enumerate(BACKENDS):
        base = baseline_for(rows, backend)
        set_rows = [set_row(rows, backend, w, s) for w, s in TOPOLOGIES]
        base_mean = f(base, "mean_final_val_ppl")
        base_vram = f(base, "mean_peak_vram_mib")
        xs = [100.0 * f(r, "mean_peak_vram_mib") / base_vram for r in set_rows]
        diffs = [f(r, "mean_final_val_ppl") - base_mean for r in set_rows]
        diff_cis = [diff_ci95(r, base) for r in set_rows]

        ax = axes[col]
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.35)
        ax.axvline(100.0, color="#777777", linestyle=":", linewidth=1.1)
        ax.scatter([100.0], [0.0], marker="*", s=90, color="#333333", zorder=5, label="Matched token")
        frontier = pareto_frontier([(100.0, 0.0), *zip(xs, diffs)])
        ax.plot(
            [x for x, _ in frontier],
            [y for _, y in frontier],
            color="#555555",
            linewidth=1.15,
            alpha=0.62,
            label="Nondominated frontier",
        )
        ax.errorbar(
            xs,
            diffs,
            yerr=diff_cis,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1.7,
            label="Set empty_only",
        )
        for row, x, diff in zip(set_rows, xs, diffs):
            dx, dy = label_offset(row)
            ax.annotate(
                topology_label(row),
                (x, diff),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.2,
                ha="right" if dx < 0 else "left",
            )
        ax.set_title(title)
        ax.set_ylabel(r"Set minus token $\Delta$PPL")
        ax.set_xlabel("Peak VRAM (% of matched token)")

    handles = [
        plt.Line2D([0], [0], color="#333333", marker="*", linestyle="", markersize=9),
        plt.Line2D([0], [0], color="#555555", linewidth=1.15),
        plt.Line2D([0], [0], color="#1b6ca8", marker="o", linestyle="", markersize=6),
    ]
    labels = [
        "Matched token baseline",
        "Nondominated memory-quality frontier",
        "Set empty_only (95% CI; 3-5 seeds)",
    ]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
