#!/usr/bin/env python3
"""Plot post-A1 causal LM diagnostic figures with seed dispersion."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
PLOTS = ROOT / "out" / "final_paper_bundle" / "plots" / "main"
PLOTS.mkdir(parents=True, exist_ok=True)

T95 = {2: 12.706204736432095, 3: 4.302652729911275}

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.family": "serif",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.8,
})

COLORS = {
    "Token Dense": "#555555",
    "Token Sparse": "#8c6d31",
    "Token Linear": "#7f7f7f",
    "Set Dense": "#1b6ca8",
    "Set Sparse": "#d95f02",
    "Set Linear": "#2f7d32",
}

MARKERS = {
    "Token Dense": "X",
    "Token Sparse": "P",
    "Token Linear": "D",
    "Set Dense": "o",
    "Set Sparse": "s",
    "Set Linear": "^",
}

LINESTYLES = {
    "Token Dense": "--",
    "Token Sparse": "--",
    "Token Linear": "--",
    "Set Dense": "-",
    "Set Sparse": "-",
    "Set Linear": "-",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def num(row: dict[str, str], key: str) -> float:
    value = row.get(key, "NA")
    if value in {"", "NA", "None", None}:
        return float("nan")
    return float(value)


def finite(value: float) -> bool:
    return math.isfinite(value)


def ci95(values: list[float]) -> float:
    values = [v for v in values if finite(v)]
    n = len(values)
    if n <= 1:
        return 0.0
    return T95.get(n, 1.96) * stdev(values) / math.sqrt(n)


def sem_band(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if finite(v)]
    if not values:
        return float("nan"), 0.0
    return mean(values), ci95(values)


def group(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k, "") for k in keys)].append(row)
    return out


def draw_line_with_seed_points(
    ax,
    x: list[float],
    y: list[float],
    yerr: list[float],
    seed_x: list[float],
    seed_y: list[float],
    label: str,
    *,
    jitter: float = 0.0,
    alpha_band: float = 0.18,
) -> None:
    order = sorted(range(len(x)), key=lambda i: x[i])
    x = [x[i] for i in order]
    y = [y[i] for i in order]
    yerr = [yerr[i] for i in order]
    color = COLORS[label]
    marker = MARKERS[label]
    ax.plot(x, y, marker=marker, color=color, linestyle=LINESTYLES[label], label=label)
    if any(e > 0 for e in yerr):
        ax.fill_between(
            x,
            [v - e for v, e in zip(y, yerr)],
            [v + e for v, e in zip(y, yerr)],
            color=color,
            alpha=alpha_band,
            linewidth=0,
        )
    if seed_x and seed_y:
        ax.scatter(
            [v + jitter for v in seed_x],
            seed_y,
            marker=marker,
            s=20,
            color=color,
            alpha=0.34,
            linewidths=0,
            zorder=2,
        )


def add_unique_legend(fig, axes, **kwargs) -> None:
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(unique.values(), unique.keys(), frameon=False, **kwargs)


def lr_family_rows() -> dict[str, list[dict[str, str]]]:
    family = read_tsv(TABLES / "a2_lrnorm_family_slice_all_runs.tsv")
    controls = read_tsv(TABLES / "a2_baseline_controls_all_runs.tsv")
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in family:
        if row["D"] != "384" or row["d_ff"] != "1536":
            continue
        label = {
            "Baseline token": "Token Dense",
            "Set Dense": "Set Dense",
            "Set Sparse": "Set Sparse",
            "Set Linear": "Set Linear",
        }.get(row["family"])
        if label:
            rows[label].append(row)
    for row in controls:
        if row["D"] != "384" or row["d_ff"] != "1536":
            continue
        label = {
            "baseline_sparse_local_band": "Token Sparse",
            "baseline_linear_landmark": "Token Linear",
        }.get(row["family_slug"])
        if label:
            rows[label].append(row)
    return rows


def plot_lr_sweep() -> Path:
    rows_by_label = lr_family_rows()
    labels = ["Token Dense", "Token Sparse", "Token Linear", "Set Dense", "Set Sparse", "Set Linear"]
    lr_order = ["1e-4", "2e-4", "3e-4", "5e-4", "7e-4"]
    lr_index = {lr: i for i, lr in enumerate(lr_order)}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.05), constrained_layout=True)

    for idx, label in enumerate(labels):
        rows = rows_by_label.get(label, [])
        grouped = group(rows, ("lr",))
        xs: list[float] = []
        ppl: list[float] = []
        ppl_ci: list[float] = []
        vram: list[float] = []
        vram_ci: list[float] = []
        time: list[float] = []
        time_ci: list[float] = []
        seed_lr: list[float] = []
        seed_ppl: list[float] = []
        seed_vram: list[float] = []
        seed_time: list[float] = []
        for (lr,), sub in grouped.items():
            x = float(lr_index[lr])
            xs.append(x)
            seed_lr.extend([x] * len(sub))
            seed_ppl.extend([num(r, "final_val_ppl") for r in sub])
            seed_vram.extend([num(r, "peak_vram_mib") / 1024.0 for r in sub])
            seed_time.extend([num(r, "time_per_epoch_s") for r in sub])
            m, c = sem_band([num(r, "final_val_ppl") for r in sub])
            ppl.append(m)
            ppl_ci.append(c)
            m, c = sem_band([num(r, "peak_vram_mib") / 1024.0 for r in sub])
            vram.append(m)
            vram_ci.append(c)
            m, c = sem_band([num(r, "time_per_epoch_s") for r in sub])
            time.append(m)
            time_ci.append(c)
        jitter = (idx - 2.5) * 0.000004
        draw_line_with_seed_points(axes[0], xs, ppl, ppl_ci, seed_lr, seed_ppl, label, jitter=jitter)
        draw_line_with_seed_points(axes[1], xs, vram, vram_ci, seed_lr, seed_vram, label, jitter=jitter)
        draw_line_with_seed_points(axes[2], xs, time, time_ci, seed_lr, seed_time, label, jitter=jitter)

    for ax in axes:
        ax.set_xlabel("Learning rate")
        ax.set_xticks(list(range(len(lr_order))))
        ax.set_xticklabels(lr_order)
    axes[0].set_title("LR sweep: validation quality")
    axes[0].set_ylabel("Validation perplexity")
    axes[1].set_title("LR sweep: peak memory")
    axes[1].set_ylabel("Peak VRAM (GiB)")
    axes[2].set_title("LR sweep: runtime")
    axes[2].set_ylabel("Time / epoch (s)")
    add_unique_legend(fig, axes, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08))
    out = PLOTS / "fig_causal_lr_sweep_dispersion.png"
    fig.savefig(out, bbox_inches="tight")
    return out


def window_rows() -> dict[str, list[dict[str, str]]]:
    set_rows = read_tsv(TABLES / "a3_window_sweep_all_runs.tsv")
    base_rows = read_tsv(TABLES / "a3_window_baseline_controls_all_runs.tsv")
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in set_rows:
        label = {
            "dense_exact": "Set Dense",
            "sparse_local_band": "Set Sparse",
            "linear_landmark": "Set Linear",
        }.get(row["family_slug"])
        if label:
            rows[label].append(row)
    for row in base_rows:
        label = {
            "baseline_sparse_local_band": "Token Sparse",
            "baseline_linear_landmark": "Token Linear",
        }.get(row["family_slug"])
        if label:
            rows[label].append(row)
    return rows


def plot_window_sweep() -> Path:
    rows_by_label = window_rows()
    labels_quality = ["Token Sparse", "Token Linear", "Set Dense", "Set Sparse", "Set Linear"]
    labels_set = ["Set Dense", "Set Sparse", "Set Linear"]
    fig, axes = plt.subplots(2, 2, figsize=(12.9, 7.1), constrained_layout=True)

    for idx, label in enumerate(labels_quality):
        grouped = group(rows_by_label.get(label, []), ("w",))
        xs: list[float] = []
        ppl: list[float] = []
        ppl_ci: list[float] = []
        vram: list[float] = []
        vram_ci: list[float] = []
        seed_x: list[float] = []
        seed_ppl: list[float] = []
        seed_vram: list[float] = []
        for (w,), sub in grouped.items():
            x = float(w)
            xs.append(x)
            seed_x.extend([x] * len(sub))
            seed_ppl.extend([num(r, "final_val_ppl") for r in sub])
            seed_vram.extend([num(r, "peak_vram_mib") / 1024.0 for r in sub])
            m, c = sem_band([num(r, "final_val_ppl") for r in sub])
            ppl.append(m)
            ppl_ci.append(c)
            m, c = sem_band([num(r, "peak_vram_mib") / 1024.0 for r in sub])
            vram.append(m)
            vram_ci.append(c)
        jitter = (idx - 2.0) * 0.08
        draw_line_with_seed_points(axes[0, 0], xs, ppl, ppl_ci, seed_x, seed_ppl, label, jitter=jitter)
        draw_line_with_seed_points(axes[0, 1], xs, vram, vram_ci, seed_x, seed_vram, label, jitter=jitter)

    for idx, label in enumerate(labels_set):
        grouped = group(rows_by_label.get(label, []), ("w",))
        xs, entropy, entropy_ci, top1, top1_ci = [], [], [], [], []
        seed_x, seed_entropy, seed_top1 = [], [], []
        for (w,), sub in grouped.items():
            x = float(w)
            xs.append(x)
            seed_x.extend([x] * len(sub))
            seed_entropy.extend([num(r, "router_entropy_norm") for r in sub])
            seed_top1.extend([num(r, "router_top1_weight") for r in sub])
            m, c = sem_band([num(r, "router_entropy_norm") for r in sub])
            entropy.append(m)
            entropy_ci.append(c)
            m, c = sem_band([num(r, "router_top1_weight") for r in sub])
            top1.append(m)
            top1_ci.append(c)
        jitter = (idx - 1.0) * 0.08
        draw_line_with_seed_points(axes[1, 0], xs, entropy, entropy_ci, seed_x, seed_entropy, label, jitter=jitter)
        draw_line_with_seed_points(axes[1, 1], xs, top1, top1_ci, seed_x, seed_top1, label, jitter=jitter)

    for ax in axes.flat:
        ax.set_xlabel(r"Window size $w$ at fixed stride $s=4$")
        ax.set_xticks([6, 8, 12, 16, 20, 24])
    axes[0, 0].set_title("Window sweep: quality")
    axes[0, 0].set_ylabel("Validation perplexity")
    axes[0, 1].set_title("Window sweep: memory")
    axes[0, 1].set_ylabel("Peak VRAM (GiB)")
    axes[1, 0].set_title("Set routing entropy")
    axes[1, 0].set_ylabel("Normalized routing entropy")
    axes[1, 1].set_title("Set routing concentration")
    axes[1, 1].set_ylabel("Router top-1 weight")
    add_unique_legend(fig, axes.flat, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.035))
    out = PLOTS / "fig_causal_window_diagnostics_dispersion.png"
    fig.savefig(out, bbox_inches="tight")
    return out


def pooltau_rows() -> dict[str, list[dict[str, str]]]:
    rows = read_tsv(TABLES / "a3_pooltau_sweep_all_runs.tsv")
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = {
            "dense_exact": "Set Dense",
            "sparse_local_band": "Set Sparse",
            "linear_landmark": "Set Linear",
        }.get(row["family_slug"])
        if label:
            out[label].append(row)
    return out


def plot_pooltau_sweep() -> Path:
    rows_by_label = pooltau_rows()
    labels = ["Set Dense", "Set Sparse", "Set Linear"]
    tau_ticks = sorted({
        float(row["tau_pool"])
        for rows in rows_by_label.values()
        for row in rows
        if row.get("tau_pool") not in {"", "NA", None}
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.9, 7.1), constrained_layout=True)
    metrics = [
        ("final_val_ppl", "Validation perplexity", "Pooling sweep: quality"),
        ("pooling_effective_support", "Pooling effective support", "Effective support"),
        ("grad_ratio_total_rho_pa", r"End-to-end transport $\rho_{pa}$", "Gradient transport"),
        ("router_entropy_norm", "Normalized routing entropy", "Routing entropy"),
    ]
    for idx, label in enumerate(labels):
        grouped = group(rows_by_label.get(label, []), ("tau_pool",))
        for ax, (metric, ylabel, title) in zip(axes.flat, metrics):
            xs, means, errs, seed_x, seed_y = [], [], [], [], []
            for (tau,), sub in grouped.items():
                x = float(tau)
                vals = [num(r, metric) for r in sub]
                xs.append(x)
                m, c = sem_band(vals)
                means.append(m)
                errs.append(c)
                seed_x.extend([x] * len(vals))
                seed_y.extend(vals)
            jitter = (idx - 1.0) * 0.002
            draw_line_with_seed_points(ax, xs, means, errs, seed_x, seed_y, label, jitter=jitter)
            ax.set_xlabel(r"Pooling temperature $\tau_{\mathrm{pool}}$")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(tau_ticks)
            ax.set_xticklabels([f"{x:g}" for x in tau_ticks])
            if tau_ticks:
                pad = max((tau_ticks[-1] - tau_ticks[0]) * 0.03, 0.01)
                ax.set_xlim(tau_ticks[0] - pad, tau_ticks[-1] + pad)
    add_unique_legend(fig, axes.flat, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.035))
    out = PLOTS / "fig_causal_pooltau_diagnostics_dispersion.png"
    fig.savefig(out, bbox_inches="tight")
    return out


def main() -> None:
    for path in [plot_lr_sweep(), plot_window_sweep(), plot_pooltau_sweep()]:
        print(path)


if __name__ == "__main__":
    main()
