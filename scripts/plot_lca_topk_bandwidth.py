#!/usr/bin/env python3
"""Plot the LCA routing-bandwidth quality and memory controls.

The b75 series uses the three-seed L=1024 top-k sweep. The token reference is
the matched three-seed prefix-supervision control. Lines show seed means and
bands/whiskers show 95% Student-t confidence intervals. Output is a compact
vector PDF that does not require matplotlib.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from plot_lca_l4096_trajectories import Canvas
except ModuleNotFoundError:  # Allow import as scripts.plot_lca_topk_bandwidth.
    from scripts.plot_lca_l4096_trajectories import Canvas


ROOT = Path(__file__).resolve().parents[1]
TOPK_TSV = ROOT / "out/lca_cmp/topksweep/topksweep_blue.tsv"
TOKEN_TSV = ROOT / "out/lca_cmp/prefix3/prefix3_blue.tsv"
OUT = ROOT / "out/final_paper_bundle/plots/main/fig_lca_topk_bandwidth.pdf"

TOPKS = (16, 32, 64, 128, 256, 512, 1023)
TOPK_LABELS = ("16", "32", "64", "128", "256", "512", "full")
T_CRIT_95_DF2 = 4.3026527299

INK = (0.12, 0.12, 0.12)
GRID = (0.88, 0.88, 0.88)
TOKEN = (0.38, 0.38, 0.38)
B75 = (0.12, 0.32, 0.78)
MEMORY = (0.18, 0.52, 0.46)


def summary(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    half_width = T_CRIT_95_DF2 * statistics.stdev(values) / len(values) ** 0.5
    return mean, half_width


def read_topk() -> dict[int, dict[str, list[float]]]:
    result: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"acc": [], "vram": []}
    )
    with TOPK_TSV.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            match = re.search(r"topk(\d+)", row["label"])
            if not match:
                continue
            topk = int(match.group(1))
            result[topk]["acc"].append(float(row["val_acc"]))
            result[topk]["vram"].append(float(row["peak_vram_mib"]))
    if set(result) != set(TOPKS) or any(len(result[k]["acc"]) != 3 for k in TOPKS):
        raise ValueError("expected exactly three rows for every registered top-k")
    return result


def read_token() -> dict[str, list[float]]:
    result = {"acc": [], "vram": []}
    with TOKEN_TSV.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["label"].startswith("prefix3_token_L1024_seed"):
                result["acc"].append(float(row["val_acc"]))
                result["vram"].append(float(row["peak_vram_mib"]))
    if len(result["acc"]) != 3:
        raise ValueError("expected three matched token-prefix rows")
    return result


def blend(rgb: tuple[float, float, float], fraction: float) -> tuple[float, float, float]:
    return tuple(channel + (1.0 - channel) * fraction for channel in rgb)


def main() -> None:
    rows = read_topk()
    token = read_token()
    c = Canvas(width=760, height=400)

    bottom, panel_h, panel_w = 72.0, 250.0, 285.0
    lefts = (62.0, 420.0)
    top = bottom + panel_h
    xs = [lefts[0] + i * panel_w / (len(TOPKS) - 1) for i in range(len(TOPKS))]

    def x_for(panel_left: float, index: int) -> float:
        return panel_left + index * panel_w / (len(TOPKS) - 1)

    def draw_axes(
        left: float,
        title: str,
        ymin: float,
        ymax: float,
        ticks: tuple[float, ...],
        formatter,
    ) -> None:
        c.color(*INK)
        c.line_width(0.9)
        c.line(left, bottom, left + panel_w, bottom)
        c.line(left, bottom, left, top)
        c.text(left + 78, top + 7, title, 11)
        c.text(left + 85, bottom - 43, "retained routing candidates", 9)
        for value in ticks:
            y = bottom + (value - ymin) / (ymax - ymin) * panel_h
            c.color(*GRID)
            c.line_width(0.5)
            c.line(left, y, left + panel_w, y)
            c.color(*INK)
            c.line(left - 4, y, left, y)
            c.text(left - 42, y - 3, formatter(value), 8)
        for index, label in enumerate(TOPK_LABELS):
            x = x_for(left, index)
            c.color(*GRID)
            c.line_width(0.4)
            c.line(x, bottom, x, top)
            c.color(*INK)
            c.text(x - (8 if label != "full" else 10), bottom - 17, label, 8)

    c.text(112, 370, "Routing bandwidth: quality changes while dense-score memory does not", 13)

    # Accuracy panel.
    acc_min, acc_max = 0.70, 1.01
    draw_axes(lefts[0], "validation accuracy", acc_min, acc_max,
              (0.70, 0.80, 0.90, 1.00), lambda value: f"{value:.2f}")

    def acc_y(value: float) -> float:
        return bottom + (value - acc_min) / (acc_max - acc_min) * panel_h

    token_acc, token_acc_ci = summary(token["acc"])
    c.color(*blend(TOKEN, 0.82))
    c.fill_rect(lefts[0], acc_y(max(acc_min, token_acc - token_acc_ci)), panel_w,
                acc_y(min(acc_max, token_acc + token_acc_ci)) -
                acc_y(max(acc_min, token_acc - token_acc_ci)))
    c.color(*TOKEN)
    c.line_width(1.4)
    c.line(lefts[0], acc_y(token_acc), lefts[0] + panel_w, acc_y(token_acc))

    acc_points = []
    lower = []
    upper = []
    for index, topk in enumerate(TOPKS):
        mean, ci = summary(rows[topk]["acc"])
        x = x_for(lefts[0], index)
        acc_points.append((x, acc_y(mean)))
        lower.append((x, acc_y(max(acc_min, mean - ci))))
        upper.append((x, acc_y(min(acc_max, mean + ci))))
    c.color(*blend(B75, 0.82))
    c.fill_polygon(lower + list(reversed(upper)))
    c.color(*B75)
    c.line_width(2.0)
    c.polyline(acc_points)
    for x, y in acc_points:
        c.fill_rect(x - 2.3, y - 2.3, 4.6, 4.6)

    # Peak-memory panel.
    mem_min, mem_max = 2300.0, 2730.0
    draw_axes(lefts[1], "peak training VRAM", mem_min, mem_max,
              (2300.0, 2400.0, 2500.0, 2600.0, 2700.0), lambda value: f"{int(value)}")

    def mem_y(value: float) -> float:
        return bottom + (value - mem_min) / (mem_max - mem_min) * panel_h

    token_mem, _ = summary(token["vram"])
    c.color(*TOKEN)
    c.line_width(1.4)
    c.line(lefts[1], mem_y(token_mem), lefts[1] + panel_w, mem_y(token_mem))
    mem_points = []
    for index, topk in enumerate(TOPKS):
        mean, ci = summary(rows[topk]["vram"])
        x = x_for(lefts[1], index)
        y = mem_y(mean)
        mem_points.append((x, y))
        if ci > 0:
            c.color(*MEMORY)
            c.line_width(0.8)
            c.line(x, mem_y(max(mem_min, mean - ci)), x, mem_y(min(mem_max, mean + ci)))
    c.color(*MEMORY)
    c.line_width(2.0)
    c.polyline(mem_points)
    for x, y in mem_points:
        c.fill_rect(x - 2.3, y - 2.3, 4.6, 4.6)
    c.color(*INK)
    c.text(lefts[1] + 58, mem_y(2440), "b75 range: 2347--2408 MiB (<3%)", 8)
    c.text(lefts[1] + 58, mem_y(2415), "dense scores are allocated before top-k", 8)

    # Shared legend and provenance note.
    c.color(*TOKEN)
    c.line_width(1.5)
    c.line(205, 347, 227, 347)
    c.text(232, 344, "matched token control", 8)
    c.color(*B75)
    c.line(342, 347, 364, 347)
    c.text(369, 344, "b75 accuracy", 8)
    c.color(*MEMORY)
    c.line(454, 347, 476, 347)
    c.text(481, 344, "b75 peak VRAM", 8)
    c.color(*INK)
    c.text(62, 8, "L=1024, B=4, 2000 updates, n=3; bands/whiskers: 95% t-CI", 8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(c.pdf_bytes())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
