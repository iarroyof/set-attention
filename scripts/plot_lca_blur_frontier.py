#!/usr/bin/env python3
"""Plot the LCA L=1024 blur-allocation accuracy-memory frontier."""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from plot_lca_l4096_trajectories import Canvas
except ModuleNotFoundError:  # Allow import as scripts.plot_lca_blur_frontier.
    from scripts.plot_lca_l4096_trajectories import Canvas


ROOT = Path(__file__).resolve().parents[1]
BLUR_TSV = ROOT / "out/lca_cmp/prefixblur/prefixblur_blue.tsv"
TOKEN_TSV = ROOT / "out/lca_cmp/prefix3/prefix3_blue.tsv"
OUT = ROOT / "out/final_paper_bundle/plots/main/fig_lca_blur_frontier.pdf"

BLURS = (25, 50, 75, 100)
T_CRIT_95_DF2 = 4.3026527299
INK = (0.12, 0.12, 0.12)
GRID = (0.88, 0.88, 0.88)
TOKEN = (0.38, 0.38, 0.38)
SET = (0.12, 0.32, 0.78)
PATH = (0.18, 0.52, 0.46)


def summarize(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    ci = T_CRIT_95_DF2 * statistics.stdev(values) / len(values) ** 0.5
    return mean, ci


def read_rows() -> tuple[dict[int, dict[str, list[float]]], dict[str, list[float]]]:
    blur: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"acc": [], "vram": []}
    )
    with BLUR_TSV.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            match = re.search(r"prefixblur_b(\d+)_", row["label"])
            if not match:
                continue
            value = int(match.group(1))
            blur[value]["acc"].append(float(row["val_acc"]))
            blur[value]["vram"].append(float(row["peak_vram_mib"]))

    token = {"acc": [], "vram": []}
    with TOKEN_TSV.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["label"].startswith("prefix3_token_L1024_seed"):
                token["acc"].append(float(row["val_acc"]))
                token["vram"].append(float(row["peak_vram_mib"]))

    if set(blur) != set(BLURS) or any(len(blur[b]["acc"]) != 3 for b in BLURS):
        raise ValueError("expected three rows for every registered blur allocation")
    if len(token["acc"]) != 3:
        raise ValueError("expected three matched token-prefix rows")
    return blur, token


def main() -> None:
    blur, token = read_rows()
    c = Canvas(width=760, height=400)
    left, bottom, width, height = 92.0, 62.0, 580.0, 265.0
    xmin, xmax = 1650.0, 3300.0
    ymin, ymax = 0.70, 1.03

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * width

    def sy(value: float) -> float:
        return bottom + (value - ymin) / (ymax - ymin) * height

    c.text(165, 371, "LCA blur allocation selects an interior quality-memory operating point", 13)
    c.text(270, 20, "peak training VRAM (MiB)", 10)
    c.text(12, 340, "validation accuracy", 10)

    c.color(*INK)
    c.line_width(0.9)
    c.line(left, bottom, left + width, bottom)
    c.line(left, bottom, left, bottom + height)
    for value in (1800, 2100, 2400, 2700, 3000, 3300):
        x = sx(value)
        c.color(*GRID)
        c.line_width(0.5)
        c.line(x, bottom, x, bottom + height)
        c.color(*INK)
        c.text(x - 14, bottom - 17, str(value), 8)
    for value in (0.70, 0.80, 0.90, 1.00):
        y = sy(value)
        c.color(*GRID)
        c.line_width(0.5)
        c.line(left, y, left + width, y)
        c.color(*INK)
        c.text(left - 36, y - 3, f"{value:.2f}", 8)

    points = []
    summaries = {}
    for value in BLURS:
        acc, ci = summarize(blur[value]["acc"])
        vram = statistics.fmean(blur[value]["vram"])
        summaries[value] = (acc, ci, vram)
        points.append((sx(vram), sy(acc)))

    c.color(*PATH)
    c.line_width(1.5)
    c.polyline(points)

    for value, (x, y) in zip(BLURS, points):
        acc, ci, _ = summaries[value]
        c.color(*SET)
        c.line_width(0.9)
        c.line(x, sy(max(ymin, acc - ci)), x, sy(min(ymax, acc + ci)))
        c.line(x - 4, sy(max(ymin, acc - ci)), x + 4, sy(max(ymin, acc - ci)))
        c.line(x - 4, sy(min(ymax, acc + ci)), x + 4, sy(min(ymax, acc + ci)))
        c.fill_rect(x - 3, y - 3, 6, 6)
        dx = -10 if value != 100 else 7
        dy = 10 if value in (25, 50, 75) else -13
        c.text(x + dx, y + dy, f"b{value}", 9)

    token_acc, token_ci = summarize(token["acc"])
    token_vram = statistics.fmean(token["vram"])
    tx, ty = sx(token_vram), sy(token_acc)
    c.color(*TOKEN)
    c.line_width(1.0)
    c.line(tx, sy(max(ymin, token_acc - token_ci)), tx, sy(min(ymax, token_acc + token_ci)))
    c.line(tx - 5, sy(max(ymin, token_acc - token_ci)), tx + 5, sy(max(ymin, token_acc - token_ci)))
    c.line(tx - 5, sy(min(ymax, token_acc + token_ci)), tx + 5, sy(min(ymax, token_acc + token_ci)))
    c.fill_polygon([(tx, ty + 5), (tx + 5, ty), (tx, ty - 5), (tx - 5, ty)])
    c.text(tx + 9, ty + 5, "token", 9)

    c.color(*INK)
    c.text(395, 138, "b75: 0.923 mean accuracy at -12.5% VRAM vs token", 9)
    c.text(395, 123, "b100: minimum memory, but lower mean accuracy", 9)

    c.color(*PATH)
    c.line_width(1.5)
    c.line(112, 349, 135, 349)
    c.text_colored(141, 346, "blur path: b25 to b100", SET, 8)
    c.color(*TOKEN)
    c.fill_polygon([(335, 354), (340, 349), (335, 344), (330, 349)])
    c.text(346, 346, "matched token control", 8)
    c.color(*INK)
    c.text(92, 7, "L=1024, B=4, 2000 updates, prefix supervision, full routing, n=3; whiskers: 95% t-CI", 8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(c.pdf_bytes())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
