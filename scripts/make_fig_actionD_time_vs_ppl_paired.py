#!/usr/bin/env python
"""
Generates:
  - out/final_paper_bundle/plots/main/fig_actionD_time_vs_ppl_paired.png

Source:
  - out/metrics/actionD_epoch10_summary.tsv

Purpose:
  Paired model-level comparison figure for Action D:
  validation perplexity vs. training time per epoch, with
  Baseline Att and Set Att shown together.

Notes:
  - Point labels show only D and d_ff.
  - Labels are placed with a pair-aware heuristic:
    horizontally close points are preferentially assigned
    opposite left/right label sides, with same-family pairs
    prioritized first.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt


SRC = Path("out/metrics/actionD_epoch10_summary.tsv")
OUT = Path("out/final_paper_bundle/plots/main/fig_actionD_time_vs_ppl_paired.png")


def load_rows():
    rows = []
    with SRC.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            try:
                rows.append(
                    {
                        "impl": r["impl"].strip().lower(),
                        "d_model": int(float(r["model.d_model"])),
                        "d_ff": int(float(r["model.dim_feedforward"])),
                        "time": float(r["train/time_per_epoch_s"]),
                        "ppl": float(r["val/ppl"]),
                    }
                )
            except Exception:
                continue
    return rows


def intersects(bb1, bb2, pad=2):
    return not (
        bb1.x1 + pad < bb2.x0
        or bb1.x0 - pad > bb2.x1
        or bb1.y1 + pad < bb2.y0
        or bb1.y0 - pad > bb2.y1
    )


def choose_offset(ax, fig, x, y, text, offsets, placed_bboxes):
    renderer = fig.canvas.get_renderer()
    best = None
    best_score = None

    for ox, oy in offsets:
        t = ax.annotate(
            text,
            (x, y),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8,
        )
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)
        t.remove()

        overlaps = sum(intersects(bb, old) for old in placed_bboxes)
        dist_penalty = abs(ox) + abs(oy)
        score = overlaps * 1000 + dist_penalty

        if best_score is None or score < best_score:
            best_score = score
            best = (ox, oy, bb)

    return best


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    baseline = [r for r in rows if r["impl"] == "baseline"]
    setonly = [r for r in rows if r["impl"] == "setonly"]

    fig, ax = plt.subplots(figsize=(8.4, 5.8))

    if baseline:
        ax.scatter(
            [r["time"] for r in baseline],
            [r["ppl"] for r in baseline],
            marker="o",
            label="Baseline Att",
        )

    if setonly:
        ax.scatter(
            [r["time"] for r in setonly],
            [r["ppl"] for r in setonly],
            marker="s",
            label="Set Att",
        )

    ax.set_xlabel("Training time per epoch (s)")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Matched Action D comparison: time vs. validation perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    all_points = []
    for r in baseline:
        all_points.append({"family": "baseline", **r})
    for r in setonly:
        all_points.append({"family": "setonly", **r})

    right_offsets = [(6, 4), (8, -6), (10, 8), (12, -10)]
    left_offsets = [(-6, 4), (-8, -6), (-10, 8), (-12, -10)]
    neutral_offsets = [(6, 6), (6, -8), (-6, 6), (-6, -8), (10, 4), (-10, 4)]

    preferred_side = {}
    pairs = []
    n = len(all_points)

    for i in range(n):
        for j in range(i + 1, n):
            p = all_points[i]
            q = all_points[j]
            dx = abs(p["time"] - q["time"])
            dy = abs(p["ppl"] - q["ppl"])
            same_family = p["family"] == q["family"]

            if dx < 2.8 and dy < 140:
                priority = 0 if same_family else 1
                pairs.append((priority, dx, dy, i, j))

    pairs.sort()
    used = set()

    for _, _, _, i, j in pairs:
        if i in used or j in used:
            continue
        if all_points[i]["time"] <= all_points[j]["time"]:
            preferred_side[i] = "left"
            preferred_side[j] = "right"
        else:
            preferred_side[i] = "right"
            preferred_side[j] = "left"
        used.add(i)
        used.add(j)

    placed_bboxes = []
    order = sorted(range(n), key=lambda idx: (-all_points[idx]["ppl"], all_points[idx]["time"]))

    for idx in order:
        p = all_points[idx]
        label = rf"$D={p['d_model']},\, d_{{ff}}={p['d_ff']}$"

        side = preferred_side.get(idx)
        if side == "left":
            offsets = left_offsets + neutral_offsets + right_offsets
        elif side == "right":
            offsets = right_offsets + neutral_offsets + left_offsets
        else:
            offsets = neutral_offsets + right_offsets + left_offsets

        ox, oy, bb = choose_offset(ax, fig, p["time"], p["ppl"], label, offsets, placed_bboxes)
        ax.annotate(
            label,
            (p["time"], p["ppl"]),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8,
        )
        placed_bboxes.append(bb)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
