#!/usr/bin/env python
"""
Generates:
  - out/final_paper_bundle/plots/main/fig_sw_family_time_vs_ppl_with_baseline_labeled.png

Sources:
  - out/metrics/action1_boundary_s3_w16_T1_lr1e4_seed0.csv
  - out/metrics/action0_tempSweep_s4_w16_T1_lr1e4_seed0.csv
  - out/metrics/action1_boundary_s5_w16_T1_lr1e4_seed0.csv
  - out/metrics/action0_learned_s6_w16_temp1.0_LR1e-4_seed0.csv
  - out/metrics/action1A_anchor_setonly_dense_LR1e-4_seed0.csv
  - out/metrics/action1A_anchor_baseline_LR1e-4_seed0.csv

Purpose:
  Composite illustrative family figure:
  validation perplexity vs. training time per epoch for representative
  Set Att operating points drawn from related post-parity sweep families,
  plus one matched Baseline Att reference point.

Notes:
  - Set Att labels include only s and w, since the plotted set-attention points
    are intended to share the same common anchor-family defaults in the figure.
  - Provenance differences, if needed, should be explained in the caption/prose,
    not in the point labels.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

SET_FILES = [
    ("out/metrics/action1_boundary_s3_w16_T1_lr1e4_seed0.csv", "s=3, w=16"),
    ("out/metrics/action0_tempSweep_s4_w16_T1_lr1e4_seed0.csv", "s=4, w=16"),
    ("out/metrics/action1_boundary_s5_w16_T1_lr1e4_seed0.csv", "s=5, w=16"),
    ("out/metrics/action0_learned_s6_w16_temp1.0_LR1e-4_seed0.csv", "s=6, w=16"),
    ("out/metrics/action1A_anchor_setonly_dense_LR1e-4_seed0.csv", "s=8, w=16"),
]

BASELINE_FILE = "out/metrics/action1A_anchor_baseline_LR1e-4_seed0.csv"
OUT = Path("out/final_paper_bundle/plots/main/fig_sw_family_time_vs_ppl_with_baseline_labeled.png")


def last_row(csv_path: str):
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    return rows[-1]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    set_rows = []
    for path, label in SET_FILES:
        r = last_row(path)
        set_rows.append(
            {
                "label": label,
                "time": float(r["train/time_per_epoch_s"]),
                "ppl": float(r["val/ppl"]),
            }
        )

    r = last_row(BASELINE_FILE)
    baseline = {
        "time": float(r["train/time_per_epoch_s"]),
        "ppl": float(r["val/ppl"]),
        "d_model": r.get("model.d_model", ""),
        "d_ff": r.get("model.dim_feedforward", ""),
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.9))

    ax.scatter(
        [r["time"] for r in set_rows],
        [r["ppl"] for r in set_rows],
        marker="s",
        label="Set Att",
    )

    # modest manual spread to reduce overlap while keeping labels close
    offsets = [(5, -10), (5, 6), (5, -12), (5, 8), (5, -10)]
    for r, (ox, oy) in zip(set_rows, offsets):
        ax.annotate(
            r["label"],
            (r["time"], r["ppl"]),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8,
        )

    ax.scatter(
        [baseline["time"]],
        [baseline["ppl"]],
        marker="o",
        label="Baseline Att",
    )
    base_label = "Baseline Att" + "\n" + rf"$D={baseline['d_model']},\, d_{{ff}}={baseline['d_ff']}$"
    ax.annotate(
        base_label,
        (baseline["time"], baseline["ppl"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
    )

    ax.set_xlabel("Training time per epoch (s)")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Representative set-attention family: time vs. validation perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
