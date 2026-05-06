#!/usr/bin/env python
"""
Generates:
  - out/final_paper_bundle/plots/main/fig_boundary_time_vs_ppl_with_baseline.png

Sources:
  - out/metrics/paper_action1_boundary_stride.tsv
  - out/metrics/action1A_anchor_baseline_LR1e-4_seed0.csv

Purpose:
  Boundary-family figure:
  validation perplexity vs. training time per epoch for representative
  Set Att boundary points, plus one matched Baseline Att reference point.

Notes:
  - Set Att points are annotated by s, w, and T when available.
  - The baseline point is annotated by D and d_ff.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt


SET_SRC = Path("out/metrics/paper_action1_boundary_stride.tsv")
BASE_SRC = Path("out/metrics/action1A_anchor_baseline_LR1e-4_seed0.csv")
OUT = Path("out/final_paper_bundle/plots/main/fig_boundary_time_vs_ppl_with_baseline.png")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    set_rows = []
    with SET_SRC.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            try:
                set_rows.append(
                    {
                        "w": r.get("model.window_size", ""),
                        "s": r.get("model.stride", ""),
                        "T": r.get("model.router_temperature", ""),
                        "time": float(r["train/time_per_epoch_s"]),
                        "ppl": float(r["val/ppl"]),
                    }
                )
            except Exception:
                continue

    with BASE_SRC.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"No rows found in {BASE_SRC}")

    r = rows[-1]
    baseline = {
        "time": float(r["train/time_per_epoch_s"]),
        "ppl": float(r["val/ppl"]),
        "d_model": r.get("model.d_model", ""),
        "d_ff": r.get("model.dim_feedforward", ""),
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    if set_rows:
        ax.scatter(
            [r["time"] for r in set_rows],
            [r["ppl"] for r in set_rows],
            marker="s",
            label="Set Att",
        )
        for r in set_rows:
            label = f"s={r['s']}, w={r['w']}"
            if r["T"]:
                label += f", T={r['T']}"
            ax.annotate(
                label,
                (r["time"], r["ppl"]),
                xytext=(5, -10),
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
    ax.set_title("Boundary-family time vs. validation perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
