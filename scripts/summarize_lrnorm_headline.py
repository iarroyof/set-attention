#!/usr/bin/env python3
"""Summarize the completed LR-normalized headline comparison.

Consumes only local CSV artifacts copied from blue-demon under
out/paper_lr_norm/paper_lr_norm_headline_* and writes compact TSV/LaTeX/plot
outputs with row-level source hashes for manuscript integration.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "out" / "paper_lr_norm"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
LATEX = OUT / "latex"
PLOTS = OUT / "plots"
CHECKS = OUT / "checks"
FINAL_PLOTS = ROOT / "out" / "final_paper_bundle" / "plots" / "main"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_last(path: Path) -> dict[str, str]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    row = dict(rows[-1])
    row["source"] = str(path.relative_to(ROOT))
    row["source_sha256"] = sha256(path)
    return row


def as_float(row: dict[str, str], key: str) -> float | None:
    val = row.get(key, "NA")
    if val in {"", "NA", "nan", "None"}:
        return None
    return float(val)


def label(row: dict[str, str]) -> str:
    return "Baseline" if row.get("model.implementation") == "baseline_token" else "Set Dense"


def spec(row: dict[str, str]) -> tuple[int, int]:
    return int(row["model.d_model"]), int(row["model.dim_feedforward"])


def collect_rows() -> list[dict[str, str]]:
    rows = []
    for path in sorted(SRC_ROOT.glob("paper_lr_norm_headline_*/*.csv")):
        rows.append(read_last(path))
    rows.sort(key=lambda r: (spec(r), label(r), r["training.lr"]))
    return rows


def fmt(val: float | None, digits: int = 1) -> str:
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def compact_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "family": label(row),
        "lr": row["training.lr"],
        "D": row["model.d_model"],
        "d_ff": row["model.dim_feedforward"],
        "val_ppl": fmt(as_float(row, "val/ppl"), 6),
        "train_ppl": fmt(as_float(row, "train/ppl"), 6),
        "time_per_epoch_s": fmt(as_float(row, "train/time_per_epoch_s"), 6),
        "peak_vram_mib": fmt(as_float(row, "train/peak_vram_mib"), 6),
        "candidate_count": fmt(as_float(row, "ausa/router_candidate_count_struct_mean"), 6),
        "routing_entropy_norm": fmt(as_float(row, "ausa/routing_entropy_norm"), 6),
        "router_top1": fmt(as_float(row, "ausa/router_top1_weight"), 6),
        "rho_pa": fmt(as_float(row, "ausa/grad_ratio_total_rho_pa"), 6),
        "source": row["source"],
        "source_sha256": row["source_sha256"],
    }


def write_tsv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_by_spec_and_family(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for d, ff in sorted({spec(r) for r in rows}):
        for fam in ["Baseline", "Set Dense"]:
            candidates = [r for r in rows if spec(r) == (d, ff) and label(r) == fam]
            out.append(min(candidates, key=lambda r: as_float(r, "val/ppl") or float("inf")))
    return out


def paired_best_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best = best_by_spec_and_family(rows)
    out = []
    by_key = {(spec(r), label(r)): r for r in best}
    for d, ff in sorted({spec(r) for r in rows}):
        b = by_key[((d, ff), "Baseline")]
        s = by_key[((d, ff), "Set Dense")]
        b_ppl = as_float(b, "val/ppl")
        s_ppl = as_float(s, "val/ppl")
        b_vram = as_float(b, "train/peak_vram_mib")
        s_vram = as_float(s, "train/peak_vram_mib")
        out.append(
            {
                "D": str(d),
                "d_ff": str(ff),
                "baseline_lr": b["training.lr"],
                "baseline_val_ppl": fmt(b_ppl, 1),
                "baseline_time_s": fmt(as_float(b, "train/time_per_epoch_s"), 1),
                "baseline_vram_mib": fmt(b_vram, 1),
                "set_lr": s["training.lr"],
                "set_val_ppl": fmt(s_ppl, 1),
                "set_time_s": fmt(as_float(s, "train/time_per_epoch_s"), 1),
                "set_vram_mib": fmt(s_vram, 1),
                "delta_ppl": fmt((s_ppl or 0.0) - (b_ppl or 0.0), 1),
                "vram_delta_pct": fmt(100.0 * ((s_vram or 0.0) - (b_vram or 0.0)) / (b_vram or 1.0), 1),
                "baseline_source": b["source"],
                "baseline_source_sha256": b["source_sha256"],
                "set_source": s["source"],
                "set_source_sha256": s["source_sha256"],
            }
        )
    return out


def latex_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{LR-normalized matched headline comparison at epoch 10. For each matched $(D,d_{\mathrm{ff}})$ pair, both baseline and set-dense models are selected by validation perplexity from the same learning-rate grid $\{10^{-4},2{\times}10^{-4},3{\times}10^{-4}\}$ under the shared WikiText-2 setup. Set rows use dense-exact set attention with learned routing, fixed $k=16$, soft-trimmed Boltzmann pooling with $\tau_{\mathrm{pool}}=0.1$, $w=16$, and $s=8$.}",
        r"\label{tab:matched-baseline-vs-setonly}",
        r"\small",
        r"\begin{tabular}{ccrrrrrrrr}",
        r"\toprule",
        r"$D$ & $d_{\mathrm{ff}}$ & Base LR & Base PPL & Base time & Base VRAM & Set LR & Set PPL & Set time & Set VRAM \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['D']} & {r['d_ff']} & {r['baseline_lr']} & {r['baseline_val_ppl']} & "
            f"{r['baseline_time_s']} & {r['baseline_vram_mib']} & {r['set_lr']} & "
            f"\\textbf{{{r['set_val_ppl']}}} & {r['set_time_s']} & {r['set_vram_mib']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def make_plot(rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is unavailable; skipping LR-normalized headline plot generation.")
        return

    PLOTS.mkdir(parents=True, exist_ok=True)
    FINAL_PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    markers = {"Baseline": "o", "Set Dense": "s"}
    colors = {"Baseline": "#4c566a", "Set Dense": "#0072b2"}
    for fam in ["Baseline", "Set Dense"]:
        fam_rows = [r for r in rows if label(r) == fam]
        ax.scatter(
            [as_float(r, "train/time_per_epoch_s") for r in fam_rows],
            [as_float(r, "val/ppl") for r in fam_rows],
            marker=markers[fam],
            color=colors[fam],
            label=fam,
            s=46,
            alpha=0.88,
        )
    for r in rows:
        d, ff = spec(r)
        ax.annotate(
            f"{d}/{ff}, {r['training.lr']}",
            (as_float(r, "train/time_per_epoch_s") or 0, as_float(r, "val/ppl") or 0),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            color=colors[label(r)],
        )
    ax.set_xlabel("Time per epoch (s)")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("LR-normalized matched headline grid")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    for path in [
        PLOTS / "fig_lrnorm_headline_time_vs_ppl.png",
        FINAL_PLOTS / "fig_lrnorm_headline_time_vs_ppl.png",
    ]:
        fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    for path in [TABLES, LATEX, PLOTS, CHECKS, FINAL_PLOTS]:
        path.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    compact_cols = [
        "family",
        "lr",
        "D",
        "d_ff",
        "val_ppl",
        "train_ppl",
        "time_per_epoch_s",
        "peak_vram_mib",
        "candidate_count",
        "routing_entropy_norm",
        "router_top1",
        "rho_pa",
        "source",
        "source_sha256",
    ]
    compact = [compact_row(r) for r in rows]
    best = paired_best_rows(rows)
    write_tsv(TABLES / "lrnorm_headline_all_runs.tsv", compact, compact_cols)
    write_tsv(TABLES / "lrnorm_headline_best_by_pair.tsv", best, list(best[0]))
    (LATEX / "table_lrnorm_headline_best_by_pair.tex").write_text(latex_table(best))
    make_plot(rows)
    generated = [
        "out/paper_integrated_evidence/tables/lrnorm_headline_all_runs.tsv",
        "out/paper_integrated_evidence/tables/lrnorm_headline_best_by_pair.tsv",
        "out/paper_integrated_evidence/latex/table_lrnorm_headline_best_by_pair.tex",
    ]
    for plot_path in [
        "out/paper_integrated_evidence/plots/fig_lrnorm_headline_time_vs_ppl.png",
        "out/final_paper_bundle/plots/main/fig_lrnorm_headline_time_vs_ppl.png",
    ]:
        if (ROOT / plot_path).exists():
            generated.append(plot_path)
    manifest = {
        "status": "completed_lr_normalized_headline_evidence",
        "selection_rule": "For each matched (D,d_ff) pair and family, select the row with lowest validation perplexity from the shared LR grid {1e-4,2e-4,3e-4}.",
        "source_count": len(rows),
        "sources": [{"path": r["source"], "sha256": r["source_sha256"]} for r in rows],
        "generated": [{"path": p, "sha256": sha256(ROOT / p)} for p in generated],
    }
    (CHECKS / "lrnorm_headline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote completed LR-normalized headline summary.")
    for p in generated + ["out/paper_integrated_evidence/checks/lrnorm_headline_manifest.json"]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
