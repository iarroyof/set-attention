#!/usr/bin/env python3
"""
Generate Stage A tables and Stage B plots from bench_summary.csv and metrics_summary.csv.
Non-fatal if matplotlib is unavailable; will still emit CSV/LaTeX fragments.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def _add_impl_label(df: pd.DataFrame) -> pd.DataFrame:
    if "model_type" in df.columns:
        def _label(row: pd.Series) -> str:
            model_type = str(row.get("model_type", "NA"))
            if model_type == "baseline":
                return f"baseline/{row.get('baseline_impl', 'NA')}"
            if model_type == "ska":
                return f"ska/{row.get('ska_backend', 'NA')}/{row.get('ska_score_mode', 'NA')}"
            return model_type

        labeled = df.copy()
        labeled["impl_label"] = labeled.apply(_label, axis=1)
        return labeled
    if "attn_impl" in df.columns:
        labeled = df.copy()
        labeled["impl_label"] = labeled["attn_impl"].astype(str)
        return labeled
    labeled = df.copy()
    labeled["impl_label"] = "unknown"
    return labeled


def stage_a_tables(metrics_df: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    # Keep only quality metrics columns
    quality_cols = [c for c in metrics_df.columns if any(k in c for k in ("loss", "ppl", "bleu", "rouge", "acc", "top5"))]
    key_cols = [
        "task",
        "dataset_id",
        "model_type",
        "baseline_impl",
        "ska_backend",
        "ska_score_mode",
        "precision",
    ]
    key_cols = [c for c in key_cols if c in metrics_df.columns]
    keep = [c for c in metrics_df.columns if c in key_cols or c in quality_cols]
    df = metrics_df[keep].copy()
    # Group by core keys and take mean/std if _mean/_std exist
    agg_cols = {}
    for c in quality_cols:
        agg_cols[c] = "mean"
    grouped = df.groupby(key_cols, dropna=False).agg(agg_cols).reset_index()
    csv_path = out_dir / "stageA_quality.csv"
    grouped.to_csv(csv_path, index=False)
    outputs["stageA_csv"] = csv_path
    # Simple LaTeX table
    latex_path = out_dir / "stageA_quality.tex"
    grouped.to_latex(latex_path, index=False, float_format="%.3f")
    outputs["stageA_tex"] = latex_path
    return outputs


def stage_b_plots(bench_df: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    bench_df = _add_impl_label(bench_df)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        # Emit a CSV summary even if plotting libs missing
        summary_path = out_dir / "stageB_throughput.csv"
        cols = [
            "task",
            "dataset_id",
            "model_type",
            "baseline_impl",
            "ska_backend",
            "ska_score_mode",
            "impl_label",
            "precision",
            "seq_len",
            "tokens_per_s",
            "scores_per_s",
            "max_vram_mb",
        ]
        keep = [c for c in bench_df.columns if c in cols]
        bench_df[keep].to_csv(summary_path, index=False)
        outputs["stageB_csv"] = summary_path
        return outputs

    # Plot throughput vs seq_len for LM tasks
    lm = bench_df[bench_df["task"] == "lm"].copy()
    if "seq_len" in lm:
        fig, ax = plt.subplots()
        for impl, sub in lm.groupby("impl_label"):
            ax.plot(sub["seq_len"], sub["tokens_per_s"], marker="o", label=impl)
        ax.set_xlabel("seq_len")
        ax.set_ylabel("tokens/s")
        ax.legend()
        ax.set_title("LM throughput vs length")
        lm_path = out_dir / "stageB_lm_throughput.png"
        fig.savefig(lm_path, dpi=200, bbox_inches="tight")
        outputs["stageB_lm_plot"] = lm_path
        plt.close(fig)
        # VRAM vs seq_len
        fig, ax = plt.subplots()
        for impl, sub in lm.groupby("impl_label"):
            ax.plot(sub["seq_len"], sub["max_vram_mb"], marker="o", label=impl)
        ax.set_xlabel("seq_len")
        ax.set_ylabel("max_vram_mb")
        ax.legend()
        ax.set_title("LM VRAM vs length")
        lm_vram = out_dir / "stageB_lm_vram.png"
        fig.savefig(lm_vram, dpi=200, bbox_inches="tight")
        outputs["stageB_lm_vram"] = lm_vram
        plt.close(fig)

    # Seq2Seq throughput/VRAM
    seq = bench_df[bench_df["task"] == "seq2seq"].copy()
    if not seq.empty:
        if "max_len" in seq:
            fig, ax = plt.subplots()
            for impl, sub in seq.groupby("impl_label"):
                ax.plot(sub["max_len"], sub["tokens_per_s"], marker="o", label=impl)
            ax.set_xlabel("seq_len")
            ax.set_ylabel("tokens/s")
            ax.legend()
            ax.set_title("Seq2Seq throughput vs length")
            seq_path = out_dir / "stageB_seq_throughput.png"
            fig.savefig(seq_path, dpi=200, bbox_inches="tight")
            outputs["stageB_seq_plot"] = seq_path
            plt.close(fig)

            fig, ax = plt.subplots()
            for impl, sub in seq.groupby("impl_label"):
                ax.plot(sub["max_len"], sub["max_vram_mb"], marker="o", label=impl)
            ax.set_xlabel("seq_len")
            ax.set_ylabel("max_vram_mb")
            ax.legend()
            ax.set_title("Seq2Seq VRAM vs length")
            seq_vram = out_dir / "stageB_seq_vram.png"
            fig.savefig(seq_vram, dpi=200, bbox_inches="tight")
            outputs["stageB_seq_vram"] = seq_vram
            plt.close(fig)

    # Text diffusion (text sequence length)
    textdiff = bench_df[bench_df["task"] == "textdiff"].copy()
    if not textdiff.empty and "text_seq_len" in textdiff:
        fig, ax = plt.subplots()
        for impl, sub in textdiff.groupby("impl_label"):
            ax.plot(sub["text_seq_len"], sub["sequences_per_s"], marker="o", label=impl)
        ax.set_xlabel("text_seq_len")
        ax.set_ylabel("seq/s")
        ax.legend()
        ax.set_title("TextDiff throughput vs length")
        td_path = out_dir / "stageB_textdiff_throughput.png"
        fig.savefig(td_path, dpi=200, bbox_inches="tight")
        outputs["stageB_textdiff_plot"] = td_path
        plt.close(fig)

        fig, ax = plt.subplots()
        for impl, sub in textdiff.groupby("impl_label"):
            ax.plot(sub["text_seq_len"], sub["max_vram_mb"], marker="o", label=impl)
        ax.set_xlabel("text_seq_len")
        ax.set_ylabel("max_vram_mb")
        ax.legend()
        ax.set_title("TextDiff VRAM vs length")
        td_vram = out_dir / "stageB_textdiff_vram.png"
        fig.savefig(td_vram, dpi=200, bbox_inches="tight")
        outputs["stageB_textdiff_vram"] = td_vram
        plt.close(fig)

    # ViT (no length axis; show throughput/VRAM bars)
    vit = bench_df[bench_df["task"] == "vit"].copy()
    if not vit.empty:
        fig, ax = plt.subplots()
        impls = list(vit["impl_label"].unique())
        xs = range(len(impls))
        vals = [vit[vit["impl_label"] == impl]["images_per_s"].mean() for impl in impls]
        ax.bar(xs, vals)
        ax.set_xticks(xs)
        ax.set_xticklabels(impls, rotation=45)
        ax.set_ylabel("images/s")
        ax.set_title("ViT throughput by variant")
        vit_path = out_dir / "stageB_vit_throughput.png"
        fig.savefig(vit_path, dpi=200, bbox_inches="tight")
        outputs["stageB_vit_plot"] = vit_path
        plt.close(fig)

        fig, ax = plt.subplots()
        vals = [vit[vit["impl_label"] == impl]["max_vram_mb"].mean() for impl in impls]
        ax.bar(xs, vals)
        ax.set_xticks(xs)
        ax.set_xticklabels(impls, rotation=45)
        ax.set_ylabel("max_vram_mb")
        ax.set_title("ViT VRAM by variant")
        vit_vram = out_dir / "stageB_vit_vram.png"
        fig.savefig(vit_vram, dpi=200, bbox_inches="tight")
        outputs["stageB_vit_vram"] = vit_vram
        plt.close(fig)

    return outputs


def main():
    ap = argparse.ArgumentParser(description="Generate Stage A tables and Stage B plots from summaries.")
    ap.add_argument("--bench-summary", type=str, required=True, help="Path to bench_summary.csv")
    ap.add_argument("--metrics-summary", type=str, required=False, help="Path to metrics_summary.csv (optional)")
    ap.add_argument("--output-dir", type=str, default="out/stage_artifacts")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    bench_df = load_csv(Path(args.bench_summary))

    artifacts: Dict[str, Path] = {}
    if args.metrics_summary:
        try:
            metrics_df = load_csv(Path(args.metrics_summary))
            artifacts.update(stage_a_tables(metrics_df, out_dir))
        except FileNotFoundError:
            print(f"[stage-artifacts] metrics summary missing: {args.metrics_summary} (skipping Stage A tables)")
    else:
        print("[stage-artifacts] metrics summary not provided; skipping Stage A tables.")
    artifacts.update(stage_b_plots(bench_df, out_dir))

    manifest = {k: str(v) for k, v in artifacts.items()}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("[stage-artifacts] wrote:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
