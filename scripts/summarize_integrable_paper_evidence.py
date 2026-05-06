#!/usr/bin/env python3
"""Summarize paper evidence that is ready to integrate.

This script deliberately consumes only local bundle artifacts. Use
scripts/fetch_blue_demon_lrnorm_partial.sh first if the partial LR-normalized
CSV files are still only present on blue-demon.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
LATEX = OUT / "latex"
CHECKS = OUT / "checks"

LRNORM_DIRS = [
    ROOT / "out" / "paper_lr_norm" / "paper_lr_norm_headline_D384_FF1536",
    ROOT / "out" / "paper_lr_norm" / "paper_lr_norm_family_D384_FF1536",
]

COMPLEMENT_TABLES = [
    ROOT / "out" / "final_paper_bundle" / "tables" / "summary" / "paper_boundary_family_complements_compact.tsv",
    ROOT / "out" / "final_paper_bundle" / "tables" / "summary" / "paper_pooltau_family_complements_compact.tsv",
]


def read_last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    row = dict(rows[-1])
    row["_source"] = str(path.relative_to(ROOT))
    row["_source_sha256"] = sha256(path)
    return row


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def f(row: dict[str, str], key: str) -> float | None:
    val = row.get(key, "NA")
    if val in {"", "NA", "nan", "None"}:
        return None
    return float(val)


def fmt_num(val: float | None, digits: int = 1) -> str:
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def impl_label(row: dict[str, str]) -> str:
    impl = row.get("model.implementation", "")
    family = row.get("model.attention_family", "")
    backend = row.get("model.backend", "")
    if impl == "baseline_token":
        return "Baseline token"
    if family == "dense" and backend == "exact":
        return "Set Dense"
    if family == "sparse" and backend == "local_band":
        return "Set Sparse"
    if family == "linear" and backend == "landmark":
        return "Set Linear"
    return f"{impl}/{family}/{backend}"


def lr_sort_key(row: dict[str, str]) -> tuple[int, str]:
    lr = row.get("training.lr", "")
    order = {"1e-4": 1, "2e-4": 2, "3e-4": 3}
    return (order.get(lr, 99), impl_label(row))


def collect_lrnorm_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for directory in LRNORM_DIRS:
        for path in sorted(directory.glob("*.csv")):
            rows.append(read_last_csv_row(path))
    rows.sort(key=lambda r: (impl_label(r), lr_sort_key(r)[0]))
    return rows


def write_tsv(path: Path, rows: Iterable[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def escape_tex(text: str) -> str:
    return text.replace("_", r"\_")


def latex_lrnorm_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Partial LR-normalized D384/F1536 slice available while the full headline rerun is still pending. All rows use WikiText-2, $L=512$, $B=16$, seed $0$, 10 epochs, warmup $1000$, $D=384$, $d_{\mathrm{ff}}=1536$, six layers, eight heads, dropout probabilities $0.1$, and vocabulary size $76618$. Set rows share hashed-count features, learned routing with fixed $k=16$, soft-trimmed Boltzmann pooling with $\tau_{\mathrm{pool}}=0.1$, $w=16$, $s=8$, multi-head routing, and non-head-aware pooling.}",
        r"\label{tab:lrnorm-d384-family-slice}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Family & LR & Val. PPL $\downarrow$ & Train PPL & Time (s) & VRAM & Ent. & $\rho_{pa}$ \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=lambda r: (lr_sort_key(r)[0], impl_label(r))):
        lines.append(
            " & ".join(
                [
                    escape_tex(impl_label(row)),
                    row.get("training.lr", "NA"),
                    fmt_num(f(row, "val/ppl"), 1),
                    fmt_num(f(row, "train/ppl"), 1),
                    fmt_num(f(row, "train/time_per_epoch_s"), 1),
                    fmt_num(f(row, "train/peak_vram_mib"), 1),
                    fmt_num(f(row, "ausa/routing_entropy_norm"), 3),
                    fmt_num(f(row, "ausa/grad_ratio_total_rho_pa"), 3),
                ]
            )
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def compact_lrnorm_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "family": impl_label(row),
                "lr": row.get("training.lr", "NA"),
                "D": row.get("model.d_model", "NA"),
                "d_ff": row.get("model.dim_feedforward", "NA"),
                "val_ppl": fmt_num(f(row, "val/ppl"), 6),
                "train_ppl": fmt_num(f(row, "train/ppl"), 6),
                "time_per_epoch_s": fmt_num(f(row, "train/time_per_epoch_s"), 6),
                "peak_vram_mib": fmt_num(f(row, "train/peak_vram_mib"), 6),
                "candidate_count": fmt_num(f(row, "ausa/router_candidate_count_struct_mean"), 6),
                "routing_entropy_norm": fmt_num(f(row, "ausa/routing_entropy_norm"), 6),
                "router_top1": fmt_num(f(row, "ausa/router_top1_weight"), 6),
                "rho_pa": fmt_num(f(row, "ausa/grad_ratio_total_rho_pa"), 6),
                "source": row["_source"],
                "source_sha256": row["_source_sha256"],
            }
        )
    return compact


def best_by_family(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for family in sorted({impl_label(r) for r in rows}):
        candidates = [r for r in rows if impl_label(r) == family and f(r, "val/ppl") is not None]
        best = min(candidates, key=lambda r: f(r, "val/ppl"))
        out.append(best)
    return out


def main() -> None:
    for path in [OUT, TABLES, LATEX, CHECKS]:
        path.mkdir(parents=True, exist_ok=True)

    rows = collect_lrnorm_rows()
    if not rows:
        raise SystemExit("No LR-normalized CSVs found under out/paper_lr_norm.")

    compact = compact_lrnorm_rows(rows)
    columns = [
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
    write_tsv(TABLES / "lrnorm_d384_family_slice.tsv", compact, columns)

    best_compact = compact_lrnorm_rows(best_by_family(rows))
    write_tsv(TABLES / "lrnorm_d384_best_by_family.tsv", best_compact, columns)

    (LATEX / "table_lrnorm_d384_family_slice.tex").write_text(latex_lrnorm_table(rows))

    manifest = {
        "status": "partial_lr_normalized_evidence",
        "decision_policy": "Do not replace the headline baseline comparison until the full GPU0 headline rerun completes.",
        "lrnorm_csv_count": len(rows),
        "lrnorm_sources": [
            {"path": r["_source"], "sha256": r["_source_sha256"]} for r in rows
        ],
        "cross_family_complement_sources": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
            for p in COMPLEMENT_TABLES
            if p.exists()
        ],
        "generated": [
            "out/paper_integrated_evidence/tables/lrnorm_d384_family_slice.tsv",
            "out/paper_integrated_evidence/tables/lrnorm_d384_best_by_family.tsv",
            "out/paper_integrated_evidence/latex/table_lrnorm_d384_family_slice.tex",
        ],
    }
    manifest["generated_with_sha256"] = [
        {"path": path, "sha256": sha256(ROOT / path)}
        for path in manifest["generated"]
    ]
    (CHECKS / "integrable_evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Wrote:")
    for path in manifest["generated"] + ["out/paper_integrated_evidence/checks/integrable_evidence_manifest.json"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
