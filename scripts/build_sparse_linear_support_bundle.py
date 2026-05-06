#!/usr/bin/env python3
import csv
import json
from pathlib import Path

ROOT = Path.home() / "set-attention"
METRICS = ROOT / "out" / "metrics"
OUTDIR = ROOT / "out" / "paper_support_sparse_linear_diag"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Rows to include.
# You can edit this list if you want fewer/more support rows.
RUN_BASENAMES = [
    # strict block
    "action1A_LR1e-4_baseline_seed0",
    "action1A_LR1e-4_set_dense_M64_seed0",
    "action1A_LR1e-4_set_linear_M64_seed0",
    "action1A_LR1e-4_set_sparse_M64_seed0",

    # same-regime support rows
    "action0A_linear_M64_seed0",
    "action0A_sparse_M64_seed0",
    "action0_g1_linear_M64_seed0",
    "action0_g1_sparse_M64_seed0",

    # optional: keep these if you want pooling-single support too
    # "action0P4_linear_pool_single_seed0",
    # "action0P4_sparse_pool_single_seed0",
]

CORE_CONFIG_FIELDS = [
    "task",
    "data.dataset",
    "data.seq_len",
    "data.batch_size",
    "training.seed",
    "training.lr",
    "training.epochs",
    "model.implementation",
    "model.attention_family",
    "model.backend",
    "model.feature_mode",
    "model.router_type",
    "model.router_topk",
    "model.pooling.mode",
    "model.pooling.tau",
    "model.router_multihead",
    "model.pooling_multihead",
    "model.d_model",
    "model.num_layers",
    "model.num_heads",
    "model.dim_feedforward",
    "model.window_size",
    "model.stride",
]

# Diagnostics of interest
DIAG_FIELDS = [
    "ausa/routing_entropy_norm",
    "ausa/router_top1_weight",
    "ausa/pooling_neff_ratio",
    "ausa/grad_ratio_pool_rho_p",
    "ausa/grad_ratio_set_stack_rho_a",
    "ausa/grad_ratio_total_rho_pa",
]

METRIC_FIELDS = [
    "epoch",
    "val/loss",
    "val/ppl",
    "train/loss",
    "train/ppl",
    "train/time_per_epoch_s",
    "train/peak_vram_mib",
]

OUTPUT_FIELDS = [
    "run_name_short",
    "source_csv",
    "source_json",
    "group_tag",
    *CORE_CONFIG_FIELDS,
    *DIAG_FIELDS,
    *METRIC_FIELDS,
    "diagnostics_present",
    "evidence_note",
]

def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)

def load_csv_rows(path: Path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def get_cfg(obj: dict, key: str):
    if key in obj:
        return obj[key]
    cfg = obj.get("config", {})
    return cfg.get(key)

def s(v):
    return "" if v is None else str(v)

rows_out = []
manifest = {
    "purpose": "sparse/linear support bundle with diagnostics",
    "diagnostics_requested": DIAG_FIELDS,
    "runs": [],
}

for base in RUN_BASENAMES:
    csv_path = METRICS / f"{base}.csv"
    json_path = METRICS / f"{base}.json"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    obj = load_json(json_path)
    rows = load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    last = rows[-1]
    header = set(rows[0].keys())

    # normalize pooling_multihead note
    pool_mh = get_cfg(obj, "model.pooling_multihead")
    if pool_mh is None and (base.startswith("action0A_") or base.startswith("action0_g1_")):
        pool_mh_effective = "effective_single_path(old_missing_key)"
    else:
        pool_mh_effective = s(pool_mh)

    if base.startswith("action1A_"):
        group_tag = "strict_action1A"
        note = "strict block row"
    elif base.startswith("action0P4_"):
        group_tag = "support_pool_single"
        note = "same-regime support row; pooling_single variant; LR differs from action1A"
    else:
        group_tag = "support_same_regime"
        note = "same-regime support row; main explicit difference vs action1A is LR"

    out = {
        "run_name_short": base,
        "source_csv": str(csv_path.relative_to(ROOT)),
        "source_json": str(json_path.relative_to(ROOT)),
        "group_tag": group_tag,
        "evidence_note": note,
    }

    for k in CORE_CONFIG_FIELDS:
        if k == "model.pooling_multihead":
            out[k] = pool_mh_effective
        else:
            out[k] = s(get_cfg(obj, k))

    for k in DIAG_FIELDS:
        out[k] = last.get(k, "")

    for k in METRIC_FIELDS:
        out[k] = last.get(k, "")

    present = [k for k in DIAG_FIELDS if k in header]
    out["diagnostics_present"] = ",".join(present)

    rows_out.append(out)

    manifest["runs"].append({
        "run_name_short": base,
        "group_tag": group_tag,
        "source_csv": str(csv_path.relative_to(ROOT)),
        "source_json": str(json_path.relative_to(ROOT)),
        "diagnostics_present": present,
        "final_val_ppl": last.get("val/ppl", ""),
        "final_val_loss": last.get("val/loss", ""),
    })

# main audit TSV
audit_tsv = OUTDIR / "paper_sparse_linear_with_diagnostics.tsv"
with open(audit_tsv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, delimiter="\t")
    w.writeheader()
    w.writerows(rows_out)

# compact paper-facing TSV
paper_fields = [
    "run_name_short",
    "group_tag",
    "training.lr",
    "model.attention_family",
    "model.backend",
    "val/ppl",
    "train/time_per_epoch_s",
    "train/peak_vram_mib",
    *DIAG_FIELDS,
]
paper_tsv = OUTDIR / "paper_sparse_linear_with_diagnostics_compact.tsv"
with open(paper_tsv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=paper_fields, delimiter="\t")
    w.writeheader()
    for r in rows_out:
        w.writerow({k: r.get(k, "") for k in paper_fields})

# manifest json
manifest_json = OUTDIR / "paper_sparse_linear_with_diagnostics_manifest.json"
with open(manifest_json, "w") as fh:
    json.dump(manifest, fh, indent=2)

# README
(OUTDIR / "README.md").write_text(
    """# Sparse/Linear Support Bundle with Diagnostics

This bundle extracts the final-row diagnostics and key metrics for:
- strict action1A rows
- same-regime support sparse/linear rows
- optional pooling-single support rows if enabled

Diagnostics requested:
- ausa/routing_entropy_norm
- ausa/router_top1_weight
- ausa/pooling_neff_ratio
- ausa/grad_ratio_pool_rho_p
- ausa/grad_ratio_set_stack_rho_a
- ausa/grad_ratio_total_rho_pa

Files:
- paper_sparse_linear_with_diagnostics.tsv
- paper_sparse_linear_with_diagnostics_compact.tsv
- paper_sparse_linear_with_diagnostics_manifest.json
"""
)

print("Wrote:")
print(" -", audit_tsv)
print(" -", paper_tsv)
print(" -", manifest_json)
print(" -", OUTDIR / "README.md")
