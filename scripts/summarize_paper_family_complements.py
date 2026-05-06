#!/usr/bin/env python3
import csv
import json
import math
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path("/workspace")
OUT_ROOT = ROOT / "out"
BUNDLE = OUT_ROOT / "paper_complements_bundle"
TABLES = BUNDLE / "tables"
LATEX = BUNDLE / "latex"
CHECKS = BUNDLE / "checks"

for d in [TABLES, LATEX, CHECKS]:
    d.mkdir(parents=True, exist_ok=True)

# Search broadly because actual output location may vary across runner versions.
CSV_FILES = sorted(OUT_ROOT.rglob("paper_boundary_*.csv")) + sorted(OUT_ROOT.rglob("paper_pooltau_*.csv"))
JSON_FILES = {p.stem: p for p in OUT_ROOT.rglob("*.json")}

def read_csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def read_json(path):
    with open(path) as fh:
        return json.load(fh)

def get_cfg(obj, key):
    if key in obj:
        return obj[key]
    cfg = obj.get("config", {})
    if key in cfg:
        return cfg[key]
    parts = key.split(".")
    cur = cfg
    ok = True
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            ok = False
            break
    if ok:
        return cur
    return None

def s(v):
    return "NA" if v is None else str(v)

def f(v):
    if v in ("", None, "NA"):
        return None
    try:
        return float(v)
    except Exception:
        return None

def latest_completed_row(rows, expected_epoch=10):
    # Prefer exact epoch match; otherwise use last row.
    by_epoch = []
    for r in rows:
        ep = f(r.get("epoch"))
        if ep is not None:
            by_epoch.append((ep, r))
    if by_epoch:
        exact = [r for ep, r in by_epoch if int(ep) == int(expected_epoch)]
        if exact:
            return exact[-1]
        by_epoch.sort(key=lambda x: x[0])
        return by_epoch[-1][1]
    return rows[-1]

def find_json_for_csv(csv_path):
    return JSON_FILES.get(csv_path.stem, None)

def detect_sweep(stem):
    if stem.startswith("paper_boundary_"):
        return "boundary"
    if stem.startswith("paper_pooltau_"):
        return "pooltau"
    return None

def build_row(csv_path, json_path):
    rows = read_csv_rows(csv_path)
    if not rows:
        return None
    obj = read_json(json_path) if json_path and json_path.exists() else {}
    last = latest_completed_row(rows, expected_epoch=10)

    row = {
        "source_csv": str(csv_path.relative_to(ROOT)),
        "source_json": str(json_path.relative_to(ROOT)) if json_path and json_path.exists() else "NA",
        "mtime": csv_path.stat().st_mtime,
        "sweep": detect_sweep(csv_path.stem),
        "run_name_short": csv_path.stem,

        "task": s(get_cfg(obj, "task")),
        "data.dataset": s(get_cfg(obj, "data.dataset")),
        "data.seq_len": s(get_cfg(obj, "data.seq_len")),
        "data.batch_size": s(get_cfg(obj, "data.batch_size")),
        "training.seed": s(get_cfg(obj, "training.seed")),
        "training.lr": s(get_cfg(obj, "training.lr")),
        "training.epochs": s(get_cfg(obj, "training.epochs")),
        "training.warmup_steps": s(get_cfg(obj, "training.warmup_steps")),

        "model.implementation": s(get_cfg(obj, "model.implementation")),
        "model.attention_family": s(get_cfg(obj, "model.attention_family")),
        "model.backend": s(get_cfg(obj, "model.backend")),
        "model.feature_mode": s(get_cfg(obj, "model.feature_mode")),
        "model.router_type": s(get_cfg(obj, "model.router_type")),
        "model.router_topk": s(get_cfg(obj, "model.router_topk")),
        "model.router_multihead": s(get_cfg(obj, "model.router_multihead")),
        "model.pooling_multihead": s(get_cfg(obj, "model.pooling_multihead")),
        "model.window_size": s(get_cfg(obj, "model.window_size")),
        "model.stride": s(get_cfg(obj, "model.stride")),
        "model.router_temperature": s(get_cfg(obj, "model.router_temperature")),
        "model.pooling.tau": s(get_cfg(obj, "model.pooling.tau")),
        "model.pooling.q": s(get_cfg(obj, "model.pooling.q")),
        "model.d_model": s(get_cfg(obj, "model.d_model")),
        "model.num_layers": s(get_cfg(obj, "model.num_layers")),
        "model.num_heads": s(get_cfg(obj, "model.num_heads")),
        "model.dim_feedforward": s(get_cfg(obj, "model.dim_feedforward")),
        "model.backend_params.radius": s(get_cfg(obj, "model.backend_params.radius")),
        "model.backend_params.num_landmarks": s(get_cfg(obj, "model.backend_params.num_landmarks")),

        "epoch": last.get("epoch", "NA"),
        "val/loss": last.get("val/loss", "NA"),
        "val/ppl": last.get("val/ppl", "NA"),
        "train/loss": last.get("train/loss", "NA"),
        "train/ppl": last.get("train/ppl", "NA"),
        "train/time_per_epoch_s": last.get("train/time_per_epoch_s", "NA"),
        "train/peak_vram_mib": last.get("train/peak_vram_mib", "NA"),

        "ausa/router_candidate_count_struct_mean": last.get("ausa/router_candidate_count_struct_mean", "NA"),
        "ausa/routing_entropy_norm": last.get("ausa/routing_entropy_norm", "NA"),
        "ausa/router_top1_weight": last.get("ausa/router_top1_weight", "NA"),
        "ausa/pooling_neff_ratio": last.get("ausa/pooling_neff_ratio", "NA"),
        "ausa/pooling_neff_l2": last.get("ausa/pooling_neff_l2", "NA"),
        "ausa/pooling_effective_support": last.get("ausa/pooling_effective_support", "NA"),
        "ausa/grad_ratio_pool_rho_p": last.get("ausa/grad_ratio_pool_rho_p", "NA"),
        "ausa/grad_ratio_set_stack_rho_a": last.get("ausa/grad_ratio_set_stack_rho_a", "NA"),
        "ausa/grad_ratio_total_rho_pa": last.get("ausa/grad_ratio_total_rho_pa", "NA"),
    }
    return row

# Collect rows
raw_rows = []
for csv_path in CSV_FILES:
    json_path = find_json_for_csv(csv_path)
    row = build_row(csv_path, json_path)
    if row is not None and row["sweep"] in {"boundary", "pooltau"}:
        raw_rows.append(row)

# Dedup by full experimental control key
def dedup_key(r):
    return (
        r["sweep"],
        r["model.attention_family"],
        r["model.backend"],
        r["data.dataset"],
        r["data.seq_len"],
        r["data.batch_size"],
        r["training.seed"],
        r["training.lr"],
        r["training.epochs"],
        r["model.d_model"],
        r["model.num_layers"],
        r["model.num_heads"],
        r["model.dim_feedforward"],
        r["model.window_size"],
        r["model.stride"],
        r["model.router_temperature"],
        r["model.pooling.tau"],
        r["model.router_topk"],
        r["model.feature_mode"],
        r["model.router_type"],
        r["model.pooling_multihead"],
        r["model.backend_params.radius"],
        r["model.backend_params.num_landmarks"],
    )

grouped = defaultdict(list)
for r in raw_rows:
    grouped[dedup_key(r)].append(r)

selected = []
audit = {"boundary": [], "pooltau": []}

for key, rows in grouped.items():
    rows_sorted = sorted(rows, key=lambda x: x["mtime"], reverse=True)
    keep = rows_sorted[0]
    selected.append(keep)
    audit[keep["sweep"]].append({
        "dedup_key": list(key),
        "selected_source_csv": keep["source_csv"],
        "selected_source_json": keep["source_json"],
        "selected_mtime": keep["mtime"],
        "duplicate_candidates": [
            {
                "source_csv": r["source_csv"],
                "source_json": r["source_json"],
                "mtime": r["mtime"],
                "val/ppl": r["val/ppl"],
            }
            for r in rows_sorted
        ],
    })

# Split
boundary_rows = sorted(
    [r for r in selected if r["sweep"] == "boundary"],
    key=lambda r: (r["model.attention_family"], float(r["model.stride"]))
)
pooltau_rows = sorted(
    [r for r in selected if r["sweep"] == "pooltau"],
    key=lambda r: (r["model.attention_family"], float(r["model.pooling.tau"]))
)

FULL_FIELDS = [
    "source_csv","source_json","run_name_short","sweep",
    "task","data.dataset","data.seq_len","data.batch_size",
    "training.seed","training.lr","training.epochs","training.warmup_steps",
    "model.implementation","model.attention_family","model.backend",
    "model.feature_mode","model.router_type","model.router_topk",
    "model.router_multihead","model.pooling_multihead",
    "model.window_size","model.stride","model.router_temperature",
    "model.pooling.tau","model.pooling.q",
    "model.d_model","model.num_layers","model.num_heads","model.dim_feedforward",
    "model.backend_params.radius","model.backend_params.num_landmarks",
    "epoch","val/loss","val/ppl","train/loss","train/ppl",
    "train/time_per_epoch_s","train/peak_vram_mib",
    "ausa/router_candidate_count_struct_mean","ausa/routing_entropy_norm",
    "ausa/router_top1_weight","ausa/pooling_neff_ratio",
    "ausa/pooling_neff_l2","ausa/pooling_effective_support",
    "ausa/grad_ratio_pool_rho_p","ausa/grad_ratio_set_stack_rho_a","ausa/grad_ratio_total_rho_pa"
]

BOUNDARY_COMPACT_FIELDS = [
    "model.attention_family","model.backend","model.stride","model.window_size","model.router_temperature",
    "ausa/router_candidate_count_struct_mean","ausa/routing_entropy_norm","ausa/router_top1_weight",
    "ausa/pooling_neff_ratio","ausa/grad_ratio_pool_rho_p","ausa/grad_ratio_set_stack_rho_a",
    "ausa/grad_ratio_total_rho_pa","val/ppl","train/time_per_epoch_s","train/peak_vram_mib","source_csv"
]

POOLTAU_COMPACT_FIELDS = [
    "model.attention_family","model.backend","model.pooling.tau","model.window_size","model.stride","model.router_temperature",
    "ausa/pooling_neff_ratio","ausa/pooling_neff_l2","ausa/pooling_effective_support",
    "ausa/routing_entropy_norm","ausa/router_top1_weight",
    "ausa/grad_ratio_pool_rho_p","ausa/grad_ratio_set_stack_rho_a","ausa/grad_ratio_total_rho_pa",
    "val/ppl","train/time_per_epoch_s","train/peak_vram_mib","source_csv"
]

def write_tsv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "NA") for k in fields})

write_tsv(TABLES / "paper_boundary_family_complements.tsv", boundary_rows, FULL_FIELDS)
write_tsv(TABLES / "paper_pooltau_family_complements.tsv", pooltau_rows, FULL_FIELDS)
write_tsv(TABLES / "paper_boundary_family_complements_compact.tsv", boundary_rows, BOUNDARY_COMPACT_FIELDS)
write_tsv(TABLES / "paper_pooltau_family_complements_compact.tsv", pooltau_rows, POOLTAU_COMPACT_FIELDS)

with open(CHECKS / "paper_boundary_family_complements.audit.json", "w") as fh:
    json.dump(audit["boundary"], fh, indent=2)
with open(CHECKS / "paper_pooltau_family_complements.audit.json", "w") as fh:
    json.dump(audit["pooltau"], fh, indent=2)

def fmt_num(x, nd=3):
    if x in ("NA", "", None):
        return "NA"
    try:
        v = float(x)
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)

def fam_pretty(r):
    fam = r["model.attention_family"]
    back = r["model.backend"]
    if fam == "dense" and back == "exact":
        return "Dense"
    if fam == "sparse" and back == "local_band":
        return "Sparse"
    if fam == "linear" and back == "landmark":
        return "Linear"
    return f"{fam}/{back}"

def write_boundary_latex(rows, path):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Expanded boundary-stride family complement sweep at fixed $w=16$, $\tau_r=1.0$, $\tau_{\mathrm{pool}}=0.1$, $q=0.85$, and the same post-parity set-only architecture/training recipe used in the Tier C anchor family. Rows are deduplicated by full control-key; if multiple source files share the same key, the newest completed run is retained and all candidates are recorded in the audit JSON.}")
    lines.append(r"\label{tab:boundary-family-complements}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Family & $s$ & Cand. & Entropy & Top-1 & $n_{\mathrm{eff}}$ ratio & $\rho_p$ & $\rho_a$ & $\rho_{pa}$ & Val. PPL $\downarrow$ \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{fam_pretty(r)} & "
            f"{fmt_num(r['model.stride'],0)} & "
            f"{fmt_num(r['ausa/router_candidate_count_struct_mean'],3)} & "
            f"{fmt_num(r['ausa/routing_entropy_norm'],3)} & "
            f"{fmt_num(r['ausa/router_top1_weight'],3)} & "
            f"{fmt_num(r['ausa/pooling_neff_ratio'],3)} & "
            f"{fmt_num(r['ausa/grad_ratio_pool_rho_p'],3)} & "
            f"{fmt_num(r['ausa/grad_ratio_set_stack_rho_a'],3)} & "
            f"{fmt_num(r['ausa/grad_ratio_total_rho_pa'],3)} & "
            f"{fmt_num(r['val/ppl'],1)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines))

def write_pooltau_latex(rows, path):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Expanded pooling-temperature family complement sweep at fixed $s=4$, $w=16$, $\tau_r=1.0$, $q=0.85$, and the same post-parity set-only architecture/training recipe used in the Tier C anchor family. Rows are deduplicated by full control-key; if multiple source files share the same key, the newest completed run is retained and all candidates are recorded in the audit JSON.}")
    lines.append(r"\label{tab:pooltau-family-complements}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Family & $\tau_{\mathrm{pool}}$ & $n_{\mathrm{eff}}$ ratio & $n_{\mathrm{eff}}$ L2 & Eff. support & Entropy & Top-1 & $\rho_{pa}$ & Val. PPL $\downarrow$ & Time / epoch (s) $\downarrow$ \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{fam_pretty(r)} & "
            f"{fmt_num(r['model.pooling.tau'],3)} & "
            f"{fmt_num(r['ausa/pooling_neff_ratio'],3)} & "
            f"{fmt_num(r['ausa/pooling_neff_l2'],3)} & "
            f"{fmt_num(r['ausa/pooling_effective_support'],3)} & "
            f"{fmt_num(r['ausa/routing_entropy_norm'],3)} & "
            f"{fmt_num(r['ausa/router_top1_weight'],3)} & "
            f"{fmt_num(r['ausa/grad_ratio_total_rho_pa'],3)} & "
            f"{fmt_num(r['val/ppl'],1)} & "
            f"{fmt_num(r['train/time_per_epoch_s'],1)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines))

write_boundary_latex(boundary_rows, LATEX / "table_boundary_family_complements.tex")
write_pooltau_latex(pooltau_rows, LATEX / "table_pooltau_family_complements.tex")

print("Wrote:")
print(" -", TABLES / "paper_boundary_family_complements.tsv")
print(" -", TABLES / "paper_pooltau_family_complements.tsv")
print(" -", TABLES / "paper_boundary_family_complements_compact.tsv")
print(" -", TABLES / "paper_pooltau_family_complements_compact.tsv")
print(" -", LATEX / "table_boundary_family_complements.tex")
print(" -", LATEX / "table_pooltau_family_complements.tex")
print(" -", CHECKS / "paper_boundary_family_complements.audit.json")
print(" -", CHECKS / "paper_pooltau_family_complements.audit.json")
