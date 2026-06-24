#!/usr/bin/env python3
"""Homogenize SD-9.x set + token-baseline runs into ONE mergeable, host-agnostic schema.

Robustness (2026): keys are derived from JSON METADATA, not file/dir names, so the same
(model_kind, variant, seq_len, seed) aggregates correctly no matter which host/dir produced it
(safe cross-host merge + safe server swapping). Guards:
  - REJECT any data.limit row (probe/smoke).
  - REJECT any malformed seed (not in {0,1,2,3,4}, e.g. the space-seeded "0 1 2" bug).
  - DEDUP by (model_kind, variant, seq_len, seed): if duplicated, keep the most-trained (max epochs)
    and print a warning — never silently double-count.
Variant label: set -> "b{blur%}" (blur% = coarse_heads/num_heads), token -> "token".

Usage: python3 scripts/normalize_sd9x_runs.py [run_root ...]   (default out/paper_mechanisms)
Writes: out/paper_integrated_evidence/tables/sd9x_homogenized_runs.tsv
"""
from __future__ import annotations
import csv, glob, json, os, sys

COMMON = ["model_kind", "variant", "blur_pct", "seq_len", "seed", "backend", "landmark_coverage",
          "fine_heads", "coarse_heads", "epochs_done", "data_limit", "final_val_ppl",
          "final_train_ppl", "peak_vram_mib", "span_ablation_delta_ppl", "csv_path"]
VALID_SEEDS = {"0", "1", "2", "3", "4"}


def heads(meta):
    g = meta.get("model.multiresolution.groups")
    fh = ch = 0
    if isinstance(g, list):
        for x in g:
            if isinstance(x, dict) and x.get("name") == "fine":
                fh += int(x.get("num_heads", 0))
            elif isinstance(x, dict) and x.get("name") == "coarse":
                ch += int(x.get("num_heads", 0))
    return fh, ch


def normalize(json_path):
    try:
        meta = json.load(open(json_path))
    except Exception:
        return None
    if str(meta.get("data.limit", meta.get("data_limit", "NA"))) in ("500", "8"):
        return None
    seed = str(meta.get("training.seed", "?")).strip()
    if seed not in VALID_SEEDS:        # drops the space-seeded malformed runs
        return None
    fh, ch = heads(meta)
    nh = int(meta.get("model.num_heads", 8) or 8)
    is_multires = str(meta.get("model.multiresolution.enabled", "")).lower() == "true"
    is_set_only = str(meta.get("model.implementation", "")).startswith("set")
    if is_set_only and not is_multires:
        return None  # legacy single-resolution set run (a3/a8) — not part of SD-9.x multi-res analysis
    is_set = is_multires
    csv_path = json_path[:-5] + ".csv"
    n = 0; ppl = tppl = vram = sab = "NA"
    if os.path.exists(csv_path):
        rows = list(csv.DictReader(open(csv_path)))
        n = len(rows)
        if rows:
            ppl = rows[-1].get("val/ppl", "NA"); tppl = rows[-1].get("train/ppl", "NA")
            vram = rows[-1].get("train/peak_vram_mib", "NA")
            sab = rows[-1].get("val/span_ablation_delta_ppl", "NA")
    if is_set:
        kind = "set"; blur = round(100 * ch / nh) if nh else "NA"; variant = f"b{blur}"
    else:
        kind = "token"; blur = "NA"; variant = "token"; fh = ch = "NA"; sab = "NA"
    return {"model_kind": kind, "variant": variant, "blur_pct": blur,
            "seq_len": meta.get("data.seq_len", "NA"), "seed": seed,
            "backend": meta.get("model.backend", "NA"),
            "landmark_coverage": meta.get("model.backend_params.landmark_coverage", "NA"),
            "fine_heads": fh, "coarse_heads": ch, "epochs_done": n,
            "data_limit": "NONE", "final_val_ppl": ppl, "final_train_ppl": tppl,
            "peak_vram_mib": vram, "span_ablation_delta_ppl": sab, "csv_path": csv_path}


def main():
    roots = sys.argv[1:] or ["out/paper_mechanisms"]
    best = {}
    for root in roots:
        for jp in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            r = normalize(jp)
            if not r:
                continue
            key = (r["model_kind"], r["variant"], str(r["seq_len"]), r["seed"])
            if key in best:
                prev = best[key]
                if int(r["epochs_done"]) > int(prev["epochs_done"]):
                    print(f"WARN dup {key}: keeping {r['epochs_done']}ep over {prev['epochs_done']}ep")
                    best[key] = r
                else:
                    print(f"WARN dup {key}: keeping {prev['epochs_done']}ep, dropping {r['epochs_done']}ep")
            else:
                best[key] = r
    rows = sorted(best.values(), key=lambda r: (r["model_kind"], str(r["seq_len"]), r["variant"], r["seed"]))
    out = "out/paper_integrated_evidence/tables/sd9x_homogenized_runs.tsv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COMMON, delimiter="\t"); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "NA") for k in COMMON})
    ns = sum(1 for r in rows if r["model_kind"] == "set"); nt = len(rows) - ns
    print(f"wrote {len(rows)} valid full-data rows ({ns} set, {nt} token) -> {out}")


if __name__ == "__main__":
    main()
