#!/usr/bin/env python3
"""Homogenize SD-9.x set + token-baseline runs into ONE mergeable, host-agnostic schema.

Robustness (2026): keys are derived from JSON METADATA, not file/dir names, so the same
(model_kind, backend, variant, seq_len, batch, applied_seed) aggregates correctly no matter which host/dir produced it
(safe cross-host merge + safe server swapping). Guards:
  - REJECT every non-null data.limit/data.val_limit row (probe/smoke).
  - REJECT any malformed seed (not in {0,1,2,3,4}, e.g. the space-seeded "0 1 2" bug).
  - REQUIRE the applied-seed and current-diagnostics contracts.
  - FAIL on duplicate (family,backend,L,batch,variant,seed) cells.
Variant label: set -> "b{blur%}" (blur% = coarse_heads/num_heads), token -> "token".

Usage: python3 scripts/normalize_sd9x_runs.py [run_root ...]
Default root: out/paper_mechanisms/sd_grid_seeded_v1
Writes: out/paper_integrated_evidence/tables/sd9x_homogenized_runs.tsv
"""
from __future__ import annotations
import csv, glob, json, os, sys

COMMON = ["model_kind", "variant", "blur_pct", "seq_len", "batch_size", "seed",
          "applied_seed", "backend", "experiment_contract", "diagnostics_contract", "landmark_coverage",
          "fine_heads", "coarse_heads", "epochs_done", "data_limit", "final_val_ppl",
          "final_train_ppl", "peak_vram_mib", "span_ablation_delta_ppl",
          "span_ablation_fine_delta_ppl", "span_ablation_coarse_delta_ppl",
          "effective_range_fine", "effective_range_coarse",
          "routing_entropy_fine", "routing_entropy_coarse",
          "routing_top1_fine", "routing_top1_coarse",
          "router_param_norm_fine", "router_param_norm_coarse",
          "router_gradient_norm_fine", "router_gradient_norm_coarse",
          "pooling_effective_support_fine", "pooling_effective_support_coarse",
          "grad_norm_set_post_pool_fine", "grad_norm_set_post_pool_coarse",
          "grad_norm_set_post_blocks_fine", "grad_norm_set_post_blocks_coarse",
          "loss_early_freq", "loss_early_rare", "loss_late_freq", "loss_late_rare",
          "baseline_attention_entropy", "baseline_attention_top1",
          "baseline_attention_gradient_norm", "baseline_attention_param_norm",
          "csv_path"]
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
    def nullish(value):
        return value is None or str(value).strip().lower() in {"", "na", "none", "null"}
    if not nullish(meta.get("data.limit", meta.get("data_limit"))):
        return None
    if not nullish(meta.get("data.val_limit")):
        return None
    seed = str(meta.get("training.seed", "?")).strip()
    if seed not in VALID_SEEDS:        # drops the space-seeded malformed runs
        return None
    fh, ch = heads(meta)
    nh = int(meta.get("model.num_heads", 8) or 8)
    implementation = str(meta.get("model.implementation", ""))
    is_multires = (
        implementation == "set_only"
        and str(meta.get("model.multiresolution.enabled", "")).lower() == "true"
    )
    if implementation not in {"set_only", "baseline_token"}:
        return None
    if implementation == "set_only" and not is_multires:
        return None
    is_set = is_multires
    if meta.get("training.experiment_contract") != "sd_grid_seeded_v1":
        return None
    if meta.get("training.diagnostics_contract") != "current_matrix_v1":
        return None
    if meta.get("training.seed_applied") is not True:
        return None
    if str(meta.get("training.applied_seed", "?")).strip() != seed:
        return None
    if str(meta.get("training.torch_initial_seed", "?")).strip() != seed:
        return None
    if meta.get("training.deterministic") is not True:
        return None
    if meta.get("training.benchmark_mode") is not False:
        return None
    if is_set:
        if str(meta.get("model.output_residual_mode")) != "anchor_span":
            return None
        if str(meta.get("model.token_mlp.enabled", "")).lower() == "true":
            return None
    csv_path = json_path[:-5] + ".csv"
    n = 0; ppl = tppl = vram = sab = "NA"
    target = {}
    if os.path.exists(csv_path):
        try:
            # strip NUL bytes so a CSV being actively written by a live worker
            # (partial flush) parses its complete rows instead of aborting the pass
            with open(csv_path, newline="") as fcsv:
                rows = list(csv.DictReader(line.replace("\0", "") for line in fcsv))
        except Exception:
            rows = []
        n = len(rows)
        if rows:
            target = next((row for row in rows if row.get("epoch") == "10"), rows[-1])
            ppl = target.get("val/ppl", "NA"); tppl = target.get("train/ppl", "NA")
            vram = target.get("train/peak_vram_mib", "NA")
            sab = target.get("val/span_ablation_delta_ppl", "NA")
    if is_set:
        kind = "set"; blur = round(100 * ch / nh) if nh else "NA"; variant = f"b{blur}"
    else:
        # backend in the variant so the dense and landmark token baselines never collide
        bk = str(meta.get("model.backend", "NA")) or "NA"
        kind = "token"; blur = "NA"; variant = f"token-{bk}"; fh = ch = "NA"; sab = "NA"
    return {"model_kind": kind, "variant": variant, "blur_pct": blur,
            "seq_len": meta.get("data.seq_len", "NA"),
            "batch_size": meta.get("data.batch_size", "NA"),
            "seed": seed, "applied_seed": meta.get("training.applied_seed", "NA"),
            "backend": meta.get("model.backend", "NA"),
            "experiment_contract": meta.get("training.experiment_contract", "NA"),
            "diagnostics_contract": meta.get("training.diagnostics_contract", "NA"),
            "landmark_coverage": meta.get("model.backend_params.landmark_coverage", "NA"),
            "fine_heads": fh, "coarse_heads": ch, "epochs_done": n,
            "data_limit": "NONE", "final_val_ppl": ppl, "final_train_ppl": tppl,
            "peak_vram_mib": vram, "span_ablation_delta_ppl": sab,
            "span_ablation_fine_delta_ppl": target.get("val/span_ablation_fine_delta_ppl", "NA"),
            "span_ablation_coarse_delta_ppl": target.get("val/span_ablation_coarse_delta_ppl", "NA"),
            "effective_range_fine": target.get("val/effective_range_fine", "NA"),
            "effective_range_coarse": target.get("val/effective_range_coarse", "NA"),
            "routing_entropy_fine": target.get("val/routing_entropy_fine", "NA"),
            "routing_entropy_coarse": target.get("val/routing_entropy_coarse", "NA"),
            "routing_top1_fine": target.get("val/routing_top1_fine", "NA"),
            "routing_top1_coarse": target.get("val/routing_top1_coarse", "NA"),
            "router_param_norm_fine": target.get("ausa/fine/router_param_norm", "NA"),
            "router_param_norm_coarse": target.get("ausa/coarse/router_param_norm", "NA"),
            "router_gradient_norm_fine": target.get("ausa/fine/router_gradient_norm", "NA"),
            "router_gradient_norm_coarse": target.get("ausa/coarse/router_gradient_norm", "NA"),
            "pooling_effective_support_fine": target.get("ausa/fine/pooling_effective_support", "NA"),
            "pooling_effective_support_coarse": target.get("ausa/coarse/pooling_effective_support", "NA"),
            "grad_norm_set_post_pool_fine": target.get("ausa/fine/grad_norm_set_post_pool", "NA"),
            "grad_norm_set_post_pool_coarse": target.get("ausa/coarse/grad_norm_set_post_pool", "NA"),
            "grad_norm_set_post_blocks_fine": target.get("ausa/fine/grad_norm_set_post_blocks", "NA"),
            "grad_norm_set_post_blocks_coarse": target.get("ausa/coarse/grad_norm_set_post_blocks", "NA"),
            "loss_early_freq": target.get("val/loss_early_freq", "NA"),
            "loss_early_rare": target.get("val/loss_early_rare", "NA"),
            "loss_late_freq": target.get("val/loss_late_freq", "NA"),
            "loss_late_rare": target.get("val/loss_late_rare", "NA"),
            "baseline_attention_entropy": target.get("baseline/attention_entropy_mean", "NA"),
            "baseline_attention_top1": target.get("baseline/attention_top1_mean", "NA"),
            "baseline_attention_gradient_norm": target.get("baseline/attention_gradient_norm", "NA"),
            "baseline_attention_param_norm": target.get("baseline/attention_param_norm", "NA"),
            "csv_path": csv_path}


def main():
    roots = sys.argv[1:] or ["out/paper_mechanisms/sd_grid_seeded_v1"]
    best = {}
    duplicates = []
    for root in roots:
        for jp in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            r = normalize(jp)
            if not r:
                continue
            key = (
                r["model_kind"],
                r["backend"],
                str(r["seq_len"]),
                str(r["batch_size"]),
                r["variant"],
                r["seed"],
            )
            if key in best:
                duplicates.append((key, best[key]["csv_path"], r["csv_path"]))
            else:
                best[key] = r
    if duplicates:
        for key, first, second in duplicates:
            print(
                f"ERROR duplicate corrected cell {key}: {first}; {second}",
                file=sys.stderr,
            )
        raise SystemExit(2)
    rows = sorted(
        best.values(),
        key=lambda r: (
            r["model_kind"],
            str(r["seq_len"]),
            str(r["batch_size"]),
            r["variant"],
            r["seed"],
        ),
    )
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
