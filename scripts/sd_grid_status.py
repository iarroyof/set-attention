#!/usr/bin/env python3
"""Backend-aware grid status scanner for the SD-9.x full-grid driver.

Emits ONE canonical line per run found under the given roots:

    cell_id <TAB> epochs <TAB> target_val_ppl <TAB> peak_vram_mib <TAB> span_abl <TAB> csv_path

cell_id = "{family}|{backend}|{L}|{variant}|{seed}" where
  family  = set | token        (set iff model.multiresolution.enabled)
  backend = model.backend      (exact = dense, landmark = linear-landmark) -> distinguishes
                                the dense vs landmark token baselines (fixes the conflation)
  variant = set:  f{fine}c{coarse}   (head split; no rounding, exact match to the driver manifest)
            token: token
  L, seed straight from metadata.

Guards (same discipline as normalize_sd9x_runs.py):
  - DROP every non-null data.limit/data.val_limit row (probe/smoke).
  - DROP malformed seeds (not in 0..4).
  - DROP non-baseline token implementations instead of conflating hybrid runs.
  - NUL-safe CSV read (a live worker's partial flush won't crash the scan).

Set SD_GRID_REQUIRE_CONTRACT=sd_grid_seeded_v1 for fail-closed validation of
the current exact-dense matrix. In that mode malformed metadata, missing
diagnostics on completed runs, or duplicate cell IDs terminate the scan.

The driver consumes this to decide which cells are already complete (epochs>=10) and must be
skipped; analysts consume it for a backend-correct set-vs-token table.

Metrics come from `SD_GRID_TARGET_EPOCHS` (default 10), even when a reusable run continued
past the registered budget. This prevents a 30-epoch run from contributing its epoch-30 PPL
to a 10-epoch comparison.
"""
from __future__ import annotations
import csv, glob, json, os, sys

VALID_SEEDS = {"0", "1", "2", "3", "4"}
TARGET_EPOCHS = int(os.environ.get("SD_GRID_TARGET_EPOCHS", "10"))
REQUIRED_CONTRACT = os.environ.get("SD_GRID_REQUIRE_CONTRACT", "")


def _is_null_limit(value):
    return value is None or str(value).strip().lower() in {"", "na", "none", "null"}


def _is_value(value):
    return value is not None and str(value).strip().lower() not in {"", "na", "none", "null"}


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


def _strict_contract_errors(meta, target_row, epochs):
    errors = []

    def expect(key, expected):
        actual = meta.get(key)
        if isinstance(expected, float):
            try:
                matches = abs(float(actual) - expected) <= 1e-12
            except (TypeError, ValueError):
                matches = False
            if not matches:
                errors.append(f"{key} expected numeric {expected!r}, got {actual!r}")
            return
        if actual != expected:
            errors.append(f"{key} expected {expected!r}, got {actual!r}")

    expect("training.experiment_contract", "sd_grid_seeded_v1")
    expect("training.diagnostics_contract", "current_matrix_v1")
    expect("training.seed_applied", True)
    expect("training.deterministic", True)
    expect("training.benchmark_mode", False)
    expect("model.attention_family", "dense")
    expect("model.backend", "exact")
    expect("model.d_model", 384)
    expect("model.dim_feedforward", 1536)
    expect("model.num_layers", 6)
    expect("model.num_heads", 8)
    expect("training.epochs", 10)
    expect("training.lr", 0.0001)
    expect("training.warmup_steps", 1000)
    expect("data.dataset", "wikitext2")
    if not _is_null_limit(meta.get("data.limit")):
        errors.append(f"data.limit must be null, got {meta.get('data.limit')!r}")
    if not _is_null_limit(meta.get("data.val_limit")):
        errors.append(f"data.val_limit must be null, got {meta.get('data.val_limit')!r}")
    seed = str(meta.get("training.seed", "?")).strip()
    if str(meta.get("training.applied_seed", "?")).strip() != seed:
        errors.append("training.applied_seed does not match training.seed")
    if str(meta.get("training.torch_initial_seed", "?")).strip() != seed:
        errors.append("training.torch_initial_seed does not match training.seed")

    implementation = str(meta.get("model.implementation", ""))
    groups = []
    if implementation == "set_only":
        for key, expected in {
            "model.output_residual_mode": "anchor_span",
            "resolved.output_residual_mode": "anchor_span",
            "model.token_mlp.enabled": False,
            "model.anchor.enabled": False,
            "resolved.anchor_enabled": False,
            "model.candidate_fiber": "endpoint_window",
            "resolved.candidate_fiber": "endpoint_window",
            "model.multiresolution.enabled": True,
            "resolved.multiresolution_enabled": True,
            "model.multivector_basis.enabled": False,
        }.items():
            expect(key, expected)
        groups = meta.get("resolved.multiresolution_groups", [])
        if not isinstance(groups, list) or not groups:
            errors.append("resolved.multiresolution_groups must be non-empty")
            groups = []
        names = set()
        total_heads = 0
        for group in groups:
            if not isinstance(group, dict):
                errors.append("resolved multiresolution group is not a mapping")
                continue
            name = str(group.get("name", ""))
            if name in names:
                errors.append(f"duplicate group name {name!r}")
            names.add(name)
            expected_ws = {"fine": (2, 1), "coarse": (4, 2)}.get(name)
            if expected_ws is None:
                errors.append(f"unexpected group name {name!r}")
                continue
            if (group.get("window_size"), group.get("stride")) != expected_ws:
                errors.append(f"group {name!r} topology mismatch")
            total_heads += int(group.get("num_heads", 0))
        if total_heads != 8:
            errors.append(f"resolved group heads total {total_heads}, expected 8")
    elif implementation == "baseline_token":
        if meta.get("model.backend_params") not in (None, {}, "NA"):
            errors.append("exact token model.backend_params must be empty/absent")
    else:
        errors.append(f"unsupported model.implementation {implementation!r}")

    if epochs >= TARGET_EPOCHS and target_row is not None:
        required = [
            "val/ppl",
            "train/peak_vram_mib",
            "val/loss_early_freq",
            "val/loss_early_rare",
            "val/loss_late_freq",
            "val/loss_late_rare",
        ]
        if implementation == "baseline_token":
            required += [
                "baseline/attention_entropy_mean",
                "baseline/attention_top1_mean",
                "baseline/attention_gradient_norm",
                "baseline/attention_param_norm",
            ]
        else:
            required += [
                "val/span_ablation_delta_ppl",
            ]
            for group in groups:
                name = str(group.get("name", ""))
                required += [
                    f"val/span_ablation_{name}_delta_ppl",
                    f"val/effective_range_{name}",
                    f"val/routing_entropy_{name}",
                    f"val/routing_top1_{name}",
                    f"ausa/{name}/routing_entropy_norm",
                    f"ausa/{name}/router_top1_weight",
                    f"ausa/{name}/pooling_effective_support",
                    f"ausa/{name}/router_gradient_norm",
                    f"ausa/{name}/router_param_norm",
                    f"ausa/{name}/grad_norm_token_pre_pool",
                    f"ausa/{name}/grad_norm_set_post_pool",
                    f"ausa/{name}/grad_norm_set_post_blocks",
                ]
        for key in required:
            if not _is_value(target_row.get(key)):
                errors.append(f"completed run missing diagnostic {key}")
    return errors


def scan(json_path):
    try:
        meta = json.load(open(json_path))
    except Exception:
        return None
    if not _is_null_limit(meta.get("data.limit", meta.get("data_limit"))):
        return None
    if not _is_null_limit(meta.get("data.val_limit")):
        return None
    seed = str(meta.get("training.seed", "?")).strip()
    if seed not in VALID_SEEDS:
        return None
    backend = str(meta.get("model.backend", "NA")) or "NA"
    L = str(meta.get("data.seq_len", "NA"))
    implementation = str(meta.get("model.implementation", ""))
    is_multires = (
        implementation == "set_only"
        and str(meta.get("model.multiresolution.enabled", "")).lower() == "true"
    )
    if implementation not in {"set_only", "baseline_token"}:
        return None
    if implementation == "set_only" and not is_multires:
        return None
    bs = str(meta.get("data.batch_size", "NA"))
    if is_multires:
        fh, ch = heads(meta)
        family, variant = "set", f"f{fh}c{ch}"
    else:
        family, variant = "token", "token"
    csv_path = json_path[:-5] + ".csv"
    n = 0
    ppl = vram = sab = "NA"
    target = None
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="") as fh2:
                rows = list(csv.DictReader(line.replace("\0", "") for line in fh2))
        except Exception:
            rows = []
        n = len(rows)
        if rows:
            target = next(
                (row for row in rows if row.get("epoch") == str(TARGET_EPOCHS)),
                rows[-1],
            )
            ppl = target.get("val/ppl", "NA")
            vram = target.get("train/peak_vram_mib", "NA")
            sab = target.get("val/span_ablation_delta_ppl", "NA")
    if REQUIRED_CONTRACT:
        if REQUIRED_CONTRACT != "sd_grid_seeded_v1":
            raise ValueError(f"unsupported SD_GRID_REQUIRE_CONTRACT={REQUIRED_CONTRACT!r}")
        errors = _strict_contract_errors(meta, target, n)
        if errors:
            raise ValueError(f"{json_path}: " + "; ".join(errors))
    cell_id = f"{family}|{backend}|{L}|{variant}|b{bs}|{seed}"
    return cell_id, n, ppl, vram, sab, csv_path


def main():
    global REQUIRED_CONTRACT
    if sys.argv[1:]:
        roots = sys.argv[1:]
    else:
        roots = ["out/paper_mechanisms/sd_grid_seeded_v1"]
        REQUIRED_CONTRACT = REQUIRED_CONTRACT or "sd_grid_seeded_v1"
    best = {}
    duplicates = {}
    errors = []
    for root in roots:
        for jp in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            try:
                r = scan(jp)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not r:
                continue
            cid, n = r[0], r[1]
            if cid in best:
                duplicates.setdefault(cid, [best[cid]]).append(r)
            if cid not in best or n > best[cid][1]:
                best[cid] = r
    if REQUIRED_CONTRACT and (errors or duplicates):
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        for cid, rows in sorted(duplicates.items()):
            paths = ", ".join(row[5] for row in rows)
            print(f"ERROR duplicate corrected cell {cid}: {paths}", file=sys.stderr)
        raise SystemExit(2)
    for cid, rows in sorted(duplicates.items()):
        paths = ", ".join(row[5] for row in rows)
        print(f"WARN duplicate cell {cid}: {paths}", file=sys.stderr)
    for cid in sorted(best):
        cid_, n, ppl, vram, sab, csvp = best[cid]
        print(f"{cid}\t{n}\t{ppl}\t{vram}\t{sab}\t{csvp}")


if __name__ == "__main__":
    main()
