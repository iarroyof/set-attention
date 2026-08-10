#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from config.experiment_contracts import validate_experiment_contract  # noqa: E402
from config.load import load_config  # noqa: E402
from data.ar_hits import (  # noqa: E402
    build_bigram_counts_from_dataset,
    evaluate_ar_hit_group_ablation,
    evaluate_ar_hits,
    write_metrics_csv,
    write_metrics_json,
)
from data.ordered_text import dataset_provenance_bundle  # noqa: E402
from run_experiment import (  # noqa: E402
    apply_training_seed,
    attach_dataset_provenance,
    attach_resolved_metadata,
    build_dataloaders,
    build_model,
)
from train.checkpoints import load_checkpoint, sha256_file, source_commit, stable_config_digest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate natural repeated-bigram AR-hit metrics")
    parser.add_argument("--config", required=True, help="Registered LM config")
    parser.add_argument("--checkpoint", required=True, help="MRP-0 checkpoint to evaluate")
    parser.add_argument("--row", required=True, help="Registered row label: token, b0, b25, b100, ...")
    parser.add_argument("--seed", type=int, required=True, help="Applied seed for this replicate")
    parser.add_argument("--override", action="append", default=[], nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--min-inferential-targets", type=int, default=1000)
    parser.add_argument("--group-ablation", action="store_true")
    return parser.parse_args()


def _flatten_overrides(groups: list[list[str]]) -> list[str]:
    out: list[str] = []
    for group in groups:
        out.extend(group if isinstance(group, list) else [group])
    return out


def _row_from_metrics(
    *,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    row: str,
    seed: int,
    checkpoint_path: Path,
    checkpoint_digest: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "task": "natural_ar_hits",
        "dataset": cfg["data"]["dataset"],
        "row": row,
        "seed": int(seed),
        "config_fingerprint": stable_config_digest(cfg),
        "model_config_digest": stable_config_digest(cfg["model"]),
        "source_commit": source_commit() or "NA",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "data.dataset_digest": provenance["dataset_digest"],
        "data.tokenizer_digest": provenance["tokenizer_digest"],
        "data.train_token_count": provenance["train"]["token_count"],
        "data.validation_token_count": provenance["validation"]["token_count"],
        "model.implementation": cfg["model"].get("implementation", "NA"),
        "model.backend": cfg["model"].get("backend", "NA"),
        "model.max_seq_len": cfg["model"].get("max_seq_len", "NA"),
        "data.seq_len": cfg["data"].get("seq_len", "NA"),
        "data.batch_size": cfg["data"].get("batch_size", "NA"),
    }
    for key, value in metrics.items():
        out[f"val/{key}"] = value
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=_flatten_overrides(args.override))
    validate_experiment_contract(cfg)
    if cfg.get("data", {}).get("dataset") != "wikitext2":
        raise ValueError("natural AR-hit evaluation currently requires data.dataset=wikitext2")
    if int(cfg.get("training", {}).get("seed", args.seed)) != int(args.seed):
        cfg.setdefault("training", {})["seed"] = int(args.seed)
    apply_training_seed(cfg)

    train_loader, val_loader, vocab_size = build_dataloaders(cfg["data"])
    provenance = dataset_provenance_bundle(train_loader.dataset, val_loader.dataset)
    attach_dataset_provenance(cfg, provenance)
    if cfg["model"].get("vocab_size", 0) in (0, None):
        cfg["model"]["vocab_size"] = vocab_size
    device = torch.device(args.device)
    model = build_model(cfg["model"]).to(device)
    attach_resolved_metadata(cfg, model)

    checkpoint = Path(args.checkpoint)
    checkpoint_digest = sha256_file(checkpoint)
    payload = load_checkpoint(
        checkpoint,
        model=model,
        map_location=device,
        expected_model_config=cfg["model"],
        expected_dataset_digest=str(provenance["dataset_digest"]),
        expected_tokenizer_digest=str(provenance["tokenizer_digest"]),
    )
    payload_seed = int(payload.get("applied_seed", payload.get("requested_seed", args.seed)))
    if payload_seed != int(args.seed):
        raise ValueError(f"checkpoint applied seed mismatch: expected {args.seed}, found {payload_seed}")

    train_bigram_counts = build_bigram_counts_from_dataset(train_loader.dataset)
    metrics = evaluate_ar_hits(
        model,
        val_loader,
        device,
        train_bigram_counts=train_bigram_counts,
        min_inferential_targets=int(args.min_inferential_targets),
        collect_blocks=True,
    )
    blocks = metrics.pop("blocks", [])
    if args.group_ablation:
        metrics.update(
            evaluate_ar_hit_group_ablation(
                model,
                val_loader,
                device,
                metrics,
                train_bigram_counts=train_bigram_counts,
                min_inferential_targets=int(args.min_inferential_targets),
            )
        )

    row = _row_from_metrics(
        cfg=cfg,
        metrics=metrics,
        row=args.row,
        seed=args.seed,
        checkpoint_path=checkpoint,
        checkpoint_digest=checkpoint_digest,
        provenance=provenance,
    )
    payload_out = {
        "row": row,
        "metrics": metrics,
        "blocks": blocks,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_digest,
            "epoch": int(payload.get("epoch", -1)),
            "global_step": int(payload.get("global_step", -1)),
        },
        "provenance": provenance,
    }
    write_metrics_json(args.out_json, payload_out)
    if args.out_csv is not None:
        write_metrics_csv(args.out_csv, row)
    print(f"ar_hit_metrics={args.out_json} checkpoint_sha256={checkpoint_digest}")


if __name__ == "__main__":
    main()
