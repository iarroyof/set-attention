#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from config.load import load_config  # noqa: E402
from data.lca_cmp import build_lca_cmp_datasets  # noqa: E402
from run_experiment import build_model  # noqa: E402
from train.experiment_logger import ExperimentLogger  # noqa: E402
from train.lca_cmp import evaluate_lca, train_lca_updates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRP-lca-cmp synthetic aggregation runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-one-step", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-tags", default="")
    return parser.parse_args()


def _flatten_overrides(groups: list[list[str]]) -> list[str]:
    out: list[str] = []
    for group in groups:
        out.extend(group if isinstance(group, list) else [group])
    return out


def _loader(dataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=_flatten_overrides(args.override))
    if cfg.get("data", {}).get("dataset") != "lca_cmp":
        raise ValueError("scripts/run_lca_cmp.py requires data.dataset=lca_cmp")

    train_ds, val_ds = build_lca_cmp_datasets(cfg["data"])
    cfg["task"] = "lca_cmp"
    cfg["data"]["task"] = "lca_cmp"
    cfg["model"]["vocab_size"] = int(cfg["data"].get("vocab_size", train_ds.vocab_size))
    cfg["model"]["max_seq_len"] = int(cfg["data"]["seq_len"])

    if args.dry_run:
        print("Dry run: LCA config and generator validated.")
        print(f"train_digest={train_ds.dataset_digest}")
        print(f"validation_digest={val_ds.dataset_digest}")
        return

    train_loader = _loader(train_ds, int(cfg["data"]["batch_size"]), shuffle=True)
    val_loader = _loader(val_ds, int(cfg["data"]["batch_size"]), shuffle=False)
    device = torch.device(args.device)
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    wandb_tags = [tag for tag in args.wandb_tags.split(",") if tag]
    logger = ExperimentLogger(
        config=cfg,
        csv_path=args.csv_path,
        wandb_project=args.wandb_project,
        wandb_tags=wandb_tags or None,
        wandb_enable=True if args.wandb else None,
    )
    logger.log_model_complexity(model)

    max_updates = 1 if args.preflight_one_step else int(cfg["training"].get("max_updates", 20_000))
    grad_accum_steps = int(cfg["training"].get("grad_accum_steps", 1) or 1)
    eval_microbatch_size = cfg["training"].get("eval_microbatch_size")
    eval_microbatch_size = None if eval_microbatch_size is None else int(eval_microbatch_size)
    try:
        logger.start_epoch(
            num_train_samples=max_updates * int(cfg["data"]["batch_size"]) * grad_accum_steps
        )
        train_metrics = train_lca_updates(
            model,
            train_loader,
            optimizer,
            device,
            max_updates=max_updates,
            clip_grad_norm=float(cfg["training"].get("clip_grad_norm", 1.0)),
            grad_accum_steps=grad_accum_steps,
        )
        completed_updates = int(train_metrics.pop("_optimizer_steps", 0))
        train_metrics.pop("_microbatches_per_optimizer_step", None)
        train_metrics["completed_updates"] = completed_updates
        train_metrics["completed_microbatches"] = completed_updates * grad_accum_steps
        train_metrics["effective_batch_size"] = int(cfg["data"]["batch_size"]) * grad_accum_steps
        val_metrics = evaluate_lca(
            model,
            val_loader,
            device,
            vocab_size=int(cfg["model"]["vocab_size"]),
            microbatch_size=eval_microbatch_size,
        )
        set_diagnostics = model.get_diagnostics() if hasattr(model, "get_diagnostics") else None
        logger.log_epoch(1, train_metrics, val_metrics, set_diagnostics)
        print(
            f"updates={completed_updates} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
