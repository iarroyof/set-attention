#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from config.load import load_config  # noqa: E402
from data.lca_cmp import build_lca_cmp_datasets  # noqa: E402
from run_experiment import apply_training_seed, build_model  # noqa: E402
from train.experiment_logger import ExperimentLogger  # noqa: E402
from train.lca_cmp import _cycle, evaluate_lca, train_lca_update_block, train_lca_updates  # noqa: E402
from train.metrics_impl import perplexity  # noqa: E402


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


def _write_curve_csv(csv_path: str | None, curve: list[tuple[int, float]]) -> Path | None:
    """Write the per-update train-loss sidecar ``<csv_stem>_curve.csv``."""
    if not curve:
        return None
    if csv_path is None:
        return None
    curve_path = Path(csv_path)
    curve_path = curve_path.with_name(f"{curve_path.stem}_curve{curve_path.suffix}")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["update", "train_loss"])
        for update, loss in curve:
            writer.writerow([update, f"{loss:.6f}"])
    return curve_path


def _write_eval_curve_csv(csv_path: str | None, eval_curve: list[tuple[int, float, float]]) -> Path | None:
    """Write the periodic-validation sidecar ``<csv_stem>_evalcurve.csv``."""
    if not eval_curve:
        return None
    if csv_path is None:
        return None
    curve_path = Path(csv_path)
    curve_path = curve_path.with_name(f"{curve_path.stem}_evalcurve{curve_path.suffix}")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["update", "val_loss", "val_acc"])
        for update, val_loss, val_acc in eval_curve:
            writer.writerow([update, f"{val_loss:.6f}", f"{val_acc:.6f}"])
    return curve_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=_flatten_overrides(args.override))
    if cfg.get("data", {}).get("dataset") != "lca_cmp":
        raise ValueError("scripts/run_lca_cmp.py requires data.dataset=lca_cmp")
    apply_training_seed(cfg)

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
    eval_every = int(cfg["training"].get("eval_every", 0) or 0)
    use_periodic_eval = eval_every > 0 and not args.preflight_one_step
    try:
        logger.start_epoch(
            num_train_samples=max_updates * int(cfg["data"]["batch_size"]) * grad_accum_steps
        )
        eval_curve: list[tuple[int, float, float]] = []
        if use_periodic_eval:
            # Chunked training over one shared data iterator so the trajectory
            # matches a single continuous run; validate after each chunk.
            clip = float(cfg["training"].get("clip_grad_norm", 1.0))
            batch_iter = _cycle(train_loader)
            total_loss = 0.0
            total_tokens = 0
            total_correct = 0.0
            completed_updates = 0
            curve: list[tuple[int, float]] = []
            val_metrics = None
            while completed_updates < max_updates:
                chunk = min(eval_every, max_updates - completed_updates)
                block = train_lca_update_block(
                    model,
                    batch_iter,
                    optimizer,
                    device,
                    max_updates=chunk,
                    clip_grad_norm=clip,
                    grad_accum_steps=grad_accum_steps,
                    record_curve=True,
                )
                steps = int(block["_optimizer_steps"])
                curve.extend((u + completed_updates, loss) for u, loss in block["_curve"])
                total_loss += float(block["loss"]) * int(block["valid_tokens"])
                total_tokens += int(block["valid_tokens"])
                total_correct += float(block["accuracy"]) * int(block["valid_tokens"])
                completed_updates += steps
                if steps < chunk:
                    break
                val_metrics = evaluate_lca(
                    model,
                    val_loader,
                    device,
                    vocab_size=int(cfg["model"]["vocab_size"]),
                    microbatch_size=eval_microbatch_size,
                )
                eval_curve.append(
                    (completed_updates, float(val_metrics["loss"]), float(val_metrics["accuracy"]))
                )
                print(
                    f"[periodic-eval] update={completed_updates} "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}",
                    flush=True,
                )
            loss_avg = total_loss / max(total_tokens, 1)
            train_metrics = {
                "loss": loss_avg,
                "ppl": perplexity(loss_avg),
                "accuracy": total_correct / max(total_tokens, 1),
                "valid_tokens": total_tokens,
                "completed_updates": completed_updates,
                "completed_microbatches": completed_updates * grad_accum_steps,
                "effective_batch_size": int(cfg["data"]["batch_size"]) * grad_accum_steps,
            }
            if val_metrics is None:
                val_metrics = evaluate_lca(
                    model,
                    val_loader,
                    device,
                    vocab_size=int(cfg["model"]["vocab_size"]),
                    microbatch_size=eval_microbatch_size,
                )
        else:
            train_metrics = train_lca_updates(
                model,
                train_loader,
                optimizer,
                device,
                max_updates=max_updates,
                clip_grad_norm=float(cfg["training"].get("clip_grad_norm", 1.0)),
                grad_accum_steps=grad_accum_steps,
                record_curve=True,
            )
            curve = train_metrics.pop("_curve", [])
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
        csv_path = args.csv_path or cfg.get("logging", {}).get("csv", {}).get("path")
        curve_path = _write_curve_csv(csv_path, curve)
        eval_curve_path = _write_eval_curve_csv(csv_path, eval_curve)
        print(
            f"updates={train_metrics['completed_updates']} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        if curve_path is not None:
            print(f"curve_csv={curve_path}")
        if eval_curve_path is not None:
            print(f"eval_curve_csv={eval_curve_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
