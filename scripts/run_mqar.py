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

from common.repro import set_seed  # noqa: E402
from config.experiment_contracts import validate_experiment_contract  # noqa: E402
from config.load import load_config  # noqa: E402
from data.mqar import build_mqar_datasets, mqar_provenance  # noqa: E402
from run_experiment import (  # noqa: E402
    append_checkpoint_manifest,
    apply_training_seed,
    attach_dataset_provenance,
    attach_resolved_metadata,
    build_model,
    checkpoint_path,
)
from set_attention.utils.repro_workers import make_worker_init_fn  # noqa: E402
from train.checkpoints import build_checkpoint_payload, save_checkpoint, source_commit  # noqa: E402
from train.experiment_logger import ExperimentLogger  # noqa: E402
from train.mqar import (  # noqa: E402
    _cycle,
    evaluate_mqar,
    evaluate_mqar_group_ablation,
    train_mqar_update_block,
    train_mqar_updates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MQAR generator/trainer runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="validate config and generator only")
    parser.add_argument("--preflight-one-step", action="store_true", help="run exactly one train update and eval")
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


def _loader(dataset, batch_size: int, *, seed: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        worker_init_fn=make_worker_init_fn(int(seed)),
        num_workers=max(0, int(num_workers)),
        persistent_workers=False,
    )


def _save_final_checkpoint(
    *,
    cfg: dict,
    logger: ExperimentLogger,
    model: torch.nn.Module,
    dataset_provenance: dict,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    global_step: int,
) -> None:
    path = checkpoint_path(cfg, "final.pt")
    payload = build_checkpoint_payload(
        model=model,
        config=cfg,
        config_fingerprint=logger.config_fingerprint,
        dataset_provenance=dataset_provenance,
        epoch=1,
        global_step=global_step,
        optimizer=optimizer,
        loaders={"train": train_loader, "validation": val_loader},
    )
    digest = save_checkpoint(payload, path)
    append_checkpoint_manifest(cfg, path=path, digest=digest, epoch=1, global_step=global_step)
    print(f"checkpoint={path} sha256={digest}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=_flatten_overrides(args.override))
    validate_experiment_contract(cfg)
    if cfg.get("data", {}).get("dataset") != "mqar":
        raise ValueError("scripts/run_mqar.py requires data.dataset=mqar")
    apply_training_seed(cfg)
    cfg.setdefault("resolved", {})["source_commit"] = source_commit() or "NA"

    train_ds, val_ds = build_mqar_datasets(cfg["data"])
    provenance = mqar_provenance(train_ds, val_ds)
    attach_dataset_provenance(cfg, provenance)
    cfg["model"]["vocab_size"] = int(cfg["data"].get("vocab_size", train_ds.vocab_size))
    cfg["model"]["max_seq_len"] = int(cfg["data"]["seq_len"])

    if args.dry_run:
        print("Dry run: MQAR config, generator, seeds, and provenance validated.")
        print(f"train_digest={train_ds.dataset_digest}")
        print(f"validation_digest={val_ds.dataset_digest}")
        print(f"dataset_digest={provenance['dataset_digest']}")
        return

    # Re-apply after generator construction so model/optimizer construction starts
    # from the registered training seed, independent of synthetic data generation.
    training_cfg = cfg["training"]
    set_seed(
        int(training_cfg["applied_seed"]),
        deterministic=bool(training_cfg.get("deterministic", False)),
        benchmark_mode=bool(training_cfg.get("benchmark_mode", False)),
        strict_deterministic=bool(training_cfg.get("strict_deterministic", False)),
    )

    train_loader = _loader(
        train_ds,
        int(cfg["data"]["batch_size"]),
        seed=int(cfg["data"].get("train_loader_seed", 0)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )
    val_loader = _loader(
        val_ds,
        int(cfg["data"]["batch_size"]),
        seed=int(cfg["data"].get("validation_loader_seed", 1)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )
    device = torch.device(args.device)
    model = build_model(cfg["model"]).to(device)
    attach_resolved_metadata(cfg, model)
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
    eval_every_updates = 0 if args.preflight_one_step else int(cfg["training"].get("eval_every_updates", 0) or 0)
    grad_accum_steps = int(cfg["training"].get("grad_accum_steps", 1) or 1)
    eval_microbatch_size = cfg["training"].get("eval_microbatch_size")
    eval_microbatch_size = None if eval_microbatch_size is None else int(eval_microbatch_size)
    calibration_threshold = float(cfg["training"].get("calibration_accuracy_threshold", 0.99))
    calibration_consecutive_required = int(cfg["training"].get("calibration_consecutive_evals", 2) or 2)
    try:
        if eval_every_updates > 0:
            batch_iter = _cycle(train_loader)
            global_step = 0
            consecutive_hits = 0
            gate_passed = False
            while global_step < max_updates:
                block_updates = min(eval_every_updates, max_updates - global_step)
                logger.start_epoch(
                    num_train_samples=block_updates
                    * int(cfg["data"]["batch_size"])
                    * grad_accum_steps
                )
                train_metrics = train_mqar_update_block(
                    model,
                    batch_iter,
                    optimizer,
                    device,
                    max_updates=block_updates,
                    clip_grad_norm=float(cfg["training"].get("clip_grad_norm", 1.0)),
                    grad_accum_steps=grad_accum_steps,
                )
                completed_now = int(train_metrics.pop("_optimizer_steps", 0))
                train_metrics.pop("_microbatches_per_optimizer_step", None)
                global_step += completed_now
                train_metrics["completed_updates"] = global_step
                train_metrics["completed_microbatches"] = global_step * grad_accum_steps
                val_metrics = evaluate_mqar(
                    model,
                    val_loader,
                    device,
                    microbatch_size=eval_microbatch_size,
                )
                if bool(cfg["training"].get("evaluate_group_ablation", False)):
                    val_metrics.update(
                        evaluate_mqar_group_ablation(
                            model,
                            val_loader,
                            device,
                            val_metrics,
                            microbatch_size=eval_microbatch_size,
                        )
                    )
                if float(val_metrics["accuracy"]) >= calibration_threshold:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0
                gate_passed = consecutive_hits >= calibration_consecutive_required
                val_metrics["calibration_consecutive_hits"] = consecutive_hits
                val_metrics["calibration_gate_passed"] = gate_passed
                val_metrics["calibration_selected_update"] = global_step if gate_passed else None
                set_diagnostics = model.get_diagnostics() if hasattr(model, "get_diagnostics") else None
                logger.log_epoch(global_step, train_metrics, val_metrics, set_diagnostics)
                print(
                    f"updates={global_step} train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                    f"calibration_hits={consecutive_hits}/{calibration_consecutive_required}"
                )
                if gate_passed or completed_now <= 0:
                    break
        else:
            logger.start_epoch(
                num_train_samples=(
                    len(train_ds)
                    if grad_accum_steps == 1
                    else max_updates * int(cfg["data"]["batch_size"]) * grad_accum_steps
                )
            )
            train_metrics = train_mqar_updates(
                model,
                train_loader,
                optimizer,
                device,
                max_updates=max_updates,
                clip_grad_norm=float(cfg["training"].get("clip_grad_norm", 1.0)),
                grad_accum_steps=grad_accum_steps,
            )
            global_step = int(train_metrics.pop("_optimizer_steps", 0))
            train_metrics.pop("_microbatches_per_optimizer_step", None)
            train_metrics["completed_updates"] = global_step
            train_metrics["completed_microbatches"] = global_step * grad_accum_steps
            val_metrics = evaluate_mqar(
                model,
                val_loader,
                device,
                microbatch_size=eval_microbatch_size,
            )
            if bool(cfg["training"].get("evaluate_group_ablation", False)):
                val_metrics.update(
                    evaluate_mqar_group_ablation(
                        model,
                        val_loader,
                        device,
                        val_metrics,
                        microbatch_size=eval_microbatch_size,
                    )
                )
            set_diagnostics = model.get_diagnostics() if hasattr(model, "get_diagnostics") else None
            logger.log_epoch(1, train_metrics, val_metrics, set_diagnostics)
            print(
                f"updates={global_step} train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )
        if cfg["training"].get("checkpoint", {}).get("save_final") and not args.preflight_one_step:
            _save_final_checkpoint(
                cfg=cfg,
                logger=logger,
                model=model,
                dataset_provenance=provenance,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                global_step=global_step,
            )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
