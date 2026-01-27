from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from config.load import load_config  # noqa: E402
from data.wikitext2 import Wikitext2Dataset  # noqa: E402
from models.baseline_token import TransformerLM  # noqa: E402
from models.set_only import SetOnlyLM  # noqa: E402
from train.experiment_logger import ExperimentLogger  # noqa: E402
from train.loop import evaluate, train_one_epoch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. model.d_model=256",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without loading data or running training",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default=None, help="W&B project override")
    parser.add_argument(
        "--wandb-tags",
        default="",
        help="Comma-separated W&B tags override",
    )
    parser.add_argument("--csv-path", default=None, help="CSV metrics path override")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(model_cfg: dict) -> torch.nn.Module:
    family = model_cfg["family"]
    if family == "baseline_token":
        return TransformerLM(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            nhead=model_cfg["nhead"],
            num_layers=model_cfg["num_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
            max_seq_len=model_cfg["max_seq_len"],
        )
    return SetOnlyLM(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        window_size=model_cfg["window_size"],
        stride=model_cfg["stride"],
        dropout=model_cfg["dropout"],
        max_seq_len=model_cfg["max_seq_len"],
        router_type=model_cfg["router_type"],
        router_topk=model_cfg["router_topk"],
        backend=model_cfg["backend"],
        backend_params=model_cfg.get("backend_params"),
        feature_mode=model_cfg.get("feature_mode", "geometry_only"),
        feature_params=model_cfg.get("feature_params"),
        adapter_type=model_cfg.get("adapter_type", "auto"),
        adapter_hidden_multiplier=model_cfg.get("adapter_hidden_multiplier", 2),
        adapter_budget_fraction=model_cfg.get("adapter_budget_fraction", 0.15),
        gamma=model_cfg.get("gamma", 1.0),
        beta=model_cfg.get("beta", 0.0),
    )


def build_dataloaders(data_cfg: dict) -> tuple[DataLoader, DataLoader, int]:
    if data_cfg["dataset"] != "wikitext2":
        raise ValueError("Only wikitext2 is supported in the unified runner")
    train_ds = Wikitext2Dataset(
        split="train",
        seq_len=data_cfg["seq_len"],
        limit=data_cfg.get("limit"),
        cache_root=data_cfg.get("cache_root"),
    )
    val_ds = Wikitext2Dataset(
        split="validation",
        seq_len=data_cfg["seq_len"],
        limit=data_cfg.get("limit"),
        cache_root=data_cfg.get("cache_root"),
    )
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=data_cfg["batch_size"], shuffle=False)
    return train_loader, val_loader, train_ds.vocab_size


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.dry_run:
        print("Dry run: config validated. No data loaded or training run.")
        print(cfg)
        return
    device = torch.device(args.device)

    train_loader, val_loader, vocab_size = build_dataloaders(cfg["data"])
    if cfg["model"].get("vocab_size", 0) in (0, None):
        cfg["model"]["vocab_size"] = vocab_size
    model = build_model(cfg["model"]).to(device)
    wandb_tags = [t for t in args.wandb_tags.split(",") if t]
    logger = ExperimentLogger(
        config=cfg,
        csv_path=args.csv_path,
        wandb_project=args.wandb_project,
        wandb_tags=wandb_tags or None,
        wandb_enable=True if args.wandb else None,
    )
    logger.log_model_complexity(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    epochs = cfg["training"]["epochs"]
    try:
        for epoch in range(1, epochs + 1):
            logger.start_epoch(num_train_samples=len(train_loader.dataset))
            train_metrics = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            set_diagnostics = model.get_diagnostics() if hasattr(model, "get_diagnostics") else None
            logger.log_epoch(epoch, train_metrics, val_metrics, set_diagnostics)
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f}"
            )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
