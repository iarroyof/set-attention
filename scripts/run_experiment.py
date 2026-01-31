from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from config.load import load_config  # noqa: E402
from data.wikitext2 import Wikitext2Dataset, Wikitext2IterableDataset  # noqa: E402
from models.baseline_token import TransformerLM  # noqa: E402
from models.seq2seq import Seq2SeqTransformer  # noqa: E402
from models.set_only import SetOnlyLM  # noqa: E402
from set_attention.training.seq_loaders import get_seq2seq_datasets  # noqa: E402
from train.experiment_logger import ExperimentLogger  # noqa: E402
from train.loop import (
    evaluate,
    evaluate_seq2seq,
    train_one_epoch,
    train_one_epoch_seq2seq,
)  # noqa: E402
from train.metrics_impl import bleu_score, rouge_l_f1  # noqa: E402
from train.metrics_schema import detect_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        nargs="+",
        help="Override config values, e.g. model.d_model=256 data.limit=10",
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
        pooling=model_cfg.get("pooling", "mean"),
        multiscale=model_cfg.get("multiscale", False),
        sig_gating=model_cfg.get("sig_gating"),
        d_phi=model_cfg.get("d_phi"),
        geometry=model_cfg.get("geometry"),
        features=model_cfg.get("features"),
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


def _make_loader(ds, batch_size: int, shuffle: bool):
    from torch.utils.data import IterableDataset

    if isinstance(ds, IterableDataset):
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_dataloaders(data_cfg: dict) -> tuple[DataLoader, DataLoader, int]:
    if data_cfg["dataset"] != "wikitext2":
        raise ValueError("Only wikitext2 is supported in the unified runner")
    streaming = bool(data_cfg.get("streaming", True))
    if streaming:
        train_ds = Wikitext2IterableDataset(
            split="train",
            seq_len=data_cfg["seq_len"],
            limit=data_cfg.get("limit"),
            cache_root=data_cfg.get("cache_root"),
        )
        val_ds = Wikitext2IterableDataset(
            split="validation",
            seq_len=data_cfg["seq_len"],
            limit=data_cfg.get("limit"),
            cache_root=data_cfg.get("cache_root"),
        )
    else:
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
    train_loader = _make_loader(train_ds, data_cfg["batch_size"], shuffle=True)
    val_loader = _make_loader(val_ds, data_cfg["batch_size"], shuffle=False)
    return train_loader, val_loader, train_ds.vocab_size


def build_seq2seq_dataloaders(data_cfg: dict, shared_vocab: bool) -> tuple[DataLoader, DataLoader, dict]:
    streaming = bool(data_cfg.get("streaming", True))
    train_ds, val_ds = get_seq2seq_datasets(
        dataset=data_cfg.get("seq_dataset", ""),
        limit=data_cfg.get("limit"),
        val_limit=data_cfg.get("val_limit"),
        demo=bool(data_cfg.get("demo", False)),
        demo_samples=int(data_cfg.get("demo_samples", 200)),
        max_len=int(data_cfg.get("seq_len", data_cfg.get("max_len", 64))),
        cache_dir=data_cfg.get("cache_root"),
        shared_vocab=shared_vocab,
        val_split=float(data_cfg.get("val_split", 0.2)),
        split_seed=int(data_cfg.get("split_seed", 42)),
        streaming=streaming,
    )
    train_loader = _make_loader(train_ds, data_cfg["batch_size"], shuffle=True)
    val_loader = _make_loader(val_ds, data_cfg["batch_size"], shuffle=False)
    vocab = {
        "vocab_size": train_ds.vocab_size,
        "pad_id": train_ds.pad_id,
        "bos_id": train_ds.bos_id,
        "eos_id": train_ds.eos_id,
        "decode": train_ds.decode,
        "max_len": data_cfg.get("seq_len", data_cfg.get("max_len", 64)),
    }
    return train_loader, val_loader, vocab


def main() -> None:
    args = parse_args()
    overrides = []
    for group in args.override:
        if isinstance(group, list):
            overrides.extend(group)
        else:
            overrides.append(group)
    cfg = load_config(args.config, overrides=overrides)
    if args.dry_run:
        print("Dry run: config validated. No data loaded or training run.")
        print(cfg)
        return
    device = torch.device(args.device)

    task = detect_task(cfg)
    if task == "seq2seq":
        shared_vocab = bool(cfg.get("model", {}).get("seq2seq", {}).get("shared_vocab", True))
        train_loader, val_loader, vocab = build_seq2seq_dataloaders(cfg["data"], shared_vocab)
        if cfg["model"].get("vocab_size", 0) in (0, None):
            cfg["model"]["vocab_size"] = vocab["vocab_size"]
        family = cfg["model"]["family"]
        num_heads = cfg["model"].get("num_heads") or cfg["model"].get("nhead", 8)
        num_layers = cfg["model"].get("num_layers", 4)
        d_model = cfg["model"].get("d_model", 512)
        dim_ff = cfg["model"].get("dim_feedforward", d_model * 4)
        dropout = cfg["model"].get("dropout", 0.1)
        max_len = cfg["data"].get("seq_len", cfg["data"].get("max_len", 64))
        encoder_family = "set_only" if family in {"set_only", "encoder_set_only"} else "baseline_token"
        set_only_cfg = cfg["model"] if encoder_family == "set_only" else None
        model = Seq2SeqTransformer(
            vocab_size=cfg["model"]["vocab_size"],
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            max_len=max_len,
            encoder_family=encoder_family,
            set_only_cfg=set_only_cfg,
            shared_embeddings=None,
            pad_id=vocab["pad_id"],
            bos_id=vocab["bos_id"],
            eos_id=vocab["eos_id"],
        ).to(device)
    else:
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
            num_samples = None
            try:
                num_samples = len(train_loader.dataset)
            except Exception:
                num_samples = cfg.get("data", {}).get("limit")
            logger.start_epoch(num_train_samples=num_samples or 0)
            if task == "seq2seq":
                train_metrics = train_one_epoch_seq2seq(
                    model, train_loader, optimizer, device, pad_id=vocab["pad_id"]
                )
                eval_bundle = evaluate_seq2seq(
                    model,
                    val_loader,
                    device,
                    pad_id=vocab["pad_id"],
                    bos_id=vocab["bos_id"],
                    eos_id=vocab["eos_id"],
                    decode_fn=vocab["decode"],
                    max_len=int(vocab["max_len"]),
                )
                val_metrics = {"loss": eval_bundle["loss"]}
                if eval_bundle["preds"]:
                    val_metrics["bleu"] = bleu_score(eval_bundle["preds"], eval_bundle["refs"])
                    val_metrics["rougeL"] = rouge_l_f1(eval_bundle["preds"], eval_bundle["refs"])
            else:
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
