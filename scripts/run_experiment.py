from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from config.load import load_config  # noqa: E402
from config.experiment_contracts import validate_experiment_contract  # noqa: E402
from common.repro import set_seed  # noqa: E402
from data.ordered_text import dataset_provenance_bundle  # noqa: E402
from data.wikitext2 import Wikitext2Dataset, Wikitext2IterableDataset  # noqa: E402
from models.baseline_token import TransformerLM  # noqa: E402
from models.hybrid_token_set_lm import HybridTokenSetLM  # noqa: E402
from models.seq2seq import Seq2SeqTransformer  # noqa: E402
from models.set_only import SetOnlyLM  # noqa: E402
from set_attention.training.seq_loaders import get_seq2seq_datasets  # noqa: E402
from set_attention.utils.repro_workers import make_worker_init_fn  # noqa: E402
from train.checkpoints import (  # noqa: E402
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    source_commit,
)
from train.experiment_logger import ExperimentLogger  # noqa: E402
from train.loop import (
    evaluate,
    evaluate_seq2seq,
    train_one_epoch,
    train_one_epoch_seq2seq,
)  # noqa: E402
from train.metrics_impl import bleu_score, perplexity, rouge_l_f1  # noqa: E402
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
    impl = model_cfg["implementation"]
    if impl == "baseline_token":
        nhead = model_cfg.get("num_heads", model_cfg.get("nhead", 8))
        return TransformerLM(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            nhead=nhead,
            num_layers=model_cfg["num_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
            attn_dropout=model_cfg.get("attn_dropout"),
            resid_dropout=model_cfg.get("resid_dropout"),
            ffn_dropout=model_cfg.get("ffn_dropout"),
            max_seq_len=model_cfg["max_seq_len"],
            attention_family=model_cfg.get("attention_family", "dense"),
            backend=model_cfg.get("backend", "exact"),
            backend_params=model_cfg.get("backend_params"),
            causal=bool(model_cfg.get("causal", True)),
        )
    if impl == "hybrid_token_set":
        return HybridTokenSetLM(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg.get("num_heads", model_cfg.get("nhead", 8)),
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
            attn_dropout=model_cfg.get("attn_dropout"),
            resid_dropout=model_cfg.get("resid_dropout"),
            ffn_dropout=model_cfg.get("ffn_dropout"),
            max_seq_len=model_cfg["max_seq_len"],
            attention_family=model_cfg.get("attention_family", "sparse"),
            backend=model_cfg.get("backend", "local_band"),
            backend_params=model_cfg.get("backend_params"),
            hybrid=model_cfg.get("hybrid"),
            pooling=model_cfg.get("pooling", "mean"),
            pooling_multihead=bool(model_cfg.get("pooling_multihead", False)),
            d_phi=model_cfg.get("d_phi"),
            set_state_dim=model_cfg.get("set_state_dim"),
            feature_mode=model_cfg.get("feature_mode", "geometry_only"),
            feature_params=model_cfg.get("feature_params"),
            router_type=model_cfg["router_type"],
            router_topk=model_cfg["router_topk"],
            router_multihead=bool(model_cfg.get("router_multihead", False)),
            router_temperature=float(model_cfg.get("router_temperature", 1.0)),
            router_min_temp=float(model_cfg.get("router", {}).get("min_temp", 0.5)),
            router_score_mode=str(
                model_cfg.get("router", {}).get("score_mode", "candidate_gather")
            ),
            adapter_type=model_cfg.get("adapter_type", "auto"),
            adapter_hidden_multiplier=model_cfg.get("adapter_hidden_multiplier", 2),
            gamma=model_cfg.get("gamma", 1.0),
            beta=model_cfg.get("beta", 0.0),
            set_causality_mode=model_cfg.get("set_causality_mode", "strict_past"),
            output_residual_mode=model_cfg.get("output_residual_mode", "empty_only"),
        )
    return SetOnlyLM(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        window_size=model_cfg["window_size"],
        stride=model_cfg["stride"],
        dropout=model_cfg["dropout"],
        attn_dropout=model_cfg.get("attn_dropout"),
        resid_dropout=model_cfg.get("resid_dropout"),
        ffn_dropout=model_cfg.get("ffn_dropout"),
        max_seq_len=model_cfg["max_seq_len"],
        dim_feedforward=model_cfg.get("dim_feedforward"),
        pooling=model_cfg.get("pooling", "mean"),
        pooling_multihead=bool(model_cfg.get("pooling_multihead", False)),
        multiscale=model_cfg.get("multiscale", False),
        sig_gating=model_cfg.get("sig_gating"),
        d_phi=model_cfg.get("d_phi"),
        set_state_dim=model_cfg.get("set_state_dim"),
        geometry=model_cfg.get("geometry"),
        features=model_cfg.get("features"),
        router_type=model_cfg["router_type"],
        router_topk=model_cfg["router_topk"],
        router_multihead=bool(model_cfg.get("router_multihead", False)),
        router_temperature=float(model_cfg.get("router_temperature", 1.0)),
        router_min_temp=float(model_cfg.get("router", {}).get("min_temp", 0.5)),
        router_score_mode=str(
            model_cfg.get("router", {}).get("score_mode", "candidate_gather")
        ),
        backend=model_cfg["backend"],
        backend_params=model_cfg.get("backend_params"),
        feature_mode=model_cfg.get("feature_mode", "geometry_only"),
        feature_params=model_cfg.get("feature_params"),
        token_mlp=model_cfg.get("token_mlp"),
        adapter_type=model_cfg.get("adapter_type", "auto"),
        adapter_hidden_multiplier=model_cfg.get("adapter_hidden_multiplier", 2),
        adapter_budget_fraction=model_cfg.get("adapter_budget_fraction", 0.15),
        gamma=model_cfg.get("gamma", 1.0),
        beta=model_cfg.get("beta", 0.0),
        allow_token_token=bool(model_cfg.get("allow_token_token", False)),
        set_causality_mode=model_cfg.get("set_causality_mode"),
        output_residual_mode=model_cfg.get("output_residual_mode", "direct"),
        anchor=model_cfg.get("anchor"),
        set_diversity=model_cfg.get("set_diversity"),
        multivector_basis=model_cfg.get("multivector_basis"),
        candidate_fiber=model_cfg.get("candidate_fiber", "endpoint_window"),
        multiresolution=model_cfg.get("multiresolution"),
    )


def apply_training_seed(cfg: dict) -> None:
    training_cfg = cfg.setdefault("training", {})
    if "seed" not in training_cfg:
        raise ValueError("training.seed is required and must be applied before construction")

    requested_seed = int(training_cfg["seed"])
    deterministic = bool(training_cfg.get("deterministic", False))
    strict_deterministic = bool(
        training_cfg.get("strict_deterministic", False)
    )
    benchmark_mode = bool(training_cfg.get("benchmark_mode", False))
    if deterministic and benchmark_mode:
        raise ValueError(
            "training.deterministic=true is incompatible with "
            "training.benchmark_mode=true"
        )

    set_seed(
        requested_seed,
        deterministic=deterministic,
        benchmark_mode=benchmark_mode,
        strict_deterministic=strict_deterministic,
    )
    training_cfg["seed"] = requested_seed
    training_cfg["seed_applied"] = True
    training_cfg["applied_seed"] = requested_seed
    training_cfg["torch_initial_seed"] = int(torch.initial_seed())
    training_cfg["deterministic"] = deterministic
    training_cfg["strict_deterministic"] = strict_deterministic
    training_cfg["benchmark_mode"] = benchmark_mode
    training_cfg["cublas_workspace_config"] = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG",
        "NA",
    )
    data_cfg = cfg.setdefault("data", {})
    data_cfg.setdefault("dataset_seed", requested_seed + 10_000)
    data_cfg.setdefault("train_loader_seed", requested_seed + 20_000)
    data_cfg.setdefault("validation_loader_seed", requested_seed + 30_000)

    resolved = cfg.setdefault("resolved", {})
    resolved.update(
        {
            "requested_seed": requested_seed,
            "applied_seed": requested_seed,
            "torch_initial_seed": int(torch.initial_seed()),
            "deterministic": deterministic,
            "strict_deterministic": strict_deterministic,
            "benchmark_mode": benchmark_mode,
            "cublas_workspace_config": training_cfg[
                "cublas_workspace_config"
            ],
            "dataset_seed": int(data_cfg["dataset_seed"]),
            "train_loader_seed": int(data_cfg["train_loader_seed"]),
            "validation_loader_seed": int(
                data_cfg["validation_loader_seed"]
            ),
        }
    )


def attach_resolved_metadata(cfg: dict, model: torch.nn.Module) -> None:
    runtime_resolved = dict(cfg.get("resolved", {}))
    if hasattr(model, "get_resolved_metadata"):
        resolved = dict(model.get_resolved_metadata())
    else:
        resolved = {}
    resolved_defaults = {
        "d_phi": "NA",
        "set_state_dim": "NA",
        "adapter_type": "NA",
        "router_min_temp": "NA",
        "router_score_mode": "NA",
        "pooling_alpha": "NA",
        "hash_seed": "NA",
        "hash_normalize": "NA",
        "hash_num_bins": "NA",
        "landmark_coverage": "NA",
        "landmark_count": "NA",
        "output_residual_mode": "NA",
        "anchor_enabled": "NA",
        "anchor_target": "NA",
        "anchor_pre_encoder_layers": "NA",
        "anchor_lambda_h": "NA",
        "anchor_lambda_pre": "NA",
        "anchor_pre_encoder_head": "NA",
        "anchor_detach_target": "NA",
        "anchor_norm": "NA",
        "anchor_teacher_enabled": "NA",
        "set_diversity_lambda_div": "NA",
        "multivector_basis_enabled": "NA",
        "multivector_basis_r": "NA",
        "candidate_fiber": "NA",
        "multiresolution_enabled": "NA",
        "multiresolution_groups": "NA",
        "multiresolution_num_groups": "NA",
    }
    resolved_defaults.update(runtime_resolved)
    resolved_defaults.update(resolved)
    cfg["resolved"] = resolved_defaults


def _make_loader(
    ds,
    batch_size: int,
    shuffle: bool,
    *,
    seed: int = 0,
    num_workers: int = 0,
):
    from torch.utils.data import IterableDataset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    worker_init_fn = make_worker_init_fn(int(seed))
    if isinstance(ds, IterableDataset):
        if int(num_workers) != 0:
            raise ValueError(
                "ordered iterable datasets require data.num_workers=0 "
                "until worker sharding is provenance-aware"
            )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            generator=generator,
            worker_init_fn=worker_init_fn,
            num_workers=0,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=worker_init_fn,
        num_workers=max(0, int(num_workers)),
        persistent_workers=False,
    )


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
            limit=data_cfg.get("val_limit", data_cfg.get("limit")),
            cache_root=data_cfg.get("cache_root"),
            vocab=(train_ds.stoi, train_ds.itos),
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
            limit=data_cfg.get("val_limit", data_cfg.get("limit")),
            cache_root=data_cfg.get("cache_root"),
            vocab=(train_ds.stoi, train_ds.itos),
        )
    train_loader = _make_loader(
        train_ds,
        data_cfg["batch_size"],
        shuffle=True,
        seed=int(data_cfg.get("train_loader_seed", 0)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    val_loader = _make_loader(
        val_ds,
        data_cfg["batch_size"],
        shuffle=False,
        seed=int(data_cfg.get("validation_loader_seed", 1)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    return train_loader, val_loader, train_ds.vocab_size


def build_unigram_counts(dataset, vocab_size: int) -> torch.Tensor:
    counts = torch.zeros(int(vocab_size), dtype=torch.long)

    def add(labels: torch.Tensor) -> None:
        valid = labels.reshape(-1)
        valid = valid[valid >= 0]
        if valid.numel() > 0:
            counts.add_(torch.bincount(valid, minlength=vocab_size))

    samples = getattr(dataset, "samples", None)
    if samples is not None:
        for _, labels in samples:
            add(labels)
        return counts
    for _, labels in dataset:
        add(labels)
    return counts


def maybe_evaluate_span_ablation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    val_metrics: dict,
) -> dict:
    if not hasattr(model, "set_span_ablation"):
        return {}
    if getattr(model, "output_residual_mode", None) != "anchor_span":
        return {}
    previous = bool(getattr(model, "span_ablation_enabled", False))
    try:
        model.set_span_ablation(True)
        ablated = evaluate(model, val_loader, device)
    finally:
        model.set_span_ablation(previous)
    ablated_loss = float(ablated["loss"])
    base_loss = float(val_metrics["loss"])
    ablated_ppl = perplexity(ablated_loss)
    base_ppl = perplexity(base_loss)
    return {
        "span_ablation_loss": ablated_loss,
        "span_ablation_ppl": ablated_ppl,
        "span_ablation_delta_loss": ablated_loss - base_loss,
        "span_ablation_delta_ppl": ablated_ppl - base_ppl,
    }


def maybe_evaluate_group_span_ablation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    val_metrics: dict,
) -> dict:
    if not hasattr(model, "set_span_ablation_mode"):
        return {}
    if not bool(getattr(model, "multiresolution_enabled", False)):
        return {}
    if getattr(model, "output_residual_mode", None) != "anchor_span":
        return {}
    groups = [str(m["name"]) for m in getattr(model, "multiresolution_group_metadata", [])]
    wanted = [name for name in ("fine", "coarse") if name in groups]
    if not wanted:
        return {}
    previous = str(getattr(model, "span_ablation_mode", "none"))
    base_loss = float(val_metrics["loss"])
    base_ppl = perplexity(base_loss)
    metrics = {}
    try:
        for group in wanted:
            model.set_span_ablation_mode(group)
            ablated = evaluate(model, val_loader, device)
            ablated_loss = float(ablated["loss"])
            ablated_ppl = perplexity(ablated_loss)
            metrics[f"span_ablation_{group}_loss"] = ablated_loss
            metrics[f"span_ablation_{group}_ppl"] = ablated_ppl
            metrics[f"span_ablation_{group}_delta_loss"] = ablated_loss - base_loss
            metrics[f"span_ablation_{group}_delta_ppl"] = ablated_ppl - base_ppl
    finally:
        model.set_span_ablation_mode(previous)
    return metrics


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
    train_loader = _make_loader(
        train_ds,
        data_cfg["batch_size"],
        shuffle=True,
        seed=int(data_cfg.get("train_loader_seed", 0)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    val_loader = _make_loader(
        val_ds,
        data_cfg["batch_size"],
        shuffle=False,
        seed=int(data_cfg.get("validation_loader_seed", 1)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    vocab = {
        "vocab_size": train_ds.vocab_size,
        "pad_id": train_ds.pad_id,
        "bos_id": train_ds.bos_id,
        "eos_id": train_ds.eos_id,
        "decode": train_ds.decode,
        "max_len": data_cfg.get("seq_len", data_cfg.get("max_len", 64)),
    }
    return train_loader, val_loader, vocab


def attach_dataset_provenance(
    cfg: dict,
    provenance: dict[str, object],
) -> None:
    data_cfg = cfg.setdefault("data", {})
    data_cfg["dataset_digest"] = provenance["dataset_digest"]
    data_cfg["tokenizer_name"] = provenance["tokenizer"]
    data_cfg["tokenizer_digest"] = provenance["tokenizer_digest"]
    train = provenance["train"]
    validation = provenance["validation"]
    if isinstance(train, dict):
        data_cfg["train_token_count"] = train["token_count"]
        data_cfg["train_record_offsets_digest"] = train[
            "record_offsets_digest"
        ]
        data_cfg["train_sample_offsets_digest"] = train[
            "sample_offsets_digest"
        ]
    if isinstance(validation, dict):
        data_cfg["validation_token_count"] = validation["token_count"]
        data_cfg["validation_record_offsets_digest"] = validation[
            "record_offsets_digest"
        ]
        data_cfg["validation_sample_offsets_digest"] = validation[
            "sample_offsets_digest"
        ]
    cfg.setdefault("resolved", {}).update(
        {
            "dataset_digest": provenance["dataset_digest"],
            "tokenizer_digest": provenance["tokenizer_digest"],
        }
    )


def checkpoint_path(cfg: dict, name: str) -> Path:
    checkpoint_cfg = cfg["training"]["checkpoint"]
    directory = checkpoint_cfg.get("directory")
    if directory is None:
        directory = Path(cfg["training"].get("output_dir", "out")) / "checkpoints"
    return Path(directory) / name


def append_checkpoint_manifest(
    cfg: dict,
    *,
    path: Path,
    digest: str,
    epoch: int,
    global_step: int,
) -> None:
    manifest = checkpoint_path(cfg, "manifest.jsonl")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(path),
        "sha256": digest,
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_training_checkpoint(
    *,
    cfg: dict,
    logger: ExperimentLogger,
    model: torch.nn.Module,
    dataset_provenance: dict[str, object],
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str,
) -> tuple[Path, str]:
    path = checkpoint_path(cfg, name)
    payload = build_checkpoint_payload(
        model=model,
        config=cfg,
        config_fingerprint=logger.config_fingerprint,
        dataset_provenance=dataset_provenance,
        epoch=epoch,
        global_step=global_step,
        optimizer=optimizer,
        loaders={"train": train_loader, "validation": val_loader},
    )
    digest = save_checkpoint(payload, path)
    append_checkpoint_manifest(
        cfg,
        path=path,
        digest=digest,
        epoch=epoch,
        global_step=global_step,
    )
    print(f"checkpoint={path} sha256={digest}")
    return path, digest


def evaluate_lm_bundle(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    unigram_counts: torch.Tensor | None,
) -> dict:
    val_metrics = evaluate(
        model,
        val_loader,
        device,
        unigram_counts=unigram_counts,
    )
    val_metrics.update(
        maybe_evaluate_span_ablation(
            model,
            val_loader,
            device,
            val_metrics,
        )
    )
    val_metrics.update(
        maybe_evaluate_group_span_ablation(
            model,
            val_loader,
            device,
            val_metrics,
        )
    )
    return val_metrics


def main() -> None:
    args = parse_args()
    overrides = []
    for group in args.override:
        if isinstance(group, list):
            overrides.extend(group)
        else:
            overrides.append(group)
    cfg = load_config(args.config, overrides=overrides)
    validate_experiment_contract(cfg)
    apply_training_seed(cfg)
    if args.dry_run:
        print("Dry run: config validated. No data loaded or training run.")
        print(cfg)
        return
    device = torch.device(args.device)

    task = detect_task(cfg)
    cfg.setdefault("resolved", {})["source_commit"] = source_commit() or "NA"
    unigram_counts = None
    dataset_provenance: dict[str, object] | None = None
    if task == "seq2seq":
        shared_vocab = bool(cfg.get("model", {}).get("seq2seq", {}).get("shared_vocab", True))
        train_loader, val_loader, vocab = build_seq2seq_dataloaders(cfg["data"], shared_vocab)
        if cfg["model"].get("vocab_size", 0) in (0, None):
            cfg["model"]["vocab_size"] = vocab["vocab_size"]
        impl = cfg["model"]["implementation"]
        num_heads = cfg["model"].get("num_heads") or cfg["model"].get("nhead", 8)
        num_layers = cfg["model"].get("num_layers", 4)
        d_model = cfg["model"].get("d_model", 512)
        dim_ff = cfg["model"].get("dim_feedforward", d_model * 4)
        dropout = cfg["model"].get("dropout", 0.1)
        attn_dropout = cfg["model"].get("attn_dropout")
        resid_dropout = cfg["model"].get("resid_dropout")
        ffn_dropout = cfg["model"].get("ffn_dropout")
        max_len = cfg["data"].get("seq_len", cfg["data"].get("max_len", 64))
        encoder_family = "baseline_token"
        decoder_family = "baseline_token"
        cross_attention = cfg["model"].get("cross_attention", "baseline")
        if impl == "set_only":
            encoder_family = "set_only"
            decoder_family = "set_only"
            cross_attention = "set_only"
        elif impl in {"encoder_set_only", "encoder_set_decoder_baseline"}:
            encoder_family = "set_only"
        elif impl in {"decoder_set_only", "encoder_baseline_decoder_set"}:
            decoder_family = "set_only"
        elif impl == "cross_attention_set_only":
            cross_attention = "set_only"
        if cfg["model"].get("cross_attention") is not None:
            cross_attention = cfg["model"]["cross_attention"]

        set_only_cfg = cfg["model"] if encoder_family == "set_only" or decoder_family == "set_only" or cross_attention == "set_only" else None
        model = Seq2SeqTransformer(
            vocab_size=cfg["model"]["vocab_size"],
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            ffn_dropout=ffn_dropout,
            max_len=max_len,
            encoder_family=encoder_family,
            decoder_family=decoder_family,
            cross_attention=cross_attention,
            set_only_cfg=set_only_cfg,
            shared_embeddings=None,
            pad_id=vocab["pad_id"],
            bos_id=vocab["bos_id"],
            eos_id=vocab["eos_id"],
            encoder_attention_family=cfg["model"].get("encoder_attention_family", cfg["model"].get("attention_family", "dense")),
            encoder_backend=cfg["model"].get("encoder_backend", cfg["model"].get("backend", "exact")),
            decoder_attention_family=cfg["model"].get("decoder_attention_family", cfg["model"].get("attention_family", "dense")),
            decoder_backend=cfg["model"].get("decoder_backend", cfg["model"].get("backend", "exact")),
            cross_attention_family=cfg["model"].get("cross_attention_family", cfg["model"].get("attention_family", "dense")),
            cross_backend=cfg["model"].get("cross_backend", cfg["model"].get("backend", "exact")),
            encoder_backend_params=cfg["model"].get("encoder_backend_params", cfg["model"].get("backend_params")),
            decoder_backend_params=cfg["model"].get("decoder_backend_params", cfg["model"].get("backend_params")),
            cross_backend_params=cfg["model"].get("cross_backend_params", cfg["model"].get("backend_params")),
        ).to(device)
        attach_resolved_metadata(cfg, model)
    else:
        train_loader, val_loader, vocab_size = build_dataloaders(cfg["data"])
        dataset_provenance = dataset_provenance_bundle(
            train_loader.dataset,
            val_loader.dataset,
        )
        attach_dataset_provenance(cfg, dataset_provenance)
        if cfg["model"].get("vocab_size", 0) in (0, None):
            cfg["model"]["vocab_size"] = vocab_size
        model = build_model(cfg["model"]).to(device)
        attach_resolved_metadata(cfg, model)
        unigram_counts = build_unigram_counts(train_loader.dataset, vocab_size)

    checkpoint_cfg = cfg["training"]["checkpoint"]
    checkpoint_requested = bool(
        checkpoint_cfg.get("save_final")
        or checkpoint_cfg.get("save_every_epochs")
        or checkpoint_cfg.get("resume_from")
        or checkpoint_cfg.get("eval_only_from")
    )
    if checkpoint_requested and dataset_provenance is None:
        raise ValueError(
            "checkpointed runs require ordered dataset provenance; "
            "the current implementation supports causal LM datasets"
        )

    eval_only_from = checkpoint_cfg.get("eval_only_from")
    resume_from = checkpoint_cfg.get("resume_from")
    if eval_only_from:
        cfg["training"]["eval_only"] = True
        cfg["training"]["checkpoint"]["loaded_sha256"] = sha256_file(
            eval_only_from
        )
        payload = load_checkpoint(
            eval_only_from,
            model=model,
            map_location=device,
            expected_model_config=cfg["model"],
            expected_dataset_digest=str(
                dataset_provenance["dataset_digest"]
            ),
            expected_tokenizer_digest=str(
                dataset_provenance["tokenizer_digest"]
            ),
        )
        cfg["training"]["checkpoint"]["loaded_epoch"] = int(payload["epoch"])
        cfg["training"]["checkpoint"]["loaded_global_step"] = int(
            payload["global_step"]
        )
    elif resume_from:
        cfg["training"]["checkpoint"]["loaded_sha256"] = sha256_file(
            resume_from
        )

    wandb_tags = [t for t in args.wandb_tags.split(",") if t]
    logger = ExperimentLogger(
        config=cfg,
        csv_path=args.csv_path,
        wandb_project=args.wandb_project,
        wandb_tags=wandb_tags or None,
        wandb_enable=True if args.wandb else None,
    )
    logger.log_model_complexity(model)

    if eval_only_from:
        try:
            logger.start_epoch(num_train_samples=0)
            if task == "seq2seq":
                raise ValueError(
                    "eval-only checkpointing is not implemented for seq2seq"
                )
            val_metrics = evaluate_lm_bundle(
                model,
                val_loader,
                device,
                unigram_counts,
            )
            logger.log_epoch(
                int(payload["epoch"]),
                {},
                val_metrics,
                None,
            )
            print(
                f"eval_only checkpoint={eval_only_from} "
                f"val_loss={val_metrics['loss']:.4f}"
            )
        finally:
            logger.finish()
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    set_diversity_weight = float(
        cfg["model"].get("set_diversity", {}).get(
            "lambda_div",
            cfg["training"].get("set_diversity_weight", 0.0),
        )
    )
    set_diversity_mode = str(cfg["training"].get("set_diversity_mode", "position_contrastive"))

    epochs = int(cfg["training"]["epochs"])
    start_epoch = 1
    global_step = 0
    if resume_from:
        payload = load_checkpoint(
            resume_from,
            model=model,
            map_location=device,
            expected_model_config=cfg["model"],
            expected_dataset_digest=str(
                dataset_provenance["dataset_digest"]
            ),
            expected_tokenizer_digest=str(
                dataset_provenance["tokenizer_digest"]
            ),
            optimizer=optimizer,
            loaders={"train": train_loader, "validation": val_loader},
            restore_training_state=True,
        )
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        if start_epoch > epochs:
            raise ValueError(
                f"resume checkpoint epoch {payload['epoch']} already reaches "
                f"training.epochs={epochs}"
            )

    try:
        for epoch in range(start_epoch, epochs + 1):
            num_samples = None
            try:
                num_samples = len(train_loader.dataset)
            except Exception:
                num_samples = cfg.get("data", {}).get("limit")
            logger.start_epoch(num_train_samples=num_samples or 0)
            if task == "seq2seq":
                train_metrics = train_one_epoch_seq2seq(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    pad_id=vocab["pad_id"],
                    set_diversity_weight=set_diversity_weight,
                    set_diversity_mode=set_diversity_mode,
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
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    set_diversity_weight=set_diversity_weight,
                    set_diversity_mode=set_diversity_mode,
                )
                global_step += int(
                    train_metrics.pop("_optimizer_steps", 0)
                )
                val_metrics = evaluate_lm_bundle(
                    model,
                    val_loader,
                    device,
                    unigram_counts,
                )
            set_diagnostics = model.get_diagnostics() if hasattr(model, "get_diagnostics") else None
            logger.log_epoch(epoch, train_metrics, val_metrics, set_diagnostics)
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f}"
            )
            save_every = int(
                checkpoint_cfg.get("save_every_epochs", 0)
            )
            if save_every > 0 and epoch % save_every == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    logger=logger,
                    model=model,
                    dataset_provenance=dataset_provenance,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    name=f"epoch_{epoch:04d}.pt",
                )
        if checkpoint_cfg.get("save_final"):
            save_training_checkpoint(
                cfg=cfg,
                logger=logger,
                model=model,
                dataset_provenance=dataset_provenance,
                epoch=epochs,
                global_step=global_step,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                name="final.pt",
            )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
