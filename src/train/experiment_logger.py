from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from train.metrics_impl import perplexity
from train.metrics_schema import (
    ATTENTION_TAGS,
    EFFICIENCY_METRICS,
    SET_DIAGNOSTICS,
    TASK_METRICS,
    UNIVERSAL_METRICS,
    canonical_dataset_name,
    detect_task,
)


def _flatten_cfg(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in cfg.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_cfg(value, path))
        else:
            flat[path] = value
    return flat


def _get_cfg_value(cfg: Dict[str, Any], key: str, default: Any = "NA") -> Any:
    parts = key.split(".")
    cur: Any = cfg
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _attention_tags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model = cfg.get("model", {})
    family = model.get("family")
    set_enabled = family == "set_only" or bool(model.get("set_enabled"))
    backend = model.get("backend", "baseline")

    if family == "baseline_token":
        attn_family = "dense"
        base = "dot_product"
    else:
        if backend in {"dense_exact"}:
            attn_family = "set_dense"
        elif backend in {"local_band", "sparse_topk"}:
            attn_family = "set_sparse"
        else:
            attn_family = "set_linear"
        base = backend

    return {
        "attention/family": attn_family,
        "attention/base_mechanism": base,
        "attention/set_enabled": bool(set_enabled),
    }


def _default_run_name(cfg: Dict[str, Any], task: str, dataset: str, attn_tags: Dict[str, Any]) -> str:
    stage = cfg.get("stage", _get_cfg_value(cfg, "training.stage", "stage"))
    model_type = cfg.get("model", {}).get("family", "model")
    backend = cfg.get("model", {}).get("backend", "baseline")
    score_mode = cfg.get("model", {}).get("ska_score_mode", "na")
    seq_len = _get_cfg_value(cfg, "data.seq_len", "na")
    batch = _get_cfg_value(cfg, "data.batch_size", "na")
    seed = _get_cfg_value(cfg, "training.seed", "na")
    attn_family = attn_tags["attention/family"]
    return f"{stage}-{task}-{dataset}-{model_type}-{attn_family}-{backend}-{score_mode}-L{seq_len}-B{batch}-S{seed}"


class ExperimentLogger:
    def __init__(
        self,
        config: Dict[str, Any],
        csv_path: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_enable: Optional[bool] = None,
    ) -> None:
        self.cfg = config
        self.task = detect_task(config)
        data_cfg = config.get("data", {})
        dataset = data_cfg.get("dataset", "unknown")
        if self.task == "lm":
            dataset = data_cfg.get("lm_dataset", dataset)
        elif self.task == "seq2seq":
            dataset = data_cfg.get("seq_dataset", dataset)
        elif self.task == "textdiff":
            dataset = data_cfg.get("textdiff_dataset", dataset)
        elif self.task == "vit":
            dataset = data_cfg.get("vit_dataset", dataset)
        self.dataset = canonical_dataset_name(str(dataset))
        self.attn_tags = _attention_tags(config)
        self.is_set_enabled = bool(self.attn_tags["attention/set_enabled"])

        run_name_cfg = _get_cfg_value(config, "logging.wandb.run_name", None)
        self.run_name = run_name_cfg or _default_run_name(
            config, self.task, self.dataset, self.attn_tags
        )

        self.csv_path = self._resolve_csv_path(csv_path)
        self.csv_file = None
        self.csv_writer = None

        self.model_metrics: Dict[str, Any] = {}
        self._epoch_start = None
        self._epoch_samples = None

        self.wandb = None
        self._init_wandb(wandb_project, wandb_tags, wandb_enable)
        self._init_csv()

    def _resolve_csv_path(self, cli_path: Optional[str]) -> Path:
        if cli_path:
            return Path(cli_path)
        cfg_path = _get_cfg_value(self.cfg, "logging.csv.path", None)
        if cfg_path:
            return Path(cfg_path)
        out_dir = _get_cfg_value(self.cfg, "training.output_dir", "out")
        return Path(out_dir) / "metrics" / f"{self.run_name}.csv"

    def _init_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._columns())
        self.csv_writer.writeheader()

    def _init_wandb(
        self,
        wandb_project: Optional[str],
        wandb_tags: Optional[List[str]],
        wandb_enable: Optional[bool],
    ) -> None:
        wandb_cfg = self.cfg.get("logging", {}).get("wandb", {}) if isinstance(self.cfg.get("logging"), dict) else {}
        enable = wandb_enable if wandb_enable is not None else bool(wandb_cfg.get("enable", False))
        if not enable:
            return
        try:
            import wandb  # type: ignore
        except Exception as exc:
            raise ImportError("wandb is required for W&B logging.") from exc

        project = (
            wandb_project
            or wandb_cfg.get("project")
            or os.environ.get("WANDB_PROJECT")
            or "set-attention"
        )
        tags = wandb_tags or wandb_cfg.get("tags") or []
        auto_tags = [
            f"task:{self.task}",
            f"dataset:{self.dataset}",
            f"family:{self.attn_tags['attention/family']}",
            f"backend:{self.cfg.get('model', {}).get('backend', 'baseline')}",
            f"score:{self.cfg.get('model', {}).get('ska_score_mode', 'na')}",
            f"precision:{_get_cfg_value(self.cfg, 'training.precision', 'na')}",
            f"stage:{self.cfg.get('stage', _get_cfg_value(self.cfg, 'training.stage', 'na'))}",
        ]
        tags = list(dict.fromkeys([*auto_tags, *tags]))
        flat_cfg = _flatten_cfg(self.cfg)
        flat_cfg.update(self.attn_tags)

        self.wandb = wandb.init(
            project=project,
            config=flat_cfg,
            tags=tags,
            reinit=True,
        )
        if self.wandb:
            self.wandb.name = self.run_name

    def _columns(self) -> List[str]:
        cfg_fields = [
            "training.seed",
            "training.epochs",
            "training.lr",
            "training.weight_decay",
            "training.warmup_steps",
            "training.precision",
            "training.grad_accum_steps",
            "training.clip_grad_norm",
            "data.dataset",
            "data.lm_dataset",
            "data.seq_dataset",
            "data.textdiff_dataset",
            "data.vit_dataset",
            "data.batch_size",
            "data.seq_len",
            "model.family",
            "model.backend",
            "model.feature_mode",
            "model.router_type",
            "model.router_topk",
            "model.d_model",
            "model.num_layers",
            "model.num_heads",
            "model.window_size",
            "model.stride",
            "model.vocab_size",
            "model.ska_score_mode",
            "model.minhash_k",
        ]
        columns = [
            "task",
            "dataset",
            "epoch",
        ]
        columns += cfg_fields
        columns += ATTENTION_TAGS
        columns += TASK_METRICS.get(self.task, [])
        columns += UNIVERSAL_METRICS
        columns += EFFICIENCY_METRICS
        columns += SET_DIAGNOSTICS
        return columns

    def log_model_complexity(self, model: torch.nn.Module) -> None:
        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        self.model_metrics["model/param_count"] = param_count
        self.model_metrics["model/memory_footprint_mb"] = param_bytes / (1024 ** 2)

    def start_epoch(self, num_train_samples: int) -> None:
        self._epoch_start = time.time()
        self._epoch_samples = num_train_samples
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        set_diagnostics: Optional[Dict[str, Any]],
    ) -> None:
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else None
        )
        samples_per_second = (
            self._epoch_samples / elapsed if self._epoch_samples and elapsed > 0 else None
        )

        metrics = {}
        for prefix, bundle in (("train", train_metrics), ("val", val_metrics)):
            for key, value in bundle.items():
                metrics[f"{prefix}/{key}"] = value

        if self.task == "lm":
            if metrics.get("train/ppl") is None and metrics.get("train/loss") is not None:
                metrics["train/ppl"] = perplexity(metrics["train/loss"])
            if metrics.get("val/ppl") is None and metrics.get("val/loss") is not None:
                metrics["val/ppl"] = perplexity(metrics["val/loss"])

        efficiency = {
            "train/time_per_epoch_s": elapsed,
            "train/peak_vram_mib": peak_vram,
            "efficiency/samples_per_second": samples_per_second,
        }

        if self.task == "lm" and metrics.get("val/ppl") is not None and elapsed > 0:
            efficiency["efficiency/ppl_per_second"] = metrics["val/ppl"] / elapsed
        if self.task == "seq2seq" and metrics.get("val/bleu") is not None and elapsed > 0:
            efficiency["efficiency/bleu_per_second"] = metrics["val/bleu"] / elapsed
        if self.task == "vit" and metrics.get("val/acc") is not None and elapsed > 0:
            efficiency["efficiency/acc_per_second"] = metrics["val/acc"] / elapsed

        row = {
            "task": self.task,
            "dataset": self.dataset,
            "epoch": epoch,
        }
        for col in self._columns():
            if col in row:
                continue
            if col in self.attn_tags:
                row[col] = self.attn_tags[col]
                continue
            if col in self.model_metrics:
                row[col] = self.model_metrics[col]
                continue
            if col in efficiency:
                row[col] = efficiency[col]
                continue
            if col in metrics:
                row[col] = metrics[col]
                continue
            if set_diagnostics and col in set_diagnostics:
                row[col] = set_diagnostics[col]
                continue
            row[col] = _get_cfg_value(self.cfg, col, "NA")

        csv_row = {k: ("NA" if v is None else v) for k, v in row.items()}
        self.csv_writer.writerow(csv_row)
        self.csv_file.flush()

        if self.wandb:
            wandb_row = {k: (None if v == "NA" else v) for k, v in csv_row.items()}
            self.wandb.log(wandb_row, step=epoch)

    def finish(self) -> None:
        if self.csv_file:
            self.csv_file.close()
        if self.wandb:
            self.wandb.finish()
