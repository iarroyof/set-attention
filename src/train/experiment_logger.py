from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from train.metrics_impl import perplexity
from train.metrics_schema import (
    ATTENTION_TAGS,
    BASELINE_DIAGNOSTICS,
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


def _config_fingerprint(flat_cfg: Dict[str, Any]) -> str:
    # Fingerprint should reflect experiment-defining hyperparameters and remain
    # stable across reruns. Exclude transient/private logging and warning fields.
    excluded = {
        "logging.wandb.run_name",
        "logging.wandb.tags",
        "logging.wandb.enable",
        "logging.wandb.project",
        "logging.csv.path",
    }
    stable_cfg = {
        k: v
        for k, v in flat_cfg.items()
        if not k.startswith("_") and k not in excluded
    }
    payload = json.dumps(stable_cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


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
    impl = model.get("implementation", "baseline_token")
    set_enabled = impl != "baseline_token" or bool(model.get("set_enabled"))
    backend = model.get("backend", "exact")
    family = model.get("attention_family", "dense")

    if impl == "baseline_token":
        attn_family = family
    else:
        attn_family = f"set_{family}"
    base = backend

    return {
        "attention/family": attn_family,
        "attention/base_mechanism": base,
        "attention/set_enabled": bool(set_enabled),
    }


def _default_run_name(cfg: Dict[str, Any], task: str, dataset: str, attn_tags: Dict[str, Any]) -> str:
    stage = cfg.get("stage", _get_cfg_value(cfg, "training.stage", "stage"))
    model_type = cfg.get("model", {}).get("implementation", "model")
    backend = cfg.get("model", {}).get("backend", "exact")
    score_mode = cfg.get("model", {}).get("ska_score_mode", "na")
    feature_mode = cfg.get("model", {}).get("feature_mode", "na")
    router_type = cfg.get("model", {}).get("router_type", "na")
    pooling_mode = _get_cfg_value(cfg, "model.pooling.mode", _get_cfg_value(cfg, "model.pooling", "na"))
    seq_len = _get_cfg_value(cfg, "data.seq_len", "na")
    batch = _get_cfg_value(cfg, "data.batch_size", "na")
    seed = _get_cfg_value(cfg, "training.seed", "na")
    attn_family = attn_tags["attention/family"]
    return (
        f"{stage}-{task}-{dataset}-{model_type}-{attn_family}-{backend}-{score_mode}-"
        f"{feature_mode}-{router_type}-{pooling_mode}-L{seq_len}-B{batch}-S{seed}"
    )


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
        self.flat_cfg = _flatten_cfg(self.cfg)
        self.flat_cfg.update(self.attn_tags)
        self.flat_cfg.setdefault("task", self.task)
        self.flat_cfg.setdefault("dataset", self.dataset)
        self.config_fingerprint = _config_fingerprint(self.flat_cfg)

        run_name_cfg = _get_cfg_value(config, "logging.wandb.run_name", None)
        self.run_name = run_name_cfg or _default_run_name(
            config, self.task, self.dataset, self.attn_tags
        )

        self.csv_file = None
        self.csv_writer = None

        self.model_metrics: Dict[str, Any] = {}
        self._epoch_start = None
        self._epoch_samples = None

        self.wandb = None
        self.run_id = None
        self._init_wandb(wandb_project, wandb_tags, wandb_enable)
        if self.run_id is None:
            self.run_id = uuid.uuid4().hex[:8]
        self.csv_path = self._resolve_csv_path(csv_path)
        self._init_csv()
        self._dump_config_json()

    def _resolve_csv_path(self, cli_path: Optional[str]) -> Path:
        if cli_path:
            return Path(cli_path)
        cfg_path = _get_cfg_value(self.cfg, "logging.csv.path", None)
        if cfg_path:
            return Path(cfg_path)
        out_dir = _get_cfg_value(self.cfg, "training.output_dir", "out")
        base = (
            Path(out_dir)
            / "metrics"
            / f"{self.run_name}-id{self.run_id}-fp{self.config_fingerprint}.csv"
        )
        if base.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            return (
                Path(out_dir)
                / "metrics"
                / f"{self.run_name}-id{self.run_id}-fp{self.config_fingerprint}_{ts}.csv"
            )
        return base

    def _init_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._columns())
        self.csv_writer.writeheader()

    def _dump_config_json(self) -> None:
        cfg_path = self.csv_path.with_suffix(".json")
        payload = dict(self.flat_cfg)
        payload["run_name"] = self.run_name
        payload["run_id"] = self.run_id
        payload["config_fingerprint"] = self.config_fingerprint
        payload["csv_path"] = str(self.csv_path)
        cfg_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _init_wandb(
        self,
        wandb_project: Optional[str],
        wandb_tags: Optional[List[str]],
        wandb_enable: Optional[bool],
    ) -> None:
        wandb_cfg = (
            self.cfg.get("logging", {}).get("wandb", {})
            if isinstance(self.cfg.get("logging"), dict)
            else {}
        )
        sweep_env = any(
            os.environ.get(key)
            for key in ("WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_AGENT_ID")
        )
        explicit_enable = (
            wandb_enable
            if wandb_enable is not None
            else wandb_cfg.get("enable")
        )
        if explicit_enable is None:
            enable = bool(wandb_cfg) or sweep_env or bool(wandb_project) or bool(wandb_tags) or bool(os.environ.get("WANDB_PROJECT"))
        else:
            enable = bool(explicit_enable)
        if enable and explicit_enable is None:
            logging_cfg = self.cfg.setdefault("logging", {})
            wandb_cfg = logging_cfg.setdefault("wandb", {})
            wandb_cfg["enable"] = True
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
        flat_cfg = dict(self.flat_cfg)
        flat_cfg["config_fingerprint"] = self.config_fingerprint

        if sweep_env:
            self.wandb = wandb.init(
                project=project,
                tags=tags,
                reinit=True,
            )
            if self.wandb:
                for key, value in flat_cfg.items():
                    if key not in self.wandb.config:
                        self.wandb.config[key] = value
        else:
            self.wandb = wandb.init(
                project=project,
                config=flat_cfg,
                tags=tags,
                reinit=True,
            )
        if self.wandb:
            wandb.define_metric("epoch")
            for prefix in ("train/*", "val/*", "efficiency/*", "ausa/*", "model/*"):
                wandb.define_metric(prefix, step_metric="epoch")
            self.wandb.name = self.run_name
            self.run_id = self.wandb.id

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
            "training.output_dir",
            "data.dataset",
            "data.lm_dataset",
            "data.seq_dataset",
            "data.textdiff_dataset",
            "data.vit_dataset",
            "data.batch_size",
            "data.seq_len",
            "data.limit",
            "data.val_limit",
            "data.streaming",
            "data.cache_root",
            "model.implementation",
            "model.attention_family",
            "model.backend",
            "model.encoder_attention_family",
            "model.encoder_backend",
            "model.decoder_attention_family",
            "model.decoder_backend",
            "model.cross_attention_family",
            "model.cross_backend",
            "model.cross_attention",
            "model.feature_mode",
            "model.feature_params.minhash_k",
            "model.feature_params.num_bins",
            "model.feature_params.fusion",
            "model.feature_params.allow_unsafe",
            "model.features.hashed_counts.fusion",
            "model.router_type",
            "model.router_topk",
            "model.d_model",
            "model.dim_feedforward",
            "model.num_layers",
            "model.num_heads",
            "model.nhead",
            "model.dropout",
            "model.attn_dropout",
            "model.resid_dropout",
            "model.ffn_dropout",
            "model.max_seq_len",
            "model.window_size",
            "model.stride",
            "model.vocab_size",
            "model.ska_score_mode",
            "model.minhash_k",
            "model.pooling.mode",
            "model.pooling.tau",
            "model.pooling.q",
            "model.pooling.alpha",
            "model.pooling.learnable_alpha",
            "model.pooling.tiny_set_n",
            "model.pooling.isotropy_eps",
            "model.sig_gating.enabled",
            "model.sig_gating.method",
            "model.sig_gating.k",
            "model.sig_gating.sig_k",
            "model.sig_gating.delta_threshold",
            "model.sig_gating.symmetric",
            "model.sig_gating.include_self",
            "model.d_phi",
            "model.geometry.enabled",
            "model.geometry.apply_as_bias",
            "model.geometry.apply_in_phi_attn",
            "model.allow_token_token",
            "model.backend_params.radius",
            "model.backend_params.k_s",
            "model.backend_params.num_landmarks",
            "model.backend_params.k",
            "model.backend_params.global_indices",
            "model.backend_params.global_set_indices",
            "model.backend_params.bias_scale",
            "model.encoder_backend_params.radius",
            "model.encoder_backend_params.k_s",
            "model.encoder_backend_params.num_landmarks",
            "model.encoder_backend_params.k",
            "model.encoder_backend_params.global_indices",
            "model.encoder_backend_params.global_set_indices",
            "model.encoder_backend_params.bias_scale",
            "model.decoder_backend_params.radius",
            "model.decoder_backend_params.k_s",
            "model.decoder_backend_params.num_landmarks",
            "model.decoder_backend_params.k",
            "model.decoder_backend_params.global_indices",
            "model.decoder_backend_params.global_set_indices",
            "model.decoder_backend_params.bias_scale",
            "model.cross_backend_params.radius",
            "model.cross_backend_params.k_s",
            "model.cross_backend_params.num_landmarks",
            "model.cross_backend_params.k",
            "model.cross_backend_params.global_indices",
            "model.cross_backend_params.global_set_indices",
            "model.cross_backend_params.bias_scale",
        ]
        # Keep the legacy core fields first for readability, then append any
        # additional flattened config keys so new hyperparameters are always
        # logged without needing manual logger updates.
        core = set(cfg_fields)
        extra_cfg_fields = sorted(
            k
            for k in self.flat_cfg.keys()
            if k not in core
            and k not in ATTENTION_TAGS
            and not k.startswith("_")
            and (
                k.startswith("model.")
                or k.startswith("data.")
                or k.startswith("training.")
                or k.startswith("logging.")
                or k == "stage"
            )
        )
        cfg_fields = cfg_fields + extra_cfg_fields
        columns = [
            "task",
            "dataset",
            "run_name",
            "run_id",
            "config_fingerprint",
            "epoch",
        ]
        columns += cfg_fields
        columns += ATTENTION_TAGS
        columns += TASK_METRICS.get(self.task, [])
        columns += UNIVERSAL_METRICS
        columns += EFFICIENCY_METRICS
        columns += SET_DIAGNOSTICS
        columns += BASELINE_DIAGNOSTICS
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
            "run_name": self.run_name,
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
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
            wandb_row = {"epoch": epoch}
            wandb_row.update(self.model_metrics)
            wandb_row.update(metrics)
            wandb_row.update(efficiency)
            if set_diagnostics:
                wandb_row.update(set_diagnostics)
            wandb_row = {k: v for k, v in wandb_row.items() if v is not None}
            self.wandb.log(wandb_row, step=epoch)
            for key, value in wandb_row.items():
                self.wandb.summary[key] = value

    def finish(self) -> None:
        if self.csv_file:
            self.csv_file.close()
        if self.wandb:
            self.wandb.finish()
