from __future__ import annotations

from typing import Dict


TASK_METRICS = {
    "lm": ["train/loss", "val/loss", "train/ppl", "val/ppl"],
    "seq2seq": ["train/loss", "val/loss", "val/bleu", "val/rougeL"],
    "textdiff": ["train/loss", "val/loss"],
    "vit": ["train/loss", "val/loss", "val/acc", "val/top5"],
}

UNIVERSAL_METRICS = [
    "model/param_count",
    "model/memory_footprint_mb",
    "train/time_per_epoch_s",
    "train/peak_vram_mib",
    "efficiency/samples_per_second",
    "train/grad_norm",
]

EFFICIENCY_METRICS = [
    "efficiency/ppl_per_second",
    "efficiency/bleu_per_second",
    "efficiency/acc_per_second",
]

SET_DIAGNOSTICS = [
    "ausa/active_set_ratio",
    "ausa/active_set_size_mean",
    "ausa/active_set_size_std",
    "ausa/routing_entropy",
    "ausa/routing_entropy_norm",
    "ausa/routing_gini",
    "ausa/routing_top1_prob_mean",
    "ausa/set_reuse_jaccard",
    "ausa/tokens_per_set_variance",
    "ausa/router_confidence_mean",
    "ausa/router_confidence_std",
    "ausa/router_entropy",
    "ausa/router_top1_weight",
    "ausa/router_set_utilization_gini",
    "ausa/router_gradient_norm",
    "ausa/router_param_norm",
    "ausa/router_weight_change",
    "ausa/top1_vs_random_kl",
    "ausa/routing_consistency",
    "ausa/set_embedding_variance",
    "ausa/set_embedding_norm_mean",
    "ausa/set_cosine_similarity_mean",
    "ausa/set_rank_effective",
    "ausa/set_gram_top_eig_ratio",
    "ausa/set_gram_spectral_entropy_norm",
    "ausa/set_gram_condition_number",
    "ausa/set_gram_logdet",
    "ausa/set_gram_powerlaw_alpha",
    "ausa/set_attention_entropy_mean",
    "ausa/set_attention_top1_mean",
    "ausa/pooling_weight_entropy",
    "ausa/pooling_top1_weight",
    "ausa/pooling_effective_support",
    "ausa/pooling_neff_l2",
    "ausa/pooling_neff_ratio",
    "ausa/pooling_norm_ratio",
    "ausa/pooling_weight_gini",
    "ausa/pooling_alpha_value",
    "ausa/delta_routing_entropy",
    "ausa/delta_set_variance",
    "ausa/delta_router_confidence",
]

ATTENTION_TAGS = [
    "attention/family",
    "attention/base_mechanism",
    "attention/set_enabled",
]

BASELINE_DIAGNOSTICS = [
    "baseline/attention_entropy_mean",
    "baseline/attention_entropy_norm",
    "baseline/attention_top1_mean",
    "baseline/attention_top1_std",
    "baseline/attention_gradient_norm",
    "baseline/attention_param_norm",
    "baseline/attention_weight_change",
    "baseline/attention_pattern_jaccard",
    "baseline/delta_attention_entropy",
    "baseline/delta_attention_confidence",
    "baseline/encoder_attention_entropy_mean",
    "baseline/encoder_attention_entropy_norm",
    "baseline/encoder_attention_top1_mean",
    "baseline/encoder_attention_top1_std",
    "baseline/encoder_attention_pattern_jaccard",
    "baseline/encoder_attention_gradient_norm",
    "baseline/encoder_attention_param_norm",
    "baseline/encoder_attention_weight_change",
    "baseline/decoder_self_attention_entropy_mean",
    "baseline/decoder_self_attention_entropy_norm",
    "baseline/decoder_self_attention_top1_mean",
    "baseline/decoder_self_attention_top1_std",
    "baseline/decoder_self_attention_pattern_jaccard",
    "baseline/decoder_self_attention_gradient_norm",
    "baseline/decoder_self_attention_param_norm",
    "baseline/decoder_self_attention_weight_change",
    "baseline/decoder_cross_attention_entropy_mean",
    "baseline/decoder_cross_attention_entropy_norm",
    "baseline/decoder_cross_attention_top1_mean",
    "baseline/decoder_cross_attention_top1_std",
    "baseline/decoder_cross_attention_pattern_jaccard",
    "baseline/decoder_cross_attention_gradient_norm",
    "baseline/decoder_cross_attention_param_norm",
    "baseline/decoder_cross_attention_weight_change",
    "baseline/encoder_delta_attention_entropy",
    "baseline/encoder_delta_attention_confidence",
    "baseline/decoder_self_delta_attention_entropy",
    "baseline/decoder_self_delta_attention_confidence",
    "baseline/decoder_cross_delta_attention_entropy",
    "baseline/decoder_cross_delta_attention_confidence",
]

DATASET_TO_TASK = {
    "wikitext2": "lm",
    "wikitext103": "lm",
    "wikitext": "lm",
    "ptb": "lm",
    "wmt14_fr_en": "seq2seq",
    "cnn_dailymail": "seq2seq",
    "opus_books_en_fr": "seq2seq",
    "cifar10": "vit",
    "cifar100": "vit",
    "imagenet": "vit",
    "imagenet1k": "vit",
}


def canonical_dataset_name(name: str) -> str:
    s = name.strip().lower().replace("-", "_")
    s = s.split(":")[0]
    if "/" in s:
        s = s.split("/")[-1]
    return s


def _find_task_from_dataset(name: str | None) -> str | None:
    if not name:
        return None
    canon = canonical_dataset_name(name)
    return DATASET_TO_TASK.get(canon)


def _find_nested_value(cfg: Dict, keys: list[str]) -> str | None:
    for key in keys:
        if key in cfg:
            return cfg[key]
    return None


def _has_diffusion_fields(cfg: Dict) -> bool:
    for key in ("num_diffusion_steps", "noise_schedule"):
        if key in cfg:
            return True
    for value in cfg.values():
        if isinstance(value, dict) and _has_diffusion_fields(value):
            return True
    return False


def detect_task(cfg: Dict) -> str:
    if "task" in cfg and cfg["task"]:
        task = cfg["task"]
        print(f"[task] detected {task} (cfg.task)")
        return task
    if "data" in cfg and isinstance(cfg["data"], dict):
        data_task = _find_nested_value(cfg["data"], ["task"])
        if data_task:
            print(f"[task] detected {data_task} (cfg.data.task)")
            return data_task
    if "task" in cfg and isinstance(cfg["task"], dict):
        data_task = _find_nested_value(cfg["task"], ["data"])
        if data_task:
            print(f"[task] detected {data_task} (cfg.task.data)")
            return data_task

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    dataset_fields = [
        ("lm_dataset", "lm"),
        ("seq_dataset", "seq2seq"),
        ("textdiff_dataset", "textdiff"),
        ("vit_dataset", "vit"),
    ]
    for field, task in dataset_fields:
        if field in data_cfg and data_cfg[field]:
            print(f"[task] detected {task} (cfg.data.{field})")
            return task

    task = _find_task_from_dataset(data_cfg.get("dataset"))
    if task:
        if task == "lm" and cfg.get("task") == "textdiff":
            task = "textdiff"
        print(f"[task] detected {task} (cfg.data.dataset)")
        return task

    if _has_diffusion_fields(cfg):
        print("[task] detected textdiff (diffusion fields)")
        return "textdiff"

    raise ValueError("Could not detect task from config.")
