from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

MODEL_TYPES: tuple[str, ...] = ("baseline", "ska", "set_kernel")
BASELINE_IMPLS: tuple[str, ...] = ("pytorch", "explicit")
SKA_BACKENDS: tuple[str, ...] = ("python", "triton", "keops")
SKA_SCORE_MODES: tuple[str, ...] = (
    "dot",
    "delta_rbf",
    "delta_plus_dot",
    "intersect_norm",
    "intersect_plus_dot",
)


@dataclass
class ModelConfig:
    model_type: str
    baseline_impl: Optional[str] = None
    ska_backend: Optional[str] = None
    ska_score_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Invalid model_type '{self.model_type}'")
        if self.model_type == "baseline":
            if self.baseline_impl not in BASELINE_IMPLS:
                raise ValueError(f"Invalid baseline_impl '{self.baseline_impl}'")
            self.ska_backend = None
            self.ska_score_mode = None
        elif self.model_type == "ska":
            if self.ska_backend not in SKA_BACKENDS:
                raise ValueError(f"Invalid ska_backend '{self.ska_backend}'")
            if self.ska_score_mode not in SKA_SCORE_MODES:
                raise ValueError(f"Invalid ska_score_mode '{self.ska_score_mode}'")
            self.baseline_impl = None
        elif self.model_type == "set_kernel":
            self.baseline_impl = None
            self.ska_backend = None
            self.ska_score_mode = None


def add_model_args(
    parser,
    *,
    default_model_type: str = "ska",
    allow_set_kernel: bool = False,
    default_baseline_impl: str = "pytorch",
    default_ska_backend: str = "python",
    default_ska_score_mode: str = "delta_plus_dot",
):
    choices = ["baseline", "ska"]
    if allow_set_kernel:
        choices.append("set_kernel")
    parser.add_argument(
        "--model-type",
        choices=choices,
        default=default_model_type,
        help="Model type: baseline (standard attention) or ska (banked set attention).",
    )
    parser.add_argument(
        "--baseline-impl",
        choices=BASELINE_IMPLS,
        default=default_baseline_impl,
        help="Baseline attention implementation (pytorch or explicit).",
    )
    parser.add_argument(
        "--ska-backend",
        choices=SKA_BACKENDS,
        default=default_ska_backend,
        help="SKA backend implementation.",
    )
    parser.add_argument(
        "--ska-score-mode",
        choices=SKA_SCORE_MODES,
        default=default_ska_score_mode,
        help="SKA score mode (kernel).",
    )
    return parser


def build_model_config(args, *, allow_set_kernel: bool = False) -> ModelConfig:
    if getattr(args, "model_type", None) is None:
        raise ValueError("--model-type is required.")
    if args.model_type == "set_kernel" and not allow_set_kernel:
        raise ValueError("--model-type set_kernel is not supported for this script.")
    return ModelConfig(
        model_type=args.model_type,
        baseline_impl=getattr(args, "baseline_impl", None),
        ska_backend=getattr(args, "ska_backend", None),
        ska_score_mode=getattr(args, "ska_score_mode", None),
    )


def model_config_fields(cfg: ModelConfig) -> dict[str, str]:
    return {
        "model_type": cfg.model_type,
        "baseline_impl": cfg.baseline_impl or "NA",
        "ska_backend": cfg.ska_backend or "NA",
        "ska_score_mode": cfg.ska_score_mode or "NA",
    }


def model_impl_label(cfg: ModelConfig) -> str:
    if cfg.model_type == "baseline":
        return f"baseline/{cfg.baseline_impl}"
    if cfg.model_type == "ska":
        return f"ska/{cfg.ska_backend}/{cfg.ska_score_mode}"
    return "set_kernel"


def ensure_model_type_allowed(cfg: ModelConfig, allowed: Sequence[str]) -> None:
    if cfg.model_type not in allowed:
        raise ValueError(f"Model type '{cfg.model_type}' not allowed (allowed: {', '.join(allowed)}).")
