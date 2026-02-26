from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from config.load import load_config  # noqa: E402
from run_experiment import build_dataloaders, build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a model through the same LM path as run_experiment.py, "
            "including vocab_size inference when model.vocab_size=0."
        )
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        nargs="+",
        help="Override config values, e.g. model.max_seq_len=32 data.seq_len=32",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Convenience alias for multiple overrides in one flag.",
    )
    return parser.parse_args()


def _flatten_overrides(args: argparse.Namespace) -> list[str]:
    flat: list[str] = []
    for group in args.override:
        flat.extend(group if isinstance(group, list) else [group])
    flat.extend(args.overrides or [])
    return flat


def main() -> None:
    """
    Example:
    docker compose exec -T set-attention bash -lc '
    cd /workspace
    PYTHONPATH=src python scripts/probe_model_forward.py \
      --config configs/set_only/wikitext2_dense_exact.yaml \
      --overrides data.seq_len=32 data.batch_size=2 model.max_seq_len=32
    '
    """
    args = parse_args()
    cfg = load_config(args.config, overrides=_flatten_overrides(args))

    train_loader, _, vocab_size = build_dataloaders(cfg["data"])
    if cfg["model"].get("vocab_size", 0) in (0, None):
        cfg["model"]["vocab_size"] = vocab_size

    model = build_model(cfg["model"]).cpu().eval()

    batch_size = int(cfg["data"]["batch_size"])
    seq_len = int(cfg["data"]["seq_len"])
    with torch.no_grad():
        input_ids = torch.randint(
            low=0,
            high=int(cfg["model"]["vocab_size"]),
            size=(batch_size, seq_len),
            dtype=torch.long,
        )
        logits = model(input_ids)

    print(f"logits.shape={tuple(logits.shape)}")
    print(f"resolved_vocab_size={cfg['model']['vocab_size']}")
    print(f"model.implementation={cfg['model'].get('implementation')}")
    print(f"model.causal(cfg)={cfg['model'].get('causal', None)}")
    print(f"model.causal(runtime)={getattr(model, 'causal', None)}")
    try:
        print(f"train_samples={len(train_loader.dataset)}")
    except Exception:
        print("train_samples=iterable_dataset")


if __name__ == "__main__":
    main()
