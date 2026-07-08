#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment import build_model  # noqa: E402
from train.checkpoints import read_checkpoint, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_a", type=Path)
    parser.add_argument("checkpoint_b", type=Path, nargs="?")
    return parser.parse_args()


def logits_from_payload(payload: dict, input_ids: torch.Tensor) -> torch.Tensor:
    model = build_model(dict(payload["model_config"])).cpu().eval()
    model.load_state_dict(payload["model_state"], strict=True)
    with torch.no_grad():
        return model(input_ids)


def main() -> None:
    args = parse_args()
    paths = [args.checkpoint_a]
    if args.checkpoint_b is not None:
        paths.append(args.checkpoint_b)
    before = {str(path): sha256_file(path) for path in paths}
    payloads = [read_checkpoint(path, map_location="cpu") for path in paths]
    first = payloads[0]
    vocab_size = int(first["model_config"]["vocab_size"])
    max_seq_len = int(first["model_config"].get("max_seq_len", 8))
    seq_len = min(8, max_seq_len)
    generator = torch.Generator().manual_seed(8675309)
    input_ids = torch.randint(
        0,
        vocab_size,
        (2, seq_len),
        generator=generator,
    )
    first_logits = logits_from_payload(first, input_ids)
    replay_logits = logits_from_payload(first, input_ids)
    if not torch.equal(first_logits, replay_logits):
        raise SystemExit("same-checkpoint CPU logit replay mismatch")

    cross_checkpoint_equal = None
    if len(payloads) == 2:
        second = payloads[1]
        for key in (
            "model_config_digest",
            "dataset_digest",
            "tokenizer_digest",
            "applied_seed",
        ):
            if first.get(key) != second.get(key):
                raise SystemExit(f"checkpoint metadata mismatch for {key}")
        first_state = first["model_state"]
        second_state = second["model_state"]
        if set(first_state) != set(second_state):
            raise SystemExit("checkpoint state-dict keys differ")
        cross_checkpoint_equal = all(
            torch.equal(first_state[key], second_state[key])
            for key in first_state
        )
        if not cross_checkpoint_equal:
            raise SystemExit("strict same-seed checkpoint tensors differ")
        second_logits = logits_from_payload(second, input_ids)
        if not torch.equal(first_logits, second_logits):
            raise SystemExit("strict same-seed checkpoint logits differ")

    after = {str(path): sha256_file(path) for path in paths}
    if before != after:
        raise SystemExit("checkpoint verification mutated an input checkpoint")
    print(
        json.dumps(
            {
                "schema": first["schema"],
                "epoch": first["epoch"],
                "global_step": first["global_step"],
                "dataset_digest": first["dataset_digest"],
                "tokenizer_digest": first["tokenizer_digest"],
                "same_checkpoint_logits_exact": True,
                "cross_checkpoint_tensors_exact": cross_checkpoint_equal,
                "input_checkpoint_sha256": before,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
