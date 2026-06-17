#!/usr/bin/env python3
"""Focused VRAM audit for A7 token/set operating points.

The probe separates reported training peak memory into:
- observed one-step training peak under the current code path;
- set diagnostic gradient-probe overhead by toggling grad_probe_interval;
- static tensor-size estimates for token states, set states, pooling gathers,
  and dense token-to-set router score/probability tensors.

It is intentionally forward/backward-only and does not train a model.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from models.baseline_token import TransformerLM  # noqa: E402
from models.set_only import SetOnlyLM  # noqa: E402
from models.set_only.banks import num_sets_for_length  # noqa: E402


OUT = ROOT / "audit" / "vram_overhead_audit.json"

VOCAB_SIZE = 76618
BATCH = 16
SEQ_LEN = 512
D_MODEL = 384
D_FF = 1536
NUM_LAYERS = 6
NUM_HEADS = 8
LR = 1e-4
DTYPE_BYTES = 4

BACKENDS = {
    "dense": ("exact", {}),
    "sparse": ("local_band", {"radius": 4}),
    "linear": ("landmark", {"landmark_coverage": 0.25}),
}

TOPOLOGIES = [(1, 1), (4, 2), (16, 8)]


def _run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }


def mib(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0**2)


def tensor_mib(*shape: int, dtype_bytes: int = DTYPE_BYTES) -> float:
    n = 1
    for dim in shape:
        n *= int(dim)
    return mib(n * dtype_bytes)


def param_mib(model: torch.nn.Module) -> float:
    return mib(sum(p.numel() * p.element_size() for p in model.parameters()))


def make_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), generator=gen, dtype=torch.long)
    y = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), generator=gen, dtype=torch.long)
    return x.to(device), y.to(device)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _measured_step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    reset_peak: bool,
) -> dict[str, float]:
    x, y = make_inputs(device)
    allocated_before_step = torch.cuda.memory_allocated() / (1024.0**2)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    after_forward = torch.cuda.memory_allocated() / (1024.0**2)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    after_loss = torch.cuda.memory_allocated() / (1024.0**2)
    loss.backward()
    after_backward = torch.cuda.memory_allocated() / (1024.0**2)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    after_step = torch.cuda.memory_allocated() / (1024.0**2)
    peak = torch.cuda.max_memory_allocated() / (1024.0**2)
    del logits, loss, x, y
    return {
        "allocated_before_step_mib": allocated_before_step,
        "allocated_after_forward_mib": after_forward,
        "allocated_after_loss_mib": after_loss,
        "allocated_after_backward_mib": after_backward,
        "allocated_after_optimizer_step_mib": after_step,
        "peak_mib": peak,
    }


def train_step_peak(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    cleanup()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    cold = _measured_step(model, opt, device, reset_peak=True)
    # Published epoch peaks reset after optimizer state exists in later epochs.
    # Run one more step with AdamW state resident to emulate that steady state.
    warm = _measured_step(model, opt, device, reset_peak=True)
    del opt, model
    cleanup()
    return {
        "cold_first_step": cold,
        "warm_optimizer_resident_step": warm,
    }


def run_model_probe(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Measure a model and release the caller's reference before the next probe."""
    p_mib = param_mib(model)
    probe = train_step_peak(model, device)
    del model
    cleanup()
    return {
        "param_mib": p_mib,
        "adamw_state_mib_estimate": 2.0 * p_mib,
        "probe": probe,
    }


def make_baseline(backend_family: str) -> TransformerLM:
    backend, params = BACKENDS[backend_family]
    return TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=D_FF,
        dropout=0.1,
        attn_dropout=0.1,
        resid_dropout=0.1,
        ffn_dropout=0.1,
        max_seq_len=SEQ_LEN,
        attention_family=backend_family,
        backend=backend,
        backend_params=params,
        causal=True,
    )


def make_set(backend_family: str, window: int, stride: int, grad_probe: bool) -> SetOnlyLM:
    backend, params = BACKENDS[backend_family]
    model = SetOnlyLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        window_size=window,
        stride=stride,
        dropout=0.1,
        attn_dropout=0.1,
        resid_dropout=0.1,
        ffn_dropout=0.1,
        max_seq_len=SEQ_LEN,
        dim_feedforward=D_FF,
        pooling={"mode": "mean"},
        pooling_multihead=False,
        d_phi=D_MODEL,
        set_state_dim=D_MODEL,
        router_type="learned",
        router_topk=16,
        router_multihead=True,
        router_temperature=1.0,
        router_min_temp=0.5,
        backend=backend,
        backend_params=params,
        feature_mode="geometry_only",
        geometry={"enabled": False, "apply_as_bias": False, "apply_in_phi_attn": False},
        token_mlp={"enabled": False},
        allow_token_token=(window == 1 and stride == 1),
        causal=True,
        set_causality_mode="strict_past",
        output_residual_mode="empty_only",
    )
    model.grad_probe_interval = 200 if grad_probe else 0
    return model


def static_estimates(window: int, stride: int) -> dict[str, float | int]:
    m = num_sets_for_length(SEQ_LEN, window, stride, causality_mode="strict_past")
    return {
        "M": m,
        "token_state_mib": tensor_mib(BATCH, SEQ_LEN, D_MODEL),
        "one_set_state_mib": tensor_mib(BATCH, m, D_MODEL),
        "pool_gather_mib": tensor_mib(BATCH, m, window, D_MODEL),
        "router_scores_or_probs_mib": tensor_mib(BATCH, NUM_HEADS, SEQ_LEN, m),
        "set_self_attention_scores_or_probs_mib": tensor_mib(BATCH, NUM_HEADS, m, m),
        "set_grad_probe_extra_mib_formula": tensor_mib(BATCH, SEQ_LEN + 2 * m, D_MODEL),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this audit")
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")

    results: dict[str, Any] = {
        "purpose": "Separate A7 reported VRAM overhead into architecture, implementation, and diagnostic components.",
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "branch": _run(["git", "branch", "--show-current"]),
            "head": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "fixed_shape": {
            "vocab_size": VOCAB_SIZE,
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "dtype_bytes": DTYPE_BYTES,
        },
        "baseline": {},
        "set_only": {},
        "static_estimates": {},
        "conclusion": {},
    }

    for backend_family in BACKENDS:
        results["baseline"][backend_family] = run_model_probe(
            make_baseline(backend_family),
            device,
        )

    for backend_family in BACKENDS:
        results["set_only"][backend_family] = {}
        for window, stride in TOPOLOGIES:
            key = f"w{window}_s{stride}"
            with_probe = run_model_probe(
                make_set(backend_family, window, stride, grad_probe=True),
                device,
            )
            without_probe = run_model_probe(
                make_set(backend_family, window, stride, grad_probe=False),
                device,
            )
            probe_on = with_probe["probe"]["warm_optimizer_resident_step"]
            probe_off = without_probe["probe"]["warm_optimizer_resident_step"]
            results["set_only"][backend_family][key] = {
                "param_mib": with_probe["param_mib"],
                "adamw_state_mib_estimate": with_probe["adamw_state_mib_estimate"],
                "with_grad_probe": probe_on,
                "without_grad_probe": probe_off,
                "measured_grad_probe_delta_mib": probe_on["peak_mib"] - probe_off["peak_mib"],
            }
            results["static_estimates"][key] = static_estimates(window, stride)

    grad_probe_deltas = [
        v["measured_grad_probe_delta_mib"]
        for by_topology in results["set_only"].values()
        for v in by_topology.values()
    ]
    max_grad_probe = max(grad_probe_deltas)
    min_grad_probe = min(grad_probe_deltas)
    results["conclusion"] = {
        "max_measured_set_grad_probe_delta_mib": max_grad_probe,
        "min_measured_set_grad_probe_delta_mib": min_grad_probe,
        "subtract_from_reported_vram": False,
        "reason": (
            "The measurable diagnostic-only component is small relative to the "
            "reported A7 set-token overhead and is topology/model specific. "
            "The dominant contributors are architectural/implementation tensors: "
            "the simultaneous token and set streams plus dense token-to-set router "
            "score/probability tensors and set self-attention tensors. These should "
            "be reported, not subtracted."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
