from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.load import load_config
from scripts.run_experiment import build_model


def _tiny_multiresolution_model():
    cfg = load_config(
        ROOT / "configs/set_dictionary/sd9_multiresolution.yaml",
        overrides=[
            "model.vocab_size=32",
            "model.d_model=32",
            "model.d_phi=32",
            "model.set_state_dim=32",
            "model.num_heads=4",
            "model.num_layers=1",
            "model.dim_feedforward=64",
            "model.max_seq_len=8",
            "model.feature_mode=geometry_only",
            "model.router_topk=2",
            "model.multiresolution.groups=[{name: fine, num_heads: 2, window_size: 2, stride: 1}, {name: coarse, num_heads: 2, window_size: 4, stride: 2}]",
            "data.seq_len=8",
            "data.batch_size=2",
        ],
    )
    model = build_model(cfg["model"])
    model.grad_probe_interval = 1
    return model


def test_multiresolution_training_diagnostics_are_grouped_and_complete() -> None:
    torch.manual_seed(3)
    model = _tiny_multiresolution_model()
    model.train()
    input_ids = torch.randint(0, 32, (2, 8))
    logits = model(input_ids, labels=input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1))
    loss.backward()
    model.collect_grad_diagnostics()
    model.update_parameter_diagnostics()
    diagnostics = model.get_diagnostics()

    for group in ("fine", "coarse"):
        for metric in (
            "routing_entropy_norm",
            "router_top1_weight",
            "pooling_effective_support",
            "router_gradient_norm",
            "router_param_norm",
            "grad_norm_token_pre_pool",
            "grad_norm_set_post_pool",
            "grad_norm_set_post_blocks",
        ):
            value = diagnostics[f"ausa/{group}/{metric}"]
            assert value == value
    assert diagnostics["ausa/router_param_norm"] > 0.0
    assert diagnostics["ausa/grad_norm_set_post_pool"] > 0.0


def test_gradient_probe_is_rearmed_after_epoch_diagnostics() -> None:
    torch.manual_seed(5)
    model = _tiny_multiresolution_model()
    model.grad_probe_interval = 200
    input_ids = torch.randint(0, 32, (2, 8))

    for _ in range(2):
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model(input_ids, labels=input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            input_ids.reshape(-1),
        )
        loss.backward()
        model.collect_grad_diagnostics()

        model.eval()
        with torch.no_grad():
            model(input_ids)
        diagnostics = model.get_diagnostics()

        for group in ("fine", "coarse"):
            assert diagnostics[f"ausa/{group}/grad_norm_token_pre_pool"] >= 0.0
            assert diagnostics[f"ausa/{group}/grad_norm_set_post_pool"] >= 0.0
            assert diagnostics[f"ausa/{group}/grad_norm_set_post_blocks"] >= 0.0
        assert model._forward_step == 0


def test_multiresolution_eval_probes_are_grouped() -> None:
    torch.manual_seed(4)
    model = _tiny_multiresolution_model()
    model.eval()
    input_ids = torch.randint(0, 32, (2, 8))
    model.reset_probe_metrics()
    with torch.no_grad():
        model(input_ids)
    probes = model.get_probe_metrics()
    for group in ("fine", "coarse"):
        assert probes[f"effective_range_{group}"] >= 0.0
        assert probes[f"routing_entropy_{group}"] >= 0.0
        assert 0.0 <= probes[f"routing_top1_{group}"] <= 1.0
