from __future__ import annotations

from pathlib import Path

import torch

from src.models.hybrid_token_set_lm import HybridTokenSetLM


def _make_model(pattern: str = "TSS") -> HybridTokenSetLM:
    set_topologies = [
        {"window_size": 4, "stride": 2},
        {"window_size": 8, "stride": 4},
    ]
    if pattern == "TST":
        set_topologies = [{"window_size": 4, "stride": 2}]
    torch.manual_seed(123)
    model = HybridTokenSetLM(
        vocab_size=97,
        d_model=32,
        num_layers=len(pattern),
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=16,
        attention_family="sparse",
        backend="local_band",
        backend_params={"radius": 1},
        hybrid={"pattern": pattern, "set_topologies": set_topologies},
        pooling="mean",
        d_phi=32,
        feature_mode="hashed_counts",
        feature_params={"num_bins": 32, "hash_seed": 13, "normalize": True},
        router_type="learned",
        router_topk=4,
        router_multihead=True,
        router_temperature=1.0,
        router_min_temp=0.5,
        adapter_type="auto",
        causal=True,
        set_causality_mode="strict_past",
        output_residual_mode="empty_only",
    )
    model.eval()
    return model


def test_hybrid_forward_resolves_metadata_and_keeps_shared_stream_shape():
    model = _make_model("TSS")
    input_ids = torch.arange(0, 16, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 16, 97)
    assert torch.isfinite(logits).all()
    metadata = model.get_resolved_metadata()
    assert metadata["hybrid_pattern"] == "TSS"
    assert metadata["hybrid_set_topologies"] == "4:2;8:4"
    assert metadata["output_residual_mode"] == "empty_only"
    assert metadata["d_phi"] == 32


def test_hybrid_strict_past_future_perturbation_is_causal():
    model = _make_model("TST")
    input_ids = torch.tensor(
        [[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]],
        dtype=torch.long,
    )
    with torch.no_grad():
        base_logits = model(input_ids)

    for t in range(input_ids.size(1)):
        perturbed = input_ids.clone()
        if t + 1 < input_ids.size(1):
            perturbed[:, t + 1 :] = (perturbed[:, t + 1 :] + 17 + t) % 97
        with torch.no_grad():
            logits = model(perturbed)
        diff = (logits[:, t] - base_logits[:, t]).abs().max().item()
        assert diff <= 1e-5, f"hybrid logits changed at t={t}: {diff}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {Path(__file__).name}:{name}")
