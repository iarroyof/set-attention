from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from models.baseline_token import TransformerLM  # noqa: E402
from models.set_only import SetOnlyLM  # noqa: E402


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_set_dictionary_model(
    *,
    d_model: int,
    num_layers: int,
    dim_feedforward: int,
    max_seq_len: int,
    window_size: int,
    stride: int,
    anchor_enabled: bool,
) -> SetOnlyLM:
    return SetOnlyLM(
        vocab_size=128,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=8 if d_model >= 64 else 4,
        window_size=window_size,
        stride=stride,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=max_seq_len,
        dim_feedforward=dim_feedforward,
        pooling="mean",
        d_phi=d_model,
        router_type="learned",
        router_topk=16,
        router_multihead=True,
        backend="exact",
        feature_mode="hashed_counts",
        feature_params={"num_bins": 32 if d_model < 128 else 128},
        token_mlp=False,
        set_causality_mode="strict_past",
        output_residual_mode="anchor_span",
        anchor={
            "enabled": anchor_enabled,
            "pre_encoder_layers": 2,
            "lambda_h": 0.1,
        },
    )


def build_token_baseline(
    *,
    d_model: int,
    num_layers: int,
    dim_feedforward: int,
    max_seq_len: int,
) -> TransformerLM:
    return TransformerLM(
        vocab_size=128,
        d_model=d_model,
        nhead=8 if d_model >= 64 else 4,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=max_seq_len,
        attention_family="dense",
        backend="exact",
        causal=True,
    )


def measure_peak_vram_mib(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    training: bool,
) -> float | None:
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    model = model.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if training:
        model.train()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if hasattr(model, "get_auxiliary_losses"):
            for name, value in model.get_auxiliary_losses().items():
                if name.endswith("_loss"):
                    loss = loss + value
        loss.backward()
    else:
        model.eval()
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def run_smoke() -> dict[str, object]:
    torch.manual_seed(2026)
    small = build_set_dictionary_model(
        d_model=32,
        num_layers=1,
        dim_feedforward=64,
        max_seq_len=16,
        window_size=4,
        stride=2,
        anchor_enabled=True,
    )
    assert small.output_residual_mode == "anchor_span"
    assert small.token_mlp_enabled is False

    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    small.eval()
    with torch.no_grad():
        logits = small(input_ids)
        normal_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        small.set_span_ablation(True)
        ablated_logits = small(input_ids)
        ablated_loss = F.cross_entropy(
            ablated_logits.reshape(-1, ablated_logits.shape[-1]),
            labels.reshape(-1),
        )
        pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        thin_anchor = small.token_emb(input_ids) + small.pos_emb(pos_ids)
        expected_ablated_logits = small.lm_head(thin_anchor)
        thin_anchor_exact = torch.allclose(ablated_logits, expected_ablated_logits, atol=1e-6)
        small.set_span_ablation(False)

    train_vram = measure_peak_vram_mib(
        build_set_dictionary_model(
            d_model=32,
            num_layers=1,
            dim_feedforward=64,
            max_seq_len=16,
            window_size=4,
            stride=2,
            anchor_enabled=True,
        ),
        input_ids,
        labels,
        training=True,
    )
    inference_vram = measure_peak_vram_mib(
        build_set_dictionary_model(
            d_model=32,
            num_layers=1,
            dim_feedforward=64,
            max_seq_len=16,
            window_size=4,
            stride=2,
            anchor_enabled=False,
        ),
        input_ids,
        labels,
        training=False,
    )

    reference_set = build_set_dictionary_model(
        d_model=384,
        num_layers=6,
        dim_feedforward=1536,
        max_seq_len=512,
        window_size=16,
        stride=4,
        anchor_enabled=True,
    )
    reference_token = build_token_baseline(
        d_model=384,
        num_layers=6,
        dim_feedforward=1536,
        max_seq_len=512,
    )
    set_inference_params = reference_set.inference_parameter_count()
    anchor_pre_encoder_params = reference_set.anchor_pre_encoder_parameter_count()
    set_train_params = count_params(reference_set)
    token_params = count_params(reference_token)
    return {
        "status": "pass",
        "checks": {
            "thin_anchor_exact_under_span_ablation": thin_anchor_exact,
            "token_mlp_disabled_for_anchor_span": small.token_mlp_enabled is False,
            "pre_encoder_excluded_from_inference_count": (
                set_train_params - set_inference_params == anchor_pre_encoder_params
            ),
        },
        "small_smoke": {
            "normal_loss": float(normal_loss.item()),
            "span_ablated_loss": float(ablated_loss.item()),
            "span_ablation_delta_loss": float((ablated_loss - normal_loss).item()),
            "train_peak_vram_mib": train_vram,
            "inference_peak_vram_mib": inference_vram,
        },
        "anchor_reference_counts": {
            "set_dictionary_inference_params": set_inference_params,
            "set_dictionary_train_params": set_train_params,
            "anchor_pre_encoder_params": anchor_pre_encoder_params,
            "matched_dense_token_params": token_params,
            "inference_minus_token_params": set_inference_params - token_params,
        },
        "notes": [
            "Synthetic span-ablation delta is a harness smoke value, not a trained PPL collapse claim.",
            "Actual ladder summaries must fail a run if trained span ablation does not sharply worsen PPL.",
        ],
    }


def write_outputs(result: dict[str, object]) -> None:
    audit_dir = ROOT / "audit"
    checks_dir = ROOT / "out" / "paper_integrated_evidence" / "checks"
    audit_dir.mkdir(parents=True, exist_ok=True)
    checks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checks_dir / "sd_fairness_harness_smoke.json"
    audit_path = audit_dir / "SD_3_fairness_harness.md"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    counts = result["anchor_reference_counts"]
    smoke = result["small_smoke"]
    lines = [
        "# SD-3 Fairness Audit Harness",
        "",
        "Status: PASS",
        "",
        "Harness checks:",
        f"- Thin-anchor span ablation exact: {result['checks']['thin_anchor_exact_under_span_ablation']}",
        f"- Token MLP disabled for anchor_span: {result['checks']['token_mlp_disabled_for_anchor_span']}",
        f"- Pre-encoder excluded from inference count: {result['checks']['pre_encoder_excluded_from_inference_count']}",
        "",
        "Anchor-reference parameter counts:",
        f"- Set-dictionary inference params: {counts['set_dictionary_inference_params']}",
        f"- Set-dictionary train params: {counts['set_dictionary_train_params']}",
        f"- Anchor pre-encoder params: {counts['anchor_pre_encoder_params']}",
        f"- Matched dense token params: {counts['matched_dense_token_params']}",
        f"- Inference minus token params: {counts['inference_minus_token_params']}",
        "",
        "Small smoke:",
        f"- Normal loss: {smoke['normal_loss']}",
        f"- Span-ablated loss: {smoke['span_ablated_loss']}",
        f"- Span-ablation delta loss: {smoke['span_ablation_delta_loss']}",
        f"- Train peak VRAM MiB: {smoke['train_peak_vram_mib']}",
        f"- Inference peak VRAM MiB: {smoke['inference_peak_vram_mib']}",
        "",
        "Notes:",
        "- Synthetic span-ablation delta is a harness smoke value, not a trained PPL collapse claim.",
        "- Actual ladder summaries must fail a run if trained span ablation does not sharply worsen PPL.",
        f"- Manifest: `{manifest_path.relative_to(ROOT)}`",
        "",
    ]
    audit_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write audit markdown and manifest")
    args = parser.parse_args()
    result = run_smoke()
    checks = result["checks"]
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        result["status"] = "fail"
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(f"SD-3 fairness harness failed checks: {', '.join(failed)}")
    if args.write:
        write_outputs(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
