import torch

from src.models.set_only import SetOnlyLM


SEQ_LEN = 10
VOCAB_SIZE = 64
WINDOW_SIZE = 4
STRIDE = 2
PERTURBATION_SEEDS = range(5)


def _make_model(backend: str, backend_params: dict | None = None) -> SetOnlyLM:
    torch.manual_seed(1234)
    model = SetOnlyLM(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        num_layers=1,
        num_heads=4,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=SEQ_LEN,
        dim_feedforward=32,
        pooling="mean",
        d_phi=16,
        geometry={"enabled": True, "apply_as_bias": True, "apply_in_phi_attn": True},
        router_type="learned",
        router_topk=3,
        router_multihead=True,
        router_temperature=1.0,
        backend=backend,
        backend_params=backend_params,
        feature_mode="hashed_counts",
        feature_params={"num_bins": 32},
        token_mlp=False,
        causal=True,
        set_causality_mode="strict_past",
    )
    model.eval()
    return model


def _base_input() -> torch.Tensor:
    return torch.tensor(
        [
            [3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        ],
        dtype=torch.long,
    )


def _perturb_future(input_ids: torch.Tensor, t: int, seed: int) -> torch.Tensor:
    perturbed = input_ids.clone()
    if t + 1 >= input_ids.size(1):
        return perturbed
    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(seed)
    offsets = torch.randint(
        low=1,
        high=VOCAB_SIZE,
        size=perturbed[:, t + 1 :].shape,
        generator=generator,
        device=input_ids.device,
    )
    perturbed[:, t + 1 :] = (perturbed[:, t + 1 :] + offsets) % VOCAB_SIZE
    return perturbed


def _states_and_logits(model: SetOnlyLM, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        states = model.encode(input_ids)
        logits = model.lm_head(states)
    return states, logits


def _assert_future_perturbations_do_not_change_prefix(
    backend: str,
    backend_params: dict | None = None,
) -> None:
    model = _make_model(backend, backend_params)
    input_ids = _base_input()
    base_states, base_logits = _states_and_logits(model, input_ids)

    for perturb_seed in PERTURBATION_SEEDS:
        for t in range(input_ids.size(1)):
            perturbed = _perturb_future(
                input_ids,
                t=t,
                seed=10_000 + 1_000 * perturb_seed + t,
            )
            states, logits = _states_and_logits(model, perturbed)
            state_diff = (states[:, t] - base_states[:, t]).abs().max().item()
            logit_diff = (logits[:, t] - base_logits[:, t]).abs().max().item()
            assert state_diff <= 1e-5, (
                f"{backend} state changed at t={t}, seed={perturb_seed}: {state_diff}"
            )
            assert logit_diff <= 1e-5, (
                f"{backend} logits changed at t={t}, seed={perturb_seed}: {logit_diff}"
            )


def test_strict_past_dense_exact_future_perturbation_causal():
    _assert_future_perturbations_do_not_change_prefix("exact")


def test_strict_past_sparse_local_band_future_perturbation_causal():
    _assert_future_perturbations_do_not_change_prefix(
        "local_band",
        {"radius": 1},
    )


def test_strict_past_linear_landmark_future_perturbation_causal():
    _assert_future_perturbations_do_not_change_prefix(
        "landmark",
        {"landmark_coverage": 0.5},
    )
