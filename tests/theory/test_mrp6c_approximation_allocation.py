from __future__ import annotations

import math


D_MODEL = 384
HEADS = 8


def num_sets(seq_len: int, window: int, stride: int) -> int:
    if seq_len < window:
        return 0
    return ((seq_len - window) // stride) + 1


def score_factor(coarse_heads: int, seq_len: int) -> int:
    fine_heads = HEADS - coarse_heads
    m_f = num_sets(seq_len, 2, 1)
    m_c = num_sets(seq_len, 4, 2)
    return fine_heads * m_f**2 + coarse_heads * m_c**2


def approximation_delta(epsilon: float, atom_bound: float, tv: float) -> float:
    return epsilon + 2.0 * atom_bound * tv


def span_bound(
    operator_norm: float,
    head_counts: list[int],
    epsilons: list[float],
    atom_bounds: list[float],
    tvs: list[float],
) -> float:
    total = 0.0
    for heads, epsilon, atom_bound, tv in zip(
        head_counts, epsilons, atom_bounds, tvs
    ):
        total += heads * approximation_delta(epsilon, atom_bound, tv) ** 2
    return operator_norm * math.sqrt(total)


def tv_distance(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def routed_scalar_error(
    implemented_values: list[float],
    reference_values: list[float],
    implemented_weights: list[float],
    reference_weights: list[float],
) -> float:
    implemented = sum(w * z for w, z in zip(implemented_weights, implemented_values))
    reference = sum(w * a for w, a in zip(reference_weights, reference_values))
    return abs(implemented - reference)


def rank_ceiling(fine_heads: int, coarse_heads: int, fine_candidates: int = 2, coarse_candidates: int = 2) -> int:
    ceiling = 0
    if fine_heads:
        ceiling += min(fine_heads, fine_candidates)
    if coarse_heads:
        ceiling += min(coarse_heads, coarse_candidates)
    return ceiling


def pre_stack_output_dim(seq_len: int, fine_heads: int, coarse_heads: int) -> int:
    fine_dim = D_MODEL * fine_heads // HEADS
    coarse_dim = D_MODEL * coarse_heads // HEADS
    return (
        num_sets(seq_len, 2, 1) * fine_dim
        + num_sets(seq_len, 4, 2) * coarse_dim
    )


def interior_minimizers(errors: list[float]) -> list[int]:
    best = min(errors)
    return [idx for idx, value in enumerate(errors) if value == best]


def marginal_gains(errors: list[float]) -> list[float]:
    return [errors[n - 1] - errors[n] for n in range(1, len(errors))]


def is_nonincreasing(values: list[float]) -> bool:
    return all(a >= b for a, b in zip(values, values[1:]))


def test_per_group_transport_bound_constant_two_for_tv() -> None:
    implemented_values = [0.8, -0.2]
    reference_values = [1.0, 0.0]
    implemented_weights = [0.75, 0.25]
    reference_weights = [0.25, 0.75]

    epsilon = max(
        abs(z - a) for z, a in zip(implemented_values, reference_values)
    )
    atom_bound = max(abs(a) for a in reference_values)
    tv = tv_distance(implemented_weights, reference_weights)

    assert math.isclose(tv, 0.5)
    assert routed_scalar_error(
        implemented_values,
        reference_values,
        implemented_weights,
        reference_weights,
    ) <= approximation_delta(epsilon, atom_bound, tv)


def test_concatenated_span_bound_matches_direct_sum_geometry() -> None:
    bound = span_bound(
        operator_norm=1.25,
        head_counts=[6, 2],
        epsilons=[0.10, 0.20],
        atom_bounds=[2.0, 1.5],
        tvs=[0.05, 0.10],
    )
    expected = 1.25 * math.sqrt(
        6 * (0.10 + 2 * 2.0 * 0.05) ** 2
        + 2 * (0.20 + 2 * 1.5 * 0.10) ** 2
    )
    assert math.isclose(bound, expected)


def test_score_allocation_strictly_decreases_with_coarse_heads() -> None:
    values = [score_factor(n, seq_len=2048) for n in range(HEADS + 1)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_registered_b25_pre_stack_dimension_is_compressive() -> None:
    for seq_len in (512, 1024, 2048, 3584, 4096):
        output_dim = pre_stack_output_dim(seq_len, fine_heads=6, coarse_heads=2)
        assert output_dim == 336 * seq_len - 384
        assert output_dim < seq_len * D_MODEL


def test_discrete_diminishing_returns_sign_change_has_interior_minimizer() -> None:
    errors = [10.0, 7.0, 5.0, 4.0, 3.5, 3.25, 3.4, 3.8, 4.4]
    gains = marginal_gains(errors)

    assert is_nonincreasing(gains)
    assert gains[0] > 0
    assert gains[-1] < 0
    assert interior_minimizers(errors) == [5]
    assert 0 not in interior_minimizers(errors)
    assert HEADS not in interior_minimizers(errors)


def test_pareto_better_than_all_fine_requires_no_greater_error() -> None:
    all_fine_error = 10.0
    all_fine_memory = score_factor(0, seq_len=1024)
    mixed_memory = score_factor(2, seq_len=1024)

    assert mixed_memory < all_fine_memory
    assert 9.9 <= all_fine_error
    assert 10.1 > all_fine_error

    mixed_good = 9.9 <= all_fine_error and mixed_memory < all_fine_memory
    mixed_bad = 10.1 <= all_fine_error and mixed_memory < all_fine_memory
    assert mixed_good
    assert not mixed_bad


def test_rank_ceiling_distinguishes_mixed_not_registered_mixed_rows() -> None:
    ceilings = {
        "b0": rank_ceiling(8, 0),
        "b25": rank_ceiling(6, 2),
        "b50": rank_ceiling(4, 4),
        "b75": rank_ceiling(2, 6),
        "b100": rank_ceiling(0, 8),
    }
    assert ceilings == {"b0": 2, "b25": 4, "b50": 4, "b75": 4, "b100": 2}
    assert ceilings["b25"] == ceilings["b50"] == ceilings["b75"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
