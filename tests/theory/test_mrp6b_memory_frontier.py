from __future__ import annotations

import math


D_MODEL = 384
D_FF = 1536
HEADS = 8
LAYERS = 6
VOCAB = 76618
HASH_BINS = 128

BLURS = {
    "b0": (8, 0),
    "b25": (6, 2),
    "b50": (4, 4),
    "b75": (2, 6),
    "b100": (0, 8),
}


def num_sets(seq_len: int, window: int, stride: int) -> int:
    if seq_len < window:
        return 0
    return ((seq_len - window) // stride) + 1


def groups_for_blur(fine_heads: int, coarse_heads: int) -> list[tuple[int, int, int]]:
    groups: list[tuple[int, int, int]] = []
    if fine_heads:
        groups.append((fine_heads, 2, 1))
    if coarse_heads:
        groups.append((coarse_heads, 4, 2))
    return groups


def score_elements(seq_len: int, batch: int, fine_heads: int, coarse_heads: int) -> int:
    return batch * LAYERS * sum(
        h * num_sets(seq_len, w, s) ** 2
        for h, w, s in groups_for_blur(fine_heads, coarse_heads)
    )


def group_params(seq_len: int, group_heads: int, window: int, stride: int) -> int:
    stream_dim = D_MODEL * group_heads // HEADS
    set_count = num_sets(seq_len, window, stride)
    input_proj = 0 if stream_dim == D_MODEL else D_MODEL * stream_dim + stream_dim
    block = LAYERS * (
        4 * stream_dim**2
        + 2 * stream_dim * D_FF
        + D_FF
        + 9 * stream_dim
    )
    feature = (
        set_count * stream_dim
        + 5 * stream_dim**2
        + (2 * HASH_BINS + 6) * stream_dim
    )
    adapter = 2 * stream_dim**2
    router = group_heads * stream_dim * (D_MODEL + stream_dim + 2)
    return input_proj + block + feature + adapter + router


def runtime_params(seq_len: int, fine_heads: int, coarse_heads: int) -> int:
    base = 2 * VOCAB * D_MODEL + seq_len * D_MODEL
    return base + sum(
        group_params(seq_len, h, w, s)
        for h, w, s in groups_for_blur(fine_heads, coarse_heads)
    )


def validate_fit_rows(rows: list[dict[str, object]]) -> None:
    strata = {(row["host"], row["batch"]) for row in rows}
    if len(strata) != 1:
        raise ValueError("MRP-6B fit rows must share one host/native-batch stratum")
    if any(row.get("primary_oom") and not row.get("exclusive_oom") for row in rows):
        raise ValueError("primary OOM rows require exclusive admission telemetry")


def test_strict_past_set_counts_match_registered_banks() -> None:
    assert num_sets(2048, 2, 1) == 2047
    assert num_sets(2048, 4, 2) == 1023
    assert num_sets(3584, 2, 1) == 3583
    assert num_sets(3584, 4, 2) == 1791
    assert num_sets(4096, 2, 1) == 4095
    assert num_sets(4096, 4, 2) == 2047


def test_leading_score_coefficients() -> None:
    cases = [
        ("b0", 1.0),
        ("b25", 0.8125),
        ("b50", 0.625),
        ("b75", 0.4375),
        ("b100", 0.25),
    ]
    for label, expected in cases:
        fine, coarse = BLURS[label]
        coeff = fine / HEADS + (coarse / HEADS) / 4
        assert coeff == expected


def test_finite_score_ratios_are_monotone_in_coarse_heads() -> None:
    seq_len = 4096
    values = [
        score_elements(seq_len, batch=4, fine_heads=f, coarse_heads=c)
        for f, c in BLURS.values()
    ]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_registered_parameter_counts_are_not_blur_identical() -> None:
    counts = {
        label: runtime_params(seq_len=2048, fine_heads=f, coarse_heads=c)
        for label, (f, c) in BLURS.items()
    }
    assert counts == {
        "b0": 74560128,
        "b25": 71796480,
        "b50": 70757376,
        "b75": 71599872,
        "b100": 74166912,
    }
    assert len(set(counts.values())) == len(counts)


def test_exact_score_formula_preserves_quadratic_class() -> None:
    fine, coarse = BLURS["b25"]
    ratios = []
    for seq_len in (512, 1024, 2048, 4096):
        score = score_elements(seq_len, batch=1, fine_heads=fine, coarse_heads=coarse)
        ratios.append(score / (LAYERS * HEADS * seq_len**2))
    assert all(ratio < 0.8125 for ratio in ratios)
    assert abs(ratios[-1] - 0.8125) < abs(ratios[0] - 0.8125)
    assert math.isclose(ratios[-1], 0.8125, rel_tol=0.002)


def test_fit_rows_reject_cross_stratum_and_uncertified_oom() -> None:
    validate_fit_rows(
        [
            {"host": "lizmark", "batch": 4, "primary_oom": False},
            {"host": "lizmark", "batch": 4, "primary_oom": False},
        ]
    )
    try:
        validate_fit_rows(
            [
                {"host": "lizmark", "batch": 4, "primary_oom": False},
                {"host": "blue", "batch": 4, "primary_oom": False},
            ]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("cross-host rows were accepted")

    try:
        validate_fit_rows(
            [
                {
                    "host": "lizmark",
                    "batch": 4,
                    "primary_oom": True,
                    "exclusive_oom": False,
                }
            ]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("uncertified primary OOM row was accepted")


if __name__ == "__main__":
    test_strict_past_set_counts_match_registered_banks()
    test_leading_score_coefficients()
    test_finite_score_ratios_are_monotone_in_coarse_heads()
    test_registered_parameter_counts_are_not_blur_identical()
    test_exact_score_formula_preserves_quadratic_class()
    test_fit_rows_reject_cross_stratum_and_uncertified_oom()
