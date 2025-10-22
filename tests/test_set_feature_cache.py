import torch
import pytest

from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher


def _make_base_tensors():
    # Universe ids need not be contiguous to ensure lookup works.
    universe_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    values = torch.tensor([10, 20, 30, 20, 40], dtype=torch.long)
    set_offsets = torch.tensor([0, 2, 4, 5], dtype=torch.long)
    seq_offsets = torch.tensor([0, 2, 3], dtype=torch.long)  # seq0 -> sets {0,1}, seq1 -> set {2}
    return universe_ids, values, set_offsets, seq_offsets


def _make_phi():
    # Simple 2-D features keyed by the universe order (10,20,30,40).
    return torch.tensor(
        [
            [1.0, 0.0],  # id 10
            [0.0, 1.0],  # id 20
            [1.0, 1.0],  # id 30
            [2.0, 0.0],  # id 40
        ],
        dtype=torch.float32,
    )


def test_compute_phi_and_gather_padded():
    universe_ids, values, set_offsets, seq_offsets = _make_base_tensors()
    universe = UniversePool(universe_ids)
    minhash = MinHasher(k=8, device=values.device)
    cache = SetFeatureCache(universe, values, set_offsets, seq_offsets, minhash=minhash)

    phi_cur = _make_phi()

    # Individual set pooling matches manual sums.
    phi_sets = cache.compute_phi_for_indices(torch.tensor([0, 1]), phi_cur)
    expected = torch.tensor([[1.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    assert torch.allclose(phi_sets, expected)

    # Padded gather groups sets per sequence.
    Phi, Sig, Size, mask = cache.gather_padded(torch.tensor([0, 1]), phi_cur)
    assert Phi.shape == (2, 2, 2)
    assert Sig.shape == (2, 2, 8)
    assert Size.shape == (2, 2)
    assert mask.shape == (2, 2)

    # Sequence 0 has two sets, sequence 1 has one (mask second slot False).
    assert torch.allclose(Phi[0, 0], expected[0])
    assert torch.allclose(Phi[0, 1], expected[1])
    assert torch.allclose(Phi[1, 0], torch.tensor([2.0, 0.0]))
    assert not mask[1, 1].item()
    assert Size[0, 0].item() == 2 and Size[1, 0].item() == 1


def test_gather_requires_minhash():
    universe_ids, values, set_offsets, seq_offsets = _make_base_tensors()
    universe = UniversePool(universe_ids)
    cache = SetFeatureCache(universe, values, set_offsets, seq_offsets)
    phi_cur = _make_phi()

    with pytest.raises(RuntimeError, match="MinHash"):
        cache.gather_padded(torch.tensor([0]), phi_cur)
