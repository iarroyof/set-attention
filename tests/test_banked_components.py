import math

import pytest
import torch

from set_attention.kernels.sketches import MinHasher
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.sets.bank_cache import SetBankCache
from set_attention.sets.bank_utils import gather_bank_batch
from set_attention.heads.token_router import TokenSetRouter
from set_attention.universe import SetFeatureCache, UniversePool


def test_gather_bank_batch_shapes_last_batch():
    sequences = [
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([2, 2, 3, 4], dtype=torch.long),
        torch.tensor([], dtype=torch.long),
    ]
    bank = build_windowed_bank_from_ids(sequences, window=2, stride=2)
    vocab_size = 16
    dim = 4
    phi_z = torch.arange(vocab_size * dim, dtype=torch.float32).reshape(vocab_size, dim)
    cache = SetBankCache(phi_z, minhash_k=4)
    cache.precompute(bank.values, bank.set_offsets)

    batch_indices = torch.tensor([0, 2], dtype=torch.long)
    Phi, Sig, Size, mask = gather_bank_batch(
        bank,
        cache,
        batch_indices,
        phi_dynamic=phi_z,
        use_adapter=False,
        atom_dim=dim,
        minhash_k=4,
    )

    assert Phi.shape[0] == batch_indices.numel()
    assert Sig.shape[:2] == mask.shape
    assert Sig.shape[-1] == 4
    assert Size.shape == mask.shape
    # Last sequence had no sets → mask row is all False
    assert mask[-1].sum().item() == 0


def test_set_feature_cache_adapter_grads_propagate():
    torch.manual_seed(0)
    sequences = [torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4])]
    bank = build_windowed_bank_from_ids(sequences, window=3, stride=3)
    universe = UniversePool(torch.arange(5, dtype=torch.long))
    minhash = MinHasher(k=4, device=bank.values.device)
    cache = SetFeatureCache(universe, bank.values, bank.set_offsets, bank.seq_offsets, minhash=minhash)

    emb = torch.nn.Embedding(5, 3)
    adapter = AtomFeatureAdapter(3, rank=2)

    phi_dynamic = adapter(emb.weight)
    batch_idx = torch.tensor([0, 1], dtype=torch.long)
    Phi, _, _, _ = cache.gather_padded(batch_idx, phi_dynamic)
    loss = Phi.sum()

    emb.zero_grad()
    adapter.zero_grad()
    loss.backward()

    assert emb.weight.grad is not None
    assert adapter.W1.weight.grad is not None
    assert adapter.W1.weight.grad.abs().sum().item() > 0
    assert adapter.W2.weight.grad.abs().sum().item() > 0
    assert adapter.alpha.grad.abs().sum().item() > 0


def test_minhash_estimates_jaccard():
    sets = [
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([2, 3, 4, 5], dtype=torch.long),
        torch.tensor([10, 11], dtype=torch.long),
    ]
    offsets = [0]
    for s in sets:
        offsets.append(offsets[-1] + s.numel())
    values = torch.cat(sets, dim=0)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)

    hasher = MinHasher(k=128)
    sigs = hasher.sketch_vec(values, offsets_tensor)
    approx = MinHasher.jaccard_from_signatures(sigs, sigs).cpu()

    def jaccard(a, b):
        sa, sb = set(a.tolist()), set(b.tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 1.0

    actual = torch.tensor(
        [[jaccard(a, b) for b in sets] for a in sets],
        dtype=torch.float32,
    )

    assert torch.allclose(torch.diag(approx), torch.ones(approx.size(0)), atol=1e-5)
    # Allow a small tolerance on off-diagonal estimates.
    diff = (approx - actual).abs()
    assert diff[0, 1] < 0.2
    assert diff[1, 0] < 0.2


def test_router_topk_softmax_zeroes_inactive():
    router = TokenSetRouter(d_model=4, num_heads=1, topk=2)
    with torch.no_grad():
        router.Wg.weight.copy_(torch.eye(4))
        router.Wg.bias.zero_()
        router.Wd.weight.copy_(torch.eye(4))
        router.Wd.bias.zero_()
        router.out.weight.copy_(torch.eye(4))
        router.out.bias.zero_()

    token_states = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    desc = torch.eye(4).unsqueeze(0)
    Z = torch.zeros(1, 4, 1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)

    _ = router(token_states, Z, desc, mask)
    with torch.no_grad():
        logits = torch.matmul(token_states, desc.transpose(1, 2))
        topk_vals, topk_idx = torch.topk(logits, k=2, dim=-1)
        tmp = torch.full_like(logits, float("-inf"))
        tmp.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        gates = torch.softmax(tmp, dim=-1)

    assert math.isclose(float(gates.sum()), 1.0, rel_tol=1e-6)
    active = gates[0, 0]
    assert active[0] > 0 and active[1] > 0
    assert active[2] == pytest.approx(0.0)
    assert active[3] == pytest.approx(0.0)
