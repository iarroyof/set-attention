import pytest
import torch
import torch.nn.functional as F

from scripts.train_toy_diffusion_banked import BankedDenoiser, make_id_sequence
from scripts.train_toy_lm_banked import TinyLMBackbone, make_char_data
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.kernels.sketches import MinHasher
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.experiments.diffusion_core import SimpleDDPM


def _build_lm_components(batch_size=16):
    torch.manual_seed(0)
    X, Y = make_char_data(n=batch_size, seq_len=16, vocab=12, seed=3)
    raw_vocab = int(X.max().item() + 1)
    token_list = [f"tok{i}" for i in range(raw_vocab)]
    vocab_tokens = ["<pad>", "<s>", "</s>", "<unk>"] + token_list
    stoi = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    vocab_size = len(vocab_tokens)

    sequences = [
        torch.tensor([stoi[f"tok{int(tok)}"] for tok in row.tolist()], dtype=torch.long)
        for row in X
    ]
    bank = build_windowed_bank_from_ids(sequences, window=6, stride=3)
    universe = UniversePool(torch.arange(vocab_size, dtype=torch.long))
    minhash = MinHasher(k=16, device=bank.values.device)
    cache = SetFeatureCache(universe, bank.values, bank.set_offsets, bank.seq_offsets, minhash=minhash)

    d_model = 32
    nhead = 2
    backbone = TinyLMBackbone(vocab_size, d_model, nhead, layers=1)
    set_attn = SetBankAttention(
        d_model=d_model,
        num_heads=nhead,
        tau=1.0,
        gamma=0.3,
        beta=0.5,
        score_mode="delta_plus_dot",
        eta=1.0,
    )
    router = TokenSetRouter(d_model=d_model, num_heads=nhead, topk=0)
    head = torch.nn.Linear(d_model, vocab_size)
    atom_emb = torch.nn.Embedding(vocab_size, d_model)
    adapter = AtomFeatureAdapter(d_model, rank=2)
    params = list(backbone.parameters()) + list(set_attn.parameters()) + list(router.parameters()) + list(head.parameters()) + list(atom_emb.parameters()) + list(adapter.parameters())
    optim = torch.optim.AdamW(params, lr=5e-4)

    batch_idx = torch.arange(batch_size, dtype=torch.long)
    src_ids = torch.stack(sequences)
    tgt_ids = torch.tensor([[stoi[f"tok{int(tok)}"] for tok in row.tolist()] for row in Y], dtype=torch.long)

    return {
        "cache": cache,
        "adapter": adapter,
        "atom_emb": atom_emb,
        "backbone": backbone,
        "set_attn": set_attn,
        "router": router,
        "head": head,
        "optim": optim,
        "batch_idx": batch_idx,
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
    }


def _lm_loss(components):
    phi_dynamic = components["adapter"](components["atom_emb"].weight)
    Phi, Sig, Size, mask = components["cache"].gather_padded(components["batch_idx"], phi_dynamic)
    tokens = components["backbone"](components["src_ids"])
    Z = components["set_attn"](Phi, Sig, Size, mask, Phi, Sig, Size, mask)
    routed = components["router"](tokens, Z, Phi, mask)
    logits = components["head"](routed)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), components["tgt_ids"].reshape(-1))


def test_toy_lm_single_step_reduces_loss():
    components = _build_lm_components(batch_size=12)
    initial_loss = float(_lm_loss(components).item())
    components["optim"].zero_grad(set_to_none=True)
    loss = _lm_loss(components)
    loss.backward()
    components["optim"].step()
    new_loss = float(_lm_loss(components).item())
    assert new_loss <= initial_loss


def test_toy_diffusion_single_step_reduces_loss():
    torch.manual_seed(0)
    n_samples = 32
    seq_len = 8
    dim = 4
    X = torch.randn(n_samples, seq_len, dim)
    train_ids = [make_id_sequence(row) for row in X]
    bank = build_windowed_bank_from_ids(train_ids, window=4, stride=2)
    vocab_size = int(bank.values.max().item() + 1) if bank.values.numel() > 0 else seq_len * 2

    atom_emb = torch.nn.Embedding(vocab_size, 16)
    adapter = AtomFeatureAdapter(16, rank=2)
    universe = UniversePool(torch.arange(vocab_size, dtype=torch.long))
    minhash = MinHasher(k=16, device=bank.values.device)
    cache = SetFeatureCache(universe, bank.values, bank.set_offsets, bank.seq_offsets, minhash=minhash)

    model = BankedDenoiser(in_dim=dim, d_model=16, nhead=2, layers=1, router_topk=0)
    ddpm = SimpleDDPM(T=10, device="cpu")
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(atom_emb.parameters()) + list(adapter.parameters()),
        lr=1e-3,
    )

    batch_idx = torch.arange(n_samples, dtype=torch.long)

    def diffusion_loss():
        phi_dynamic = adapter(atom_emb.weight)
        Phi, Sig, Size, mask = cache.gather_padded(batch_idx, phi_dynamic)
        model.set_current_bank(Phi, Sig, Size, mask)
        return ddpm.loss(model, X, lambda t, d: torch.zeros(t.size(0), d, device=X.device), d_model=16)

    with torch.no_grad():
        base_loss = float(diffusion_loss().item())
    optim.zero_grad(set_to_none=True)
    loss = diffusion_loss()
    loss.backward()
    optim.step()
    new_loss = float(diffusion_loss().item())
    assert new_loss <= base_loss


@pytest.mark.skip(reason="Seq2Seq demo integration pending lighter harness.")
def test_seq2seq_demo_step_reduces_loss():
    pass


@pytest.mark.skip(reason="Tiny ViT integration depends on torchvision + CIFAR assets.")
def test_tiny_vit_step_reduces_loss():
    pass
