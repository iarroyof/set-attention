import torch
import torch.nn.functional as F

from scripts.train_toy_diffusion_banked import BankedDenoiser, make_id_sequence
from scripts.train_toy_lm_banked import TinyLMBackbone, make_char_data
from scripts.train_seq2seq_text_banked import TinySeq2SeqBackbone
from scripts.train_tiny_vit_banked import TinyViTBackbone, build_patch_banks
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.kernels.sketches import MinHasher
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.experiments.diffusion_core import SimpleDDPM
from set_attention.training.text_utils import encode_sentence, text_batch_iterator


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


def _build_seq2seq_components():
    torch.manual_seed(1)
    base_sequences = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
    ]
    train_pairs = []
    for seq in base_sequences:
        src = " ".join(f"tok{v}" for v in seq)
        tgt = " ".join(f"tok{v}" for v in reversed(seq))
        train_pairs.append((src, tgt))

    token_set = set()
    for src, tgt in train_pairs:
        token_set.update(src.split())
        token_set.update(tgt.split())
    vocab_tokens = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(token_set)
    stoi = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    vocab_size = len(vocab_tokens)
    pad_id = stoi["<pad>"]
    max_len = 10

    def encode_trim(text: str) -> torch.Tensor:
        ids = encode_sentence(text, stoi, max_len)
        return ids[ids != pad_id]

    src_sequences = [encode_trim(src) for src, _ in train_pairs]
    tgt_sequences = [encode_trim(tgt) for _, tgt in train_pairs]

    src_bank = build_windowed_bank_from_ids(src_sequences, window=3, stride=1)
    tgt_bank = build_windowed_bank_from_ids(tgt_sequences, window=3, stride=1)

    universe = UniversePool(torch.arange(vocab_size, dtype=torch.long))
    minhash = MinHasher(k=16, device=src_bank.values.device)
    cache_src = SetFeatureCache(universe, src_bank.values, src_bank.set_offsets, src_bank.seq_offsets, minhash=minhash)
    cache_tgt = SetFeatureCache(universe, tgt_bank.values, tgt_bank.set_offsets, tgt_bank.seq_offsets, minhash=minhash)

    refs = [tgt.split() for _, tgt in train_pairs]
    iterator = text_batch_iterator(
        train_pairs,
        stoi,
        stoi,
        refs,
        max_len,
        batch_size=len(train_pairs),
        shuffle=False,
    )
    batch_idx, src_ids, tgt_ids, _ = next(iterator)
    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

    d_model = 48
    nhead = 4
    backbone = TinySeq2SeqBackbone(vocab_size, vocab_size, d_model=d_model, nhead=nhead, layers=1)
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
    out_proj = torch.nn.Linear(d_model, vocab_size)
    atom_emb = torch.nn.Embedding(vocab_size, d_model)
    adapter = AtomFeatureAdapter(d_model, rank=2)

    params = (
        list(backbone.parameters())
        + list(set_attn.parameters())
        + list(router.parameters())
        + list(out_proj.parameters())
        + list(atom_emb.parameters())
        + list(adapter.parameters())
    )
    optim = torch.optim.AdamW(params, lr=5e-4)

    return {
        "cache_src": cache_src,
        "cache_tgt": cache_tgt,
        "adapter": adapter,
        "atom_emb": atom_emb,
        "backbone": backbone,
        "set_attn": set_attn,
        "router": router,
        "out_proj": out_proj,
        "optim": optim,
        "batch_idx": batch_idx,
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "tgt_in": tgt_in,
    }


def _seq_loss(components):
    phi_dynamic = components["adapter"](components["atom_emb"].weight)
    Phi_q, Sig_q, Size_q, mask_q = components["cache_tgt"].gather_padded(components["batch_idx"], phi_dynamic)
    Phi_k, Sig_k, Size_k, mask_k = components["cache_src"].gather_padded(components["batch_idx"], phi_dynamic)
    dec_h, _ = components["backbone"](components["src_ids"], components["tgt_in"])
    Z_sets = components["set_attn"](Phi_q, Sig_q, Size_q, mask_q, Phi_k, Sig_k, Size_k, mask_k)
    tok_out = components["router"](dec_h, Z_sets, Phi_q, mask_q)
    logits = components["out_proj"](tok_out)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), components["tgt_ids"].reshape(-1))


class _SyntheticVisionDataset:
    def __init__(self, n_samples: int = 32, num_classes: int = 5, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand((n_samples, 3, 32, 32), generator=g)
        self.labels = torch.randint(0, num_classes, (n_samples,), generator=g)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx].item())


def _build_vit_components():
    dataset = _SyntheticVisionDataset()
    bank = build_patch_banks(dataset, patch=4, limit=len(dataset))
    vocab_size = int(bank.values.max().item() + 1) if bank.values.numel() > 0 else 2

    universe = UniversePool(torch.arange(vocab_size, dtype=torch.long))
    minhash = MinHasher(k=16, device=bank.values.device)
    cache = SetFeatureCache(universe, bank.values, bank.set_offsets, bank.seq_offsets, minhash=minhash)

    d_model = 64
    num_heads = 4
    backbone = TinyViTBackbone(dim=d_model, depth=2, heads=num_heads, patch=4)
    set_attn = SetBankAttention(
        d_model=d_model,
        num_heads=num_heads,
        tau=1.0,
        gamma=0.3,
        beta=0.5,
        score_mode="delta_plus_dot",
        eta=1.0,
    )
    router = TokenSetRouter(d_model=d_model, num_heads=num_heads, topk=0)
    head = torch.nn.Linear(d_model, dataset.num_classes)
    atom_emb = torch.nn.Embedding(vocab_size, d_model)
    adapter = AtomFeatureAdapter(d_model, rank=2)
    params = (
        list(backbone.parameters())
        + list(set_attn.parameters())
        + list(router.parameters())
        + list(head.parameters())
        + list(atom_emb.parameters())
        + list(adapter.parameters())
    )
    optim = torch.optim.AdamW(params, lr=3e-4)

    batch_idx = torch.arange(len(dataset), dtype=torch.long)

    return {
        "dataset": dataset,
        "cache": cache,
        "adapter": adapter,
        "atom_emb": atom_emb,
        "backbone": backbone,
        "set_attn": set_attn,
        "router": router,
        "head": head,
        "optim": optim,
        "batch_idx": batch_idx,
    }


def _vit_loss(components):
    dataset = components["dataset"]
    images = dataset.images
    labels = dataset.labels
    phi_dynamic = components["adapter"](components["atom_emb"].weight)
    Phi, Sig, Size, mask = components["cache"].gather_padded(components["batch_idx"], phi_dynamic)
    tokens = components["backbone"](images)
    Z_sets = components["set_attn"](Phi, Sig, Size, mask, Phi, Sig, Size, mask)
    routed = components["router"](tokens, Z_sets, Phi, mask)
    cls_repr = routed.mean(dim=1)
    logits = components["head"](cls_repr)
    return F.cross_entropy(logits, labels)


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
        torch.manual_seed(1234)
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


def test_seq2seq_demo_step_reduces_loss():
    components = _build_seq2seq_components()
    initial_loss = float(_seq_loss(components).item())
    for _ in range(3):
        components["optim"].zero_grad(set_to_none=True)
        loss = _seq_loss(components)
        loss.backward()
        components["optim"].step()
    final_loss = float(_seq_loss(components).item())
    assert final_loss <= initial_loss + 1e-4


def test_tiny_vit_step_reduces_loss():
    components = _build_vit_components()
    initial_loss = float(_vit_loss(components).item())
    for _ in range(3):
        components["optim"].zero_grad(set_to_none=True)
        loss = _vit_loss(components)
        loss.backward()
        components["optim"].step()
    final_loss = float(_vit_loss(components).item())
    assert final_loss <= initial_loss + 1e-4
