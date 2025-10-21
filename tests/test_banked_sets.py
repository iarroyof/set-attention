import torch

from set_attention.sets.banked import BankedSetBatch
from set_attention.sets.bank_cache import SetBankCache
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter


def test_banked_block_slicing_and_attention():
    # Two sequences; first has 2 sets {1,2} and {3}; second has 1 set {2}
    values = torch.tensor([1, 2, 3, 2], dtype=torch.long)
    set_offsets = torch.tensor([0, 2, 3, 4], dtype=torch.long)
    seq_offsets = torch.tensor([0, 2, 3], dtype=torch.long)
    bank = BankedSetBatch(values, set_offsets, seq_offsets)

    # Atom features: V=5, D=8
    V, D = 5, 8
    phi_z = torch.randn(V, D)
    cache = SetBankCache(phi_z, minhash_k=16)
    cache.precompute(bank.values, bank.set_offsets)

    # Gather both sequences as a batch
    batch_indices = torch.tensor([0, 1], dtype=torch.long)
    Phi_q, Sig_q, Size_q, q_ptrs = cache.gather(bank.seq_offsets, batch_indices)
    Phi_k, Sig_k, Size_k, k_ptrs = cache.gather(bank.seq_offsets, batch_indices)

    assert q_ptrs.tolist() == [0, 2, 3]
    assert k_ptrs.tolist() == [0, 2, 3]
    assert Phi_q.shape[0] == 3 and Sig_q.shape[0] == 3

    attn = SetBankAttention(d_model=D, num_heads=2)
    Z_sets, q_ptrs_out = attn(Phi_q, Sig_q, Size_q, q_ptrs, Phi_k, Sig_k, Size_k, k_ptrs)

    # Z_sets should have Nq rows and shape (Nq, H, Dh)
    assert Z_sets.shape[0] == 3
    assert Z_sets.shape[1] == 2
    assert Z_sets.shape[2] == D // 2

    # Route tokens (L=4 per sequence) to sets using Phi_q as descriptors
    router = TokenSetRouter(d_model=D, num_heads=2, topk=1)
    token_states = torch.randn(2, 4, D)
    out_tok = router(token_states, Z_sets, Phi_q, q_ptrs_out)
    assert out_tok.shape == (2, 4, D)

