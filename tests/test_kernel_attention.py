import torch
from set_attention.heads.kernel_attention import KernelMultiheadAttention


def test_kernel_mha_shapes():
    B, L, E, H = 2, 8, 64, 4
    attn = KernelMultiheadAttention(E, H, batch_first=True, sim="rbf", rbf_gamma=0.5)
    x = torch.randn(B, L, E)
    y, w = attn(x, x, x, need_weights=True)
    assert y.shape == (B, L, E)
    assert w.shape == (B, L, L)

