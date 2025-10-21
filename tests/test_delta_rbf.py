import torch
from set_attention.kernels.delta_rbf import delta_rbf_exact


def test_delta_rbf_self_similarity():
    ids = torch.tensor([1, 2, 3, 2, 3, 4], dtype=torch.long)
    offs = torch.tensor([0, 3, 6], dtype=torch.long)
    K = delta_rbf_exact(ids, offs, ids, offs, gamma=0.3)
    diag = torch.diag(K)
    assert torch.allclose(diag, torch.ones_like(diag))

