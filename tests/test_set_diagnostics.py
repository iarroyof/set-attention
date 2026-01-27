import math

import torch

from src.models.set_only.diagnostics import SetDiagnostics


def test_set_diagnostics_ranges():
    diag = SetDiagnostics()
    bank_indices = torch.tensor([[0, 1, 1, 2], [2, 2, 1, 0]])
    diag.update(bank_indices, num_sets=4)
    stats = diag.get_epoch_stats()
    assert 0.0 <= stats["ausa/active_set_ratio"] <= 1.0
    assert 0.0 <= stats["ausa/routing_entropy_norm"] <= 1.0
    assert 0.0 <= stats["ausa/routing_gini"] <= 1.0
    jaccard = stats["ausa/set_reuse_jaccard"]
    assert math.isnan(jaccard)
