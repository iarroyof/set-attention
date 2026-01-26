import pytest
import torch

from set_attention.core import assert_set_only_scores
from models.set_only import SetOnlyLM


def test_set_only_invariant_trips_on_token_attention():
    scores = torch.zeros(1, 1, 4, 4)
    with pytest.raises(RuntimeError):
        assert_set_only_scores(scores, seq_len=4)


def test_set_only_forward_shapes():
    model = SetOnlyLM(
        vocab_size=50,
        d_model=32,
        num_layers=2,
        num_heads=4,
        window_size=4,
        stride=2,
        max_seq_len=8,
    )
    input_ids = torch.randint(0, 50, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, 50)
