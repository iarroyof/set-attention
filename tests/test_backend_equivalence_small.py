import torch

from set_attention.backends.dense_exact import DenseExactBackend
from set_attention.backends.local_band import LocalBandBackend


def test_dense_matches_band_with_full_radius():
    torch.manual_seed(0)
    z = torch.randn(2, 6, 16)
    dense = DenseExactBackend(d_model=16, num_heads=4)
    band = LocalBandBackend(d_model=16, num_heads=4, radius=10)
    band.load_state_dict(dense.state_dict())

    out_dense = dense(z, geom_bias=None, content_bias=None, sig_mask=None, seq_len=10)
    out_band = band(z, geom_bias=None, content_bias=None, sig_mask=None, seq_len=10)

    torch.testing.assert_close(out_dense, out_band, rtol=1e-5, atol=1e-5)
