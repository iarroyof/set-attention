from __future__ import annotations


def fused_online_softmax(*args, **kwargs):
    raise NotImplementedError(
        "Triton SKA kernels are not yet implemented in this environment. "
        "Install Triton and provide optimized kernels to enable the 'triton' backend."
    )
