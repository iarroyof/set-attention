from __future__ import annotations


def keops_ska_lazy(*args, **kwargs):
    raise NotImplementedError(
        "KeOps backend not available. Install pykeops and implement keops_ska_lazy "
        "to enable the 'keops' SKA backend."
    )
