from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torch import nn


class SetAttentionBackend(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        geom_bias: torch.Tensor | None,
        content_bias: torch.Tensor | None,
        sig_mask: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor:
        raise NotImplementedError
