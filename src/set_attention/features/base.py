from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class SetFeatures:
    phi_attn: torch.Tensor | None
    desc_router: torch.Tensor | None
    geom_bias: torch.Tensor | None
    sig_mask: torch.Tensor | None = None

    def to(self, device: torch.device) -> "SetFeatures":
        def _move(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device) if t is not None else None

        return SetFeatures(
            phi_attn=_move(self.phi_attn),
            desc_router=_move(self.desc_router),
            geom_bias=_move(self.geom_bias),
            sig_mask=_move(self.sig_mask),
        )
