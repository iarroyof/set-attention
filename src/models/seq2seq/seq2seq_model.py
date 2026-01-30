from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from models.set_only import SetOnlyLM


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 256,
        encoder_family: str = "baseline_token",
        set_only_cfg: Optional[dict] = None,
        shared_embeddings: Optional[nn.Embedding] = None,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_len = max_len

        self.token_emb = shared_embeddings or nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        ff = dim_feedforward or d_model * 4
        self.encoder_family = encoder_family
        if encoder_family == "set_only":
            if set_only_cfg is None:
                raise ValueError("set_only encoder requires set_only_cfg")
            self.encoder = SetOnlyLM(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=set_only_cfg.get("num_layers", num_layers),
                num_heads=set_only_cfg.get("num_heads", num_heads),
                window_size=set_only_cfg.get("window_size", 32),
                stride=set_only_cfg.get("stride", 16),
                dropout=set_only_cfg.get("dropout", dropout),
                max_seq_len=set_only_cfg.get("max_seq_len", max_len),
                pooling=set_only_cfg.get("pooling", "mean"),
                multiscale=set_only_cfg.get("multiscale", False),
                sig_gating=set_only_cfg.get("sig_gating"),
                d_phi=set_only_cfg.get("d_phi"),
                geometry=set_only_cfg.get("geometry"),
                features=set_only_cfg.get("features"),
                router_type=set_only_cfg.get("router_type", "uniform"),
                router_topk=set_only_cfg.get("router_topk", 0),
                backend=set_only_cfg.get("backend", "dense_exact"),
                backend_params=set_only_cfg.get("backend_params"),
                feature_mode=set_only_cfg.get("feature_mode", "geometry_only"),
                feature_params=set_only_cfg.get("feature_params"),
                adapter_type=set_only_cfg.get("adapter_type", "auto"),
                adapter_hidden_multiplier=set_only_cfg.get("adapter_hidden_multiplier", 2),
                adapter_budget_fraction=set_only_cfg.get("adapter_budget_fraction", 0.15),
                gamma=set_only_cfg.get("gamma", 1.0),
                beta=set_only_cfg.get("beta", 0.0),
                token_embedding=self.token_emb,
            )
            self._encoder_is_set_only = True
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self._encoder_is_set_only = False

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _positional(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.token_emb(x) + self.pos_emb(positions)

    def _generate_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def encode(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._encoder_is_set_only:
            enc = self.encoder.encode(src_ids)
            return enc, src_pad_mask
        src = self.dropout(self._positional(src_ids))
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        return memory, src_pad_mask

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory, src_pad_mask = self.encode(src_ids, src_pad_mask)
        tgt = self.dropout(self._positional(tgt_ids))
        tgt_mask = self._generate_subsequent_mask(tgt_ids.size(1), tgt_ids.device)
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.lm_head(out)

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor],
        max_len: int,
    ) -> torch.Tensor:
        memory, src_pad_mask = self.encode(src_ids, src_pad_mask)
        ys = torch.full((src_ids.size(0), 1), self.bos_id, device=src_ids.device, dtype=torch.long)
        for _ in range(max_len - 1):
            tgt = self.dropout(self._positional(ys))
            tgt_mask = self._generate_subsequent_mask(ys.size(1), ys.device)
            out = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=ys.eq(self.pad_id),
                memory_key_padding_mask=src_pad_mask,
            )
            logits = self.lm_head(out[:, -1])
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_id], dim=1)
            if (next_id.squeeze(1) == self.eos_id).all():
                break
        return ys

    def get_diagnostics(self) -> Optional[dict]:
        if self._encoder_is_set_only:
            return self.encoder.get_diagnostics()
        return None
