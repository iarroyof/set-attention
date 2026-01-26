from __future__ import annotations

import torch
from torch import nn


class TransformerLM(nn.Module):
    """Clean baseline transformer for language modeling (token attention only)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch, seq]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(batch_size, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError("attention_mask must be [batch, seq]")
            key_padding_mask = attention_mask == 0

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.lm_head(x)
