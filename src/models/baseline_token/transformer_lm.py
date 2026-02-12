from __future__ import annotations

import torch
from torch import nn

from .attention import BaselineAttention
from .diagnostics import BaselineAttentionDiagnostics


class BaselineEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn_dropout: float | None,
        resid_dropout: float | None,
        ffn_dropout: float | None,
        attention_family: str,
        backend: str,
        backend_params: dict | None,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        attn_drop = attn_dropout if attn_dropout is not None else dropout
        resid_drop = resid_dropout if resid_dropout is not None else dropout
        ffn_drop = ffn_dropout if ffn_dropout is not None else dropout
        self.self_attn = BaselineAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=attn_drop,
            attention_family=attention_family,
            backend=backend,
            backend_params=backend_params,
            max_seq_len=max_seq_len,
            causal=False,
            is_cross=False,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(ffn_drop)
        self.dropout1 = nn.Dropout(resid_drop)
        self.dropout2 = nn.Dropout(resid_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(x)
        attn_output, attn_weights = self.self_attn(
            attn_input,
            memory=None,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout1(attn_output)
        ff_input = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ff_input))))
        x = x + self.dropout2(ff_output)
        return x, attn_weights


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
        attn_dropout: float | None = None,
        resid_dropout: float | None = None,
        ffn_dropout: float | None = None,
        max_seq_len: int = 512,
        attention_family: str = "dense",
        backend: str = "exact",
        backend_params: dict | None = None,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [
                BaselineEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    ffn_dropout=ffn_dropout,
                    attention_family=attention_family,
                    backend=backend,
                    backend_params=backend_params,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self.diagnostics = BaselineAttentionDiagnostics()

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

        attn_sum = None
        for layer in self.layers:
            x, attn = layer(x, key_padding_mask=key_padding_mask)
            attn_sum = attn if attn_sum is None else attn_sum + attn
        if attn_sum is None:
            attn_sum = torch.zeros(
                (batch_size, seq_len, seq_len), device=x.device, dtype=x.dtype
            )
        attn_mean = attn_sum / max(len(self.layers), 1)
        if self.training:
            self.diagnostics.update(attn_mean.detach())
        return self.lm_head(x)

    def get_diagnostics(self) -> dict[str, float]:
        stats = self.diagnostics.get_epoch_stats()
        self.diagnostics.reset()
        return stats

    def attention_params(self) -> dict[str, torch.Tensor]:
        params: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers):
            for name, param in layer.self_attn.named_parameters():
                params[f"layer{idx}.{name}"] = param
        return params
