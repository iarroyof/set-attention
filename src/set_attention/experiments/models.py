from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 2, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L) token ids
        h = self.emb(x)
        h = self.pos(h)
        h = self.enc(h)
        # CLS: mean pool
        h = h.mean(dim=1)
        return self.head(h)


class TinyTransformerDenoiser(nn.Module):
    """Simple transformer denoiser for DDPM-style training on continuous sequences.
    """

    def __init__(self, d_model: int = 64, nhead: int = 2, num_layers: int = 2, in_dim: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(d_model, in_dim)

    def forward(self, x_t: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        # x_t: (B, L, D_in), t_embed: (B, 1, d_model) or (B, d_model)
        h = self.proj_in(x_t)
        h = self.pos(h)
        if t_embed.dim() == 2:
            t_embed = t_embed.unsqueeze(1)
        h = h + t_embed
        h = self.enc(h)
        return self.proj_out(h)

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb

