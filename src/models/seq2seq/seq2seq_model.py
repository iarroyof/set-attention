from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from models.set_only import SetOnlyLM
from models.baseline_token.attention import BaselineAttention
from .set_only_cross_attention import SetOnlyCrossAttention, SetOnlyCrossLayer
from .diagnostics import BaselineSeq2SeqDiagnostics


class Seq2SeqEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attention_family: str,
        backend: str,
        backend_params: Optional[dict],
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.self_attn = BaselineAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            attention_family=attention_family,
            backend=backend,
            backend_params=backend_params,
            max_seq_len=max_seq_len,
            causal=False,
            is_cross=False,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
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


class Seq2SeqDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        self_attention_family: str,
        self_backend: str,
        self_backend_params: Optional[dict],
        cross_attention_family: str,
        cross_backend: str,
        cross_backend_params: Optional[dict],
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.self_attn = BaselineAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            attention_family=self_attention_family,
            backend=self_backend,
            backend_params=self_backend_params,
            max_seq_len=max_seq_len,
            causal=True,
            is_cross=False,
        )
        self.cross_attn = BaselineAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            attention_family=cross_attention_family,
            backend=cross_backend,
            backend_params=cross_backend_params,
            max_seq_len=max_seq_len,
            causal=False,
            is_cross=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(x)
        self_out, self_weights = self.self_attn(
            attn_input,
            memory=None,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.dropout1(self_out)
        cross_input = self.norm2(x)
        cross_out, cross_weights = self.cross_attn(
            cross_input,
            memory=memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.dropout2(cross_out)
        ff_input = self.norm3(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ff_input))))
        x = x + self.dropout3(ff_output)
        return x, self_weights, cross_weights


class Seq2SeqDecoderLayerSetOnlyCross(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        self_attention_family: str,
        self_backend: str,
        self_backend_params: Optional[dict],
        cross_attn: SetOnlyCrossAttention,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.self_attn = BaselineAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            attention_family=self_attention_family,
            backend=self_backend,
            backend_params=self_backend_params,
            max_seq_len=max_seq_len,
            causal=True,
            is_cross=False,
        )
        self.cross_attn = cross_attn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None,
        src_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(x)
        self_out, self_weights = self.self_attn(
            attn_input,
            memory=None,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.dropout1(self_out)
        cross_input = self.norm2(x)
        cross_out, cross_weights = self.cross_attn(cross_input, memory, src_ids)
        x = x + self.dropout2(cross_out)
        ff_input = self.norm3(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ff_input))))
        x = x + self.dropout3(ff_output)
        return x, self_weights, cross_weights


class Seq2SeqCrossOnlyLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cross_input = self.norm1(x)
        cross_out, cross_weights = self.cross_attn(
            cross_input,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = x + self.dropout1(cross_out)
        ff_input = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ff_input))))
        x = x + self.dropout2(ff_output)
        return x, cross_weights


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
        decoder_family: str | None = None,
        cross_attention: str = "baseline",
        set_only_cfg: Optional[dict] = None,
        shared_embeddings: Optional[nn.Embedding] = None,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        encoder_attention_family: str = "dense",
        encoder_backend: str = "exact",
        decoder_attention_family: str = "dense",
        decoder_backend: str = "exact",
        cross_attention_family: str = "dense",
        cross_backend: str = "exact",
        encoder_backend_params: Optional[dict] = None,
        decoder_backend_params: Optional[dict] = None,
        cross_backend_params: Optional[dict] = None,
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
        self.decoder_family = "baseline_token"
        if encoder_family not in {"baseline_token", "set_only"}:
            raise ValueError("encoder_family must be baseline_token or set_only")
        if cross_attention not in {"baseline", "set_only"}:
            raise ValueError("cross_attention must be baseline or set_only")
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
                backend=set_only_cfg.get("backend", "exact"),
                backend_params=set_only_cfg.get("backend_params"),
                feature_mode=set_only_cfg.get("feature_mode", "geometry_only"),
                feature_params=set_only_cfg.get("feature_params"),
                adapter_type=set_only_cfg.get("adapter_type", "auto"),
                adapter_hidden_multiplier=set_only_cfg.get("adapter_hidden_multiplier", 2),
                adapter_budget_fraction=set_only_cfg.get("adapter_budget_fraction", 0.15),
                gamma=set_only_cfg.get("gamma", 1.0),
                beta=set_only_cfg.get("beta", 0.0),
                allow_token_token=bool(set_only_cfg.get("allow_token_token", False)),
                token_embedding=self.token_emb,
            )
            self._encoder_is_set_only = True
        else:
            self.encoder_layers = nn.ModuleList(
                [
                    Seq2SeqEncoderLayer(
                        d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=ff,
                        dropout=dropout,
                        attention_family=encoder_attention_family,
                        backend=encoder_backend,
                        backend_params=encoder_backend_params,
                        max_seq_len=max_len,
                    )
                    for _ in range(num_layers)
                ]
            )
            self._encoder_is_set_only = False
        self.decoder = None
        self._decoder_is_set_only = False
        self._decoder_cross_is_set_only = False
        if decoder_family is not None:
            self.decoder_family = decoder_family
        if self.decoder_family == "set_only":
            decoder_cfg = set_only_cfg
            if decoder_cfg is None:
                raise ValueError("set_only decoder requires set_only_cfg")
            if cross_attention == "set_only":
                cross_cfg = decoder_cfg
                cross_attn = SetOnlyCrossAttention(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    window_size=cross_cfg.get("window_size", 32),
                    stride=cross_cfg.get("stride", 16),
                    max_seq_len=cross_cfg.get("max_seq_len", max_len),
                    pooling=cross_cfg.get("pooling", "mean"),
                    feature_mode=cross_cfg.get("feature_mode", "geometry_only"),
                    feature_params=cross_cfg.get("feature_params"),
                    router_type=cross_cfg.get("router_type", "uniform"),
                    router_topk=cross_cfg.get("router_topk", 0),
                    sig_gating=cross_cfg.get("sig_gating"),
                    d_phi=cross_cfg.get("d_phi"),
                    gamma=cross_cfg.get("gamma", 1.0),
                    beta=cross_cfg.get("beta", 0.0),
                )
                self.decoder_cross_layers = nn.ModuleList(
                    [
                        SetOnlyCrossLayer(
                            d_model=d_model,
                            dim_feedforward=ff,
                            dropout=dropout,
                            cross_attn=cross_attn,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self._decoder_cross_is_set_only = True
            else:
                self.decoder_cross_layers = nn.ModuleList(
                    [
                        Seq2SeqCrossOnlyLayer(
                            d_model=d_model,
                            nhead=num_heads,
                            dim_feedforward=ff,
                            dropout=dropout,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self._decoder_cross_is_set_only = False
            self.decoder = SetOnlyLM(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=decoder_cfg.get("num_layers", num_layers),
                num_heads=decoder_cfg.get("num_heads", num_heads),
                window_size=decoder_cfg.get("window_size", 32),
                stride=decoder_cfg.get("stride", 16),
                dropout=decoder_cfg.get("dropout", dropout),
                max_seq_len=decoder_cfg.get("max_seq_len", max_len),
                pooling=decoder_cfg.get("pooling", "mean"),
                multiscale=decoder_cfg.get("multiscale", False),
                sig_gating=decoder_cfg.get("sig_gating"),
                d_phi=decoder_cfg.get("d_phi"),
                geometry=decoder_cfg.get("geometry"),
                features=decoder_cfg.get("features"),
                router_type=decoder_cfg.get("router_type", "uniform"),
                router_topk=decoder_cfg.get("router_topk", 0),
                backend=decoder_cfg.get("backend", "exact"),
                backend_params=decoder_cfg.get("backend_params"),
                feature_mode=decoder_cfg.get("feature_mode", "geometry_only"),
                feature_params=decoder_cfg.get("feature_params"),
                adapter_type=decoder_cfg.get("adapter_type", "auto"),
                adapter_hidden_multiplier=decoder_cfg.get("adapter_hidden_multiplier", 2),
                adapter_budget_fraction=decoder_cfg.get("adapter_budget_fraction", 0.15),
                gamma=decoder_cfg.get("gamma", 1.0),
                beta=decoder_cfg.get("beta", 0.0),
                allow_token_token=bool(decoder_cfg.get("allow_token_token", False)),
                token_embedding=self.token_emb,
                causal=True,
            )
            self._decoder_is_set_only = True
        else:
            if cross_attention == "set_only":
                if set_only_cfg is None:
                    raise ValueError("set_only cross-attention requires set_only_cfg")
                cross_cfg = set_only_cfg
                cross_attn = SetOnlyCrossAttention(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    window_size=cross_cfg.get("window_size", 32),
                    stride=cross_cfg.get("stride", 16),
                    max_seq_len=cross_cfg.get("max_seq_len", max_len),
                    pooling=cross_cfg.get("pooling", "mean"),
                    feature_mode=cross_cfg.get("feature_mode", "geometry_only"),
                    feature_params=cross_cfg.get("feature_params"),
                    router_type=cross_cfg.get("router_type", "uniform"),
                    router_topk=cross_cfg.get("router_topk", 0),
                    sig_gating=cross_cfg.get("sig_gating"),
                    d_phi=cross_cfg.get("d_phi"),
                    gamma=cross_cfg.get("gamma", 1.0),
                    beta=cross_cfg.get("beta", 0.0),
                )
                self.decoder_layers = nn.ModuleList(
                    [
                        Seq2SeqDecoderLayerSetOnlyCross(
                            d_model=d_model,
                            nhead=num_heads,
                            dim_feedforward=ff,
                            dropout=dropout,
                            self_attention_family=decoder_attention_family,
                            self_backend=decoder_backend,
                            self_backend_params=decoder_backend_params,
                            cross_attn=cross_attn,
                            max_seq_len=max_len,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self._decoder_cross_is_set_only = True
            else:
                self.decoder_layers = nn.ModuleList(
                    [
                    Seq2SeqDecoderLayer(
                        d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=ff,
                        dropout=dropout,
                        self_attention_family=decoder_attention_family,
                        self_backend=decoder_backend,
                        self_backend_params=decoder_backend_params,
                        cross_attention_family=cross_attention_family,
                        cross_backend=cross_backend,
                        cross_backend_params=cross_backend_params,
                        max_seq_len=max_len,
                    )
                        for _ in range(num_layers)
                    ]
                )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.diagnostics = BaselineSeq2SeqDiagnostics()

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
        attn_sum = None
        for layer in self.encoder_layers:
            src, attn = layer(src, key_padding_mask=src_pad_mask)
            attn_sum = attn if attn_sum is None else attn_sum + attn
        if attn_sum is None:
            attn_sum = torch.zeros(
                (src.size(0), src.size(1), src.size(1)), device=src.device, dtype=src.dtype
            )
        attn_mean = attn_sum / max(len(self.encoder_layers), 1)
        if self.training:
            self.diagnostics.update(attn_mean.detach(), None, None)
        return src, src_pad_mask

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory, src_pad_mask = self.encode(src_ids, src_pad_mask)
        if self._decoder_is_set_only:
            x = self.decoder.encode(tgt_ids)
            cross_attn_sum = None
            for layer in self.decoder_cross_layers:
                if self._decoder_cross_is_set_only:
                    x, cross_attn = layer(x, memory, src_ids)
                else:
                    x, cross_attn = layer(
                        x,
                        memory,
                        memory_key_padding_mask=src_pad_mask,
                    )
                cross_attn_sum = cross_attn if cross_attn_sum is None else cross_attn_sum + cross_attn
            if self.training and cross_attn_sum is not None:
                cross_attn_mean = cross_attn_sum / max(len(self.decoder_cross_layers), 1)
                self.diagnostics.update(None, None, cross_attn_mean.detach())
            return self.lm_head(x)

        tgt = self.dropout(self._positional(tgt_ids))
        tgt_mask = self._generate_subsequent_mask(tgt_ids.size(1), tgt_ids.device)
        self_attn_sum = None
        cross_attn_sum = None
        x = tgt
        for layer in self.decoder_layers:
            if self._decoder_cross_is_set_only:
                x, self_attn, cross_attn = layer(
                    x,
                    memory,
                    tgt_key_padding_mask=tgt_pad_mask,
                    src_ids=src_ids,
                )
            else:
                x, self_attn, cross_attn = layer(
                    x,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_pad_mask,
                )
            self_attn_sum = self_attn if self_attn_sum is None else self_attn_sum + self_attn
            cross_attn_sum = cross_attn if cross_attn_sum is None else cross_attn_sum + cross_attn
        if self.training:
            self_attn_mean = None
            cross_attn_mean = None
            if self_attn_sum is not None:
                self_attn_mean = self_attn_sum / max(len(self.decoder_layers), 1)
            if cross_attn_sum is not None:
                cross_attn_mean = cross_attn_sum / max(len(self.decoder_layers), 1)
            self.diagnostics.update(
                None,
                self_attn_mean.detach() if self_attn_mean is not None else None,
                cross_attn_mean.detach() if cross_attn_mean is not None else None,
            )
        return self.lm_head(x)

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
            if self._decoder_is_set_only:
                x = self.decoder.encode(ys)
                for layer in self.decoder_cross_layers:
                    if self._decoder_cross_is_set_only:
                        x, _ = layer(x, memory, src_ids)
                    else:
                        x, _ = layer(
                            x,
                            memory,
                            memory_key_padding_mask=src_pad_mask,
                        )
            else:
                tgt = self.dropout(self._positional(ys))
                tgt_mask = self._generate_subsequent_mask(ys.size(1), ys.device)
                x = tgt
                for layer in self.decoder_layers:
                    if self._decoder_cross_is_set_only:
                        x, _, _ = layer(
                            x,
                            memory,
                            tgt_key_padding_mask=ys.eq(self.pad_id),
                            src_ids=src_ids,
                        )
                    else:
                        x, _, _ = layer(
                            x,
                            memory,
                            tgt_mask=tgt_mask,
                            tgt_key_padding_mask=ys.eq(self.pad_id),
                            memory_key_padding_mask=src_pad_mask,
                        )
            logits = self.lm_head(x[:, -1])
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_id], dim=1)
            if (next_id.squeeze(1) == self.eos_id).all():
                break
        return ys

    def get_diagnostics(self) -> Optional[dict]:
        encoder_stats: dict[str, float] = {}
        if self._encoder_is_set_only:
            encoder_stats = self.encoder.get_diagnostics() or {}
        stats = self.diagnostics.get_epoch_stats()
        self.diagnostics.reset()
        if encoder_stats:
            stats = {**stats, **encoder_stats}
        return stats

    def get_last_set_embeddings(self) -> Optional[torch.Tensor]:
        if self._encoder_is_set_only:
            return self.encoder.get_last_set_embeddings()
        if self._decoder_is_set_only and self.decoder is not None:
            return self.decoder.get_last_set_embeddings()
        return None

    def attention_params(self) -> dict[str, torch.Tensor]:
        params: dict[str, torch.Tensor] = {}
        if not self._encoder_is_set_only:
            for idx, layer in enumerate(self.encoder_layers):
                for name, param in layer.self_attn.named_parameters():
                    params[f"encoder.layer{idx}.{name}"] = param
        if self._decoder_is_set_only:
            for idx, layer in enumerate(self.decoder_cross_layers):
                for name, param in layer.cross_attn.named_parameters():
                    params[f"decoder_cross.layer{idx}.{name}"] = param
        else:
            for idx, layer in enumerate(self.decoder_layers):
                for name, param in layer.self_attn.named_parameters():
                    params[f"decoder_self.layer{idx}.{name}"] = param
                for name, param in layer.cross_attn.named_parameters():
                    params[f"decoder_cross.layer{idx}.{name}"] = param
        return params
