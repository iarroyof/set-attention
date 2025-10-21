from __future__ import annotations
from typing import Callable, Optional

import torch.nn as nn

from .heads.kernel_attention import KernelMultiheadAttention
from .heads.ska import SetKernelMultiheadAttention
from .heads.ska_tokenized import SetKernelMultiheadAttentionTokenized


def _clone_mha_config(mha: nn.MultiheadAttention) -> dict:
    # Retrieve key config to recreate compatible KernelMultiheadAttention
    cfg = {
        "embed_dim": mha.embed_dim,
        "num_heads": mha.num_heads,
        "batch_first": getattr(mha, "batch_first", False),
    }
    return cfg


def replace_multihead_attn(
    module: nn.Module,
    sim: str = "rbf",
    temperature: float = 1.0,
    rbf_gamma: float = 0.5,
    inter_topk: int = 16,
    inter_normalize: bool = True,
    predicate: Optional[Callable[[str, nn.Module], bool]] = None,
) -> int:
    """
    Replace nn.MultiheadAttention modules in a model with KernelMultiheadAttention.

    Args:
        module: root module to search
        sim: similarity ("dot"|"cosine"|"rbf")
        temperature: softmax temperature (dot/cosine)
        rbf_gamma: gamma for RBF attention
        predicate: optional filter (name, submodule) -> bool to select targets

    Returns:
        count of replaced modules
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention) and (predicate is None or predicate(name, child)):
            cfg = _clone_mha_config(child)
            if sim == "ska_true":
                new = SetKernelMultiheadAttention(
                    embed_dim=cfg["embed_dim"],
                    num_heads=cfg["num_heads"],
                    batch_first=cfg["batch_first"],
                    ska_topk=inter_topk,
                    ska_gamma=rbf_gamma,
                    ska_rank=32,
                    ska_ridge=1e-4,
                    dropout=getattr(child, "dropout", 0.0),
                    bias=True,
                )
            elif sim == "ska_tok":
                new = SetKernelMultiheadAttentionTokenized(
                    embed_dim=cfg["embed_dim"],
                    num_heads=cfg["num_heads"],
                    batch_first=cfg["batch_first"],
                    gamma=rbf_gamma,
                    minhash_k=128,
                    dropout=getattr(child, "dropout", 0.0),
                    bias=True,
                )
            else:
                new = KernelMultiheadAttention(
                    embed_dim=cfg["embed_dim"],
                    num_heads=cfg["num_heads"],
                    batch_first=cfg["batch_first"],
                    sim=sim,
                    temperature=temperature,
                    rbf_gamma=rbf_gamma,
                    dropout=getattr(child, "dropout", 0.0),
                    bias=True,
                    inter_topk=inter_topk,
                    inter_normalize=inter_normalize,
                )
            # Attempt to copy weights where shapes match (q,k,v,out projections)
            try:
                new.in_proj_q.weight.data.copy_(child.in_proj_weight.data[: cfg["embed_dim"], :])
                new.in_proj_k.weight.data.copy_(child.in_proj_weight.data[cfg["embed_dim"] : 2 * cfg["embed_dim"], :])
                new.in_proj_v.weight.data.copy_(child.in_proj_weight.data[2 * cfg["embed_dim"] :, :])
                if child.in_proj_bias is not None:
                    new.in_proj_q.bias.data.copy_(child.in_proj_bias.data[: cfg["embed_dim"]])
                    new.in_proj_k.bias.data.copy_(child.in_proj_bias.data[cfg["embed_dim"] : 2 * cfg["embed_dim"]])
                    new.in_proj_v.bias.data.copy_(child.in_proj_bias.data[2 * cfg["embed_dim"] :])
                new.out_proj.weight.data.copy_(child.out_proj.weight.data)
                if child.out_proj.bias is not None:
                    new.out_proj.bias.data.copy_(child.out_proj.bias.data)
            except Exception:
                # If in_proj_weight layout differs (PyTorch versions), skip copying silently
                pass

            setattr(module, name, new)
            replaced += 1
        else:
            replaced += replace_multihead_attn(
                child,
                sim,
                temperature,
                rbf_gamma,
                inter_topk,
                inter_normalize,
                predicate,
            )
    return replaced
