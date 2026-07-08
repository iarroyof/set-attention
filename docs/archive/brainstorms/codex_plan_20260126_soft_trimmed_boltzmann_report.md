# Soft-Trimmed Boltzmann Pooling Integration Report

This report documents the implementation aligned to the **Codex Plan — Informative Soft-Trimmed Boltzmann Pooling Integration**.

It includes:
- Intended math meaning
- Implemented interpretation
- Code evidence (snippets)
- Notes on any deviations

---

## 1) Pooling module (InformativeBoltzmannPooling)

**Intended math**
- Compute typical center `mu`
- Energies `E_k = ||e_k - mu||^2`
- Trim threshold at quantile `q`
- Soft trim gate `m_k = sigmoid(alpha * (tau_trim - E_k))`
- Boltzmann weights `w_k = softmax(-E_k/tau + log(m_k))`
- Pooled state `a_j = sum_k w_k e_k`
- Gradients flow through everything **except** trim threshold

**Implementation**
- Implemented as `InformativeBoltzmannPooling` in `src/models/set_only/banks.py`.
- Uses `torch.topk` for threshold (fixed window, vectorized).
- Trim threshold computed under `torch.no_grad()`.

**Code evidence**

```python
# src/models/set_only/banks.py
class InformativeBoltzmannPooling(nn.Module):
    def __init__(self, tau: float = 0.1, q: float = 0.8, alpha: float = 10.0) -> None:
        ...

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            mu = x.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        else:
            mu = x.mean(dim=1, keepdim=True)

        d2 = ((x - mu) ** 2).sum(dim=-1) / x.shape[-1]
        if mask is not None:
            d2 = d2.masked_fill(~mask, float("inf"))

        with torch.no_grad():
            k = max(1, int(self.q * d2.shape[1]))
            thresh = torch.topk(d2, k, dim=1, largest=False).values[:, -1:]

        mask_soft = torch.sigmoid(self.alpha * (thresh - d2))
        logits = (-d2 / self.tau) + torch.log(mask_soft + 1e-8)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)
```

**Notes**
- Uses `topk` for threshold selection to avoid `quantile` overhead.
- Masked entries are assigned `inf` energy, and masked in logits, so they do not contribute.

---

## 2) Integration into bank pooling

**Intended math**
- Pooling applies only to compute set state `a_j`.
- Only affects routing fusion and optional feature fusion.

**Implementation**
- `Bank.pool` now accepts `mode` and `params`.
- For `soft_trimmed_boltzmann`, it applies the new pooling module to each set.

**Code evidence**

```python
# src/models/set_only/banks.py
if mode == "soft_trimmed_boltzmann":
    params = params or {}
    pooling = InformativeBoltzmannPooling(
        tau=float(params.get("tau", 0.1)),
        q=float(params.get("q", 0.8)),
        alpha=float(params.get("alpha", 10.0)),
    )
    b, m, w, d = gathered.shape
    flat = gathered.reshape(b * m, w, d)
    flat_mask = (self.set_indices >= 0).unsqueeze(0).expand(b, -1, -1)
    flat_mask = flat_mask.reshape(b * m, w)
    pooled = pooling(flat, mask=flat_mask)
    return pooled.view(b, m, d)
```

---

## 3) Configurable pooling in SetOnlyLM

**Intended math**
- `pooling.mode` is configurable.
- Defaults to mean pooling for backward compatibility.

**Implementation**
- `SetOnlyLM` accepts `pooling` as string or dict.
- Parses `mode`, `tau`, `q`, `alpha` for soft-trimmed Boltzmann.

**Code evidence**

```python
# src/models/set_only/set_only_lm.py
if isinstance(pooling, dict):
    self.pooling_mode = pooling.get("mode", "mean")
    self.pooling_params = {
        "tau": pooling.get("tau", 0.1),
        "q": pooling.get("q", 0.8),
        "alpha": pooling.get("alpha", 10.0),
    }
else:
    self.pooling_mode = pooling
    self.pooling_params = {}
...
set_states = bank.pool(
    token_states, mode=self.pooling_mode, params=self.pooling_params
)
```

**Runner plumbing**

```python
# scripts/run_experiment.py
pooling=model_cfg.get("pooling", "mean"),
```

---

## 4) Config validation

**Intended math**
- Pooling mode must be one of `mean` or `soft_trimmed_boltzmann`.

**Implementation**

```python
# src/config/compatibility.py
pooling_cfg = model.get("pooling", "mean")
if isinstance(pooling_cfg, dict):
    pooling_mode = pooling_cfg.get("mode", "mean")
else:
    pooling_mode = pooling_cfg
require(
    pooling_mode in {"mean", "soft_trimmed_boltzmann"},
    "set_only: pooling.mode must be 'mean' or 'soft_trimmed_boltzmann'",
)
```

---

## 5) Routing descriptor fusion still aligned

**Intended math**
- `phi_route` uses `[a_j ; c_j]`. `a_j` now comes from pooling and can use soft-trimmed Boltzmann.

**Implementation evidence**

```python
# src/set_attention/features/hashed_counts.py
router_counts = self.router_count_proj(counts)
if set_states is None:
    desc_router = router_counts
else:
    desc_router = self.router_fuse(torch.cat([set_states, router_counts], dim=-1))
```

---

## Status

- Soft-trimmed Boltzmann pooling is **implemented and configurable**.
- Mean pooling remains the default.
- No attention or kernel paths were modified.
- Multiscale remains de-scoped.

If you want unit tests or a benchmark toggle, I can add them next.
