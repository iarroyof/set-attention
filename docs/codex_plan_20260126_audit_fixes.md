# Codex Plan 20260126 — Notation Audit Fixes (Implementation Report)

This report summarizes the **fixes applied** after the notation audit, including:
- What was corrected in code
- The intended math meaning (correct interpretation)
- Evidence via code snippets

Scope: set-only path (features, routing, pooling). Plan source: `docs/codex_plan_20260126.md`.

---

## Fix 1 — A2 fusion for `phi_attn` (geometry row + counts)

**Plan intent (correct meaning):**
- `\phi^{attn}_j` should encode **both** geometry row features `r_j` and hashed counts `c_j`, optionally via an MLP.
- Plan lines: L135, L472.

**What was missing:**
- `phi_attn` was computed from counts only; geometry row used only as `geom_bias`.

**Fix applied:**
- Added **geometry-row projection** and **counts projection**, then fused (MLP by default) into `phi_attn`.
- `geom_bias` is still computed and applied as the canonical additive score bias.

**Code evidence:**

```python
# src/set_attention/features/hashed_counts.py
self.geom_proj = nn.Linear(max_sets, d_model)
self.count_proj = nn.Linear(num_bins, d_model)
...
geom_bias = geom_bias_from_delta(delta, gamma=self.gamma, beta=self.beta)
geom_row = torch.exp(geom_bias)
if m < self.max_sets:
    pad = torch.zeros((m, self.max_sets - m), device=geom_row.device)
    geom_row = torch.cat([geom_row, pad], dim=-1)

proj_geom = self.geom_proj(geom_row)
proj_counts = self.count_proj(counts)
fused = torch.cat([proj_geom, proj_counts], dim=-1)
phi_attn = self.fuse_attn(fused)
```

**Notes:**
- Fusion mode is configurable via `feature_params.fusion` (`"mlp"` default; `"linear"` supported).

---

## Fix 2 — Router descriptor uses pooled set state `a_j` + counts

**Plan intent (correct meaning):**
- `\phi^{route}_j = W_route [a_j ; c_j]`, where `a_j` is pooled set state and `c_j` counts.
- Plan line: L476.

**What was missing:**
- Router descriptor was computed only from counts (or positional embedding), ignoring pooled set state.

**Fix applied:**
- Added a fusion path that concatenates pooled set state `a_j` with projected counts, then projects to `d_route`.

**Code evidence:**

```python
# src/set_attention/features/hashed_counts.py
self.router_count_proj = nn.Linear(num_bins, d_model)
self.router_fuse = nn.Linear(d_model * 2, d_model)
...
router_counts = self.router_count_proj(counts)
if set_states is None:
    desc_router = router_counts
else:
    desc_router = self.router_fuse(torch.cat([set_states, router_counts], dim=-1))
```

**Call site evidence:**

```python
# src/models/set_only/set_only_lm.py
per_batch = [
    self.feature_builder(input_ids[i], bank, set_states[i])
    for i in range(batch)
]
```

---

## Fix 3 — Explicit de-scope of multiscale head→scale mapping

**Plan intent (correct meaning):**
- `Head_h(Z) = SetAttn(Z_{s(h)})` requires multiscale banks and head-to-scale routing.

**Status before:**
- Multiscale not implemented, but no explicit guard.

**Fix applied:**
- Added a **hard runtime guard** and **config validation** for `model.multiscale`.

**Code evidence:**

```python
# src/models/set_only/set_only_lm.py
self.multiscale = multiscale
if self.multiscale:
    raise ValueError("multiscale is not implemented in SetOnlyLM")
```

```python
# src/config/compatibility.py
if model.get("multiscale"):
    raise ConfigError("set_only: multiscale is not implemented in this runner")
```

---

## Fix 4 — Pooling is configurable; soft-trimmed Boltzmann prepared

**Plan request:**
- Keep average pooling default, but allow future “Soft-trimmed Boltzmann selection”.

**Fix applied:**
- Added `model.pooling` config key with default `"mean"`.
- `Bank.pool` now supports a new mode placeholder for `soft_trimmed_boltzmann`.

**Code evidence:**

```python
# src/models/set_only/banks.py
def pool(self, token_embeddings: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    ...
    if mode == "mean":
        ...
        return summed / denom
    if mode == "soft_trimmed_boltzmann":
        raise NotImplementedError(
            "Soft-trimmed Boltzmann selection is not implemented yet."
        )
```

```python
# src/models/set_only/set_only_lm.py
self.pooling = pooling
...
set_states = bank.pool(token_states, mode=self.pooling)
```

```python
# src/config/schema.py (set-only keys)
"pooling",
"multiscale",
```

---

## Additional checks requested in your note

**1) Star notation interpretation**
- Confirmed: `*j`, `*t`, `*{ij}`, `*\phi`, `d*phi` were treated as **subscripts or named dimensions**, not multiplication.

**2) d_phi vs d_model**
- Implementation currently **sets d_phi = d_model** by construction. This is consistent with the adapter interfaces, but should be documented explicitly in configs or docs if needed.

**3) Geometry double-counting**
- `geom_bias` remains as canonical score bias and geometry is now also injected into `phi_attn` via the fused row features. If you want to disable geometry bias when fusion is enabled, I can add a config flag (e.g., `feature_params.geom_bias_in_scores = false`).

---

## Files changed

- `src/set_attention/features/hashed_counts.py`
- `src/models/set_only/set_only_lm.py`
- `src/models/set_only/banks.py`
- `src/config/schema.py`
- `src/config/compatibility.py`

---

If you want, I can:
- add `geom_bias_in_scores` flag to prevent geometry double-counting
- implement a real Soft-trimmed Boltzmann pooling algorithm
- add tests for fusion and router descriptor behavior

