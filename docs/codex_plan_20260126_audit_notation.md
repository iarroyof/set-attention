# Codex Plan 20260126 notation audit (implementation vs intended math)

Scope: `docs/codex_plan_20260126.md` only.
Goal: list all notation artifacts that use `*x` (e.g., `*j`, `*t`, `*{...}`, `*\phi`) and document:
- what I **interpreted** and implemented
- the **code snippet + source** that resulted
- the **correct interpretation** and **correct implementation**
- a **judgment** of correctness (both interpretation and implementation) with rationale

Legend for judgments:
- Correct = interpretation matches intended math AND implementation matches that interpretation.
- Partially correct = interpretation ok, implementation incomplete or deviates.
- Incorrect = interpretation wrong and/or implementation contradicts plan.

---

## Audit table

| Plan line (literal) | Implemented interpretation (what I understood) | Implemented code snippet + source | Correct interpretation | Correct implementation (what it should be) | Judgment (correct/incorrect + why) |
|---|---|---|---|---|---|
| L100: `\phi^{\text{attn}}*j \in \mathbb R^{d*\phi}` | Read as **per-set attention feature** `phi_attn[j]`, with `*j` meaning subscript `_{j}` and `d*phi` meaning a single feature dimension **d_phi** (not multiplication). | `phi_attn = self.attn_proj(counts)` in `src/set_attention/features/hashed_counts.py` and `phi_attn = self.attn_proj(k_delta)` in `src/set_attention/features/kernel_features.py`. `geometry_only` uses `phi_attn=None` in `src/set_attention/features/geometry_only.py`. | `\phi^{\text{attn}}_{j} \in \mathbb R^{d_\phi}` | Ensure `phi_attn` is an explicit per-set vector (shape `[m, d_phi]`) constructed from the chosen feature family. This is already true for `hashed_counts` and `kernel`; for `geometry_only`, `phi_attn` is intentionally absent per the plan’s A1 description. | **Correct.** Interpretation and implementation align with the intended meaning of a per-set vector. |
| L104: `\phi^{\text{route}}*j \in \mathbb R^{d*\text{route}}` | Read as **per-set router descriptor** `desc_router[j]`, `*j` as subscript and `d*route` as **d_route**. Implemented with `d_route = d_model`. | `desc_router = self.router_emb(set_positions)` in `src/set_attention/features/geometry_only.py`; `desc_router = self.router_proj(counts)` in `src/set_attention/features/hashed_counts.py`; `desc_router = self.router_proj(k_delta)` in `src/set_attention/features/kernel_features.py`. | `\phi^{\text{route}}_{j} \in \mathbb R^{d_\text{route}}` | Per-set router descriptor, typically `d_route = d_model` (as specified). Current implementation matches. | **Correct.** Interpretation and implementation match intended meaning. |
| L120: `\phi^{\text{attn}}*j = W*\phi r_j \in \mathbb R^{d_\phi}` | Read `W*phi` as `W_\phi`, and `*j` as subscript. Implemented as **linear projection** of kernel row features to `d_model`. | `self.attn_proj = nn.Linear(max_sets, d_model)` and `phi_attn = self.attn_proj(k_delta)` in `src/set_attention/features/kernel_features.py`. | `\phi^{\text{attn}}_{j} = W_\phi r_j \in \mathbb R^{d_\phi}` | Same as implemented: linear projection of kernel row `r_j` into feature space. | **Correct.** Interpretation and implementation align. |
| L135: `\phi^{\text{attn}}*j = [W*\phi r_j \ ;\ W_c c_j]` | Interpreted as **concat of projected geometry row + counts**, but I did **not** implement the concat; I used counts only for `phi_attn`, and geometry is a separate bias. | `phi_attn = self.attn_proj(counts)` and `geom_bias = geom_bias_from_delta(...)` in `src/set_attention/features/hashed_counts.py`. | `\phi^{\text{attn}}_{j} = [W_\phi r_j ; W_c c_j]` | Implement concatenation and (optionally) an MLP over the concatenated vector if desired by plan: e.g., `phi_attn = MLP(concat(W_phi r_j, W_c c_j))`. | **Partially correct.** Interpretation was correct, but implementation does not concatenate geometry row `r_j` into `phi_attn`. |
| L272: `A_{ij} = \mathrm{softmax}*j(s*{ij})` | Read as **softmax over j** of `s_{ij}` (i.e., softmax along last dimension). | `attn = torch.softmax(scores, dim=-1)` in `src/set_attention/backends/dense_exact.py` and other backends. | `A_{ij} = \mathrm{softmax}_j(s_{ij})` | Softmax over the key/set dimension (`j`). Implemented correctly. | **Correct.** Interpretation and implementation align. |
| L443: `\phi^{\text{attn}}*j = W*\phi r_j` | Same as L120: per-set attention feature via `W_\phi` projection of geometry/kernel row. | `phi_attn = self.attn_proj(k_delta)` in `src/set_attention/features/kernel_features.py`. | `\phi^{\text{attn}}_{j} = W_\phi r_j` | Same as implemented. | **Correct.** |
| L472: `\phi^{\text{attn}}*j = \mathrm{MLP}([W*\phi r_j ; W_c c_j])` | Interpreted as MLP over concatenated geometry+counts. **Not implemented**; current code uses counts only and no MLP. | `phi_attn = self.attn_proj(counts)` in `src/set_attention/features/hashed_counts.py`. | `\phi^{\text{attn}}_{j} = \mathrm{MLP}([W_\phi r_j ; W_c c_j])` | Implement concatenation and MLP over `[W_phi r_j ; W_c c_j]`. | **Partially correct.** Interpretation correct, implementation incomplete. |
| L476: `\phi^{\text{route}}*j = W*{route} [a_j ; c_j]` | Interpreted as router descriptor from concatenated pooled set state `a_j` and counts `c_j`. **Not implemented**; routing descriptor uses counts only (or position embedding). | `desc_router = self.router_proj(counts)` in `src/set_attention/features/hashed_counts.py`; `desc_router = self.router_emb(set_positions)` in `src/set_attention/features/geometry_only.py`. | `\phi^{\text{route}}_{j} = W_{\text{route}} [a_j ; c_j]` | Implement concatenation of pooled set state `a_j` (from bank pooling) and counts `c_j` and project with `W_route`. | **Partially correct.** Interpretation correct, but implementation does not use `a_j` and does not concatenate. |
| L523: `w_{t,j} = \frac{\langle q_t, \phi^{\text{route}}*j\rangle}{\sqrt{d}} + b*{t,j}` | Read `\phi^{route}*j` as `\phi^{route}_j` and `b*{t,j}` as bias/mask `b_{t,j}` (0 for valid, −inf for invalid). Implemented dot product + mask. | `scores = torch.matmul(q, desc_router.transpose(-2, -1)) * self.scale` and `scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))` in `src/models/set_only/router.py`. | `w_{t,j} = <q_t, \phi^{\text{route}}_{j}>/\sqrt{d} + b_{t,j}` | Same as implemented: add mask bias for invalid sets; no learned bias term exists. | **Correct.** Interpretation and implementation align with the intended mask bias. |
| L573: `\mathrm{Head}*h(Z) = \mathrm{SetAttn}\big(Z*{s(h)}\big)` | Read as `Head_h` and `Z_{s(h)}` (head to scale mapping). **Not implemented**; multiscale banks are not present in current set-only runner. | No corresponding code. `SetOnlyLM` uses a single bank + standard multihead set attention in `src/models/set_only/set_only_lm.py` and `src/set_attention/backends/*`. | `\mathrm{Head}_{h}(Z) = \mathrm{SetAttn}(Z_{s(h)})` | Implement multi-scale banks and head-to-scale mapping (`banks_multiscale.py` + routing). | **Incorrect (not implemented).** Interpretation was correct, but the functionality is missing. |

---

## Summary of mismatches (updated after fixes)

1) **Feature fusion mismatch (A2):**
   - **Resolved.** `phi_attn` now fuses projected geometry rows + counts with an MLP by default.
   - Affected lines: L135, L472.

2) **Router descriptor mismatch (A2):**
   - **Resolved.** `desc_router` now fuses pooled set states `a_j` with counts when available.
   - Affected line: L476.

3) **Multiscale head-to-scale mapping not implemented:**
   - Plan specifies `Head_h` operates on `Z_{s(h)}`.
   - No multiscale bank code in set-only runner (explicitly de-scoped).
   - Affected line: L573.

---

## Notes on notation artifacts

All `*x` artifacts (e.g., `*j`, `*t`, `*{ij}`, `*\phi`) were interpreted as subscripts or identifiers, **not** as multiplication. This is consistent with how the math is described in the surrounding text, and matches the actual implementation for the cases above, except where functionality was not implemented.

---

## Additional ambiguous math inventory (non-`*x` issues)

This extends the audit with other plan math that could be misread or only partially implemented.

| Plan line (literal) | Implemented interpretation (what I understood) | Implemented code snippet + source | Correct interpretation | Judgment (correct/incorrect + why) |
|---|---|---|---|---|
| `q_t = \\rho(e_t, p_t)` (router query) | Treated `\\rho` as attention‑free MLP on token+pos. | `token_states = self.token_mlp(self.token_emb(input_ids) + self.pos_emb(pos_ids))` in `src/models/set_only/set_only_lm.py` | `\\rho` can be any attention‑free mixing (MLP/conv/gated MLP). | **Correct.** One valid instantiation. |
| `Z^{(0)}_j = \\psi(S_j)` (pooling) | Implemented mean pooling; now configurable with soft‑trimmed Boltzmann. | `Bank.pool(..., mode=...)` in `src/models/set_only/banks.py` | `\\psi` is any attention‑free invariant pooling; mean is default. | **Correct.** Mean is valid, with new optional pooling. |
| `b_{sig}(i,j)` neighbor gating | Implemented support in `apply_score_biases`, but **never constructed** a `sig_mask` in set‑only. | `apply_score_biases(..., sig_mask=None)` in `src/set_attention/core.py` | If MinHash neighbor gating is used, `b_sig` must mask disallowed pairs. | **Incorrect / missing.** Hook exists, but no mask is built in set‑only. |
| `g(i,j) = -\\gamma\\Delta(i,j) + \\beta` additive bias | Implemented directly as geometry bias added to scores. | `geom_bias_from_delta` in `src/set_attention/geometry.py` and `apply_score_biases` in `src/set_attention/core.py` | Additive geometry bias. | **Correct.** |
| `K_\\Delta(i,j)=\\exp(g(i,j))` kernel row features | Implemented as exp(−gamma * delta_hat + beta) over MinHash‑derived delta. | `k_delta = torch.exp(-self.gamma * delta_hat + self.beta)` in `src/set_attention/features/kernel_features.py` | Kernel row features are correct; delta_hat is acceptable approximation. | **Correct.** |
| `\\phi^{route}_j = EmbedSetIndex(j) or W_r r_j` | Implemented embedding of set positions in geometry‑only. | `desc_router = self.router_emb(set_positions)` in `src/set_attention/features/geometry_only.py` | Either embedding or row projection is allowed. | **Correct.** |
| `w_{t,j}` bias term `b_{t,j}` | Interpreted as mask bias (0 for valid, −inf for invalid). | `scores = scores.masked_fill(~mask.unsqueeze(0), float(\"-inf\"))` in `src/models/set_only/router.py` | Mask bias is valid; no learned bias specified in plan. | **Correct.** |
| `Head_h(Z)=SetAttn(Z_{s(h)})` (multiscale) | Interpreted as head→scale mapping, but explicitly de‑scoped. | `raise ValueError(\"multiscale is not implemented in SetOnlyLM\")` in `src/models/set_only/set_only_lm.py` and `ConfigError` in `src/config/compatibility.py` | Requires multiscale bank plumbing. | **Incorrect / missing.** De‑scoped by design. |
| `d_\\phi` vs `d_model` | Treated `d_\\phi = d_model` to keep adapters simple. | Projections to `d_model` in `hashed_counts`/`kernel_features`. | Plan allows separate `d_\\phi`; not wired. | **Partially correct.** Valid simplification but should be documented. |

---

## Final notes (updated)

- A2 fusion for `phi_attn` and `desc_router` is now implemented.
- Soft‑trimmed Boltzmann pooling is implemented and configurable (see `model.pooling`).
- Multiscale remains explicitly de‑scoped.
- Signature neighbor gating (`b_sig`) is still not wired in set‑only; only the hook exists.

