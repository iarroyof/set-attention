# Improvements Matrix Checklist (Set-Attention)

This file captures a **matrix checklist** of improvements vs **backend/feature/task** and includes a **Codex plan** of TODOs to keep future updates consistent and modular.

---

## Legend

- ✅ = applied / centralized
- 🟡 = applies, but verify or wire in this path
- ❌ = not applicable / not present
- ⚠️ = backend-specific; must update per backend

Backends:
- **DE** = dense_exact
- **LB** = local_band
- **ST** = sparse_topk
- **LM** = landmark
- **NY** = nystrom

Feature modes:
- **GO** = geometry_only
- **HC** = hashed_counts
- **KF** = kernel

Tasks:
- **LM** = language modeling
- **S2S** = seq2seq

---

## Matrix: Improvements × Backend/Feature/Task

| Improvement | Location (primary) | DE | LB | ST | LM | NY | GO | HC | KF | LM task | S2S task | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Pooling (mean + soft_trimmed_boltzmann) | `src/models/set_only/banks.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Backend-agnostic |
| Pooling diagnostics (entropy, gini, support, alpha) | `src/models/set_only/banks.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Backend-agnostic |
| Router temperature + debug prints | `src/models/set_only/router.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Learned router only |
| Sig-gating / neighbor mask (pos/minhash) | `src/models/set_only/banks.py` + `set_only_lm.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Feature-mode dependent |
| Set diversity loss | `src/models/set_only/losses.py` + `src/train/loop.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Shared hook |
| Set-only invariant guard bypass | `src/set_attention/core.py` + backends | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Per-backend wiring done |
| Set-level causal masking (decoder) | `src/models/set_only/set_only_lm.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | Task-specific (decoder use) |
| Minhash gating in non-kernel modes | `src/models/set_only/set_only_lm.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ✅ | Needs feature-path checks |
| Baseline diagnostics (attention stats) | `src/models/baseline_token/diagnostics.py` + `src/models/seq2seq/diagnostics.py` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | Task-specific |
| Encoder/Decoder diagnostics merge | `src/models/seq2seq/seq2seq_model.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | Seq2seq only |
| Seq2Seq Set-only decoder (cross-only) | `src/models/seq2seq/seq2seq_model.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | Decoder-family only |

---

## Backend TODO Matrix (per-backend work items)

### Core backends (Set-only)
- `src/set_attention/backends/dense_exact.py`
  - TODO: keep guard bypass wired to `allow_token_token`
  - TODO: ensure `sig_mask` shape semantics match future changes

- `src/set_attention/backends/local_band.py`
  - TODO: propagate any future mask/score changes
  - TODO: verify band mask + sig mask interactions if modified

- `src/set_attention/backends/sparse_topk.py`
  - TODO: keep guard bypass wired to `allow_token_token`
  - TODO: ensure top-k masking compatible with new score semantics

- `src/set_attention/backends/landmark.py`
  - TODO: verify bias broadcasting when geom/content updates occur

- `src/set_attention/backends/nystrom.py`
  - TODO: verify normalization path if future score changes occur

---

## Feature-Path TODOs

- `src/set_attention/features/hashed_counts.py`
  - TODO: keep fusion interface consistent with `d_phi` / `d_model`
  - TODO: keep router descriptor fusion consistent with pooling changes

- `src/set_attention/features/kernel_features.py`
  - TODO: keep signature-based gating aligned with minhash params
  - TODO: verify `sig_k` is the single source of truth

- `src/set_attention/features/geometry_only.py`
  - TODO: if geometry changes (bias scaling), ensure no double-counting

---

## Task TODOs (LM / Seq2Seq parity)

- `src/train/loop.py`
  - TODO: keep diversity loss hooks shared between LM + Seq2Seq
  - TODO: keep diagnostics updates shared for encoder/router and baseline attention

- `src/models/seq2seq/seq2seq_model.py`
  - TODO: keep set-only decoder causal + cross-only path in sync with SetOnlyLM updates
  - TODO: merge encoder + decoder diagnostics consistently

- `src/train/metrics_schema.py`
  - TODO: add new metric keys once and ensure logger handles NA correctly

---

## Codex Plan (Future Improvement Workflow)

1) **Classify change**  
   - Is it **backend-agnostic** (router/pooling/diagnostics)?  
     → Update shared modules only.
   - Is it **backend-specific** (score computation / mask semantics)?  
     → Update all five backends.
   - Is it **feature-path specific** (hashed_counts / kernel / geometry)?  
     → Update feature module + `SetOnlyLM` wiring.
   - Is it **task-specific** (Seq2Seq only)?  
     → Update `seq2seq_model.py` and training loop hooks.

2) **Apply shared updates**
   - `src/models/set_only/banks.py`
   - `src/models/set_only/router.py`
   - `src/models/set_only/diagnostics.py`
   - `src/train/loop.py`

3) **Apply backend updates (if needed)**
   - `dense_exact.py`, `local_band.py`, `sparse_topk.py`, `landmark.py`, `nystrom.py`

4) **Apply feature-path updates (if needed)**
   - `hashed_counts.py`, `kernel_features.py`, `geometry_only.py`

5) **Apply task updates**
   - `seq2seq_model.py`, `diagnostics.py`, `metrics_schema.py`

6) **Run small smoke tests**
   - LM: `configs/ablations/pooling_effect_minimal.yaml` with `data.limit`
   - S2S: OPUS Books config with `data.limit`

---
