# A8 Plan: Favorable Set-Attention Conditions

Date: 2026-06-13  
Status: planned, not launched  
Motivation: A7/A4 show that set attention is not competitive as a short-context drop-in token-attention replacement, but the long-context comparison narrows the set-vs-token gap while preserving a VRAM reduction. The next experiments should test whether set attention approaches matched baselines while saving memory under conditions that favor compressed set states. Before broad experiments, the most promising engineering target is to remove dense token-to-set routing tensors, because the current router materializes dense `[B,H,L,M]` scores/probs even though strict-past Option-1 only needs the candidate fiber `C_t`.

## Gate From Existing Evidence

Proceed with A8 because Table `tab:long-context` now shows:

- dense SKA gap shrinks from `+705.7` PPL at `L=512` to `+583.4` at `L=2048`;
- sparse SKA gap shrinks from `+737.3` to `+615.2`;
- linear SKA gap shrinks from `+526.5` to `+450.0`;
- long-context SKA uses `59.2%` to `72.3%` of matched token VRAM.

This is not a quality win, but it is a plausible signal that longer context and compression pressure are more favorable than the short-context replacement test.

## Global Rules

- Use only post-A1 causal LM protocol: `set_causality_mode=strict_past`, T1 dropped trailing windows, explicit residual policy.
- Initial scope: dense exact and linear landmark only.
- Seeds: `0,1,2,3,4`.
- Always include matched token baselines where applicable.
- Report raw `train/peak_vram_mib`; do not subtract VRAM unless a future audit proves an exact non-architectural correction.
- Primary question: does SKA approach matched-token PPL while using less VRAM?
- Secondary question: does the favorable condition improve set-minus-token gap relative to A7/A4?

## First Implementation Focus: Near-2 and Near-4 Compression

The first implementation experiments after candidate-gather routing should focus
on the lowest-PPL observed operating points near compression rates 2 and 4, not
on broad topology sweeps. The current A7 tables show that `L/M ~= 2` is the
first point where memory saving becomes meaningful while still leaving enough
set coverage to have a plausible quality path. The exact tested topology
`(w,s)=(4,2)` is the strongest observed near-2 compressed row for both dense and
linear set families. In contrast, `(w,s)=(2,2)` also has `L/M=2`, but its mean
candidate count is about one and quality is much worse. This means the target is
not only compression factor; it is compression with enough candidate support.
For the near-4 regime, `(w,s)=(8,4)` is the lowest-PPL observed row and gives a
clearer memory-saving test than near-2.

Evidence from `a7_backend_family_empty_only_augmented_summary.tsv`:

| Family | Topology `(w,s)` | `M` | `L/M` | mean candidate count | seeds | mean val PPL | mean VRAM MiB | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Set Dense | `(2,2)` | 256 | 2.000 | 0.998 | 3 | 1429.0 | 11813.9 | Near-2 but too few candidates; weak quality. |
| Set Dense | `(4,2)` | 255 | 2.008 | 1.984 | 3 | 1273.6 | 11807.3 | Best observed dense near-2 compressed point. |
| Set Dense | `(8,4)` | 127 | 4.031 | 1.965 | 3 | 1311.7 | 11071.9 | Lowest-PPL observed dense near-4 point. |
| Set Linear | `(2,2)` | 256 | 2.000 | 0.998 | 3 | 1693.6 | 11624.7 | Near-2 but too few candidates; weak quality. |
| Set Linear | `(4,2)` | 255 | 2.008 | 1.984 | 3 | 1356.2 | 11620.6 | Best observed linear near-2 compressed point. |
| Set Linear | `(8,4)` | 127 | 4.031 | 1.965 | 3 | 1394.9 | 11032.6 | Lowest-PPL observed linear near-4 point. |

Compression targets to test first:

1. `(w,s)=(4,2)`:
   - already validated;
   - `M=255` for `L=512`, so `L/M=2.008`;
   - mean candidate count is about `1.984`;
   - use as the primary near-2 quality-preserving compression target.
2. `(w,s)=(8,4)`:
   - already validated;
   - `M=127` for `L=512`, so `L/M=4.031`;
   - mean candidate count is about `1.965`;
   - use as the primary near-4 memory-saving compression target.
3. `(w,s)=(5,2)` or `(6,2)`:
   - not yet validated, but both keep `L/M` close to 2 because stride remains 2;
   - expected `M=254` for `L=512`, so `L/M=2.016`;
   - expected mean candidate count is roughly `w/s`, i.e. about `2.5` for `(5,2)` and about `3.0` for `(6,2)`;
   - rationale: preserve near-2 compression while increasing overlap/candidate support relative to `(4,2)`, which may reduce PPL without giving up the intended compression regime.

Initial recommendation: run `(4,2)` and `(8,4)` first for dense and linear at
larger context. These are the lowest-PPL observed near-2 and near-4 compression
points. Run `(6,2)` as the first unobserved near-2 overlap variant after the
direct large-L check, or in parallel only if compute is available. Use `(5,2)`
only if the extra overlap of `(6,2)` improves PPL but costs too much time/VRAM or
over-smooths routing diagnostics.

## A8.0 Memory-Path Remediation

Goal: remove the implementation overhead that prevents set attention from realizing its intended compression advantage.

Priority 1: candidate-gather routing.

- Current behavior: learned routing builds dense `[B,H,L,M]` score/probability tensors.
- Desired behavior: use the actual strict-past candidate fiber and gather candidates into `[B,H,L,Cmax]`, where `Cmax` is the maximum candidate count in the batch/topology.
- Preserve semantics: outputs must match dense routing up to numerical tolerance for the same model, inputs, masks, and candidate fiber.
- Tests:
  - dense-vs-gather route output equality for representative `w,s,L`, with first coverage at `(4,2)`, `(6,2)`, and `(16,8)`;
  - empty candidate fiber behavior remains identical;
  - strict-past causality tests still pass;
  - VRAM smoke confirms routing memory drops for compressed topologies, especially the near-2 targets.
- Initial benchmark families:
  - `set_dense_exact`
  - `set_linear_landmark`
- Initial benchmark topologies:
  - primary: `(w,s)=(4,2)`;
  - primary near-4 reference: `(w,s)=(8,4)`;
  - near-2 overlap variant: `(w,s)=(6,2)`;
  - optional midpoint if `(6,2)` is promising but too expensive: `(w,s)=(5,2)`;
  - reference only: `(w,s)=(16,8)` from the old LR-norm headline topology.
- Seeds for quality follow-up: `0,1,2,3,4`.

Priority 2: fused/memory-efficient routing and set attention.

- Use PyTorch SDPA/FlashAttention-style fused paths where masks and tensor layouts allow it.
- This is an implementation-memory target, not a quality fix.
- Validate exact or numerically equivalent outputs before using for paper-bound experiments.

Priority 3: activation-sharing variants, only after candidate-gather routing is measured.

- `routing_query_mode=light`: compute routing queries from a lightweight descriptor instead of keeping a full token stream only for routing.
- `token_state_checkpointed`: checkpoint or recompute token states used only for routing.
- `set_only_after_pool`: after pooling, drop most token-stream activations except positions with `C_t=0`.
- These are invasive and must be reported as architectural variants, not as pure implementation optimizations.

Pass signal:

- same or statistically indistinguishable PPL as the current dense-router SKA at matched topology;
- lower peak VRAM, especially for compressed topologies;
- no causality regressions;
- no hidden batch-size changes.

## A8.1 Compression Curriculum

Goal: test whether set models need to learn the token-equivalent regime before being compressed.

Families:

- `baseline_dense_exact`
- `baseline_linear_landmark`
- `set_dense_exact`
- `set_linear_landmark`

Fixed:

- `D=384`, `d_ff=1536`
- `L=512` for first pass, then repeat best setting at `L=2048` if positive
- `landmark_coverage=0.25`
- `output_residual_mode=empty_only`
- 10 epochs per stage or equivalent fixed update budget
- 5 seeds

Curriculum:

1. start at `(w,s)=(1,1)`;
2. continue to `(2,1)`;
3. continue to `(4,2)`;
4. continue to `(6,2)` if `(4,2)` remains stable;
5. optionally continue to `(8,4)` only as a higher-compression reference if validation does not collapse.

Comparison:

- same final topology trained from scratch;
- matched token backend baseline.

Pass signal:

- lower PPL than from-scratch compressed SKA at equal final topology;
- set-minus-token gap narrows by at least 15% relative to A7/A4;
- VRAM remains below matched token baseline for compressed stages.

## A8.2 Token-Teacher Distillation

Goal: force set states and routed token states to preserve predictive information from a matched token-attention teacher.

Teachers:

- dense token teacher for SetDense;
- linear landmark token teacher for SetLinear.

Student families:

- `set_dense_exact`
- `set_linear_landmark`

Losses:

- causal LM cross-entropy;
- logit KL distillation from matched token teacher;
- optional hidden-state matching at final token states, gated by memory budget.

Topologies:

- primary: `(w,s)=(4,2)` and `(6,2)`;
- reference: `(w,s)=(8,4)` only if the near-2 rows improve;
- diagnostic: `(w,s)=(2,1)` if the primary rows fail.

Seeds:

- `0,1,2,3,4`.

Pass signal:

- distillation reduces set-minus-token gap versus non-distilled A7 rows;
- memory saving remains visible at compressed topologies;
- no regression in causality checks.

## A8.3 Fixed-VRAM Long-Context Protocol

Goal: test the setting where set attention should have the clearest advantage: long context under a fixed memory budget.

Families:

- `baseline_dense_exact` if it fits;
- `baseline_linear_landmark`;
- `set_dense_exact`;
- `set_linear_landmark`.

Lengths:

- `L=2048` as anchor;
- `L=8192` as the first direct large-L insight target on a 48 GiB GPU host;
- `L=4096` only as fallback if `L=8192` OOMs or has unsupported sequence-length behavior;
- `L=16384` is out of scope until `L=8192` has a clean smoke and an explicit memory plan.

Lizmark capacity note: `audit/A8_lizmark_largeL_capacity_check.md`. As of 2026-06-13, host `192.168.241.205` has two idle RTX 6000 Ada GPUs with about 48.6 GiB free each. After removing old stopped containers, root has about 137 GiB free. There is still no `~/set-attention` checkout, no Torch install in system Python, and no set-attention container. The inserted 64 GB USB drive is `vfat` and should not be used for Docker storage or unsplit Docker image tars. Prefer streaming the Docker image and copying the minimal repo/cache to root or a proper ext4 disk.

Rules:

- keep token and set batch policy explicit;
- if dense token cannot fit at the set batch size, record the OOM/unsupported result rather than silently lowering batch size;
- compare at equal VRAM tiers when exact equal batch is impossible.

Seeds:

- 5 seeds for `L=2048`;
- for `L=8192`, run seed-0 smoke first for all four matched rows, then run 5 seeds only if the set-vs-token gap narrows versus `L=512`/`L=2048` and the VRAM saving remains real.

Pass signal:

- set-minus-token gap continues to shrink as context increases;
- SKA provides a clear memory or fit advantage;
- at least one dense or linear SKA row approaches the matched-token confidence interval while saving memory.

## A8.4 Auxiliary Compression Objectives

Condition: launch if A8.1-A8.3 or existing long-context comparison shows continued gap narrowing under favorable conditions. Current table gives enough signal to plan the implementation, but launch after A8.1/A8.2 results.

Goal: make set states explicitly information-preserving.

Candidate losses:

- reconstruct teacher final hidden states from routed set context;
- reconstruct window token-state means from set states;
- predict teacher next-token logits from set states at endpoints;
- contrastive retention loss for windows that share entities or repeated spans.

Topologies:

- `(w,s)=(4,2)`;
- `(w,s)=(6,2)`;
- `(w,s)=(8,4)` only as a higher-compression reference.

Families:

- SetDense;
- SetLinear.

Seeds:

- `0,1,2,3,4`.

Pass signal:

- lower PPL at same topology and VRAM;
- improved candidate/routing diagnostics without entropy collapse;
- set-state probes show better reconstruction or retrieval accuracy.

## A8.5 Set-State Compression / Retrieval Evaluation

Goal: evaluate set embeddings as compressed memories directly, not only through next-token PPL.

Tasks:

- associative recall with distractors;
- needle-in-context retrieval over long sequences;
- repeated-entity continuation;
- book/code/math continuation subsets where long-range memory is useful;
- set-state nearest-neighbor retrieval of future-relevant windows.

Metrics:

- retrieval accuracy or rank of relevant window;
- downstream PPL with and without retrieved/compressed set states;
- memory per retained token;
- latency and peak VRAM.

Why separate from A8.1-A8.4:

These tasks test whether set states are useful compressed representations even if short-context PPL remains worse. This is the most direct test of the memory/compression hypothesis.

## A8.6 Hybrid Attention Layer Patterns

Goal: test whether set attention is more useful as a long-context/compression component than as a full replacement for token attention.

Setup:

- 6 layers total.
- Patterns:
  - `TTSSSS`
  - `TSTSTS`
  - `TTTSSS`
  - `SSSTTT`
- Backends:
  - dense token + dense set;
  - landmark token + landmark set.
- Keep the same strict-past bank semantics and `output_residual_mode=empty_only` for set layers.
- Seeds: `0,1,2,3,4` after one smoke seed passes.

Pass signal:

- hybrid rows narrow the SKA-token gap while using less VRAM than all-token dense/landmark baselines;
- no layer pattern is claimed superior without five-seed support.

## A8.7 Larger-Data / Fixed-Token-Budget Training

Goal: test whether SKA is undertrained on WikiText-2 scale rather than fundamentally limited by the current architecture.

Dataset options:

- WikiText-103;
- fixed OpenWebText subset.

Rules:

- compare by fixed token budget, not epochs;
- use matched token baselines and SKA families;
- keep the grid compact:
  - token dense/sparse/linear;
  - set dense/sparse/linear;
  - topologies `(1,1)`, `(2,1)`, `(4,2)`, `(6,2)`, `(16,8)`;
  - current `output_residual_mode=empty_only`;
  - candidate-gather router if implemented and validated.
- Seeds: `0,1,2` if budget permits, otherwise one smoke plus documented budget blocker.

Pass signal:

- SKA gap narrows materially under larger data/fixed-token-budget training;
- no result is interpreted without matched token controls and raw VRAM.

## Initial Launch Recommendation

Launch order:

1. A8.0 candidate-gather routing implementation and dense-vs-gather equivalence tests.
2. A8.0 memory smoke for `set_dense_exact` and `set_linear_landmark` at `(4,2)` and `(8,4)`.
3. Direct `L=8192` current-implementation smoke for dense and linear matched rows at `(4,2)` and `(8,4)`, seed 0 first.
4. If `L=8192` shows statistically meaningful gap narrowing after 5 seeds, continue with A8.0 candidate-gather quality reruns at `(4,2)` and `(8,4)`.
5. Test `(6,2)` as the first unobserved near-2 overlap variant only after the direct large-L result, unless spare compute makes it cheap to include.
6. A8.1 dense+linear curriculum at `L=512`, five seeds, ending first at `(4,2)` and `(8,4)`.
7. A8.2 dense+linear teacher distillation at `(4,2)` and `(8,4)`, five seeds.
8. A8.3 fixed-VRAM large-L dense+linear, five seeds, carrying forward whichever near-2/near-4 topology is better.
9. If at least one of A8.1-A8.3 narrows the gap while saving memory, launch A8.4 auxiliary compression objectives.
10. Run A8.5 as a diagnostic benchmark track before making any new performance claim.
11. Run A8.6 hybrid layers and A8.7 larger-data/fixed-token-budget only after the candidate-gather path is stable, unless the goal is explicitly exploratory rather than paper-bound.

Do not start broad sparse-family sweeps until dense and linear establish a positive signal.
