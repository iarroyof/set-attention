# A7 Empty-Only Calibration Plan

Status: feature implemented; long experiments not launched.

## Motivation

The calibrated convergence question is empirical convergence as `M/L -> 1`, while avoiding architecture differences that would prevent a clean comparison. Under strict-past endpoint routing, the first sealed endpoint is at position `w`, so exactly the first `w-1` positions have `C_t=0`. Therefore `P(C_t=0)=(w-1)/L` under a uniform token-position view. In the singleton limit `w=1,s=1`, `M=L` and `P(C_t=0)=0`.

This supports a named output policy:

- `direct`: existing A1-A6 behavior, `h_t^(0) + r_t` in strict-past mode.
- `empty_only`: `h_t^(0)` only for `C_t=0`, otherwise `r_t`.
- `none`: `r_t` for all strict-past positions.

Use `empty_only` for calibrated token-limit and compression-limit experiments because it keeps early empty-fiber tokens defined without adding a direct token residual when routed set information exists.

## Experiment-Design Directives

1. Keep A1-A6 results reproducible under the default `direct` architecture.
2. Treat A7 as a separate calibration extension, not as a replacement for completed causal results.
3. Study empirical convergence as `M/L -> 1`; do not claim exact Transformer equality unless all architectural differences are removed or proved irrelevant.
4. Avoid architecture confounds in the token-limit setting: use explicit named knobs rather than silent code changes.
5. Use matched token baselines under the same `D`, `d_ff`, `L`, learning rate, seed, epoch budget, and backend family.
6. Log and report structural diagnostics with performance: `M`, `M/L`, compression `L/M`, candidate-count mean/max, fraction `C_t=0`, routing entropy/top1, pooling effective support, VRAM, and time.
7. Separate calibration/token-limit artifacts from ordinary compressed SKA artifacts in paths and names.

## A7.0 Smoke Gate

Run before long grids:

- Config parse with `model.output_residual_mode in {direct, empty_only, none}`.
- Tiny strict-past forward for `w=1,s=1`, `allow_token_token=true`, `output_residual_mode=empty_only`.
- Confirm `resolved.output_residual_mode` is logged.
- Confirm `empty_only == none` at `w=1,s=1`.
- Confirm `direct == empty_only + h_t^(0)` when all tokens have nonempty singleton candidates.

## A7.1 Token-Limit Calibration

Goal: test the no-compression/singleton-set limit while keeping the set pipeline explicit.

Fixed:

- `L=512`
- `w=1`
- `s=1`
- `M=L`
- `allow_token_token=true`
- `output_residual_mode=empty_only`
- `pooling.mode=mean`
- `token_mlp.enabled=false`
- `feature_mode=geometry_only`
- `geometry.enabled=false`
- `set_state_dim=d_model`
- `d_phi=d_model`
- `backend=exact`
- `set_causality_mode=strict_past`

Families:

- `baseline_dense_exact`
- `set_dense_exact_calibrated_empty_only`

Run policy:

- Start with seed 0, best LR from the matched dense A2/A2.4 summaries.
- If stable and plausible, run seeds 0,1,2.

Expected interpretation:

- This is not literal equality to the baseline Transformer because the computation still goes through set pooling, set-attention blocks, and routing projections.
- It is the cleanest empirical limit available without implementing a new exact-equivalence architecture.

## A7.2 Compression Path Sweep

Goal: show performance and diagnostics as compression increases away from `M/L=1`.

Fixed:

- `D=384`, `d_ff=1536`
- `L=512`
- `strict_past`
- `output_residual_mode=empty_only`
- exact backend first
- best LR per family from prior summaries unless a fixed-LR ablation is explicitly desired
- 10 epochs

Sweep:

- `(w,s) in {(1,1), (2,1), (4,2), (8,4), (16,8), (32,16)}`

This keeps approximately `w/s=2` after the singleton endpoint and changes `M/L` from `1` downward. Report `M=floor((L-w)/s)+1` for every point.

Families:

- Required: `baseline_dense_exact`, `set_dense_exact`
- Optional if compute permits: matched sparse and landmark families with their token controls.

Seeds:

- Seed 0 for all points first.
- Then seeds 0,1,2 for the most informative points: `(1,1)`, `(4,2)`, `(16,8)`, `(32,16)`.

## A7.3 Window/Stride Factorization

Goal: distinguish compression rate from candidate-count effects.

Two controlled views:

- Fixed stride, vary window: reuse the A3.1 pattern under `empty_only`.
- Fixed compression target, vary `w/s`: choose pairs with similar `M/L` but different candidate count.

Candidate pairs for fixed `L=512`:

- Similar compression near `M/L ~= 0.5`: `(w=2,s=2)`, `(w=4,s=2)`, `(w=8,s=2)`.
- Similar compression near `M/L ~= 0.25`: `(w=4,s=4)`, `(w=8,s=4)`, `(w=16,s=4)`.

This tests whether degradation tracks compression `M/L`, routing support `C_t`, or pooling window size.

## Deliverables

Suggested artifact names:

- `out/paper_integrated_evidence/tables/a7_empty_only_token_limit_all_runs.tsv`
- `out/paper_integrated_evidence/tables/a7_empty_only_token_limit_summary.tsv`
- `out/paper_integrated_evidence/checks/a7_empty_only_token_limit_manifest.json`
- `out/paper_integrated_evidence/tables/a7_compression_path_all_runs.tsv`
- `out/paper_integrated_evidence/tables/a7_compression_path_summary.tsv`
- `out/paper_integrated_evidence/checks/a7_compression_path_manifest.json`
- `audit/A7_empty_only_calibration.md`

Do not launch the full A7 grid until the A7.0 smoke gate passes and the run matrix is confirmed.

## Confirmed Launch Matrix

This section records the matrix selected for the first A7 launch.

Baseline comparison provenance:

- Reuse the validated A2 dense token baseline rows, not a new baseline run.
- Required match keys: `family=Baseline token`, `backend=exact`, `D=384`, `d_ff=1536`, `L=512`, `epochs=10`, `lr=1e-4`, `seeds={0,1,2}`, Wikitext-2 LM, causal token attention.
- Source summary: `out/paper_integrated_evidence/tables/a2_lrnorm_headline_all_runs.tsv`.
- Source run artifacts are the `out/paper_lr_norm/paper_lr_norm_headline_A2_D384_FF1536/paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed*.{csv,json}` rows recorded in that summary.

Set-side rows to run:

- Family: `set_dense_exact` only for the first calibrated convergence test.
- `D=384`, `d_ff=1536`, `L=512`, `num_layers=6`, `num_heads=8`, `epochs=10`, `lr=1e-4`.
- `set_causality_mode=strict_past`, `output_residual_mode=empty_only`.
- `pooling.mode=mean`, `feature_mode=geometry_only`, `geometry.enabled=false`, `token_mlp.enabled=false`.
- Learned router remains active. With `feature_mode=geometry_only`, router descriptors are learned set-position embeddings; geometry bias and content-bias adapters are disabled for this calibration track.
- `d_phi=384`, `set_state_dim=384`.
- `backend=exact`; `allow_token_token=true` only for `w=1,s=1`.

Compression path rows:

| `w` | `s` | seeds | Reason |
| ---: | ---: | --- | --- |
| 1 | 1 | 0,1,2 | singleton token-limit, `M/L=1`, no empty fibers |
| 2 | 1 | 0 | high-resolution interior point |
| 4 | 2 | 0,1,2 | moderate high-resolution point |
| 8 | 4 | 0 | interior point |
| 16 | 8 | 0,1,2 | LR-norm headline compression point, but with `empty_only` |
| 32 | 16 | 0,1,2 | stronger compression endpoint |

Expected new set-side run count: 14.

Primary outputs:

- Raw CSV/JSON: `out/paper_mechanisms/a7_empty_only_calibration/`.
- Logs: `logs/a7_empty_only_calibration/`.
- Summary: `out/paper_integrated_evidence/tables/a7_empty_only_calibration_all_runs.tsv`.
- Aggregates: `out/paper_integrated_evidence/tables/a7_empty_only_calibration_summary.tsv`.
- Manifest: `out/paper_integrated_evidence/checks/a7_empty_only_calibration_manifest.json`.
- Audit: `audit/A7_empty_only_calibration.md`.
