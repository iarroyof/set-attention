# A8 Hybrid Sparse Progressive Sweep

Date: 2026-06-14  
Status: PASS  
Host: `iarroyof@192.168.241.149` (`blue-demon`)  

## Rationale

This sweep tests the concern that stacked set layers may filter fine-grained token information before later layers can recover it. The implemented hybrid model uses one shared token residual stream: token layers and set layers both update the same `[B,L,D]` hidden sequence, so token information can bridge from early token layers into mid/end token or set layers. The set layers still compress positions with nonempty candidate fibers under `output_residual_mode=empty_only`, so this is an empirical test rather than an equivalence claim.

The first family is sparse/local-band because the post-A1 causal evidence showed it is the strongest non-dense matched family near token-equivalence:

- best nondense token baseline at `L=512`: `baseline_sparse_local_band`;
- best nondense set no-compression point: `set_sparse_local_band_empty_only` at `(w,s)=(1,1)`;
- linear/landmark had some favorable compressed points at `L=512`, but the `L=8192` 5-seed follow-up showed large quality degradation despite memory savings.

## Implementation

New model path:

- `src/models/hybrid_token_set_lm.py`
- `model.implementation: hybrid_token_set`

Config/run plumbing:

- `scripts/run_experiment.py`
- `src/config/normalize.py`
- `src/config/schema.py`
- `src/config/compatibility.py`

Configs:

- `configs/a8_hybrid/sparse_progressive_TTSSSS.yaml`
- `configs/a8_hybrid/sparse_progressive_TSTSTS.yaml`
- `configs/a8_hybrid/sparse_progressive_TTTTSS.yaml`
- `configs/a8_hybrid/README.md`

Config discipline:

- The YAML configs are the source of truth for model, data, and training
  hyperparameters.
- The launcher overrides only per-run provenance fields: seed, output directory,
  W&B run identity/project, and CSV path.
- This correction was applied after the initial launch because the first
  launcher duplicated model/data/training values as shell overrides. Existing
  artifacts can still be validated from their JSON metadata, but future resumes
  or reruns should use the config-first launcher.

Launcher and validator:

- `scripts/run_a8_hybrid_sparse_progressive.sh`
- `scripts/summarize_a8_hybrid_sparse_progressive.py`

Focused test:

- `tests/test_hybrid_token_set_lm.py`

## Matrix

Fixed:

- family: `hybrid_sparse_local_band`
- `D=384`
- `d_ff=1536`
- `L=512`
- `batch_size=16`
- `epochs=10`
- `lr=1e-4`
- `set_causality_mode=strict_past`
- `output_residual_mode=empty_only`
- set backend: `local_band`, `radius=4`
- pooling: `soft_trimmed_boltzmann`, `tau=0.1`, `q=0.85`
- feature mode: `hashed_counts`

Rows:

| Pattern | Interpretation | Set topologies | Seeds |
| --- | --- | --- | --- |
| `TTSSSS` | two token layers followed by four set layers | `4:2;4:2;8:4;8:4` | `0,1,2` |
| `TSTSTS` | alternating token/set bridge | `4:2;4:2;8:4` | `0,1,2` |
| `TTTTSS` | four token layers followed by two set layers | `4:2;8:4` | `0,1,2` |

Seeds `3,4` are intentionally held for follow-up approval after this first 3-seed pass.

## Verification Before Launch

Local:

- `python3 -m py_compile src/models/hybrid_token_set_lm.py scripts/run_experiment.py src/config/schema.py src/config/normalize.py src/config/compatibility.py tests/test_hybrid_token_set_lm.py scripts/summarize_a8_hybrid_sparse_progressive.py`
- `bash -n scripts/run_a8_hybrid_sparse_progressive.sh`
- `git diff --check` on edited hybrid/config/script/test files

Blue-demon Docker:

- Python compile of edited runtime/test files passed.
- `tests/test_hybrid_token_set_lm.py` direct execution passed.
- All three hybrid configs passed `scripts/run_experiment.py --dry-run`.
- CUDA forward smoke passed for all three configs and logged:
  - `resolved.d_phi=384`
  - `resolved.set_state_dim=384`
  - `resolved.adapter_type=linear`
  - `resolved.output_residual_mode=empty_only`
  - `resolved.hybrid_pattern`
  - `resolved.hybrid_set_topologies`

## Incident and Fix

The first launch attempt failed immediately for the two first `TTSSSS` rows. Cause:

```text
RuntimeError: The size of tensor a (127) must match the size of tensor b (255)
```

This came from using one shared `SetDiagnostics` object across multiple hybrid set layers with different bank sizes (`M=255` for `(w,s)=(4,2)` and `M=127` for `(w,s)=(8,4)`). The fix was scoped to the hybrid model: each `HybridSetLayer` owns its own diagnostics accumulator, and `HybridTokenSetLM.get_diagnostics()` averages same-named finite metrics across set layers. Set-only diagnostics were not changed.

Post-fix Docker checks:

- direct hybrid test passed;
- training-mode forward/backward smoke passed;
- aggregated diagnostics were finite and emitted expected `ausa/*` keys.

## Relaunch

Command:

```bash
cd ~/set-attention
nohup bash scripts/run_a8_hybrid_sparse_progressive.sh \
  > logs/a8_hybrid_sparse_progressive/nohup_launcher.log 2>&1 &
```

Wrapper PID printed: `3602947`.

First health check at 2026-06-14 12:06 CST:

- active worker/script PIDs present;
- GPU0 and GPU1 active at about `16574 MiB` each;
- first two CSV/JSON artifacts existed;
- status TSV reset and present;
- log scan found no OOM, traceback, standalone nonfinite token, W&B step issue, or permission denial;
- active logs showed both first runs had loaded cached WikiText-2 and entered training.

Per the monitoring rule, no repeated polling was performed after this health check.

## Expected Outputs After Completion

- Raw CSV/JSON root: `out/paper_mechanisms/a8_hybrid_sparse_progressive/`
- Logs: `logs/a8_hybrid_sparse_progressive/`
- Status TSV: `out/paper_mechanisms/a8_hybrid_sparse_progressive/a8_hybrid_sparse_progressive_status.tsv`
- All runs TSV: `out/paper_integrated_evidence/tables/a8_hybrid_sparse_progressive_all_runs.tsv`
- Summary TSV: `out/paper_integrated_evidence/tables/a8_hybrid_sparse_progressive_summary.tsv`
- Manifest: `out/paper_integrated_evidence/checks/a8_hybrid_sparse_progressive_manifest.json`

## Completion Validation

Validated on 2026-06-14 after user reported GPUs were free.

- No active launcher or experiment process remained.
- GPU0 and GPU1 were idle.
- Raw artifacts: `9` CSVs and `9` JSONs.
- Status TSV had `9/9` rows with exit code `0`.
- Log scan found no OOM, traceback, standalone nonfinite token, W&B step issue, or permission denial.
- `scripts/summarize_a8_hybrid_sparse_progressive.py` passed after a validator-only patch to compare `training.lr` numerically; the completed launch stored the LR as string `'1e-4'`, while future config-first launches may store it as numeric `0.0001`.

Summary artifact hashes:

| Artifact | SHA256 |
| --- | --- |
| `out/paper_integrated_evidence/tables/a8_hybrid_sparse_progressive_all_runs.tsv` | `ce5d07056a2db214b6fc73ae8f9da27648e879c39537221a0bbb81935ec3ea55` |
| `out/paper_integrated_evidence/tables/a8_hybrid_sparse_progressive_summary.tsv` | `7eeffd250962f2629552eb851f77a6caf933c612cfd90499a5964bca31440a42` |
| `out/paper_integrated_evidence/checks/a8_hybrid_sparse_progressive_manifest.json` | `9bb00e0b816f5df5ec94ddb2914b674babb5b90cee3fb284c262d131d20d4c59` |

## Result Summary

| Pattern | Set topologies | n | Mean val PPL | Std val PPL | 95% CI half-width | Mean train PPL | Mean peak VRAM MiB | Mean sec/epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TTSSSS` | `4:2;4:2;8:4;8:4` | 3 | 2623.9402 | 65.7246 | 74.3744 | 946.2727 | 13407.0723 | 267.6816 |
| `TSTSTS` | `4:2;4:2;8:4` | 3 | 2576.8607 | 27.3868 | 30.9910 | 911.7574 | 13517.1582 | 240.2123 |
| `TTTTSS` | `4:2;8:4` | 3 | 2393.3585 | 30.2067 | 34.1821 | 881.4876 | 13401.1309 | 150.9459 |

## Interpretation

The shared-token-stream hybrid implementation works and the experiment completed cleanly, but this sparse progressive operating point does not recover quality. `TTTTSS` is the best of the three tested patterns, which supports the intuition that preserving more early token layers is less harmful, but its mean validation PPL remains far above the matched sparse token and near-token set operating points from A7/A2.4.

Recommendation: do not spend seeds `3,4` on this exact hybrid sparse progressive sweep unless a confirmatory negative result is explicitly desired. The next higher-value engineering path remains reducing the dense token-to-set routing memory overhead via candidate-gather routing, or testing memory/retrieval-favorable tasks rather than broadening these poor-quality hybrid patterns.
