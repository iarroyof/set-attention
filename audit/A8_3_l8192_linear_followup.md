# A8.3 L=8192 Linear Follow-Up

Date: 2026-06-13  
Status: PASS  
Host: `iarroyof@192.168.241.205` (`lizmark`)  
Source HEAD: `1c68dfa9b86b351007da285252ef823fe87c9227`  

## Rationale

The A8.3 seed-0 `L=8192` smoke passed for all matched dense/linear rows. The only row with both a strong memory reduction and a lower one-epoch smoke PPL than its matched token backend was `set_linear_landmark` at `(w,s)=(8,4)`. This follow-up is therefore intentionally narrow: it tests that operating point against the matched `baseline_linear_landmark` control with five seeds.

This is not a broad A8 grid and does not launch candidate-gather routing, curriculum, distillation, or dense-family reruns.

## Matrix

Fixed:

- `L=8192`
- `D=384`
- `d_ff=1536`
- `batch_size=1`
- `epochs=10`
- `lr=1e-4`
- full cached WikiText-2 (`data.limit` not set)
- seeds `{0,1,2,3,4}`

Rows:

| Family | Backend | Window | Stride | Causality | Output residual | Landmark coverage | Seeds |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| `baseline_linear_landmark` | `landmark` | NA | NA | causal token LM | NA | 0.25 | 0,1,2,3,4 |
| `set_linear_landmark` | `landmark` | 8 | 4 | `strict_past` | `empty_only` | 0.25 | 0,1,2,3,4 |

For the set row, post-T1 topology at `L=8192,w=8,s=4` is `M=floor((8192-8)/4)+1=2047`; landmark count resolves to `round(0.25*M)=512`.

## Launch

Launcher: `scripts/run_a8_l8192_linear_followup_lizmark.sh`

Command pattern:

```bash
cd ~/set-attention
SOURCE_HEAD=1c68dfa9b86b351007da285252ef823fe87c9227 \
  nohup bash scripts/run_a8_l8192_linear_followup_lizmark.sh \
  > logs/a8_l8192_linear_followup/nohup_launcher.log 2>&1 &
```

Launcher PID printed by SSH: `95803`.

Worker PIDs at first health check:

- wrapper: `95805`
- GPU0 worker: `95810`
- GPU1 worker: `95811`
- active GPU0 docker process at health check: `95822`
- active GPU1 docker process at health check: `95823`

Initial active rows:

- GPU0: `baseline_linear_landmark_L8192_seed0`
- GPU1: `set_linear_landmark_L8192_w8_s4_seed1`

## First Health Check

Time: 2026-06-13 17:27 CST

- GPU0: `29805 MiB` used, `100%` utilization.
- GPU1: `16637 MiB` used, `15%` utilization.
- Initial CSV/JSON files existed for both active rows.
- Status TSV existed at `out/paper_mechanisms/a8_l8192_linear_followup/a8_l8192_linear_followup_status.tsv`.
- First warning scan found no OOM, traceback, nonfinite metric token, W&B step issue, or permission denial.

Per the monitoring rule, no repeated polling should be done. Wait for explicit user notification before validation/sync/summarization.

## Expected Artifacts

- Raw CSV/JSON root: `out/paper_mechanisms/a8_l8192_linear_followup/`
- Logs: `logs/a8_l8192_linear_followup/`
- Status TSV: `out/paper_mechanisms/a8_l8192_linear_followup/a8_l8192_linear_followup_status.tsv`
- Prelaunch JSON on lizmark: `audit/A8_3_l8192_linear_followup_prelaunch.json`

## Completion Validation

Completed on 2026-06-13 and validated on 2026-06-14.

- All 10 rows exited `0`.
- Every CSV has 10 epochs.
- Final `train/loss`, `val/loss`, `train/ppl`, `val/ppl`, `train/peak_vram_mib`, and `train/time_per_epoch_s` are finite.
- Log scan found no OOM, traceback, standalone nonfinite token, W&B step issue, or permission denial.
- JSON metadata matched:
  - `data.seq_len=8192`
  - `data.batch_size=1`
  - `training.seed in {0,1,2,3,4}`
  - `model.backend=landmark`
  - `model.backend_params.landmark_coverage=0.25`
  - set rows have `model.window_size=8`, `model.stride=4`, `model.set_causality_mode=strict_past`, `resolved.output_residual_mode=empty_only`
  - baseline landmark rows have `resolved.landmark_count=2048`
  - set landmark rows have `resolved.landmark_count=512`

Validation/summarization script:

- `scripts/summarize_a8_l8192_linear_followup.py`

Summary artifacts:

- All runs TSV: `out/paper_integrated_evidence/tables/a8_l8192_linear_followup_all_runs.tsv`
- Summary TSV: `out/paper_integrated_evidence/tables/a8_l8192_linear_followup_summary.tsv`
- Manifest: `out/paper_integrated_evidence/checks/a8_l8192_linear_followup_manifest.json`

## Final Summary

| Family | n | Window | Stride | M | Landmark count | Mean val PPL | Std val PPL | 95% CI half-width | Mean train PPL | Mean peak VRAM MiB | VRAM ratio vs baseline | Mean sec/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_linear_landmark` | 5 | NA | NA | NA | 2048 | 1048.4202 | 81.9369 | 71.8209 | 321.7302 | 25785.1895 | 1.0000 | 107.1446 |
| `set_linear_landmark` | 5 | 8 | 4 | 2047 | 512 | 2181.3303 | 56.0769 | 49.1536 | 668.7101 | 12982.2896 | 0.5035 | 253.1671 |

## Interpretation

The 5-seed result does not confirm the favorable one-epoch smoke quality signal. At `L=8192`, `set_linear_landmark` with `(w,s)=(8,4)` uses about half the peak VRAM of the matched linear token baseline, but its mean validation PPL is worse by `+1132.91`.

This supports the memory-compression side of A8 but not the near-baseline-quality side. The one-epoch smoke was therefore a noisy fit/provenance signal, not a reliable quality predictor.

Recommended next step: do not broaden current-implementation `L=8192` quality grids. Prioritize A8.0 candidate-gather routing or a task that rewards compressed memory/retrieval before spending more compute on the same architecture/training recipe.
