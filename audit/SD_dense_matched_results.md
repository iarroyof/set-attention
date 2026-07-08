# Exact-Dense Multiresolution Matched Results

Status: primary three-replicate matrix complete; legacy OOM exclusivity caveat
added.

Verified: 2026-07-02.

## Contract

- Exact dense backend for both set dictionary and token attention.
- Full WikiText-2, no `data.limit`, 10 registered epochs, seeds 0--2.
- `D=384`, `d_ff=1536`, 6 layers, 8 heads, LR `1e-4`, warmup metadata `1000`.
- Set rows use `anchor_span`, token MLP off, anchor off, CE-only, endpoint-window.
- B16 and B4 are separate comparison islands.

The L512/B16 token seed 0 is reused from the protocol-matched 30-epoch convergence run at its epoch-10
row. This is valid because architecture/data/batch/seed/LR/warmup match and the trainer has no
epoch-count-dependent scheduler. Seeds 1--2 are the corrected grid runs.

LR-schedule audit: `scripts/run_experiment.py` constructs one `torch.optim.AdamW` with the configured
constant LR and never constructs or steps an LR scheduler. `training.warmup_steps` is logged metadata
only in this trainer path; it does not alter optimization. `src/train/loop.py` calls
`optimizer.step()` once per batch, and the configured total epoch count is used only as the outer-loop
stop bound. Therefore the first 10 epochs of the 30-epoch seed-0 run follow the same optimization
schedule as a 10-epoch run under the matched seed/data/config. Reuse is restricted to its epoch-10 row.

## Validation

- Blue-demon corrected queue: 8/8 cells completed with 10 epochs and exit 0.
- Lizmark corrected queue: L2048 token 3/3 completed; L4096 token 3/3 OOM before epoch 1.
- Completed token metadata failures: 0.
- Unexpected NaN/Inf/traceback findings: 0.
- Exact token rows have absent/empty `backend_params`; no landmark config reached training.
- Every completed row contains independent `train/peak_vram_mib`.

The earlier config-error attempts produced zero epochs and are excluded.

The `L4096/B4` OOMs occurred on 2026-06-26 (set b0/b25) and 2026-06-29
(token), before the unrelated 2026-07-02 dual-GPU process documented in
`audit/incident_lizmark_gpu_contention_20260702.md`. That incident did not
cause these OOMs. The legacy launchers did not archive external-process
telemetry, so the 3/3 outcomes are repeated observed feasibility results, not
retrospectively certified exclusive-capacity measurements.

## Matched comparison

PPL uncertainty is a two-sided 95% t interval over three seeds. VRAM is mean peak train memory.

| Island | Lowest-PPL feasible set | Set PPL | Token PPL | Set-token PPL | Set VRAM MiB | Token VRAM MiB | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| L512/B16 | b25 | 860.629 +/- 60.954 | 800.508 +/- 62.712 | +60.121 | 13790.5 | 13415.3 | token dominates b25 on both means |
| L512/B4 | b25 | 961.859 +/- 58.559 | 1023.556 +/- 152.877 | -61.697 | 4074.8 | 3981.9 | quality/memory tradeoff; PPL intervals overlap |
| L1024/B4 | b25 | 931.052 +/- 92.984 | 934.909 +/- 60.654 | -3.857 | 8037.4 | 8001.6 | effectively tied; b25 costs 35.7 MiB |
| L2048/B4 | b25 | 920.105 +/- 69.890 | 921.377 +/- 55.600 | -1.272 | 18097.3 | 18633.3 | mean-level Pareto point; quality difference unresolved |
| L4096/B4 | b50 | 900.563 +/- 25.283 | OOM 3/3 | -- | 41369.5 | OOM 3/3 | set feasibility win; no token-quality comparison |

At L4096, set b0 and b25 also OOM 3/3. Set b50/b62/b75/b100 completed. Therefore mixed/coarse
resolution is what moves the exact-dense frontier past the memory ceiling; this is stronger than a
claim based on a landmark approximation.

## Interpretation

1. Mixed resolution is a valid set-vs-set mechanism: b25 is the lowest-PPL set row through L2048 and
   uses less VRAM than all-fine.
2. Dense set dictionary does not establish a consistent PPL advantage over dense token attention.
   The B4 mean differences at L1024/L2048 are negligible relative to seed uncertainty.
3. The defensible cross-family result is a legacy observed
   memory-feasibility result: at L4096/B4, dense token and fine-heavy set rows
   OOM, while b50 and coarser set rows train successfully. New terminal
   capacity claims require the corrected admission telemetry.
4. Batch materially changes PPL at L512. B16 and B4 must remain separate; neither can calibrate the
   other's absolute quality.
5. Coverage-scaled landmark results are unnecessary for this conclusion and remain historical only.

## Five-seed confirmation

The earlier winner-only recommendation is superseded by
`docs/sd_dense_paper5_matrix.md`. The active paper matrix five-seeds
`{b0,b25,b50,b75,b100}` plus exact token at every supported island, reusing all valid completed seeds.

At L4096/B4, token/b0/b25 remain repeated 3/3 legacy OOM outcomes; supported
b50/b75/b100 are topped up.
Existing b62 rows remain exploratory and are not part of the regular five-row comparison.
