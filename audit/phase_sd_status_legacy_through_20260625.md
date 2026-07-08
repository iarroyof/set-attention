# HISTORICAL Set-Dictionary Branch Status Tracker (through 2026-06-25)

This snapshot is retained for provenance only. It contains stale live PIDs, landmark-era queues, and
pre-pivot instructions. Do not use it to decide what to run. The current authoritative tracker is
`audit/phase_sd_status.md`.

Last updated: 2026-06-23 22:15 CST by raptor-mini — homogenizer re-run and CSV verification performed; no STATUS flips applied.

Scope: `set-dictionary/anchor-span` branch for the causal set-dictionary revision plan v3.0.
This tracker follows the same transition discipline as `audit/phase_a_status.md`.

## Prerequisite State

- A9 candidate-gather validation is experimentally complete in `audit/phase_a_status.md` and
  `audit/A9_candidate_gather.md` (`Status: PASS`, validated_runs=18/18).
- Blocking prerequisite is recorded locally as commit `483464a` on branch
  `a9/candidate-gather-router`.
- Current branch: `set-dictionary/anchor-span`, created from `483464a`.
- Set-dictionary implementation may proceed from SD-1, subject to the v3.0 read discipline.

## Summary

| Phase | Task | Status | Runs | Audit / Notes |
| --- | --- | --- | --- | --- |
| SD-0.1 | A9 candidate-gather prerequisite commit (`a9/candidate-gather-router`) | ✅ DONE | — | Local commit `483464a` (`Add A9 candidate-gather router prerequisite`) records the validated A9 state. A9 PASS remains recorded in `audit/phase_a_status.md` and `audit/A9_candidate_gather.md`. |
| SD-0.2 | Create `set-dictionary/anchor-span` from the A9 commit tip | ✅ DONE | — | Branch `set-dictionary/anchor-span` created from `483464a` on 2026-06-16 18:45 CST. No merge to `main`. |
| SD-1 | D-causal flag collapse (`causal` derived from `set_causality_mode`) | ✅ DONE | — | `audit/SD_1_d_causal.md` — Status: PASS. Container checks passed on blue-demon for output residual modes, config propagation, and dense/sparse/landmark causality probes. |
| SD-2 | §4 numerical leakage probe for `anchor_span` and anchor pre-encoder path | ✅ DONE | — | `audit/SD_2_anchor_span_causality.md` — Status: PASS. `anchor_span` formula, causal pre-encoder target, config propagation, training aux-loss smoke, and dense/sparse/landmark causality probes passed in blue-demon container. |
| SD-3 | §5 fairness audit harness (parameter parity, inference/train VRAM split, span ablation) | ✅ DONE | — | `audit/SD_3_fairness_harness.md` — Status: PASS. Harness validates thin-anchor ablation, token-MLP exclusion for `anchor_span`, anchor pre-encoder exclusion from inference parameter count, reference parameter counts, and train/inference VRAM smoke accounting. |
| SD-4 | Config contract for set-dictionary keys | ✅ DONE | — | Landed with SD-2. New `anchor`, `set_diversity`, `multivector_basis`, and `candidate_fiber` keys are normalized, schema/compatibility-checked, documented in `configs/hyperparameters.md`, emitted by `get_resolved_metadata()`, and included in CSV metadata/fingerprints. |
| SD-5 | S1 ladder: `anchor_span`, anchor disabled, CE only, dense backend | ✅ DONE | Smoke 1/1 PASS; full 6/6 PASS | `audit/SD_5_s1_anchor_span_dense.md` — Status: PASS as a validated run package, but the result is null/negative. S1 mean val PPL: `(4,2)` = 1297.866 vs old set dense empty_only ref 1273.6; `(16,8)` = 1510.900 vs old SKA dense direct ref 1422.8 and token baseline 781.1. Span ablation sharply worsened PPL, so the span is carrying prediction, but S1 does not satisfy the adoption gate. |
| SD-6 | S2 ladder: S1 + causal pre-encoder anchoring (`lambda_h=0.1`) — **pre-registered rescue, gated decision** | ✅ DONE | Full 6/6 PASS (validated 2026-06-17) | `audit/SD_6_s2_anchoring.md`, manifest `sd6_s2_anchoring_manifest.json`, verdict `sd6_s2_anchoring_verdict.tsv` (all synced local). **Result: anchoring did NOT rescue the bottleneck.** `recon_error_norm` floors at ~1.36–1.39 (>1: span barely matches the predictive target) and is essentially flat (slope ~−0.002/epoch) at both topologies → the loss is not reducing reconstruction error. Verdict: **(16,8) = Branch B (capacity-limited)** — PPL 1510.9→1437.2 (Δ−73.7, within combined CI 122, not significant), recon high+flat → SD-8. **(4,2) = MIXED** — PPL 1297.9→1276.6 (Δ−21.3 beats tight CI 17.8) BUT recon high+flat, so it is a marginal regularization gain, not a signal-limited rescue. Both point to span capacity, not learning signal, as the binding constraint (consistent with S1). **⚠️ CONFOUND FOUND (post-hoc code audit): the verdict is INVALID.** The anchor target is produced by `CausalPreEncoder`, which has no head/loss, is not in the forward path, and is fed a `detach()`ed target in `_update_anchor_loss` (`set_only_lm.py:585-587`) — so it receives zero gradient and stays at random init. `h*` is a fixed random projection; `recon_error_norm≈√2` (cosine≈0.03) is the uncorrelated-vectors fingerprint. The anchoring rescue was never validly tested. Do NOT treat "capacity-limited" as established. |
| SD-6.5 | **Fix the anchor target**: give `CausalPreEncoder` its own causal LM head + auxiliary CE (`L_CE_pre`, e.g. `lambda_pre=1.0`), keep `detach_target=true`, training-only/excluded from inference; rerun the S2 ladder | ✅ DONE | Smoke 1/1 PASS; full 6/6 PASS | `audit/SD_6_5_s2_anchoring_fixed.md`, manifest `out/paper_integrated_evidence/checks/sd6_5_s2_anchoring_fixed_manifest.json`, verdict `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_verdict.tsv` (all synced local). Blue-demon checks passed: py_compile, SD-2 leakage + pre-encoder-logit causality + nonzero pre-encoder gradient, hyperparameter propagation, output-residual tests, SD-3 fairness harness, dry-runs for `(16,8)`, `(4,2)`, and smoke. Fixed S2 is VALID under the anchor guard: final recon is below 1.2 at both topologies (`(16,8)`=1.176980, `(4,2)`=1.184113), so the random-target confound is resolved. Verdict: **Branch B for both topologies**. `(16,8)` PPL 1510.899740→1510.373413 (Δ−0.526327, within combined CI 100.834607); `(4,2)` PPL 1297.866252→1288.973145 (Δ−8.893107, within combined CI 75.443707). Span-ablation remains large (`~40k–44k` ΔPPL), so no token bypass. Recommended next step for user review: **SD-8 = D-fiber all_past CE-only first; r=2 only secondary**. No SD-7, SD-8, lambda_h=1.0, wider fiber, or multivector follow-up was launched. |
| SD-7 | Follow-up winner-only knobs (`lambda_h=1.0`, then `lambda_div>0`) | ⏳ PENDING | — | **Only if SD-6.5 reruns classify Branch A.** Do not launch otherwise. |
| SD-8 | Capacity pivot: D-fiber `all_past` (CE-only first), then `window_plus_landmarks`; `multivector_basis.r=2` only as a secondary floor test | ✅ DONE | Candidate-gather smoke 1/1 PASS; candidate-gather full OOM; dense-router smoke 1/1 PASS; dense-router full 6/6 PASS | `audit/SD_8_all_past_dense_routerdense.md`, manifest `out/paper_integrated_evidence/checks/sd8_all_past_dense_routerdense_manifest.json`, run table `out/paper_integrated_evidence/tables/sd8_all_past_dense_routerdense_runs.tsv`, summary table `out/paper_integrated_evidence/tables/sd8_all_past_dense_routerdense_summary.tsv`. The first full run OOMed with `router.score_mode=candidate_gather`; `audit/incident_sd_8_20260618.md` is RESOLVED by a clean dense-router rerun. Result: **validated PASS package, mixed capacity result rather than adoption PASS.** `(16,8)` mean PPL `1363.931559` improves materially vs S1/fixed S2 (`~1510`) and old SKA dense direct ref `1422.8`, but remains far above dense token baseline `781.1`. `(4,2)` mean PPL `1288.603190` is essentially flat vs fixed S2 `1288.973145` and worse than old set dense empty_only ref `1273.6`. Span ablations remain large (`~48.9k` and `~62.7k` ΔPPL), so prediction remains span-carried. No `window_plus_landmarks`, `r=2`, SD-7, `lambda_h=1.0`, or multivector follow-up launched. |
| SD-8.1 | User-requested capacity check: `(4,2)` all_past CE-only with doubled dictionary atom width (`d_phi=set_state_dim=768`) | ✅ DONE | Smoke 1/1 PASS; full 3/3 PASS | `audit/SD_8_1_dphi768_w4s2.md`, manifest `out/paper_integrated_evidence/checks/sd8_all_past_dense_dphi768_w4s2_manifest.json`, run table `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_runs.tsv`, summary table `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_summary.tsv`, comparison table `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_comparison.tsv`. Result: mean val PPL `1241.522583` (std `44.528320`), mean peak train VRAM `12764.080078` MiB, mean span-ablation ΔPPL `71758.745646`. Compared with best SD `(4,2)` so far (SD-8 all_past `d_phi=setdim=384`: PPL `1288.603190`, VRAM `11913.547852` MiB), doubled width improves PPL by `47.080607` but costs `850.532226` MiB. Compared with old Set Dense empty_only ref (`1273.6` PPL, `11807.3` MiB), it improves PPL by `32.077417` but costs `956.780078` MiB. Compared with dense token baseline (`781.109436` PPL, `13407.220703` MiB), it is still `460.413147` PPL worse while using `643.140625` MiB less VRAM. |
| SD-9 | **Multi-resolution (mixed-blur) frontier test** — heads split into fine `(2,1)` + coarse/blurred `(4,2)` groups at the same depth. `%blur` = coarse-head fraction. **Short: L=512 on blue-demon, dense exact, batch=16, 25% coarse (6 fine + 2 coarse). Long: L=8192 on lizmark, LANDMARK backend (coverage 0.25), batch=1, 65% coarse (3 fine + 5 coarse), matching A8.3 `audit/A8_3_l8192_linear_followup.md`.** Baseline = the two uniform extremes (all-fine `(2,1)`, all-coarse `(4,2)`) per context → PPL–memory Pareto check. CE-only, endpoint_window fiber, 3 seeds. **Two-server concurrent.** | ✅ DONE | Smoke 2/2 PASS; full 18/18 PASS | `audit/SD_9_multiresolution.md`; manifests `out/paper_integrated_evidence/checks/sd9_multiresolution_short_manifest.json` and `out/paper_integrated_evidence/checks/sd9_multiresolution_long_manifest.json`. Completed 2026-06-20: short 9/9 exit 0 on blue-demon, long 9/9 exit 0 on lizmark; no SD-9 processes remain and GPUs idle. Guard confirmed: YAML and launcher pin `output_residual_mode=anchor_span`; summarizer asserts `model.output_residual_mode` and `resolved.output_residual_mode` are `anchor_span`, plus `token_mlp.enabled=false`, `anchor.enabled=false`, and `candidate_fiber=endpoint_window`. Verdict: mixed improves PPL vs fine↔coarse interpolation in both contexts but uses more VRAM than interpolation, so neither context is Pareto-better. Short mixed-25: PPL `862.1083` vs interp `1001.6285`, VRAM `13790.4800` vs interp `13435.5135` MiB. Long mixed-65: PPL `1008.9868` vs interp `1282.0271`, VRAM `20192.8843` vs interp `19948.0794` MiB. **Claude reframe (2026-06-20): the registered verdict is vs the synthetic interpolation; against the ACHIEVABLE all-fine endpoint, mixed Pareto-DOMINATES on BOTH axes at BOTH contexts** — short all-fine 912.9/13933 → mixed 862.1/13790 (−51 PPL, −143 MiB, super-additive; span-abl Δ 58368→64489 = coarse heads carry more used prediction); long all-fine 1033.1/27928 → mixed 1009.0/20193 (−24 PPL AND −28% VRAM). Multi-scale hypothesis SUPPORTED; long-context coarse-head win de-risks SD-10. Actionable: scale L (16k/32k), global-dependency/recall tasks, per-head-group ablation + receptive-field probes, learned fine/coarse gate. |
| SD-9.5 | **Mechanism probes on the current winner** (plan §7d): per-head-group span-ablation (fine vs coarse), effective-range probe, token-type stratified loss, and scale-L sweep L∈{16k,32k} (lizmark, landmark, smoke first). Reuse SD-9 checkpoints if saved, else retrain mixed seeds with eval instrumentation. | 🚧 RUNNING | blue short 9/9 validated; lizmark `L=12288` PID 621175 on GPU0 and `L=8192` PID 625216 on GPU1 | 2026-06-20: SD-9 checkpoints checked locally, blue-demon, and lizmark; none found, so probe retraining path selected. Implemented eval-only group span ablations, effective-range probe, token-type stratified val loss, `scripts/run_sd9_5_probes.sh`, and `scripts/summarize_sd9_5_probes.py`. Local `py_compile`/`bash -n` PASS; runnable local smoke blocked because this host has namespace-only `torch` and no Docker. Synced to blue-demon and lizmark; remote syntax/hash checks PASS. Blue short full probe matrix completed 9/9 exit 0 and artifacts/logs were pulled local; local `python scripts/summarize_sd9_5_probes.py --mode short` validated 9/9, no log findings, guard metadata accepted. Lizmark `L=32768` mixed-65 smoke OOMed under landmark coverage 0.25, batch 1 (`audit/incident_SD_9_5_scaleL_32768_oom.md`). Lizmark `L=16384` full scale: mixed-65 and all-fine OOMed under the same guard; all-coarse completed 10/10 (`audit/incident_SD_9_5_scaleL_16384_full_oom.md`). Per user direction after mixed long OOM, launched lower mixed-only full runs with landmark coverage 0.25, batch 1, `VARIANTS=mixed`: `L=12288` on GPU0 (~41.9 GiB/49.1 GiB, active, no epoch row yet) and `L=8192` on GPU1 (~23.3 GiB/49.1 GiB, active, no epoch row yet). Guards remain `output_residual_mode=anchor_span`, `token_mlp.enabled=false`, `anchor.enabled=false`, CE-only, endpoint_window. Output `audit/SD_9_5_probes.md` after fallback completes or fails. |
| SD-9.6 | **Long-context multiresolution support/combination probes** requested after SD-9.5 OOMs: use the SD-9/9.5 instrumentation and landmark backend to find supported operating candidates on 24GB GPUs for corrected blue interval `L in (8192 downto 2048] = {4096,2048}`, plus lizmark recovery for blue-failed L=8192 rows. | ✅ PROBE DONE / ANALYSIS READY | blue primary seed-0 support+blur probes 12/12 exit 0; lizmark recovery 3/3 exit 0; both hosts idle | `audit/SD_9_6_blue_long_multiresolution.md`, `audit/SD_9_6_long_context_multiresolution_plan.md`, `scripts/run_sd9_6_blue_long_multires_queue.sh`. Correction: initial launch mistakenly included `L=8192`; since `L=8192` is lizmark-owned, blue `L=8192` rows are retained only as capacity sanity checks and excluded from blue operating-point selection. Primary corrected blue rows at `L=4096,2048` used landmark coverage 0.25, batch 1, seed 0, **`data.limit=500`**; support rows `mixed65/all_fine/all_coarse` plus blur rows `mixed25/mixed50/mixed75` completed exit 0. Lizmark recovery for `L=8192 all_fine/mixed50/mixed75` completed 3/3 exit 0, also **`data.limit=500`**. These are limited probes only, not conclusive quality/Pareto results. Guards confirmed in CSV metadata: `output_residual_mode=anchor_span`, `token_mlp.enabled=false`, `anchor.enabled=false`, CE-only, endpoint_window, no re-read/all_past/multivector. No NaN/Inf/traceback/OOM findings in primary/recovery probe logs. Research direction: run full-dataset 5-seed validation at `L=4096` and `L=2048` for `all_fine/all_coarse/mixed50/mixed65`; use `L=8192` mixed50/mixed75 full rows only if a long-context blur curve is needed beyond validated SD-9 mixed65/all-fine/all-coarse. |
| SD-9.7 | **Full-data gap completion** (Claude-launched 2026-06-21): promote the 1-epoch SD-9.6 operating points to FULL WikiText-2, 10 epochs. Set runs {all_fine, all_coarse, mixed50, mixed65} × seeds {0,1,2}, landmark cov 0.25 b1, via `run_sd9_6_blue_long_multires_queue.sh MODE=full`. | 🚧 RUNNING | **L=2048 on blue-demon** (launcher PID 3145277, GPU0+GPU1) and **L=4096 on lizmark** (PID 783895, GPU0+GPU1) | Launched + 1 health check 2026-06-21: both alive, GPUs active, CSVs writing, **prelaunch `data_limit:"NA"` (full data) confirmed both hosts**; out roots `out/paper_mechanisms/sd9_6_{blue,lizmark}_long/blur_sweep/L{2048,4096}_full/`. Stop polling until user reports completion, then validate with a FULL-DATA-asserting summarizer (reject any `data.limit`), sync, write `audit/SD_9_7_gap_completion.md`. **GAPS STILL OPEN (next launches):** (D) matched landmark token baselines at L=2048/4096 — NO launcher yet, must build (adapt `run_a8_l8192_linear_followup_lizmark.sh`); (A) short L=512 dense 5-seed (seeds 3,4); (E) long L=8192 5-seed (seeds 3,4, mind provenance merge with `sd9_multiresolution_long` + warmup match); (F) scale-L frontier mixed L≥16384 with reduced coverage. Without (D) the new L=2048/4096 set PPLs are not yet interpretable. **D PREPARED 2026-06-21 (not launched — no free GPU, gated):** `scripts/run_sd9_7_token_baseline.sh` (matched landmark token baseline L=2048/4096, full-data) + `scripts/normalize_sd9x_runs.py` (generic HOMOGENIZER: one schema for set+token, model_kind auto-detected, set-fields=NA for token, rejects any data.limit) built, synced to both servers, syntax-checked. Homogenizer validated on existing corpus: 452 full-data rows (395 set / 57 token), limit-500 rows rejected. Fire D on first free GPU: `HOST_TAG=blue LENGTHS="2048" SEEDS="0 1 2" GPU=<free> bash scripts/run_sd9_7_token_baseline.sh` (and L=4096 on lizmark). Use `normalize_sd9x_runs.py` for the merged set-vs-baseline table — no specialized summarizer. **SET RUNS DONE 2026-06-21: 24/24 full-data 10-epoch (L=2048 blue, L=4096 lizmark).** Means (PPL/VRAM): L=2048 all_fine 1205.6/4610, mixed50 1136.0/4277, mixed62 1249.4/4191, all_coarse 1721.2/3767; L=4096 all_fine 1116.0/10164, mixed50 1073.7/8893, mixed62 1180.1/8532, all_coarse 1524.2/7114. **mixed50 Pareto-dominates all_fine on both axes (L=2048 −70 PPL/−7% VRAM; L=4096 −42/−12.5%).** Optimal blur SHIFTS with L: mixed50 wins at 2048/4096, mixed62 at 8192 → more coarse heads optimal as L grows. **D token baselines RUNNING** (relaunched after wandb fix: added `logging.wandb.enable=false` to `run_sd9_7_token_baseline.sh`; blue L=2048 PID 37101, lizmark L=4096 PID 792674, GPU0 each). On D completion: run homogenizer, report set-vs-token at L∈{2048,4096}. **2026-06-21: D-2048 baseline DONE (blue), D-4096 running (lizmark). Blue gap queue launched PID 57617** — G3 (mixed25/75 seeds 0-2) + G4 (5-seed seeds 3,4, all 6 variants) at L=2048/4096, GPU0+GPU1. Handoff `audit/SD_9_7_handoff.md` rewritten as the **raptor-mini standing playbook** (STANDING LOOP run-every-check + status-tracked gap matrix incl. yesterday's pending: short L=512 dense 5-seed, scale-L frontier G5, sparse/linear controls G7; lizmark gaps G2 mixed50@8192 + G3b/G4b@8192 PENDING until lizmark GPU0 frees). **2026-06-23 AUTHORITATIVE RESET (Claude): all prior gap runs were MALFORMED** — the queue launcher needed comma seeds but every gap launch used spaces, so `IFS=',' read` made `training.seed="0 1 2"` (run names with spaces). Killed all running (0 valid lost), DELETED malformed artifacts (blue 78, lizmark 63). FIXED launcher (`read -a seeds <<< "${seeds_csv//,/ }"`, accepts space|comma) and homogenizer (canonical metadata keys, malformed-seed filter, multires-only, dedup → safe cross-host merge; OLD aggregation TSVs are corrupt, regenerate). Valid seeds 0-2 preserved (L=512/2048/4096/8192 cores + baselines). RELAUNCHED CORRECTLY + VERIFIED single-seed: blue=L2048+L4096 (mixed25/75 s0-4, mains s3,4) PID 2088098; lizmark=L8192 (mixed50 s0-4 incl. G2, mixed25/75 s0-4, mains s3,4) PID 1173143. Guards added to `audit/SD_9_7_handoff.md §6` (no over-launch, GPU-check-first, no plaintext creds, comma seeds, canonical merge). |
### 2026-06-23 agent update

- Ran `normalize_sd9x_runs.py` on both hosts and pulled the produced TSVs:
  - `out/blue_sd9x_homogenized_runs.tsv` (fetched locally)
  - `out/lizmark_sd9x_homogenized_runs.tsv` (fetched locally)
- Created local aggregation artifacts: `out/merged_sd9x_homogenized_runs.tsv`, `out/aggregation_sd9x_summary.tsv`, and `out/missing_cells_sd9x.tsv`.
- Quick coverage counts (both hosts combined): `2048`: 53 rows; `4096`: 25; `8192`: 30; `12288`: 1; `16384`: 3.
- Launch activity: started remaining blue-supported SD-9.6 full queue on blue; started migrated G4 queue on lizmark. Launcher logs:
  - `logs/sd9_6_blue_long_blur_sweep_launch.log` (blue)
  - `logs/sd9_7_G4_migrated_lizmark_launch.log` (lizmark)

Status: still `RUNNING` for SD-9.7 until merged table shows complete 3-seed (or 5-seed where applicable) cells for each matrix entry. Use `out/missing_cells_sd9x.tsv` to flip STATUS rows to `DONE` and record provenance.

### 2026-06-25 update (Claude) — corrected optimum + duplication-proof grid scheduler

- **Live verify (both hosts active 11:51 CST):** blue 1 clean queue (L=4096→2048 5-seed ext); lizmark 2
  DISJOINT legacy queues (PID 1173146=`mixed50`@8192, PID 1191943=`mixed25 mixed75`@8192) — confirmed NOT
  duplicating each other (different variants), both fit (GPU0 2 workers @46/49GB, GPU1 1), both progressing.
  No kill (per user: kill only if true duplicates). Only stale `L8192_probe` 1-row stubs found, inert.
- **Homogenizer fixes:** (a) NUL-safe CSV read (a live worker's partial flush crashed the pass on lizmark);
  (b) **token variant now `token-{backend}`** so dense vs landmark token baselines stop colliding
  (`normalize_sd9x_runs.py`).
- **CORRECTED HEADLINE (supersedes "mixed50 wins"):** completing the **b25** wing flips the optimum.
  Mean val PPL / VRAM: L=2048 b25 **1046.4/4479** (was best b50 1136.0); L=4096 b25 **1002.4/9611**
  (was b50 1073.7); L=8192 b62 **1009.0/20193**. So argmin blur ≈ **25% for L≤4096 → ~62% at 8192** (optimum
  still rises with L; short-L sweet spot is lower than the 3-point sweep implied). b25 Pareto-dominates
  all-fine on both axes. Set b62@8192 beats matched landmark token (1048.4/25785) on both axes.
- **DUPLICATION-PROOF GRID SCHEDULER (new, canonical going forward):** `scripts/run_sd_grid.sh` +
  `scripts/sd_grid_status.py`. Guarantees: (i) 100% mutually exclusive — each (family,backend,L,variant,seed)
  cell assigned to ONE host (manifest), atomic `mkdir` lock per cell intra-host, triple-skip
  (done-marker / metadata-complete / **live-process**) so it never duplicates a legacy run; (ii) 100%
  resumable — `.done` markers + lock release on failure; (iii) OOM-mapped — failures classified, OOMs append
  `out/paper_mechanisms/sd_grid/oom_registry.tsv` (VRAM-ceiling map) + `.oom` marker (record-and-skip);
  (iv) 3-seed default; (v) full grid = set blur sweep {b0,b25,b50,b62,b75,b100} × L{512,2048,4096,8192,16384,
  32768} + landmark token baselines, families/backends split by L. **Dry-run VERIFIED** both hosts: blue plans
  only L≤4096, lizmark only L≥8192 (zero overlap), all prior runs skipped as `SKIP done`.
- **ROLLOUT = staged cutover (NOT hot-started):** legacy launchers don't honor the new lock and there is no
  free GPU (blue 2×10.6GB used; lizmark 46/49+22/49, L=16384 needs ~37GB). Hot-starting would re-introduce
  the race and OOM. Plan: let legacy queues DRAIN their valid in-flight 8192/4096 cells, then the grid is the
  SOLE scheduler. Start commands (one queue per host):
  `HOST_TAG=blue GPU0=0 GPU1=1 nohup bash scripts/run_sd_grid.sh > logs/sd_grid_blue.log 2>&1 &`
  `HOST_TAG=lizmark GPU0=0 GPU1=1 nohup bash scripts/run_sd_grid.sh > logs/sd_grid_lizmark.log 2>&1 &`
  Use `DRY_RUN=1` first each time to re-confirm the pending plan. 5-seed top-ups (seeds 3,4) only for the
  best variant per L, later, via `SEEDS="3 4"`.
- Math-agent handoff written: `docs/set_dictionary_model_provenance_for_math_agent.md` (code-grounded model
  spec + empirical-claims-to-prove + theorem-audit status).
- **AUTO-CUTOVER WATCHER deployed (2026-06-25):** `scripts/sd_grid_autostart.sh` running on BOTH hosts
  (flock-deduped, one per host; verified alive by log freshness, both waiting on legacy queues). When a
  host's legacy queue drains and GPUs go idle (<4GB), it auto-starts `run_sd_grid.sh` for that host then
  exits. Logs `logs/sd_grid_autostart_<host>.log`. The auto-started grid uses the VALIDATED manifest only
  (dense-short + landmark-long); sparse is NOT auto-run.
- **BACKEND COMPLEXITY AUDIT (2026-06-25, important for claims):** verified actual memory cost of each
  backend. `landmark`@`landmark_coverage=0.25` uses `k=round(0.25·M)` landmarks (landmark.py:52) →
  materializes `0.25·M²` score blocks = a ~4× CONSTANT reduction, **NOT linear/sub-quadratic**.
  `local_band` & `sparse_topk` build the FULL M×M then mask (local_band.py:51/55; baseline attention.py:138/155)
  → O(M²), **no memory benefit**. Only `linformer`/`nystrom` / fixed-`num_landmarks` are genuinely linear.
  Implications: (a) all PPL results valid (approximations are mathematically sound); (b) set-vs-token VRAM at
  L≥2048 is VALID as a RELATIVE comparison — token baseline uses the SAME landmark@0.25 backend
  (baseline_linear_landmark.yaml), so the set win is the multi-res atom-count/stride reduction, not backend
  trickery; (c) dense set vs dense token (L=512) is the only both-honest-O(M²) point; (d) DO NOT claim
  "linear/sub-quadratic" — reframe as constant-factor + atom-count reduction, OR add a true-linear family
  (fixed num_landmarks / nystrom / linformer) to substantiate sub-quadratic scaling; (e) SKIP sparse for the
  efficiency story (no benefit) — only a quality control. The 16384/32768 landmark OOMs (coming) will
  empirically document the quadratic wall. Math-agent report + this tracker updated accordingly.
- **PIVOT TO DENSE (2026-06-25, user-directed):** since landmark@0.25 is quadratic (not a contribution) the
  landmark grid was TRIMMED — all running/enqueued landmark queues KILLED on both hosts (blue 3 containers,
  lizmark required an aggressive -9 loop: 3 launcher children had survived and re-dispatched a landmark cell),
  landmark removed from the active matrix. Completed landmark@0.25 CSVs are PRESERVED and repurposed as the
  high-fidelity (quadratic) REFERENCE for the compression-robustness story (compression = blur/stride: more
  coarse heads ⇒ fewer atoms M≈L/s). `run_sd_grid.sh` manifest rewritten to the **DENSE matrix** and LAUNCHED
  resumably both hosts: blue = set dense L∈{512(b16),1024(b1)} all blurs + dense token; lizmark = set dense
  L∈{2048,4096(b1)} all blurs + dense token. OOM rows map the dense ceiling on 24GB(blue)/49GB(lizmark).
  Verified running: blue 2×19.7GB@97%, lizmark worker on sd_grid/set/L2048 backend=exact. Token branch made
  backend-aware (exact/landmark/local_band). Watchers stopped during pivot.
- **DENSE GRID STATUS (verified 2026-06-29):** both hosts were idle before the targeted retry. Dense SET
  rows are finished at `B4,L∈{512,1024,2048}` for all six blur fractions (3/3 seeds each). At `B4,L=4096`,
  b50/b62/b75/b100 finished 3/3; b0 and b25 OOMed 3/3 on lizmark (49 GiB), defining the current exact-dense
  frontier. The `B16,L=512` six-blur set island is also complete. Across 77 completed grid set artifacts,
  metadata/guard validation found 0 failures and completed logs had 0 NaN/Inf/traceback findings; every row
  is full-data, 10 epochs, exact, `anchor_span`, `token_mlp=false`, `anchor=false`, `endpoint_window`, with
  peak VRAM. **Token control incident:** all newly requested exact-token rows had failed before model
  construction because the launcher inherited landmark `backend_params`; this did NOT touch set training or
  set-vs-set results, but it blocks every new dense set-vs-token conclusion. Fixed by selecting the native
  `baseline_dense_exact.yaml` for exact token cells; `sd_grid_status.py` now selects registered epoch 10 from
  reusable longer runs. Container config checks and token-only dry runs passed on both hosts. Corrected queues
  launched: blue PID `3489171`, log `logs/sd_grid_blue_token_retry.log` (8 missing cells); lizmark PID
  `2153424`, log `logs/sd_grid_lizmark_token_retry.log` (6 cells). One health check passed: blue started
  L512/B16 seeds 1,2 at ~15.97 GiB/GPU; lizmark started L2048/B4 seeds 0,1 at ~21.9 GiB/GPU. Stop polling
  until the next explicit status request. Incident: `audit/incident_sd_dense_token_exact_backend_params.md`.
- **LINEAR (fixed-k) MATRIX PLANNED, not launched:** `docs/archive/deferred/sd_linear_matrix_plan.md`. Vehicle = landmark with
  fixed `num_landmarks` (now exposed in `set_only_lm.py`; reuses proven code, avoids crash-prone nystrom /
  set-awkward linformer) → genuinely O(M·k); whole model linear in L. Repurpose verdict: coverage-0.25 runs
  are NOT fixed-k points (k=0.25·M scales) — reuse them only as the quadratic reference (zero compute); all
  fixed-k cells are new. REQUIRED before launch: cell-id disambiguation in `sd_grid_status.py`/driver
  (`landmark_k{K}` vs `landmark_c{cov}`) so fixed-k and coverage runs don't conflate. Awaiting user scope
  confirm (k set, blurs) + smoke-first.
| SD-10a | **Minimal re-read ablation** (plan §7e): add per-layer `ReadXAttn` (set states ← `h₀`, causal `j≤eₘ`) on top of the verified winner; seed=pool, route + `anchor_span` identity unchanged; flag `set_reread.enabled`. One comparison vs pool-once at SD-9 short mixed `(2,1)/(4,2)`, 3 seeds, dense, CE-only. | ⏸️ HOLD (explicit user go only) | — | **UNTESTED architecture — held at user request (2026-06-21): do not implement or launch without an explicit per-task go.** This is the ONLY new architecture proposed beyond the verified SD-1…9 set; it would be run *to find out* whether re-read helps, not because it is known to. Circuit-safe by design (does NOT drop identity / un-materialize tokens / use learned latents). If green-lit: DoD = SD-2 leakage probe + nonzero-grad on ReadXAttn; verdict gates SD-11. |
| SD-11 | **Full latent-dictionary redesign** (plan §7c): learned/un-materialized latents, drop identity, generate tokens only as atom combinations, decode-by-query, compressed-memory eval. | ⏳ DEFERRED | — | **Conditional on SD-10a positive.** Has open circuit questions (seeding; raw E[x] as KV/query/identity). Do NOT implement until SD-10a green-lights. |

## Blocking Dependencies

```
SD-6 verdict CONFOUNDED (random anchor target) --> SD-6.5 (train the pre-encoder), then rerun S2
SD-6.5 --Branch A (recon drops + PPL beats S1)--> SD-7
SD-6.5 --Branch B (recon floors high w/ TRAINED target + PPL flat)--> SD-8 (all_past first; r=2 only after)
```

## Incidents

| File | Phase | Summary |
| --- | --- | --- |
| `audit/incident_sd_8_20260618.md` | SD-8 | First all_past full run used `router.score_mode=candidate_gather` and OOMed on `(4,2)` due to `[B,H,L,C,d_phi]` key expansion. Rerun once with dense router score path in clean artifact roots. |

## Current Next Step

SD-8.1 user-requested doubled-width `(4,2)` check is complete and validated. **Claude assessment
(2026-06-18): confirmatory, not a path — do NOT sweep width.** Doubling `set_state_dim/d_phi` 384→768
gave PPL 1288.6→1241.5 (−47.1) for +850.5 MiB. Frontier slope ≈ 18 MiB/PPL ⇒ closing the remaining 460
PPL to the token baseline would need ~8 GB more (total ~20.6 GB vs token 13.4 GB) — self-defeating, and
returns diminish. `set_state_dim=768` is already WIDER than the matched token baseline's `D=384`, yet
still +460 behind; a matched (D=768) token control would widen the gap. Compression reframing also
fails here: +59% PPL for only ~5% (−643 MiB) memory. This strengthens the upstream-pooling diagnosis
(over-provisioned atoms recover little because pooling already discarded token detail). Fold SD-8.1 into
the write-up's capacity-diagnostics story. Optional single `d=1536` point only for the paper's
diminishing-returns curve, not for the branch decision. **Decision still pending: (a) write-up vs
(b) contextualize-before-pool branch.** Do not launch `window_plus_landmarks`, `r=2`, SD-7,
`lambda_h=1.0`, or multivector follow-ups.

SD-8 all_past CE-only is complete and validated. **Synthesis (Claude review 2026-06-18): the capacity
branch has topped out and the SD ladder should stop.** Three now-VALID data points converge: S1
(endpoint, no anchor) PPL 1510.9/1297.9; SD-6.5 (endpoint, TRAINED anchor — recon 1.18, cos≈0.30, guard
✓) PPL 1510.4/1289.0 with PPL flat → the predictive signal now arrives but the span absorbs only
cos≈0.30 of it; SD-8 (all_past = MAXIMAL causal fiber) PPL 1363.9/1288.6 → helps (16,8) by −146 but
plateaus far above the 781.1 token baseline, (4,2) flat. Since `window_plus_landmarks ⊂ all_past` and
`r=2` only increments an already-saturated fiber, neither can close the ~500–700 PPL gap; and all_past
OOM'd under candidate-gather (forced dense router), so the quality lever also destroys the compression
rationale. **Diagnosis: the bottleneck is UPSTREAM at pooling** — atoms summarize raw `emb+pos`
(token_mlp off, no token tower), so token-resolution predictive detail was never in them; routing /
anchoring / fiber width cannot recover it. **Recommendation: stop the SD capacity ladder.** Do NOT
launch `window_plus_landmarks`/`r=2` for the quality question (settled). Two paths for user decision:
(a) write up the negative/diagnostic result (clean, valid mechanism); (b) open a NEW branch
**contextualize-before-pool** — pool the SD-6.5 causal pre-encoder's states into atoms instead of raw
`emb+pos` (needs its own fairness framing, drifts toward token-attention + pooled memory). Optional:
one `window_plus_landmarks` run only if the paper wants a memory-vs-quality figure.
