# Set-Dictionary Branch Status Tracker

Last updated: 2026-06-16 19:12 CST by Codex after validating SD-1 on blue-demon.

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
| SD-2 | §4 numerical leakage probe for `anchor_span` and anchor pre-encoder path | ⏳ PENDING | — | Blocks all ladder/fairness work. |
| SD-3 | §5 fairness audit harness (parameter parity, inference/train VRAM split, span ablation) | ⏳ PENDING | — | No ladder launches until SD-2 and SD-3 PASS. |
| SD-4 | Config contract for set-dictionary keys | ⏳ PENDING | — | Update `src/config/*`, `configs/hyperparameters.md`, `get_resolved_metadata()`, and CSV fingerprint when keys land. |
| SD-5 | S1 ladder: `anchor_span`, anchor disabled, CE only, dense backend | ⏳ PENDING | — | Reuse existing `direct` refs; dense backend first; no launch until SD-2/SD-3 PASS. |
| SD-6 | S2 ladder: S1 + causal pre-encoder anchoring (`lambda_h=0.1`) | ⏳ PENDING | — | Teacher path must remain disabled. |
| SD-7 | Follow-up winner-only knobs (`lambda_h=1.0`, then `lambda_div>0`) | ⏳ PENDING | — | Only if S2 gives signal. |
| SD-8 | Deferred floor tests (`multivector_basis.r=2` or wider candidate fiber) | ⏳ PENDING | — | Only if S2 reconstruction error floors high and PPL gap persists. |

## Blocking Dependencies

```
SD-1 -> SD-2 -> SD-3 -> SD-5 -> SD-6 -> SD-7/SD-8
```

## Incidents

| File | Phase | Summary |
| --- | --- | --- |

## Current Next Step

Start SD-2: implement `output_residual_mode="anchor_span"` and the causal anchor pre-encoder path,
then add the §4 numerical leakage probe for both paths. Do not launch ladder runs until SD-2 and SD-3
are PASS.
