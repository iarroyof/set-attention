# Incident SD-8 2026-06-18

Status: RESOLVED

Phase: SD-8 all_past dense CE-only

Summary:
- The first SD-8 full launch used `candidate_fiber=all_past` with `router.score_mode=candidate_gather`.
- The `(16,8)` topology completed 3/3 seeds.
- The `(4,2)` topology OOMed immediately for seeds 0/1, before epoch 1 metrics were written; seed 2 did not launch.

Failure signature:
- Logs: `logs/sd8_all_past_dense/worker_gpu0.log`, `logs/sd8_all_past_dense/worker_gpu1.log`
- Error: `torch.OutOfMemoryError` in `src/models/set_only/router.py` candidate-gather key expansion.
- Allocation attempted: about 23.91 GiB on a 24 GB GPU.

Cause:
- `all_past` widens `C_t` to all sealed past atoms.
- `router.score_mode=candidate_gather` materializes gathered keys with shape proportional to
  `[B, H, L, C, d_phi]`, which is too large for `(4,2)` at `L=512`, `B=16`, `D=384`.

One-time fix:
- Rerun SD-8 in a clean artifact root with `router.score_mode=dense`.
- The dense score path uses the same masked causal support but scores over `[B, H, L, M]`, avoiding the
  candidate-gather key expansion.
- Do not mix the failed candidate-gather partial outputs into the validated SD-8 package.

Guard:
- Rerun smoke and full validation before updating SD-8 to DONE.
- Do not launch `window_plus_landmarks`, `r=2`, SD-7, `lambda_h=1.0`, or multivector follow-ups.

Resolution:
- Dense-router smoke validated 1/1 in clean roots.
- Dense-router full ladder validated 6/6 with manifest
  `out/paper_integrated_evidence/checks/sd8_all_past_dense_routerdense_manifest.json`.
- Audit package: `audit/SD_8_all_past_dense_routerdense.md`.
