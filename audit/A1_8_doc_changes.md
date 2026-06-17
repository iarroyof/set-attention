# A1.8 Documentation/Config/Test Scan Audit

Date: 2026-05-09

Scope: active docs, configs, tests, scripts, and README surface for A1.1-A1.7 consistency. Archived planning drafts, old paper snapshots, and paper bundle artifacts were scanned but left unchanged when clearly historical.

## Modified Files

- `configs/hyperparameters.md`: clarified that the active revision backend surface is `exact`, `local_band`, and `landmark`; marked `nystrom` and `linformer` as legacy/deprecated schema paths rather than active tested backends.
- `docs/revision_source_of_truth_definitions.md`: clarified active linear backend semantics, diagnostics use of the supplied candidate fiber, strict-past partial-window wording, and deprecated Nyström status.
- `Context For Revision Agent after NeurIPS2026 LLM feedback.md`: documented the current blue-demon Docker fallback to direct function execution because pytest is unavailable in the container.
- `scripts/create_paper_complements_yamls.sh`: changed generated active linear-landmark config from `backend_params.num_landmarks` to `backend_params.landmark_coverage: 0.25`.
- `scripts/export_main_results_provenance.sh`: changed landmark provenance field and paper-ready statement from `num_landmarks` to `landmark_coverage`.
- `scripts/summarize_paper_family_complements.py`: changed active linear-landmark summary/dedup fields from `num_landmarks` to `landmark_coverage`.
- `scripts/postrun_collect_family_complements.sh`: changed active linear-landmark summary/dedup fields from `num_landmarks` to `landmark_coverage`.
- `audit/A1_8_doc_changes.md`: this audit summary.

## Scan Results

- `num_landmarks`: remaining active-surface hits are limited to explicit deprecated/Nyström documentation, the legacy `configs/set_only/wikitext2_nystrom.yaml`, and tests that enforce active landmark configs do not use it.
- `pydantic`: no active-surface hits outside the locked plan text.
- `M=64` / `ceil(512 / 8)`: no active source-of-truth/config/test/script hits for strict-past LR-norm.
- Old landmark example `[0, 4, 8, ..., 60]`: no active-surface hits.
- Router top-1 as head agreement: no active-surface hits; source-of-truth defines it as head-averaged probability concentration.
- `reference recipe`: no active-surface hits.
- `clipped trailing windows`: active source-of-truth now uses partial-window wording for strict-past; noncausal clipping remains described as historical mode.
- `token_to_sets` legacy membership: active source-of-truth now explicitly says diagnostics consume the supplied candidate fiber and must not reconstruct containing-token membership.
- Hardcoded alpha/hash/min-temp references: active docs describe canonical normalized keys and resolved logging fields.
- Nyström: remaining active-surface references are marked legacy/deprecated or are in the explicit Nyström legacy config/test path.

## Verification Notes

No runtime model or test logic was changed for A1.8. The only Python change was a provenance/summary helper field rename from `model.backend_params.num_landmarks` to `model.backend_params.landmark_coverage`; compile verification should cover that script.
