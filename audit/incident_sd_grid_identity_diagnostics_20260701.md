# Incident: Grid Identity And Multiresolution Diagnostics

Status: MITIGATED on blue-demon; lizmark deployment pending its active legacy
queue exit.

Discovered: 2026-07-01 during corrected-seed duplicate audit.

## Findings

1. `scripts/sd_grid_status.py` used a deliberately small cell key and treated
   every non-multiresolution implementation as token. Across the historical
   tree this conflated hybrid/window experiments and made distinct campaigns
   look duplicated.
2. `scripts/normalize_sd9x_runs.py` omitted native batch from its deduplication
   key and selected one duplicate by epoch count. That could pool or silently
   discard B3/B4 islands.
3. Any non-null `data.limit` other than the two historical values `8` and
   `500` passed the scanners.
4. The multiresolution forward path collected group-specific eval probes, but
   training-time `SetDiagnostics` mixed group updates and omitted router
   parameter and gradient probes because no multiresolution tensors were
   registered with the gradient probe.
5. The first strict relaunch exposed PyYAML parsing launcher `LR=1e-4` as a
   string. The numeric contract compared it directly with float `0.0001`,
   causing fail-fast configuration exits before training.

## Impact

- No confirmed duplicate successful result was found in either host registry.
- Historical files reported as duplicates by the old scanner include distinct
  planned campaigns and were not deleted.
- The first corrected blue run produced two four-epoch rows with the old
  diagnostics schema. They are excluded and archived.
- The first strict blue relaunch produced 34 configuration-failure records,
  zero CSV result rows, and no GPU training. The whole attempt is archived.
- Existing legacy results lack the new group training diagnostics and remain
  historical/unpaired evidence only.

## Remediation

- Require `training.experiment_contract=sd_grid_seeded_v1` and
  `training.diagnostics_contract=current_matrix_v1`.
- Validate full data, exact backend, architecture, residual/anchor/token-MLP
  guards, topology, deterministic seed settings, and supported `(L,batch)`.
- Make metadata scanners reject hybrid misclassification, every non-null data
  limit, malformed provenance, missing completed-run diagnostics, and
  duplicate corrected cell IDs.
- Include backend, native batch, and applied seed in normalization identity.
- Emit fine/coarse routing, pooling, parameter, and gradient diagnostics and
  export them in the downstream normalized table.
- Run set and token configs through `run_experiment.py --dry-run` before grid
  workers start. A local follow-up guard also stops the host queue on the first
  contract error rather than iterating all cells; deploy it only after the
  currently active blue queue exits.
- Normalize numeric contract comparisons so scientific notation represented as
  a YAML string is accepted only when numerically equal.

## Validation

- Functional fine/coarse training diagnostics: PASS.
- Functional fine/coarse eval probes: PASS.
- Functional token baseline diagnostics required by `current_matrix_v1`: PASS.
- Applied-seed regression tests: PASS.
- Set and token experiment contracts: PASS.
- Non-null data-limit rejection: PASS.
- Scanner implementation/limit/provenance tests: PASS.
- Normalizer group-diagnostic export test: PASS.
- Registry cleanup test: PASS.
- Blue strict manifest after archive: 120 plans, zero skips.
- Lizmark strict manifest in isolated source: 135 plans, zero skips.
- Blue runner-path set/token preflight: PASS before final relaunch.

## Remote State

- Blue final corrected queue: driver PID `1914084`; workers `1914382` and
  `1914383`; one health check passed with zero worker failures.
- Lizmark legacy queue: PID `2389855` remained active at the authorized check.
  Its executable source and active registry were not modified. Corrected
  135-cell work is enqueued under `/tmp` watcher PID `2631600`; deployment
  cannot occur until active training exits and the GPUs are idle.
