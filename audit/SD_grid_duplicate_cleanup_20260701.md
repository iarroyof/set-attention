# SD Grid Duplicate And Superseded-Artifact Cleanup

Status: PARTIAL. Blue cleanup complete; lizmark active-registry cleanup waits
for its legacy queue to exit.

## Classification

The audit used immutable `grid_runs_<host>.tsv` records, metadata, epochs,
native batch, implementation, backend, topology, applied-seed provenance, and
paths. A shared reduced scanner cell ID alone is not sufficient proof of a
duplicate.

| Host | Repeated registry cells | Extra records | Duplicate successful results | Classification |
|---|---:|---:|---:|---|
| blue-demon legacy registry | 8 | 24 | 0 | failed exact-token config/retry attempts |
| lizmark legacy registry | 6 | 12 | 0 | failed exact-token config/retry/OOM attempts |

The old blue metadata scan also reported 16 duplicate cell IDs and 42 extra
paths. Those were false positives for cleanup purposes: distinct SD-9/SD-9.5
campaigns, historical window/hybrid configurations omitted by the old key, or
the active corrected partial rows. None was deleted as a successful duplicate.

## Archived

Blue:

- raw 151-row legacy registry and 24 superseded failed records:
  `out/_archive/duplicate_launch_records_20260701_blue/`;
- cleaned live registry: 127 rows, one retained record per cell;
- duplicate successful records archived: zero;
- old-schema partial corrected rows:
  `out/_archive/incomplete_diagnostics_schema_v0_20260701_123252_blue/`;
- fail-fast numeric-LR contract attempt:
  `out/_archive/failfast_lr_contract_20260701_123635_blue/`;
- corresponding logs under `logs/_archive/`;
- superseded initial paper5 launch logs:
  `logs/_archive/superseded_paper5_launch_20260701_blue/`.

Lizmark:

- superseded initial paper5 launch logs:
  `logs/_archive/superseded_paper5_launch_20260701_lizmark/`.

The lizmark active registry has not been rewritten. Deferred-deployer PID
`2631600` will run the tested registry cleanup tool and record its manifest
after PID `2389855` exits, before deploying corrected source.

Local:

- empty dry-run-only `sd_grid_seeded_v1` scaffolding moved to
  `out/_archive/local_seeded_v1_dryrun_20260701/`.

## Aggregation Guards

- Current scanners default to `out/paper_mechanisms/sd_grid_seeded_v1`.
- Corrected rows require explicit applied-seed and diagnostics contracts.
- Duplicate corrected IDs fail instead of selecting one path.
- Native batches remain separate identity fields.
- Archive roots are outside `out/paper_mechanisms` and cannot enter the
  current status scanner or normalizer.

No valid completed scientific result was removed.
