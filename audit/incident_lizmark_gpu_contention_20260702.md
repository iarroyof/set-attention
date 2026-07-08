# Incident: Untracked Lizmark GPU Co-Residency

Status: contained; corrected queue resumed with strict GPU exclusivity and
deferred external-workload restart.

Date: 2026-07-02.

## Detection

The first corrected Lizmark launch used one grid worker per physical GPU, but
an unrelated process in Docker container `cancer_rl_agent` used both GPUs:

- host PID `2859013`;
- start time `2026-07-02 13:46:37`;
- initial observed allocation: 14,444 MiB on GPU 0 and 13,000 MiB on GPU 1;
- later observed allocation: 17,018 MiB on GPU 0 and 15,572 MiB on GPU 1.

Docker's host process and its child Python process were not two CUDA
workloads. The CUDA query showed one grid Python process and the unrelated
Python process on each occupied device.

## Result Disposition

Corrected `L2048/B4 b0` seeds 0 and 1 completed at 12:32 and 12:34, before the
unrelated process started. They remain accepted.

The following corrected artifacts overlapped, or were partially executed
during, unrecorded co-residency and were removed from the live aggregation
tree:

- `b0` seed 2, complete;
- `b0` seeds 3 and 4, partial;
- `b25` seed 1, partial orphaned-container attempt.

They are preserved with checksums under
`out/_archive/contention_exposed_20260702_lizmark/`. The seed-2 done marker,
affected registry rows, stale locks, CSV/JSON/output directories, and per-run
logs were also moved or filtered. The live corrected registry therefore
contained only accepted seeds 0 and 1 before resume.

No corrected OOM was observed. Both
`out/paper_mechanisms/sd_grid_seeded_v1/oom_registry.tsv` and
`contention_oom_registry.tsv` were header-only at the post-resume health
check.

## Launcher Remediation

`scripts/run_sd_grid.sh` now:

1. records per-device total, used, free, and CUDA-process allocations before
   every cell;
2. rejects another `run_experiment.py` process on the selected GPU;
3. defaults to `REQUIRE_EXCLUSIVE_GPU=1`, rejects the conflicting
   `ALLOW_GPU_CORESIDENCY=1`, and defers whenever any CUDA process is present;
4. repeats the occupancy query immediately before `docker run` to narrow the
   check/start race;
5. treats every OOM that starts or ends with another CUDA process present, or
   whose end occupancy cannot be queried, as
   `NON_TERMINAL_CONTENTION_OR_UNKNOWN_OOM`, writes it to a separate registry,
   creates no `.oom` marker, and leaves the cell queued;
6. creates a terminal memory-ceiling marker only for an exclusive OOM; and
7. signals workers so they cannot advance, then stops host-prefixed named
   Docker containers before waiting on those workers, so a worker blocked in
   foreground `docker run` cannot deadlock queue termination or leave orphaned
   matrix work.

Admission-deferred cells are reconsidered in later queue passes. The final
Lizmark launch explicitly uses strict exclusivity and no co-residency.

## External Workload Handoff

The external evaluation had no checkpoint/resume path and was observed at
least through item 108/120. Per user priority, it was terminated and
`cancer_rl_agent` was stopped before the final matrix launch.

Its script, partial CSV/JSON outputs, and log are checksummed under
`/app/_gpu_handoff_archive/20260702_set_attention_priority`. Its process
environment is stored mode `0600` under
`~/.local/state/gpu-handoff/cancer_rl_agent_semantic_eval.env`.

`scripts/restart_deferred_cancer_after_sd_grid.sh` runs as one-shot watcher PID
`3049751`. It waits for all three conditions:

1. grid driver PID `2879441` has exited;
2. no `sdgrid_lizmark_*` container remains; and
3. `nvidia-smi` reports no CUDA process.

The original container was manually started again at
`2026-07-03T07:22:24Z`, but no external CUDA process or evaluation was
started. It was stopped again and renamed
`cancer_rl_agent__deferred_until_sd_grid_release`, preventing casual restart
through the old name. At release, the watcher restores `cancer_rl_agent`,
starts it, and relaunches the evaluation from the beginning with the preserved
environment. Do not start or rename the deferred container manually while the
grid is active.

## Validation And Resume

- shell syntax: passed;
- final launcher/handoff contract tests: 6 passed;
- corrected dry plan: 133 plans, 2 accepted skips, 0 OOM skips;
- final resumed driver: PID `2879441`;
- workers: PIDs `2879862` and `2879863`;
- external restart watcher: PID `3049751`;
- queue log: `logs/sd_grid_lizmark_paper5_seeded_v1.log`.

The initial guarded driver `2868098` was stopped after review found that a
failed end-of-cell GPU query could default the end process count to zero. Its
interrupted `b0` seed 2/3 attempts are archived under
`out/_archive/contention_exposed_20260702_lizmark/fail_closed_restart_1426/`.
The final policy treats unknown end occupancy as non-terminal, and driver
cleanup signals workers before stopping named containers and waiting for
them.

Driver `2873222` was subsequently stopped when the user required full
exclusive ownership. Its interrupted attempts and admission telemetry are
archived under
`out/_archive/contention_exposed_20260702_lizmark/strict_exclusive_restart/`.

At the one permitted final post-launch check, `b0` seeds 2 and 3 were admitted
with `cuda_processes=none`, about 48.6 GiB free per device, and
`co_resident=0`. The only later GPU processes were one set-attention Python
process per device. `cancer_rl_agent` remained exited, and both OOM registries
were header-only. Stop-polling is active until the user requests status.

## Historical OOM Scope

The legacy `L4096/B4` set `b0/b25` OOMs occurred on 2026-06-26 and the token
OOMs on 2026-06-29. They predate PID `2859013`, so this incident did not cause
them. Those launchers did not archive per-cell external-process telemetry,
however. They remain repeated 3/3 legacy feasibility observations, not
retrospectively certified exclusive-capacity measurements. New terminal OOM
claims require the admission telemetry above.
