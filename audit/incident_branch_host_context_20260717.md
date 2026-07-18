# Incident: Branch And Host Context Drift

Date: 2026-07-17

## Summary

The LCA comparison scaffold was initially created on `mrp-lca-cmp` from
`origin/main`, which omitted the current set-dictionary/anchor-span model
implementation. That branch is a false-start validation branch and must not be
used for current experiments.

The corrected branch is `mrp-lca-cmp-sd`, based on
`set-dictionary/anchor-span`, with current HEAD:

```text
e221ec2 Correct LCA matrix to set-dictionary multiresolution
```

The user verified interactive GitHub pushes on 2026-07-17:

- `mrp-lca-cmp-sd` pushed to `origin/mrp-lca-cmp-sd`
- `set-dictionary/anchor-span` pushed to `origin/set-dictionary/anchor-span`

## Host Audit

Audited without modifying host files.

| Host | Path | State |
|---|---|---|
| blue-demon | `~/set-attention` | git repo on `paper/final-results-bundle@1174947`, 421 dirty/untracked status entries, `ACTIVE_RUNTIME_COPY.md` present |
| blue-demon | `~/set-attention-anchor-span-sync` | git repo on `set-dictionary/anchor-span@8107a7b`, deprecated marker present |
| blue-demon | `~/set-attention-mrp0-validation` | not a git repo, deprecated marker present |
| Lizmark | `~/set-attention` | not a git repo, `ACTIVE_RUNTIME_COPY.md` present |
| Lizmark | `~/set-attention-anchor-span-sync` | not a git repo, deprecated marker present |

## Impact

No LCA scientific sweeps were launched from the false-start branch. The only
executions were dry-runs and one-step preflights used for validation. However,
generic instructions to pull directly on Blue/Lizmark were unsafe for the
actual host state:

- Blue correctly refused checkout because local dirty/untracked files would be
  overwritten.
- Lizmark could not run git commands because `~/set-attention` is not a git
  repository.

## Required Guard

Before any new GPU launch or host sync:

1. Verify local authoritative branch and commit:
   `mrp-lca-cmp-sd@e221ec2` for LCA comparison, or
   `set-dictionary/anchor-span@3e7edb4` for the base set-dictionary line.
2. Verify the host target is a git repository and is clean, or explicitly
   archive/reconcile it before checkout/sync.
3. Never run `git switch`, `git reset`, `git pull`, source sync, compose run, or
   experiment launch from a host directory just because its path is
   `~/set-attention`.
4. Do not use `mrp-lca-cmp` for current set-dictionary experiments.
5. Do not use deprecated alternate directories for new launches.

## Recovery Direction

The next safe operational step is not a launch. It is a deliberate host repair
plan:

- Blue: archive or reconcile the dirty `~/set-attention` tree before switching
  to `origin/mrp-lca-cmp-sd`.
- Lizmark: replace or recreate `~/set-attention` as a proper git checkout, or
  record an intentional rsync-only runtime policy before launch.

No destructive cleanup is authorized by this incident note alone.

## Resolution (2026-07-17/18)

Repair executed after the user pushed `d0f7ae8` and approved host repair:

1. Both hosts verified idle (no project processes; all GPUs 1 MiB used, 0%).
2. Old unsafe `~/set-attention` directories preserved by timestamped rename to
   `~/repo_audit_copies/set-attention_pre_mrp_lca_cmp_sd_repair_20260717_175927`
   on both hosts. Nothing was deleted.
3. Clean `~/set-attention` checkouts recreated on both hosts from the verified
   host-side git bundle (no GitHub credentials needed on hosts), then
   fast-forwarded from bundle
   `~/repo_audit_bundles/set-attention-mrp-lca-cmp-sd-2ded5d1.bundle` to
   `2ded5d1` ("Relax LCA PPL roundoff validation", not yet pushed to origin).
4. Compose validation passed on both hosts at `2ded5d1`: `py_compile` of the
   LCA modules, LCA dry-run, and `scripts/check_lca_batching_preservation.py`
   (PPL roundoff fixed by the tolerance commit).
5. Runtime directory fixes: root-owned empty `.hf` (created by the docker
   volume mount) was replaced with user-owned `.hf/datasets` and `.hf/hub`;
   `out/lca_cmp`, `logs/lca_cmp`, `subsets`, `wandb` created. In-container
   write test to `out/lca_cmp` passes on both hosts.
6. Lizmark-specific guard: the host user is uid/gid 1001 while the container
   image default is 1000. Compose launches on Lizmark must set the
   `UID`/`GID` environment (e.g. `UID=$(id -u) GID=$(id -g) docker compose
   run ...`), because `docker-compose.yml` uses `user: "${UID}:${GID}"`.
   Without it, container writes to mounted `out/` fail with EACCES. Blue is
   uid 1000 and matches the image default. `.nv/` (root-owned NVIDIA toolkit
   artifact) was added to `.git/info/exclude` on Lizmark to silence a
   git-status warning.

Both `~/set-attention` checkouts are now launch-ready git repos on
`mrp-lca-cmp-sd@2ded5d1` (ahead of origin by the unpushed validation commit).
Both hosts remain idle; no scientific launch has been made from the repaired
copies.
