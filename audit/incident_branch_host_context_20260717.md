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
