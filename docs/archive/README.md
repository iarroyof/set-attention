# Documentation Archive

Everything below this directory is non-canonical.

- `brainstorms/`: superseded generated plans, exploratory architecture proposals, and untitled drafts.
- `deferred/`: designs that were discussed but are not approved for implementation or launch.
- `legacy_root_tools/`: misplaced root-level tools or ad hoc scripts that are retained only for provenance after replacement by canonical scripts.
- `legacy_context/`: replaced agent context retained for provenance.

Do not use archived files to select experiments, infer live status, or support current paper claims.
Current guidance begins at `docs/README.md`.

## Deprecation Policy

Move a file into this archive only when all of the following are true:

1. A current canonical driver, plan, audit, or manuscript section replaces it.
2. A repository-wide reference search shows that active scripts do not import or call it.
3. The archive destination has a descriptive name and, when needed, a short README explaining the replacement.
4. Generated outputs under `out/` remain artifact provenance unless they are explicitly superseded by a newer artifact bundle.

Prefer `git mv` for tracked files so history records the rename. Do not delete
legacy material merely because it is old; archive it with enough context for
future agents to understand why it is no longer canonical.
