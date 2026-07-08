# Set-Dictionary Agent Subplans

These files divide the canonical multiresolution research program into
independently resumable, end-to-end tasks. The main authority is
`../set_dictionary_research_main_plan.md`.

## Session Start

Read:

1. `../set_dictionary_research_main_plan.md`;
2. `../../audit/phase_sd_status.md`;
3. only the assigned subplan;
4. the subplan's required retrieval files.

Do not infer state from chat history. Do not execute an archived plan.

## Ownership

Each subplan declares a write scope. Agents working concurrently must not edit
outside that scope. Shared-file changes are owned by MRP-0, formal TeX changes
by MRP-6D, and final narrative/bundle changes by MRP-7.

Claiming a task means replacing `Owner: UNASSIGNED` with a durable role label
and setting `Status: RUNNING` in that subplan. Personal names and
product/vendor names are not required.

Only the current program-integration/experiment-operations role edits
`audit/phase_sd_status.md` or the main plan. Task agents record transitions in
their own subplan/audit and hand them to that role. Shared-tracker ownership
must be explicitly transferred in the tracker before another agent edits it.

## Status Vocabulary

- `READY`: prerequisites satisfied; work may begin.
- `RUNNING`: an owner is actively executing the subplan.
- `BLOCKED`: a named prerequisite is incomplete.
- `HOLD`: execute only if the subplan's deterministic trigger fires.
- `NOT_TRIGGERED`: the trigger did not fire; no run is needed.
- `PASS`: all deliverables and validation gates passed.
- `FAILED`: the protocol completed but its scientific hypothesis failed.
- `INCIDENT`: execution stopped because a correctness or infrastructure guard
  failed.

`FAILED` is a valid research result. `PASS` describes protocol completion, not
whether a preferred hypothesis was supported.

## Handoff

Before stopping, update the subplan status block and append:

```text
Last completed action:
Files changed:
Commands/tests and outcomes:
Artifacts and digests:
Host/PID/log/ETA:
Decision or gate result:
Known incident or limitation:
Next atomic action:
Inputs required:
```

Do not write secrets, passwords, or private credentials.
