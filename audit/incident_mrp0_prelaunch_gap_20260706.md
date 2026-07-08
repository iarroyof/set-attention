# Incident: MRP-0 Was Not A Completed Prelaunch Platform

Status: MITIGATED IN LOCAL SOURCE; CONTAINER VALIDATION AND DOWNSTREAM GATE PENDING

Discovered: 2026-07-06 while distinguishing MRP-1 diagnostics validation from
the registered MRP-0 reproducibility/checkpoint platform.

## Pre-Mitigation Finding

Before the corrected MRP-1 matrix, the repository implemented and tested only
the MRP-1-specific subset:

- apply `training.seed` before data/model construction;
- log requested/applied/Torch seeds;
- enforce full WikiText-2, exact backend, architecture, and set guards;
- require family-specific endpoint diagnostics;
- reject malformed, duplicate, limited-data, and incomplete corrected rows.

The broader MRP-0 platform was not implemented:

- `src/train/checkpoints.py` is absent;
- `src/data/ordered_text.py` is absent;
- checkpoint save/load and eval-only modes are absent;
- dataset, tokenizer/vocabulary, and checkpoint digests are absent;
- dedicated loader-generator seeds and first-batch reproducibility tests are
  absent;
- masked `-100` metric denominators are not implemented for the LM loop;
- the required MRP-0 audit is absent.

No corrected MRP-1 artifact is a model checkpoint. The pulled Blue package
contains CSV, JSON, JSONL, logs, registries, and markers only.

## Determinism Qualification

All 120 Blue corrected logs contain PyTorch warnings that CuBLAS operations
are not deterministic without `CUBLAS_WORKSPACE_CONFIG`. The runner calls
`torch.use_deterministic_algorithms(..., warn_only=True)`, so
`training.deterministic=true` records a requested mode, not a fail-closed
bitwise-reproducibility guarantee.

Applied seed provenance is still real: all 120 Blue JSON files have
`seed_applied=true`, requested/applied/Torch seed equality, full data, exact
backend, and the registered experiment/diagnostics contracts. CUDA
nondeterminism adds uncontrolled run noise and prevents exact replay; it does
not turn the five distinct runs back into the legacy unseeded replicates.

## Impact On Completed MRP-1

Still supported:

- within-island PPL and peak-VRAM estimates;
- set-vs-token and set-vs-set mean/CI comparisons;
- span ablations, routing/pooling probes, and group diagnostics for rows that
  pass the endpoint contract;
- the registered `L2048,B4` selection of `b*=b25`.

Qualifications:

- exact same-seed replay is not established;
- cross-host data/tokenizer identity is not cryptographically proven, although
  corrected Blue rows uniformly report WikiText-2 and vocabulary size 76618;
- paired-seed inference contains extra CUDA nondeterminism;
- completed training cannot be reused for new token-level MRP-2 evaluation
  because no checkpoints were saved.

The missing masked-target metric does not affect current WikiText-2 PPL,
because those LM rows do not use `-100` labels. It directly blocks MQAR and
other masked-target tasks.

The Lizmark epoch-gradient incident is separate: it affects endpoint
diagnostic completeness for 15 mixed rows, not MRP-0 checkpoint/provenance.

## Disposition

1. Do not describe MRP-0 as validated or as `PASS`.
   The current source may be described only as locally implemented pending
   project-container validation.
2. Do not launch MRP-2, MRP-3, or MRP-5.
3. Let the registered Lizmark MRP-1 queue and diagnostic replacements finish;
   do not retroactively change their runtime contract.
4. Implement MRP-0 locally and validate it on idle Blue only after focused
   tests pass. This is new infrastructure work, not post-hoc validation of the
   completed matrix.
5. Require fail-closed deterministic configuration, checkpoint round-trip,
   loader-batch reproduction, digest mismatch rejection, eval-only behavior,
   masked metrics, and ordered-token identity before downstream approval.
6. Because MRP-1 checkpoints do not exist, MRP-2 must use its registered
   retraining path after MRP-0 passes.

Local implementation and validation status:
`audit/MRP_0_reproducibility_platform.md`.
