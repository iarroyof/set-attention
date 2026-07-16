# MRP-5: Tokenizer-Matched WikiText-2 And PG-19 Transfer

Status: BLOCKED on MRP-2 completion and explicit transfer launch approval

Owner: UNASSIGNED

Updated: 2026-07-15 after MRP-3 completed as null/inconclusive.

## Mission

Test whether the interior multiresolution frontier transfers from WikiText-2
to a different long-document domain without retuning blur. PG-19 is the only
new natural corpus in this program.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `mrp0_reproducibility_platform.md`
3. `mrp1_matrix_closure.md`
4. `mrp2_natural_ar_hits.md` and its final evaluator/audit
5. `../../audit/MRP_3_mqar_mechanism.md`
6. PG-19 source: `https://arxiv.org/abs/1911.05507`
7. `../../src/set_attention/data/wikitext.py`
8. `../../src/data/wikitext2.py`
9. `../../scripts/prefetch_datasets.py`

## Write Scope

- `src/data/pg19.py` and `src/data/fixed_tokenizer.py` (new)
- PG-19 registry support in `src/set_attention/data/`
- `scripts/prefetch_datasets.py`
- `scripts/run_lm_transfer.py` (new)
- `scripts/summarize_lm_transfer.py` (new)
- `configs/transfer/pg19/` (new)
- transfer/tokenizer tests
- `audit/MRP_5_pg19_transfer.md`

Use MRP-0 checkpoint/provenance APIs. Do not edit model architecture.
Use the MRP-0 ordered-text interface read-only; do not modify
`src/data/wikitext2.py` or `src/data/ordered_text.py`.

## Tokenizer Contract

Use the public GPT-2 byte-level BPE tokenizer unchanged for both corpora and
all rows. Prefetch and cache its complete artifact set, record file SHA-256
digests, assign an explicit pad token policy, and freeze it before dataset
tokenization.

The tokenizer:

- is not trained or expanded on either corpus;
- has no target-corpus vocabulary growth;
- is serialized with every checkpoint;
- must match by digest at evaluation.

Existing word-level WikiText-2 runs cannot be reused in this study.

## Registered Rows And Island

This matrix is a registration, not launch authorization. Do not prefetch from
the network or launch training until approval is recorded by the tracker write
owner in `audit/phase_sd_status.md`.

For each dataset, run:

- exact token;
- b0;
- frozen `b*`;
- b100.

Use:

- `L=2048,B=4`;
- applied seeds `0,1,2`;
- exact dense backend;
- current model shape, LR, warmup, and set guards.

This is 12 WikiText-2 runs plus 12 PG-19 runs. No b25/b50 substitution is
allowed after `b*` is frozen.

## Matched Training Budget

1. Tokenize the full WikiText-2 training split with the frozen tokenizer.
2. Let `T_WT2` be its number of next-token targets.
3. Set the training budget `T_train=10*T_WT2`.
4. WikiText-2 consumes exactly ten ordered passes.
5. PG-19 uses a fixed applied-seed document shuffle, preserves within-document
   order, drops cross-document chunks, and consumes exactly `T_train` targets.
6. Evaluate on each full official validation split without a line/document
   limit.

Report the realized update and target counts. `data.limit` remains absent.

## Metrics

Per dataset, row, and seed:

- validation NLL/PPL;
- bits per byte;
- peak train VRAM;
- processed training and validation targets/bytes/documents;
- checkpoint, tokenizer, dataset, and config digests;
- feasible PPL/VRAM Pareto frontier.

For b*, also report fine/coarse group ablation and group-local routing
diagnostics. Apply the completed MRP-2 AR-hit evaluator to every
tokenizer-matched WT2 and PG-19 checkpoint, reporting the same count and lag
slices with corpus-specific training counts. This adds no training rows.
Do not compare absolute PPL between datasets.

## Transfer Gate

The cross-corpus mechanism is supported when frozen b* lies on the set-family
PPL/VRAM Pareto frontier on both tokenizer-matched WikiText-2 and PG-19 and its
paired applied-seed NLL difference has a 95% Student-t CI strictly below zero
against both b0 and b100 on PG-19.

Token superiority is not required. Report b* versus token as a separate
paired applied-seed NLL and quality/memory comparison with 95% CIs.

If the gate fails, the paper must restrict the interior-optimum claim to
WikiText-2 and present PG-19 as a negative transfer result. Do not tune blur on
PG-19.

## Tests

1. Frozen tokenizer produces identical IDs across datasets and runs.
2. Tokenizer digest mismatch fails closed.
3. Ordered chunking never crosses documents.
4. Training target budget is exact.
5. Full validation is used and `data.limit` is rejected.
6. Mocked PG-19 splits preserve document identity/order.
7. Bits-per-byte and PPL match hand-computed token NLLs.
8. Summaries reject cross-tokenizer or cross-batch pooling.

## Definition Of Done

All 24 runs and full validation evaluations pass metadata/log checks, the
transfer gate is applied without retuning, and
`audit/MRP_5_pg19_transfer.md` records both supportive and limiting evidence.

## Durable Handoff

Status: BLOCKED.

Last completed action: tokenizer, token budget, rows, transfer gate, and AR-hit
reuse registered. MRP-0 has since passed, MRP-1 has frozen `b*=b25`, the
short-B3 bridge does not change the transfer rows or gate, and MRP-3 is
reviewed as null/inconclusive.

Files changed: this subplan only during registration.

Commands/tests and outcomes: PG-19 literature and current data-runner support
were reviewed; no data was prefetched.

Artifacts and digests: none.

Host/PID/log/ETA: none.

Decision or gate result: network/prefetch and launch approval have not been
granted. Transfer remains gated on MRP-2 completion and explicit transfer
approval.

Known incident or limitation: current runner supports only WikiText-2 for the
active matrix; the fixed-tokenizer PG-19 path still needs implementation and
approval before any network/prefetch or training action.

Next atomic action: wait for MRP-2 completion and explicit transfer approval.

Inputs required: frozen GPT-2 tokenizer artifacts, approved PG-19 access/host
allocation, and completed upstream audits.
