# Short-B3 Exact-Dense Bridge Completion

Date: 2026-07-13

Scope: post-run validation for the later `short_b3` bridge extension under
`out/paper_mechanisms/sd_grid_seeded_v1`, covering the missing exact-dense
`L512/B3` and `L1024/B3` islands. These rows are outside the closed 255-row
paper5 bundle and were launched to complete the descriptive batch bridge.

## Host State

Both GPU servers were idle at verification:

| Host | GPU state |
|---|---|
| blue-demon | both GPUs 0%, 1 MiB used |
| lizmark | both GPUs 0%, 1 MiB used |

No `sdgrid`, MRP, `run_experiment.py`, or `run_mqar.py` processes remained.

## Endpoint Validation

| Island | Host | Expected cells | Endpoint-valid cells | OOM registry | Contention-OOM registry |
|---|---|---:|---:|---|---|
| `L512/B3` | lizmark | 30 | 30 | header-only | header-only |
| `L1024/B3` | blue-demon | 30 | 30 | header-only | header-only |

The validation includes token rows under `token/L*` and set rows under
`set/L*`. A first directory-local check that looked only under `set/` falsely
reported missing token rows; the corrected check includes both subtrees.

No metric NaN/Inf was found. The only apparent non-finite hits were hex
`run_id` strings parsed by a broad numeric-looking regex, not metric fields.

## Aggregate Results

Mean +/- sample standard deviation over five seeds.

### L512/B3

| Row | Val PPL | Peak VRAM MiB | Span-ablation delta PPL |
|---|---:|---:|---:|
| token | 1048.040 +/- 30.003 | 3194.990 +/- 0.000 | NA |
| b0 | 1039.951 +/- 42.091 | 3325.671 +/- 0.000 | 50438.592 +/- 2444.918 |
| b25 | 943.695 +/- 26.439 | 3272.788 +/- 0.000 | 57644.663 +/- 2735.924 |
| b50 | 955.754 +/- 40.509 | 3185.149 +/- 0.000 | 58429.771 +/- 1897.532 |
| b75 | 1080.618 +/- 15.915 | 3122.127 +/- 0.123 | 52898.955 +/- 1782.183 |
| b100 | 1517.896 +/- 50.467 | 2953.479 +/- 0.000 | 38618.189 +/- 1650.662 |

### L1024/B3

| Row | Val PPL | Peak VRAM MiB | Span-ablation delta PPL |
|---|---:|---:|---:|
| token | 988.047 +/- 23.773 | 6207.766 +/- 0.000 | NA |
| b0 | 987.612 +/- 22.510 | 6442.156 +/- 0.000 | 53379.376 +/- 1579.178 |
| b25 | 969.176 +/- 41.925 | 6250.296 +/- 0.000 | 60453.776 +/- 2184.252 |
| b50 | 979.152 +/- 30.093 | 5955.508 +/- 0.000 | 58520.780 +/- 2851.244 |
| b75 | 1101.028 +/- 37.818 | 5691.994 +/- 1.217 | 55932.385 +/- 2223.910 |
| b100 | 1490.520 +/- 54.496 | 5193.230 +/- 0.000 | 43623.176 +/- 1481.122 |

## Interpretation

The short-B3 bridge reinforces the main exact-dense pattern rather than
reversing it. The best mean-PPL row is `b25` at both scales. All-coarse
`b100` is the cheapest memory row but pays a large quality penalty. The
fine-only `b0` row approaches token at `L1024/B3`, as expected, but does not
dominate the mixed row. These rows are useful for the appendix/batch bridge
story and for plotting behavior across `B3/B4/B16`; they do not replace the
closed paper5 selection island.

The runs were co-resident/accelerated, including manually locked auxiliary
cells after the admission-control bug was identified. Therefore the PPL and
per-process peak VRAM fields are usable, but wall-time/throughput and
exclusive-capacity/OOM interpretations are not.

## Remaining MRP State

MRP-2 checkpoint retraining is complete: 12/12 registered rows have epoch-10
CSV endpoints and final checkpoints.

MRP-3 primary MQAR remains incomplete: 8/18 rows have final checkpoints. The
first incomplete row is `b25_seed1_B4`; its CSV is an empty stub with no final
checkpoint. Since the short-B3 queues are now complete and both servers are
idle, the MRP-3 resume is no longer blocked by short-B3 GPU pressure.
