# MRP-lca-cmp Initial Exact-Dense Matrix

This matrix is the first supported LCA comparison on `mrp-lca-cmp-sd`. It uses
the current set-dictionary/anchor-span implementation: exact-dense
`baseline_token` versus exact-dense set-dictionary rows with multiresolution
fine/coarse head allocation.

All rows use `D=384`, `d_ff=1536`, `6L`, `8H`, `lr=1e-4`, `token_mlp=false`
for set rows, `output_residual_mode=anchor_span`, `anchor.enabled=false`,
`candidate_fiber=endpoint_window`, `grad_accum_steps=1`, and
`eval_microbatch_size=null` unless a cell explicitly needs a memory-control
override.

| Island | Families | Seeds | Purpose |
|---|---|---:|---|
| `L=1024,B=4` | token, set b0/b25/b50/b75/b100 | 0,1,2 | small-context learnability and diagnostics |
| `L=2048,B=4` | token, set b0/b25/b50/b75/b100 | 0,1,2 | compare with main exact-dense SD scale |
| `L=3584,B=4` | token, set b0/b25/b50/b75/b100 | 0,1,2 | main long exact-dense island |
| `L=4096,B=3` | token, set b0/b25/b50/b75/b100 | 0,1,2 | largest full feasible island |

Set blur rows:

| Row | Fine heads `(w,s)=(2,1)` | Coarse heads `(w,s)=(4,2)` |
|---|---:|---:|
| b0 | 8 | 0 |
| b25 | 6 | 2 |
| b50 | 4 | 4 |
| b75 | 2 | 6 |
| b100 | 0 | 8 |

Five-seed extension: run seeds 3 and 4 only for families that show a stable
candidate advantage over token after the 3-seed matrix.
