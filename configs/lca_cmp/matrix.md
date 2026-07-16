# MRP-lca-cmp Initial Exact-Dense Matrix

This matrix is the first supported LCA comparison on `mrp-lca-cmp`. It uses the
model families available on `origin/main`: exact-dense `baseline_token` and
exact-dense `set_only`. Native set-dictionary mixed-head blur rows require the
set-dictionary implementation to be merged into this branch before launch.

All rows use `D=384`, `d_ff=1536`, `6L`, `8H`, `lr=1e-4`, `token_mlp=false`
for set rows, `grad_accum_steps=1`, and `eval_microbatch_size=null` unless a
cell explicitly needs a memory-control override.

| Island | Families | Seeds | Purpose |
|---|---|---:|---|
| `L=1024,B=4` | token, set `(w,s)=(2,1),(4,2),(8,4)` | 0,1,2 | small-context learnability and diagnostics |
| `L=2048,B=4` | token, set `(w,s)=(2,1),(4,2),(8,4)` | 0,1,2 | compare with main exact-dense SD scale |
| `L=3584,B=4` | token, set `(w,s)=(2,1),(4,2),(8,4)` | 0,1,2 | main long exact-dense island |
| `L=4096,B=3` | token, set `(w,s)=(2,1),(4,2),(8,4)` | 0,1,2 | largest full feasible island |

Five-seed extension: run seeds 3 and 4 only for families that show a stable
candidate advantage over token after the 3-seed matrix.
