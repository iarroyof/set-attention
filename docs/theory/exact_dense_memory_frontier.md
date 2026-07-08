# Exact-Dense Memory Frontier For Multiresolution Set Dictionary

Status: analytic MRP-6B complete; empirical fit and final table pending MRP-1
replacement closure.

Updated: 2026-07-07.

This memo covers only the analytic part of MRP-6B. It is code-faithful to the
active exact-dense `set-dictionary/anchor-span` branch and does not edit the
canonical TeX. Empirical VRAM fits, bootstrap intervals, and final table
replacement are intentionally left pending until the 15 `L3584,B4` diagnostic
replacement rows close.

## Implemented Variables

Active set rows use `SetOnlyLM` with exact dense set attention, `D=384`,
`d_ff=1536`, `K=6` set-attention blocks, `H=8` routing/set heads,
`set_state_dim=d_phi=384`, `output_residual_mode=anchor_span`, token MLP
disabled, anchor disabled, `candidate_fiber=endpoint_window`, and
`strict_past` banks.

Each multiresolution group `g` has:

- head count `H_g`;
- stream width `D_g = D H_g/H`;
- bank window and stride `(w_g,s_g)`;
- strict-past set count
  `M_g(L)=floor((L-w_g)/s_g)+1` when `L >= w_g`, and zero otherwise;
- independent bank, pooling module, feature builder, set stack, router, and
  content-bias adapter.

The active blur rows are:

| Row | Fine heads `(w,s)=(2,1)` | Coarse heads `(w,s)=(4,2)` | Coarse fraction `p` |
|---|---:|---:|---:|
| b0 | 8 | 0 | 0.00 |
| b25 | 6 | 2 | 0.25 |
| b50 | 4 | 4 | 0.50 |
| b75 | 2 | 6 | 0.75 |
| b100 | 0 | 8 | 1.00 |

For the active fine/coarse banks,

```text
M_f(L)=L-1
M_c(L)=floor((L-4)/2)+1.
```

For even registered lengths, `M_c=L/2-1`.

## Exact Score Tensor Count

In `DenseExactBackend.forward`, each group computes `q`, `k`, and then
materializes

```text
scores_g in R^{B x H_g x M_g x M_g}
```

before score biases and causal masks are applied. Therefore the materialized
dense score-element count across batch and layers is

```text
A_score(L,B) = B K sum_g H_g M_g(L)^2.
```

This is a full-square count. The implementation does not store triangular
causal matrices, so a triangular-storage count would be incorrect for this
branch.

For a coarse-head fraction `p=H_c/H`, with fine stride `s_f=1` and coarse
stride `s_c=2`,

```text
A_score / (B K H L^2)
  = (1-p) / s_f^2 + p / s_c^2 + O(1/L)
  = 1 - 3p/4 + O(1/L).
```

The leading coefficients are:

| Row | Leading coefficient |
|---|---:|
| b0 | 1.0000 |
| b25 | 0.8125 |
| b50 | 0.6250 |
| b75 | 0.4375 |
| b100 | 0.2500 |

Replacing one fine head by one coarse head changes the finite score count by

```text
Delta = B K (M_c^2 - M_f^2).
```

For every registered length `L >= 4`, `M_f=L-1` and `M_c <= L/2-1`, hence
`M_c < M_f` and `Delta < 0`. Coarser blur strictly decreases the exact dense
score tensor count, while every row remains `Theta(B K H L^2)` because at
least one active group has `M_g = Theta(L)`.

## Activation And Tensor Scaling

The score tensor is the leading exact-dense allocation:

```text
scores, attention probabilities: O(B sum_g H_g M_g^2) per layer.
```

Other forward tensors are lower order in `L` for fixed `D,H,K`:

```text
q,k,v,out set states per layer: O(B sum_g M_g D_g)
pooled/set states across streams: O(B sum_g M_g D_g)
content feature phi_attn: O(B sum_g M_g D_g)
router candidate scores: O(B L sum_g H_g C_g)
```

For `candidate_fiber=endpoint_window`, the structural candidate count per token
is bounded by a small constant determined by `(w_g,s_g)`:

```text
C_f <= 2, C_c <= 2
```

before `router_topk=16`, so router candidate-gather scores are linear in `L`
under the active endpoint-window path. This statement is not true for
`all_past` or dense router scoring, which are outside the active branch.

The set-state linear allocation term used by the empirical model is

```text
A_linear(L,B) = B K sum_g M_g(L) D_g.
```

For registered even `L`,

```text
sum_g M_g D_g
  = (1-p)(L-1)D + p(L/2-1)D.
```

This decreases with blur but is lower order than the score term.

## Exact Parameter Accounting

The blur rows are not parameter-identical. Stream width changes with the head
split, and mixed rows instantiate two independent streams.

Let `V=76618` be the active WikiText-2 vocabulary size observed in current
artifacts, `N=128` the hashed-count bin count, and `L` the configured
`max_seq_len`. With anchor disabled, token MLP disabled, and `set_output_proj`
an identity because the concatenated stream width is `D`, the shared base is

```text
P_base(L,V) = 2 V D + L D
```

for token embeddings, untied LM head, and position embeddings.

For one group with width `d=D_g`, head count `h=H_g`, and set count `M=M_g(L)`,
the implemented inference parameter count is:

```text
P_input(d)   = 0 if d=D else Dd+d
P_block(d)   = K [4d^2 + 2 d d_ff + d_ff + 9d]
P_feature(M,d)
             = Md + 5d^2 + (2N+6)d
P_adapter(d) = 2d^2
P_router(h,d)= h d (D+d+2)
P_group      = P_input + P_block + P_feature + P_adapter + P_router.
```

The terms correspond respectively to the per-group input projection, six
`SetAttentionBlock`s, `HashedCountFeatureBuilder`, the auto-selected linear
content-bias adapter, and multihead learned router. The router formula follows
the current code: `d_phi` is the stream `d_phi`, so the query/key projections
output `h*d_phi_g`, not merely `h*d_head`.

The exact active total is:

```text
P_runtime(L,V) = P_base(L,V) + sum_g P_group(M_g(L),D_g,H_g).
```

Training and inference parameter counts are equal in the active runs because
`anchor.enabled=false`; the anchor pre-encoder is not constructed.

Concrete counts for `V=76618`:

| L | b0 | b25 | b50 | b75 | b100 |
|---:|---:|---:|---:|---:|---:|
| 512 | 73,380,480 | 70,690,560 | 69,725,184 | 70,641,408 | 73,282,176 |
| 1024 | 73,773,696 | 71,059,200 | 70,069,248 | 70,960,896 | 73,577,088 |
| 2048 | 74,560,128 | 71,796,480 | 70,757,376 | 71,599,872 | 74,166,912 |
| 3584 | 75,739,776 | 72,902,400 | 71,789,568 | 72,558,336 | 75,051,648 |
| 4096 | 76,132,992 | 73,271,040 | 72,133,632 | 72,877,824 | 75,346,560 |

The U-shape is expected: score memory decreases monotonically with coarse
heads, but parameter count also reflects stream splitting, feature projections,
and router projections.

## Token Dense Baseline Comparison

Matched exact token attention has `L` tokens and materializes full dense token
attention scores per transformer layer:

```text
A_token_score(L,B) = B K H L^2.
```

The set rows have leading score coefficient `1-3p/4` relative to this token
score count, plus finite `O(1/L)` corrections from windows. This is a
constant-factor reduction inside an exact-dense `Theta(L^2)` family. The
analysis does not imply subquadratic complexity and does not make b0
equivalent to token attention: b0 still pools `(w,s)=(2,1)` sets and routes
through set atoms.

## Peak-VRAM Interpretation Limits

The theorem above determines only tensor cardinalities and exact parameter
counts. Peak VRAM also depends on allocator state, dtype, autograd saved
tensors, optimizer state, CUDA kernels, dropout/softmax workspaces, data
loader behavior, fragmentation, and external process occupancy.

Therefore:

- a token OOM is a fixed `(L,batch,hardware,implementation,admission)` result,
  not an intrinsic impossibility theorem for the context length;
- legacy OOMs without archived external-process telemetry are observed
  feasibility outcomes, not primary certified capacity inequalities;
- B3, B4, B16, blue-demon, and lizmark are separate empirical strata;
- VRAM cannot be used to infer wall-clock speed or asymptotic speedups.

The primary empirical model, once MRP-1 closes, should be fit only within the
registered lizmark `B=4` stratum:

```text
V_peak =
  alpha
  + beta_linear B K sum_g M_g D_g
  + beta_score B K sum_g H_g M_g^2
  + beta_param P_runtime(L,V)
  + error,
```

with nonnegative coefficients. B3, B16, and blue rows are held-out/descriptive
checks, not additional independently fit strata.

## Pareto And Frontier Definitions

MRP-1 uses an within-island memory-quality frontier. An island fixes dataset,
tokenizer, objective, architecture width/depth, backend family, `L`, native
batch, effective batch, optimizer, LR, warmup, training-token budget, seed
contract, hardware class, and metric implementation.

For rows `a` and `b` in the same island, row `a` Pareto-dominates `b` on
mean PPL and peak VRAM when:

```text
mean_ppl(a) <= mean_ppl(b)
mean_peak_vram(a) <= mean_peak_vram(b)
```

and at least one inequality is strict. The nondominated rows form the empirical
frontier. OOM rows can define observed feasibility only under their admission
status; they do not receive a finite PPL/VRAM point unless the run completes.

The analytic memory frontier is the ordered set of blur rows by exact count
tuple, for example

```text
(A_score, A_linear, P_runtime)
```

at fixed `(L,B,V)`. This analytic frontier explains monotone score-memory
pressure with blur. It does not select the lowest-PPL blur row and does not
claim an interior quality optimum; that selection remains empirical and, for
MRP-6C, mechanistic.

## Pending Empirical Replacement Closure

Pending after MRP-1 closure:

1. fit the lizmark `B=4` nonnegative VRAM model using successful regular blur
   cells at `L in {2048,3584,4096}`;
2. use only admission-certified exclusive OOMs as primary censored capacity
   observations;
3. bootstrap cell-level uncertainty and report leave-one-length-out residuals;
4. report B3, B16, and blue residuals without refitting;
5. replace provisional/frozen tables with final MRP-1 rows only after strict
   validation accepts the 15 replacement cells.
