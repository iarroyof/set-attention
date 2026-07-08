# MRP-6A Multiresolution Formal Model

Status: MRP-6A proof memo package.

Scope: code-faithful formal definitions and proofs for the active exact-dense
multiresolution set-dictionary model with `output_residual_mode=anchor_span`,
`candidate_fiber=endpoint_window`, token MLP disabled, trained anchor disabled,
and CE-only readout. This memo is not canonical TeX; MRP-6D owns integration.

## 1. Code Correspondence Ledger

Primary locked configuration:

- `configs/set_dictionary/sd9_multiresolution.yaml`: exact backend, `D=384`,
  `H=8`, `K=6`, fine `(w,s,H)=(2,1,6)`, coarse `(4,2,2)`,
  `output_residual_mode=anchor_span`, `candidate_fiber=endpoint_window`,
  `token_mlp.enabled=false`, `anchor.enabled=false`, `feature_mode=hashed_counts`.
- `src/models/set_only/banks.py`: strict-past bank construction,
  `M=floor((L-w)/s)+1`, starts, endpoints, and endpoint-window fibers.
- `src/models/set_only/set_only_lm.py`: token MLP guard, causality/fiber guards,
  multiresolution stream construction, per-group width accounting, causal set
  stack mask, concatenation, output projection, and `anchor_span` addition.
- `src/models/set_only/router.py`: learned router candidate gather, empty-fiber
  zero weights, top-k masking, per-group multihead routing, and routed vector
  formation.
- `src/models/set_only/ska_block.py`: pre-LN set-attention residual block and
  set FFN.
- `src/set_attention/features/hashed_counts.py`: hashed-count and geometry
  descriptors used by the set-attention content-bias path.

The formal model below intentionally distinguishes router heads from
set-attention heads. In the active configuration both counts are supplied by
`num_heads`, but the proofs use router heads for routing-capacity statements and
set-attention heads only inside the causal set-stack operator.

## 2. Indices, Widths, And Resolution Groups

Let the token sequence be `x_0,...,x_{L-1}`. Let `D` be the model width and let
`H` be the total number of router heads. A finite set of resolution groups is
`G`. Each group `g in G` has:

- a window `w_g >= 1` and stride `s_g >= 1`;
- a nonempty disjoint router-head set `U_g`;
- head count `H_g=|U_g|`;
- `sum_g H_g = H`;
- stream width `D_g = D_set H_g / H`, where `D_set` is the configured
  `set_state_dim`;
- router feature width `P_g = d_phi H_g / H`.

The code enforces `sum_g H_g=H`, `D_set % H=0`, `D_g>0`,
`D_g % H_g=0`, and `P_g>0`; invalid cases raise construction errors. In the
active runs `D_set=D=384`, so concatenating all group routes returns width
`sum_g D_g=D`, and `W_o` is the identity. The definitions keep `W_o` explicit
because the implementation supports `D_set != D`.

The active groups are:

```text
fine:   w_f=2, s_f=1, H_f=6, D_f=288
coarse: w_c=4, s_c=2, H_c=2, D_c=96
```

## 3. Strict-Past Window Banks

For a group `g`, define the strict-past starts, endpoints, and sets by

```text
M_g = 0,                                      if L < w_g,
M_g = floor((L - w_g) / s_g) + 1,             if L >= w_g,
a_m^g = m s_g,
e_m^g = a_m^g + w_g - 1,
S_m^g = {a_m^g, a_m^g+1, ..., e_m^g},
```

for `m=0,...,M_g-1`. These are exactly the `strict_past` starts and endpoints
created by `build_window_bank`. Partial trailing windows are dropped.

The endpoint-window candidate fiber at token `t` is

```text
C_t^g = {m: t - w_g < e_m^g <= t}.
```

Write `c_t^g=|C_t^g|`. The active configuration uses `s_g <= w_g` for both
groups, which is the standing assumption for the exact/interior count lemmas.

## 4. Pooling, Features, Set Stacks, And Routing

The thin anchor is

```text
h_t^0 = e(x_t) + p_t.
```

With token MLP disabled, the input to pooling is `h_t^0`; the constructor
replaces the token MLP with the identity. In group `g`, pooled atoms are

```text
z_m^{0,g} = P_g({h_j^0: j in S_m^g}),
```

where `P_g` is the configured pooling operator. For
`soft_trimmed_boltzmann`, the implementation computes a differentiable trimmed
Boltzmann weighted average over the tokens in `S_m^g`; for proofs here, only
the support property `z_m^{0,g}` depends on `x_j` with `j <= e_m^g` is needed.

The group also constructs endpoint-local geometry and content descriptors:

- geometry uses only set positions and endpoint/order geometry inside the bank;
- hashed counts are computed from token IDs in `S_m^g`, normalized by set size
  in the active config;
- `phi_attn` and attention content-bias tensors are deterministic functions of
  these group-local descriptors and pooled states.

Let `Phi_g` denote the `K`-layer causal set stack. It maps pooled atoms to
processed atoms

```text
z^{K,g} = Phi_g(z^{0,g}, descriptors_g).
```

The code applies a causal set mask

```text
set_positions_i >= set_positions_j
```

before each set-attention block. Because positions increase with endpoints,
atom `i` may attend only to atoms `j` with `e_j^g <= e_i^g`. The block then
applies the residual set-attention update and the residual set FFN.

For a router head `u in U_g`, the learned router forms scores only over
`C_t^g` in `candidate_gather` mode, applies optional top-k masking within that
candidate list, divides by an effective temperature bounded below by
`min_temp`, and softmaxes. Let

```text
pi_{t,m}^{u,g} >= 0,    sum_{m in C_t^g} pi_{t,m}^{u,g}=1
```

when `C_t^g` is nonempty. If `C_t^g` is empty, the implementation has no finite
row; it replaces the row's softmax output by zeros. The head route is

```text
r_t^{u,g} = sum_{m in C_t^g} pi_{t,m}^{u,g} slice_u(z_m^{K,g}),
```

and is zero when `C_t^g` is empty. The routed group vector is

```text
r_t^g = Concat_{u in U_g} r_t^{u,g} in R^{D_g}.
```

The full routed context and span are

```text
r_t = Concat_{g in G} r_t^g in R^{D_set},
span_t = W_o r_t in R^D.
```

The active readout is

```text
f_t = h_t^0 + span_t,
ell_t = W_lm f_t,
p(x_{t+1}|x_<=t) = softmax(ell_t).
```

There is no direct token-state residual and no `empty_only` branch in this
model. Empty-fiber repair occurs inside each group router: an empty group
contributes a zero routed group vector, while nonempty groups still contribute
their spans and the thin anchor is always present.

## 5. Bank Lemmas

### Lemma 1: Empty-Fiber Characterization

For strict-past endpoint-window group `g`, `C_t^g` is empty exactly when there
is no integer `m` satisfying

```text
0 <= m <= M_g-1
and
t - w_g < m s_g + w_g - 1 <= t.
```

Equivalently, if `M_g>0`,

```text
ceil((t - 2w_g + 2)/s_g) <= m <= floor((t - w_g + 1)/s_g)
```

has no solution inside `[0,M_g-1]`. In particular, all `t < w_g-1` are empty,
and `t=w_g-1` is nonempty with candidate `m=0`.

Proof. Substitute `e_m^g=m s_g+w_g-1` into
`t-w_g < e_m^g <= t` and rearrange the two inequalities. The early-token
claim follows because the first endpoint is `w_g-1`. At `t=w_g-1`, endpoint
`e_0^g=w_g-1` satisfies `-1 < e_0^g <= w_g-1`. ∎

### Lemma 2: Exact Candidate Count

Assume `M_g>0`. Define

```text
lo_t^g = max(0, ceil((t - 2w_g + 2)/s_g)),
hi_t^g = min(M_g-1, floor((t - w_g + 1)/s_g)).
```

Then

```text
c_t^g = max(0, hi_t^g - lo_t^g + 1).
```

Proof. Lemma 1 gives exactly the feasible integer interval for candidate
indices. Intersecting that interval with valid bank indices gives
`[lo_t^g, hi_t^g]`. The number of integers in a nonempty closed integer
interval is `hi-lo+1`; otherwise it is zero. ∎

### Lemma 3: Interior Candidate Count And Endpoint Locality

Assume `s_g <= w_g` and `t` is an interior token for which the truncations in
Lemma 2 are inactive:

```text
lo_t^g = ceil((t - 2w_g + 2)/s_g),
hi_t^g = floor((t - w_g + 1)/s_g).
```

Then

```text
c_t^g =
floor((t - w_g + 1)/s_g) - ceil((t - 2w_g + 2)/s_g) + 1,
```

and every candidate endpoint lies in the endpoint-local interval

```text
e_m^g in {t-w_g+1, ..., t}.
```

Consequently `c_t^g` is either `floor(w_g/s_g)` or `ceil(w_g/s_g)` depending
on the phase of `t` modulo `s_g`; in the active groups it is exactly `2` for
all interior tokens in both fine `(2,1)` and coarse `(4,2)` streams.

Proof. The exact formula follows by removing the boundary clamps from Lemma 2.
Endpoint locality is the defining inequality `t-w_g < e_m^g <= t` with integer
endpoints. The interval has length `w_g` and contains endpoints from the stride
grid `w_g-1 + s_g Z`; a length-`w_g` integer interval intersects such a grid
in either `floor(w_g/s_g)` or `ceil(w_g/s_g)` points when `s_g <= w_g`. For
the active pairs, `w_g/s_g=2`, so every fully interior window contains exactly
two stride-grid endpoints. ∎

This proof repairs the legacy error that treated the first endpoint as present
in every later fiber. The fiber moves with `t`; only the most recent stride-grid
endpoints in `(t-w_g,t]` remain candidates.

## 6. Causal Closure Theorem

Theorem. For every group `g`, layer `k`, atom `m`, and token `t`:

1. `z_m^{0,g}` is measurable with respect to `sigma(x_0,...,x_{e_m^g})`.
2. `z_m^{k,g}` is measurable with respect to `sigma(x_0,...,x_{e_m^g})`.
3. The routed group vector `r_t^g` is measurable with respect to
   `sigma(x_0,...,x_t)`.
4. The logits `ell_t` are measurable with respect to `sigma(x_0,...,x_t)`.

Therefore the model is next-token causal. The conclusion permits dependence on
the current input token `x_t`; it excludes only future tokens.

Proof. The pooled atom `z_m^{0,g}` is computed from anchors in `S_m^g`, whose
largest token index is `e_m^g`; the pooling weights are deterministic
functions of the same anchors. Hashed-count features use only token IDs in
`S_m^g`, and geometry features use only bank positions/endpoints, so the
content and geometry descriptors are also endpoint-measurable.

Induct on set-stack layer `k`. The claim is true for `k=0`. At layer `k+1`,
the causal set-attention mask permits atom `m` to consume only atoms `j` with
`e_j^g <= e_m^g`. By the induction hypothesis, each such source atom is
measurable with respect to a sub-sigma-field of
`sigma(x_0,...,x_{e_m^g})`. The attention scores, geometry bias,
hashed-count content bias, residual addition, layer norms, dropout-disabled
evaluation map or fixed training randomness, and set FFN are deterministic
functions of these endpoint-measurable variables. Thus
`z_m^{k+1,g}` is endpoint-measurable.

For routing at token `t`, every candidate satisfies `e_m^g <= t`. Router
queries use the current token state derived from `h_t^0`, and keys/descriptors
and values are candidate atoms whose endpoints are at most `t`. Thus scores,
masked softmax weights, and the weighted sum are `sigma(x_0,...,x_t)`
measurable. Empty fibers produce the zero vector, which is deterministic.
Concatenation over groups, `W_o`, anchor addition, and the LM head are
deterministic maps, so the logits are causal. ∎

## 7. Contextual-Path Factorization

Theorem. Consider two histories `x` and `x'` with the same current token and
position at `t`: `x_t=x'_t` and the same positional index `t`. In the active
model with token MLP disabled and trained anchor disabled,

```text
ell_t(x) - ell_t(x') = W_lm W_o (r_t(x) - r_t(x')).
```

Consequently every dependence of `ell_t` on `x_<t` factors through the routed
span `span_t=W_o r_t`.

Proof. The readout is `ell_t(x)=W_lm(h_t^0(x)+W_o r_t(x))`. Equal current token
and equal position imply `h_t^0(x)=h_t^0(x')`. Subtracting the two logit
vectors cancels the anchor and leaves the stated identity. ∎

Observability limitation. Softmax probabilities cannot distinguish span
differences mapped by `W_lm W_o` to a constant-logit vector `c 1`, because
softmax is invariant under adding the same constant to all vocabulary logits.
This is a factorization statement, not an unqualified identifiability theorem.

Span-ablation corollary. If evaluation sets `r_t:=0` for all `t`, all
historical-context paths are removed from the readout. The remaining model is
`ell_t=W_lm h_t^0`, which still depends on the current token and position; it
should not be called a pure unigram model.

## 8. Multigroup Routing Capacity

For each group, let `Pi_t^g` be the matrix with rows indexed by router heads
`u in U_g` and columns indexed by atoms in `C_t^g`, containing
`pi_{t,m}^{u,g}`. For empty fibers, `Pi_t^g` has zero columns and rank zero.

Define the tagged disjoint atom universe

```text
Omega_t = disjoint union over g of ({g} x C_t^g).
```

The block-supported routing matrix `Pi_hat_t` has rows indexed by all router
heads and columns by `Omega_t`; row `u in U_g` is zero outside the block
`{g} x C_t^g` and equals `Pi_t^g` inside that block.

Theorem.

```text
rank(Pi_hat_t) = sum_g rank(Pi_t^g)
              <= sum_g min(H_g, c_t^g).
```

Proof. After permuting rows and columns by group, `Pi_hat_t` is block diagonal
with diagonal blocks `Pi_t^g`. Rank is additive over block diagonal matrices,
which proves the equality. Each block has at most `H_g` rows and `c_t^g`
columns, so `rank(Pi_t^g) <= min(H_g,c_t^g)`. Summing gives the bound. ∎

Product-simplex dimension. The admissible assignment space at token `t` is

```text
prod_g prod_{u in U_g} Delta(C_t^g).
```

Its relative interior has dimension

```text
sum_g H_g (c_t^g - 1)
```

over nonempty fibers, with zero contribution from empty or singleton fibers.
Finite-temperature dot-product routers parameterize smooth subsets of this
space; limits can approach boundary Dirac laws, but finite temperature does not
realize every boundary distribution exactly. Row-wise top-k limits each row's
support and entropy. It does not, by itself, bound the matrix rank by `k`.

## 9. Per-Group Diagnostics

Entropy and top-1 diagnostics are group-local because different groups route
over incompatible atom spaces. For `u in U_g` and nonempty `C_t^g`,

```text
H_t^{u,g} = - sum_{m in C_t^g} pi_{t,m}^{u,g} log pi_{t,m}^{u,g}
          <= log min(c_t^g, k)
```

when top-k is active with `k < c_t^g`, and `<= log c_t^g` otherwise. Top-1
weight and top-1 gap are likewise computed inside `C_t^g`. Aggregating
probability vectors across groups is meaningful only after passing to the
tagged disjoint union `Omega_t`; otherwise the atom labels are not comparable.

## 10. Legacy Statement Replacement Notes

Replace or restate:

- routing entropy: keep only the per-group/fiber statement;
- multihead capacity: replace the single-bank rank ceiling by the tagged
  block-rank theorem above;
- finite-temperature realization: qualify boundary/Dirac claims as limits;
- empty-fiber proof: use the endpoint-grid interval in Lemmas 1-3;
- context path: replace direct/empty-only residual statements by
  `anchor_span` factorization and span-ablation corollary.

Retain outside MRP-6A scope pending MRP-6D audit:

- pooling maximum-entropy theorem;
- pooling gradient/transport theorems;
- empirical feasible-region claims, after multigroup constants are threaded
  through by the appropriate theory task.
