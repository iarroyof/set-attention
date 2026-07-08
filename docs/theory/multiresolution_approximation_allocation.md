# MRP-6C Multiresolution Approximation And Allocation

Status: analytic MRP-6C proof memo package; empirical specialization pending
MRP-3.

Updated: 2026-07-07.

This memo is a conditional proof package for the active exact-dense
multiresolution set-dictionary model. It uses the objects defined in
`docs/theory/multiresolution_formal_model.md` and the score-memory law in
`docs/theory/exact_dense_memory_frontier.md`. It is not canonical TeX; MRP-6D
owns integration.

The results below do not claim that language decomposes into smooth
low-frequency and high-frequency signals. They also do not select b25, b50, or
any blur row from rank, topology, or memory alone. They give conditions under
which a mixed fine/coarse allocation can have lower memory and no worse
approximation error than a uniform endpoint, and they state which premises
need empirical support.

## 1. Code-Faithful Objects

Let `G` be the set of resolution groups. Group `g` has window/stride
`(w_g,s_g)`, router-head set `U_g`, head count `H_g`, stream width `D_g`, bank
size `M_g`, endpoint-window fiber `C_t^g`, processed atoms `z_m^{K,g}`, and
routing laws `pi_t^{u,g}` for heads `u in U_g`. The routed group and span are

```text
r_t^{u,g} = sum_{m in C_t^g} pi_{t,m}^{u,g} slice_u(z_m^{K,g}),
r_t^g     = Concat_{u in U_g} r_t^{u,g},
r_t       = Concat_g r_t^g,
span_t    = W_o r_t.
```

The active split is fine `(w,s,H,D)=(2,1,6,288)` and coarse
`(4,2,2,96)` with total `H=8`, `D=384`, `D_set=D`, exact dense set attention,
`candidate_fiber=endpoint_window`, token MLP disabled, trained anchor disabled,
and `output_residual_mode=anchor_span`.

## 2. Approximation Definitions

The theory is relative to a reference contextual operator family. For each
group `g`, token `t`, and head `u in U_g`, assume a reference atom family

```text
a_i^{u,g}(x) in R^{D_g/H_g},    i in I_t^g,
```

and a reference routing law

```text
rho_t^{u,g}(x) in Delta(I_t^g).
```

The implemented group bank is compared to the reference through an explicit
coupling

```text
Gamma_t^g subset C_t^g x I_t^g.
```

For this memo the simplest case is a bijection `kappa_t^g:C_t^g -> I_t^g`.
Many-to-one or partial couplings are also admissible if the suprema below are
taken over paired atoms and unmatched mass is charged to the transport term.

For a fixed token/history domain `X` and time set `T`, define:

```text
epsilon_g = sup ||slice_u(z_m^{K,g}(x)) - a_{kappa_t^g(m)}^{u,g}(x)||
B_g       = sup ||a_i^{u,g}(x)||
xi_g      = sup TV(kappa_{t#}^g pi_t^{u,g}(x), rho_t^{u,g}(x)).
```

The suprema range over `x in X`, `t in T`, `u in U_g`, and valid coupled atoms.
`TV(mu,nu)=0.5 ||mu-nu||_1` on the common finite reference index set. If a
fiber is empty, both implemented and reference laws are required to route the
zero vector for the bound to apply; otherwise the zero-vector mismatch is
included in `epsilon_g` or in an explicit empty-fiber residual.

Interpretation of quantities:

- Observable diagnostics: group bank sizes, fiber counts, route entropies,
  top-1 weights, top-1 gaps, span-ablation effects, per-group ablation
  effects, atom norms, pooled within-set diameters, and empirical transport
  proxies computed after choosing a diagnostic reference/coupling.
- Theoretical proxies: the reference atoms, exact coupling, `epsilon_g`,
  `xi_g`, local Lipschitz constants, and the decomposition of approximation
  error into per-group reference terms.
- Pooling distortion: a data-dependent within-set diameter or coupling error,
  for example
  `diam_m^g(x)=sup_{i,j in S_m^g} ||h_i^0(x)-h_j^0(x)||`, propagated through
  the pooling and set-stack Lipschitz constants. It is not a temporal
  smoothness or frequency assumption.

## 3. Per-Group Routed Approximation Lemma

Lemma. Fix group `g`, head `u`, token `t`, and history `x`. Assume
`kappa_t^g` maps implemented candidates to the reference index set,

```text
||slice_u(z_m^{K,g}) - a_{kappa_t^g(m)}^{u,g}|| <= epsilon_g,
||a_i^{u,g}|| <= B_g,
TV(kappa_{t#}^g pi_t^{u,g}, rho_t^{u,g}) <= xi_g.
```

Then the routed head error satisfies

```text
||r_t^{u,g} - r_{*,t}^{u,g}|| <= epsilon_g + 2 B_g xi_g,
```

where

```text
r_{*,t}^{u,g} = sum_{i in I_t^g} rho_{t,i}^{u,g} a_i^{u,g}.
```

Proof. Let `mu=kappa_{t#}^g pi_t^{u,g}`. Insert and subtract the coupled
implemented route over reference labels:

```text
r_t^{u,g} - r_{*,t}^{u,g}
 = sum_i mu_i (z_i-a_i) + sum_i (mu_i-rho_i) a_i,
```

where `z_i` denotes the implemented slice paired with `i`; duplicate coupled
mass is aggregated if the coupling is many-to-one. The first term has norm at
most `sum_i mu_i epsilon_g = epsilon_g`. The second term is bounded by

```text
sum_i |mu_i-rho_i| ||a_i|| <= B_g ||mu-rho||_1 = 2 B_g TV(mu,rho).
```

Using `TV(mu,rho)<=xi_g` gives the result. Endpoints with empty fibers are
covered by the stated zero-route or residual convention. QED.

Corollary. With

```text
delta_g = epsilon_g + 2 B_g xi_g,
```

the group-vector error obeys

```text
||r_t^g-r_{*,t}^g|| <= sqrt(H_g) delta_g
```

because `r_t^g` is the direct concatenation of `H_g` head slices.

## 4. Concatenated Span And Loss Bounds

Theorem. Under the lemma assumptions for every group and head,

```text
||Delta span_t||
 <= ||W_o|| [
      sum_g H_g (epsilon_g + 2 B_g xi_g)^2
    ]^(1/2),
```

where `Delta span_t = W_o(r_t-r_{*,t})`.

Proof. Direct-sum head geometry gives

```text
||r_t-r_{*,t}||^2 = sum_g ||r_t^g-r_{*,t}^g||^2
                 <= sum_g H_g delta_g^2.
```

Applying the operator norm of `W_o` proves the bound. QED.

### Two-Family Lipschitz Assumptions

The legacy one-sided proof treated only the implemented model as Lipschitz.
For a meaningful conditional loss comparison, both the implemented and
reference families must be locally controlled on the same neighborhood.

Assume:

1. Model-family readout stability. The implemented readout map
   `Q_theta(h_t^0,span_t)=CE(W_lm(h_t^0+span_t), y_t)` is locally Lipschitz in
   span with constant `L_model(t,x)` on a ball containing both implemented and
   reference spans.
2. Reference-family readout stability. The target/reference loss functional
   `Q_*(h_t^0,span_t)` is locally Lipschitz in its span argument with constant
   `L_ref(t,x)` on the same ball.
3. Calibration residual. At the reference span, the implemented and reference
   readouts differ by
   `b_t(x)=|Q_theta(h_t^0,span_{*,t})-Q_*(h_t^0,span_{*,t})|`.

Then

```text
|Q_theta(h_t^0,span_t) - Q_*(h_t^0,span_{*,t})|
 <= b_t(x) + L_model(t,x) ||Delta span_t||.
```

If the comparison is between two operators evaluated at their own perturbed
spans, a symmetric triangle inequality gives

```text
|Q_theta(h_t^0,span_t) - Q_*(h_t^0,span'_{*,t})|
 <= b_t(x)
    + L_model(t,x) ||span_t-span_{*,t}||
    + L_ref(t,x) ||span'_{*,t}-span_{*,t}||.
```

Thus any specialization claiming small excess loss must bound both the
implemented approximation error and any reference-family perturbation. No rank
or topology statement supplies these Lipschitz or calibration quantities by
itself.

For cross-entropy with a fixed linear LM head, one admissible span-Euclidean
constant is `L_model <= sqrt(2) ||W_lm||_2` because the gradient of
cross-entropy with respect to logits is `p-e_y`, whose Euclidean norm is at
most `sqrt(2)`. A sharper or normalized convention may be used, but it must be
stated with its norm.

## 5. Pooling Collision And Token-Recovery Boundary

### Identical Dictionaries Imply Identical Downstream Span

Theorem. Consider two histories `x,x'` with the same current token and
position at `t`. If their multigroup dictionaries are identical at every
downstream input consumed by the set stacks and routers, namely the same
processed atoms, descriptors, fibers, and router query inputs for all groups up
to time `t`, then every deterministic downstream stack/router produces the
same `r_t`, `span_t`, and logits.

Proof. The group stacks, routers, concatenation, `W_o`, anchor addition, and LM
head are deterministic maps of those inputs. Equal current token and position
make the thin anchors equal. Applying the same deterministic maps to equal
inputs gives equal spans and logits. QED.

Consequence. A deterministic downstream model cannot separate two histories
after the exact multigroup dictionary state it receives has collapsed them.
This is a structural non-recovery statement conditional on identical
dictionaries, not a claim that such collisions must occur on the discrete
vocabulary support.

### Continuous Dimension Boundary

Let `F` be the continuous pre-stack pooled-state map from an open subset of
`R^{L D}` to the concatenated group pooled states

```text
F: R^{L D} -> product_g R^{M_g D_g}.
```

If

```text
sum_g M_g D_g < L D,
```

then `F` has no global continuous left inverse on any open set.

Proof. If a continuous left inverse `G` existed on an open set `U`, then
`G(F(x))=x` for all `x in U`, so `F` would be a continuous injection from an
open subset of `R^{L D}` into `R^q` with `q < L D`. This contradicts
invariance of domain and the standard dimension obstruction for continuous
embeddings of open Euclidean sets into lower-dimensional Euclidean spaces.
QED.

Registered 6-fine/2-coarse calculation. For even `L >= 4`,

```text
M_f=L-1,         D_f=288,
M_c=L/2-1,       D_c=96,
sum_g M_g D_g = 288(L-1) + 96(L/2-1)
              = 336L - 384
              < 384L = L D.
```

The continuous pooled-state map is therefore dimension-compressing for the
active b25 split at registered lengths. This calculation assumes the map's
continuous input is the token-anchor state in `R^{L D}` and the output is the
concatenated pooled state before the set stacks. It does not treat discrete
token IDs, hashed-count descriptors, or geometry features as a continuous
invertible channel.

Limitations:

- The dimension theorem does not prove collisions on the finite discrete
  vocabulary domain; finite sets can inject into lower-dimensional Euclidean
  spaces.
- `M=L` alone does not imply token-attention equivalence. Pooling, set-stack
  processing, endpoint-window routing, head slicing, and the `anchor_span`
  readout can still differ from token attention.
- `(w,s)=(1,1)` alone also does not imply token-attention equivalence unless
  pooling, features, set stack, routing, residual paths, and readout are
  specialized to reproduce token attention.

## 6. Discrete Interior-Allocation Theorem

Fix total heads `H`, fine bank size `M_f`, coarse bank size `M_c`, and
`0 <= n <= H` coarse heads. Define

```text
A(n) = (H-n) M_f^2 + n M_c^2,
E(n) = E_f(H-n) + E_c(n) + E_int(n).
```

Here `A(n)` is the exact dense score-count factor up to the common multiplier
`B K`, and `E(n)` is an abstract approximation or validation error. The
terms may encode fine approximation, coarse approximation, and interaction
cost/benefit. They are not determined by rank alone.

### Memory Monotonicity

If `M_c < M_f`, then

```text
A(n+1)-A(n) = M_c^2 - M_f^2 < 0.
```

Thus score memory is strictly decreasing in the number of coarse heads. For
the active fine/coarse banks at registered lengths, `M_f=L-1` and
`M_c=floor((L-4)/2)+1`, hence `M_c<M_f`.

### Interior Error Minimizer

Let the discrete marginal quality gain of replacing the `n`th fine head by a
coarse head be

```text
G(n) = E(n-1) - E(n),       n=1,...,H.
```

Assume discrete diminishing returns:

```text
G(1) >= G(2) >= ... >= G(H).
```

Assume a sign change:

```text
G(1) > 0 and G(H) < 0.
```

Then there exists an interior minimizer `n* in {1,...,H-1}` of `E(n)`.

Proof. Since `G(1)>0`, `E(1)<E(0)`, so `n=0` is not a minimizer. Since
`G(H)<0`, `E(H)>E(H-1)`, so `n=H` is not a minimizer. The finite set
`{0,...,H}` has at least one minimizer, and neither endpoint can be one; hence
some minimizer is interior. Diminishing returns further implies the minimizers
form a contiguous plateau around the first index where `G(n)` stops being
positive. QED.

This theorem is an existence result. It does not locate b25 or b50 without
estimating the marginal gains under the fixed data, tokenizer, architecture,
optimization, and metric island.

### Pareto Conditions

For any baseline `m`, allocation `n` is Pareto-better in error-memory space
exactly when

```text
E(n) <= E(m) and A(n) < A(m),
```

or

```text
E(n) < E(m) and A(n) <= A(m).
```

Against all-fine `m=0`, every active mixed allocation `n>0` has lower score
memory, so the condition reduces to

```text
E(n) <= E(0) and A(n) < A(0),
```

or equivalently no greater error and strictly lower memory.

Against all-coarse `n=H`, a mixed allocation `n<H` is Pareto-better exactly
when it has no greater error and no greater memory with at least one strict
inequality. Since `A(n)>A(H)` whenever `n<H`, mixed rows can dominate
all-coarse only through strictly lower error together with an accepted memory
tradeoff definition, or in a multi-metric frontier where all-coarse is
infeasible or fails another constraint. Under the strict two-objective
`(E,A)` definition, higher memory prevents dominance of all-coarse.

### Rank Ceilings

MRP-6A gives the per-token routing-rank ceiling

```text
rank(Pi_hat_t) <= sum_g min(H_g, c_t^g).
```

For active interior tokens, both fine and coarse groups have `c_t^g=2`. The
registered blur ceilings are therefore:

```text
b0:   min(8,2) = 2
b25:  min(6,2)+min(2,2) = 4
b50:  min(4,2)+min(4,2) = 4
b75:  min(2,2)+min(6,2) = 4
b100: min(8,2) = 2
```

The ceiling distinguishes mixed rows from uniform endpoints, but it cannot
select among b25, b50, and b75 because their registered ceiling is equal. Any
claim that b25 is optimal must come from empirical loss/frontier evidence or
from additional assumptions on `E(n)`, not from this ceiling.

Blur-optimum movement with sequence length remains a conjecture unless dense
multi-length evidence establishes it under the registered statistical
contract.

## 7. MQAR Interpretation Limits

MRP-3 can test whether coarse contextualized atoms preserve distant discrete
associations and whether group ablation effects vary with lag. Such evidence
would support or refute specialization premises for `epsilon_g`, `xi_g`, and
the allocation marginal gains `G(n)` on the synthetic task.

MRP-3 must not be described as validating a high-frequency/slow-signal
language decomposition. Lag is not signal frequency, and a distant key-value
association can be discrete and non-smooth.

If MRP-3 is null, the conditional theorems above remain true, but the project
must state that the specialization premises for coarse preservation of
distant associations were not empirically established.

## 8. Legacy Theorem Disposition

| Legacy item | Disposition | Replacement or reason |
|---|---|---|
| Mechanistic decomposition / feasible-region theorem | Repair | Restate per group with `epsilon_g`, `xi_g`, `B_g`, direct-sum concatenation, and `anchor_span` readout. Require two-family Lipschitz assumptions before converting span error to loss error. |
| Pooling-gradient formulas omitting overlapping sets and anchor/router paths | Repair with named partial derivatives | Any gradient statement must distinguish pooling parameters, overlapping window memberships, set-stack derivatives, router-score derivatives, and the readout derivative through `W_lm W_o`. Pooling is one path, not the only path. |
| Jacobian sandwich assuming pooling is the only path | Remove or repair | Valid only for a deliberately ablated operator where set-stack, router, and anchor-span paths are frozen or named. It is not a theorem about the active model as written. |
| Inference from low effective support to large approximation error | Remove | Low support or low entropy can coexist with accurate routing if the selected atom is correct. Error requires a reference/coupling and a bound like `epsilon_g+2B_g xi_g`. |
| Token-limit or convergence statements | Remove from theory | OOM is a fixed hardware/software feasibility observation. Convergence and blur selection are empirical under the control tuple, not consequences of rank or topology alone. |
| Pooling transport theorem | Retain per group if assumptions are stated | It may be used as one source of `epsilon_g` through within-set diameter/coupling distortion, without temporal smoothness language. |
| Routing entropy and rank statements | Retain in MRP-6A form | They are per-group or tagged-disjoint-union structural statements and not perplexity theorems. |

## 9. Diagnostic Map

The approximation theorem needs, at minimum:

- chosen reference family and atom coupling;
- atom norm diagnostics or bounds for `B_g`;
- atom approximation diagnostics or proxies for `epsilon_g`;
- routing transport diagnostics or proxies for `xi_g`;
- local readout Lipschitz/calibration assumptions for loss conversion.

The allocation theorem needs:

- exact `M_f,M_c,H` counts from MRP-6B;
- empirical or task-specific estimates of `E(n)` or marginal gains `G(n)`;
- Pareto comparison inside a fixed control tuple.

The collision boundary needs:

- explicit definition of the dictionary state being compared;
- a discrete-domain caveat whenever token IDs, hashes, or finite vocabulary
  support are involved.
