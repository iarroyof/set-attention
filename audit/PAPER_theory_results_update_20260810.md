# Paper Theory and Results Update Audit

Status: PASS.

Updated: 2026-08-13.

## Scope

This update makes the current paper self-contained without changing an
experiment or model implementation.  It places the theoretical background
under Model Overview, fixes LCA table layout, replaces per-seed L4096 traces
with seed-mean confidence bands, adds two missing mechanism figures, and
reconciles the canonical program trackers with completed evidence.

The background was subsequently corrected so that ``theoretical foundations''
means the originating or canonical prior work, not a reconstruction from this
project's notation.  Each imported concept is now stated before its local
specialization:

- Transformer scaled dot-product and multi-head self-attention: Vaswani et al.
  (2017);
- neural probabilistic language modeling: Bengio et al. (2003), with WikiText
  attributed to Merity et al. (2017);
- permutation-invariant set aggregation: Deep Sets (Zaheer et al., 2017);
- attention-based set interaction and PMA: Set Transformer (Lee et al., 2019);
- variable-granularity grouped memory: Area Attention (Li et al., 2019);
- content-cluster routing and latent-query readback, kept distinct: Routing
  Transformer (Roy et al., 2021) and Perceiver IO (Jaegle et al., 2022);
- MQAR and the natural-text AR-hit diagnostic: Zoology (Arora et al., 2024).

Candidate fiber, routing bandwidth, blur allocation, and the LCA marker-counting
task are explicitly marked as definitions introduced in this paper.  The paper
does not attribute its robust local pooling rule to Deep Sets or PMA, does not
claim sequence-level permutation invariance, and does not identify its
token-to-atom readback with Routing Transformer's online-clustering algorithm.

## Benchmark Mechanism Diagrams

The paper's benchmark background is now a full subsection with one code-faithful
TikZ diagram per task:

- WikiText-2 LM compares direct causal token attention with both the registered
  local set readback and the separately labeled global recipe-regression path,
  while retaining causal dense interaction inside both atom stacks;
- synthetic MQAR shows the earlier key--value region, repeated supervised
  queries, and the binding-transport requirement imposed by local readback;
- natural repeated-bigram AR is shown as an evaluation partition of the same LM
  checkpoints under token, local-set, and global-set paths, including
  fine/coarse span ablations rather than a new loss;
- LCA compares dense token reach, the failed local/final-query calibration, and
  the repaired b75 recipe with `all_past`, dense scores, full routing, prefix
  supervision, and zero dropout.

The diagrams no longer represent these cases as identical blocks with renamed
labels or as one-dimensional stage lists.  Every model column now uses the same
top-to-bottom computation direction.  Their common two-dimensional grammar
shows token rows fanning into overlapping windows, an upper fine-atom row and a
lower coarse-atom row contextualizing in parallel, separate token-query and
atom-candidate inputs to each group router, and the later group merge.  Fine
candidate rails always run above the fine row and coarse candidate rails always
run below the coarse row; their orthogonal paths are derived from the atom
coordinates rather than hard-coded page heights.  Dotted stubs mark
representative sealed atoms excluded from direct readback; amber and violet
rails mark local and all-past eligibility respectively.  Blue rounded boxes
identify query projections, dashed blue edges carry group queries, and thick
green/red edges carry routed group values to concatenation.  Thin pale pooling
edges and medium-weight causal set-stack edges remain visually subordinate.
Prefix supervision uses a magenta bus.  Explanatory boxes are outside model
frames, node spacing leaves visible connector segments, and a single terminal
arrow is used for each causal arc, removing ambiguous coincident arrowheads and
crossings without changing the represented computation.
The connectivity was then audited against the executed code.  Token controls now
show only the current layer state projecting to $q_t^\ell$, with the causal
states entering attention separately as $K_{\le t}^\ell,V_{\le t}^\ell$.  Every
set query box is connected to the current thin state $h_t^0$, and that state
also follows the explicit `anchor_span` residual path to the post-$W_o$
addition.  The generic $q_t^g$ label denotes the fine/coarse router's own
head-wise query projections.  Candidate rails denote eligible atom indices:
under the executed hashed-count feature path their keys are projected from
pre-stack pooled states and their routed values come from post-stack set states.
In particular, the local `(2,1)` and `(4,2)` fibers have at most two candidates,
so their configured top-k=16 is explicitly shown as nonbinding; the global
panels instead connect all sealed bank atoms to the group candidate rails,
mask future atoms, and use full routing.  MQAR highlights transport of one binding through an atom
bank, natural AR keeps three contextual paths above one common evaluator, and
LCA contrasts final-target supervision with a prefix-target fan-out.  The
benchmark prose, captions, and corresponding Results subsections explain these
task-specific visual paths directly.

Canonical editable source:
`docs/benchmark_task_diagrams.tikz.tex`.  Reproducible builder:
`scripts/build_benchmark_task_diagrams.sh`.  The builder creates a four-page
vector PDF and copies both PDF and TikZ source into the Overleaf-ready bundle.
The main paper compiler invokes this builder before `latexmk`, and float
barriers keep each diagram inside the subsection that defines its task.

## Natural-AR Global-Recipe Bridge

The 2026-08-12 bridge adds 12 separately labeled rows (token, b0, b25, b75;
seeds 0--2) under `all_past`, dense router scoring, full routing, and zero
dropout for set rows, with a matched zero-dropout token control.  All training
and evaluation rows ran on blue-demon.  The paper reports absolute AR and
non-AR NLL, paired sequence-block intervals, and the AR-minus-non-AR
difference-in-differences.

The claim is deliberately narrower than the original run summary.  Every set
row has higher absolute AR NLL than token, but b0/b25 DiDs versus token are near
zero, so most of the gap tracks overall language-model quality.  The
global-minus-local comparison changes candidate fiber, score mode, routing
bandwidth, and dropout together and is therefore called a routing-recipe
contrast, not a fiber-only effect.  Whole-group ablations follow the six-head
group in b25/b75 and do not identify the presence or absence of a dedicated
retrieval circuit.  The supported conclusion is a recipe-robust null for a
demonstrated set-specific repeated-bigram advantage; LCA aggregation success
does not transfer to this retrieval proxy.

## Vocabulary Contract

The paper now defines and keeps distinct:

- token attention, pooling, set attention, and routing;
- candidate reachability, post-score routing bandwidth, and score
  materialization mode;
- fine/coarse atom banks and blur head allocation;
- LM, MQAR, natural repeated-bigram AR, and LCA;
- endpoint and prefix supervision;
- NLL, PPL, peak allocated VRAM, Pareto dominance, and matched-quality
  tradeoffs.

`MRP-*` names experiment packages only.  They are not benchmark names.  The
registered WikiText-2 matrix uses dense all-position next-token supervision;
only the initial LCA calibration uses final-query-only endpoint supervision.

## Theory Correction

The exact-dense theorem now counts two executed tensors separately:

```text
A_set = B K sum_g H_g M_g(L)^2
A_route_dense = B L sum_g H_g M_g(L)
```

For coarse-head fraction `p`, their leading coefficients relative to full
token-length square score counts are respectively `1 - 3p/4` and `1 - p/2`,
up to `O(1/L)` boundary terms.  Both are quadratic constant-factor laws and
neither is a total peak-VRAM formula.  The local endpoint-window candidate
gather instead has bounded candidates per query and a linear router-score
count.  The paper does not claim subquadratic scaling, b0/token equivalence,
or an architecture-wide Pareto result independent of routing recipe and task.

## Evidence Map

### Table 6: LCA quality-memory frontier

Primary record: `audit/LCA_calibration_20260718.md`.

- L1024 blur and matched token controls:
  `out/lca_cmp/prefixblur/prefixblur_blue.tsv` and
  `out/lca_cmp/prefix3/prefix3_blue.tsv`.
- L2048 with-dropout rows:
  `out/lca_cmp/l2048budget/` and the registered audit summary.
- L4096 with-dropout trajectories:
  `out/lca_cmp/l4096trajectory/`.
- L2048 dropout-free confirmation:
  `out/lca_cmp/l2048nodrop3seed/l2048nodrop3seed_blue.tsv`.
- L4096 dropout-free confirmation:
  `out/lca_cmp/l4096nodrop/l4096nodrop_lizmark.tsv` and its eval curves.

The table has six declared columns for six data fields, reports MiB rather
than labeling it GiB, and states the one single-seed exception.

### Figure 10: L4096 validation trajectories

Generator: `scripts/plot_lca_l4096_trajectories.py`.

Inputs are the six with-dropout and six dropout-free eval-curve CSVs under
`out/lca_cmp/l4096trajectory/` and `out/lca_cmp/l4096nodrop/`.  Each family is
one seed-mean line with a pointwise 95% Student-t confidence band (`n=3`).  The
inset uses measured peaks from the same recipe pairs: token 33746 to 20766 MiB
and b75 24916 to 17925 MiB after removing dropout.

### Figure 11: L1024 blur quality-memory frontier

Generator: `scripts/plot_lca_blur_frontier.py`.

Inputs are 12 set rows (b25/b50/b75/b100 by three seeds) from
`prefixblur_blue.tsv` and three matched token rows from `prefix3_blue.tsv`.
The figure shows mean accuracy against mean peak VRAM with 95% Student-t
accuracy intervals.  It visualizes the non-monotone quality ordering used to
select b75; it is not a scale plot and does not replace Table 6.

### Figure 12: L1024 routing-bandwidth control

Generator: `scripts/plot_lca_topk_bandwidth.py`.

Inputs are all 21 rows of `out/lca_cmp/topksweep/topksweep_blue.tsv` plus the
three matched token rows.  The quality panel and memory panel answer a joint
mechanism question: accuracy rises toward full routing, while the measured b75
peak stays within 2347--2408 MiB because dense scores are allocated before
top-k pruning.

## Canonical Status Reconciliation

The main plan, phase tracker, MRP-2 plan/audit, LCA plan/story, and MRP-6B/6D
records now agree that:

- MRP-2 is complete and null for set-specific repeated-bigram advantage under
  both the registered local recipe and the separately labeled global bridge;
- MRP-lca-cmp and its WikiText-2 reverse bridge are complete;
- MRP-5 is dependency-unblocked but still needs explicit launch approval;
- landmark and sparse backends remain historical and cannot support current
  efficiency claims.

## Validation

Final validation record:

- direct MRP-6B and MRP-6C theory tests passed;
- Python compilation of all three LCA plot generators passed;
- the blur input has 12 registered set rows plus three matched token rows and
  the bandwidth input has 21 registered set rows plus three token rows;
- `scripts/compile_paper_bundle.sh` produced the 60-page final PDF in
  `out/final_paper_bundle/checks/compile_logs/run_1QgOkn/` with no undefined
  citation/reference, overfull box, or TeX error (underfull page-layout notices
  only);
- the canonical diagram source and the Overleaf-ready source are byte-identical,
  as are their four-page vector PDFs;
- Model Overview, the four benchmark diagrams in their final paper placement
  on pages 6--9,
  the natural-AR bridge and Table 5, Table 6, Figures 10--12, and the completed
  NeurIPS checklist were visually inspected without clipping or overlap;
- the rendered PDF contains no unresolved paper placeholder;
- `git diff --check` passed.

No experiment was launched and no scientific artifact was reclassified.
