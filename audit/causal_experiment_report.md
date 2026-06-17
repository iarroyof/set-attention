# Stabilized Post-A1 Causal LM Experiment Report

Overall status: PASS

This report consolidates the post-A1 causal LM experiment record so the work is not lost even if the current SKA implementation is not presented as a perplexity improvement.

## Reporting Principles

- Use strict-past, T1 dropped trailing windows, and explicit output residual policy labels.
- Distinguish dense-baseline-only historical artifacts from v2.7 matched token-backend controls.
- Treat A6 capacity sweeps as diagnostics, not as evidence of a simple missing-capacity fix.
- Treat A7 as empirical convergence under the calibrated set pipeline, not exact Transformer equivalence.
- Preserve source CSV/JSON hashes and manifest status for every reported table or figure.
- Exclude pre-A1, noncausal, and causality-unverified artifacts from reviewer-facing causal LM claims unless they are rebuilt and revalidated under the post-A1 causal LM protocol.

## Artifact Index

| Phase | Status | Rows | Intended paper use | Caveat |
| --- | --- | ---: | --- | --- |
| A2 | pass | 153 | LR-normalized dense baseline and set-family provenance; dense-baseline-only historical context. | Use matched v2.7 controls for reviewer-facing backend-family claims. |
| A2.4 | pass | 10 | Matched sparse and linear token-backend controls at the LR-normalized headline point. | Controls are at D=384,d_ff=1536,L=512,w=16,s=8 only. |
| A3.1 | pass | 18 | Fixed-stride window-size mechanism sweep for candidate-count effects. | Set-family mechanism evidence; not a headline token-baseline comparison. |
| A3.1-control | pass | 12 | Token sparse/linear overlays for the A3.1 window sweep. | Use to distinguish set mechanism from backend attribution. |
| A3.2 | pass | 15 | Pooling-temperature sweep with error bars across set families. | Mechanism evidence for pooling support and transport. |
| A3.3 | pass | 12 | Stride/candidate-count complement sweep. | Demoted complement; useful for topology diagnostics. |
| A4.1 | pass | 2 | Long-context smoke/proof of feasible batch policy. | Not a final comparison table. |
| A4.2 | pass | 12 | Long-context set-family quality/memory slice. | Pair with A4.2 controls for matched backend claims. |
| A4.2-control | pass | 2 | Long-context matched sparse/linear token controls. | Dense token baseline remains memory-heavy but strongest in PPL. |
| A4.3 | pass | 7 | Thirty-epoch convergence panel. | Convergence favors token baselines under tested settings. |
| A6.1 | pass | 9 | d_phi set-token interface capacity ablation. | Moderate d_phi gains are family-specific, not a complete bottleneck fix. |
| A6.2 | pass | 12 | Explicit set-state dimensionality sweep at fixed token width. | Wider set state does not reliably close the PPL gap. |
| A6.3 | pass | 18 | Joint set-state and d_phi interface bottleneck sweep. | Matched d_phi=set_state_dim often worsens validation PPL. |
| A6.4 | pass | 18 | Set-stack depth bottleneck test. | Depth 8/10 worsens validation PPL versus depth 6. |
| A7 | pass | 7 | Calibrated empty_only token-limit and compression path. | Empirical convergence as M/L->1, not exact Transformer equivalence. |

## Writing Caveats

- The current causal SKA implementation is strongest as a diagnostic framework for compression, routing support, pooling support, and memory tradeoffs.
- Matched token baselines remain stronger in perplexity at the tested operating points.
- The A7 singleton limit shows SetDense `empty_only` approaching the dense token baseline but still trailing it.
- Long-context results support a memory advantage, not a quality advantage.
- Historical unreconciled appendix slices should be dropped or rebuilt from canonical LR-normalized artifacts.
