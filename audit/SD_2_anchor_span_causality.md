# SD-2 Anchor-Span Causality Gate

Status: PASS

Branch: `set-dictionary/anchor-span`

Implementation summary:

- Added `output_residual_mode="anchor_span"` to `SetOnlyLM`.
- `anchor_span` computes `f_t = emb(x_t) + pos_t + span_t`, where `span_t` is the existing routed set output after `set_output_proj`.
- Added a training-only shallow causal pre-encoder for the predictive anchor target. It is constructed only when `model.anchor.enabled=true`; S1 (`anchor.enabled=false`) carries no pre-encoder parameters.
- Added anchor auxiliary loss plumbing:
  - `anchor_loss = lambda_h * MSE(LN(span_t), stopgrad(LN(h_t*)))`
  - logs `anchor_loss`, `anchor_mse`, and `anchor/recon_error_norm` during LM training.
- Added config/default/schema/compatibility keys for `anchor`, `set_diversity`, `multivector_basis`, and `candidate_fiber`.
- `candidate_fiber` remains locked to `endpoint_window`; wider fibers are schema-known but rejected by compatibility until a later gate.
- `multivector_basis` remains deferred and must stay disabled with `r=1`.

Container checks on blue-demon:

- PASS: `docker compose exec -T set-attention python tests/test_output_residual_mode.py`
- PASS: `docker compose exec -T set-attention python tests/test_hyperparameter_propagation.py`
- PASS: `docker compose exec -T set-attention python tests/test_set_dictionary_causality.py`
- PASS: existing dense/sparse/landmark future-perturbation probes from `tests/test_causality.py`
- PASS: tiny CPU training-loop smoke with `anchor.enabled=true` produced nonzero `anchor_loss` and `anchor/recon_error_norm`.

Local checks:

- PASS: `python -m py_compile src/models/set_only/set_only_lm.py src/config/normalize.py src/config/schema.py src/config/compatibility.py scripts/run_experiment.py src/train/loop.py src/train/experiment_logger.py src/train/metrics_schema.py tests/test_output_residual_mode.py tests/test_hyperparameter_propagation.py tests/test_set_dictionary_causality.py`

Notes:

- The pre-encoder is not run during evaluation/inference; it is only used for training loss computation and explicit causality tests via `compute_anchor_target()`.
- `output_residual_mode=anchor_span` requires `token_mlp.enabled=false` in compatibility validation to preserve the thin-anchor fairness constraint.
