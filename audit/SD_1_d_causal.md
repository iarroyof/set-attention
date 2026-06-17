# SD-1 D-causal Flag Collapse

Status: PASS

Branch: `set-dictionary/anchor-span`

Prerequisite commit:

- `483464a` on `a9/candidate-gather-router` records the A9 candidate-gather prerequisite.

Implementation summary:

- `src/config/normalize.py` now removes legacy `model.causal` for set-only/hybrid configs, records a deprecation warning, and lets explicit `model.set_causality_mode` win.
- `src/config/schema.py` no longer treats `causal` as a set-only schema key.
- `src/config/compatibility.py` no longer uses `causal=False` to bypass strict-past LM validation.
- `src/models/set_only/set_only_lm.py` derives `self.causal` from `self.set_causality_mode`.
- `src/models/hybrid_token_set_lm.py` applies the same derivation for hybrid set layers and derives token-layer causal masking from the same mode.
- `scripts/run_experiment.py` no longer passes `causal` into set-only or hybrid model constructors.
- `configs/hyperparameters.md` documents `set_causality_mode` as the single source of truth.

Local checks:

- PASS: `python -m py_compile src/config/normalize.py src/config/compatibility.py src/config/schema.py src/models/set_only/set_only_lm.py src/models/hybrid_token_set_lm.py scripts/run_experiment.py tests/test_hyperparameter_propagation.py tests/test_output_residual_mode.py`
- PASS: direct-file normalization/schema smoke confirmed legacy `causal=False` is removed and does not override explicit `set_causality_mode=strict_past`.

Container checks on blue-demon:

- PASS: `docker compose exec -T set-attention python tests/test_output_residual_mode.py`
- PASS: `docker compose exec -T set-attention python tests/test_hyperparameter_propagation.py`
- PASS: dense/sparse/landmark future-perturbation probes from `tests/test_causality.py`

Local runtime limitation:

- Full runtime tests could not be run locally. Conda Python 3.13 has an incomplete namespace `torch` package with no `torch.nn`, and `/usr/bin/python3` has no `torch`.
- `docker` is not available locally.

Remaining gate work:

- Add the §4 set-dictionary leakage probe once `output_residual_mode="anchor_span"` and the anchor pre-encoder path exist. This is tracked as SD-2, not SD-1.
