# A8 Hybrid Sparse Progressive Configs

These configs define the layer-level token/set hybrid sweep. They are active
post-A1 causal LM configs and should be treated as the source of truth for
model, data, and training hyperparameters.

Launcher rule:

- Keep paper-relevant hyperparameters in YAML.
- The launcher may override only per-run provenance values:
  - `training.seed`
  - `training.output_dir`
  - `logging.wandb.run_name`
  - `logging.wandb.project`
  - `logging.csv.path` / `--csv-path`
- Do not duplicate model topology, pooling, router, feature, data length, LR,
  or epoch values in shell scripts.

Current operating point:

- `model.implementation: hybrid_token_set`
- `attention_family/backend: sparse/local_band`
- `D=384`, `d_ff=1536`, `L=512`
- `set_causality_mode: strict_past`
- `output_residual_mode: empty_only`
- set layer topologies use near-2 and near-4 compression:
  - `(w,s)=(4,2)`
  - `(w,s)=(8,4)`
- `pooling.mode: soft_trimmed_boltzmann`
- `pooling.tau: 0.1`, `pooling.q: 0.85`, `pooling.alpha: 10.0`
- `feature_mode: hashed_counts`
- `feature_params.hash_seed: 13`
- `feature_params.normalize: true`
- `feature_params.num_bins: 128`
- `router.min_temp: 0.5`
- `d_phi: 384`
- `set_state_dim: 384`
