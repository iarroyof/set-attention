# A1.3 Reconciliation: App D.2 786.3 vs Table 8 836.1

## Verdict

Both values are recoverable from repository artifacts. They are not the same
artifact:

- `786.3` comes from the older Action1A parity/support bundle.
- `836.1` comes from the later LR-normalized headline/Table 8 bundle.

The logged model/data/training settings that matter scientifically are the same
for both baseline runs: Wikitext-2 LM, `baseline_token`, dense exact causal
attention, `D=384`, `d_ff=1536`, 6 layers, 8 heads, sequence length 512, batch
size 16, seed 0, 10 epochs, LR `1e-4`, warmup 1000, dropout settings 0.1, and
vocab size 76618.

The concrete recoverable difference is run/artifact provenance, not an
effective model configuration difference. The two CSVs have different run IDs,
config fingerprints, output roots, hashes, and summary pipelines. The
fingerprint difference is explained by logged metadata such as W&B and
`training.output_dir`; no logged architecture/data/training hyperparameter
difference explains the PPL gap.

Recommendation: do not present App D.2 as the same baseline as Table 8. Use the
LR-normalized Table 8 artifact as canonical for the matched LR-normalized
comparison, and drop App D.2 or rewrite it as a clearly labeled historical
Action1A parity rerun. The safer paper action is to drop App D.2 unless it is
rerun/rebuilt from the canonical Table 8 baseline artifact.

## Value Provenance

### App D.2 / parity value: 786.3

- Rounded value: `786.3`
- Exact value: `786.3329467773438`
- Source CSV: `out/metrics/action1A_LR1e-4_baseline_seed0.csv`
- Source JSON: `out/metrics/action1A_LR1e-4_baseline_seed0.json`
- CSV SHA256: `6fe0d1aae9cb4edc5880be31281344006189ba7bb479b1cc94dd65817ed36181`
- Run ID: `84m1cdjx`
- Run name: `stage-lm-wikitext2-baseline_token-dense-exact-na-na-na-na-L512-B16-S0`
- Config fingerprint: `64527b3263329159059c73dc7b552b81d5bc4df3`
- Summary TSV: `out/metrics/paper_action1A_parity_LR1e-4_epoch10.tsv`
- Final bundle TSV: `out/final_paper_bundle/tables/summary/paper_action1A_parity_LR1e-4_epoch10.tsv`
- Final paper reference: `out/final_paper_bundle/overleaf_ready/example_paper.tex`

Final epoch row:

| metric | value |
| --- | --- |
| `val/loss` | `6.66738041184789` |
| `val/ppl` | `786.3329467773438` |
| `train/ppl` | `252.10321044921875` |
| `train/time_per_epoch_s` | `46.18159770965576` |
| `train/peak_vram_mib` | `13407.220703125` |

### Table 8 / LR-normalized value: 836.1

- Rounded value: `836.1`
- Exact value: `836.1092529296875`
- Source CSV: `out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536/paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0.csv`
- Source JSON: `out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536/paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0.json`
- CSV SHA256: `8b17217dbed92fc3a9aef037b0afb3036c80762b185ded728ebd035111d67478`
- Run ID: `uiuinikf`
- Run name: `paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0`
- Config fingerprint: `7b08baed2d8ddf27bae9f748736947c71ff35906`
- Summary TSV: `out/paper_integrated_evidence/tables/lrnorm_headline_best_by_pair.tsv`
- All-runs TSV: `out/paper_integrated_evidence/tables/lrnorm_headline_all_runs.tsv`
- LaTeX table: `out/paper_integrated_evidence/latex/table_lrnorm_headline_best_by_pair.tex`
- Manifest: `out/paper_integrated_evidence/checks/lrnorm_headline_manifest.json`
- Final paper reference: `out/final_paper_bundle/overleaf_ready/example_paper.tex`

Final epoch row:

| metric | value |
| --- | --- |
| `val/loss` | `6.728759206456246` |
| `val/ppl` | `836.1092529296875` |
| `train/ppl` | `255.10484313964844` |
| `train/time_per_epoch_s` | `36.745994567871094` |
| `train/peak_vram_mib` | `13407.220703125` |

## Config/Artifact Diff

Common logged scientific settings:

| field | value |
| --- | --- |
| `task` | `lm` |
| `data.dataset` | `wikitext2` |
| `data.seq_len` | `512` |
| `data.batch_size` | `16` |
| `training.seed` | `0` |
| `training.epochs` | `10` |
| `training.lr` | `1e-4` |
| `training.warmup_steps` | `1000` |
| `training.set_diversity_mode` | `position_contrastive` |
| `training.set_diversity_weight` | `0.0` |
| `model.implementation` | `baseline_token` |
| `model.attention_family` | `dense` |
| `model.backend` | `exact` |
| `model.architecture` | `transformer_lm` |
| `model.causal` | `true` |
| `model.d_model` | `384` |
| `model.dim_feedforward` | `1536` |
| `model.num_layers` | `6` |
| `model.num_heads` | `8` |
| `model.dropout` | `0.1` |
| `model.attn_dropout` | `0.1` |
| `model.resid_dropout` | `0.1` |
| `model.ffn_dropout` | `0.1` |
| `model.max_seq_len` | `512` |
| `model.vocab_size` | `76618` |

Differing logged metadata:

| field | Action1A parity (`786.3`) | LR-normalized Table 8 (`836.1`) |
| --- | --- | --- |
| `csv_path` | `out/metrics/action1A_LR1e-4_baseline_seed0.csv` | `out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536/paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0.csv` |
| CSV SHA256 | `6fe0d1aae9cb4edc5880be31281344006189ba7bb479b1cc94dd65817ed36181` | `8b17217dbed92fc3a9aef037b0afb3036c80762b185ded728ebd035111d67478` |
| `run_id` | `84m1cdjx` | `uiuinikf` |
| `run_name` | `stage-lm-wikitext2-baseline_token-dense-exact-na-na-na-na-L512-B16-S0` | `paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0` |
| `config_fingerprint` | `64527b3263329159059c73dc7b552b81d5bc4df3` | `7b08baed2d8ddf27bae9f748736947c71ff35906` |
| `training.output_dir` | missing/`NA` | `out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536/paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0` |
| `training.wandb_init_timeout` | `180` | missing/`NA` |
| `logging.wandb.enable` | missing | `true` |
| `logging.wandb.project` | missing | `set-attention` |
| `logging.wandb.run_name` | missing | `paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0` |

The LR-normalized run script provenance is
`scripts/gpu0_run_lrnorm_headline_pairs.sh`, which invokes
`configs/paper_lr_norm/baseline_dense_exact.yaml` with overrides for the LR grid,
W&B metadata, output path, and the matched `(D,d_ff)` pairs.

The Action1A parity value is collected by the older provenance/export path
around `out/metrics/paper_action1A_parity_LR1e-4_epoch10.tsv`,
`scripts/export_main_results_provenance.sh`, and final-bundle copy rules in
`scripts/build_final_paper_bundle.sh`.

## Searches/Checks Performed

- Searched for `786.3`, `836.1`, `Table 8`, `App D.2`, `Appendix D.2`, and
  related TSV/CSV/JSON artifacts.
- Inspected:
  - `out/metrics/action1A_LR1e-4_baseline_seed0.csv`
  - `out/metrics/action1A_LR1e-4_baseline_seed0.json`
  - `out/metrics/paper_action1A_parity_LR1e-4_epoch10.tsv`
  - `out/final_paper_bundle/tables/summary/paper_action1A_parity_LR1e-4_epoch10.tsv`
  - `out/final_paper_bundle/checks/audit_appendix_parity.json`
  - `out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536/paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0.csv`
  - `out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536/paper_lrnorm_baseline_D384_FF1536_lr1e-4_seed0.json`
  - `out/paper_integrated_evidence/tables/lrnorm_headline_all_runs.tsv`
  - `out/paper_integrated_evidence/tables/lrnorm_headline_best_by_pair.tsv`
  - `out/paper_integrated_evidence/tables/lrnorm_d384_family_slice.tsv`
  - `out/paper_integrated_evidence/tables/lrnorm_d384_best_by_family.tsv`
  - `out/paper_integrated_evidence/latex/table_lrnorm_headline_best_by_pair.tex`
  - `out/paper_integrated_evidence/checks/lrnorm_headline_manifest.json`
  - `out/final_paper_bundle/overleaf_ready/example_paper.tex`
  - `configs/paper_lr_norm/baseline_dense_exact.yaml`
  - `scripts/gpu0_run_lrnorm_headline_pairs.sh`
  - `scripts/summarize_lrnorm_headline.py`
  - `scripts/export_main_results_provenance.sh`
  - `scripts/build_final_paper_bundle.sh`

No training or expensive evaluation was run for A1.3.
