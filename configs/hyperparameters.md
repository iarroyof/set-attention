# Hyperparameters (Normalized Names)

This table uses **LM naming** as the canonical reference. Seq2Seq configs have been normalized to match these names.

## Shared Hyperparameters (LM + Seq2Seq)

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `training.epochs` | Number of training epochs | LM: 5, Seq2Seq: 5 | Recommended: `1–50` |
| `training.lr` | Optimizer learning rate | `0.0003` | Recommended: `1e-5–5e-4` |
| `training.seed` | Random seed for reproducibility | `0` | Any integer |
| `data.batch_size` | Batch size | `16` | Recommended: `1–64` |
| `data.seq_len` | Sequence length (LM tokens / Seq2Seq max length) | `256` | Recommended: `64–1024` |
| `data.streaming` | Stream from HF datasets (no full RAM load) | Default `true` | `true`, `false` |
| `data.cache_root` | HF cache root (HF_HOME) | Default from env | Any path |
| `model.vocab_size` | Vocab size (0 means auto‑infer) | `0` | `0` or positive integer |
| `model.d_model` | Model hidden size | `256` | Recommended: `128–1024` |
| `model.num_layers` | Number of layers | `4` | Recommended: `2–12` |
| `model.num_heads` | Number of attention heads | `8` | Recommended: divisors of `d_model` |
| `model.dropout` | Dropout probability | `0.1` | Recommended: `0.0–0.3` |
| `model.max_seq_len` | Max length for positional embeddings | `256` | Recommended: `>= data.seq_len` |

## LM (Set‑Only) Hyperparameters

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `model.family` | Model family | `set_only` (LM), `encoder_set_only` (Seq2Seq encoder‑only) | `baseline_token`, `set_only`, `encoder_set_only` |
| `model.window_size` | Set window size | `32` | Recommended: `8–128` |
| `model.stride` | Set stride | `16` | Recommended: `1–window_size` |
| `model.backend` | Set attention backend | `dense_exact` | `dense_exact`, `local_band`, `nystrom`, `landmark`, `sparse_topk` |
| `model.router_type` | Router type | `learned` | `uniform`, `learned` |
| `model.router_topk` | Router top‑k | `4` | `>=1` and `<= max_sets` (required for learned) |
| `model.feature_mode` | Feature mode | `hashed_counts` | `geometry_only`, `hashed_counts`, `kernel` |
| `model.pooling.mode` | Pooling mode | `mean` | `mean`, `soft_trimmed_boltzmann` |
| `model.geometry.enabled` | Geometry features enabled | `true` | `true`, `false` |
| `model.geometry.apply_as_bias` | Apply geometry as bias | `false` | `true`, `false` |
| `model.geometry.apply_in_phi_attn` | Include geometry in phi_attn | `true` | `true`, `false` |
| `model.sig_gating.enabled` | Signature gating | `false` | `true`, `false` |

## Seq2Seq (Baseline) Hyperparameters

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `model.family` | Model family | `baseline_token` | `baseline_token`, `set_only`, `encoder_set_only` |
| `model.architecture` | Architecture variant | `transformer_seq2seq` | `transformer_seq2seq` |
| `model.dim_feedforward` | FFN hidden size | `1024` | Recommended: `2x–8x d_model` |
| `model.seq2seq.shared_vocab` | Shared src/tgt vocab | `true` | `true` (separate vocab not implemented) |
| `data.seq_dataset` | Seq2Seq dataset key | `opus_books_en_fr` | `opus_books_en_fr`, `wmt14_fr_en`, `cnn_dailymail` |

## Seq2Seq (Set‑Only Encoder) Hyperparameters

Same as LM set‑only hyperparameters, plus:

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `model.seq2seq.shared_vocab` | Shared src/tgt vocab | `true` | `true` |
| `data.seq_dataset` | Seq2Seq dataset key | `opus_books_en_fr` | `opus_books_en_fr`, `wmt14_fr_en`, `cnn_dailymail` |
| `model.family` | Encoder only set‑only family selector | `encoder_set_only` | `encoder_set_only` |

## Compatibility Rules (Hyperparameter Combinations)

| Rule | Applies When | Requirement |
| --- | --- | --- |
| `d_model % num_heads == 0` | All transformer models | Must be divisible |
| `head_dim >= 8` | All transformer models | `d_model / num_heads >= 8` |
| `window_size <= seq_len` | Set‑only models | Must hold |
| `stride <= window_size` | Set‑only models | Must hold |
| `router_topk` required | `router_type=learned` | `1 <= router_topk <= max_sets` |
| `backend_params` required | `backend=local_band` | Must set `backend_params.radius >= 1` |
| `backend_params` required | `backend in {nystrom, landmark}` | Must set `backend_params.num_landmarks >= 2` and `< max_sets` |
| `backend_params` forbidden | `backend=dense_exact` | Must be empty/absent |
| `feature_mode=kernel` + large sets | Set‑only | Requires `max_sets <= 500` unless override |
| `sig_gating` top‑k | `sig_gating.method=*topk` | `k <= max_sets` |
| `sig_gating` threshold | `sig_gating.method=*threshold` | `0 <= delta_threshold <= 1` |
| `sig_gating` minhash | `sig_gating.method=minhash_*` | Must set `sig_gating.sig_k` |
