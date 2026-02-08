# Hyperparameters (Normalized Names)

This table uses the **canonical naming**:

- `implementation`: baseline vs set-only placement
- `attention_family`: dense / sparse / linear
- `backend`: exact / local_band / sparse_topk / landmark / nystrom

## Shared Hyperparameters (LM + Seq2Seq)

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `training.epochs` | Number of training epochs | LM: 5, Seq2Seq: 5 | Recommended: `1–50` |
| `training.lr` | Optimizer learning rate | `0.0003` | Recommended: `1e-5–5e-4` |
| `training.seed` | Random seed | `0` | Any integer |
| `data.batch_size` | Batch size | `16` | Recommended: `1–64` |
| `data.seq_len` | Sequence length (LM tokens / Seq2Seq max length) | `256` | Recommended: `64–1024` |
| `data.streaming` | Stream from HF datasets | Default `true` | `true`, `false` |
| `data.cache_root` | HF cache root | Default from env | Any path |
| `model.vocab_size` | Vocab size (0 = auto infer) | `0` | `0` or positive integer |
| `model.d_model` | Hidden size | `256` | Recommended: `128–1024` |
| `model.num_layers` | Number of layers | `4` | Recommended: `2–12` |
| `model.num_heads` | Number of attention heads | `8` | Divisors of `d_model` |
| `model.dropout` | Dropout probability | `0.1` | Recommended: `0.0–0.3` |
| `model.max_seq_len` | Max length for positional embeddings | `256` | Recommended: `>= data.seq_len` |

## Core Attention Naming (New)

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `model.implementation` | Where set-only is used | LM: `baseline_token` or `set_only` | `baseline_token`, `set_only`, `encoder_set_only`, `decoder_set_only`, `cross_attention_set_only`, `encoder_set_decoder_baseline`, `encoder_baseline_decoder_set` |
| `model.attention_family` | Complexity family | `dense` | `dense`, `sparse`, `linear` |
| `model.backend` | Concrete backend | `exact` | `exact`, `local_band`, `sparse_topk` (deprecated), `landmark`, `nystrom`, `linformer` |
| `model.encoder_attention_family` | Encoder attention family | defaults to `model.attention_family` | `dense`, `sparse`, `linear` |
| `model.encoder_backend` | Encoder backend | defaults to `model.backend` | `exact`, `local_band`, `sparse_topk` (deprecated), `landmark`, `nystrom`, `linformer` |
| `model.decoder_attention_family` | Decoder self-attn family | defaults to `model.attention_family` | `dense`, `sparse`, `linear` |
| `model.decoder_backend` | Decoder self-attn backend | defaults to `model.backend` | `exact`, `local_band`, `sparse_topk` (deprecated), `landmark`, `nystrom`, `linformer` |
| `model.cross_attention_family` | Cross-attn family | defaults to `model.attention_family` | `dense`, `sparse`, `linear` |
| `model.cross_backend` | Cross-attn backend | defaults to `model.backend` | `exact`, `local_band`, `sparse_topk` (deprecated), `landmark`, `nystrom`, `linformer` |
| `model.cross_attention` | Cross-attn implementation | default derives from `implementation` | `baseline`, `set_only` |

## Set-Only Hyperparameters (shared across encoder/decoder/cross)

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `model.window_size` | Set window size | `32` | Recommended: `4–128` |
| `model.stride` | Set stride | `16` | Recommended: `1–window_size` |
| `model.router_type` | Router type | `learned` | `uniform`, `learned` |
| `model.router_topk` | Router top‑k | `4` | `>=1` and `<= max_sets` (required for learned) |
| `model.feature_mode` | Feature mode | `hashed_counts` | `geometry_only`, `hashed_counts`, `kernel` |
| `model.pooling.mode` | Pooling mode | `mean` | `mean`, `soft_trimmed_boltzmann` |
| `model.geometry.enabled` | Geometry enabled | `true` | `true`, `false` |
| `model.geometry.apply_as_bias` | Apply geometry as bias | `false` | `true`, `false` |
| `model.geometry.apply_in_phi_attn` | Include geometry in phi_attn | `true` | `true`, `false` |
| `model.sig_gating.enabled` | Signature gating | `false` | `true`, `false` |

## Seq2Seq-Specific

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `model.architecture` | Seq2Seq architecture | `transformer_seq2seq` | `transformer_seq2seq` |
| `model.seq2seq.shared_vocab` | Shared vocab | `true` | `true` (separate vocab not implemented) |
| `data.seq_dataset` | Seq2Seq dataset key | `opus_books_en_fr` | `opus_books_en_fr`, `wmt14_fr_en`, `cnn_dailymail` |

## Compatibility Rules (Hyperparameter Combinations)

| Rule | Applies When | Requirement |
| --- | --- | --- |
| `d_model % num_heads == 0` | All transformer models | Must be divisible |
| `head_dim >= 8` | All transformer models | `d_model / num_heads >= 8` |
| `window_size <= seq_len` | Set‑only models | Must hold |
| `stride <= window_size` | Set‑only models | Must hold |
| `router_topk` required | `router_type=learned` | `1 <= router_topk <= max_sets` |
| `attention_family=dense` | Any component | `backend=exact` |
| `attention_family=sparse` | Any component | `backend in {local_band, sparse_topk}` |
| `attention_family=linear` | Any component | `backend in {landmark, nystrom, linformer}` |
| `backend_params` required | `backend=local_band` | Must set `backend_params.radius >= 1`; optional `global_indices` (tokens) or `global_set_indices` (sets) |
| `backend_params` required | `backend in {nystrom, landmark}` | Must set `backend_params.num_landmarks >= 2` and `< max_sets` |
| `backend_params` required | `backend=linformer` | Must set `backend_params.k >= 2` |
| `backend_params` forbidden | `backend=exact` | Must be empty/absent |
| `feature_mode=kernel` + large sets | Set‑only | Requires `max_sets <= 500` unless override |
| `sig_gating` top‑k | `sig_gating.method=*topk` | `k <= max_sets` |
| `sig_gating` threshold | `sig_gating.method=*threshold` | `0 <= delta_threshold <= 1` |
| `sig_gating` minhash | `sig_gating.method=minhash_*` | Must set `sig_gating.sig_k` |
