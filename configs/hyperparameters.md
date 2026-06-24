# Hyperparameters (Normalized Names)

This table uses the **canonical naming**:

- `implementation`: baseline vs set-only placement
- `attention_family`: dense / sparse / linear
- `backend`: exact / local_band / landmark for the active revision surface; legacy schema values `nystrom` and `linformer` are not active tested backends for this revision

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
| `model.dim_feedforward` | FFN hidden size | `d_model * 4` | Recommended: `d_model*2–d_model*8` |
| `model.dropout` | Dropout probability | `0.1` | Recommended: `0.0–0.3` |
| `model.max_seq_len` | Max length for positional embeddings | `256` | Recommended: `>= data.seq_len` |

## Core Attention Naming (New)

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `model.implementation` | Where set-only is used | LM: `baseline_token` or `set_only` | `baseline_token`, `set_only`, `hybrid_token_set`, `encoder_set_only`, `decoder_set_only`, `cross_attention_set_only`, `encoder_set_decoder_baseline`, `encoder_baseline_decoder_set` |
| `model.attention_family` | Complexity family | `dense` | `dense`, `sparse`, `linear` |
| `model.backend` | Concrete backend | `exact` | Active revision: `exact`, `local_band`, `landmark`; legacy/deprecated schema values: `nystrom`, `linformer` |
| `model.encoder_attention_family` | Encoder attention family | defaults to `model.attention_family` | `dense`, `sparse`, `linear` |
| `model.encoder_backend` | Encoder backend | defaults to `model.backend` | Active revision: `exact`, `local_band`, `landmark`; legacy/deprecated schema values: `nystrom`, `linformer` |
| `model.decoder_attention_family` | Decoder self-attn family | defaults to `model.attention_family` | `dense`, `sparse`, `linear` |
| `model.decoder_backend` | Decoder self-attn backend | defaults to `model.backend` | Active revision: `exact`, `local_band`, `landmark`; legacy/deprecated schema values: `nystrom`, `linformer` |
| `model.cross_attention_family` | Cross-attn family | defaults to `model.attention_family` | `dense`, `sparse`, `linear` |
| `model.cross_backend` | Cross-attn backend | defaults to `model.backend` | Active revision: `exact`, `local_band`, `landmark`; legacy/deprecated schema values: `nystrom`, `linformer` |
| `model.cross_attention` | Cross-attn implementation | default derives from `implementation` | `baseline`, `set_only` |

## Set-Only Hyperparameters (shared across encoder/decoder/cross)

| Name | Meaning | Default / Current | Choices / Range |
| --- | --- | --- | --- |
| `model.window_size` | Set window size | `32` | Recommended: `4–128` |
| `model.stride` | Set stride | `16` | Recommended: `1–window_size` |
| `model.router_type` | Router type | `learned` | `uniform`, `learned` |
| `model.router_topk` | Router top‑k | `4` | `>=1` and `<= max_sets` (required for learned) |
| `model.router.min_temp` | Learned-router temperature floor | `0.5` | `> 0` |
| `model.router.score_mode` | Learned-router score implementation | `candidate_gather` | `candidate_gather`, `dense` |
| `model.feature_mode` | Feature mode | `hashed_counts` | `geometry_only`, `hashed_counts`, `kernel` |
| `model.feature_params.num_bins` | Hashed-count feature bins | `128` | Positive integer |
| `model.feature_params.hash_seed` | Hashed-count deterministic hash seed | `13` | Integer |
| `model.feature_params.normalize` | Normalize hashed counts by set size | `true` | `true`, `false` |
| `model.pooling.mode` | Pooling mode | `mean` | `mean`, `soft_trimmed_boltzmann` |
| `model.pooling.alpha` | Soft-trimmed Boltzmann trim sharpness | `10.0` | `> 0` |
| `model.pooling.learnable_alpha` | Learn trim sharpness | `false` | `true`, `false` |
| `model.set_causality_mode` | Single source of truth for set-bank/routing causality mode | LM causal default: `strict_past`; noncausal default: `noncausal` | `strict_past`, `noncausal`; legacy `model.causal` is deprecated for set-only models and cannot override this value |
| `model.output_residual_mode` | Strict-past output policy before the LM head | `direct` | `direct`, `empty_only`, `none`, `anchor_span` |
| `model.anchor.enabled` | Enable predictive anchoring auxiliary loss | `false` | `true`, `false`; when false, no pre-encoder is constructed |
| `model.anchor.target` | Anchor target source | `pre_encoder` | `pre_encoder` only |
| `model.anchor.pre_encoder_layers` | Shallow causal token pre-encoder depth | `2` | `1` or `2` |
| `model.anchor.lambda_h` | Anchor loss weight | `0.1` | `>= 0` |
| `model.anchor.lambda_pre` | Pre-encoder next-token CE weight (`L_CE_pre`) | `1.0` | `>= 0`; must be `> 0` when `anchor.enabled=true` |
| `model.anchor.pre_encoder_head` | Enable the pre-encoder's own causal LM head | `true` | Required when `anchor.enabled=true`; training-only and excluded from inference accounting |
| `model.anchor.detach_target` | Stop gradients through anchor target | `true` | `true`, `false` |
| `model.anchor.norm` | Normalization before anchor MSE | `layernorm` | `layernorm` |
| `model.anchor.teacher.enabled` | External teacher anchor path | `false` | Deferred; must remain `false` |
| `model.set_diversity.lambda_div` | Set diversity regularizer weight | `0.0` | `>= 0` |
| `model.multivector_basis.enabled` | Multi-vector basis floor-test knob | `false` | Deferred; must remain `false` |
| `model.multivector_basis.r` | Value sub-vectors per atom per head | `1` | `1–4`; stays `1` while disabled |
| `model.candidate_fiber` | Candidate fiber policy | `endpoint_window` | `endpoint_window`, `all_past`; `window_plus_landmarks` is deferred |
| `model.multiresolution.enabled` | Enable parallel set streams with different topologies inside one set-only LM | `false` | `true`, `false`; experimental SD-9 path |
| `model.multiresolution.groups` | Parallel set stream definitions | `[]` | List of `{name, num_heads, window_size, stride}`; group head counts must sum to `model.num_heads` |
| `model.hybrid.pattern` | Layer pattern for `hybrid_token_set` LM; `T` means token-attention layer and `S` means set layer | Required for `hybrid_token_set` | String of `T`/`S` with length `model.num_layers` |
| `model.hybrid.set_topologies` | Per-set-layer topology list for `hybrid_token_set`; one entry per `S` in `model.hybrid.pattern` | Required for `hybrid_token_set` | List of `{window_size, stride}` mappings |
| `model.d_phi` | Set feature / adapter dimension | `null`, resolved to `model.d_model` | `null` or positive integer |
| `model.set_state_dim` | Pooled set-state / set-stack width | `null`, resolved to `model.d_model` | `null` or positive integer divisible by `model.num_heads` |
| `model.adapter_type` | Content-bias adapter type | `auto` | `auto`, `linear`, `nonlinear`, `hybrid` |
| `model.backend_params.landmark_coverage` | Landmark coverage fraction for `backend=landmark` | `0.25` | `> 0`; runtime `K=max(round(coverage*M),2)`, capped at `M` |
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
| `set_causality_mode=strict_past` | Autoregressive set-only LM | Drops partial trailing windows; candidates are endpoint sets `{m: t-window_size < endpoint_m <= t}`; set self-attention masking is derived from this mode |
| `output_residual_mode` | Strict-past set-only LM | `direct` uses `h_t^(0)+r_t`; `empty_only` uses `h_t^(0)` only for `C_t=0` and otherwise `r_t`; `none` uses only `r_t`; `anchor_span` uses `emb(x_t)+pos_t+span_t` and requires `token_mlp.enabled=false` |
| `anchor.enabled=true` | Set-dictionary anchoring | Constructs a shallow causal token pre-encoder with its own LM head and `L_CE_pre`; `anchor.lambda_pre>0`, `anchor.pre_encoder_head=true`, and `anchor.teacher.enabled=false` are required |
| `candidate_fiber` | Set-dictionary routing support | `endpoint_window` uses sealed endpoints in the local window; `all_past` uses every sealed past set (`endpoint_m <= t`); `window_plus_landmarks` is deferred |
| `multiresolution.enabled=true` | Set-only LM | Builds one set stream per `model.multiresolution.groups` entry, sharing token embeddings/token MLP and concatenating routed stream outputs before the LM head; each group's `d_phi` and set width are proportional to `num_heads` |
| `multiresolution.groups` | Set-only LM | Head counts must sum to `model.num_heads`; every group needs positive `window_size` and `stride`, `stride <= window_size <= data.seq_len`; proportional `d_model`, `set_state_dim`, and `d_phi` shares must be integral |
| `hybrid_token_set` topology | Hybrid LM | `model.hybrid.pattern` length must equal `model.num_layers`; `model.hybrid.set_topologies` length must equal the number of `S` layers; every topology requires positive `window_size` and `stride`, with `stride <= window_size <= data.seq_len` |
| `router_topk` required | `router_type=learned` | `1 <= router_topk <= max_sets` |
| `router.min_temp` | `router_type=learned` | Must be numeric and `> 0`; default `0.5` |
| `router.score_mode` | `router_type=learned` | `candidate_gather` computes scores only over the supplied candidate fiber; `dense` preserves the historical dense `[B,H,L,M]` masked score tensor for debugging/comparison |
| `pooling.alpha` | `pooling.mode=soft_trimmed_boltzmann` | Must be numeric and `> 0`; default `10.0` |
| `pooling.learnable_alpha` | Set-only pooling | Must be boolean; default `false` |
| `feature_params.num_bins` | `feature_mode=hashed_counts` | Must be positive integer; default `128` |
| `feature_params.hash_seed` | `feature_mode=hashed_counts` | Must be an integer; default `13` |
| `feature_params.normalize` | `feature_mode=hashed_counts` | Must be boolean; default `true` |
| `d_phi` | Set-only models | `null` resolves at runtime to `d_model`; otherwise positive integer |
| `set_state_dim` | Set-only models | `null` resolves at runtime to `d_model`; otherwise positive integer divisible by `num_heads`; set states are projected back to `d_model` before the LM head |
| `adapter_type` | Set-only models with content-bias features | `auto`, `linear`, `nonlinear`, or `hybrid`; resolved adapter type is logged |
| `attention_family=dense` | Any component | `backend=exact` |
| `attention_family=sparse` | Any component | `backend in {local_band}` |
| `attention_family=linear` | Any component | Active revision uses `backend=landmark`; legacy/deprecated schema values `nystrom` and `linformer` are not active tested backends |
| `backend_params` required | `backend=local_band` | Must set `backend_params.radius >= 1`; optional `global_indices` (tokens) or `global_set_indices` (sets) |
| `backend_params` required | `backend=landmark` | Uses `backend_params.landmark_coverage > 0`; default `0.25`; `num_landmarks` is not used by the active landmark backend |
| `backend_params` required | `backend=nystrom` | Deprecated and rejected for this revision cycle; historical YAMLs may exist only under `configs/_deprecated/` and must not be active launch configs |
| `backend_params` required | `backend=linformer` | Legacy schema path; not an active tested backend for this revision; must set `backend_params.k >= 2` |
| `backend_params` forbidden | `backend=exact` | Must be empty/absent |
| `feature_mode=kernel` + large sets | Set‑only | Requires `max_sets <= 500` unless override |
| `sig_gating` top‑k | `sig_gating.method=*topk` | `k <= max_sets` |
| `sig_gating` threshold | `sig_gating.method=*threshold` | `0 <= delta_threshold <= 1` |
| `sig_gating` minhash | `sig_gating.method=minhash_*` | Must set `sig_gating.sig_k` |
