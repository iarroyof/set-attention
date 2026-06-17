# A8.3 Large-L Smoke on Lizmark

Date: 2026-06-13  
Status: PASS  
Host: `iarroyof@192.168.241.205` (`lizmark`)  
Source HEAD: `1c68dfa9b86b351007da285252ef823fe87c9227`  

## Setup

- Streamed Docker image from blue-demon to lizmark with `docker save set-attention:latest | docker load`.
- Loaded image on lizmark: `set-attention:latest`, size `35.7GB`.
- Copied `~/set-attention` source/config/scripts and `.hf` cache from blue-demon to lizmark, excluding `.git`, `out`, `logs`, `wandb`, and pycache.
- Container sanity check passed inside `set-attention:latest`:
  - PyTorch `2.5.1+cu124`;
  - CUDA available;
  - 2 GPUs visible;
  - both GPUs are NVIDIA RTX 6000 Ada Generation cards with about 51 GB total memory.
- Offline WikiText cache probe passed. At `L=8192`, `data.limit=500` yields at least two train chunks; smaller line limits may yield zero or one chunk.

## Launcher

Script: `scripts/run_a8_largeL_smoke_lizmark.sh`

The script runs one-epoch, two-chunk smoke rows at:

- `L=8192`
- `D=384`
- `d_ff=1536`
- `batch_size=1`
- `seed=0`
- `lr=1e-4`
- set rows use `set_causality_mode=strict_past`
- set rows use `output_residual_mode=empty_only`
- landmark rows use `landmark_coverage=0.25`

The launcher records nonzero exits without stopping later rows on the same GPU, so an unsupported dense baseline fit would not hide set-family smoke results.

## Results

All six rows exited with code `0`; CSV and JSON metadata files were produced and copied locally.

| Row | Window | Stride | Landmark count | Val PPL | Peak VRAM MiB | Time/epoch s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_dense_exact_L8192_seed0` | NA | NA | NA | 4965.8828 | 33223.9590 | 3.6452 |
| `baseline_linear_landmark_L8192_seed0` | NA | NA | 2048 | 5144.7622 | 16839.0718 | 2.3358 |
| `set_dense_exact_L8192_w4_s2_seed0` | 4 | 2 | NA | 5218.8623 | 18113.0825 | 12.5359 |
| `set_dense_exact_L8192_w8_s4_seed0` | 8 | 4 | NA | 5235.7070 | 7491.3477 | 6.9406 |
| `set_linear_landmark_L8192_w4_s2_seed0` | 4 | 2 | 1024 | 5569.8955 | 14647.6318 | 12.8480 |
| `set_linear_landmark_L8192_w8_s4_seed0` | 8 | 4 | 512 | 3498.5229 | 6628.4136 | 6.8329 |

These values are smoke metrics, not paper-bound quality estimates: the run uses one epoch and only two long WikiText chunks to test fit, metadata, and approximate memory behavior.

## Artifacts

- Status TSV: `out/paper_mechanisms/a8_largeL_smoke/a8_largeL_smoke_status.tsv`
- Raw CSV/JSON root: `out/paper_mechanisms/a8_largeL_smoke/`
- Logs: `logs/a8_largeL_smoke/`
- Prelaunch metadata: `audit/A8_3_largeL_smoke_prelaunch.json` on lizmark

## Notes

- Initial launch failed before running because a dry-run container had created `out/` as root. Ownership was fixed with a one-shot root container over the mounted repo.
- The launcher now runs experiment containers with the host UID/GID to avoid future root-owned artifacts.
- A harmless W&B/Weave cache warning appeared because the numeric container user had no home directory. The launcher now sets `HOME=/workspace` and `XDG_CACHE_HOME=/workspace/.cache` for future runs; no rerun was needed because all smoke rows exited `0` and metrics were finite.
- Temporary probe files were removed after validation.

## Recommendation

The direct `L=8192` smoke is viable on lizmark. The most informative next paper-bound experiment is not a broad grid yet; run 5-seed follow-up only for operating points whose smoke behavior supports the A8 question. From this smoke:

- `(w,s)=(8,4)` has a much larger memory reduction than `(4,2)`.
- SetLinear `(8,4)` is the only smoke row with lower one-epoch PPL than its matched token baseline, but the metric is too small-sample to claim quality.
- If continuing empirically before candidate-gather routing, prioritize a 5-seed `L=8192` follow-up for linear landmark at `(8,4)` plus its matched token linear baseline; include dense `(8,4)` only if the goal is memory-scaling characterization rather than near-baseline quality.
