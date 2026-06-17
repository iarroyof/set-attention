# VRAM Overhead Audit

Date: 2026-06-12  
Scope: A7 backend-matched `empty_only` operating points at `D=384`, `d_ff=1536`, `L=512`, batch size 16, vocab size 76618.

## Result

No paper VRAM subtraction is justified.

The measured near-token set overhead is real architecture/implementation cost, not a removable logging artifact. The dominant contributors are:

- simultaneous token states and set states in `SetOnlyLM`;
- set self-attention over `M` set states;
- dense learned token-to-set router score/probability tensors of shape `[B,H,L,M]`;
- normal optimizer-resident training memory shared by both token and set models.

The only specifically tested set diagnostic artifact, the gradient-probe `retain_grad()` path, contributes `0.0` MiB to the warm optimizer-resident peak for the measured A7 topologies. Therefore the reported A7 VRAM values remain raw `torch.cuda.max_memory_allocated()` epoch peaks.

## Probe

Command run on blue-demon Docker:

```bash
cd ~/set-attention
docker compose exec -T -e CUDA_VISIBLE_DEVICES=0 set-attention python scripts/audit_vram_overhead.py
```

Machine-readable output:

```text
audit/vram_overhead_audit.json
```

The probe runs synthetic one-step LM training with the same model width, sequence length, batch size, backend family, vocab size, and A7 set topologies. It reports both:

- cold first-step peak, before AdamW state is resident;
- warm optimizer-resident peak, matching the paper's later-epoch `train/peak_vram_mib` measurement.

## Warm Peaks Versus Paper Values

| Family | Row | Warm probe MiB | Paper mean MiB | Difference |
| --- | ---: | ---: | ---: | ---: |
| Dense token | baseline | 13393.7 | 13407.2 | +13.5 |
| Sparse token | baseline | 13395.2 | 13408.7 | +13.5 |
| Linear token | baseline | 12582.9 | 12596.5 | +13.5 |
| SetDense | `(1,1)`, `M=512` | 13915.6 | 13936.6 | +21.0 |
| SetDense | `(4,2)`, `M=255` | 11792.3 | 11807.3 | +15.0 |
| SetDense | `(16,8)`, `M=63` | 10774.7 | 10785.2 | +10.5 |
| SetSparse | `(1,1)`, `M=512` | 13915.6 | 13936.6 | +21.0 |
| SetSparse | `(4,2)`, `M=255` | 11792.3 | 11807.3 | +15.0 |
| SetSparse | `(16,8)`, `M=63` | 10774.7 | 10785.2 | +10.5 |
| SetLinear | `(1,1)`, `M=512` | 13104.8 | 13125.9 | +21.0 |
| SetLinear | `(4,2)`, `M=255` | 11605.6 | 11620.6 | +15.0 |
| SetLinear | `(16,8)`, `M=63` | 10768.6 | 10779.1 | +10.5 |

The warm probe matches the paper values closely enough to validate the measurement path. The small residual is consistent with full epoch/run-path variation and is not a backend-specific removable overhead.

## Diagnostic Artifact Check

| Topology | `M` | Gradient-probe delta at warm peak |
| --- | ---: | ---: |
| `(1,1)` | 512 | 0.0 MiB |
| `(4,2)` | 255 | 0.0 MiB |
| `(16,8)` | 63 | 0.0 MiB |

The raw retained-gradient tensor-size formula is small relative to the observed overhead:

```text
B * (L + 2M) * D * 4 bytes
```

This is 36.0 MiB for `M=512`, 24.0 MiB for `M=255`, and 15.0 MiB for `M=63`, and it does not move the warm peak in the actual probe.

## Architectural / Implementation Contributors

For `B=16`, `H=8`, `L=512`, `D=384`:

| Topology | `M` | Token state | One set-state tensor | Pool gather | Router score/prob tensor `[B,H,L,M]` | Set attention score/prob tensor `[B,H,M,M]` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1,1)` | 512 | 12.0 MiB | 12.0 MiB | 12.0 MiB | 128.0 MiB | 128.0 MiB |
| `(4,2)` | 255 | 12.0 MiB | 6.0 MiB | 23.9 MiB | 63.8 MiB | 31.8 MiB |
| `(16,8)` | 63 | 12.0 MiB | 1.5 MiB | 23.6 MiB | 15.8 MiB | 1.9 MiB |

At the singleton limit, set attention is not memory-identical to token attention. It carries a token stream and a set stream, and the learned router materializes an additional token-to-set interaction tensor. This explains why `(w,s)=(1,1)` can approach matched-token quality while using about 104% of matched-token VRAM.

## Reporting Rule

Report raw `train/peak_vram_mib` unless an audit identifies a non-architectural GPU allocation with an exact per-row correction formula and provenance. If such an artifact is found later, report both raw and adjusted VRAM, document the formula, and apply it only to affected rows. Do not apply a blanket subtraction across token and set families.
