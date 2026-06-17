# A8 Lizmark Large-L Capacity Check

Date: 2026-06-13  
Host: `iarroyof@192.168.241.205` (`lizmark`)  
Purpose: determine what larger context lengths are reasonable to try for the near-2 compression experiments.

## Hardware Status

Command method:

```bash
sshpass -f ~/.ssh/.sshpass ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/set_attention_known_hosts_205 iarroyof@192.168.241.205 'date; hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader; docker ps --format "{{.Names}}\t{{.Status}}\t{{.Image}}" 2>/dev/null || true; ls -d ~/set-attention 2>/dev/null || true'
```

Observed GPUs:

| GPU | Model | Total MiB | Used MiB | Free MiB |
| ---: | --- | ---: | ---: | ---: |
| 0 | NVIDIA RTX 6000 Ada Generation | 49140 | 1 | 48637 |
| 1 | NVIDIA RTX 6000 Ada Generation | 49140 | 1 | 48645 |

No GPU jobs were running at the time of the check.

## Software / Filesystem Status

- `~/set-attention` was not present.
- No `set-attention` Docker container was running.
- System Python exists but `torch` is not installed.
- Initial root filesystem was full: `/dev/nvme0n1p3 456G 433G 0 100% /`.

This means no real model smoke was run on lizmark during the first check.

## Cleanup Update

The user asked to remove all stopped containers except `cesar2_container` and `cancer_rl_mistral`. Removed:

- `elo-training-test`
- `loving_lalande`
- `bogdan_indexer`
- `bogdan_es`
- `es-evaluacion`
- `entrenamiento-gpt`

Kept stopped containers:

- `cesar2_container`
- `cancer_rl_mistral`

Running containers were not removed:

- `bogdan_postgres`
- `cancer_rl_agent`
- `es_cancer_db`

After cleanup:

| Filesystem | Size | Used | Free | Use |
| --- | ---: | ---: | ---: | ---: |
| `/` | 456G | 297G | 137G | 69% |
| `/mnt/data` | 458G | 432G | 3.0G | 100% |
| `/mnt/nuevo_disco` | 466G | 13G | 453G | 3% |

This makes a minimal set-attention transfer feasible on root if we stream the Docker image instead of storing an extra image tar.

## USB Pendrive Status

Inserted drive:

| Device | Size | Transport | Filesystem | Label | USB negotiated speed |
| --- | ---: | --- | --- | --- | --- |
| `/dev/sdc1` | 57.8G | USB | `vfat` | `LUBUNTU 25_` | 5000M |

The pendrive is not a good Docker target as-is:

- `vfat` has a 4 GB file-size limit, so it cannot store a normal Docker image tar.
- `vfat` is not appropriate for Docker overlay storage.
- 57.8 GB is too tight for the image plus repo/cache plus working headroom.

It could be reformatted to ext4 and used for small repo/cache staging, but after container cleanup the root filesystem is now a better immediate target. Prefer an external SSD, or use `/mnt/nuevo_disco` for non-Docker staging if needed.

## Recommended Large-L Sequence

For the near-2 / near-4 compression focus:

1. Start with direct `L=8192` smoke tests if the set-attention environment can be installed on root without filling the disk.
   - topologies: `(w,s)=(4,2)` and `(8,4)`;
   - families: `set_dense_exact`, `set_linear_landmark`, matched `baseline_dense_exact`, matched `baseline_linear_landmark`;
   - use forward-only or 1-batch train/validation smoke first;
   - record exact batch size and raw `train/peak_vram_mib`.
2. Use `L=4096` only as fallback if `L=8192` OOMs or exposes unsupported sequence-length behavior.
3. If seed-0 `L=8192` is viable, run 5 seeds only if the set-vs-token gap narrows versus `L=512`/`L=2048` while keeping real VRAM savings.
4. Test `(w,s)=(6,2)` after the direct observed-topology comparison, or in parallel only if compute and setup time are not bottlenecks.
5. Do not run `L=16384` until `L=8192` has a clean smoke and there is an explicit memory plan.

Rationale: compared with blue-demon RTX 4090 GPUs, lizmark has roughly twice the per-GPU memory. Existing validated `L=2048` runs fit on 24 GiB-class GPUs at reduced batch size. A 48 GiB RTX 6000 Ada host is therefore a plausible target for direct `L=8192` smokes, but feasibility must be verified empirically after syncing the repo/container.

## A8.3 Smoke Update

The direct `L=8192` smoke was completed on lizmark after streaming the Docker image and copying the repo/cache from blue-demon. See `audit/A8_3_largeL_smoke.md`.

All six seed-0 smoke rows completed with exit code `0`:

- `baseline_dense_exact`
- `baseline_linear_landmark`
- `set_dense_exact` at `(w,s)=(4,2)` and `(8,4)`
- `set_linear_landmark` at `(w,s)=(4,2)` and `(8,4)`

Smoke artifacts are under `out/paper_mechanisms/a8_largeL_smoke/`; logs are under `logs/a8_largeL_smoke/`. These are one-epoch, two-long-chunk fit checks, not paper-bound quality runs.
