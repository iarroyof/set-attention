# SD-9.5 32k Scale-L OOM

Date: 2026-06-20

Host: lizmark (`iarroyof@192.168.241.205`)

Run: `ROLE=scale SEQ_LEN=32768 SMOKE=1 IMAGE=set-attention:latest bash scripts/run_sd9_5_probes.sh`

Result: mixed-65 seed 0 exited `1` before completing epoch 1.

Failure: CUDA OOM in the landmark backend with `landmark_coverage=0.25`, batch 1.

Log excerpt summary: PyTorch attempted to allocate an additional 768 MiB while GPU 0 had about 341 MiB free; the process had about 47.16 GiB in use. The traceback is in `logs/sd9_5_scaleL_L32768_smoke/sd9_5_scale_mixed_blur62_L32768_B1_landmark_seed0.log` on lizmark.

Guard decision: do not silently change topology, backend, coverage, or batch. Record 32768 as OOM and keep the 16384 landmark scale row.
