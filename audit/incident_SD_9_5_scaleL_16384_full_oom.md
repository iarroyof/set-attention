# SD-9.5 16k Scale-L Full-Data OOM

Date: 2026-06-20

Host: lizmark (`iarroyof@192.168.241.205`)

Run: `ROLE=scale SEQ_LEN=16384 SMOKE=0 IMAGE=set-attention:latest bash scripts/run_sd9_5_probes.sh`

Result: the `mixed-65` and `all-fine` seed-0 rows exited `1` before completing epoch 1 under the required landmark backend, `landmark_coverage=0.25`, batch 1. The `all-coarse` seed-0 row completed all 10 epochs.

Failure summary:
- `mixed-65`: CUDA OOM during the full WikiText evaluation/training path after the 16k smoke had passed; the larger full-run token vocabulary made the LM-head/CE allocation exceed available memory.
- `all-fine`: CUDA OOM in the landmark path at batch 1.

Guard decision: do not change backend, coverage, topology, batch size, or enable any non-SD-9.5 architecture knob. Record the failed rows as scale results and summarize the completed all-coarse row separately.
