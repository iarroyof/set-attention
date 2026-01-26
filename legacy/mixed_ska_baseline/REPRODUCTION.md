# Legacy Reproduction Notes

These commands reproduce the original Stage A/B sweeps. They are kept for
backward-compatibility only.

## Stage A Quality Sweep (legacy)

```bash
cd legacy/mixed_ska_baseline
python scripts/run_stageA_sweeps.py --sweep-yaml configs/stageA_quality.yaml
```

## Stage B Scaling Sweep (legacy)

```bash
cd legacy/mixed_ska_baseline
python scripts/run_stageB_sweeps.py --sweep-yaml configs/stageB_scaling.yaml
```
