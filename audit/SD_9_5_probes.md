# SD-9.5 Mechanism Probes

Guards: CE-only, `anchor.enabled=false`, `token_mlp.enabled=false`, `candidate_fiber=endpoint_window`, `output_residual_mode=anchor_span`, multiresolution only.

## Short Mechanism Attribution
- Mixed short PPL: `860.6289876302084`.
- Fine ablation ΔPPL: `22624.807861328125`.
- Coarse ablation ΔPPL: `84.09566243489583`.
- Effective range fine/coarse: `0.634298667061529` / `2.9737754710477224`.
- Routing entropy fine/coarse: `0.2817320933254312` / `0.6674390426166147`.
- Routing top-1 fine/coarse: `0.8885414274476415` / `0.5816219256078596`.

## Scale-L Sweep
- No completed scale rows summarized yet.

## Validation
- `scan_logs()` found no nan/inf/traceback/OOM markers in summarized logs.
