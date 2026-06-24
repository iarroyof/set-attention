# SD-3 Fairness Audit Harness

Status: PASS

Harness checks:
- Thin-anchor span ablation exact: True
- Token MLP disabled for anchor_span: True
- Pre-encoder/head excluded from inference count: True

Anchor-reference parameter counts:
- Set-dictionary inference params: 14487936
- Set-dictionary train params: 18086016
- Anchor pre-encoder/head params: 3598080
- Matched dense token params: 10941696
- Inference minus token params: 3546240

Small smoke:
- Normal loss: 5.325502872467041
- Span-ablated loss: 5.268815517425537
- Span-ablation delta loss: -0.056687355041503906
- Train peak VRAM MiB: 17.03857421875
- Inference peak VRAM MiB: 17.5126953125

Notes:
- Synthetic span-ablation delta is a harness smoke value, not a trained PPL collapse claim.
- Actual ladder summaries must fail a run if trained span ablation does not sharply worsen PPL.
- Manifest: `out/paper_integrated_evidence/checks/sd_fairness_harness_smoke.json`
