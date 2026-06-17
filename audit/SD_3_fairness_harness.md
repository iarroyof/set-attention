# SD-3 Fairness Audit Harness

Status: PASS

Harness checks:
- Thin-anchor span ablation exact: True
- Token MLP disabled for anchor_span: True
- Pre-encoder excluded from inference count: True

Anchor-reference parameter counts:
- Set-dictionary inference params: 14487936
- Set-dictionary train params: 18036864
- Anchor pre-encoder params: 3548928
- Matched dense token params: 10941696
- Inference minus token params: 3546240

Small smoke:
- Normal loss: 5.503087997436523
- Span-ablated loss: 5.339049339294434
- Span-ablation delta loss: -0.16403865814208984
- Train peak VRAM MiB: 16.8193359375
- Inference peak VRAM MiB: 17.5126953125

Notes:
- Synthetic span-ablation delta is a harness smoke value, not a trained PPL collapse claim.
- Actual ladder summaries must fail a run if trained span ablation does not sharply worsen PPL.
- Manifest: `out/paper_integrated_evidence/checks/sd_fairness_harness_smoke.json`
