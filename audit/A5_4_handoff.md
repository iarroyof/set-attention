# A5.4 Final Reproducibility Handoff

Status: PASS

## Scope

This handoff consolidates completed Phase A evidence after A1-A4, the v2.7 matched token-backend controls, A4.3 convergence, and A6 capacity/bottleneck ablations.

## Validation Summary

- Required manifests checked: 14 / 14
- Source CSVs checked: 428
- Source JSON metadata files checked: 428
- Indexed artifacts: 61
- Final artifact index: `out/paper_integrated_evidence/checks/final_artifact_index.tsv`
- Final reproducibility manifest: `out/paper_integrated_evidence/checks/final_reproducibility_manifest.json`

## Manifest Checks

| phase | manifest | status | validated | expected |
| --- | --- | --- | ---: | ---: |
| A2 | `out/paper_integrated_evidence/checks/a2_grid_manifest.json` | pass | 153 | 153 |
| A2.4 | `out/paper_integrated_evidence/checks/a2_baseline_controls_manifest.json` | pass | 30 | 30 |
| A3.1 | `out/paper_integrated_evidence/checks/a3_window_sweep_manifest.json` | pass | 36 | 36 |
| A3.1-control | `out/paper_integrated_evidence/checks/a3_window_baseline_controls_manifest.json` | pass | 24 | 24 |
| A3.2 | `out/paper_integrated_evidence/checks/a3_pooltau_sweep_manifest.json` | pass | 27 | 27 |
| A3.3 | `out/paper_integrated_evidence/checks/a3_stride_sweep_manifest.json` | pass | 24 | 24 |
| A4.1 | `out/paper_integrated_evidence/checks/a41_smoke_manifest.json` | pass | 2 | 2 |
| A4.2 | `out/paper_integrated_evidence/checks/a42_slice_manifest.json` | pass | 12 | 12 |
| A4.2-control | `out/paper_integrated_evidence/checks/a4_long_context_baseline_controls_manifest.json` | pass | 6 | 6 |
| A4.3 | `out/paper_integrated_evidence/checks/a4_convergence_manifest.json` | pass | 6 | 6 |
| A6.1 | `out/paper_integrated_evidence/checks/a6_dphi_capacity_manifest.json` | pass | 27 | 27 |
| A6.2 | `out/paper_integrated_evidence/checks/a6_set_state_width_manifest.json` | pass | 36 | 36 |
| A6.3 | `out/paper_integrated_evidence/checks/a6_interface_bottleneck_manifest.json` | pass | 54 | 54 |
| A6.4 | `out/paper_integrated_evidence/checks/a64_depth_sweep_manifest.json` | pass | 54 | 54 |

## TSV / Artifact Index

| artifact | rows | sha256 | intended use |
| --- | ---: | --- | --- |
| `out/paper_integrated_evidence/tables/a2_anchor_stability.tsv` | 3 | `f04918c7b134cb38918da8b6633a012113b7e880ff0b284321dfd5f1266c7547` | LR-normalized headline/family grids for main tables |
| `out/paper_integrated_evidence/tables/a2_grid_all_runs.tsv` | 153 | `fba64fbaede84e86ded86d7735529cce83f548b11e79e617ec4a361a79ea8510` | LR-normalized headline/family grids for main tables |
| `out/paper_integrated_evidence/tables/a2_lrnorm_family_best_by_family.tsv` | 4 | `acc5e65a59eee362168f1b3f52272ee1a77061c05397594d7f10c3231e888ecc` | LR-normalized headline/family grids for main tables |
| `out/paper_integrated_evidence/tables/a2_lrnorm_family_slice_all_runs.tsv` | 60 | `0142605028b5fca9dfb0e1ac533e104ee38e91b7dcb18eb636a0a7e0474969d4` | LR-normalized headline/family grids for main tables |
| `out/paper_integrated_evidence/tables/a2_lrnorm_headline_all_runs.tsv` | 120 | `85d41157590ba850a732a5b1935652abe3c6b924638181af04c74b61db7d6077` | LR-normalized headline/family grids for main tables |
| `out/paper_integrated_evidence/tables/a2_lrnorm_headline_best_by_pair.tsv` | 8 | `e757f168618b0cfc870e6dc26e0289557e1af266086e32a08e9698a7fa6ecdd0` | LR-normalized headline/family grids for main tables |
| `out/paper_integrated_evidence/tables/a2_baseline_controls_all_runs.tsv` | 30 | `63fe67926c31b398eabc5e8f83a443a83f31929728773d561f3d376ba62700e1` | v2.7 matched sparse/linear token controls for LR-normalized comparisons |
| `out/paper_integrated_evidence/tables/a2_baseline_controls_summary.tsv` | 10 | `9b05f736fbc2fe33b3b84f3ce26d171b64d8f99d021e275a899288c03058113a` | v2.7 matched sparse/linear token controls for LR-normalized comparisons |
| `out/paper_integrated_evidence/tables/a3_window_sweep_all_runs.tsv` | 36 | `e807fd71b3c3e90df8ecbe2709c7cebc47ea6cf767eae73e7ed0d534524d96dd` | fixed-stride set-family window sweep for mechanism figure |
| `out/paper_integrated_evidence/tables/a3_window_sweep_summary.tsv` | 18 | `0dd06a8ab4967a9c963ab0176951bef205d58480a774d8d64e950a76efc0d2f0` | fixed-stride set-family window sweep for mechanism figure |
| `out/paper_integrated_evidence/tables/a3_window_baseline_controls_all_runs.tsv` | 24 | `d2bba1f8c5eef8fc2638c111a4d70863ad16e90f3f2958b1131cc979a97b8da6` | matched token-control window overlays |
| `out/paper_integrated_evidence/tables/a3_window_baseline_controls_summary.tsv` | 12 | `77069a24fda22efa569b1584fa18cc676da81a41d5c39314af937b109396f4d8` | matched token-control window overlays |
| `out/paper_integrated_evidence/tables/a3_pooltau_sweep_all_runs.tsv` | 27 | `46ab57ddb39dde56e62c20a0ac3b9d72fcca8a7e6ca0187394d718a0e004d643` | pooling-temperature sweep with error bars |
| `out/paper_integrated_evidence/tables/a3_pooltau_sweep_summary.tsv` | 9 | `264d46cdd2df93aea87484592eadb5cdb1c1f05aeb1c7f9d1238e90eb378ebcd` | pooling-temperature sweep with error bars |
| `out/paper_integrated_evidence/tables/a3_stride_sweep_all_runs.tsv` | 24 | `74ffc789e6e2f3472c088dd3ceb1b8aa0dbee4585ec7cc0e8a53555df47fbdb0` | demoted stride-sweep complement |
| `out/paper_integrated_evidence/tables/a3_stride_sweep_summary.tsv` | 12 | `59c0ea30d105b03ec21903eca20ef1ddeaffc46e133241edcdd7448505a60956` | demoted stride-sweep complement |
| `out/paper_integrated_evidence/tables/a41_smoke_all_runs.tsv` | 2 | `ee0a908e162c06e51e09e1c45795f83e507c895530b20e027a27faa77ea70217` | long-context smoke gate |
| `out/paper_integrated_evidence/tables/a42_slice_all_runs.tsv` | 12 | `340bfd2ad579fcaae5ce2a776f83b0427cfecea09e9c281d72e5c81da9c5ce16` | long-context family slice |
| `out/paper_integrated_evidence/tables/a4_long_context_baseline_controls_all_runs.tsv` | 6 | `8e57791bf70375587d3b8e2250f81b3ca30cb7684be65687be287affd9074bbb` | matched long-context sparse/linear token controls |
| `out/paper_integrated_evidence/tables/a4_long_context_baseline_controls_summary.tsv` | 2 | `01f4f8fc4f9e0e204e0a83141808a54672058dddb4f448335160eddd8ee266ad` | matched long-context sparse/linear token controls |
| `out/paper_integrated_evidence/tables/a4_convergence_all_runs.tsv` | 6 | `8218766915d4268c26e914818f700a69107d512522d2fe45bac404175029d798` | 30-epoch convergence panel |
| `out/paper_integrated_evidence/tables/a4_convergence_summary.tsv` | 7 | `f05a5aac131f936ef70ee3756de7450bf4c74c4a01122343ce78bee23165dd36` | 30-epoch convergence panel |
| `out/paper_integrated_evidence/tables/a6_dphi_capacity_all_runs.tsv` | 27 | `d48711513b1993b8f00c67142fd39670260d77f64dae0a25f4922ec1c402ec63` | d_phi set-token interface capacity ablation |
| `out/paper_integrated_evidence/tables/a6_dphi_capacity_summary.tsv` | 9 | `e4c38d523037417c63d5dafc3a77ab1a8ac1be3e6e312744ed8b0a1553c4776b` | d_phi set-token interface capacity ablation |
| `out/paper_integrated_evidence/tables/a6_set_state_width_all_runs.tsv` | 36 | `11499b0f06855304bab5eaab7e3819b839fa1bef570bbbc5e4dfe35f2681f954` | explicit set-state dimensionality capacity ablation |
| `out/paper_integrated_evidence/tables/a6_set_state_width_summary.tsv` | 12 | `a939d4820ffabee1b30f39db5e1232d22b8965924e67d96a8e0fa68834f10f5d` | explicit set-state dimensionality capacity ablation |
| `out/paper_integrated_evidence/tables/a6_interface_bottleneck_all_runs.tsv` | 54 | `2c737641b9a73a6f3c8df87450efab0bcaa8d9c00a4a74037924a8b1025c24e8` | set-token interface bottleneck ablation |
| `out/paper_integrated_evidence/tables/a6_interface_bottleneck_summary.tsv` | 18 | `def1b7f0cf31b128add04c48825dee818b3b3c79dca9caf3caec92d7f5d03d5b` | set-token interface bottleneck ablation |
| `out/paper_integrated_evidence/tables/a64_depth_sweep_all_runs.tsv` | 54 | `d88b0d923682114b08af485dc3bfc2b93e4c877f62a9dea335dbe63083a17d21` | set-processing stack depth ablation |
| `out/paper_integrated_evidence/tables/a64_depth_sweep_summary.tsv` | 18 | `d58bffa5906f100389645838a83996082777cf5f01b3bb0393a48791fbd3a425` | set-processing stack depth ablation |

## Audits Checked

- `audit/A1_9_gate.json` sha256=c4bec99ad174feb45b972bd63e350f89e4388c4da0283eafb95b5e79f5d8ddd8
- `audit/A2_grid_handoff.md` sha256=8d2710552fb8ba3797dbeb642e406efd2a121834c1a14e97e5c3d841e441814f
- `audit/A2_4_baseline_controls.md` sha256=49d9f42595a8051185639c31adbc4e3117859d080c7707a1cbbe5080d888d2c0
- `audit/A3_1_window_sweep.md` sha256=319db4427ba832ba888d95f46da167cd20f347e725f38c6c1f58223464f268e6
- `audit/A3_1_baseline_controls.md` sha256=b17ec0ae7df15fcea9a65e8b65763317d1bc8bb7c1110e5bd0a77a68b60d57ae
- `audit/A3_2_pooltau_sweep.md` sha256=5bc8077a42648cc5b288b859e298138eb9fc9e0ebae16af6c99139cb2af7a1c9
- `audit/A3_3_stride_sweep.md` sha256=cfd22b185c0059bb10ec161b65930454bee8d038610283bfff9f7488b6f9bff0
- `audit/A4_1_smoke.md` sha256=79871f060e3c96eb580da260071527f30daf4fde29dd0626e1aa440af9dda73a
- `audit/A4_2_slice.md` sha256=298afc721a09e6f9190337e76bbaabbbc7aef1183da57d7cdee63c46ceae1c1d
- `audit/A4_2_baseline_controls.md` sha256=735ef022b7e32fbdb5b032d829f1a296ea9592da7f1fa2acf96d534911229f56
- `audit/A4_3_convergence.md` sha256=02f81b99f9d87ec085b948185ef56d45db1c4e6257d0346c731c013b47993cf7
- `audit/A6_1_dphi_capacity.md` sha256=996c1c46418db48491562c724f64f45625dd9e8ac3293dc31c8f8f45c15c6a8e
- `audit/A6_2_set_state_width.md` sha256=b2a5153569bce664a6e1402b21d0d76568ddf5160155011230170881e54cd93e
- `audit/A6_3_interface_bottleneck.md` sha256=7b8129897d0719699578b5990033c9fc4d8fb69ae6a061f01c773e0ac863a600
- `audit/A6_4_depth_sweep.md` sha256=17f39ccea7e364d14aaa4f045eebaabf2b796097c8a512e79e4b8f95b0e48663

## Matched-Controls Coverage Statement

- A2.4 supplies matched `baseline_sparse_local_band` and `baseline_linear_landmark` controls for LR-normalized headline/family comparisons.
- A3.1-control supplies matched token-backend overlays for the fixed-stride window sweep.
- A4.2-control supplies matched sparse/linear token controls at long context (`L=2048`).
- A4.3 supplies a 30-epoch panel containing dense/sparse/linear token baselines and dense/sparse/linear SKA variants.
- Historical A2/A3/A4 set-family artifacts that predate v2.7 are still valid, but any backend-family interpretation must use or cite the matched-control artifacts above.

## A6 Capacity Ablation Statement

- A6.1 shows that increasing `d_phi` helps some set families but does not produce a broad monotonic gain.
- A6.2 shows explicit `set_state_dim` helps SetSparse at 512 but does not broadly improve SKA.
- A6.3 shows moderate `d_phi` increases partially relieve interface bottlenecks, especially for SetLinear, but matched `d_phi=set_state_dim` often worsens PPL.
- A6.4 rejects the set-stack-depth bottleneck under this budget: depth 8/10 worsens validation PPL versus depth 6 across all tested families and capacity pairs.

## Writing-Agent Caveats

- A4.3 convergence favors token baselines over SKA at the 30-epoch LR-normalized reference; do not overclaim convergence wins for SKA.
- SKA memory advantage appears in the long-context slice: compare `audit/A4_2_slice.md` and `audit/A4_2_baseline_controls.md` before writing the long-context claim.
- Tables/figures must distinguish dense-baseline-only historical artifacts from v2.7 matched-control artifacts.
- A6 capacity ablations should be written as diagnostics, not as evidence of a simple missing-capacity fix.
- App D.2 should be dropped or rebuilt from the canonical LR-normalized baseline per `audit/A1_3_reconciliation.md`.

## Failures

- None.
