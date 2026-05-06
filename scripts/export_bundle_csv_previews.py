#!/usr/bin/env python3
import csv
from pathlib import Path

ROOT = Path.home() / "set-attention" / "out" / "final_paper_bundle"
SUM = ROOT / "tables" / "summary"
OUT = ROOT / "checks" / "csv_previews"
OUT.mkdir(parents=True, exist_ok=True)

files = [
    "paper_action0_anchor_seed_sweep.tsv",
    "paper_action0_topology_temp.tsv",
    "paper_action1_boundary_stride.tsv",
    "paper_action1_pooltau_sweep.tsv",
    "actionD_epoch10_summary.tsv",
    "paper_action1A_parity_LR1e-4_epoch10.tsv",
]

for name in files:
    src = SUM / name
    dst = OUT / (src.stem + ".preview.txt")
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    lines = []
    lines.append(f"# FILE: {name}")
    lines.append(f"# ROWS: {len(rows)}")
    lines.append(f"# COLUMNS: {list(rows[0].keys()) if rows else []}")
    lines.append("")
    for r in rows[:20]:
        lines.append(str(r))
    dst.write_text("\n".join(lines))

print(f"Wrote previews to {OUT}")
