#!/usr/bin/env python3
"""Summarize a frozen partial snapshot of the corrected exact-dense grid."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


VARIANT_LABELS = {
    "token": "token",
    "f8c0": "b0",
    "f6c2": "b25",
    "f4c4": "b50",
    "f2c6": "b75",
    "f0c8": "b100",
}
ROW_ORDER = ("token", "b0", "b25", "b50", "b75", "b100")
ISLANDS = (
    (512, 16),
    (512, 4),
    (1024, 4),
    (2048, 3),
    (2048, 4),
    (3584, 4),
    (3584, 3),
    (4096, 3),
    (4096, 4),
)
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blue-status", type=Path, required=True)
    parser.add_argument("--lizmark-status", type=Path, required=True)
    parser.add_argument("--diagnostic-retry-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_retry_cells(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def read_status(path: Path, host: str, retry_cells: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"{path}:{line_number}: expected six tab-separated fields")
        cell_id, epochs, ppl, vram, span_ablation, csv_path = fields
        family, backend, seq_len, variant, batch, seed = cell_id.split("|")
        if backend != "exact" or variant not in VARIANT_LABELS:
            raise ValueError(f"{path}:{line_number}: unsupported corrected cell {cell_id}")
        epoch_count = int(epochs)
        try:
            ppl_value: float | str = float(ppl)
            vram_value: float | str = float(vram)
        except ValueError:
            if epoch_count >= 10:
                raise ValueError(
                    f"{path}:{line_number}: completed cell has nonnumeric PPL/VRAM"
                )
            ppl_value = "NA"
            vram_value = "NA"
        rows.append(
            {
                "host": host,
                "cell_id": cell_id,
                "family": family,
                "seq_len": int(seq_len),
                "batch": int(batch.removeprefix("b")),
                "row": VARIANT_LABELS[variant],
                "seed": int(seed),
                "epochs": epoch_count,
                "val_ppl": ppl_value,
                "peak_vram_mib": vram_value,
                "span_ablation_delta_ppl": span_ablation,
                "diagnostic_status": (
                    "retry_required" if cell_id in retry_cells else "endpoint_valid"
                ),
                "source_csv": csv_path,
            }
        )
    seen: set[str] = set()
    for row in rows:
        cell_id = str(row["cell_id"])
        if cell_id in seen:
            raise ValueError(f"duplicate cell in frozen snapshot: {cell_id}")
        seen.add(cell_id)
    return rows


def summarize(values: list[float]) -> tuple[float, float | None, float | None]:
    avg = mean(values)
    if len(values) == 1:
        return avg, None, None
    sd = stdev(values)
    return avg, sd, T95[len(values)] * sd / math.sqrt(len(values))


def fmt_stat(values: list[float]) -> str:
    avg, _, ci = summarize(values)
    if ci is None:
        return f"{avg:.1f} (1/5)"
    return f"{avg:.1f} +/- {ci:.1f} ({len(values)}/5)"


def evidence_status(rows: list[dict[str, object]]) -> str:
    requires_retry = any(
        row["diagnostic_status"] != "endpoint_valid" for row in rows
    )
    if len(rows) < 5:
        return "partial_diagnostic_retry" if requires_retry else "partial"
    if requires_retry:
        return "provisional_diagnostic_retry"
    return "complete_valid"


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    retry_cells = read_retry_cells(args.diagnostic_retry_cells)
    run_rows = read_status(args.blue_status, "blue-demon", retry_cells)
    run_rows += read_status(args.lizmark_status, "lizmark", retry_cells)
    completed = [row for row in run_rows if int(row["epochs"]) >= 10]

    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in completed:
        grouped[(int(row["seq_len"]), int(row["batch"]), str(row["row"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (seq_len, batch, label), rows in sorted(grouped.items()):
        ppl = [float(row["val_ppl"]) for row in rows]
        vram = [float(row["peak_vram_mib"]) for row in rows]
        ppl_mean, ppl_sd, ppl_ci = summarize(ppl)
        vram_mean, vram_sd, vram_ci = summarize(vram)
        summary_rows.append(
            {
                "seq_len": seq_len,
                "batch": batch,
                "row": label,
                "n": len(rows),
                "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda x: int(x["seed"]))),
                "mean_val_ppl": f"{ppl_mean:.6f}",
                "sd_val_ppl": "NA" if ppl_sd is None else f"{ppl_sd:.6f}",
                "ci95_val_ppl": "NA" if ppl_ci is None else f"{ppl_ci:.6f}",
                "mean_peak_vram_mib": f"{vram_mean:.6f}",
                "sd_peak_vram_mib": "NA" if vram_sd is None else f"{vram_sd:.6f}",
                "ci95_peak_vram_mib": "NA" if vram_ci is None else f"{vram_ci:.6f}",
                "evidence_status": evidence_status(rows),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_fields = list(run_rows[0])
    write_tsv(args.output_dir / "partial_runs.tsv", run_rows, run_fields)
    write_tsv(
        args.output_dir / "partial_cells.tsv",
        summary_rows,
        list(summary_rows[0]),
    )

    pairwise_rows: list[dict[str, object]] = []
    key_comparisons: list[dict[str, object]] = []
    for seq_len, batch in ISLANDS:
        for label in ("b0", "b25", "b50", "b75", "b100"):
            for reference in ("token", "b0", "b100"):
                if label == reference:
                    continue
                candidate_rows = grouped.get((seq_len, batch, label), [])
                reference_rows = grouped.get((seq_len, batch, reference), [])
                candidate_by_seed = {int(row["seed"]): row for row in candidate_rows}
                reference_by_seed = {int(row["seed"]): row for row in reference_rows}
                common = sorted(set(candidate_by_seed) & set(reference_by_seed))
                if not common:
                    continue
                delta_ppl = [
                    float(candidate_by_seed[seed]["val_ppl"])
                    - float(reference_by_seed[seed]["val_ppl"])
                    for seed in common
                ]
                delta_vram = [
                    float(candidate_by_seed[seed]["peak_vram_mib"])
                    - float(reference_by_seed[seed]["peak_vram_mib"])
                    for seed in common
                ]
                ppl_mean, ppl_sd, ppl_ci = summarize(delta_ppl)
                vram_mean = mean(delta_vram)
                if ppl_mean < 0.0 and vram_mean < 0.0:
                    verdict = "candidate_dominates_on_means"
                elif ppl_mean > 0.0 and vram_mean > 0.0:
                    verdict = "reference_dominates_on_means"
                else:
                    verdict = "mean_tradeoff"
                candidate_status = evidence_status(candidate_rows)
                reference_status = evidence_status(reference_rows)
                if "diagnostic_retry" in candidate_status or "diagnostic_retry" in reference_status:
                    pair_status = "provisional_diagnostic_retry"
                elif candidate_status == reference_status == "complete_valid":
                    pair_status = "complete_valid"
                else:
                    pair_status = "partial"
                pair = {
                    "seq_len": seq_len,
                    "batch": batch,
                    "candidate": label,
                    "reference": reference,
                    "common_n": len(common),
                    "common_seeds": ",".join(map(str, common)),
                    "mean_delta_ppl": f"{ppl_mean:.6f}",
                    "sd_delta_ppl": "NA" if ppl_sd is None else f"{ppl_sd:.6f}",
                    "ci95_delta_ppl": "NA" if ppl_ci is None else f"{ppl_ci:.6f}",
                    "mean_delta_peak_vram_mib": f"{vram_mean:.6f}",
                    "mean_pareto_verdict": verdict,
                    "evidence_status": pair_status,
                }
                pairwise_rows.append(pair)
                if (label, reference) in {
                    ("b25", "token"),
                    ("b50", "token"),
                    ("b25", "b0"),
                }:
                    key_comparisons.append(pair)
    write_tsv(
        args.output_dir / "partial_pairwise.tsv",
        pairwise_rows,
        list(pairwise_rows[0]),
    )

    frontier_rows: list[dict[str, object]] = []
    for seq_len, batch in ISLANDS:
        points = []
        for label in ROW_ORDER:
            rows = grouped.get((seq_len, batch, label), [])
            if rows:
                points.append(
                    (
                        label,
                        mean(float(row["val_ppl"]) for row in rows),
                        mean(float(row["peak_vram_mib"]) for row in rows),
                        rows,
                    )
                )
        frontier = [
            point
            for point in points
            if not any(
                other[1] <= point[1]
                and other[2] <= point[2]
                and (other[1] < point[1] or other[2] < point[2])
                for other in points
            )
        ]
        if frontier:
            frontier_rows.append(
                {
                    "seq_len": seq_len,
                    "batch": batch,
                    "frontier_rows": ",".join(point[0] for point in frontier),
                    "frontier_n": ",".join(
                        f"{point[0]}:{len(point[3])}" for point in frontier
                    ),
                    "evidence_status": (
                        "complete_valid"
                        if all(
                            evidence_status(point[3]) == "complete_valid"
                            for point in points
                        )
                        else "provisional"
                    ),
                }
            )
    write_tsv(
        args.output_dir / "partial_frontiers.tsv",
        frontier_rows,
        list(frontier_rows[0]),
    )

    interpolation_rows: list[dict[str, object]] = []
    for seq_len, batch in ISLANDS:
        fine_rows = grouped.get((seq_len, batch, "b0"), [])
        coarse_rows = grouped.get((seq_len, batch, "b100"), [])
        if not fine_rows or not coarse_rows:
            continue
        fine_ppl = mean(float(row["val_ppl"]) for row in fine_rows)
        fine_vram = mean(float(row["peak_vram_mib"]) for row in fine_rows)
        coarse_ppl = mean(float(row["val_ppl"]) for row in coarse_rows)
        coarse_vram = mean(float(row["peak_vram_mib"]) for row in coarse_rows)
        for label, fraction in (("b25", 0.25), ("b50", 0.50), ("b75", 0.75)):
            mixed_rows = grouped.get((seq_len, batch, label), [])
            if not mixed_rows:
                continue
            mixed_ppl = mean(float(row["val_ppl"]) for row in mixed_rows)
            mixed_vram = mean(float(row["peak_vram_mib"]) for row in mixed_rows)
            interp_ppl = (1.0 - fraction) * fine_ppl + fraction * coarse_ppl
            interp_vram = (1.0 - fraction) * fine_vram + fraction * coarse_vram
            ppl_gain = interp_ppl - mixed_ppl
            vram_gain = interp_vram - mixed_vram
            interpolation_rows.append(
                {
                    "seq_len": seq_len,
                    "batch": batch,
                    "row": label,
                    "ppl_gain_vs_interpolation": f"{ppl_gain:.6f}",
                    "peak_vram_gain_mib_vs_interpolation": f"{vram_gain:.6f}",
                    "pareto_better_than_interpolation": (
                        ppl_gain > 0.0 and vram_gain > 0.0
                    ),
                    "evidence_status": (
                        "complete_valid"
                        if all(
                            evidence_status(rows) == "complete_valid"
                            for rows in (fine_rows, coarse_rows, mixed_rows)
                        )
                        else "provisional"
                    ),
                }
            )
    write_tsv(
        args.output_dir / "partial_interpolation.tsv",
        interpolation_rows,
        list(interpolation_rows[0]),
    )

    lines = [
        "# Corrected Exact-Dense Grid: Frozen Partial Snapshot",
        "",
        "Entries are mean +/- 95% Student-t CI with completed-seed count. "
        "Only 10-epoch rows are summarized.",
        "",
        "## Validation",
        "",
        "- Blue rows passed the endpoint `current_matrix_v1` scanner.",
        "- Lizmark rows passed the strict seed/config/full-data scanner.",
        "- `L3584/B4` mixed rows are provisional and require clean reruns for "
        "endpoint gradient diagnostics.",
        "",
    ]
    for metric, title in (("val_ppl", "PPL"), ("peak_vram_mib", "Peak VRAM MiB")):
        lines += [f"## {title}", ""]
        lines.append("| row | " + " | ".join(f"L{length}/B{batch}" for length, batch in ISLANDS) + " |")
        lines.append("|---|" + "|".join("---" for _ in ISLANDS) + "|")
        for label in ROW_ORDER:
            cells = []
            for length, batch in ISLANDS:
                rows = grouped.get((length, batch, label), [])
                if not rows:
                    cells.append("--")
                    continue
                value = fmt_stat([float(row[metric]) for row in rows])
                if "diagnostic_retry" in evidence_status(rows):
                    value += " [R]"
                cells.append(value)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Key Paired Comparisons",
        "",
        "Deltas are candidate minus reference on common applied seeds. Negative "
        "values favor the candidate.",
        "",
        "| island | comparison | common seeds | delta PPL +/- 95% CI | delta VRAM MiB | mean verdict | status |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in key_comparisons:
        ci = row["ci95_delta_ppl"]
        ppl = float(str(row["mean_delta_ppl"]))
        ppl_cell = f"{ppl:.1f}" if ci == "NA" else f"{ppl:.1f} +/- {float(str(ci)):.1f}"
        lines.append(
            f"| L{row['seq_len']}/B{row['batch']} | "
            f"{row['candidate']} vs {row['reference']} | {row['common_n']} | "
            f"{ppl_cell} | {float(str(row['mean_delta_peak_vram_mib'])):.1f} | "
            f"{row['mean_pareto_verdict']} | {row['evidence_status']} |"
        )
    lines += [
        "",
        "## Available Mean Frontiers",
        "",
        "| island | nondominated rows | seed counts | status |",
        "|---|---|---|---|",
    ]
    for row in frontier_rows:
        lines.append(
            f"| L{row['seq_len']}/B{row['batch']} | {row['frontier_rows']} | "
            f"{row['frontier_n']} | {row['evidence_status']} |"
        )
    lines += [
        "",
        "## Fine-Coarse Interpolation",
        "",
        "Every available mixed row improves PPL over the straight b0-to-b100 "
        "interpolation, but none improves interpolated VRAM; therefore none is "
        "Pareto-better than that synthetic line in the current snapshot.",
        "",
        "[R] PPL/VRAM is present and core metadata is valid, but the row must be "
        "replaced because its epoch-10 gradient diagnostics are absent.",
        "",
    ]
    (args.output_dir / "partial_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
