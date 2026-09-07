#!/usr/bin/env python3
"""Build current MRP-1 paper assets from the strict final summary TSV.

This script intentionally uses only the Python standard library and LaTeX/TikZ.
It creates:

- a compact main-text table for the selected frontier rows;
- a blur-path PPL/VRAM frontier figure;
- an operating-regime heatmap;
- an allocation/mechanism diagnostic figure.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "out/paper_integrated_evidence/checks/sd_grid_seeded_v1_final_20260708/cells.tsv"
RUNS_ROOT = ROOT / "out/paper_mechanisms/sd_grid_seeded_v1"
TABLE_OUT = ROOT / "out/final_paper_bundle/overleaf_ready/tables/sd_grid_compact_frontier.tex"
FINAL_MATRIX_OUT = ROOT / "out/final_paper_bundle/overleaf_ready/tables/sd_grid_final_matrix.tex"
PLOT_TEX = ROOT / "out/final_paper_bundle/plots/main/fig_sd_exact_dense_frontier.tex"
BLUR_PATH_TEX = ROOT / "out/final_paper_bundle/plots/main/fig_sd_blur_path_frontier.tex"
REGIME_TEX = ROOT / "out/final_paper_bundle/plots/main/fig_sd_operating_regime_map.tex"
MECHANISM_TEX = ROOT / "out/final_paper_bundle/plots/main/fig_sd_mechanism_allocation.tex"
ROUTING_TEX = ROOT / "out/final_paper_bundle/plots/main/fig_sd_group_routing_diagnostics.tex"
ABLATION_TEX = ROOT / "out/final_paper_bundle/plots/main/fig_sd_span_ablation_diagnostics.tex"
BUCKET_TEX = ROOT / "out/final_paper_bundle/plots/main/fig_sd_bucket_diagnostics.tex"

ROWS = ["token", "b0", "b25", "b50", "b75", "b100"]
ROW_LABEL = {
    "token": "Token",
    "b0": "b0",
    "b25": "b25",
    "b50": "b50",
    "b75": "b75",
    "b100": "b100",
}
BLUR_VALUE = {"b0": 0, "b25": 25, "b50": 50, "b75": 75, "b100": 100}
SET_ROWS = ["b0", "b25", "b50", "b75", "b100"]
ROW_COLOR = {
    "token": "tokenblack",
    "b0": "fineblue",
    "b25": "fineteal",
    "b50": "coarseamber",
    "b75": "coarsevermillion",
    "b100": "coarsegray",
}
ROW_MARK = {
    "token": "*",
    "b0": "square*",
    "b25": "triangle*",
    "b50": "diamond*",
    "b75": "pentagon*",
    "b100": "*",
}
LABEL_OFFSET = {
    "token": (0.12, -0.20, "west"),
    "b0": (-0.58, 0.12, "west"),
    "b25": (0.12, -0.22, "west"),
    "b50": (0.12, 0.08, "west"),
    "b75": (0.12, 0.08, "west"),
    "b100": (0.12, 0.02, "west"),
}
PLOT_ISLANDS = [(2048, 4), (3584, 4), (4096, 3), (4096, 4)]
REGIME_ISLANDS = [(512, 3), (512, 4), (512, 16), (1024, 3), (1024, 4), (2048, 3), (2048, 4), (3584, 3), (3584, 4), (4096, 3), (4096, 4)]
MATRIX_ISLANDS = [(512, 3), (512, 4), (512, 16), (1024, 3), (1024, 4), (2048, 3), (2048, 4), (3584, 3), (3584, 4), (4096, 3), (4096, 4)]
COMPACT_ISLANDS = [(2048, 4), (3584, 4), (4096, 3), (4096, 4)]
DIAG_ISLANDS = [(3584, 4), (4096, 3)]
DIAG_ALL_ISLANDS = [(512, 3), (512, 4), (512, 16), (1024, 3), (1024, 4), (2048, 3), (2048, 4), (3584, 3), (3584, 4), (4096, 3), (4096, 4)]
DIAG_ROWS = ["b25", "b50", "b75"]
RUN_RE = re.compile(
    r"sdgrid_seeded_v1_(?:(?P<set_family>set)_(?P<row>b\d+)_L(?P<set_L>\d+)_exact_b(?P<set_B>\d+)|"
    r"(?P<token_family>token)_exact_L(?P<token_L>\d+)_b(?P<token_B>\d+))_seed(?P<seed>\d+)\.csv$"
)


def load_cells() -> dict[tuple[int, int, str], dict[str, str]]:
    with CELLS.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        out: dict[tuple[int, int, str], dict[str, str]] = {}
        for row in reader:
            key = (int(row["L"]), int(row["B"]), row["row"])
            out[key] = row
        return out


def load_seed_rows() -> dict[tuple[int, int, str], list[dict[str, str]]]:
    out: dict[tuple[int, int, str], list[dict[str, str]]] = {}
    for path in RUNS_ROOT.glob("*/*/*.csv"):
        match = RUN_RE.match(path.name)
        if not match:
            continue
        if match.group("set_family"):
            row_name = match.group("row")
            key = (int(match.group("set_L")), int(match.group("set_B")), row_name)
        else:
            key = (int(match.group("token_L")), int(match.group("token_B")), "token")
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        last = max(rows, key=lambda r: int(float(r.get("epoch", "0") or "0")))
        out.setdefault(key, []).append(last)
    return out


def _numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", "NA", "nan", "None"):
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def _mean_string(rows: list[dict[str, str]], key: str) -> str:
    values = _numeric_values(rows, key)
    return "" if not values else str(statistics.fmean(values))


def _sd_string(rows: list[dict[str, str]], key: str) -> str:
    values = _numeric_values(rows, key)
    if len(values) < 2:
        return "0.0" if values else ""
    return str(statistics.stdev(values))


def summarize_seed_cells(
    seed_rows: dict[tuple[int, int, str], list[dict[str, str]]],
    fallback: dict[tuple[int, int, str], dict[str, str]],
) -> dict[tuple[int, int, str], dict[str, str]]:
    metric_map = {
        "span_mean": "val/span_ablation_delta_ppl",
        "span_fine_mean": "val/span_ablation_fine_delta_ppl",
        "span_coarse_mean": "val/span_ablation_coarse_delta_ppl",
        "range_fine_mean": "val/effective_range_fine",
        "range_coarse_mean": "val/effective_range_coarse",
        "ent_fine_mean": "val/routing_entropy_fine",
        "ent_coarse_mean": "val/routing_entropy_coarse",
        "top1_fine_mean": "val/routing_top1_fine",
        "top1_coarse_mean": "val/routing_top1_coarse",
        "early_freq": "val/loss_early_freq",
        "early_rare": "val/loss_early_rare",
        "late_freq": "val/loss_late_freq",
        "late_rare": "val/loss_late_rare",
    }
    out = dict(fallback)
    for key, rows in seed_rows.items():
        if not rows:
            continue
        L, B, row_name = key
        seeds = sorted({int(float(row.get("training.seed", "-1"))) for row in rows})
        entry = {
            "L": str(L),
            "B": str(B),
            "row": row_name,
            "family": "token" if row_name == "token" else "set",
            "n": str(len(rows)),
            "seeds": ",".join(str(seed) for seed in seeds),
            "ppl_mean": _mean_string(rows, "val/ppl"),
            "ppl_sd": _sd_string(rows, "val/ppl"),
            "vram_mean": _mean_string(rows, "train/peak_vram_mib"),
            "vram_sd": _sd_string(rows, "train/peak_vram_mib"),
        }
        for out_key, metric in metric_map.items():
            entry[out_key] = _mean_string(rows, metric)
        out[key] = entry
    return out


def write_cells_tsv(cells: dict[tuple[int, int, str], dict[str, str]]) -> None:
    fieldnames = [
        "L",
        "B",
        "row",
        "family",
        "n",
        "seeds",
        "ppl_mean",
        "ppl_sd",
        "vram_mean",
        "vram_sd",
        "span_mean",
        "span_fine_mean",
        "span_coarse_mean",
        "range_fine_mean",
        "range_coarse_mean",
        "ent_fine_mean",
        "ent_coarse_mean",
        "top1_fine_mean",
        "top1_coarse_mean",
        "early_freq",
        "early_rare",
        "late_freq",
        "late_rare",
    ]
    CELLS.parent.mkdir(parents=True, exist_ok=True)
    with CELLS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for key in sorted(cells):
            writer.writerow({name: cells[key].get(name, "") for name in fieldnames})


def _mean_ci(rows: list[dict[str, str]], key: str, *, scale: float = 1.0) -> tuple[float, float, int] | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", "NA", "nan", "None"):
            continue
        values.append(float(value) * scale)
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0, 1
    sd = statistics.stdev(values)
    # 95% t critical value for n=5. The final matrix is five-seed; this remains
    # conservative for any smaller censored subset.
    tcrit = 2.776 if len(values) <= 5 else 1.96
    return mean, tcrit * sd / math.sqrt(len(values)), len(values)


def _island_label(L: int, B: int) -> str:
    if L % 1024 == 0:
        return f"{L // 1024}k/B{B}"
    return f"{L / 1024:.1f}k/B{B}"


def fmt_ppl(row: dict[str, str] | None) -> str:
    if row is None:
        return "--"
    return f"{float(row['ppl_mean']):.1f} $\\pm$ {float(row['ppl_sd']):.1f}"


def fmt_vram(row: dict[str, str] | None) -> str:
    if row is None:
        return "--"
    return f"{float(row['vram_mean']):.0f}"


def _bold_best(values: list[tuple[str, float | None]], formatted: dict[str, str], *, lower_is_better: bool = True) -> dict[str, str]:
    numeric = [(name, value) for name, value in values if value is not None]
    if not numeric:
        return formatted
    best = min(value for _name, value in numeric) if lower_is_better else max(value for _name, value in numeric)
    out = dict(formatted)
    for name, value in numeric:
        if value is not None and math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12):
            out[name] = f"\\textbf{{{out[name]}}}"
    return out


def write_final_matrix_table(cells: dict[tuple[int, int, str], dict[str, str]]) -> None:
    FINAL_MATRIX_OUT.parent.mkdir(parents=True, exist_ok=True)

    def fmt_combined(row: dict[str, str] | None, *, best_ppl: bool, best_vram: bool) -> str:
        if row is None or row.get("ppl_mean", "") == "" or row.get("vram_mean", "") == "":
            return "--"
        ppl = f"{float(row['ppl_mean']):.1f}$\\pm${float(row['ppl_sd']):.1f}"
        gib = f"{float(row['vram_mean']) / 1024.0:.1f} GiB"
        if best_ppl:
            ppl = f"\\textbf{{{ppl}}}"
        if best_vram:
            gib = f"\\textbf{{{gib}}}"
        return f"\\begin{{tabular}}{{@{{}}c@{{}}}}{ppl}\\\\{{\\scriptsize {gib}}}\\end{{tabular}}"

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\renewcommand{\\arraystretch}{1.18}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "$L$ / batch & Token & b0 & b25 & b50 & b75 & b100 \\\\",
        "\\midrule",
    ]
    for L, B in MATRIX_ISLANDS:
        row_cells = {name: cells.get((L, B, name)) for name in ROWS}
        if not any(row_cells.values()):
            continue
        ppl_values = {
            name: (None if row is None or row.get("ppl_mean", "") == "" else float(row["ppl_mean"]))
            for name, row in row_cells.items()
        }
        vram_values = {
            name: (None if row is None or row.get("vram_mean", "") == "" else float(row["vram_mean"]))
            for name, row in row_cells.items()
        }
        valid_ppl = [value for value in ppl_values.values() if value is not None]
        valid_vram = [value for value in vram_values.values() if value is not None]
        best_ppl = min(valid_ppl) if valid_ppl else None
        best_vram = min(valid_vram) if valid_vram else None
        cells_tex = []
        for name in ROWS:
            cells_tex.append(
                fmt_combined(
                    row_cells[name],
                    best_ppl=ppl_values[name] is not None and math.isclose(ppl_values[name], best_ppl or 0.0, rel_tol=1e-12, abs_tol=1e-12),
                    best_vram=vram_values[name] is not None and math.isclose(vram_values[name], best_vram or 0.0, rel_tol=1e-12, abs_tol=1e-12),
                )
            )
        lines.append(f"{L}/B{B} & " + " & ".join(cells_tex) + " \\\\")
        lines.append("\\addlinespace[1pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Exact-dense multiresolution set-dictionary matrix. Each cell gives validation PPL on the first line and peak training memory on the second line. Each endpoint-valid cell uses seeds 0--4, 10 epochs, matched D=384/6L/8H/CE-only/anchor-span/token-MLP-off guards. b0 is all-fine; b25--b75 are mixed fine/coarse allocations; b100 is all-coarse. Bold on the first line marks the lowest available PPL in that island; bold on the second line marks the lowest peak memory. Missing L4096/B4 token, b0, and b25 entries are the dense memory boundary rather than failed statistical endpoints.}",
            "\\label{tab:sd-grid-final-matrix}",
            "\\end{table*}",
            "",
        ]
    )
    FINAL_MATRIX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_compact_table(cells: dict[tuple[int, int, str], dict[str, str]]) -> None:
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Island & Metric & Token & b0 & b25 & b50 \\\\",
        "\\midrule",
    ]
    for L, B in COMPACT_ISLANDS:
        token = cells.get((L, B, "token"))
        b0 = cells.get((L, B, "b0"))
        b25 = cells.get((L, B, "b25"))
        b50 = cells.get((L, B, "b50"))
        lines.append(
            f"L{L}/B{B} & PPL & {fmt_ppl(token)} & {fmt_ppl(b0)} & {fmt_ppl(b25)} & {fmt_ppl(b50)} \\\\"
        )
        lines.append(
            f" & peak VRAM & {fmt_vram(token)} & {fmt_vram(b0)} & {fmt_vram(b25)} & {fmt_vram(b50)} \\\\"
        )
        if (L, B) != COMPACT_ISLANDS[-1]:
            lines.append("\\addlinespace[1pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Compact exact-dense frontier controls.  b0 is the all-fine set endpoint, b25 is the frozen mixed operating point, and b50 is the first feasible moderate-blur row at the L4096/B4 boundary.  Missing L4096/B4 token, b0, and b25 entries are censored exact-dense feasibility observations, not assigned losses.}",
            "\\label{tab:sd-compact-frontier}",
            "\\end{table}",
            "",
        ]
    )
    TABLE_OUT.write_text("\n".join(lines), encoding="utf-8")


def _panel_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max(0.35, 0.12 * (xmax - xmin))
    ypad = max(20.0, 0.08 * (ymax - ymin))
    if math.isclose(xmin, xmax):
        xpad = max(1.0, 0.08 * abs(xmin))
    if math.isclose(ymin, ymax):
        ypad = max(20.0, 0.08 * abs(ymin))
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _scale(value: float, lo: float, hi: float, size: float) -> float:
    if math.isclose(lo, hi):
        return size / 2.0
    return (value - lo) / (hi - lo) * size


def _num(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def _cell_ci(row: dict[str, str], sd_key: str = "ppl_sd") -> float:
    """95% t half-width from the summarized five-seed cells."""
    try:
        sd = float(row.get(sd_key, "") or "0")
    except ValueError:
        return 0.0
    try:
        n = int(float(row.get("n", "") or "5"))
    except ValueError:
        n = 5
    if n <= 1:
        return 0.0
    tcrit = 2.776 if n <= 5 else 1.96
    return tcrit * sd / math.sqrt(n)


def _axis_ticks(lo: float, hi: float, n: int = 3) -> list[tuple[float, str]]:
    if n <= 1:
        return [(lo, f"{lo:.0f}")]
    return [(lo + i * (hi - lo) / (n - 1), f"{lo + i * (hi - lo) / (n - 1):.1f}") for i in range(n)]


def _color_for_delta(delta: float | None) -> str:
    if delta is None:
        return "gray!18"
    if delta <= -40:
        return "winstrong"
    if delta <= -15:
        return "winsoft"
    if delta <= 15:
        return "neutralcell"
    if delta <= 80:
        return "costsoft"
    return "coststrong"


def _fmt_signed(value: float) -> str:
    return f"{value:+.0f}"


def write_frontier_plot(cells: dict[tuple[int, int, str], dict[str, str]]) -> None:
    PLOT_TEX.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 5.7
    panel_h = 3.75
    x_gap = 1.15
    y_gap = 1.15
    lines = [
        "\\documentclass[border=7pt]{standalone}",
        "\\usepackage{tikz}",
        "\\usetikzlibrary{plotmarks}",
        "\\definecolor{tokenblack}{HTML}{374151}",
        "\\definecolor{fineblue}{HTML}{3B6EA8}",
        "\\definecolor{fineteal}{HTML}{4B9E8A}",
        "\\definecolor{coarseamber}{HTML}{C58A2B}",
        "\\definecolor{coarsevermillion}{HTML}{B65A4A}",
        "\\definecolor{coarsegray}{HTML}{6B7280}",
        "\\begin{document}",
        "\\begin{tikzpicture}[font=\\sffamily\\small]",
    ]

    for idx, (L, B) in enumerate(PLOT_ISLANDS):
        col = idx % 2
        row = idx // 2
        x0 = col * (panel_w + x_gap)
        y0 = -row * (panel_h + y_gap)
        token = cells.get((L, B, "token"))
        present: list[tuple[str, float, float, float]] = []
        for name in ROWS:
            item = cells.get((L, B, name))
            if item is None:
                continue
            if token is not None:
                x_value = (float(token["vram_mean"]) - float(item["vram_mean"])) / 1024.0
                y_value = float(item["ppl_mean"]) - float(token["ppl_mean"])
            else:
                x_value = float(item["vram_mean"]) / 1024.0
                y_value = float(item["ppl_mean"])
            present.append(
                (
                    name,
                    x_value,
                    y_value,
                    float(item["ppl_sd"]),
                )
            )
        xmin, xmax, ymin, ymax = _panel_bounds([(x, y) for _, x, y, _ in present])
        lines.append(f"\\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]")
        lines.append(f"\\draw[gray!45] (0,0) rectangle ({panel_w:.2f},{panel_h:.2f});")
        lines.append(
            f"\\node[anchor=west,font=\\sffamily\\bfseries\\small] at (0.08,{panel_h + 0.25:.2f}) {{$L={L}$, B={B}}};"
        )
        xlabel = "GiB saved vs token" if token is not None else "GiB used"
        ylabel = "$\\Delta$PPL vs token" if token is not None else "PPL"
        lines.append(f"\\draw[->] (0,0) -- ({panel_w + 0.18:.2f},0) node[right] {{{xlabel}}};")
        lines.append(f"\\draw[->] (0,0) -- (0,{panel_h + 0.18:.2f}) node[above] {{{ylabel}}};")
        if token is not None and xmin < 0 < xmax:
            zx = _scale(0.0, xmin, xmax, panel_w)
            lines.append(f"\\draw[gray!50,dashed] ({zx:.2f},0) -- ({zx:.2f},{panel_h:.2f});")
        if token is not None and ymin < 0 < ymax:
            zy = _scale(0.0, ymin, ymax, panel_h)
            lines.append(f"\\draw[gray!50,dashed] (0,{zy:.2f}) -- ({panel_w:.2f},{zy:.2f});")
        for frac in (0.0, 0.5, 1.0):
            xt = frac * panel_w
            xv = xmin + frac * (xmax - xmin)
            lines.append(f"\\draw[gray!30] ({xt:.2f},0) -- ({xt:.2f},-0.05) node[below,font=\\scriptsize,text=gray!65!black] {{{xv:.1f}}};")
            yt = frac * panel_h
            yv = ymin + frac * (ymax - ymin)
            lines.append(f"\\draw[gray!30] (0,{yt:.2f}) -- (-0.05,{yt:.2f}) node[left,font=\\scriptsize,text=gray!65!black] {{{yv:.0f}}};")
        for name, x, y, sd in present:
            px = _scale(x, xmin, xmax, panel_w)
            py = _scale(y, ymin, ymax, panel_h)
            err = min(panel_h * 0.18, _scale(y + sd, ymin, ymax, panel_h) - py)
            color = ROW_COLOR[name]
            mark = ROW_MARK[name]
            dx, dy, anchor = LABEL_OFFSET[name]
            lines.append(f"\\draw[{color},line width=0.45pt] ({px:.2f},{py-err:.2f}) -- ({px:.2f},{py+err:.2f});")
            lines.append(f"\\node[{color}] at ({px:.2f},{py:.2f}) {{\\Large\\pgfuseplotmark{{{mark}}}}};")
            lines.append(f"\\node[anchor={anchor},font=\\scriptsize,{color}] at ({px+dx:.2f},{py+dy:.2f}) {{{ROW_LABEL[name]}}};")
        if (L, B) == (4096, 4):
            lines.append(
                f"\\node[anchor=west,align=left,font=\\scriptsize,text=gray!65!black] at (0.10,0.55) {{token, b0, b25\\\\censored/OOM}};"
            )
        lines.append("\\end{scope}")

    legend_x = 0.0
    legend_y = -2 * (panel_h + y_gap) + 0.30
    lines.append(f"\\begin{{scope}}[shift={{({legend_x:.2f},{legend_y:.2f})}}]")
    lines.append("\\node[anchor=west,font=\\scriptsize,align=left] at (0,0) {Panels with token baselines show relative memory savings and PPL change; censored rows are omitted rather than assigned losses.};")
    lines.append("\\end{scope}")
    lines.extend(["\\end{tikzpicture}", "\\end{document}", ""])
    PLOT_TEX.write_text("\n".join(lines), encoding="utf-8")


def write_blur_path_frontier(cells: dict[tuple[int, int, str], dict[str, str]]) -> None:
    """A matrix-wide quality/efficiency frontier: follow the blur path."""
    BLUR_PATH_TEX.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 4.05
    panel_h = 2.10
    x_gap = 0.72
    y_gap = 0.54
    left_pad = 1.18
    top_pad = 0.48
    row_Ls = [512, 1024, 2048, 3584, 4096]
    col_Bs = [3, 4, 16]
    lines = [
        "\\documentclass[border=4pt]{standalone}",
        "\\usepackage{tikz}",
        "\\usetikzlibrary{plotmarks}",
        "\\definecolor{tokenblack}{HTML}{374151}",
        "\\definecolor{fineblue}{HTML}{3B6EA8}",
        "\\definecolor{fineteal}{HTML}{4B9E8A}",
        "\\definecolor{coarseamber}{HTML}{C58A2B}",
        "\\definecolor{coarsevermillion}{HTML}{B65A4A}",
        "\\definecolor{coarsegray}{HTML}{6B7280}",
        "\\begin{document}",
        "\\begin{tikzpicture}[font=\\sffamily\\small]",
    ]
    for col, B in enumerate(col_Bs):
        x = left_pad + col * (panel_w + x_gap) + panel_w / 2
        lines.append(f"\\node[font=\\bfseries\\scriptsize] at ({x:.2f},{top_pad:.2f}) {{batch {B}}};")
    for row, L in enumerate(row_Ls):
        y = -top_pad - row * (panel_h + y_gap) - panel_h / 2
        lines.append(f"\\node[anchor=east,font=\\bfseries\\scriptsize] at ({left_pad-0.56:.2f},{y:.2f}) {{$L={L}$}};")
    for row, L in enumerate(row_Ls):
        for col, B in enumerate(col_Bs):
            x0 = left_pad + col * (panel_w + x_gap)
            y0 = -top_pad - row * (panel_h + y_gap) - panel_h
            if (L, B) not in MATRIX_ISLANDS:
                lines.append(f"\\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]")
                lines.append(f"\\filldraw[fill=gray!8,draw=gray!35,line width=0.30pt] (0,0) rectangle ({panel_w:.2f},{panel_h:.2f});")
                lines.append(f"\\node[font=\\scriptsize,text=gray!55!black] at ({panel_w/2:.2f},{panel_h/2:.2f}) {{not run}};")
                lines.append("\\end{scope}")
                continue
            points: list[tuple[str, float, float, float]] = []
            for name in ROWS:
                item = cells.get((L, B, name))
                if item is None:
                    continue
                points.append((name, float(item["vram_mean"]) / 1024.0, float(item["ppl_mean"]), _cell_ci(item)))
            if not points:
                lines.append(f"\\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]")
                lines.append(f"\\filldraw[fill=gray!8,draw=gray!35,line width=0.30pt] (0,0) rectangle ({panel_w:.2f},{panel_h:.2f});")
                lines.append(f"\\node[font=\\scriptsize,text=gray!55!black] at ({panel_w/2:.2f},{panel_h/2:.2f}) {{no endpoint rows}};")
                lines.append("\\end{scope}")
                continue
            xmin, xmax, ymin, ymax = _panel_bounds([(x, y) for _, x, y, _ in points])
            lines.append(f"\\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]")
            lines.append(f"\\draw[gray!45,line width=0.34pt] (0,0) rectangle ({panel_w:.2f},{panel_h:.2f});")
            lines.append(f"\\draw[->] (0,0) -- ({panel_w+0.11:.2f},0);")
            lines.append(f"\\draw[->] (0,0) -- (0,{panel_h+0.11:.2f});")
            for frac in (0.0, 0.5, 1.0):
                xt = frac * panel_w
                xv = xmin + frac * (xmax - xmin)
                yt = frac * panel_h
                yv = ymin + frac * (ymax - ymin)
                lines.append(f"\\draw[gray!30] ({xt:.2f},0) -- ({xt:.2f},-0.04) node[below,font=\\tiny,text=gray!65!black] {{{xv:.1f}}};")
                lines.append(f"\\draw[gray!30] (0,{yt:.2f}) -- (-0.04,{yt:.2f}) node[left,font=\\tiny,text=gray!65!black] {{{yv:.0f}}};")
            path_points: list[tuple[float, float]] = []
            for name in SET_ROWS:
                item = cells.get((L, B, name))
                if item is None:
                    continue
                px = _scale(float(item["vram_mean"]) / 1024.0, xmin, xmax, panel_w)
                py = _scale(float(item["ppl_mean"]), ymin, ymax, panel_h)
                path_points.append((px, py))
            if len(path_points) > 1:
                coords = " -- ".join(f"({px:.2f},{py:.2f})" for px, py in path_points)
                lines.append(f"\\draw[gray!65,line width=0.65pt,->] {coords};")
            for name, x, y, sd in points:
                px = _scale(x, xmin, xmax, panel_w)
                py = _scale(y, ymin, ymax, panel_h)
                err = min(panel_h * 0.22, abs(_scale(y + sd, ymin, ymax, panel_h) - py))
                color = ROW_COLOR[name]
                mark = ROW_MARK[name]
                if name == "token":
                    lines.append(f"\\draw[{color},dashed,line width=0.42pt] ({px:.2f},0) -- ({px:.2f},{panel_h:.2f});")
                lines.append(f"\\draw[{color},line width=0.48pt] ({px:.2f},{py-err:.2f}) -- ({px:.2f},{py+err:.2f});")
                lines.append(f"\\draw[{color},line width=0.42pt] ({px-0.04:.2f},{py-err:.2f}) -- ({px+0.04:.2f},{py-err:.2f});")
                lines.append(f"\\draw[{color},line width=0.42pt] ({px-0.04:.2f},{py+err:.2f}) -- ({px+0.04:.2f},{py+err:.2f});")
                lines.append(f"\\node[{color}] at ({px:.2f},{py:.2f}) {{\\small\\pgfuseplotmark{{{mark}}}}};")
            if (L, B) == (4096, 4):
                lines.append("\\node[anchor=west,align=left,font=\\scriptsize,text=gray!65!black] at (0.10,0.33) {token, b0, b25 censored};")
            lines.append("\\end{scope}")
    legend_y = -top_pad - len(row_Ls) * (panel_h + y_gap) + 0.18
    legend_x = left_pad
    for name in ROWS:
        color = ROW_COLOR[name]
        mark = ROW_MARK[name]
        lines.append(f"\\node[{color}] at ({legend_x:.2f},{legend_y:.2f}) {{\\small\\pgfuseplotmark{{{mark}}}}};")
        lines.append(f"\\node[anchor=west,font=\\scriptsize] at ({legend_x+0.15:.2f},{legend_y:.2f}) {{{ROW_LABEL[name]}}};")
        legend_x += 1.02 if name != "token" else 1.30
    lines.append(f"\\node[anchor=west,font=\\scriptsize,text=gray!65!black] at ({left_pad:.2f},{legend_y-0.34:.2f}) {{Vertical bars are 95\\% t-intervals; dashed vertical line marks token peak GiB.}};")
    lines.extend(["\\end{tikzpicture}", "\\end{document}", ""])
    BLUR_PATH_TEX.write_text("\n".join(lines), encoding="utf-8")


def write_operating_regime_map(cells: dict[tuple[int, int, str], dict[str, str]]) -> None:
    """Heatmap: where each blur setting beats/fails token and how much memory it saves."""
    REGIME_TEX.parent.mkdir(parents=True, exist_ok=True)
    cell_w = 1.55
    cell_h = 0.86
    left = 2.0
    top = 0.0
    lines = [
        "\\documentclass[border=4pt]{standalone}",
        "\\usepackage{tikz}",
        "\\definecolor{winstrong}{HTML}{4B9E8A}",
        "\\definecolor{winsoft}{HTML}{BFE3D8}",
        "\\definecolor{neutralcell}{HTML}{F3E8B6}",
        "\\definecolor{costsoft}{HTML}{E8C08A}",
        "\\definecolor{coststrong}{HTML}{D18A7A}",
        "\\begin{document}",
        "\\begin{tikzpicture}[font=\\sffamily\\small]",
        "\\node[anchor=west,font=\\sffamily\\bfseries] at (0,0.85) {Exact-dense matrix map: quality and memory relative to token};",
        "\\node[anchor=west,font=\\scriptsize] at (0,0.35) {Cell text: $\\Delta$PPL vs token on top, GiB saved vs token below. Gray = censored/OOM or no token control.};",
    ]
    for j, name in enumerate(SET_ROWS):
        x = left + j * cell_w + cell_w / 2
        lines.append(f"\\node[font=\\bfseries\\scriptsize] at ({x:.2f},{top-0.20:.2f}) {{{ROW_LABEL[name]}}};")
    for i, (L, B) in enumerate(REGIME_ISLANDS):
        y = top - 0.75 - i * cell_h
        token = cells.get((L, B, "token"))
        lines.append(f"\\node[anchor=east,font=\\scriptsize] at ({left-0.18:.2f},{y-cell_h/2:.2f}) {{$L{L}$/B{B}}};")
        for j, name in enumerate(SET_ROWS):
            x = left + j * cell_w
            row = cells.get((L, B, name))
            delta = None
            saved = None
            if token is not None and row is not None:
                delta = float(row["ppl_mean"]) - float(token["ppl_mean"])
                saved = (float(token["vram_mean"]) - float(row["vram_mean"])) / 1024.0
            color = _color_for_delta(delta)
            lines.append(f"\\filldraw[fill={color},draw=white,line width=0.6pt] ({x:.2f},{y-cell_h:.2f}) rectangle ({x+cell_w:.2f},{y:.2f});")
            if row is None:
                text = "OOM"
                lines.append(f"\\node[font=\\scriptsize,text=gray!65!black] at ({x+cell_w/2:.2f},{y-cell_h/2:.2f}) {{{text}}};")
            elif token is None:
                ppl = float(row["ppl_mean"])
                gib = float(row["vram_mean"]) / 1024.0
                lines.append(f"\\node[align=center,font=\\tiny,text=gray!75!black] at ({x+cell_w/2:.2f},{y-cell_h/2:.2f}) {{PPL {ppl:.0f}\\\\{gib:.1f} GiB}};")
            else:
                lines.append(f"\\node[align=center,font=\\tiny] at ({x+cell_w/2:.2f},{y-cell_h/2:.2f}) {{{_fmt_signed(delta)} PPL\\\\{saved:+.1f} GiB}};")
    lx = left + len(SET_ROWS) * cell_w + 0.40
    ly = top - 0.95
    legend = [("winstrong", "$\\Delta$PPL $\\le -40$"), ("winsoft", "$-40..-15$"), ("neutralcell", "$\\pm 15$"), ("costsoft", "$15..80$"), ("coststrong", "$>80$")]
    for k, (color, label) in enumerate(legend):
        yy = ly - k * 0.42
        lines.append(f"\\filldraw[fill={color},draw=gray!60] ({lx:.2f},{yy:.2f}) rectangle ({lx+0.35:.2f},{yy+0.25:.2f});")
        lines.append(f"\\node[anchor=west,font=\\scriptsize] at ({lx+0.45:.2f},{yy+0.12:.2f}) {{{label}}};")
    lines.extend(["\\end{tikzpicture}", "\\end{document}", ""])
    REGIME_TEX.write_text("\n".join(lines), encoding="utf-8")


def write_mechanism_allocation(cells: dict[tuple[int, int, str], dict[str, str]], seed_rows: dict[tuple[int, int, str], list[dict[str, str]]]) -> None:
    """Link normalized allocation frontiers with route-removal line diagnostics."""
    MECHANISM_TEX.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 7.10
    panel_h = 3.42
    x_gap = 1.15
    y_gap = 1.45
    top_y = -0.36
    bottom_y = -(panel_h + y_gap)
    color_by_L = {
        512: "Lfive",
        1024: "Lone",
        2048: "Ltwo",
        3584: "Lthree",
        4096: "Lfour",
    }
    dash_by_B = {
        3: "dash pattern=on 3.6pt off 2.0pt",
        4: "solid",
        16: "dash pattern=on 0.8pt off 1.6pt",
    }
    lines = [
        "\\documentclass[border=4pt]{standalone}",
        "\\usepackage{tikz}",
        "\\usetikzlibrary{plotmarks}",
        "\\definecolor{Lfive}{HTML}{6B7280}",
        "\\definecolor{Lone}{HTML}{3B6EA8}",
        "\\definecolor{Ltwo}{HTML}{4B9E8A}",
        "\\definecolor{Lthree}{HTML}{C58A2B}",
        "\\definecolor{Lfour}{HTML}{B65A4A}",
        "\\begin{document}",
        "\\begin{tikzpicture}[font=\\sffamily\\small]",
        f"\\node[anchor=west,font=\\scriptsize,text=gray!70!black] at (0,{panel_h+0.38:.2f}) {{Top: $x=\\mathrm{{GiB}}_{{token}}-\\mathrm{{GiB}}_{{set}}$, $y=\\mathrm{{PPL}}_{{set}}-\\mathrm{{PPL}}_{{token}}$; lower/right is better. Bottom: route-removal $\\Delta\\mathrm{{PPL}}=\\mathrm{{PPL}}_{{removed}}-\\mathrm{{PPL}}_{{original}}$.}};",
    ]

    def draw_axes(x0: float, y0: float, w: float, h: float, title: str, xlabel: str, ylabel: str) -> None:
        lines.append(f"\\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]")
        lines.append(f"\\draw[gray!50,line width=0.35pt] (0,0) rectangle ({w:.2f},{h:.2f});")
        lines.append(f"\\node[anchor=west,font=\\bfseries\\small] at (0.05,{h+0.22:.2f}) {{{title}}};")
        lines.append(f"\\draw[->] (0,0) -- ({w+0.16:.2f},0);")
        if xlabel:
            lines.append(f"\\node[anchor=east,font=\\scriptsize] at ({w:.2f},-0.62) {{{xlabel}}};")
        lines.append(f"\\draw[->] (0,0) -- (0,{h+0.16:.2f});")
        lines.append(f"\\node[rotate=90,anchor=center,font=\\scriptsize] at (-0.68,{h/2:.2f}) {{{ylabel}}};")

    def draw_frontier_panel(idx: int, B: int, title: str, note: str = "") -> None:
        x0 = idx * (panel_w + x_gap)
        y0 = top_y
        frontier_rows = SET_ROWS
        points: list[tuple[int, str, float, float, float]] = []
        for L in (512, 1024, 2048, 3584, 4096):
            token = cells.get((L, B, "token"))
            if token is None:
                continue
            token_ppl = float(token["ppl_mean"])
            token_vram = float(token["vram_mean"])
            for name in frontier_rows:
                row = cells.get((L, B, name))
                if row is None or row.get("ppl_mean", "") == "" or row.get("vram_mean", "") == "":
                    continue
                x_value = (token_vram - float(row["vram_mean"])) / 1024.0
                y_value = float(row["ppl_mean"]) - token_ppl
                points.append((L, name, x_value, y_value, _cell_ci(row)))
        if not points:
            return
        xmin, xmax, ymin, ymax = _panel_bounds([(x, y) for *_rest, x, y, _sd in points])
        xmin = min(xmin, -0.75)
        xmax = max(xmax, 2.75)
        ymin = min(ymin, -125.0)
        ymax = max(ymax, 540.0)
        draw_axes(x0, y0, panel_w, panel_h, title, "$\\Delta$GiB vs token", "$\\Delta$PPL vs token")
        for frac in (0.0, 0.5, 1.0):
            xt = frac * panel_w
            xv = xmin + frac * (xmax - xmin)
            yt = frac * panel_h
            yv = ymin + frac * (ymax - ymin)
            lines.append(f"\\draw[gray!30] ({xt:.2f},0) -- ({xt:.2f},-0.05) node[below,font=\\scriptsize,text=gray!65!black] {{{xv:.1f}}};")
            lines.append(f"\\draw[gray!30] (0,{yt:.2f}) -- (-0.05,{yt:.2f}) node[left,font=\\scriptsize,text=gray!65!black] {{{yv:.0f}}};")
        if xmin < 0 < xmax:
            zx = _scale(0.0, xmin, xmax, panel_w)
            lines.append(f"\\draw[gray!50,dashed] ({zx:.2f},0) -- ({zx:.2f},{panel_h:.2f});")
        if ymin < 0 < ymax:
            zy = _scale(0.0, ymin, ymax, panel_h)
            lines.append(f"\\draw[gray!50,dashed] (0,{zy:.2f}) -- ({panel_w:.2f},{zy:.2f});")
        for L in (512, 1024, 2048, 3584, 4096):
            token = cells.get((L, B, "token"))
            if token is None:
                continue
            token_ppl = float(token["ppl_mean"])
            token_vram = float(token["vram_mean"])
            coords: list[str] = []
            for name in frontier_rows:
                row = cells.get((L, B, name))
                if row is None:
                    continue
                px = _scale((token_vram - float(row["vram_mean"])) / 1024.0, xmin, xmax, panel_w)
                py = _scale(float(row["ppl_mean"]) - token_ppl, ymin, ymax, panel_h)
                coords.append(f"({px:.2f},{py:.2f})")
            if len(coords) > 1:
                lines.append(f"\\draw[{color_by_L.get(L,'black')},line width=0.75pt,opacity=0.74] " + " -- ".join(coords) + ";")
        for L, name, x, y, sd in points:
            px = _scale(x, xmin, xmax, panel_w)
            py = _scale(y, ymin, ymax, panel_h)
            ylo = _scale(y - sd, ymin, ymax, panel_h)
            yhi = _scale(y + sd, ymin, ymax, panel_h)
            color = color_by_L.get(L, "black")
            mark = ROW_MARK[name]
            lines.append(f"\\draw[{color},line width=0.42pt,opacity=0.75] ({px:.2f},{ylo:.2f}) -- ({px:.2f},{yhi:.2f});")
            lines.append(f"\\draw[{color},line width=0.36pt,opacity=0.75] ({px-0.05:.2f},{ylo:.2f}) -- ({px+0.05:.2f},{ylo:.2f});")
            lines.append(f"\\draw[{color},line width=0.36pt,opacity=0.75] ({px-0.05:.2f},{yhi:.2f}) -- ({px+0.05:.2f},{yhi:.2f});")
            lines.append(f"\\node[{color}] at ({px:.2f},{py:.2f}) {{\\Large\\pgfuseplotmark{{{mark}}}}};")
        if note:
            lines.append(f"\\node[anchor=west,font=\\scriptsize,text=gray!65!black,align=left,text width={panel_w-0.10:.2f}cm] at (0.04,-0.48) {{{note}}};")
        lines.append("\\end{scope}")

    draw_frontier_panel(0, 4, "B=4: allocation vs token", "L4096/B4 token,b0,b25 are censored.")
    draw_frontier_panel(1, 3, "B=3: allocation vs token")

    def collect_route_series(group: str) -> tuple[list[tuple[int, int, list[tuple[float, float, float]]]], float, float]:
        metric = f"val/span_ablation_{group}_delta_ppl"
        row_names = ["b0", "b25", "b50", "b75"] if group == "fine" else ["b25", "b50", "b75", "b100"]
        series: list[tuple[int, int, list[tuple[float, float, float]]]] = []
        values: list[float] = []
        for L, B in DIAG_ALL_ISLANDS:
            if B == 16:
                continue
            pts: list[tuple[float, float, float]] = []
            for name in row_names:
                runs = seed_rows.get((L, B, name), [])
                stat = _mean_ci(runs, metric, scale=1.0 / 1000.0) if runs else None
                if stat is None:
                    continue
                mean, ci, _n = stat
                pts.append((BLUR_VALUE[name], mean, ci))
                values.extend([mean - ci, mean + ci])
            if pts:
                series.append((L, B, pts))
        ymin, ymax = (min(values), max(values)) if values else (0.0, 1.0)
        pad = max(0.10 * (ymax - ymin), 0.25)
        return series, min(0.0, ymin - pad), ymax + pad

    def draw_route_panel(idx: int, group: str, color: str) -> None:
        x0 = idx * (panel_w + x_gap)
        y0 = bottom_y
        series, ymin, ymax = collect_route_series(group)
        draw_axes(x0, y0, panel_w, panel_h, f"{group.capitalize()} contribution removed", "", "$\\Delta$PPL$_{remove}$ / 1000")
        if ymin < 0 < ymax:
            zy = _scale(0.0, ymin, ymax, panel_h)
            lines.append(f"\\draw[gray!50,dashed] (0,{zy:.2f}) -- ({panel_w:.2f},{zy:.2f});")
        for xtick in (0, 25, 50, 75, 100):
            x = _scale(xtick, -5.0, 105.0, panel_w)
            lines.append(f"\\draw[gray!30] ({x:.2f},0) -- ({x:.2f},-0.05) node[below,font=\\scriptsize,text=gray!65!black] {{{xtick}}};")
        lines.append(f"\\node[anchor=center,font=\\scriptsize] at ({panel_w/2:.2f},-0.48) {{\\% coarse heads}};")
        for frac in (0.0, 0.5, 1.0):
            yt = frac * panel_h
            yv = ymin + frac * (ymax - ymin)
            lines.append(f"\\draw[gray!30] (0,{yt:.2f}) -- (-0.05,{yt:.2f}) node[left,font=\\scriptsize,text=gray!65!black] {{{yv:.1f}}};")
        for L, B, pts in series:
            coords: list[str] = []
            path_color = color_by_L.get(L, color)
            dash = dash_by_B.get(B, "solid")
            for blur, mean, ci in pts:
                px = _scale(blur, -5.0, 105.0, panel_w)
                py = _scale(mean, ymin, ymax, panel_h)
                ylo = _scale(mean - ci, ymin, ymax, panel_h)
                yhi = _scale(mean + ci, ymin, ymax, panel_h)
                coords.append(f"({px:.2f},{py:.2f})")
                lines.append(f"\\draw[{path_color},line width=0.42pt,opacity=0.76] ({px:.2f},{ylo:.2f}) -- ({px:.2f},{yhi:.2f});")
                lines.append(f"\\draw[{path_color},line width=0.36pt,opacity=0.76] ({px-0.05:.2f},{ylo:.2f}) -- ({px+0.05:.2f},{ylo:.2f});")
                lines.append(f"\\draw[{path_color},line width=0.36pt,opacity=0.76] ({px-0.05:.2f},{yhi:.2f}) -- ({px+0.05:.2f},{yhi:.2f});")
                lines.append(f"\\filldraw[fill={path_color},draw=white,line width=0.25pt,opacity=0.92] ({px:.2f},{py:.2f}) circle (1.55pt);")
            if len(coords) > 1:
                lines.append(f"\\draw[{path_color},{dash},line width=0.88pt,opacity=0.82] " + " -- ".join(coords) + ";")
        lines.append(f"\\node[anchor=west,font=\\scriptsize,text={color}] at (0.12,{panel_h-0.30:.2f}) {{{group} route}};")
        lines.append("\\end{scope}")

    draw_route_panel(0, "fine", "blue!70")
    draw_route_panel(1, "coarse", "orange!85")

    legend_y = bottom_y - 0.72
    lx = 0.0
    for L in (512, 1024, 2048, 3584, 4096):
        lines.append(f"\\draw[{color_by_L[L]},line width=0.95pt] ({lx:.2f},{legend_y:.2f}) -- ({lx+0.38:.2f},{legend_y:.2f});")
        lines.append(f"\\node[anchor=west,font=\\scriptsize] at ({lx+0.45:.2f},{legend_y:.2f}) {{$L={L}$}};")
        lx += 1.42
    lines.append(f"\\draw[black,dash pattern=on 3.6pt off 2.0pt,line width=0.85pt] ({lx+0.28:.2f},{legend_y:.2f}) -- ({lx+0.75:.2f},{legend_y:.2f});")
    lines.append(f"\\node[anchor=west,font=\\scriptsize] at ({lx+0.84:.2f},{legend_y:.2f}) {{B3}};")
    lines.append(f"\\draw[black,line width=0.85pt] ({lx+1.55:.2f},{legend_y:.2f}) -- ({lx+2.02:.2f},{legend_y:.2f});")
    lines.append(f"\\node[anchor=west,font=\\scriptsize] at ({lx+2.11:.2f},{legend_y:.2f}) {{B4}};")
    row_legend_y = legend_y - 0.36
    lines.append(f"\\node[anchor=west,font=\\scriptsize,text=gray!70!black] at (0.00,{row_legend_y:.2f}) {{top markers: square b0, triangle b25, diamond b50, pentagon b75, circle b100}};")
    lines.extend(["\\end{tikzpicture}", "\\end{document}", ""])
    MECHANISM_TEX.write_text("\n".join(lines), encoding="utf-8")


def _write_bar_figure(
    tex_path: Path,
    title: str,
    panels: list[tuple[str, list[tuple[str, float, str]]]],
    *,
    ylabel: str,
    include_zero: bool = True,
) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 6.25
    panel_h = 3.1
    x_gap = 1.45
    y_gap = 2.05
    lines = [
        "\\documentclass[border=4pt]{standalone}",
        "\\usepackage{tikz}",
        "\\definecolor{fineblue}{HTML}{3B6EA8}",
        "\\definecolor{fineteal}{HTML}{4B9E8A}",
        "\\definecolor{coarseamber}{HTML}{C58A2B}",
        "\\definecolor{coarsevermillion}{HTML}{B65A4A}",
        "\\definecolor{winsoft}{HTML}{BFE3D8}",
        "\\definecolor{costsoft}{HTML}{E8C08A}",
        "\\begin{document}",
        "\\begin{tikzpicture}[font=\\sffamily\\small]",
        f"\\node[anchor=west,font=\\sffamily\\bfseries] at (0,{panel_h + 0.85:.2f}) {{{title}}};",
    ]
    for idx, (panel_title, points) in enumerate(panels):
        col = idx % 2
        row = idx // 2
        x0 = col * (panel_w + x_gap)
        y0 = -row * (panel_h + y_gap)
        values = [value for _, value, _ in points]
        ymin = min(0.0, min(values)) if include_zero else min(values)
        ymax = max(0.0, max(values)) if include_zero else max(values)
        pad = max(0.05 * (ymax - ymin), 0.1)
        ymin -= pad
        ymax += pad
        lines.append(f"\\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]")
        lines.append(f"\\draw[gray!45] (0,0) rectangle ({panel_w:.2f},{panel_h:.2f});")
        lines.append(f"\\node[anchor=west,font=\\bfseries\\scriptsize] at (0.05,{panel_h+0.20:.2f}) {{{panel_title}}};")
        lines.append(f"\\draw[->] (0,0) -- (0,{panel_h+0.18:.2f}) node[above,font=\\scriptsize] {{{ylabel}}};")
        if ymin < 0 < ymax:
            zy = _scale(0.0, ymin, ymax, panel_h)
            lines.append(f"\\draw[gray!55] (0,{zy:.2f}) -- ({panel_w:.2f},{zy:.2f});")
        for tick, label in _axis_ticks(ymin, ymax, 3):
            yt = _scale(tick, ymin, ymax, panel_h)
            lines.append(f"\\draw[gray!30] (0,{yt:.2f}) -- (-0.05,{yt:.2f}) node[left,font=\\scriptsize,text=gray!65!black] {{{label}}};")
        step = panel_w / max(1, len(points))
        zero_y = _scale(0.0, ymin, ymax, panel_h)
        for i, (label, value, color) in enumerate(points):
            cx = step * (i + 0.5)
            yv = _scale(value, ymin, ymax, panel_h)
            if include_zero:
                lines.append(f"\\draw[{color},line width=0.58pt,opacity=0.70] ({cx:.2f},{zero_y:.2f}) -- ({cx:.2f},{yv:.2f});")
            else:
                lines.append(f"\\draw[gray!22,densely dotted] ({cx:.2f},0) -- ({cx:.2f},{panel_h:.2f});")
            lines.append(f"\\filldraw[fill={color},draw=white,line width=0.35pt] ({cx:.2f},{yv:.2f}) circle (2.25pt);")
            lines.append(f"\\node[rotate=45,anchor=east,font=\\tiny] at ({cx:.2f},-0.12) {{{label}}};")
        lines.append("\\end{scope}")
    lines.extend(["\\end{tikzpicture}", "\\end{document}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def write_routing_diagnostics(cells: dict[tuple[int, int, str], dict[str, str]], seed_rows: dict[tuple[int, int, str], list[dict[str, str]]]) -> None:
    ROUTING_TEX.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 3.18
    panel_h = 1.88
    x_gap = 0.50
    y_gap = 0.92
    metrics = [
        ("val/effective_range", "effective range", "{:.1f}", True),
        ("val/routing_entropy", "routing entropy", "{:.2f}", False),
        ("val/routing_top1", "top-1 probability", "{:.2f}", False),
    ]
    columns = [512, 1024, 2048, 3584, 4096]
    blur_ticks = [("b0", 0), ("b25", 25), ("b50", 50), ("b75", 75), ("b100", 100)]
    group_blurs = {"fine": ["b0", "b25", "b50", "b75"], "coarse": ["b25", "b50", "b75", "b100"]}
    batch_styles = {
        4: {"offset": 0.0, "dash": "solid", "shape": "square", "opacity": 0.55, "width": 0.55},
        3: {"offset": -6.5, "dash": "dash pattern=on 3.6pt off 2.0pt", "shape": "circle", "opacity": 0.95, "width": 0.70},
        16: {"offset": 6.5, "dash": "dash pattern=on 0.8pt off 1.6pt", "shape": "triangle", "opacity": 0.95, "width": 0.70},
    }
    batch_order = [4, 3, 16]
    colors = {"fine": "blue!72", "coarse": "orange!88"}
    x_min, x_max = -10.0, 110.0

    def y_map(value: float, ymin: float, ymax: float, is_log: bool) -> float:
        if is_log:
            return (math.log10(max(value, ymin * 1.0001)) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * panel_h
        return (value - ymin) / (ymax - ymin) * panel_h

    def ticks_for(ymin: float, ymax: float, is_log: bool) -> list[float]:
        if is_log:
            candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            return [v for v in candidates if ymin <= v <= ymax]
        return [ymin, 0.5 * (ymin + ymax), ymax]

    def draw_mark(shape: str, x: float, y: float, color: str, *, opacity: float) -> str:
        if shape == "square":
            return f"\\filldraw[fill={color},draw=black,line width=0.20pt,opacity={opacity:.2f}] ({x-0.045:.3f},{y-0.045:.3f}) rectangle ({x+0.045:.3f},{y+0.045:.3f});"
        if shape == "triangle":
            return f"\\filldraw[fill={color},draw=white,line width=0.28pt,opacity={opacity:.2f}] ({x:.3f},{y+0.060:.3f}) -- ({x-0.060:.3f},{y-0.050:.3f}) -- ({x+0.060:.3f},{y-0.050:.3f}) -- cycle;"
        return f"\\filldraw[fill={color},draw=white,line width=0.28pt,opacity={opacity:.2f}] ({x:.3f},{y:.3f}) circle (0.052);"

    lines = [
        "\\documentclass[border=4pt]{standalone}",
        "\\usepackage{tikz}",
        "\\begin{document}",
        "\\begin{tikzpicture}[font=\\sffamily\\small]",
        f"\\node[anchor=west,font=\\sffamily\\bfseries] at (0,{panel_h + 0.98:.2f}) {{Fine/coarse routing diagnostics over blur within fixed islands}};",
        f"\\node[anchor=west,font=\\scriptsize,text=gray!65!black] at (0,{panel_h + 0.58:.2f}) {{Batch is a sub-position around each blur tick: B3 left, B4 center, B16 right. Lines connect only fixed $(L,B)$ blur paths.}};",
    ]
    for c_idx, L in enumerate(columns):
        x0 = c_idx * (panel_w + x_gap)
        lines.append(f"\\node[font=\\sffamily\\bfseries\\scriptsize] at ({x0+panel_w/2:.2f},{panel_h+0.20:.2f}) {{$L={L}$}};")

    for r_idx, (metric, title, tick_fmt, is_log) in enumerate(metrics):
        values: list[float] = []
        for (L, _B, row_name), runs in seed_rows.items():
            if L not in columns or row_name not in SET_ROWS:
                continue
            for group in ("fine", "coarse"):
                stat = _mean_ci(runs, f"{metric}_{group}")
                if stat is not None:
                    mean, ci, _ = stat
                    values.extend([mean - ci, mean + ci])
        if values:
            ymin, ymax = min(values), max(values)
        else:
            ymin, ymax = 0.0, 1.0
        if is_log:
            ymin = max(1e-6, ymin / 1.12)
            ymax *= 1.12
        else:
            pad = max(0.18 * (ymax - ymin), ymax * 0.018)
            ymin = max(0.0, ymin - pad)
            ymax += pad
        row_label = "range (log)" if is_log else "entropy" if "entropy" in title else "top-1"
        y_label = -r_idx * (panel_h + y_gap) + panel_h - 0.16
        lines.append(f"\\node[anchor=east,font=\\bfseries\\scriptsize] at (-0.52,{y_label:.2f}) {{{row_label}}};")

        for c_idx, L in enumerate(columns):
            x0 = c_idx * (panel_w + x_gap)
            y0 = -r_idx * (panel_h + y_gap)
            lines.append(f"\\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]")
            lines.append(f"\\draw[gray!45] (0,0) rectangle ({panel_w:.2f},{panel_h:.2f});")
            for tick in ticks_for(ymin, ymax, is_log):
                yt = y_map(tick, ymin, ymax, is_log)
                lines.append(f"\\draw[gray!35] (0,{yt:.2f}) -- ({panel_w:.2f},{yt:.2f});")
                lines.append(f"\\node[anchor=east,font=\\tiny] at (-0.07,{yt:.2f}) {{{tick_fmt.format(tick)}}};")
            for row_name, blur in blur_ticks:
                x = _scale(blur, x_min, x_max, panel_w)
                lines.append(f"\\draw[gray!28,densely dotted] ({x:.2f},0) -- ({x:.2f},{panel_h:.2f});")
                if r_idx == len(metrics) - 1:
                    lines.append(f"\\node[anchor=north,font=\\tiny] at ({x:.2f},-0.07) {{{row_name}}};")
            marker_lines: list[str] = []
            available_batches = sorted({B for (LL, B, row_name) in seed_rows if LL == L and row_name in SET_ROWS})
            for B in batch_order:
                if B not in available_batches:
                    continue
                style = batch_styles[B]
                for group in ("fine", "coarse"):
                    pts: list[tuple[float, float, float, float]] = []
                    for row_name in group_blurs[group]:
                        runs = seed_rows.get((L, B, row_name), [])
                        if not runs:
                            continue
                        stat = _mean_ci(runs, f"{metric}_{group}")
                        if stat is None:
                            continue
                        mean, ci, _ = stat
                        px = _scale(BLUR_VALUE[row_name] + style["offset"], x_min, x_max, panel_w)
                        py = y_map(mean, ymin, ymax, is_log)
                        ylo = y_map(mean - ci, ymin, ymax, is_log)
                        yhi = y_map(mean + ci, ymin, ymax, is_log)
                        pts.append((px, py, ylo, yhi))
                    if not pts:
                        continue
                    color = colors[group]
                    if len(pts) > 1:
                        coords = " -- ".join(f"({px:.2f},{py:.2f})" for px, py, _ylo, _yhi in pts)
                        lines.append(
                            f"\\draw[{color},{style['dash']},line width={style['width']:.2f}pt,opacity={style['opacity']:.2f}] {coords};"
                        )
                    for px, py, ylo, yhi in pts:
                        lines.append(f"\\draw[{color},line width=0.36pt,opacity=0.85] ({px:.2f},{ylo:.2f}) -- ({px:.2f},{yhi:.2f});")
                        marker_lines.append(draw_mark(style["shape"], px, py, color, opacity=float(style["opacity"])))
            lines.extend(marker_lines)
            if L == 1024 and available_batches == [4]:
                lines.append(f"\\node[anchor=east,font=\\tiny,text=gray!65!black,fill=white,inner sep=0.5pt] at ({panel_w-0.05:.2f},{panel_h-0.14:.2f}) {{single batch}};")
            if L == 4096:
                lines.append(f"\\node[anchor=east,font=\\tiny,text=gray!65!black,align=right,fill=white,inner sep=0.5pt] at ({panel_w-0.05:.2f},{panel_h-0.14:.2f}) {{B4 lacks\\\\b0/b25}};")
            if r_idx == len(metrics) - 1:
                lines.append(f"\\node[anchor=north,font=\\tiny,text=gray!70!black] at ({panel_w/2:.2f},-0.38) {{coarse-head fraction}};")
            lines.append("\\end{scope}")
    legend_y = -len(metrics) * (panel_h + y_gap) + 0.20
    lines.append(f"\\fill[blue!72] (0,{legend_y:.2f}) rectangle (0.18,{legend_y+0.12:.2f});")
    lines.append(f"\\node[anchor=west,font=\\scriptsize] at (0.25,{legend_y+0.06:.2f}) {{fine w2/s1}};")
    lines.append(f"\\fill[orange!88] (1.55,{legend_y:.2f}) rectangle (1.73,{legend_y+0.12:.2f});")
    lines.append(f"\\node[anchor=west,font=\\scriptsize] at (1.80,{legend_y+0.06:.2f}) {{coarse w4/s2}};")
    lines.append(f"\\draw[black,dash pattern=on 3.6pt off 2.0pt,line width=0.70pt] (3.35,{legend_y+0.06:.2f}) -- (3.85,{legend_y+0.06:.2f});")
    lines.append(f"\\filldraw[fill=black,draw=white,line width=0.28pt] (3.60,{legend_y+0.06:.2f}) circle (0.052);")
    lines.append(f"\\node[anchor=west,font=\\scriptsize] at (3.98,{legend_y+0.06:.2f}) {{B3}};")
    lines.append(f"\\draw[black,line width=0.55pt,opacity=0.60] (4.65,{legend_y+0.06:.2f}) -- (5.15,{legend_y+0.06:.2f});")
    lines.append(f"\\filldraw[fill=black,draw=black,line width=0.20pt,opacity=0.60] (4.87,{legend_y+0.015:.2f}) rectangle (4.96,{legend_y+0.105:.2f});")
    lines.append(f"\\node[anchor=west,font=\\scriptsize] at (5.27,{legend_y+0.06:.2f}) {{B4}};")
    lines.append(f"\\draw[black,dash pattern=on 0.8pt off 1.6pt,line width=0.70pt] (5.95,{legend_y+0.06:.2f}) -- (6.45,{legend_y+0.06:.2f});")
    lines.append(f"\\filldraw[fill=black,draw=white,line width=0.28pt] (6.20,{legend_y+0.12:.2f}) -- (6.14,{legend_y+0.01:.2f}) -- (6.26,{legend_y+0.01:.2f}) -- cycle;")
    lines.append(f"\\node[anchor=west,font=\\scriptsize] at (6.58,{legend_y+0.06:.2f}) {{B16}};")
    lines.append(f"\\node[anchor=west,align=left,font=\\scriptsize,text width=17.3cm] at (0,{legend_y-0.42:.2f}) {{Five-seed means with 95\\% t-intervals. Effective range uses log y; entropy/top-1 use zoomed linear y. Fine traces run b0$\\to$b75, coarse traces run b25$\\to$b100; endpoints appear only when that group exists.}};")
    lines.extend(["\\end{tikzpicture}", "\\end{document}", ""])
    ROUTING_TEX.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_diagnostics(cells: dict[tuple[int, int, str], dict[str, str]]) -> None:
    fine_points: list[tuple[str, float, str]] = []
    coarse_points: list[tuple[str, float, str]] = []
    for L, B in DIAG_ISLANDS:
        island = f"{L/1024:.1f}k/B{B}".replace(".0", "")
        for name in DIAG_ROWS:
            row = cells[(L, B, name)]
            fine = _num(row, "span_fine_mean")
            coarse = _num(row, "span_coarse_mean")
            if fine is not None:
                fine_points.append((f"{island} {name}", fine / 1000.0, "fineblue"))
            if coarse is not None:
                coarse_points.append((f"{island} {name}", coarse / 1000.0, "coarseamber"))
    _write_bar_figure(
        ABLATION_TEX,
        "Group span-ablation diagnostics",
        [("Fine ablation", fine_points), ("Coarse ablation", coarse_points)],
        ylabel="$\\Delta$PPL / 1000",
    )


def write_bucket_diagnostics(cells: dict[tuple[int, int, str], dict[str, str]]) -> None:
    panels: list[tuple[str, list[tuple[str, float, str]]]] = []
    for L, B in DIAG_ISLANDS:
        for name in ["token", "b25", "b50", "b100"]:
            row = cells[(L, B, name)]
            points = [
                ("early freq", float(row["early_freq"]), "fineteal"),
                ("late freq", float(row["late_freq"]), "winsoft"),
                ("early rare", float(row["early_rare"]), "coarsevermillion"),
                ("late rare", float(row["late_rare"]), "costsoft"),
            ]
            panels.append((f"L{L}/B{B} {ROW_LABEL[name]}", points))
    _write_bar_figure(BUCKET_TEX, "Token-type validation loss buckets", panels, ylabel="loss", include_zero=False)


def main() -> None:
    cells = load_cells()
    seed_rows = load_seed_rows()
    cells = summarize_seed_cells(seed_rows, cells)
    write_cells_tsv(cells)
    write_final_matrix_table(cells)
    write_compact_table(cells)
    write_frontier_plot(cells)
    write_blur_path_frontier(cells)
    write_operating_regime_map(cells)
    write_mechanism_allocation(cells, seed_rows)
    write_routing_diagnostics(cells, seed_rows)
    write_ablation_diagnostics(cells)
    write_bucket_diagnostics(cells)
    print(PLOT_TEX)
    print(FINAL_MATRIX_OUT)
    print(BLUR_PATH_TEX)
    print(REGIME_TEX)
    print(MECHANISM_TEX)
    print(ROUTING_TEX)
    print(ABLATION_TEX)
    print(BUCKET_TEX)


if __name__ == "__main__":
    main()
