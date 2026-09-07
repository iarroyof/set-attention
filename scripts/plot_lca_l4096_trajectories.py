#!/usr/bin/env python3
"""Draw the LCA L=4096 validation-trajectory figure: with-dropout vs dropout-free.

Two panels, one shared y axis. Left: the original dropout=0.1 rows
(l4096trajectory grid). Right: the confirmed dropout-free recipe
(l4096nodrop grid). Three seeds per family per panel are summarized by one
seed-mean line with a pointwise 95% Student-t confidence band. The figure makes
the two paper points
visually: (a) the with-dropout endpoint is oscillation phase luck for both
families; (b) under dropout=0 the set row's endpoint variance collapses
(every seed ends at its ceiling) while token still oscillates at update
8000. Follows the repo convention of hand-rolled PDF output (no
matplotlib) so the figure regenerates in minimal environments.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/final_paper_bundle/plots/main/fig_lca_l4096_trajectories.pdf"

PANELS = [
    (
        "with dropout (0.1)",
        ROOT / "out/lca_cmp/l4096trajectory",
        {"b75": "l4096tj_b75full_L4096_seed{d}_evalcurve.csv",
         "token": "l4096tj_token_L4096_seed{d}_evalcurve.csv"},
    ),
    (
        "dropout-free",
        ROOT / "out/lca_cmp/l4096nodrop",
        {"b75": "l4096nd_b75nodrop_L4096_seed{d}_evalcurve.csv",
         "token": "l4096nd_tokennodrop_L4096_seed{d}_evalcurve.csv"},
    ),
]
SEEDS = (0, 1, 2)
T_CRIT_95_DF2 = 4.3026527299

COLORS = {
    "token": (0.45, 0.45, 0.45),
    "b75": (0.12, 0.32, 0.78),
}
LABELS = {"token": "token", "b75": "b75 full routing"}


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Canvas:
    def __init__(self, width: int = 760, height: int = 400) -> None:
        self.w = width
        self.h = height
        self.ops: list[str] = []

    def color(self, r: float, g: float, b: float) -> None:
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {r:.3f} {g:.3f} {b:.3f} RG")

    def line_width(self, width: float) -> None:
        self.ops.append(f"{width:.2f} w")

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.ops.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def polyline(self, pts: list[tuple[float, float]]) -> None:
        if len(pts) < 2:
            return
        parts = [f"{pts[0][0]:.2f} {pts[0][1]:.2f} m"]
        parts += [f"{x:.2f} {y:.2f} l" for x, y in pts[1:]]
        self.ops.append(" ".join(parts) + " S")

    def fill_polygon(self, pts: list[tuple[float, float]]) -> None:
        if len(pts) < 3:
            return
        parts = [f"{pts[0][0]:.2f} {pts[0][1]:.2f} m"]
        parts += [f"{x:.2f} {y:.2f} l" for x, y in pts[1:]]
        self.ops.append(" ".join(parts) + " h f")

    def fill_rect(self, x: float, y: float, width: float, height: float) -> None:
        self.ops.append(f"{x:.2f} {y:.2f} {width:.2f} {height:.2f} re f")

    def text(self, x: float, y: float, text: str, size: int = 10) -> None:
        self.ops.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({esc(text)}) Tj ET")

    def text_colored(self, x: float, y: float, text: str, rgb: tuple[float, float, float], size: int = 10) -> None:
        self.color(*rgb)
        self.text(x, y, text, size)

    def pdf_bytes(self) -> bytes:
        stream = "\n".join(self.ops).encode()
        objs: list[bytes] = []
        objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w} {self.h}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode()
        )
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, obj in enumerate(objs, start=1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode())
            out.extend(obj)
            out.extend(b"\nendobj\n")
        xref = len(out)
        out.extend(f"xref\n0 {len(objs)+1}\n".encode())
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode())
        out.extend(
            f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        )
        return bytes(out)


def read_curve(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [(float(r["update"]), float(r["val_acc"])) for r in rows]


def mean_ci_curve(
    curves: list[list[tuple[float, float]]],
) -> list[tuple[float, float, float, float]]:
    result = []
    for i in range(len(curves[0])):
        values = [curve[i][1] for curve in curves]
        mean = statistics.fmean(values)
        half_width = T_CRIT_95_DF2 * statistics.stdev(values) / len(values) ** 0.5
        result.append((curves[0][i][0], mean, mean - half_width, mean + half_width))
    return result


def main() -> None:
    c = Canvas()
    xmin, xmax = 0.0, 8000.0
    ymin, ymax = 0.55, 1.0
    panel_w, panel_h = 285, 270
    bottoms = 70
    lefts = [60, 415]
    top = bottoms + panel_h

    def sx(x: float, left: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * panel_w

    def sy(y: float) -> float:
        return bottoms + (y - ymin) / (ymax - ymin) * panel_h

    c.text(150, 372, "LCA L=4096 validation trajectories (3 seeds, one host)", 13)

    for (title, grid, patterns), left in zip(PANELS, lefts):
        # axes
        c.color(0.08, 0.08, 0.08)
        c.line_width(0.9)
        c.line(left, bottoms, left + panel_w, bottoms)
        c.line(left, bottoms, left, top)
        c.text(left + 62, bottoms - 36, "training update", 10)
        c.text(left + 40, top + 14, title, 11)

        for tick in (0, 2000, 4000, 6000, 8000):
            x = sx(tick, left)
            c.color(0.86, 0.86, 0.86)
            c.line_width(0.5)
            c.line(x, bottoms, x, top)
            c.color(0.08, 0.08, 0.08)
            c.line(x, bottoms - 4, x, bottoms)
            c.text(x - 10, bottoms - 18, str(tick), 8)
        for tick in (0.6, 0.7, 0.8, 0.9, 1.0):
            y = sy(tick)
            c.color(0.90, 0.90, 0.90)
            c.line_width(0.5)
            c.line(left, y, left + panel_w, y)
            c.color(0.08, 0.08, 0.08)
            c.line(left - 4, y, left, y)
            c.text(left - 34, y - 3, f"{tick:.1f}", 8)

        summaries = {}
        for family in ("token", "b75"):
            curves = [
                read_curve(grid / family / "L4096" / patterns[family].format(d=s))
                for s in SEEDS
            ]
            summaries[family] = mean_ci_curve(curves)

        # Draw both bands first so neither band's fill can hide a mean line.
        for family in ("token", "b75"):
            rgb = COLORS[family]
            summary = summaries[family]
            lower = [(sx(x, left), sy(max(ymin, lo))) for x, _, lo, _ in summary]
            upper = [(sx(x, left), sy(min(ymax, hi))) for x, _, _, hi in summary]
            c.color(
                rgb[0] + (1 - rgb[0]) * 0.82,
                rgb[1] + (1 - rgb[1]) * 0.82,
                rgb[2] + (1 - rgb[2]) * 0.82,
            )
            c.fill_polygon(lower + list(reversed(upper)))

        for family in ("token", "b75"):
            rgb = COLORS[family]
            summary = summaries[family]
            c.color(*rgb)
            c.line_width(2.0)
            c.polyline([(sx(x, left), sy(mean)) for x, mean, _, _ in summary])

        if title == "dropout-free":
            box_x, box_y, box_w, box_h = left + 10, sy(0.565), panel_w - 20, 46
            c.color(1.0, 1.0, 1.0)
            c.fill_rect(box_x, box_y, box_w, box_h)
            c.color(0.25, 0.25, 0.25)
            c.text(box_x + 8, box_y + 32, "peak VRAM after removing dropout", 8)
            c.text(box_x + 8, box_y + 19, "token: 33746 -> 20766 MiB (-38.5%)", 8)
            c.text(box_x + 8, box_y + 7, "b75:   24916 -> 17925 MiB (-28.1%)", 8)

    # legend under the right panel title row
    lx = lefts[1] + 150
    c.color(*COLORS["token"])
    c.line_width(1.8)
    c.line(lx, top + 22, lx + 22, top + 22)
    c.text(lx + 26, top + 19, LABELS["token"], 9)
    c.color(*COLORS["b75"])
    c.line(lx + 70, top + 22, lx + 92, top + 22)
    c.text(lx + 96, top + 19, LABELS["b75"], 9)
    c.color(0.08, 0.08, 0.08)
    c.text(lefts[0] + 4, 22, "line: seed mean; band: pointwise 95% t-CI; validation every 500 updates", 9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(c.pdf_bytes())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
