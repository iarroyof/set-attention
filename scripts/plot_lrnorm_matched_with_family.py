#!/usr/bin/env python3
"""Draw the LR-normalized matched-grid figure with the reference-size family slice.

The original LR-normalized headline plot labeled points by model size and
learning rate. This script keeps that style while adding the verified
reference-size sparse/linear Set Attention points. It intentionally avoids
matplotlib/PIL so the figure can be regenerated in minimal WSL/CI environments.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADLINE = ROOT / "out/paper_integrated_evidence/tables/lrnorm_headline_all_runs.tsv"
FAMILY = ROOT / "out/paper_integrated_evidence/tables/lrnorm_d384_family_slice.tsv"
OUT = ROOT / "out/final_paper_bundle/plots/main/fig_lrnorm_matched_with_family_slice.pdf"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Canvas:
    def __init__(self, width: int = 760, height: int = 420) -> None:
        self.w = width
        self.h = height
        self.ops: list[str] = []

    def color(self, r: float, g: float, b: float) -> None:
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {r:.3f} {g:.3f} {b:.3f} RG")

    def line_width(self, width: float) -> None:
        self.ops.append(f"{width:.2f} w")

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.ops.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def rect(self, x: float, y: float, w: float, h: float, fill: bool = False) -> None:
        self.ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {'f' if fill else 'S'}")

    def circle(self, x: float, y: float, r: float, fill: bool = True) -> None:
        # Cubic Bezier circle approximation.
        k = 0.5522847498 * r
        self.ops.append(
            " ".join(
                [
                    f"{x+r:.2f} {y:.2f} m",
                    f"{x+r:.2f} {y+k:.2f} {x+k:.2f} {y+r:.2f} {x:.2f} {y+r:.2f} c",
                    f"{x-k:.2f} {y+r:.2f} {x-r:.2f} {y+k:.2f} {x-r:.2f} {y:.2f} c",
                    f"{x-r:.2f} {y-k:.2f} {x-k:.2f} {y-r:.2f} {x:.2f} {y-r:.2f} c",
                    f"{x+k:.2f} {y-r:.2f} {x+r:.2f} {y-k:.2f} {x+r:.2f} {y:.2f} c",
                    "f" if fill else "S",
                ]
            )
        )

    def diamond(self, x: float, y: float, r: float) -> None:
        self.ops.append(
            f"{x:.2f} {y+r:.2f} m {x+r:.2f} {y:.2f} l {x:.2f} {y-r:.2f} l {x-r:.2f} {y:.2f} l h f"
        )

    def cross(self, x: float, y: float, r: float) -> None:
        self.line(x - r, y - r, x + r, y + r)
        self.line(x - r, y + r, x + r, y - r)

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


def main() -> None:
    headline = read_tsv(HEADLINE)
    family = read_tsv(FAMILY)
    rows: list[tuple[dict[str, str], str]] = []
    rows += [(r, "full") for r in headline]
    for r in family:
        if r["family"] in {"Set Linear", "Set Sparse"}:
            rows.append((r, "family"))

    xvals = [f(r, "time_per_epoch_s") for r, _ in rows]
    yvals = [f(r, "val_ppl") for r, _ in rows]
    xmin, xmax = min(xvals) - 2, max(xvals) + 3
    ymin, ymax = min(yvals) - 40, max(yvals) + 70

    c = Canvas()
    left, right, bottom, top = 78, 32, 64, 54
    pw, ph = c.w - left - right, c.h - bottom - top

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * pw

    def sy(y: float) -> float:
        return bottom + (y - ymin) / (ymax - ymin) * ph

    c.color(0.08, 0.08, 0.08)
    c.line_width(0.9)
    c.line(left, bottom, left + pw, bottom)
    c.line(left, bottom, left, bottom + ph)
    c.text(220, 22, "Time per epoch (s)", 11)
    c.text(8, 218, "Validation perplexity", 11)
    c.text(118, 392, "LR-normalized matched runs with reference-size family slice", 13)

    for tick in [35, 40, 45, 50, 55]:
        x = sx(tick)
        c.color(0.82, 0.82, 0.82)
        c.line(x, bottom, x, bottom + ph)
        c.color(0.08, 0.08, 0.08)
        c.line(x, bottom - 4, x, bottom)
        c.text(x - 8, bottom - 20, str(tick), 9)
    for tick in [500, 700, 900, 1100, 1300]:
        y = sy(tick)
        c.color(0.88, 0.88, 0.88)
        c.line(left, y, left + pw, y)
        c.color(0.08, 0.08, 0.08)
        c.line(left - 4, y, left, y)
        c.text(left - 42, y - 3, str(tick), 9)

    colors = {
        "Baseline": (0.38, 0.38, 0.38),
        "Baseline token": (0.38, 0.38, 0.38),
        "Set Dense": (0.12, 0.32, 0.78),
        "Set Linear": (0.05, 0.55, 0.26),
        "Set Sparse": (0.86, 0.38, 0.05),
    }
    for r, source in rows:
        family_name = r["family"]
        c.color(*colors[family_name])
        x, y = sx(f(r, "time_per_epoch_s")), sy(f(r, "val_ppl"))
        if family_name.startswith("Baseline"):
            c.cross(x, y, 5)
        elif family_name == "Set Dense":
            c.circle(x, y, 4.4 if source == "full" else 5.0)
        elif family_name == "Set Linear":
            c.diamond(x, y, 5.3)
        elif family_name == "Set Sparse":
            c.rect(x - 4.8, y - 4.8, 9.6, 9.6, fill=True)

    label_offsets = {
        ("Baseline", "384", "1536", "1e-4"): (-24, -22),
        ("Baseline", "384", "1536", "2e-4"): (-36, 10),
        ("Baseline", "384", "1536", "3e-4"): (-42, 10),
        ("Set Dense", "384", "1536", "1e-4"): (8, 10),
        ("Set Dense", "384", "1536", "2e-4"): (8, 2),
        ("Set Dense", "384", "1536", "3e-4"): (8, -15),
        ("Baseline", "384", "3072", "1e-4"): (-56, -6),
        ("Baseline", "384", "3072", "2e-4"): (-48, 12),
        ("Baseline", "384", "3072", "3e-4"): (-56, 8),
        ("Set Dense", "384", "3072", "1e-4"): (7, 8),
        ("Set Dense", "384", "3072", "2e-4"): (7, -4),
        ("Set Dense", "384", "3072", "3e-4"): (7, -17),
        ("Baseline", "512", "1024", "1e-4"): (-44, -18),
        ("Baseline", "512", "1024", "2e-4"): (-44, 10),
        ("Baseline", "512", "1024", "3e-4"): (-52, -14),
        ("Set Dense", "512", "1024", "1e-4"): (7, 9),
        ("Set Dense", "512", "1024", "2e-4"): (7, 0),
        ("Set Dense", "512", "1024", "3e-4"): (7, -13),
        ("Baseline", "512", "2048", "1e-4"): (-50, -18),
        ("Baseline", "512", "2048", "2e-4"): (-42, 10),
        ("Baseline", "512", "2048", "3e-4"): (-52, 8),
        ("Set Dense", "512", "2048", "1e-4"): (7, 8),
        ("Set Dense", "512", "2048", "2e-4"): (7, -16),
        ("Set Dense", "512", "2048", "3e-4"): (7, 4),
        ("Set Linear", "384", "1536", "1e-4"): (7, 10),
        ("Set Linear", "384", "1536", "2e-4"): (-88, -4),
        ("Set Linear", "384", "1536", "3e-4"): (-82, -20),
        ("Set Sparse", "384", "1536", "1e-4"): (7, -16),
        ("Set Sparse", "384", "1536", "2e-4"): (-78, -18),
        ("Set Sparse", "384", "1536", "3e-4"): (7, 8),
    }
    short_name = {
        "Baseline": "",
        "Baseline token": "",
        "Set Dense": "",
        "Set Linear": "Lin ",
        "Set Sparse": "Sparse ",
    }
    for r, source in rows:
        family_name = r["family"]
        key = (family_name, r["D"], r["d_ff"], r["lr"])
        dx, dy = label_offsets.get(key, (6, 6))
        x, y = sx(f(r, "time_per_epoch_s")), sy(f(r, "val_ppl"))
        label = f"{short_name[family_name]}{r['D']}/{r['d_ff']}, {r['lr']}"
        c.text_colored(x + dx, y + dy, label, colors[family_name], 7)

    # Legend.
    lx, ly = 500, 344
    c.color(1, 1, 1)
    c.rect(lx - 12, ly - 80, 220, 92, fill=True)
    c.color(0.1, 0.1, 0.1)
    c.rect(lx - 12, ly - 80, 220, 92)
    legend = [
        ("Baseline token, full grid", "Baseline"),
        ("Dense-exact Set Attention, full grid", "Set Dense"),
        ("Linear landmark, D=384 d_ff=1536", "Set Linear"),
        ("Sparse local-band, D=384 d_ff=1536", "Set Sparse"),
    ]
    for i, (label, fam) in enumerate(legend):
        y = ly - i * 20
        c.color(*colors[fam])
        if fam == "Baseline":
            c.cross(lx, y + 4, 5)
        elif fam == "Set Dense":
            c.circle(lx, y + 4, 4.5)
        elif fam == "Set Linear":
            c.diamond(lx, y + 4, 5)
        else:
            c.rect(lx - 4.5, y - 0.5, 9, 9, fill=True)
        c.color(0.08, 0.08, 0.08)
        c.text(lx + 14, y, label, 9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(c.pdf_bytes())
    print(OUT)


if __name__ == "__main__":
    main()
