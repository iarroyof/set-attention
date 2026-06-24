#!/usr/bin/env python3
"""Build the validated set-dictionary capacity-ladder figure for the paper."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "final_paper_bundle" / "plots" / "main"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / "fig_sd_capacity_ladder.tex"
    tex_path.write_text(
        r"""\documentclass[tikz,border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  width=6.7in,
  height=2.85in,
  ymin=720,
  ymax=1585,
  xmin=-0.2,
  xmax=3.25,
  ylabel={Validation PPL},
  xtick={0,1,2,3},
  xticklabels={endpoint CE, trained anchor, all\_past fiber, all\_past wide atoms},
  x tick label style={font=\scriptsize, align=center, text width=0.8in},
  ymajorgrids=true,
  grid style={draw=gray!20},
  axis line style={gray!55},
  tick style={gray!55},
  legend style={draw=none, fill=none, font=\scriptsize, at={(0.98,0.98)}, anchor=north east},
  every axis plot/.append style={line width=1.2pt},
]
\addplot+[mark=o, color=blue!65!black] coordinates {
  (0,1510.899740)
  (1,1510.373413)
  (2,1363.931559)
};
\addlegendentry{$(w,s)=(16,8)$}
\addplot+[mark=square*, color=orange!75!black] coordinates {
  (0,1297.866252)
  (1,1288.973145)
  (2,1288.603190)
  (3,1241.522583)
};
\addlegendentry{$(w,s)=(4,2)$}
\addplot+[black, dashed, mark=none] coordinates {(-0.2,781.109436) (3.25,781.109436)};
\addlegendentry{dense token baseline}
\node[font=\scriptsize, align=left, anchor=west] at (axis cs:0.55,1435) {valid anchor\\$\cos\approx0.30$};
\draw[->, gray!70] (axis cs:0.95,1425) -- (axis cs:1.0,1288.973145);
\node[font=\scriptsize, anchor=west] at (axis cs:2.46,1040) {$+460$ PPL};
\draw[->, gray!70] (axis cs:2.73,1060) -- (axis cs:3.0,1241.522583);
\end{axis}
\end{tikzpicture}
\end{document}
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ],
        cwd=OUT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stdout + "\n" + proc.stderr)


if __name__ == "__main__":
    main()
