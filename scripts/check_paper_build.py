#!/usr/bin/env python3
import re
import sys
from pathlib import Path

log_dir = Path("out/final_paper_bundle/checks/compile_logs")
latest_run_file = log_dir / "latest_run_dir.txt"
latest_main_file = log_dir / "latest_main_tex.txt"
stdout_log = log_dir / "latexmk.stdout.txt"
main_tex = "example_paper.tex"

if latest_main_file.exists():
    main_tex = latest_main_file.read_text().strip() or main_tex

log = stdout_log
if latest_run_file.exists():
    latest_run_dir = Path(latest_run_file.read_text().strip())
    final_log = latest_run_dir / f"{Path(main_tex).stem}.log"
    if final_log.exists():
        log = final_log

if not log.exists():
    print("FAIL: compile log not found:", log)
    sys.exit(1)

text = log.read_text(errors="ignore")

fatal_patterns = [
    r"^!",
    r"LaTeX Error",
    r"Undefined control sequence",
    r"Emergency stop",
    r"Fatal error occurred",
]
warning_patterns = [
    r"Citation .* undefined",
    r"Reference .* undefined",
    r"There were undefined references",
    r"Overfull \\hbox",
    r"Underfull \\hbox",
    r"Underfull \\vbox",
    r"Warning",
]

fatal_hits = []
for p in fatal_patterns:
    fatal_hits.extend(re.findall(p, text, flags=re.MULTILINE))

warning_hits = []
for p in warning_patterns:
    warning_hits.extend(re.findall(p, text, flags=re.MULTILINE))

print("Log:", log)
print("Fatal issues:", len(fatal_hits))
print("Warnings:", len(warning_hits))

if fatal_hits:
    print("BUILD_STATUS=FAIL")
    sys.exit(1)

print("BUILD_STATUS=OK")
sys.exit(0)
