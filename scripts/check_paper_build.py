#!/usr/bin/env python3
import re
import sys
from pathlib import Path

log = Path("out/final_paper_bundle/checks/compile_logs/latexmk.stdout.txt")
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
]

fatal_hits = []
for p in fatal_patterns:
    fatal_hits.extend(re.findall(p, text, flags=re.MULTILINE))

warning_hits = []
for p in warning_patterns:
    warning_hits.extend(re.findall(p, text, flags=re.MULTILINE))

print("Fatal issues:", len(fatal_hits))
print("Warnings:", len(warning_hits))

if fatal_hits:
    print("BUILD_STATUS=FAIL")
    sys.exit(1)

print("BUILD_STATUS=OK")
sys.exit(0)
