import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


def test_diffusion_cli_runs():
    cmd = [
        sys.executable,
        "scripts/train_toy_diffusion_banked.py",
        "--config",
        "",
        "--epochs",
        "1",
        "--steps",
        "10",
        "--device",
        "cpu",
        "--d-model",
        "32",
        "--nhead",
        "2",
        "--layers",
        "1",
        "--window",
        "4",
        "--stride",
        "2",
        "--minhash-k",
        "8",
        "--samples",
        "16",
        "--data-seq-len",
        "8",
        "--data-dim",
        "4",
        "--data-batch-size",
        "8",
        "--data-val-frac",
        "0.25",
        "--data-modes",
        "2",
        "--data-seed",
        "0",
        "--router-topk",
        "0",
        "--adapter-rank",
        "0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"Diffusion CLI failed ({result.returncode}):\n{result.stdout}")
