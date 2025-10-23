import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


def test_vit_cli_runs_synthetic():
    cmd = [
        sys.executable,
        "scripts/train_tiny_vit_banked.py",
        "--data-mode",
        "synthetic",
        "--demo-samples",
        "32",
        "--limit",
        "32",
        "--epochs",
        "1",
        "--batch",
        "4",
        "--device",
        "cpu",
        "--patch",
        "4",
        "--window",
        "4",
        "--stride",
        "2",
        "--minhash-k",
        "8",
        "--router-topk",
        "0",
        "--adapter-rank",
        "0",
        "--num-workers",
        "0",
        "--seed",
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
        raise AssertionError(f"ViT CLI failed ({result.returncode}):\n{result.stdout}")
