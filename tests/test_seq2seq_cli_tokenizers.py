import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

TOKENIZER_TYPES = ("ausa", "hf_unigram", "hf_bpe")

LANGUAGE_SCRIPTS = (
    (
        "scripts/train_seq2seq_text.py",
        [
            "--attn",
            "ska_tok",
            "--epochs",
            "1",
            "--batch",
            "4",
            "--demo",
            "--demo-samples",
            "8",
            "--device",
            "cpu",
            "--d-model",
            "64",
            "--nhead",
            "2",
            "--layers",
            "1",
            "--gate-topk",
            "4",
            "--token-set-k",
            "16",
        ],
    ),
    (
        "scripts/train_seq2seq_text_banked.py",
        [
            "--epochs",
            "1",
            "--batch",
            "4",
            "--demo",
            "--demo-samples",
            "8",
            "--device",
            "cpu",
            "--atom-dim",
            "64",
            "--heads",
            "2",
            "--layers",
            "1",
            "--minhash-k",
            "16",
            "--window",
            "4",
            "--stride",
            "2",
        ],
    ),
    (
        "scripts/train_toy_seq2seq.py",
        [
            "--attn",
            "ska_tok",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--d-model",
            "64",
            "--nhead",
            "2",
            "--layers",
            "1",
            "--gamma",
            "0.3",
            "--token-set-k",
            "16",
        ],
    ),
)


@pytest.mark.parametrize("script,base_args", LANGUAGE_SCRIPTS, ids=lambda item: Path(item[0]).stem)
@pytest.mark.parametrize("tokenizer_type", TOKENIZER_TYPES)
def test_language_scripts_support_tokenizers(tmp_path, script, base_args, tokenizer_type):
    tok_dir = tmp_path / f"{Path(script).stem}_{tokenizer_type}"
    cmd = [
        sys.executable,
        script,
        "--tokenizer-type",
        tokenizer_type,
        "--tokenizer-dir",
        str(tok_dir),
        "--tokenizer-config",
        "{}",
    ] + base_args
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
        pytest.fail(f"Command failed ({result.returncode}):\n{result.stdout}")
