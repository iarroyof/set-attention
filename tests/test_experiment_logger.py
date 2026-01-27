from pathlib import Path

import torch

from src.train.experiment_logger import ExperimentLogger


def test_experiment_logger_writes_csv(tmp_path: Path):
    cfg = {
        "model": {
            "family": "baseline_token",
            "architecture": "transformer_lm",
            "vocab_size": 100,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_seq_len": 16,
        },
        "data": {"dataset": "wikitext2", "batch_size": 2, "seq_len": 8},
        "training": {"epochs": 1, "lr": 1e-3, "output_dir": str(tmp_path)},
        "logging": {"wandb": {"enable": False}, "csv": {"path": str(tmp_path / "metrics.csv")}},
    }
    logger = ExperimentLogger(config=cfg)
    logger.log_model_complexity(torch.nn.Linear(4, 4))
    logger.start_epoch(num_train_samples=10)
    train_metrics = {"loss": 1.0, "grad_norm": 0.5}
    val_metrics = {"loss": 1.2}
    logger.log_epoch(1, train_metrics, val_metrics, set_diagnostics=None)
    logger.finish()

    content = (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()
    header = content[0].split(",")
    row = content[1].split(",")
    data = dict(zip(header, row))

    assert data["train/loss"] != "NA"
    assert data["val/loss"] != "NA"
    assert data["train/ppl"] != "NA"
    assert data["val/ppl"] != "NA"
    assert data["ausa/active_set_ratio"] == "NA"
