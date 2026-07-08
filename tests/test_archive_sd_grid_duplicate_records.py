from __future__ import annotations

import csv
from pathlib import Path

from scripts.archive_sd_grid_duplicate_records import clean_registry


FIELDS = ["ts", "host", "cell_id", "gpu", "exit", "epochs", "csv"]


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_registry_cleanup_archives_failed_retries_and_keeps_success(tmp_path: Path) -> None:
    registry = tmp_path / "grid_runs_blue.tsv"
    _write(
        registry,
        [
            {"ts": "1", "host": "blue", "cell_id": "a", "gpu": "0", "exit": "1", "epochs": "0", "csv": "a.csv"},
            {"ts": "2", "host": "blue", "cell_id": "a", "gpu": "0", "exit": "0", "epochs": "10", "csv": "a.csv"},
            {"ts": "3", "host": "blue", "cell_id": "b", "gpu": "1", "exit": "1", "epochs": "0", "csv": "b.csv"},
            {"ts": "4", "host": "blue", "cell_id": "b", "gpu": "1", "exit": "1", "epochs": "0", "csv": "b.csv"},
        ],
    )
    manifest = clean_registry(registry, tmp_path / "archive")
    rows = list(csv.DictReader(registry.open(), delimiter="\t"))
    assert [(row["cell_id"], row["exit"]) for row in rows] == [("a", "0"), ("b", "1")]
    assert manifest["archived_rows"] == 2
    assert manifest["duplicate_success_records"] == 0
