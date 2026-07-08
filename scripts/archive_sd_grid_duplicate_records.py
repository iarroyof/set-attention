#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import tempfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_index(rows: list[dict[str, str]]) -> int:
    successful = [
        (index, int(row.get("epochs", "0") or 0))
        for index, row in enumerate(rows)
        if row.get("exit") == "0"
    ]
    if successful:
        return max(successful, key=lambda item: (item[1], item[0]))[0]
    return len(rows) - 1


def clean_registry(registry: Path, archive_root: Path) -> dict[str, object]:
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    fieldnames = list(rows[0]) if rows else [
        "ts",
        "host",
        "cell_id",
        "gpu",
        "exit",
        "epochs",
        "csv",
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("cell_id", "")].append(row)

    kept: list[dict[str, str]] = []
    archived: list[dict[str, str]] = []
    duplicate_successes = 0
    for cell_id, cell_rows in grouped.items():
        if not cell_id or len(cell_rows) == 1:
            kept.extend(cell_rows)
            continue
        keep_index = _canonical_index(cell_rows)
        for index, row in enumerate(cell_rows):
            if index == keep_index:
                kept.append(row)
                continue
            archived_row = dict(row)
            archived_row["archive_reason"] = (
                "superseded_duplicate_success"
                if row.get("exit") == "0"
                else "superseded_failed_attempt"
            )
            duplicate_successes += int(row.get("exit") == "0")
            archived.append(archived_row)

    archive_root.mkdir(parents=True, exist_ok=True)
    raw_copy = archive_root / registry.name
    raw_copy.write_bytes(registry.read_bytes())
    archived_path = archive_root / f"{registry.stem}_archived_records.tsv"
    archived_fields = [*fieldnames, "archive_reason"]
    with archived_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=archived_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(archived)

    kept.sort(key=lambda row: (row.get("ts", ""), row.get("cell_id", "")))
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{registry.name}.",
        dir=registry.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(kept)
        os.replace(temporary_name, registry)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registry": str(registry),
        "raw_archive": str(raw_copy),
        "archived_records": str(archived_path),
        "input_rows": len(rows),
        "retained_rows": len(kept),
        "archived_rows": len(archived),
        "duplicate_success_records": duplicate_successes,
        "raw_sha256": _sha256(raw_copy),
        "cleaned_sha256": _sha256(registry),
    }
    manifest_path = archive_root / f"{registry.stem}_cleanup_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive superseded SD-grid registry records and retain one canonical record per cell."
    )
    parser.add_argument("registry", type=Path)
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()
    manifest = clean_registry(args.registry, args.archive_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
