#!/usr/bin/env python3
"""Validate and summarize A8 L=8192 linear follow-up artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "paper_mechanisms" / "a8_l8192_linear_followup"
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"

ALL_RUNS = TABLES / "a8_l8192_linear_followup_all_runs.tsv"
SUMMARY = TABLES / "a8_l8192_linear_followup_summary.tsv"
MANIFEST = CHECKS / "a8_l8192_linear_followup_manifest.json"


EXPECTED = [
    ("baseline_linear_landmark", "baseline_linear_landmark_L8192_seed{seed}", None, None),
    ("set_linear_landmark", "set_linear_landmark_L8192_w8_s4_seed{seed}", 8, 4),
]

NUMERIC_KEYS = [
    "train/loss",
    "val/loss",
    "train/ppl",
    "val/ppl",
    "train/peak_vram_mib",
    "train/time_per_epoch_s",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        out = float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{key} is not numeric in {row.get('run_name', '<unknown>')}: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{key} is not finite in {row.get('run_name', '<unknown>')}: {value!r}")
    return out


def find_csv(slug: str) -> Path:
    matches = sorted(RAW.glob(f"**/{slug}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one CSV for {slug}, found {len(matches)}")
    return matches[0]


def validate_metadata(meta: dict[str, object], family: str, seed: int, window: int | None, stride: int | None) -> None:
    required = {
        "data.seq_len": 8192,
        "data.batch_size": 1,
        "training.epochs": 10,
        "training.seed": seed,
        "model.backend": "landmark",
        "model.backend_params.landmark_coverage": 0.25,
        "resolved.landmark_coverage": 0.25,
    }
    for key, expected in required.items():
        observed = meta.get(key)
        if observed != expected:
            raise ValueError(f"{family} seed {seed}: {key} expected {expected!r}, observed {observed!r}")
    if family.startswith("baseline"):
        if meta.get("model.implementation") != "baseline_token":
            raise ValueError(f"{family} seed {seed}: expected baseline_token")
        if meta.get("resolved.landmark_count") != 2048:
            raise ValueError(f"{family} seed {seed}: expected landmark_count=2048")
        return
    if meta.get("model.implementation") != "set_only":
        raise ValueError(f"{family} seed {seed}: expected set_only")
    expected_set = {
        "model.window_size": window,
        "model.stride": stride,
        "model.set_causality_mode": "strict_past",
        "resolved.output_residual_mode": "empty_only",
        "resolved.landmark_count": 512,
    }
    for key, expected in expected_set.items():
        observed = meta.get(key)
        if observed != expected:
            raise ValueError(f"{family} seed {seed}: {key} expected {expected!r}, observed {observed!r}")


def summarize(values: list[float]) -> dict[str, float]:
    n = len(values)
    sd = stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": mean(values), "std": sd, "ci95": ci95}


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    all_rows: list[dict[str, object]] = []

    status_path = RAW / "a8_l8192_linear_followup_status.tsv"
    if not status_path.exists():
        failures.append(f"missing status TSV: {status_path}")
    else:
        status_rows = list(csv.DictReader(status_path.open(encoding="utf-8"), delimiter="\t"))
        if len(status_rows) != 10:
            failures.append(f"expected 10 status rows, found {len(status_rows)}")
        bad = [r for r in status_rows if r.get("exit_code") != "0"]
        if bad:
            failures.append(f"nonzero exits: {bad}")

    for family, template, window, stride in EXPECTED:
        for seed in range(5):
            slug = template.format(seed=seed)
            try:
                csv_path = find_csv(slug)
                json_path = csv_path.with_suffix(".json")
                if not json_path.exists():
                    raise FileNotFoundError(f"missing JSON for {slug}")
                rows = read_csv(csv_path)
                if len(rows) != 10:
                    raise ValueError(f"{slug}: expected 10 epochs, found {len(rows)}")
                last = rows[-1]
                if int(float(last.get("epoch", "nan"))) != 10:
                    raise ValueError(f"{slug}: final epoch is {last.get('epoch')!r}, expected 10")
                for key in NUMERIC_KEYS:
                    as_float(last, key)
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                validate_metadata(meta, family, seed, window, stride)
                all_rows.append(
                    {
                        "family": family,
                        "backend": "landmark",
                        "seed": seed,
                        "L": 8192,
                        "D": 384,
                        "d_ff": 1536,
                        "window_size": window if window is not None else "NA",
                        "stride": stride if stride is not None else "NA",
                        "M": 2047 if window is not None else "NA",
                        "landmark_count": meta.get("resolved.landmark_count", "NA"),
                        "epochs": 10,
                        "lr": meta.get("training.lr"),
                        "final_train_ppl": as_float(last, "train/ppl"),
                        "final_val_ppl": as_float(last, "val/ppl"),
                        "peak_vram_mib": as_float(last, "train/peak_vram_mib"),
                        "time_per_epoch_s": as_float(last, "train/time_per_epoch_s"),
                        "csv_path": str(csv_path.relative_to(ROOT)),
                        "json_path": str(json_path.relative_to(ROOT)),
                        "csv_sha256": sha256(csv_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(str(exc))

    with ALL_RUNS.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "family",
            "backend",
            "seed",
            "L",
            "D",
            "d_ff",
            "window_size",
            "stride",
            "M",
            "landmark_count",
            "epochs",
            "lr",
            "final_train_ppl",
            "final_val_ppl",
            "peak_vram_mib",
            "time_per_epoch_s",
            "csv_path",
            "json_path",
            "csv_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows: list[dict[str, object]] = []
    for family in sorted({str(r["family"]) for r in all_rows}):
        group = [r for r in all_rows if r["family"] == family]
        val = summarize([float(r["final_val_ppl"]) for r in group])
        train = summarize([float(r["final_train_ppl"]) for r in group])
        vram = summarize([float(r["peak_vram_mib"]) for r in group])
        time_s = summarize([float(r["time_per_epoch_s"]) for r in group])
        first = group[0]
        summary_rows.append(
            {
                "family": family,
                "backend": "landmark",
                "n": len(group),
                "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda r: int(r["seed"]))),
                "L": 8192,
                "D": 384,
                "d_ff": 1536,
                "window_size": first["window_size"],
                "stride": first["stride"],
                "M": first["M"],
                "landmark_count": first["landmark_count"],
                "mean_final_val_ppl": val["mean"],
                "std_final_val_ppl": val["std"],
                "ci95_final_val_ppl": val["ci95"],
                "mean_final_train_ppl": train["mean"],
                "std_final_train_ppl": train["std"],
                "mean_peak_vram_mib": vram["mean"],
                "std_peak_vram_mib": vram["std"],
                "mean_time_per_epoch_s": time_s["mean"],
                "std_time_per_epoch_s": time_s["std"],
            }
        )

    by_family = {r["family"]: r for r in summary_rows}
    if {"baseline_linear_landmark", "set_linear_landmark"}.issubset(by_family):
        base = by_family["baseline_linear_landmark"]
        setrow = by_family["set_linear_landmark"]
        setrow["delta_vs_baseline_val_ppl"] = (
            float(setrow["mean_final_val_ppl"]) - float(base["mean_final_val_ppl"])
        )
        setrow["vram_ratio_vs_baseline"] = (
            float(setrow["mean_peak_vram_mib"]) / float(base["mean_peak_vram_mib"])
        )
        base["delta_vs_baseline_val_ppl"] = 0.0
        base["vram_ratio_vs_baseline"] = 1.0

    with SUMMARY.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "family",
            "backend",
            "n",
            "seeds",
            "L",
            "D",
            "d_ff",
            "window_size",
            "stride",
            "M",
            "landmark_count",
            "mean_final_val_ppl",
            "std_final_val_ppl",
            "ci95_final_val_ppl",
            "mean_final_train_ppl",
            "std_final_train_ppl",
            "mean_peak_vram_mib",
            "std_peak_vram_mib",
            "mean_time_per_epoch_s",
            "std_time_per_epoch_s",
            "delta_vs_baseline_val_ppl",
            "vram_ratio_vs_baseline",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "phase": "A8.3 L8192 linear follow-up",
        "status": "pass" if not failures else "fail",
        "expected_runs": 10,
        "validated_runs": len(all_rows),
        "failures": failures,
        "all_runs_tsv": str(ALL_RUNS.relative_to(ROOT)),
        "summary_tsv": str(SUMMARY.relative_to(ROOT)),
        "raw_root": str(RAW.relative_to(ROOT)),
        "families": summary_rows,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
