import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="Path to sweep yaml (transformer/diffusion/vit)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.sweep, "r", encoding="utf-8"))
    program = cfg["program"]
    params = cfg["parameters"]

    attns = params.get("attn", {}).get("values", ["dot"])  # type: ignore
    seeds = params.get("seed", {}).get("values", ["1337"])  # type: ignore
    fixed = {k: v for k, v in params.items() if isinstance(v, dict) and "value" in v}

    def append_status(csv_path: Path, row: dict) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        if csv_path.exists():
            with csv_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                existing = reader.fieldnames or []
                rows = list(reader)
            extras = [c for c in fieldnames if c not in existing]
            if extras:
                new_fields = existing + extras
                with csv_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=new_fields)
                    writer.writeheader()
                    for prev in rows:
                        writer.writerow({c: prev.get(c, "") for c in new_fields})
                existing = new_fields
            with csv_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=existing)
                writer.writerow({c: row.get(c, "") for c in existing})
            return
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    csv_path = Path("out/sweeps/wandb.csv")
    for a in attns:
        for s in seeds:
            cmd = ["python", program, "--attn", a]
            if "epochs" in fixed:
                cmd += ["--epochs", str(fixed["epochs"]["value"])]
            cmd += ["--seed", str(s)] if program.endswith("transformer.py") else []
            print("→", " ".join(cmd))
            env = os.environ.copy()
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                append_status(
                    csv_path,
                    {
                        "script": program,
                        "task": "wandb_sweep",
                        "seed": s,
                        "run_uid": f"exitcode-{int(time.time())}-{s}",
                        "status": "exitcode",
                        "exit_code": result.returncode,
                        "skip_reason": f"exitcode={result.returncode}",
                    },
                )


if __name__ == "__main__":
    main()
