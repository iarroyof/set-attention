import argparse
import csv
import itertools
import subprocess
import time
from pathlib import Path


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


def run(cmd, csv_path: Path):
    print("→", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        append_status(
            csv_path,
            {
                "script": cmd[1] if len(cmd) > 1 else "",
                "task": "sweep",
                "seed": cmd[-1] if "--seed" in cmd else "",
                "run_uid": f"exitcode-{int(time.time())}",
                "status": "exitcode",
                "exit_code": result.returncode,
                "skip_reason": f"exitcode={result.returncode}",
            },
        )
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["transformer", "diffusion", "vit"], default="transformer")
    ap.add_argument("--attn", nargs="*", default=["dot", "cosine", "rbf", "intersect", "ska", "ska_true"])  # ska alias to rbf
    ap.add_argument("--seeds", nargs="*", default=["1337", "2024", "7"])  # strings for convenience
    args = ap.parse_args()

    if args.which == "transformer":
        csv_path = Path("out/sweeps/transformer.csv")
        for a, s in itertools.product(args.attn, args.seeds):
            run(["python", "scripts/train_toy_transformer.py", "--attn", a, "--seed", s], csv_path)
    elif args.which == "diffusion":
        csv_path = Path("out/sweeps/diffusion.csv")
        for a, s in itertools.product(args.attn, args.seeds):
            run(["python", "scripts/train_toy_diffusion.py", "--attn", a, "--seed", s], csv_path)
    else:
        csv_path = Path("out/sweeps/vit.csv")
        for a, s in itertools.product(args.attn, args.seeds):
            run(["python", "scripts/train_tiny_vit_cifar.py", "--attn", a], csv_path)


if __name__ == "__main__":
    main()
