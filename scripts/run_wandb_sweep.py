import argparse
import os
import yaml
import subprocess


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

    for a in attns:
        for s in seeds:
            cmd = ["python", program, "--attn", a]
            if "epochs" in fixed:
                cmd += ["--epochs", str(fixed["epochs"]["value"])]
            cmd += ["--seed", str(s)] if program.endswith("transformer.py") else []
            print("→", " ".join(cmd))
            env = os.environ.copy()
            subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()

