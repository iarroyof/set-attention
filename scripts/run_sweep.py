import argparse
import subprocess
import itertools


def run(cmd):
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["transformer", "diffusion", "vit"], default="transformer")
    ap.add_argument("--attn", nargs="*", default=["dot", "cosine", "rbf", "intersect", "ska", "ska_true"])  # ska alias to rbf
    ap.add_argument("--seeds", nargs="*", default=["1337", "2024", "7"])  # strings for convenience
    args = ap.parse_args()

    if args.which == "transformer":
        for a, s in itertools.product(args.attn, args.seeds):
            run(["python", "scripts/train_toy_transformer.py", "--attn", a, "--seed", s])
    elif args.which == "diffusion":
        for a, s in itertools.product(args.attn, args.seeds):
            run(["python", "scripts/train_toy_diffusion.py", "--attn", a, "--seed", s])
    else:
        for a, s in itertools.product(args.attn, args.seeds):
            run(["python", "scripts/train_tiny_vit_cifar.py", "--attn", a])


if __name__ == "__main__":
    main()
