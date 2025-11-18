import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import torch

from set_attention.heads.banked_attention import SetBankAttention


def build_random_bank(num_seqs: int, sets_per_seq: int, d_model: int, sig_len: int, max_size: int, device):
    total_sets = num_seqs * sets_per_seq
    ptrs = torch.arange(0, total_sets + 1, sets_per_seq, device=device, dtype=torch.long)
    sig = torch.randint(0, 2**16, (total_sets, sig_len), device=device, dtype=torch.long)
    size = torch.randint(1, max_size + 1, (total_sets,), device=device, dtype=torch.long)
    phi = torch.randn(total_sets, d_model, device=device)
    return phi, sig, size, ptrs


def run_benchmark(args, backend: str) -> Dict[str, float]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    if args.precision == "fp16" and device.type == "cuda":
        dtype = torch.float16
    elif args.precision == "bf16" and device.type == "cuda":
        dtype = torch.bfloat16

    torch.manual_seed(args.seed)
    phi_q, sig_q, size_q, q_ptrs = build_random_bank(
        args.seqs,
        args.sets_q,
        args.d_model,
        args.sig_len,
        args.max_set_size,
        device,
    )
    torch.manual_seed(args.seed + 123)
    phi_k, sig_k, size_k, k_ptrs = build_random_bank(
        args.seqs,
        args.sets_k,
        args.d_model,
        args.sig_len,
        args.max_set_size,
        device,
    )
    phi_q = phi_q.to(dtype).requires_grad_(True)
    phi_k = phi_k.to(dtype).requires_grad_(True)

    attn = SetBankAttention(
        d_model=args.d_model,
        num_heads=args.heads,
        tau=args.tau,
        gamma=args.gamma,
        beta=args.beta,
        score_mode=args.score_mode,
        eta=args.eta,
        backend=backend,
        precision=args.precision,
    ).to(device)

    def step():
        attn.zero_grad(set_to_none=True)
        out, _ = attn(phi_q, sig_q, size_q, q_ptrs, phi_k, sig_k, size_k, k_ptrs)
        loss = out.sum()
        loss.backward()
        if phi_q.grad is not None:
            phi_q.grad.zero_()
        if phi_k.grad is not None:
            phi_k.grad.zero_()

    for _ in range(args.warmup):
        step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_sets = args.seqs * args.sets_q * args.steps
    sets_per_sec = total_sets / elapsed if elapsed > 0 else 0.0
    return {
        "backend": backend,
        "precision": args.precision,
        "device": device.type,
        "d_model": args.d_model,
        "heads": args.heads,
        "sig_len": args.sig_len,
        "sets_q": args.sets_q,
        "sets_k": args.sets_k,
        "seqs": args.seqs,
        "steps": args.steps,
        "elapsed_s": elapsed,
        "sets_per_sec": sets_per_sec,
    }


def maybe_write_csv(path: Path, rows: List[Dict[str, float]]):
    if path is None:
        return
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Microbenchmark SetBankAttention.")
    parser.add_argument("--backends", nargs="+", default=["python"], help="Backends to benchmark.")
    parser.add_argument("--device", default="auto", help="Device to run on (auto|cuda|cpu).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sets-q", type=int, default=32)
    parser.add_argument("--sets-k", type=int, default=32)
    parser.add_argument("--seqs", type=int, default=16)
    parser.add_argument("--sig-len", type=int, default=64)
    parser.add_argument("--max-set-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--score-mode", choices=["delta_rbf", "delta_plus_dot", "intersect_norm", "intersect_plus_dot", "dot"], default="delta_plus_dot")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    results: List[Dict[str, float]] = []
    for backend in args.backends:
        try:
            row = run_benchmark(args, backend)
            print(
                f"[bench] backend={row['backend']:>6s} precision={row['precision']} "
                f"device={row['device']} sets/s={row['sets_per_sec']:.1f} elapsed={row['elapsed_s']:.3f}s"
            )
            results.append(row)
        except RuntimeError as exc:
            print(f"[bench] backend={backend} failed: {exc}")
    if results and csv_path:
        maybe_write_csv(csv_path, results)


if __name__ == "__main__":
    main()
