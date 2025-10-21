import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from set_attention.patch import replace_multihead_attn
from set_attention.utils.profiling import profiler


def make_synthetic_sets(n=2000, n_points=64, seed=42):
    g = torch.Generator().manual_seed(seed)
    data = []
    labels = []
    for i in range(n):
        cls = int(torch.randint(0, 3, (1,), generator=g).item())
        labels.append(cls)
        if cls == 0:
            # circle
            theta = torch.rand(n_points, generator=g) * 2 * math.pi
            r = 0.3 + 0.05 * torch.randn(n_points, generator=g)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
        elif cls == 1:
            # horizontal bar
            x = torch.linspace(-0.3, 0.3, n_points)
            y = 0.05 * torch.randn(n_points, generator=g)
        else:
            # vertical bar
            y = torch.linspace(-0.3, 0.3, n_points)
            x = 0.05 * torch.randn(n_points, generator=g)
        pts = torch.stack([x, y], dim=1) + 0.02 * torch.randn(n_points, 2, generator=g)
        data.append(pts)
    return torch.stack(data), torch.tensor(labels)


class SetEncoder(nn.Module):
    def __init__(self, d_in=2, d_model=64, nhead=4, layers=3):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls = nn.Linear(d_model, 3)

    def forward(self, x):
        h = self.proj(x)
        h = self.enc(h)
        h = h.mean(dim=1)
        return self.cls(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true"], default="dot")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    X, y = make_synthetic_sets()
    model = SetEncoder()
    if args.attn != "dot":
        replace_multihead_attn(model, sim=args.attn)
    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for ep in range(1, args.epochs + 1):
        with profiler(args.profile) as prof:
            model.train()
            xb = X[:256].to(device)
            yb = y[:256].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        pred = logits.argmax(dim=-1)
        acc = (pred == yb).float().mean().item()
        msg = f"[PatchSet][{args.attn}] epoch {ep:02d} loss {loss.item():.4f} acc {acc:.3f}"
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)


if __name__ == "__main__":
    main()

