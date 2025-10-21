import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchvision.transforms as T
except Exception as e:
    torchvision = None

from set_attention.patch import replace_multihead_attn


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, patch=4, dim=64, img_size=28):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_size // patch) * (img_size // patch)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TinyViT(nn.Module):
    def __init__(self, num_classes=10, dim=64, depth=3, heads=2, patch=4, img_size=28):
        super().__init__()
        self.patch = PatchEmbed(1, patch, dim, img_size)
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.cls = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = self.enc(x)
        x = x.mean(dim=1)
        return self.cls(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true"], default="dot")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    if torchvision is None:
        raise RuntimeError("torchvision is required for MNIST demo. Please install torchvision.")

    device = torch.device(args.device)
    tfm = T.Compose([T.ToTensor()])
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2)

    model = TinyViT()
    if args.attn != "dot":
        replace_multihead_attn(model, sim=args.attn)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for ep in range(1, args.epochs + 1):
        if args.profile and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.size(0)
            n += xb.size(0)
            if n >= 8192:
                break
        elapsed = time.time() - t0
        msg = f"[ViT-MNIST][{args.attn}] epoch {ep:02d} loss {total/max(1,n):.4f}"
        if args.profile:
            msg += f" | time {elapsed:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {torch.cuda.max_memory_allocated()/(1024**2):.1f} MiB"
        print(msg)


if __name__ == "__main__":
    main()

