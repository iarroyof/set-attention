import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None

from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.sets.bank_cache import SetBankCache
from set_attention.sets.bank_utils import gather_bank_batch
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.sets.banked import BankedSetBatch
from set_attention.utils.profiling import profiler


def patch_ids_from_image(img: torch.Tensor, patch: int) -> torch.Tensor:
    # img: (3,H,W)
    C, H, W = img.shape
    gh = H // patch
    gw = W // patch
    ids = []
    for i in range(gh):
        for j in range(gw):
            patch_block = img[:, i * patch : (i + 1) * patch, j * patch : (j + 1) * patch]
            mean_val = patch_block.mean()
            base = i * gw + j
            ids.append(base * 2 + int(mean_val > 0.5))
    return torch.tensor(ids, dtype=torch.long)


def build_patch_banks(dataset, patch: int, limit: int) -> Tuple[BankedSetBatch, List[int]]:
    sequences = []
    for idx in range(limit):
        img, _ = dataset[idx]
        sequences.append(patch_ids_from_image(img, patch))
    bank = build_windowed_bank_from_ids(sequences, window=8, stride=4)
    return bank, [i for i in range(limit)]


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, patch=4, dim=128, img_size=32):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_size // patch) * (img_size // patch)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TinyViTBackbone(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, patch=4, img_size=32):
        super().__init__()
        self.patch = PatchEmbed(3, patch, dim, img_size)
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)

    def forward(self, x):
        tokens = self.patch(x)
        return self.enc(tokens)


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, limit: int):
        self.base = base_dataset
        self.limit = min(limit, len(base_dataset))

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return idx, img, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true"], default="dot")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=2048)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--minhash-k", type=int, default=64)
    ap.add_argument("--adapter-rank", type=int, default=0)
    ap.add_argument("--router-topk", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    if torchvision is None:
        raise RuntimeError("torchvision is required for the ViT banked demo.")

    device = torch.device(args.device)
    transform = T.Compose([T.ToTensor()])
    base_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    indexed_dataset = IndexedDataset(base_dataset, args.limit)
    loader = torch.utils.data.DataLoader(indexed_dataset, batch_size=args.batch, shuffle=True, num_workers=2)

    bank, order = build_patch_banks(base_dataset, args.patch, args.limit)
    bank = bank.to(device)
    vocab_size = int(bank.values.max().item() + 1) if bank.values.numel() > 0 else (args.patch * args.patch * 2)

    atom_emb = nn.Embedding(vocab_size, 128).to(device)
    adapter = None
    if args.adapter_rank > 0:
        adapter = AtomFeatureAdapter(128, rank=args.adapter_rank).to(device)
        with torch.no_grad():
            phi_snapshot = adapter(atom_emb.weight).detach()
    else:
        phi_snapshot = atom_emb.weight
    cache = SetBankCache(phi_snapshot, minhash_k=args.minhash_k)
    cache.precompute(bank.values, bank.set_offsets)

    backbone = TinyViTBackbone(dim=128, depth=4, heads=4, patch=args.patch).to(device)
    set_attn = SetBankAttention(
        d_model=128,
        num_heads=4,
        tau=1.0,
        gamma=0.3,
        beta=1.0,
        score_mode="delta_plus_dot",
        eta=1.0,
    ).to(device)
    router = TokenSetRouter(d_model=128, num_heads=4, topk=args.router_topk).to(device)
    head = nn.Linear(128, 10).to(device)

    params = list(backbone.parameters()) + list(set_attn.parameters()) + list(router.parameters()) + list(head.parameters()) + list(atom_emb.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    optim = torch.optim.AdamW(params, lr=3e-4)

    for ep in range(1, args.epochs + 1):
        backbone.train()
        total_loss, total_acc, total_n = 0.0, 0.0, 0
        with profiler(args.profile) as prof:
            if args.profile and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for idx_batch, xb, yb in loader:
                batch_idx = idx_batch.to(device)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                tokens = backbone(xb)
                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi, Sig, Size, mask = gather_bank_batch(bank, cache, batch_idx, phi_dynamic, adapter is not None, 128, args.minhash_k)
                Z = set_attn(Phi, Sig, Size, mask, Phi, Sig, Size, mask)
                routed = router(tokens, Z, Phi, mask)
                logits = head(routed)
                loss = F.cross_entropy(logits, yb)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                pred = logits.argmax(dim=-1)
                total_loss += float(loss.detach()) * xb.size(0)
                total_acc += float((pred == yb).sum().item())
                total_n += xb.size(0)
                if total_n >= args.limit:
                    break
        msg = f"[ViT-Banked][{args.attn}] epoch {ep:02d} loss {total_loss/max(1,total_n):.4f} acc {total_acc/max(1,total_n):.3f}"
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)


if __name__ == "__main__":
    main()
