import argparse
import os
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None

from set_attention.data import resolve_data_root
from set_attention.utils.wandb import init_wandb

from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.sets.bank_utils import pad_segments_from_ptrs, update_coverage_stats
from set_attention.sets.banked import BankedSetBatch
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher
from set_attention.utils.sample_logging import select_sample_indices
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


def build_patch_banks(dataset, patch: int, limit: int) -> BankedSetBatch:
    sequences = []
    for idx in range(limit):
        img, _ = dataset[idx]
        sequences.append(patch_ids_from_image(img, patch))
    bank = build_windowed_bank_from_ids(sequences, window=8, stride=4)
    return bank


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


class SyntheticVisionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, img_size: int = 32, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(num_samples, 3, img_size, img_size, generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return int(self.images.size(0))

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices: List[int]):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, label = self.base[base_idx]
        return base_idx, img, label


def run_vit_benchmark(
    args,
    backbone,
    set_attn,
    router,
    head,
    adapter,
    atom_emb,
    phi_snapshot,
    cache,
    train_dataset,
    optim,
    device,
    wandb_run,
):
    bench_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    try:
        idx_batch, xb, yb = next(iter(bench_loader))
    except StopIteration:
        print("[benchmark] empty dataset.")
        return
    batch_idx = idx_batch.to(device)
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    def step():
        optim.zero_grad(set_to_none=True)
        tokens = backbone(xb)
        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
        Phi, Sig, Size, q_ptrs = cache.gather_flat(batch_idx, phi_dynamic)
        Z, q_ptrs = set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
        routed = router(tokens, Z, Phi, q_ptrs)
        cls_repr = routed.mean(dim=1)
        logits = head(cls_repr)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optim.step()

    for _ in range(args.bench_warmup):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.bench_iters):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    images = xb.size(0) * args.bench_iters
    throughput = images / elapsed if elapsed > 0 else 0.0
    print(
        f"[benchmark][vit] backend={args.ska_backend} precision={args.precision} "
        f"imgs/s={throughput:.1f} elapsed={elapsed:.3f}s"
    )
    if wandb_run.enabled:
        wandb_run.log(
            {
                "benchmark/images_per_s": throughput,
                "benchmark/elapsed_s": elapsed,
            }
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true"], default="dot")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--minhash-k", type=int, default=64)
    ap.add_argument("--adapter-rank", type=int, default=0)
    ap.add_argument("--router-topk", type=int, default=0)
    ap.add_argument("--profile", "--prof", action="store_true", dest="profile")
    ap.add_argument("--data-mode", choices=["cifar10", "synthetic"], default="cifar10")
    ap.add_argument("--demo-samples", type=int, default=512, help="Used when --data-mode synthetic is selected.")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--ska-backend", choices=["python", "triton", "keops"], default="python")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--data-root", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="")
    ap.add_argument("--wandb-run-name", type=str, default="")
    ap.add_argument("--wandb-tags", type=str, default="")
    ap.add_argument("--sample-count", type=int, default=10, help="Number of validation samples to log.")
    ap.add_argument("--sample-seed", type=int, default=1337, help="Seed for selecting logged samples.")
    args = ap.parse_args()

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_config = {
        "script": "train_tiny_vit_banked",
        "data_mode": args.data_mode,
        "ska_backend": args.ska_backend,
        "precision": args.precision,
        "window": args.window,
        "stride": args.stride,
        "minhash_k": args.minhash_k,
        "router_topk": args.router_topk,
        "adapter_rank": args.adapter_rank,
        "batch": args.batch,
        "sample_count": args.sample_count,
    }
    wandb_run = init_wandb(
        args.wandb,
        args.wandb_project or None,
        args.wandb_run_name or None,
        wandb_config,
        wandb_tags,
    )

    data_root = resolve_data_root(args.data_root)
    device = torch.device(args.device)
    pin_memory = device.type == "cuda" and args.num_workers > 0

    if args.data_mode == "cifar10":
        if torchvision is None:
            raise RuntimeError("torchvision is required for CIFAR10 data mode. Install torchvision or use --data-mode synthetic.")
        transform = T.Compose([T.ToTensor()])
        vision_root = data_root / "vision" / "cifar10"
        vision_root.mkdir(parents=True, exist_ok=True)
        base_dataset = torchvision.datasets.CIFAR10(root=str(vision_root), train=True, download=True, transform=transform)
        dataset_len = len(base_dataset)
    else:
        base_dataset = SyntheticVisionDataset(num_samples=args.demo_samples, img_size=32, num_classes=args.num_classes, seed=args.seed)
        dataset_len = len(base_dataset)

    total_limit = min(args.limit, dataset_len)
    indices_full = list(range(total_limit))
    if args.val_frac <= 0.0 or total_limit < 2:
        train_indices = indices_full
        val_indices: List[int] = []
    else:
        val_count = max(1, int(total_limit * args.val_frac))
        if val_count >= total_limit:
            val_count = max(1, total_limit // 5)
        train_indices = indices_full[:-val_count] if val_count < total_limit else indices_full
        val_indices = indices_full[-val_count:] if val_count < total_limit else []

    train_dataset = IndexedDataset(base_dataset, train_indices)
    val_dataset = IndexedDataset(base_dataset, val_indices) if val_indices else None
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if val_dataset is not None
        else None
    )

    bank = build_patch_banks(base_dataset, args.patch, total_limit).to(device)
    val_count_log = len(val_dataset) if val_dataset is not None else 0
    print(f"[ViT-Banked] dataset prepared | train samples {len(train_dataset)} | val samples {val_count_log} | total banked {total_limit}")
    vocab_size = int(bank.values.max().item() + 1) if bank.values.numel() > 0 else (args.patch * args.patch * 2)

    atom_emb = nn.Embedding(vocab_size, 128).to(device)
    adapter = None
    if args.adapter_rank > 0:
        adapter = AtomFeatureAdapter(128, rank=args.adapter_rank).to(device)
        with torch.no_grad():
            phi_snapshot = adapter(atom_emb.weight).detach()
    else:
        phi_snapshot = atom_emb.weight
    universe_ids = torch.arange(vocab_size, device=device, dtype=torch.long)
    universe = UniversePool(universe_ids, metadata={"task": "tiny_vit"}).to(device)
    minhash = MinHasher(k=args.minhash_k, device=bank.values.device)
    cache = SetFeatureCache(universe, bank.values, bank.set_offsets, bank.seq_offsets, minhash=minhash).to(device)

    backbone = TinyViTBackbone(dim=128, depth=4, heads=4, patch=args.patch).to(device)
    set_attn = SetBankAttention(
        d_model=128,
        num_heads=4,
        tau=1.0,
        gamma=0.3,
        beta=1.0,
        score_mode="delta_plus_dot",
        eta=1.0,
        backend=args.ska_backend,
        precision=args.precision,
    ).to(device)
    router = TokenSetRouter(d_model=128, num_heads=4, topk=args.router_topk).to(device)
    head = nn.Linear(128, 10).to(device)

    params = list(backbone.parameters()) + list(set_attn.parameters()) + list(router.parameters()) + list(head.parameters()) + list(atom_emb.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    optim = torch.optim.AdamW(params, lr=3e-4)

    if args.benchmark:
        run_vit_benchmark(
            args,
            backbone,
            set_attn,
            router,
            head,
            adapter,
            atom_emb,
            phi_snapshot,
            cache,
            train_dataset,
            optim,
            device,
            wandb_run,
        )
        wandb_run.finish()
        return

    modules = [backbone, set_attn, router, head]
    if adapter is not None:
        modules.append(adapter)

    def set_mode(train_flag: bool) -> None:
        for mod in modules:
            mod.train(train_flag)

    coverage_size = cache.num_sets()

    def evaluate(loader, sample_indices=None):
        if loader is None or len(loader.dataset) == 0:
            return None
        set_mode(False)
        total_loss = 0.0
        total_acc = 0.0
        total_top5 = 0.0
        total_n = 0
        sample_lookup = {}
        sample_records = {}
        if sample_indices:
            sample_lookup = {idx: order for order, idx in enumerate(sample_indices)}
        with torch.no_grad():
            for idx_batch, xb, yb in loader:
                batch_idx = idx_batch.to(device)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                tokens = backbone(xb)
                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi, Sig, Size, q_ptrs = cache.gather_flat(batch_idx, phi_dynamic)
                Z, q_ptrs = set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
                routed = router(tokens, Z, Phi, q_ptrs)
                cls_repr = routed.mean(dim=1)
                logits = head(cls_repr)
                loss = F.cross_entropy(logits, yb)
                pred = logits.argmax(dim=-1)
                topk = logits.topk(min(5, logits.size(-1)), dim=-1).indices
                total_loss += float(loss.detach()) * xb.size(0)
                total_acc += float((pred == yb).sum().item())
                total_top5 += float((topk == yb.unsqueeze(-1)).any(dim=-1).sum().item())
                total_n += xb.size(0)
                if sample_lookup:
                    idx_list = batch_idx.detach().cpu().tolist()
                    for local_pos, global_idx in enumerate(idx_list):
                        order = sample_lookup.get(int(global_idx))
                        if order is None or order in sample_records:
                            continue
                        sample_records[order] = {
                            "idx": int(global_idx),
                            "image": xb[local_pos].detach().cpu(),
                            "target": int(yb[local_pos]),
                            "pred": int(pred[local_pos]),
                        }
        set_mode(True)
        denom = max(1, total_n)
        metrics = {
            "loss": total_loss / denom,
            "acc": total_acc / denom,
            "top5": total_top5 / denom,
        }
        if sample_records:
            ordered = [sample_records[i] for i in sorted(sample_records)]
            metrics["samples"] = ordered
        return metrics

    for ep in range(1, args.epochs + 1):
        set_mode(True)
        total_loss, total_acc, total_top5, total_n = 0.0, 0.0, 0.0, 0
        sets_seen_total = 0.0
        seq_seen_total = 0
        coverage_mask = torch.zeros(coverage_size, dtype=torch.bool)
        with profiler(args.profile) as prof:
            if args.profile and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for idx_batch, xb, yb in loader:
                batch_idx = idx_batch.to(device)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                tokens = backbone(xb)
                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi, Sig, Size, q_ptrs = cache.gather_flat(batch_idx, phi_dynamic)
                Z, q_ptrs = set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
                routed = router(tokens, Z, Phi, q_ptrs)
                cls_repr = routed.mean(dim=1)
                logits = head(cls_repr)
                loss = F.cross_entropy(logits, yb)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                pred = logits.argmax(dim=-1)
                topk = logits.topk(min(5, logits.size(-1)), dim=-1).indices
                batch_top5 = (topk == yb.unsqueeze(-1)).any(dim=-1).sum().item()
                with torch.no_grad():
                    set_idx_batch, _ = cache.gather_sequence_sets(batch_idx)
                    _, mask = pad_segments_from_ptrs(Phi.new_zeros(Phi.size(0), 1), q_ptrs)
                batch_sets, batch_seqs, coverage_mask = update_coverage_stats(mask, set_idx_batch, coverage_mask)
                sets_seen_total += float(batch_sets)
                seq_seen_total += batch_seqs
                total_loss += float(loss.detach()) * xb.size(0)
                total_acc += float((pred == yb).sum().item())
                total_top5 += float(batch_top5)
                total_n += xb.size(0)
        denom = max(1, total_n)
        avg_loss = total_loss / denom
        avg_acc = total_acc / denom
        avg_top5 = total_top5 / denom
        avg_sets_per_seq = sets_seen_total / max(1, seq_seen_total)
        coverage_ratio = (coverage_mask.float().mean().item() if coverage_mask.numel() > 0 else 0.0)
        if val_loader is not None:
            sample_indices = select_sample_indices(len(val_loader.dataset), args.sample_count, args.sample_seed + ep)
        else:
            sample_indices = None
        val_metrics = evaluate(val_loader, sample_indices)
        val_samples = None
        if val_metrics is not None and "samples" in val_metrics:
            val_samples = val_metrics["samples"]
            del val_metrics["samples"]
        msg = (
            f"[ViT-Banked][{args.attn}] epoch {ep:02d} "
            f"train loss {avg_loss:.4f} acc {avg_acc:.3f} top5 {avg_top5:.3f} "
            f"| sets/seq {avg_sets_per_seq:.2f} | coverage {coverage_ratio*100:.1f}%"
        )
        if val_metrics is not None:
            msg += (
                f" | val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f} "
                f"top5 {val_metrics['top5']:.3f}"
            )
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)
        if wandb_run.enabled:
            payload = {
                "train/loss": avg_loss,
                "train/acc": avg_acc,
                "train/top5": avg_top5,
                "train/sets_per_seq": avg_sets_per_seq,
                "train/coverage": coverage_ratio,
            }
            if val_metrics is not None:
                payload.update(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/acc": val_metrics["acc"],
                        "val/top5": val_metrics["top5"],
                    }
                )
            if val_samples:
                preview_lines = []
                wandb_images = []
                for order, sample in enumerate(val_samples):
                    caption = f"[{order}] idx={sample['idx']} target={sample['target']} pred={sample['pred']}"
                    preview_lines.append(caption)
                    try:
                        import wandb  # type: ignore

                        wandb_images.append(wandb.Image(sample["image"], caption=caption))
                    except Exception:
                        pass
                if preview_lines:
                    payload["samples/val_preview"] = "\n".join(preview_lines)
                if wandb_images:
                    payload["samples/val_images"] = wandb_images
            if args.profile:
                payload["train/time_s"] = prof["time_s"]
                if torch.cuda.is_available():
                    payload["train/peak_vram_mib"] = prof["gpu_peak_mem_mib"]
            wandb_run.log(payload, step=ep)

    wandb_run.finish()


if __name__ == "__main__":
    main()
