import argparse
import os
import random
import time
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.experiments.data_toy import ToyDiffConfig, make_toy_continuous_sequences
from set_attention.experiments.diffusion_core import SimpleDDPM
from set_attention.experiments.models import PositionalEncoding, timestep_embedding
from set_attention.eval.mmd_simple import mmd2_unbiased_from_feats
from set_attention.utils.metrics import chamfer_l2, one_nn_two_sample
from set_attention.utils.profiling import profiler
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher


def make_id_sequence(x: torch.Tensor) -> torch.Tensor:
    # x: (L, D)
    mean = x.mean(dim=1)
    sign = (mean > 0).long()
    base = torch.arange(x.size(0), dtype=torch.long) * 2
    return base + sign


class BankedDenoiser(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        layers: int,
        router_topk: int,
        ska_backend: str = "python",
        precision: str = "fp32",
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.router = TokenSetRouter(d_model=d_model, num_heads=nhead, topk=router_topk)
        self.set_attn = SetBankAttention(
            d_model=d_model,
            num_heads=nhead,
            tau=1.0,
            gamma=0.3,
            beta=1.0,
            score_mode="delta_plus_dot",
            eta=1.0,
            backend=ska_backend,
            precision=precision,
        )
        self.proj_out = nn.Linear(d_model, in_dim)
        self._bank = None

    def set_current_bank(self, Phi, Sig, Size, q_ptrs):
        self._bank = (Phi, Sig, Size, q_ptrs)

    def forward(self, x_t: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        assert self._bank is not None, "bank not set"
        Phi, Sig, Size, q_ptrs = self._bank
        h = self.proj_in(x_t)
        h = self.pos_enc(h)
        if t_embed.dim() == 2:
            t_embed = t_embed.unsqueeze(1)
        h = h + t_embed
        h = self.enc(h)
        Z, q_ptrs = self.set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
        routed = self.router(h, Z, Phi, q_ptrs)
        return self.proj_out(routed)


def run_diffusion_benchmark(
    args,
    model,
    ddpm,
    adapter,
    atom_emb,
    phi_snapshot,
    cache,
    train_data,
    optimizer,
    device,
):
    bench_batch = min(args.batch, train_data.size(0))
    if bench_batch == 0:
        print("[benchmark] no training data available.")
        return
    batch_idx = torch.arange(bench_batch, dtype=torch.long, device=device)
    xb = train_data[:bench_batch].to(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
        Phi, Sig, Size, q_ptrs = cache.gather_flat(batch_idx, phi_dynamic)
        model.set_current_bank(Phi, Sig, Size, q_ptrs)
        loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, args.d_model), args.d_model)
        loss.backward()
        optimizer.step()

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
    sequences = bench_batch * args.bench_iters
    throughput = sequences / elapsed if elapsed > 0 else 0.0
    print(
        f"[benchmark][diffusion] backend={args.ska_backend} precision={args.precision} "
        f"seq/s={throughput:.1f} elapsed={elapsed:.3f}s"
    )


def tensor_batch_iterator(
    data: torch.Tensor, batch_size: int, shuffle: bool, generator: Optional[torch.Generator] = None
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    if data.numel() == 0:
        return
    if shuffle:
        perm = torch.randperm(data.size(0), generator=generator)
        indices = perm
    else:
        indices = torch.arange(data.size(0), dtype=torch.long)
    for start in range(0, data.size(0), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield batch_idx.clone(), data[batch_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--minhash-k", type=int, default=32)
    ap.add_argument("--adapter-rank", type=int, default=0)
    ap.add_argument("--router-topk", type=int, default=0)
    ap.add_argument("--profile", "--prof", action="store_true", dest="profile")
    ap.add_argument("--config", type=str, default="configs/diffusion_toy.yaml")
    ap.add_argument("--ska-backend", choices=["python", "triton", "keops"], default="python")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--samples", type=int, default=None, help="Override number of synthetic sequences.")
    ap.add_argument("--data-seq-len", type=int, default=None, help="Override synthetic sequence length.")
    ap.add_argument("--data-dim", type=int, default=None, help="Override synthetic feature dimensionality.")
    ap.add_argument("--data-batch-size", type=int, default=None, help="Override synthetic batch size.")
    ap.add_argument("--data-val-frac", type=float, default=None, help="Override validation fraction.")
    ap.add_argument("--data-modes", type=int, default=None, help="Override number of mixture modes.")
    ap.add_argument("--data-seed", type=int, default=None, help="Override synthetic data seed.")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=20)
    defaults = ap.parse_args([])
    args = ap.parse_args()

    cfg_yaml = {}
    yaml_path = args.config
    if os.path.isfile(args.config):
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as handle:
            cfg_yaml = yaml.safe_load(handle) or {}

    def override_from_cfg(cfg_key: str, arg_name: str):
        if cfg_key in cfg_yaml and getattr(args, arg_name) == getattr(defaults, arg_name):
            setattr(args, arg_name, cfg_yaml[cfg_key])

    override_from_cfg("steps", "steps")
    override_from_cfg("d_model", "d_model")
    override_from_cfg("nhead", "nhead")
    override_from_cfg("layers", "layers")

    seed = int(cfg_yaml.get("seed", 2024))
    if args.data_seed is not None:
        seed = int(args.data_seed)
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(args.device)

    data_cfg = ToyDiffConfig(
        n_samples=int(cfg_yaml.get("n_samples", 1000)),
        seq_len=int(cfg_yaml.get("seq_len", 16)),
        dim=int(cfg_yaml.get("dim", 8)),
        batch_size=int(cfg_yaml.get("batch_size", 64)),
        val_frac=float(cfg_yaml.get("val_frac", 0.2)),
        seed=seed,
        n_modes=int(cfg_yaml.get("n_modes", 4)),
    )
    if args.samples is not None:
        data_cfg.n_samples = int(args.samples)
    if args.data_seq_len is not None:
        data_cfg.seq_len = int(args.data_seq_len)
    if args.data_dim is not None:
        data_cfg.dim = int(args.data_dim)
    if args.data_batch_size is not None:
        data_cfg.batch_size = int(args.data_batch_size)
    if args.data_val_frac is not None:
        data_cfg.val_frac = float(args.data_val_frac)
    if args.data_modes is not None:
        data_cfg.n_modes = int(args.data_modes)
    data_cfg.seed = seed
    train_loader, val_loader = make_toy_continuous_sequences(data_cfg)

    def subset_tensor(loader):
        ds = loader.dataset
        if hasattr(ds, "dataset") and hasattr(ds, "indices"):
            base_tensor = ds.dataset.tensors[0]
            return base_tensor[ds.indices].clone()
        if hasattr(ds, "tensors"):
            return ds.tensors[0].clone()
        raise ValueError("Unsupported dataset type for tensor extraction")

    train_data = subset_tensor(train_loader)
    val_data = subset_tensor(val_loader)

    seq_len = train_data.size(1) if train_data.dim() > 1 else 0
    feat_dim = train_data.size(2) if train_data.dim() > 2 else 0
    print(
        f"[Diffusion-Banked] dataset loaded | train sequences {train_data.size(0)} | "
        f"val sequences {val_data.size(0)} | seq len {seq_len} | dim {feat_dim}"
    )

    train_ids = [make_id_sequence(x) for x in train_data]
    val_ids = [make_id_sequence(x) for x in val_data]
    train_bank = build_windowed_bank_from_ids(train_ids, window=args.window, stride=args.stride).to(device)
    val_bank = build_windowed_bank_from_ids(val_ids, window=args.window, stride=args.stride).to(device)

    vocab_size = int(max(train_bank.values.max(), val_bank.values.max()).item() + 1) if train_bank.values.numel() > 0 else data_cfg.seq_len * 2
    atom_emb = nn.Embedding(vocab_size, args.d_model).to(device)
    adapter = None
    if args.adapter_rank > 0:
        adapter = AtomFeatureAdapter(args.d_model, rank=args.adapter_rank).to(device)
        with torch.no_grad():
            phi_snapshot = adapter(atom_emb.weight).detach()
    else:
        phi_snapshot = atom_emb.weight

    universe_ids = torch.arange(vocab_size, device=device, dtype=torch.long)
    universe = UniversePool(universe_ids, metadata={"task": "toy_diffusion"}).to(device)
    train_minhash = MinHasher(k=args.minhash_k, device=train_bank.values.device)
    val_minhash = MinHasher(k=args.minhash_k, device=val_bank.values.device)
    train_cache = SetFeatureCache(universe, train_bank.values, train_bank.set_offsets, train_bank.seq_offsets, minhash=train_minhash).to(device)
    val_cache = SetFeatureCache(universe, val_bank.values, val_bank.set_offsets, val_bank.seq_offsets, minhash=val_minhash).to(device)

    model = BankedDenoiser(
        in_dim=data_cfg.dim,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        router_topk=args.router_topk,
        ska_backend=args.ska_backend,
        precision=args.precision,
    ).to(device)
    ddpm = SimpleDDPM(T=args.steps, device=device)
    params = list(model.parameters()) + list(atom_emb.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    if args.benchmark:
        run_diffusion_benchmark(
            args,
            model,
            ddpm,
            adapter,
            atom_emb,
            phi_snapshot,
            train_cache,
            train_data,
            optimizer,
            device,
        )
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        train_gen = torch.Generator().manual_seed(seed + epoch)
        with profiler(args.profile) as prof:
            for batch_idx, xb in tensor_batch_iterator(train_data, data_cfg.batch_size, shuffle=True, generator=train_gen):
                xb = xb.to(device)
                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi, Sig, Size, q_ptrs = train_cache.gather_flat(batch_idx.to(device), phi_dynamic)
                model.set_current_bank(Phi, Sig, Size, q_ptrs)
                loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, args.d_model), args.d_model)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * xb.size(0)
                count += xb.size(0)
        train_loss = total_loss / max(1, count)

        model.eval()
        with torch.no_grad():
            mmds, chamfers, nn1s = [], [], []
            for batch_idx, xb in tensor_batch_iterator(val_data, data_cfg.batch_size, shuffle=False):
                xb = xb.to(device)
                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi, Sig, Size, q_ptrs = val_cache.gather_flat(batch_idx.to(device), phi_dynamic)
                model.set_current_bank(Phi, Sig, Size, q_ptrs)
                val_mmd, val_chamfer, val_nn1 = eval_suite(ddpm, model, xb, args.d_model)
                mmds.append(val_mmd)
                chamfers.append(val_chamfer)
                nn1s.append(val_nn1)

        def safe_mean(values):
            return float(sum(values) / len(values)) if values else float("nan")

        val_mmd_mean = safe_mean(mmds)
        val_chamfer_mean = safe_mean(chamfers)
        val_nn1_mean = safe_mean(nn1s)

        msg = (
            f"[Diffusion-Banked] epoch {epoch:02d} "
            f"train loss {train_loss:.4f} | val MMD {val_mmd_mean:.4f} | "
            f"Chamfer {val_chamfer_mean:.4f} | 1NN {val_nn1_mean:.3f}"
        )
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)


def eval_suite(ddpm: SimpleDDPM, model: BankedDenoiser, X: torch.Tensor, d_model: int):
    shape = X.shape
    x_gen = ddpm.sample(model, shape, lambda t, dim: timestep_embedding(t, d_model), d_model)
    x_flat = X.contiguous().view(shape[0], -1)
    g_flat = x_gen.contiguous().view(shape[0], -1)
    if x_flat.shape[1] != g_flat.shape[1]:
        d = min(x_flat.shape[1], g_flat.shape[1])
        x_flat = x_flat[:, :d]
        g_flat = g_flat[:, :d]
    mmd = float(mmd2_unbiased_from_feats(x_flat, g_flat, gamma=0.5).detach().cpu())
    chamfer = chamfer_l2(X, x_gen)
    nn1 = one_nn_two_sample(x_flat, g_flat)
    return mmd, chamfer, nn1


if __name__ == "__main__":
    main()
