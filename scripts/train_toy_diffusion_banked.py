import argparse
import os
import random

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
from set_attention.sets.bank_cache import SetBankCache
from set_attention.sets.bank_utils import gather_bank_batch
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter


def make_id_sequence(x: torch.Tensor) -> torch.Tensor:
    # x: (L, D)
    mean = x.mean(dim=1)
    sign = (mean > 0).long()
    base = torch.arange(x.size(0), dtype=torch.long) * 2
    return base + sign


class BankedDenoiser(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, layers: int, router_topk: int):
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
        )
        self.proj_out = nn.Linear(d_model, in_dim)
        self._bank = None

    def set_current_bank(self, Phi, Sig, Size, mask):
        self._bank = (Phi, Sig, Size, mask)

    def forward(self, x_t: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        assert self._bank is not None, "bank not set"
        Phi, Sig, Size, mask = self._bank
        h = self.proj_in(x_t)
        h = self.pos_enc(h)
        if t_embed.dim() == 2:
            t_embed = t_embed.unsqueeze(1)
        h = h + t_embed
        h = self.enc(h)
        Z = self.set_attn(Phi, Sig, Size, mask, Phi, Sig, Size, mask)
        routed = self.router(h, Z, Phi, mask)
        return self.proj_out(routed)


def tensor_batch_iterator(data: torch.Tensor, batch_size: int, shuffle: bool) -> torch.Iterator:
    indices = list(range(data.size(0)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = data[batch_idx]
        yield torch.tensor(batch_idx, dtype=torch.long), batch


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
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--config", type=str, default="configs/diffusion_toy.yaml")
    args = ap.parse_args()

    cfg_yaml = {}
    if os.path.isfile(args.config):
        import yaml

        cfg_yaml = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    torch.manual_seed(int(cfg_yaml.get("seed", 2024)))
    data_cfg = ToyDiffConfig(
        n_samples=int(cfg_yaml.get("n_samples", 1000)),
        seq_len=int(cfg_yaml.get("seq_len", 16)),
        dim=int(cfg_yaml.get("dim", 8)),
        batch_size=int(cfg_yaml.get("batch_size", 64)),
    )
    train_loader, val_loader = make_toy_continuous_sequences(data_cfg)
    train_data = torch.cat([b[0] for b in train_loader], dim=0)
    val_data = torch.cat([b[0] for b in val_loader], dim=0)

    train_ids = [make_id_sequence(x) for x in train_data]
    val_ids = [make_id_sequence(x) for x in val_data]
    train_bank = build_windowed_bank_from_ids(train_ids, window=args.window, stride=args.stride).to(args.device)
    val_bank = build_windowed_bank_from_ids(val_ids, window=args.window, stride=args.stride).to(args.device)

    vocab_size = int(max(train_bank.values.max(), val_bank.values.max()).item() + 1) if train_bank.values.numel() > 0 else data_cfg.seq_len * 2
    atom_emb = nn.Embedding(vocab_size, args.d_model).to(args.device)
    adapter = None
    if args.adapter_rank > 0:
        adapter = AtomFeatureAdapter(args.d_model, rank=args.adapter_rank).to(args.device)
        with torch.no_grad():
            phi_snapshot = adapter(atom_emb.weight).detach()
    else:
        phi_snapshot = atom_emb.weight

    train_cache = SetBankCache(phi_snapshot, minhash_k=args.minhash_k)
    train_cache.precompute(train_bank.values, train_bank.set_offsets)
    val_cache = SetBankCache(phi_snapshot, minhash_k=args.minhash_k)
    val_cache.precompute(val_bank.values, val_bank.set_offsets)

    model = BankedDenoiser(in_dim=data_cfg.dim, d_model=args.d_model, nhead=args.nhead, layers=args.layers, router_topk=args.router_topk).to(args.device)
    ddpm = SimpleDDPM(T=args.steps, device=args.device)
    params = list(model.parameters()) + list(atom_emb.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        with profiler(args.profile) as prof:
            for batch_idx, xb in tensor_batch_iterator(train_data, data_cfg.batch_size, shuffle=True):
                xb = xb.to(args.device)
                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi, Sig, Size, mask = gather_bank_batch(train_bank, train_cache, batch_idx.to(args.device), phi_dynamic, adapter is not None, args.d_model, args.minhash_k)
                model.set_current_bank(Phi, Sig, Size, mask)
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
                xb = xb.to(args.device)
                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi, Sig, Size, mask = gather_bank_batch(val_bank, val_cache, batch_idx.to(args.device), phi_dynamic, adapter is not None, args.d_model, args.minhash_k)
                model.set_current_bank(Phi, Sig, Size, mask)
                val_mmd, val_chamfer, val_nn1 = eval_suite(ddpm, model, xb, args.d_model)
                mmds.append(val_mmd)
                chamfers.append(val_chamfer)
                nn1s.append(val_nn1)
        msg = f"[Diffusion-Banked] epoch {epoch:02d} train loss {train_loss:.4f} | val MMD {sum(mmds)/len(mmds):.4f} | Chamfer {sum(chamfers)/len(chamfers):.4f} | 1NN {sum(nn1s)/len(nn1s):.3f}"
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
