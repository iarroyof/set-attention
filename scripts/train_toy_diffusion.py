import os
import argparse
import yaml
import torch
from set_attention.patch import replace_multihead_attn
from set_attention.experiments.data_toy import ToyDiffConfig, make_toy_continuous_sequences
from set_attention.experiments.models import TinyTransformerDenoiser, timestep_embedding
from set_attention.experiments.diffusion_core import SimpleDDPM
from set_attention.eval.mmd_simple import mmd2_unbiased_from_feats
from set_attention.utils.metrics import chamfer_l2, one_nn_two_sample
from set_attention.utils.profiling import profiler


def train_epoch(ddpm: SimpleDDPM, model, opt, loader, device, d_model: int):
    model.train()
    total, n = 0.0, 0
    for (xb,) in loader:
        xb = xb.to(device)
        loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, d_model), d_model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)


@torch.no_grad()
def eval_suite(ddpm: SimpleDDPM, model, val_loader, device, n_gen: int = 256, d_model: int = 64):
    model.eval()
    # Gather a validation batch
    Xs = []
    for (xb,) in val_loader:
        Xs.append(xb)
        if sum(x.shape[0] for x in Xs) >= n_gen:
            break
    X = torch.cat(Xs, dim=0)[:n_gen].to(device)
    # Generate samples and compute MMD on flattened vectors
    shape = (n_gen, X.shape[1], X.shape[2])
    x_gen = ddpm.sample(model, shape, lambda t, dim: timestep_embedding(t, d_model), d_model)
    x_flat = X.contiguous().view(n_gen, -1)
    g_flat = x_gen.contiguous().view(n_gen, -1)
    # Align feature dims if they somehow differ (robustness)
    if x_flat.shape[1] != g_flat.shape[1]:
        d = min(x_flat.shape[1], g_flat.shape[1])
        x_flat = x_flat[:, :d]
        g_flat = g_flat[:, :d]
    mmd = float(mmd2_unbiased_from_feats(x_flat, g_flat, gamma=0.5).detach().cpu())
    chamfer = chamfer_l2(X, x_gen)
    nn1 = one_nn_two_sample(x_flat, g_flat)
    return mmd, chamfer, nn1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska", "ska_intersect", "ska_true"], default="dot")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--inter-topk", type=int, default=16)
    ap.add_argument("--inter-norm", type=int, default=1)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--config", type=str, default="configs/diffusion_toy.yaml")
    ap.add_argument("--outdir", type=str, default="runs/diffusion_toy")
    args = ap.parse_args()

    cfg_yaml = yaml.safe_load(open(args.config, "r", encoding="utf-8")) if os.path.isfile(args.config) else {}
    torch.manual_seed(int(cfg_yaml.get("seed", args.seed)))
    data_cfg = ToyDiffConfig(n_samples=int(cfg_yaml.get("n_samples", 1000)),
                             seq_len=int(cfg_yaml.get("seq_len", 16)),
                             dim=int(cfg_yaml.get("dim", 8)),
                             batch_size=int(cfg_yaml.get("batch_size", 64)))
    train_loader, val_loader = make_toy_continuous_sequences(data_cfg)

    d_model = args.d_model
    model = TinyTransformerDenoiser(d_model=d_model, nhead=args.nhead, num_layers=args.layers, in_dim=data_cfg.dim)
    sim_mode = args.attn
    if sim_mode == "ska":
        sim_mode = "rbf"
    elif sim_mode == "ska_intersect":
        sim_mode = "intersect"
    if sim_mode != "dot":
        replace_multihead_attn(
            model,
            sim=sim_mode,
            rbf_gamma=args.gamma,
            temperature=args.temperature,
            inter_topk=args.inter_topk,
            inter_normalize=bool(args.inter_norm),
        )

    device = torch.device(args.device)
    model.to(device)
    os.makedirs(args.outdir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ddpm = SimpleDDPM(T=args.steps, device=device)

    for epoch in range(1, args.epochs + 1):
        with profiler(args.profile) as prof:
            tr_loss = train_epoch(ddpm, model, opt, train_loader, device, d_model)
            val_mmd, val_chamfer, val_nn1 = eval_suite(ddpm, model, val_loader, device, n_gen=256, d_model=d_model)
        msg = f"[Diffusion][{args.attn}] epoch {epoch:02d} | train loss {tr_loss:.4f} | val MMD {val_mmd:.4f} | Chamfer {val_chamfer:.4f} | 1NN {val_nn1:.3f}"
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
            if prof.get("gpu_power_w") is not None:
                msg += f" | power {prof['gpu_power_w']:.1f} W"
        print(msg)

    if os.getenv("WANDB_PROJECT"):
        try:
            import wandb

            wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True, config={
                "attn": args.attn, "lr": args.lr, "d_model": args.d_model, "nhead": args.nhead,
                "layers": args.layers, "gamma": args.gamma, "temperature": args.temperature,
                "steps": args.steps,
            })
            log = {"val/mmd": val_mmd, "val/chamfer": val_chamfer, "val/nn1": val_nn1}
            if args.profile:
                log.update({
                    "time/epoch_s": prof["time_s"],
                    "gpu/peak_mem_mib": prof["gpu_peak_mem_mib"],
                    "gpu/power_w": prof.get("gpu_power_w"),
                })
            wandb.log(log)
            wandb.finish()
        except Exception as e:
            print("WandB logging skipped:", e)


if __name__ == "__main__":
    main()
