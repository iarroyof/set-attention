import os
import argparse
import yaml
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from set_attention.patch import replace_multihead_attn
from set_attention.experiments.data_toy import ToySeqConfig, make_toy_sequence_classification
from set_attention.experiments.models import TinyTransformerClassifier
from set_attention.utils.metrics import confusion_matrix, ece, binary_roc_auc, binary_pr_auc
from set_attention.utils.profiling import profiler


def train_epoch(model, opt, loader, device):
    model.train()
    total, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * xb.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        n += xb.size(0)
    return total / max(1, n), correct / max(1, n)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total += float(loss.detach()) * xb.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        n += xb.size(0)
    return total / max(1, n), correct / max(1, n)


def load_config(path: str | None):
    if path and os.path.isfile(path):
        return yaml.safe_load(open(path, "r", encoding="utf-8"))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska", "ska_intersect", "ska_true"], default="dot")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--inter-topk", type=int, default=16)
    ap.add_argument("--inter-norm", type=int, default=1, help="1 to normalize overlap by topk, 0 otherwise")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--config", type=str, default="configs/transformer_toy.yaml")
    ap.add_argument("--outdir", type=str, default="runs/transformer_toy")
    args = ap.parse_args()

    cfg_yaml = load_config(args.config)
    torch.manual_seed(int(cfg_yaml.get("seed", args.seed)))
    data_cfg = ToySeqConfig(n_samples=int(cfg_yaml.get("n_samples", 1000)),
                            seq_len=int(cfg_yaml.get("seq_len", 32)),
                            vocab_size=int(cfg_yaml.get("vocab_size", 64)),
                            batch_size=int(cfg_yaml.get("batch_size", args.batch)))
    train_loader, val_loader = make_toy_sequence_classification(data_cfg)

    model = TinyTransformerClassifier(vocab_size=data_cfg.vocab_size,
                                      d_model=args.d_model, nhead=args.nhead, num_layers=args.layers)
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

    for epoch in range(1, args.epochs + 1):
        with profiler(args.profile) as prof:
            tr_loss, tr_acc = train_epoch(model, opt, train_loader, device)
            va_loss, va_acc = eval_epoch(model, val_loader, device)
        # extra metrics on val
        y_true, y_prob = [], []
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
            y_true.append(yb)
            y_prob.append(prob1)
        y_true = torch.cat(y_true)
        y_prob = torch.cat(y_prob)
        y_pred = (y_prob >= 0.5).long()
        cm = confusion_matrix(y_pred, y_true, num_classes=2).cpu().numpy()
        val_ece = float(ece(y_prob, y_true).cpu())
        roc = binary_roc_auc(y_prob, y_true)
        pr = binary_pr_auc(y_prob, y_true)

        msg = f"[Transformer][{args.attn}] epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} | ECE {val_ece:.3f} | ROC {roc:.3f} | PR {pr:.3f}"
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
            if prof.get("gpu_power_w") is not None:
                msg += f" | power {prof['gpu_power_w']:.1f} W"
        print(msg)

        # save confusion matrix CSV
        with open(os.path.join(args.outdir, f"cm_epoch{epoch:02d}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tn", "fp"]) ; w.writerow(list(cm[0]))
            w.writerow(["fn", "tp"]) ; w.writerow(list(cm[1]))

    if os.getenv("WANDB_PROJECT"):
        try:
            import wandb

            wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True, config={
                "attn": args.attn, "lr": args.lr, "d_model": args.d_model, "nhead": args.nhead,
                "layers": args.layers, "gamma": args.gamma, "temperature": args.temperature,
            })
            log = {"val/acc": va_acc, "val/loss": va_loss, "val/ece": val_ece, "val/roc_auc": roc, "val/pr_auc": pr}
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
