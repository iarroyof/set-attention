import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from set_attention.patch import replace_multihead_attn
from set_attention.experiments.data_toy import ToySeqConfig, make_toy_sequence_classification
from set_attention.experiments.models import TinyTransformerClassifier


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf"], default="dot")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = ToySeqConfig(n_samples=1000)
    train_loader, val_loader = make_toy_sequence_classification(cfg)

    model = TinyTransformerClassifier(vocab_size=cfg.vocab_size, d_model=64, nhead=2, num_layers=2)
    if args.attn != "dot":
        # Replace dot-product attention with kernel attention
        replace_multihead_attn(model, sim=args.attn, rbf_gamma=0.5, temperature=1.0)

    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, opt, train_loader, device)
        va_loss, va_acc = eval_epoch(model, val_loader, device)
        print(f"[Transformer][{args.attn}] epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    if os.getenv("WANDB_PROJECT"):
        try:
            import wandb

            wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True, config={"attn": args.attn})
            wandb.log({"val/acc": va_acc, "val/loss": va_loss})
            wandb.finish()
        except Exception as e:
            print("WandB logging skipped:", e)


if __name__ == "__main__":
    main()

