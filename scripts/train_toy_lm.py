import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from set_attention.patch import replace_multihead_attn
from set_attention.utils.profiling import profiler


def make_char_data(n=10000, seq_len=64, vocab=32, seed=3):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, vocab, (n, seq_len), generator=g)
    Y = torch.roll(X, shifts=-1, dims=1)  # next-char prediction
    return X, Y


class TinyLM(nn.Module):
    def __init__(self, vocab=32, d_model=128, nhead=4, layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.emb(x)
        h = self.enc(h)
        return self.proj(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska", "ska_intersect", "ska_true"], default="dot")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    X, Y = make_char_data(n=2000)
    model = TinyLM()
    sim_mode = args.attn
    if sim_mode == "ska":
        sim_mode = "rbf"
    elif sim_mode == "ska_intersect":
        sim_mode = "intersect"
    if sim_mode != "dot":
        replace_multihead_attn(model, sim=sim_mode)
    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for ep in range(1, args.epochs + 1):
        with profiler(args.profile) as prof:
            xb = X[:64].to(device)
            yb = Y[:64].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        msg = f"[LM][{args.attn}] epoch {ep:02d} loss {loss.item():.4f}"
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
            if prof.get("gpu_power_w") is not None:
                msg += f" | power {prof['gpu_power_w']:.1f} W"
        print(msg)


if __name__ == "__main__":
    main()
