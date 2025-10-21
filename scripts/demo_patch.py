import os
import torch
from torch import nn

from set_attention.patch import replace_multihead_attn


def main():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
    n = replace_multihead_attn(layer, sim="rbf", rbf_gamma=0.3)
    print(f"Replaced {n} MultiheadAttention modules")

    x = torch.randn(2, 16, 128)
    y = layer(x)
    print("Output shape:", y.shape)

    # Optional W&B quick log
    if os.getenv("WANDB_PROJECT"):
        try:
            import wandb

            wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
            wandb.log({"demo/mean_abs": y.abs().mean().item()})
            wandb.finish()
            print("Logged to Weights & Biases")
        except Exception as e:
            print("WandB logging skipped:", e)


if __name__ == "__main__":
    main()

