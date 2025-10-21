from __future__ import annotations
import torch
import torch.nn.functional as F


def cosine_beta_schedule(T: int, s: float = 0.008, device=None):
    steps = torch.arange(T + 1, dtype=torch.float32, device=device)
    f = torch.cos(((steps / T) + s) / (1 + s) * 3.14159265 / 2) ** 2
    alphas = f / f[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return betas.clamp(1e-5, 0.02)


class SimpleDDPM:
    def __init__(self, T: int = 100, device=None):
        self.device = device
        self.T = T
        self.betas = cosine_beta_schedule(T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar_t = self.alphas_bar[t].view(-1, 1, 1).to(x0.device)
        return a_bar_t.sqrt() * x0 + (1 - a_bar_t).sqrt() * noise, noise

    def loss(self, model, x0: torch.Tensor, t_embed_fn, d_model: int = 64):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, eps = self.q_sample(x0, t)
        t_emb = t_embed_fn(t, d_model)
        pred = model(x_t, t_emb)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def sample(self, model, shape, t_embed_fn, d_model: int = 64):
        x = torch.randn(shape, device=self.betas.device)
        for ti in reversed(range(self.T)):
            t = torch.full((shape[0],), ti, device=x.device, dtype=torch.long)
            t_emb = t_embed_fn(t, d_model)
            eps = model(x, t_emb)
            beta_t = self.betas[ti]
            alpha_t = self.alphas[ti]
            alpha_bar_t = self.alphas_bar[ti]
            x = (1.0 / alpha_t.sqrt()) * (x - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * eps)
            if ti > 0:
                x = x + beta_t.sqrt() * torch.randn_like(x)
        return x

