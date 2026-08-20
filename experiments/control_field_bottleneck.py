from __future__ import annotations

"""Model-free bottleneck experiment for compact ControlField inputs.

Eight synthetic control features are compressed to d dimensions, then expanded by a
small nonlinear decoder into 12 adaptive Gaussian RBFs. The experiment estimates the
minimum compact control dimensionality before long-range curve reconstruction degrades.
"""

import math
import statistics
import torch
from torch import nn
from torch.nn import functional as F

K = 12
T = 128


def dataset(n: int = 1200, seed: int = 123):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, T)
    trend = (torch.rand(n, 1, generator=g) * 2 - 1) * 0.7
    wave_amp = torch.rand(n, 1, generator=g) * 0.8
    wave_freq = torch.randint(1, 5, (n, 1), generator=g).float()
    phase = torch.rand(n, 1, generator=g) * 2 * math.pi
    bump_amp = torch.rand(n, 1, generator=g) * 2 - 1
    bump_center = 0.1 + 0.8 * torch.rand(n, 1, generator=g)
    bump_width = 0.02 + 0.12 * torch.rand(n, 1, generator=g)
    x = torch.cat([trend, wave_amp, wave_freq / 4, torch.sin(phase), torch.cos(phase), bump_amp, bump_center, bump_width], 1)
    y = (
        trend * (t - 0.5)
        + wave_amp * torch.sin(2 * math.pi * wave_freq * t + phase)
        + bump_amp * torch.exp(-0.5 * ((t - bump_center) / bump_width) ** 2)
    )
    perm = torch.randperm(n, generator=g)
    return t, x[perm[:900]], y[perm[:900]], x[perm[900:]], y[perm[900:]]


class BottleneckRBF(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, dim))
        self.decoder = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3 * K))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        z = self.encoder(x)
        center_raw, width_raw, amplitude = self.decoder(z).split(K, -1)
        centers = torch.sigmoid(center_raw)
        widths = 0.015 + 0.25 * torch.sigmoid(width_raw)
        basis = torch.exp(-0.5 * ((t[None, None, :] - centers[:, :, None]) / widths[:, :, None]) ** 2)
        return (amplitude[:, :, None] * basis).sum(1), z


def run(dim: int, seed: int, steps: int = 1000):
    t, xtr, ytr, xva, yva = dataset()
    torch.manual_seed(seed)
    model = BottleneckRBF(dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 500)
    for _ in range(steps):
        idx = torch.randint(0, len(xtr), (64,), generator=g)
        pred, z = model(xtr[idx], t)
        loss = F.mse_loss(pred, ytr[idx]) + 1e-5 * z.square().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        pred, _ = model(xva, t)
        mse = ((pred - yva) ** 2).mean(1)
        sharp = xva[:, -1] < 0.05
        return mse.mean().item(), mse[sharp].mean().item()


if __name__ == "__main__":
    torch.set_num_threads(5)
    for dim in (4, 5, 6, 7, 8):
        values = [run(dim, seed) for seed in (0, 1, 2)]
        print(dim, "mse", statistics.mean(v[0] for v in values), "+/-", statistics.stdev(v[0] for v in values), "sharp", statistics.mean(v[1] for v in values))
