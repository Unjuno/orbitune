from __future__ import annotations

"""Model-free ablation for adaptive RBF ControlField parameterization.

Compares which RBF degrees of freedom should remain adaptive:
- center + width + amplitude
- width + amplitude (fixed centers)
- center + amplitude (fixed widths)
- amplitude only

Also compares unconstrained centers against structurally monotonic centers and verifies
that post-hoc sorting of (center, width, amplitude) tuples leaves the represented field
unchanged up to floating-point summation order.
"""

import math
import statistics
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

K = 12
T = 128


def make_dataset(n: int = 1200, seed: int = 123):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, T)
    trend = (torch.rand(n, 1, generator=g) * 2 - 1) * 0.7
    wave_amp = torch.rand(n, 1, generator=g) * 0.8
    wave_freq = torch.randint(1, 5, (n, 1), generator=g).float()
    phase = torch.rand(n, 1, generator=g) * 2 * math.pi
    bump_amp = (torch.rand(n, 1, generator=g) * 2 - 1)
    bump_center = 0.1 + 0.8 * torch.rand(n, 1, generator=g)
    bump_width = 0.02 + 0.12 * torch.rand(n, 1, generator=g)
    x = torch.cat(
        [trend, wave_amp, wave_freq / 4, torch.sin(phase), torch.cos(phase), bump_amp, bump_center, bump_width],
        dim=1,
    )
    y = (
        trend * (t - 0.5)
        + wave_amp * torch.sin(2 * math.pi * wave_freq * t + phase)
        + bump_amp * torch.exp(-0.5 * ((t - bump_center) / bump_width) ** 2)
    )
    perm = torch.randperm(n, generator=g)
    return t, x[perm[:900]], y[perm[:900]], x[perm[900:]], y[perm[900:]]


class Controller(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        out = {"all": 3 * K, "width_amp": 2 * K, "center_amp": 2 * K, "amp": K}[mode]
        self.net = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, out))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        raw = self.net(x)
        b = x.shape[0]
        fixed_centers = torch.linspace(0.03, 0.97, K, device=x.device).expand(b, -1)
        fixed_widths = torch.full((b, K), 0.09, device=x.device)
        if self.mode == "all":
            c_raw, w_raw, amp = raw.split(K, dim=-1)
            centers = torch.sigmoid(c_raw)
            widths = 0.015 + 0.25 * torch.sigmoid(w_raw)
        elif self.mode == "width_amp":
            w_raw, amp = raw.split(K, dim=-1)
            centers = fixed_centers
            widths = 0.015 + 0.25 * torch.sigmoid(w_raw)
        elif self.mode == "center_amp":
            c_raw, amp = raw.split(K, dim=-1)
            centers = torch.sigmoid(c_raw)
            widths = fixed_widths
        else:
            amp = raw
            centers = fixed_centers
            widths = fixed_widths
        basis = torch.exp(-0.5 * ((t[None, None, :] - centers[:, :, None]) / widths[:, :, None]) ** 2)
        return (amp[:, :, None] * basis).sum(dim=1), centers, widths, amp


def train(mode: str, seed: int, steps: int = 500):
    t, xtr, ytr, xva, yva = make_dataset()
    torch.manual_seed(seed)
    model = Controller(mode)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 99)
    for _ in range(steps):
        idx = torch.randint(0, len(xtr), (64,), generator=g)
        pred, _, _, amp = model(xtr[idx], t)
        loss = F.mse_loss(pred, ytr[idx]) + 1e-5 * amp.square().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        pred, centers, widths, amp = model(xva, t)
        mse = ((pred - yva) ** 2).mean(dim=1)
        sharp = xva[:, -1] < 0.05
        return {
            "mode": mode,
            "seed": seed,
            "mse": mse.mean().item(),
            "sharp_mse": mse[sharp].mean().item(),
            "params": sum(p.numel() for p in model.parameters()),
            "crossing_rate": (centers[:, 1:] < centers[:, :-1]).float().mean().item(),
        }


def verify_sort_invariance(seed: int = 77):
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    centers = torch.rand(64, K)
    widths = 0.02 + 0.2 * torch.rand(64, K)
    amp = torch.randn(64, K)
    field = (amp[:, :, None] * torch.exp(-0.5 * ((t[None, None, :] - centers[:, :, None]) / widths[:, :, None]) ** 2)).sum(1)
    idx = torch.argsort(centers, dim=1)
    cs = torch.gather(centers, 1, idx)
    ws = torch.gather(widths, 1, idx)
    amps = torch.gather(amp, 1, idx)
    sorted_field = (amps[:, :, None] * torch.exp(-0.5 * ((t[None, None, :] - cs[:, :, None]) / ws[:, :, None]) ** 2)).sum(1)
    return (field - sorted_field).abs().max().item()


if __name__ == "__main__":
    torch.set_num_threads(5)
    rows = [train(mode, seed) for mode in ("all", "width_amp", "center_amp", "amp") for seed in range(5)]
    for mode in ("all", "width_amp", "center_amp", "amp"):
        values = [r["mse"] for r in rows if r["mode"] == mode]
        sharp = [r["sharp_mse"] for r in rows if r["mode"] == mode]
        params = next(r["params"] for r in rows if r["mode"] == mode)
        crossings = statistics.mean(r["crossing_rate"] for r in rows if r["mode"] == mode)
        print(mode, "mse", statistics.mean(values), "+/-", statistics.stdev(values), "sharp", statistics.mean(sharp), "params", params, "crossings", crossings)
    print("sort max abs diff", verify_sort_invariance())
