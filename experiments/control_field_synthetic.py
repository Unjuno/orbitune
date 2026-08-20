from __future__ import annotations

"""Synthetic end-to-end ControlField experiment for Orbitune.

This is not a production training path. It verifies that a small learned control
function can modulate the ~10M reference Transformer through FiLM and that the
control can be learned from causal-LM loss alone.
"""

import json
import math
import random
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1234
N_TIME = 8
N_PITCH = 13
N_VEL = 8
TIME0 = 0
PITCH0 = N_TIME
VEL0 = N_TIME + N_PITCH
VOCAB = N_TIME + N_PITCH + N_VEL
REST_PITCH = PITCH0 + 12
SEQ = 48
N_CTRL = 4  # NONE, SLOW, BUILDUP, WAVE


@dataclass
class Config:
    vocab_size: int = VOCAB
    max_seq_len: int = 1024
    n_layer: int = 4
    n_embd: int = 448
    n_head: int = 7

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        x = (1 + gamma) * x + beta
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ControlFieldNet(nn.Module):
    def __init__(self, hidden: int = 448, basis: int = 8, channels: int = 3, emb: int = 32) -> None:
        super().__init__()
        self.basis = basis
        self.channels = channels
        self.embedding = nn.Embedding(N_CTRL, emb)
        self.net = nn.Sequential(nn.Linear(emb, 64), nn.GELU(), nn.Linear(64, basis * channels))
        self.to_gamma = nn.Linear(channels, hidden)
        self.to_beta = nn.Linear(channels, hidden)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def _basis(self, length: int, device: torch.device) -> torch.Tensor:
        x = torch.linspace(0, 1, length, device=device)
        values = [torch.ones_like(x)]
        for k in range(1, self.basis):
            values.append(torch.cos(math.pi * k * x))
        return torch.stack(values, dim=-1)

    def forward(self, control_ids: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coeff = self.net(self.embedding(control_ids)).view(-1, self.basis, self.channels)
        field = torch.einsum("tk,bkc->btc", self._basis(length, control_ids.device), coeff)
        # Structural guarantee: NONE means exactly zero modulation.
        field = field * (control_ids != 0).float()[:, None, None]
        return field, self.to_gamma(field), self.to_beta(field)


class Model(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.control = ControlFieldNet(hidden=cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, ids: torch.Tensor, controls: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = ids.shape
        pos = torch.arange(t, device=ids.device)
        x = self.token(ids) + self.position(pos)[None]
        field, gamma, beta = self.control(controls, t)
        for block in self.blocks:
            x = block(x, gamma, beta)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, field


def target_field(control: int, slots: int = SEQ // 3):
    x = torch.linspace(0, 1, slots)
    if control == 0:
        return torch.full_like(x, 0.5), torch.full_like(x, 0.65), torch.full_like(x, 0.5)
    if control == 1:
        return torch.full_like(x, 0.82), torch.full_like(x, 0.35), torch.full_like(x, 0.35)
    if control == 2:
        return 0.82 - 0.64 * x, 0.25 + 0.65 * x, 0.3 + 0.6 * x
    s = (torch.sin(2 * math.pi * x) + 1) / 2
    return 0.2 + 0.65 * s, 0.85 - 0.55 * s, 0.25 + 0.65 * (1 - s)


def make_sequence(control: int) -> list[int]:
    timing, density, velocity = target_field(control)
    out: list[int] = []
    for i in range(len(timing)):
        time_id = int(round(float(timing[i]) * (N_TIME - 1)))
        out.append(TIME0 + max(0, min(N_TIME - 1, time_id + random.choice([-1, 0, 0, 0, 1]))))
        pitch = random.randrange(12) if random.random() < float(density[i]) else 12
        out.append(PITCH0 + pitch)
        vel_id = int(round(float(velocity[i]) * (N_VEL - 1)))
        out.append(VEL0 + max(0, min(N_VEL - 1, vel_id + random.choice([-1, 0, 0, 0, 1]))))
    return out[:SEQ]


def batch(batch_size: int):
    controls = [random.randrange(N_CTRL) for _ in range(batch_size)]
    seqs = [make_sequence(c) for c in controls]
    x = torch.tensor([[0] + s[:-1] for s in seqs], dtype=torch.long)
    y = torch.tensor(seqs, dtype=torch.long)
    return x, y, torch.tensor(controls)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(5)
    model = Model(Config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    start = time.time()
    for step in range(151):
        x, y, controls = batch(4)
        _, loss, _ = model(x, controls, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0:
            print(step, float(loss.detach()))

    model.eval()
    diagnostics = {}
    with torch.no_grad():
        for control in range(N_CTRL):
            on_losses, off_losses, wave_corr = [], [], []
            for _ in range(10):
                seqs = [make_sequence(control) for _ in range(4)]
                x = torch.tensor([[0] + s[:-1] for s in seqs])
                y = torch.tensor(seqs)
                c = torch.full((4,), control, dtype=torch.long)
                logits, loss_on, field = model(x, c, y)
                _, loss_off, _ = model(x, torch.zeros_like(c), y)
                assert loss_on is not None and loss_off is not None
                on_losses.append(float(loss_on))
                off_losses.append(float(loss_off))
                if control == 0:
                    assert field.abs().max().item() == 0.0
                if control == 3:
                    pred = logits.argmax(-1)
                    time_curve = ((pred[:, 0::3] - TIME0).clamp(0, N_TIME - 1).float() / (N_TIME - 1)).mean(0)
                    target, _, _ = target_field(control)
                    wave_corr.append(float(torch.corrcoef(torch.stack([time_curve, target]))[0, 1]))
            diagnostics[control] = {
                "loss_control_on": statistics.mean(on_losses),
                "loss_control_forced_none": statistics.mean(off_losses),
                "loss_penalty_without_control": statistics.mean(off_losses) - statistics.mean(on_losses),
                "wave_time_corr": statistics.mean(wave_corr) if wave_corr else None,
            }
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "control_parameters": sum(p.numel() for p in model.control.parameters()), "elapsed_seconds": time.time() - start, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
