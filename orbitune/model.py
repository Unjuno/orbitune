from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class OrbituneConfig:
    vocab_size: int
    max_seq_len: int = 512
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: OrbituneConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, cfg: OrbituneConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class OrbituneGPT(nn.Module):
    architecture = "orbitune-midi-gpt-v0"

    def __init__(self, cfg: OrbituneConfig) -> None:
        super().__init__()
        self.config = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        _, t = input_ids.shape
        if t > self.config.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.config.max_seq_len}")
        pos = torch.arange(t, device=input_ids.device)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int, temperature: float = 0.85, top_p: float = 0.92, eos_id: int | None = None) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = sorted_probs.cumsum(dim=-1)
                remove = cumulative - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_sorted = torch.multinomial(sorted_probs, 1)
                next_id = sorted_idx.gather(-1, next_sorted)
            else:
                next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_id is not None and bool(torch.all(next_id == eos_id)):
                break
        return input_ids

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"architecture": self.architecture, "config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "OrbituneGPT":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("architecture") != cls.architecture:
            raise ValueError("checkpoint architecture mismatch")
        model = cls(OrbituneConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model
