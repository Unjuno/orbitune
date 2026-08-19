from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(slots=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.dropout(x) @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + delta * self.scaling


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in cfg.target_modules and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout))
                replaced.append(f"{module_name}.{child_name}".lstrip("."))
    if not replaced:
        raise ValueError(f"no target modules found: {cfg.target_modules}")
    return replaced


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.endswith("lora_a") or name.endswith("lora_b")
    }


def save_adapter(model: nn.Module, path: str | Path, cfg: LoRAConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "orbitune-lora-v0",
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "dropout": cfg.dropout,
            "target_modules": list(cfg.target_modules),
            "state_dict": adapter_state_dict(model),
        },
        path,
    )


def load_adapter(model: nn.Module, path: str | Path, *, map_location: str | torch.device = "cpu") -> LoRAConfig:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != "orbitune-lora-v0":
        raise ValueError("unsupported adapter format")
    cfg = LoRAConfig(
        rank=int(payload["rank"]),
        alpha=float(payload["alpha"]),
        dropout=float(payload.get("dropout", 0.0)),
        target_modules=tuple(payload["target_modules"]),
    )
    inject_lora(model, cfg)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    unexpected = [name for name in unexpected if "lora_" in name]
    missing_lora = [name for name in missing if "lora_" in name]
    if unexpected or missing_lora:
        raise ValueError(f"adapter state mismatch: missing={missing_lora}, unexpected={unexpected}")
    return cfg
