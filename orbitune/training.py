from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from orbitune.lora import LoRAConfig, inject_lora, save_adapter, trainable_parameter_count
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab


@dataclass(slots=True)
class TrainConfig:
    steps: int = 100
    batch_size: int = 8
    seq_len: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    device: str = "cpu"
    seed: int = 1234


def read_token_ids(paths: list[str | Path], vocab: TheoryRemiVocab) -> list[int]:
    ids: list[int] = []
    for path in paths:
        tokens = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        ids.extend(vocab.encode(tokens, add_special_tokens=True))
    if len(ids) < 3:
        raise ValueError("training corpus is too small")
    return ids


def _sample_batch(ids: list[int], *, batch_size: int, seq_len: int, device: torch.device, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    usable = min(seq_len, len(ids) - 1)
    if usable < 2:
        raise ValueError("not enough tokens for a training batch")
    starts_max = len(ids) - usable - 1
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    for _ in range(batch_size):
        start = rng.randint(0, max(0, starts_max))
        xs.append(ids[start : start + usable])
        ys.append(ids[start + 1 : start + usable + 1])
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


def train_model(model: OrbituneGPT, ids: list[int], cfg: TrainConfig, *, trainable_only: bool = False) -> list[float]:
    device = torch.device(cfg.device)
    model.to(device)
    model.train()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    parameters = [p for p in model.parameters() if p.requires_grad] if trainable_only else list(model.parameters())
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    losses: list[float] = []
    for _ in range(cfg.steps):
        x, y = _sample_batch(ids, batch_size=cfg.batch_size, seq_len=cfg.seq_len, device=device, rng=rng)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def train_base(token_paths: list[str | Path], out: str | Path, *, model_cfg: OrbituneConfig, train_cfg: TrainConfig) -> dict[str, float | int]:
    vocab = TheoryRemiVocab()
    if model_cfg.vocab_size != len(vocab):
        raise ValueError(f"model vocab_size={model_cfg.vocab_size} != Theory-REMI vocab size={len(vocab)}")
    ids = read_token_ids(token_paths, vocab)
    model = OrbituneGPT(model_cfg)
    losses = train_model(model, ids, train_cfg)
    model.save_checkpoint(out)
    return {"parameters": model.parameter_count(), "tokens": len(ids), "steps": len(losses), "initial_loss": losses[0], "final_loss": losses[-1]}


def train_adapter(base: str | Path, token_paths: list[str | Path], out: str | Path, *, lora_cfg: LoRAConfig, train_cfg: TrainConfig) -> dict[str, float | int]:
    vocab = TheoryRemiVocab()
    ids = read_token_ids(token_paths, vocab)
    model = OrbituneGPT.load_checkpoint(base)
    inject_lora(model, lora_cfg)
    losses = train_model(model, ids, train_cfg, trainable_only=True)
    save_adapter(model, out, lora_cfg)
    return {"trainable_parameters": trainable_parameter_count(model), "tokens": len(ids), "steps": len(losses), "initial_loss": losses[0], "final_loss": losses[-1]}
