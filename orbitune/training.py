from __future__ import annotations

import copy
import random
import time
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
    validation_interval: int = 0


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


def _optimizer_parameters(model: OrbituneGPT, *, trainable_only: bool) -> list[torch.nn.Parameter]:
    parameters = [p for p in model.parameters() if p.requires_grad] if trainable_only else list(model.parameters())
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def train_model(model: OrbituneGPT, ids: list[int], cfg: TrainConfig, *, trainable_only: bool = False) -> list[float]:
    device = torch.device(cfg.device)
    model.to(device)
    model.train()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    parameters = _optimizer_parameters(model, trainable_only=trainable_only)
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


@torch.no_grad()
def evaluate_token_loss(
    model: OrbituneGPT,
    ids: list[int],
    *,
    seq_len: int,
    device: str = "cpu",
    max_windows: int = 32,
) -> float:
    if len(ids) < 3:
        raise ValueError("validation corpus is too small")
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()
    usable = min(seq_len, model.config.max_seq_len, len(ids) - 1)
    if usable < 2:
        raise ValueError("validation sequence is too short")

    stride = usable
    losses: list[float] = []
    for start in range(0, len(ids) - 1, stride):
        if len(losses) >= max_windows:
            break
        end = min(start + usable, len(ids) - 1)
        if end - start < 2:
            break
        x = torch.tensor([ids[start:end]], dtype=torch.long, device=device_obj)
        y = torch.tensor([ids[start + 1 : end + 1]], dtype=torch.long, device=device_obj)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("validation corpus produced no evaluation windows")
    return sum(losses) / len(losses)


def _train_model_tracked(
    model: OrbituneGPT,
    ids: list[int],
    validation_ids: list[int],
    cfg: TrainConfig,
    *,
    trainable_only: bool,
) -> tuple[list[float], list[dict[str, float | int]], int, float]:
    if cfg.validation_interval <= 0:
        losses = train_model(model, ids, cfg, trainable_only=trainable_only)
        validation_loss = evaluate_token_loss(
            model,
            validation_ids,
            seq_len=cfg.seq_len,
            device=cfg.device,
        )
        return losses, [{"step": cfg.steps, "validation_loss": validation_loss}], cfg.steps, validation_loss

    device = torch.device(cfg.device)
    model.to(device)
    model.train()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    parameters = _optimizer_parameters(model, trainable_only=trainable_only)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    losses: list[float] = []
    history: list[dict[str, float | int]] = []
    best_step = 0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for step in range(1, cfg.steps + 1):
        model.train()
        x, y = _sample_batch(ids, batch_size=cfg.batch_size, seq_len=cfg.seq_len, device=device, rng=rng)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        should_validate = step % cfg.validation_interval == 0 or step == cfg.steps
        if should_validate:
            validation_loss = evaluate_token_loss(
                model,
                validation_ids,
                seq_len=cfg.seq_len,
                device=cfg.device,
            )
            history.append({"step": step, "validation_loss": validation_loss})
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("periodic validation produced no checkpoint candidate")
    model.load_state_dict(best_state)
    model.eval()
    return losses, history, best_step, best_loss


def _training_metrics(losses: list[float], ids: list[int], cfg: TrainConfig, elapsed_seconds: float) -> dict[str, float | int]:
    usable = min(cfg.seq_len, len(ids) - 1)
    processed_tokens = cfg.steps * cfg.batch_size * usable
    return {
        "tokens": len(ids),
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "elapsed_seconds": elapsed_seconds,
        "processed_tokens": processed_tokens,
        "tokens_per_second": processed_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0,
    }


def train_base(
    token_paths: list[str | Path],
    out: str | Path,
    *,
    model_cfg: OrbituneConfig,
    train_cfg: TrainConfig,
    validation_token_paths: list[str | Path] | None = None,
) -> dict[str, object]:
    vocab = TheoryRemiVocab()
    if model_cfg.vocab_size != len(vocab):
        raise ValueError(f"model vocab_size={model_cfg.vocab_size} != Theory-REMI vocab size={len(vocab)}")
    ids = read_token_ids(token_paths, vocab)
    # The training seed must cover parameter initialization as well as batch
    # sampling/dropout. Seeding only inside train_model() is too late because
    # the Base weights have already been created by then.
    torch.manual_seed(train_cfg.seed)
    model = OrbituneGPT(model_cfg)
    validation_ids = read_token_ids(validation_token_paths, vocab) if validation_token_paths else None
    start = time.perf_counter()
    if validation_ids is not None:
        losses, history, best_step, best_loss = _train_model_tracked(
            model,
            ids,
            validation_ids,
            train_cfg,
            trainable_only=False,
        )
    else:
        losses = train_model(model, ids, train_cfg)
        history = []
        best_step = train_cfg.steps
        best_loss = float("nan")
    elapsed = time.perf_counter() - start
    model.save_checkpoint(out)
    report: dict[str, object] = _training_metrics(losses, ids, train_cfg, elapsed)
    report["parameters"] = model.parameter_count()
    if validation_ids is not None:
        report["validation_tokens"] = len(validation_ids)
        report["validation_loss"] = best_loss
        report["best_validation_step"] = best_step
        report["validation_history"] = history
        report["checkpoint_selection"] = "minimum_validation_loss"
    else:
        report["checkpoint_selection"] = "final_step"
    return report


def train_adapter(
    base: str | Path,
    token_paths: list[str | Path],
    out: str | Path,
    *,
    lora_cfg: LoRAConfig,
    train_cfg: TrainConfig,
    validation_token_paths: list[str | Path] | None = None,
) -> dict[str, object]:
    vocab = TheoryRemiVocab()
    ids = read_token_ids(token_paths, vocab)
    validation_ids = read_token_ids(validation_token_paths, vocab) if validation_token_paths else None
    model = OrbituneGPT.load_checkpoint(base)
    # LoRA A uses random Kaiming initialization, so seed before injection.
    torch.manual_seed(train_cfg.seed)
    inject_lora(model, lora_cfg)
    start = time.perf_counter()
    if validation_ids is not None:
        losses, history, best_step, best_loss = _train_model_tracked(
            model,
            ids,
            validation_ids,
            train_cfg,
            trainable_only=True,
        )
    else:
        losses = train_model(model, ids, train_cfg, trainable_only=True)
        history = []
        best_step = train_cfg.steps
        best_loss = float("nan")
    elapsed = time.perf_counter() - start
    save_adapter(model, out, lora_cfg)
    report: dict[str, object] = _training_metrics(losses, ids, train_cfg, elapsed)
    report["trainable_parameters"] = trainable_parameter_count(model)
    if validation_ids is not None:
        report["validation_tokens"] = len(validation_ids)
        report["validation_loss"] = best_loss
        report["best_validation_step"] = best_step
        report["validation_history"] = history
        report["checkpoint_selection"] = "minimum_validation_loss"
    else:
        report["checkpoint_selection"] = "final_step"
    return report