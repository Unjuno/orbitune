from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_STATE = 4
D_MODEL = 32
SLOTS = 6
BANK_SLOTS = 2
SEQ_LEN = 256
SLOW_BASE = 0
MEDIUM_BASE = N_STATE
FAST_BASE = 2 * N_STATE
FILL_BASE = 3 * N_STATE
VOCAB = FILL_BASE + 16


def make_batch(batch: int, device: torch.device, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    slow = torch.randint(0, N_STATE, (batch,), generator=generator)
    medium = torch.randint(0, N_STATE, (batch,), generator=generator)
    fast = torch.randint(0, N_STATE, (batch,), generator=generator)
    ids = torch.empty((batch, SEQ_LEN), dtype=torch.long)
    slow_targets = torch.empty_like(ids)
    medium_targets = torch.empty_like(ids)
    fast_targets = torch.empty_like(ids)
    for row in range(batch):
        s = int(slow[row])
        m = int(medium[row])
        f = int(fast[row])
        for step in range(SEQ_LEN):
            if step == 0:
                token = SLOW_BASE + s
            elif step % 96 == 0:
                m = int(torch.randint(0, N_STATE, (1,), generator=generator))
                token = MEDIUM_BASE + m
            elif step % 24 == 0:
                f = int(torch.randint(0, N_STATE, (1,), generator=generator))
                token = FAST_BASE + f
            else:
                token = FILL_BASE + ((step + 3 * row) % 16)
            ids[row, step] = token
            slow_targets[row, step] = s
            medium_targets[row, step] = m
            fast_targets[row, step] = f
    return (
        ids.to(device),
        slow_targets.to(device),
        medium_targets.to(device),
        fast_targets.to(device),
    )


class RoutedMemoryBank(nn.Module):
    def __init__(self, slots: int, decay: float | None = None) -> None:
        super().__init__()
        self.slots = slots
        self.q = nn.Linear(D_MODEL, slots, bias=False)
        self.k = nn.Linear(D_MODEL, slots, bias=False)
        self.v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.write = nn.Linear(D_MODEL, 1)
        if decay is None:
            self.logit_decay = nn.Parameter(torch.full((slots,), 4.0))
            self.register_buffer("fixed_decay", None)
        else:
            self.logit_decay = None
            self.register_buffer("fixed_decay", torch.full((slots,), float(decay)))

    def decay(self) -> torch.Tensor:
        if self.logit_decay is None:
            assert self.fixed_decay is not None
            return self.fixed_decay
        return torch.sigmoid(self.logit_decay).clamp(0.90, 0.9999)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = F.elu(self.q(hidden)) + 1.0
        key = F.elu(self.k(hidden)) + 1.0
        value = self.v(hidden)
        write = torch.sigmoid(self.write(hidden))
        decay = self.decay()
        steps = hidden.shape[1]
        index = torch.arange(steps, device=hidden.device, dtype=hidden.dtype)
        inverse = decay[None, :].pow(-index[:, None])
        forward = decay[None, :].pow(index[:, None])
        contribution = write[:, :, :, None] * torch.einsum("btk,btd->btkd", key, value)
        normalizer_contribution = write * key
        state = torch.cumsum(contribution * inverse[None, :, :, None], dim=1) * forward[None, :, :, None]
        normalizer = torch.cumsum(normalizer_contribution * inverse[None, :, :], dim=1) * forward[None, :, :]
        return torch.einsum("btk,btkd->btd", query, state) / (
            torch.einsum("btk,btk->bt", query, normalizer)[:, :, None] + 1e-5
        )


class MemoryProxyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.slow_head = nn.Linear(D_MODEL, N_STATE)
        self.medium_head = nn.Linear(D_MODEL, N_STATE)
        self.fast_head = nn.Linear(D_MODEL, N_STATE)
        self.reconstruction_head = nn.Linear(D_MODEL, VOCAB)

    def heads(self, hidden: torch.Tensor):
        return (
            self.slow_head(hidden),
            self.medium_head(hidden),
            self.fast_head(hidden),
            self.reconstruction_head(hidden),
        )


class SingleMemory(MemoryProxyBase):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.memory = RoutedMemoryBank(SLOTS)
        if mode == "shared_decay":
            self.memory.logit_decay = nn.Parameter(torch.tensor(4.0))
        elif mode == "fixed_multiband":
            self.memory.logit_decay = None
            self.memory.fixed_decay = torch.tensor([0.90, 0.90, 0.97, 0.97, 0.995, 0.995])
        elif mode != "per_slot_decay":
            raise ValueError(f"unsupported single-memory mode: {mode}")
        self.mix = nn.Linear(2 * D_MODEL, D_MODEL)

    def forward(self, ids: torch.Tensor):
        hidden = self.norm(self.embedding(ids))
        read = self.memory(hidden)
        return self.heads(self.mix(torch.cat([hidden, read], dim=-1)))


class IndependentMultiBank(MemoryProxyBase):
    def __init__(self) -> None:
        super().__init__()
        self.banks = nn.ModuleList(
            [
                RoutedMemoryBank(BANK_SLOTS, 0.90),
                RoutedMemoryBank(BANK_SLOTS, 0.97),
                RoutedMemoryBank(BANK_SLOTS, 0.995),
            ]
        )
        self.mix = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, ids: torch.Tensor):
        hidden = self.norm(self.embedding(ids))
        reads = [bank(hidden) for bank in self.banks]
        return self.heads(self.mix(torch.cat([hidden, *reads], dim=-1)))


MODELS = {
    "shared_decay": lambda: SingleMemory("shared_decay"),
    "per_slot_decay": lambda: SingleMemory("per_slot_decay"),
    "fixed_multiband": lambda: SingleMemory("fixed_multiband"),
    "independent_multibank": IndependentMultiBank,
}


@dataclass
class Result:
    mode: str
    seed: int
    parameters: int
    slow_late: float
    medium_late: float
    fast_late: float
    reconstruction: float
    decays: list[float]


def _decays(model: nn.Module) -> list[float]:
    if isinstance(model, IndependentMultiBank):
        return [float(value) for bank in model.banks for value in bank.decay().detach().cpu()]
    assert isinstance(model, SingleMemory)
    value = model.memory.decay().detach().cpu()
    if value.ndim == 0:
        return [float(value)]
    return [float(item) for item in value]


def run(mode: str, seed: int, steps: int, batch: int, device: torch.device) -> Result:
    torch.manual_seed(seed)
    random.seed(seed)
    model = MODELS[mode]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)
    model.train()
    for step in range(steps):
        ids, slow, medium, fast = make_batch(batch, device, seed * 100000 + step)
        slow_logits, medium_logits, fast_logits, reconstruction = model(ids)
        active = torch.arange(SEQ_LEN, device=device) >= 16
        loss = (
            F.cross_entropy(slow_logits[:, active].reshape(-1, N_STATE), slow[:, active].reshape(-1))
            + F.cross_entropy(medium_logits[:, active].reshape(-1, N_STATE), medium[:, active].reshape(-1))
            + F.cross_entropy(fast_logits[:, active].reshape(-1, N_STATE), fast[:, active].reshape(-1))
            + 0.5 * F.cross_entropy(reconstruction.reshape(-1, VOCAB), ids.reshape(-1))
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    ids, slow, medium, fast = make_batch(128, device, 9999)
    model.eval()
    with torch.no_grad():
        slow_logits, medium_logits, fast_logits, reconstruction = model(ids)
        late = torch.arange(SEQ_LEN, device=device) >= 160
        accuracy = lambda logits, target: float(
            (logits.argmax(-1)[:, late] == target[:, late]).float().mean()
        )
        return Result(
            mode=mode,
            seed=seed,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            slow_late=accuracy(slow_logits, slow),
            medium_late=accuracy(medium_logits, medium),
            fast_late=accuracy(fast_logits, fast),
            reconstruction=float((reconstruction.argmax(-1) == ids).float().mean()),
            decays=_decays(model),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run(args.mode, args.seed, args.steps, args.batch, device)
    payload = {
        "schema_version": 1,
        "device": str(device),
        "task": {
            "sequence_length": SEQ_LEN,
            "slow_update": "position 0 only",
            "medium_update_period": 96,
            "fast_update_period": 24,
            "late_eval_start": 160,
        },
        "result": asdict(result),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
