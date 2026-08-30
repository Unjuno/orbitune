from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 32
SEQ_LEN = 128
BANK_SLOTS = 2
TOTAL_SLOTS = 6
ACTIVE_START = 24
LATE_START = 80

# Current experimental Compound-v0 field widths/cardinalities. This proxy uses
# the 12-field layout directly but generates synthetic records; it is not a
# substitute for the real-MIDI/Compound target experiment.
FIELD_CARDINALITIES = (10, 16, 7, 16, 128, 128, 128, 16, 7, 16, 8, 8)
NOTE = 0
PROGRAM = 2
TEMPO = 4
PEDAL = 5
N_STATE = 4


@dataclass(slots=True)
class Result:
    mode: str
    seed: int
    parameters: int
    slow_late: float
    medium_late: float
    fast_late: float
    next_event_type: float


def make_batch(batch: int, device: torch.device, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    records = torch.zeros(batch, SEQ_LEN, 12, dtype=torch.long)
    slow = torch.zeros(batch, SEQ_LEN, dtype=torch.long)
    medium = torch.zeros(batch, SEQ_LEN, dtype=torch.long)
    fast = torch.zeros(batch, SEQ_LEN, dtype=torch.long)

    for row in range(batch):
        root = int(torch.randint(0, N_STATE, (1,), generator=generator))
        program_family = int(torch.randint(0, N_STATE, (1,), generator=generator))
        velocity_class = int(torch.randint(0, N_STATE, (1,), generator=generator))
        tempo_class = int(torch.randint(0, N_STATE, (1,), generator=generator))
        pedal = 0

        for step in range(SEQ_LEN):
            if step == 0 or step % 64 == 0:
                if step > 0:
                    program_family = int(
                        torch.randint(0, N_STATE, (1,), generator=generator)
                    )
                    tempo_class = int(
                        torch.randint(0, N_STATE, (1,), generator=generator)
                    )
                records[row, step, 0] = PROGRAM
                records[row, step, 1] = program_family % 2
                records[row, step, 4] = program_family * 8
            elif step % 16 == 0:
                velocity_class = int(
                    torch.randint(0, N_STATE, (1,), generator=generator)
                )
                pedal ^= 1
                records[row, step, 0] = PEDAL
                records[row, step, 1] = program_family % 2
                records[row, step, 4] = pedal
            elif step % 48 == 1:
                records[row, step, 0] = TEMPO
                records[row, step, 4] = (72, 96, 120, 144)[tempo_class]
            else:
                scale = (0, 2, 4, 7, 9)
                degree = scale[(step + 3 * row) % len(scale)]
                pitch = 48 + 12 * ((step // 24) % 3) + ((root * 3 + degree) % 12)
                velocity = (32, 56, 80, 112)[velocity_class]
                records[row, step, 0] = NOTE
                records[row, step, 1] = program_family % 2
                records[row, step, 2] = 1
                records[row, step, 3] = 8
                records[row, step, 4] = pitch
                records[row, step, 6] = velocity
                records[row, step, 8] = 1
                records[row, step, 9] = 8

            slow[row, step] = root
            medium[row, step] = program_family
            fast[row, step] = velocity_class

    return (
        records.to(device),
        slow.to(device),
        medium.to(device),
        fast.to(device),
    )


class FactorEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, D_MODEL)
            for cardinality in FIELD_CARDINALITIES
        )
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, records: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            *records.shape[:2], D_MODEL, device=records.device, dtype=torch.float32
        )
        for index, embedding in enumerate(self.embeddings):
            hidden = hidden + embedding(
                records[:, :, index].clamp_max(FIELD_CARDINALITIES[index] - 1)
            )
        return self.norm(hidden)


class RecurrentBank(nn.Module):
    def __init__(self, slots: int, decay: float) -> None:
        super().__init__()
        self.slots = slots
        self.decay = decay
        self.q = nn.Linear(D_MODEL, slots, bias=False)
        self.k = nn.Linear(D_MODEL, slots, bias=False)
        self.v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.write = nn.Linear(D_MODEL, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = F.elu(self.q(hidden)) + 1.0
        key = F.elu(self.k(hidden)) + 1.0
        value = self.v(hidden)
        write = torch.sigmoid(self.write(hidden))
        decay = torch.tensor(self.decay, device=hidden.device, dtype=hidden.dtype)
        index = torch.arange(
            hidden.shape[1], device=hidden.device, dtype=hidden.dtype
        )
        inverse = decay.pow(-index)
        forward = decay.pow(index)
        contribution = write[:, :, :, None] * torch.einsum(
            "btk,btd->btkd", key, value
        )
        normalizer_contribution = write * key
        state = torch.cumsum(
            contribution * inverse[None, :, None, None], dim=1
        ) * forward[None, :, None, None]
        normalizer = torch.cumsum(
            normalizer_contribution * inverse[None, :, None], dim=1
        ) * forward[None, :, None]
        return torch.einsum("btk,btkd->btd", query, state) / (
            torch.einsum("btk,btk->bt", query, normalizer)[:, :, None] + 1e-5
        )


class ProxyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = FactorEmbedding()
        self.slow_head = nn.Linear(D_MODEL, N_STATE)
        self.medium_head = nn.Linear(D_MODEL, N_STATE)
        self.fast_head = nn.Linear(D_MODEL, N_STATE)
        self.event_head = nn.Linear(D_MODEL, 10)


class SharedMatched(ProxyBase):
    """One six-slot memory with capacity matched to the routed model."""

    def __init__(self) -> None:
        super().__init__()
        self.memory = RecurrentBank(TOTAL_SLOTS, 0.97)
        self.adapter = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.fast_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.medium_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.slow_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.event_mix = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, records: torch.Tensor):
        hidden = self.embedding(records)
        memory = self.memory(hidden)
        batch, steps, width = memory.shape
        memory = self.adapter(memory).view(batch, steps, 2, width).mean(dim=2)
        slow = self.slow_head(self.slow_mix(torch.cat([hidden, memory], dim=-1)))
        medium = self.medium_head(
            self.medium_mix(torch.cat([hidden, memory], dim=-1))
        )
        fast = self.fast_head(self.fast_mix(torch.cat([hidden, memory], dim=-1)))
        event = self.event_head(
            self.event_mix(torch.cat([hidden, memory, memory, memory], dim=-1))
        )
        return slow, medium, fast, event


class RoutedMultiBank(ProxyBase):
    """Fast/medium/slow banks with target-specific read paths."""

    def __init__(self) -> None:
        super().__init__()
        self.banks = nn.ModuleList(
            [
                RecurrentBank(BANK_SLOTS, 0.90),
                RecurrentBank(BANK_SLOTS, 0.97),
                RecurrentBank(BANK_SLOTS, 0.997),
            ]
        )
        self.fast_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.medium_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.slow_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.event_mix = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, records: torch.Tensor):
        hidden = self.embedding(records)
        fast_memory, medium_memory, slow_memory = [
            bank(hidden) for bank in self.banks
        ]
        slow = self.slow_head(
            self.slow_mix(torch.cat([hidden, slow_memory], dim=-1))
        )
        medium = self.medium_head(
            self.medium_mix(torch.cat([hidden, medium_memory], dim=-1))
        )
        fast = self.fast_head(
            self.fast_mix(torch.cat([hidden, fast_memory], dim=-1))
        )
        event = self.event_head(
            self.event_mix(
                torch.cat(
                    [hidden, fast_memory, medium_memory, slow_memory], dim=-1
                )
            )
        )
        return slow, medium, fast, event


MODELS = {
    "shared_matched": SharedMatched,
    "multibank_routed": RoutedMultiBank,
}


def run(
    mode: str,
    seed: int,
    steps: int,
    batch: int,
    device: torch.device,
) -> Result:
    torch.manual_seed(seed)
    random.seed(seed)
    model = MODELS[mode]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    active = torch.arange(SEQ_LEN, device=device) >= ACTIVE_START

    model.train()
    for step in range(steps):
        records, slow, medium, fast = make_batch(
            batch, device, seed * 10000 + step
        )
        slow_logits, medium_logits, fast_logits, event_logits = model(records)
        loss = (
            F.cross_entropy(
                slow_logits[:, active].reshape(-1, N_STATE),
                slow[:, active].reshape(-1),
            )
            + F.cross_entropy(
                medium_logits[:, active].reshape(-1, N_STATE),
                medium[:, active].reshape(-1),
            )
            + F.cross_entropy(
                fast_logits[:, active].reshape(-1, N_STATE),
                fast[:, active].reshape(-1),
            )
            + 0.5
            * F.cross_entropy(
                event_logits[:, :-1].reshape(-1, 10),
                records[:, 1:, 0].reshape(-1),
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    records, slow, medium, fast = make_batch(48, device, 99991 + seed)
    late = torch.arange(SEQ_LEN, device=device) >= LATE_START
    model.eval()
    with torch.no_grad():
        slow_logits, medium_logits, fast_logits, event_logits = model(records)

        def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
            return float(
                (logits.argmax(-1)[:, late] == targets[:, late]).float().mean()
            )

        event_accuracy = float(
            (event_logits[:, :-1].argmax(-1) == records[:, 1:, 0]).float().mean()
        )

    return Result(
        mode=mode,
        seed=seed,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        slow_late=accuracy(slow_logits, slow),
        medium_late=accuracy(medium_logits, medium),
        fast_late=accuracy(fast_logits, fast),
        next_event_type=event_accuracy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = run(args.mode, args.seed, args.steps, args.batch, device)
    payload = {
        "schema_version": 1,
        "device": str(device),
        "task": {
            "record_width": 12,
            "sequence_length": SEQ_LEN,
            "active_start": ACTIVE_START,
            "late_eval_start": LATE_START,
            "slow_target": "composition root class encoded through note pitch",
            "medium_target": "program-family state changed every 64 events",
            "fast_target": "velocity class changed every 16 events",
            "next_event_target": "Compound event_type",
            "scope": "synthetic Compound-field proxy; not real MIDI",
        },
        "result": asdict(result),
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
