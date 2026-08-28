from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from orbitune.compound import CompoundEventType, quantize_time
from orbitune.compound_memory_targets import derive_compound_memory_targets
from orbitune.tokenizer.compound_event import CompoundRecord

D_MODEL = 24
SEQ_LEN = 128
ACTIVE_START = 24
LATE_START = 80
FAST_CARDS = (7, 9, 8)
MEDIUM_CARDS = (9, 17, 6, 2, 16)
SLOW_CARDS = (13, 8, 6)
FIELD_CARDINALITIES = (10, 16, 7, 16, 1024, 128, 128, 16, 7, 16, 8, 8)


@dataclass(slots=True)
class Result:
    mode: str
    seed: int
    parameters: int
    fast_macro_recall: float
    medium_macro_recall: float
    slow_macro_recall: float
    next_event_type_accuracy: float


def _record(
    event_type: CompoundEventType,
    *,
    channel: int = 0,
    delta: int = 0,
    a1: int = 0,
    a2: int = 0,
    a3: int = 0,
    a4: int = 0,
    duration_coarse: int = 0,
    duration_residual: int = 0,
    continuous_coarse: int = 0,
    continuous_residual: int = 0,
) -> CompoundRecord:
    factor = quantize_time(delta)
    return CompoundRecord(
        event_type=int(event_type),
        channel=channel,
        delta_coarse=factor.coarse,
        delta_residual=factor.residual,
        a1=a1,
        a2=a2,
        a3=a3,
        a4=a4,
        duration_coarse=duration_coarse,
        duration_residual=duration_residual,
        continuous_coarse=continuous_coarse,
        continuous_residual=continuous_residual,
    )


def make_batch(batch: int, device: torch.device, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    spacing_values = (6, 12, 24, 48, 96, 192, 384)
    velocity_values = (24, 48, 72, 96, 120)
    tempo_values = (60, 120, 240, 480, 720, 960)
    pitch_set_sizes = (1, 2, 4, 7, 12)
    register_bases = (32, 48, 64, 80)
    channel_counts = (1, 2, 4)
    family_counts = (1, 2, 4)
    rest_lengths = (0, 2, 4, 8)

    tensor_rows: list[list[tuple[int, ...]]] = []
    fast_rows: list[list[tuple[int, ...]]] = []
    medium_rows: list[list[tuple[int, ...]]] = []
    slow_rows: list[list[tuple[int, ...]]] = []

    for row in range(batch):
        root = int(torch.randint(0, 12, (1,), generator=generator))
        spacing = spacing_values[
            int(torch.randint(0, len(spacing_values), (1,), generator=generator))
        ]
        pitch_set_size = pitch_set_sizes[
            int(torch.randint(0, len(pitch_set_sizes), (1,), generator=generator))
        ]
        register_base = register_bases[
            int(torch.randint(0, len(register_bases), (1,), generator=generator))
        ]
        channel_count = channel_counts[
            int(torch.randint(0, len(channel_counts), (1,), generator=generator))
        ]
        family_count = family_counts[
            int(torch.randint(0, len(family_counts), (1,), generator=generator))
        ]
        rest_length = rest_lengths[
            int(torch.randint(0, len(rest_lengths), (1,), generator=generator))
        ]
        family_base = int(torch.randint(0, 12, (1,), generator=generator))
        families = [(family_base + 3 * index) % 16 for index in range(family_count)]
        velocity_index = int(
            torch.randint(0, len(velocity_values), (1,), generator=generator)
        )
        tempo_index = int(
            torch.randint(0, len(tempo_values), (1,), generator=generator)
        )
        pedal = 0
        records: list[CompoundRecord] = []

        for step in range(SEQ_LEN):
            delta = 0 if step == 0 else spacing
            if step and step % 24 < rest_length:
                record = _record(
                    CompoundEventType.CC,
                    channel=step % channel_count,
                    delta=delta,
                    a1=1,
                    continuous_coarse=3,
                    continuous_residual=4,
                )
            elif step == 0 or step % 32 == 0:
                family = families[(step // 32) % family_count]
                record = _record(
                    CompoundEventType.PROGRAM,
                    channel=(step // 32) % channel_count,
                    delta=delta,
                    a1=family * 8,
                )
            elif step % 47 == 1:
                tempo_index = (tempo_index + 1) % len(tempo_values)
                record = _record(
                    CompoundEventType.TEMPO,
                    delta=delta,
                    a1=tempo_values[tempo_index],
                )
            elif step % 37 == 0:
                pedal ^= 1
                record = _record(
                    CompoundEventType.PEDAL,
                    channel=step % channel_count,
                    delta=delta,
                    a1=pedal,
                )
            else:
                if step % 16 == 0:
                    velocity_index = (velocity_index + 1) % len(velocity_values)
                offset = 0 if step % 4 == 0 else ((5 * step + 3 * row) % pitch_set_size)
                pitch_class = (root + offset) % 12
                pitch = max(21, min(108, register_base + pitch_class))
                record = _record(
                    CompoundEventType.NOTE,
                    channel=step % channel_count,
                    delta=delta,
                    a1=pitch,
                    a3=velocity_values[velocity_index],
                    duration_coarse=1,
                    duration_residual=8,
                )
            records.append(record)

        targets = derive_compound_memory_targets(records)
        tensor_rows.append([record.as_tuple() for record in records])
        fast_rows.append(
            [
                (
                    target.fast.note_density_bin,
                    target.fast.mean_velocity_bin,
                    target.fast.note_gap_bin,
                )
                for target in targets
            ]
        )
        medium_rows.append(
            [
                (
                    target.medium.mean_register_bin,
                    target.medium.dominant_program_family,
                    target.medium.channel_diversity_bin,
                    target.medium.pedal_any,
                    target.medium.tempo_bin,
                )
                for target in targets
            ]
        )
        slow_rows.append(
            [
                (
                    target.slow.dominant_pitch_class,
                    target.slow.pitch_class_entropy_bin,
                    target.slow.program_family_diversity_bin,
                )
                for target in targets
            ]
        )

    return (
        torch.tensor(tensor_rows, dtype=torch.long, device=device),
        torch.tensor(fast_rows, dtype=torch.long, device=device),
        torch.tensor(medium_rows, dtype=torch.long, device=device),
        torch.tensor(slow_rows, dtype=torch.long, device=device),
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
        self.q = nn.Linear(D_MODEL, slots, bias=False)
        self.k = nn.Linear(D_MODEL, slots, bias=False)
        self.v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.write = nn.Linear(D_MODEL, 1)
        self.decay = float(decay)

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


class TargetHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fast_heads = nn.ModuleList(nn.Linear(D_MODEL, card) for card in FAST_CARDS)
        self.medium_heads = nn.ModuleList(
            nn.Linear(D_MODEL, card) for card in MEDIUM_CARDS
        )
        self.slow_heads = nn.ModuleList(nn.Linear(D_MODEL, card) for card in SLOW_CARDS)
        self.event_head = nn.Linear(D_MODEL, 10)


class SharedMatched(TargetHeads):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = FactorEmbedding()
        self.memory = RecurrentBank(6, 0.97)
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
        fast_hidden = self.fast_mix(torch.cat([memory, memory], dim=-1))
        medium_hidden = self.medium_mix(torch.cat([memory, memory], dim=-1))
        slow_hidden = self.slow_mix(torch.cat([memory, memory], dim=-1))
        event_hidden = self.event_mix(
            torch.cat([hidden, memory, memory, memory], dim=-1)
        )
        return (
            [head(fast_hidden) for head in self.fast_heads],
            [head(medium_hidden) for head in self.medium_heads],
            [head(slow_hidden) for head in self.slow_heads],
            self.event_head(event_hidden),
        )


class RoutedMultiBank(TargetHeads):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = FactorEmbedding()
        self.fast_memory = RecurrentBank(2, 0.90)
        self.medium_memory = RecurrentBank(2, 0.97)
        self.slow_memory = RecurrentBank(2, 0.997)
        self.fast_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.medium_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.slow_mix = nn.Linear(2 * D_MODEL, D_MODEL)
        self.event_mix = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, records: torch.Tensor):
        hidden = self.embedding(records)
        fast_memory = self.fast_memory(hidden)
        medium_memory = self.medium_memory(hidden)
        slow_memory = self.slow_memory(hidden)
        fast_hidden = self.fast_mix(torch.cat([fast_memory, fast_memory], dim=-1))
        medium_hidden = self.medium_mix(
            torch.cat([medium_memory, medium_memory], dim=-1)
        )
        slow_hidden = self.slow_mix(torch.cat([slow_memory, slow_memory], dim=-1))
        event_hidden = self.event_mix(
            torch.cat([hidden, fast_memory, medium_memory, slow_memory], dim=-1)
        )
        return (
            [head(fast_hidden) for head in self.fast_heads],
            [head(medium_hidden) for head in self.medium_heads],
            [head(slow_hidden) for head in self.slow_heads],
            self.event_head(event_hidden),
        )


MODELS = {"shared_matched": SharedMatched, "multibank_routed": RoutedMultiBank}


def _balanced_loss(
    logits: list[torch.Tensor],
    targets: torch.Tensor,
    cards: tuple[int, ...],
    mask: torch.Tensor,
) -> torch.Tensor:
    total = logits[0].new_zeros(())
    for index, (head_logits, card) in enumerate(zip(logits, cards)):
        labels = targets[:, mask, index].reshape(-1)
        counts = torch.bincount(labels, minlength=card).float()
        used = counts > 0
        weights = torch.zeros_like(counts)
        weights[used] = counts[used].sum() / (used.sum() * counts[used])
        weights = weights.clamp_max(8.0)
        total = total + F.cross_entropy(
            head_logits[:, mask].reshape(-1, card), labels, weight=weights
        )
    return total


def _macro_recall(
    logits: list[torch.Tensor], targets: torch.Tensor, mask: torch.Tensor
) -> float:
    head_scores: list[float] = []
    for index, head_logits in enumerate(logits):
        labels = targets[:, mask, index].reshape(-1)
        predictions = head_logits.argmax(-1)[:, mask].reshape(-1)
        recalls: list[float] = []
        for cls in labels.unique():
            cls_mask = labels == cls
            recalls.append(
                float((predictions[cls_mask] == labels[cls_mask]).float().mean())
            )
        head_scores.append(sum(recalls) / len(recalls))
    return sum(head_scores) / len(head_scores)


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
        records, fast, medium, slow = make_batch(
            batch, device, seed * 10000 + step
        )
        fast_logits, medium_logits, slow_logits, event_logits = model(records)
        loss = (
            _balanced_loss(fast_logits, fast, FAST_CARDS, active)
            + _balanced_loss(medium_logits, medium, MEDIUM_CARDS, active)
            + _balanced_loss(slow_logits, slow, SLOW_CARDS, active)
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

    records, fast, medium, slow = make_batch(48, device, 99991 + seed)
    late = torch.arange(SEQ_LEN, device=device) >= LATE_START
    model.eval()
    with torch.no_grad():
        fast_logits, medium_logits, slow_logits, event_logits = model(records)
        event_accuracy = float(
            (event_logits[:, :-1].argmax(-1) == records[:, 1:, 0]).float().mean()
        )

    return Result(
        mode=mode,
        seed=seed,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        fast_macro_recall=_macro_recall(fast_logits, fast, late),
        medium_macro_recall=_macro_recall(medium_logits, medium, late),
        slow_macro_recall=_macro_recall(slow_logits, slow, late),
        next_event_type_accuracy=event_accuracy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=60)
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
            "target_schema": "orbitune-compound-memory-targets-v0-experimental",
            "record_width": 12,
            "sequence_length": SEQ_LEN,
            "late_eval_start": LATE_START,
            "loss": "class-balanced per-head cross entropy; memory heads see memory only",
            "metric": "mean macro recall across heads in each memory tier",
            "scope": "balanced synthetic Compound records; real MIDI remains required",
        },
        "result": asdict(result),
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
