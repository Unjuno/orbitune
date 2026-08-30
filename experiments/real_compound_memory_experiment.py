from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from orbitune.compound_memory_targets import derive_compound_memory_targets, target_cardinalities
from orbitune.compound_training import CompoundSong, load_compound_jsonl
from orbitune.tokenizer.compound_event import CompoundRecord

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from recurrent_memory_chunkwise_scan import chunkwise_discounted_scan  # noqa: E402

D_MODEL = 48
FAST_NAMES = ("note_density_bin", "mean_velocity_bin", "note_gap_bin")
MEDIUM_NAMES = (
    "mean_register_bin",
    "dominant_program_family",
    "channel_diversity_bin",
    "pedal_any",
    "tempo_bin",
)
SLOW_NAMES = (
    "dominant_pitch_class",
    "pitch_class_entropy_bin",
    "program_family_diversity_bin",
)
CARDINALITIES = target_cardinalities()
FAST_CARDS = tuple(CARDINALITIES["fast"][name] for name in FAST_NAMES)
MEDIUM_CARDS = tuple(CARDINALITIES["medium"][name] for name in MEDIUM_NAMES)
SLOW_CARDS = tuple(CARDINALITIES["slow"][name] for name in SLOW_NAMES)
# a1 must preserve TEMPO 1..999. a2 is deliberately bounded here because the
# memory targets do not depend on arbitrary large time-signature denominators.
FIELD_CARDINALITIES = (10, 16, 7, 16, 1024, 1024, 128, 256, 7, 16, 8, 8)


@dataclass(frozen=True, slots=True)
class PreparedSong:
    path: str
    sha256: str
    records: torch.Tensor
    fast: torch.Tensor
    medium: torch.Tensor
    slow: torch.Tensor


@dataclass(frozen=True, slots=True)
class Metrics:
    fast_macro_recall: float
    medium_macro_recall: float
    slow_macro_recall: float
    next_event_type_accuracy: float
    events_evaluated: int


BankState = tuple[torch.Tensor, torch.Tensor]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_record(values: tuple[int, ...]) -> CompoundRecord:
    return CompoundRecord(*values)


def prepare_song(song: CompoundSong) -> PreparedSong:
    records = [_to_record(record) for record in song.records]
    targets = derive_compound_memory_targets(records)
    fast = [
        (
            target.fast.note_density_bin,
            target.fast.mean_velocity_bin,
            target.fast.note_gap_bin,
        )
        for target in targets
    ]
    medium = [
        (
            target.medium.mean_register_bin,
            target.medium.dominant_program_family,
            target.medium.channel_diversity_bin,
            target.medium.pedal_any,
            target.medium.tempo_bin,
        )
        for target in targets
    ]
    slow = [
        (
            target.slow.dominant_pitch_class,
            target.slow.pitch_class_entropy_bin,
            target.slow.program_family_diversity_bin,
        )
        for target in targets
    ]
    return PreparedSong(
        path=song.path,
        sha256=song.sha256,
        records=torch.tensor(song.records, dtype=torch.long),
        fast=torch.tensor(fast, dtype=torch.long),
        medium=torch.tensor(medium, dtype=torch.long),
        slow=torch.tensor(slow, dtype=torch.long),
    )


def _limit_songs(songs: list[CompoundSong], limit: int) -> list[CompoundSong]:
    ordered = sorted(songs, key=lambda song: (song.sha256, song.path))
    return ordered if limit <= 0 else ordered[:limit]


def load_splits(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    max_train_songs: int = 0,
    max_validation_songs: int = 0,
) -> tuple[list[PreparedSong], list[PreparedSong]]:
    train_raw = _limit_songs(load_compound_jsonl(train_path), max_train_songs)
    validation_raw = _limit_songs(
        load_compound_jsonl(validation_path), max_validation_songs
    )
    train_hashes = {song.sha256 for song in train_raw}
    validation_hashes = {song.sha256 for song in validation_raw}
    if "" in train_hashes or "" in validation_hashes:
        raise ValueError("real-data experiment requires non-empty MIDI SHA-256 values")
    overlap = train_hashes & validation_hashes
    if overlap:
        raise ValueError(
            f"exact MIDI content leaked across train/validation: {sorted(overlap)[:3]}"
        )
    return (
        [prepare_song(song) for song in train_raw],
        [prepare_song(song) for song in validation_raw],
    )


def _profile_group(
    songs: list[PreparedSong],
    attribute: str,
    names: tuple[str, ...],
    cards: tuple[int, ...],
    warmup_events: int,
) -> dict[str, object]:
    counts = [torch.zeros(card, dtype=torch.long) for card in cards]
    total_events = 0
    for song in songs:
        values = getattr(song, attribute)
        if values.shape[0] <= warmup_events:
            continue
        values = values[warmup_events:]
        total_events += values.shape[0]
        for head_index, card in enumerate(cards):
            counts[head_index] += torch.bincount(
                values[:, head_index], minlength=card
            )
    heads: dict[str, object] = {}
    for name, count in zip(names, counts):
        total = int(count.sum())
        majority = float(count.max() / total) if total else 0.0
        heads[name] = {
            "counts": count.tolist(),
            "classes_observed": int((count > 0).sum()),
            "majority_baseline": majority,
        }
    return {"events": total_events, "heads": heads}


def target_profile(
    songs: list[PreparedSong], warmup_events: int
) -> dict[str, object]:
    return {
        "fast": _profile_group(
            songs, "fast", FAST_NAMES, FAST_CARDS, warmup_events
        ),
        "medium": _profile_group(
            songs, "medium", MEDIUM_NAMES, MEDIUM_CARDS, warmup_events
        ),
        "slow": _profile_group(
            songs, "slow", SLOW_NAMES, SLOW_CARDS, warmup_events
        ),
    }


def _class_weights(
    songs: list[PreparedSong],
    attribute: str,
    cards: tuple[int, ...],
    warmup_events: int,
    device: torch.device,
) -> list[torch.Tensor]:
    counts = [torch.zeros(card, dtype=torch.float64) for card in cards]
    for song in songs:
        values = getattr(song, attribute)
        if values.shape[0] <= warmup_events:
            continue
        values = values[warmup_events:]
        for index, card in enumerate(cards):
            counts[index] += torch.bincount(
                values[:, index], minlength=card
            ).to(torch.float64)
    weights: list[torch.Tensor] = []
    for count in counts:
        used = count > 0
        weight = torch.zeros_like(count)
        if used.any():
            weight[used] = count[used].sum() / (used.sum() * count[used])
            weight = weight.clamp_max(8.0)
        weights.append(weight.to(device=device, dtype=torch.float32))
    return weights


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
        self.decay_value = float(decay)
        self.norm = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, slots, bias=False)
        self.k = nn.Linear(D_MODEL, slots, bias=False)
        self.v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.write = nn.Linear(D_MODEL, 1)

    def forward_chunk(
        self, hidden: torch.Tensor, state: BankState | None
    ) -> tuple[torch.Tensor, BankState]:
        x = self.norm(hidden)
        query = F.elu(self.q(x)) + 1.0
        key = F.elu(self.k(x)) + 1.0
        value = self.v(x)
        write = torch.sigmoid(self.write(x))
        contributions = write.unsqueeze(-1) * torch.einsum(
            "btk,btd->btkd", key, value
        )
        normalizer_contributions = write * key
        initial_state = None if state is None else state[0]
        initial_normalizer = None if state is None else state[1]
        decay = torch.tensor(
            self.decay_value, device=hidden.device, dtype=torch.float32
        )
        scan = chunkwise_discounted_scan(
            contributions.float(),
            normalizer_contributions.float(),
            decay,
            chunk_size=max(1, hidden.shape[1]),
            initial_state=initial_state,
            initial_normalizer=initial_normalizer,
        )
        read = torch.einsum(
            "btk,btkd->btd", query.float(), scan.states
        ) / (
            torch.einsum(
                "btk,btk->bt", query.float(), scan.normalizers
            ).unsqueeze(-1)
            + 1e-5
        )
        return read.to(hidden.dtype), (scan.final_state, scan.final_normalizer)


class TargetHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fast_heads = nn.ModuleList(
            nn.Linear(D_MODEL, card) for card in FAST_CARDS
        )
        self.medium_heads = nn.ModuleList(
            nn.Linear(D_MODEL, card) for card in MEDIUM_CARDS
        )
        self.slow_heads = nn.ModuleList(
            nn.Linear(D_MODEL, card) for card in SLOW_CARDS
        )
        self.event_head = nn.Linear(D_MODEL, 10)

    def memory_logits(
        self,
        fast_hidden: torch.Tensor,
        medium_hidden: torch.Tensor,
        slow_hidden: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        return (
            [head(fast_hidden) for head in self.fast_heads],
            [head(medium_hidden) for head in self.medium_heads],
            [head(slow_hidden) for head in self.slow_heads],
        )


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

    def forward_chunk(self, records: torch.Tensor, state: BankState | None):
        hidden = self.embedding(records)
        memory, next_state = self.memory.forward_chunk(hidden, state)
        batch, steps, width = memory.shape
        memory = self.adapter(memory).view(batch, steps, 2, width).mean(dim=2)
        fast_hidden = self.fast_mix(torch.cat([memory, memory], dim=-1))
        medium_hidden = self.medium_mix(torch.cat([memory, memory], dim=-1))
        slow_hidden = self.slow_mix(torch.cat([memory, memory], dim=-1))
        event_hidden = self.event_mix(
            torch.cat([hidden, memory, memory, memory], dim=-1)
        )
        return (
            *self.memory_logits(fast_hidden, medium_hidden, slow_hidden),
            self.event_head(event_hidden),
            next_state,
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

    def forward_chunk(
        self,
        records: torch.Tensor,
        state: tuple[BankState | None, BankState | None, BankState | None] | None,
    ):
        hidden = self.embedding(records)
        if state is None:
            state = (None, None, None)
        fast_memory, fast_state = self.fast_memory.forward_chunk(hidden, state[0])
        medium_memory, medium_state = self.medium_memory.forward_chunk(
            hidden, state[1]
        )
        slow_memory, slow_state = self.slow_memory.forward_chunk(hidden, state[2])
        fast_hidden = self.fast_mix(torch.cat([fast_memory, fast_memory], dim=-1))
        medium_hidden = self.medium_mix(
            torch.cat([medium_memory, medium_memory], dim=-1)
        )
        slow_hidden = self.slow_mix(torch.cat([slow_memory, slow_memory], dim=-1))
        event_hidden = self.event_mix(
            torch.cat([hidden, fast_memory, medium_memory, slow_memory], dim=-1)
        )
        return (
            *self.memory_logits(fast_hidden, medium_hidden, slow_hidden),
            self.event_head(event_hidden),
            (fast_state, medium_state, slow_state),
        )


MODELS = {"shared_matched": SharedMatched, "multibank_routed": RoutedMultiBank}


def _detach_state(state):  # type: ignore[no-untyped-def]
    if state is None:
        return None
    if isinstance(state, tuple) and len(state) == 2 and all(
        isinstance(value, torch.Tensor) for value in state
    ):
        return state[0].detach(), state[1].detach()
    return tuple(_detach_state(value) for value in state)


def _head_loss(
    logits: list[torch.Tensor],
    labels: torch.Tensor,
    weights: list[torch.Tensor],
) -> torch.Tensor:
    total = logits[0].new_zeros(())
    for index, (head_logits, weight) in enumerate(zip(logits, weights)):
        total = total + F.cross_entropy(
            head_logits.reshape(-1, head_logits.shape[-1]),
            labels[:, :, index].reshape(-1),
            weight=weight,
        )
    return total


def train_memory_stage(
    model: nn.Module,
    songs: list[PreparedSong],
    *,
    epochs: int,
    chunk_size: int,
    warmup_events: int,
    seed: int,
    device: torch.device,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    fast_weights = _class_weights(
        songs, "fast", FAST_CARDS, warmup_events, device
    )
    medium_weights = _class_weights(
        songs, "medium", MEDIUM_CARDS, warmup_events, device
    )
    slow_weights = _class_weights(
        songs, "slow", SLOW_CARDS, warmup_events, device
    )
    rng = random.Random(seed)
    model.train()
    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            state = None
            for start in range(0, len(song.records), chunk_size):
                stop = min(len(song.records), start + chunk_size)
                records = song.records[start:stop].unsqueeze(0).to(device)
                fast = song.fast[start:stop].unsqueeze(0).to(device)
                medium = song.medium[start:stop].unsqueeze(0).to(device)
                slow = song.slow[start:stop].unsqueeze(0).to(device)
                fast_logits, medium_logits, slow_logits, _, next_state = model.forward_chunk(
                    records, state
                )
                active_start = max(0, warmup_events - start)
                if active_start < stop - start:
                    loss = (
                        _head_loss(
                            [logits[:, active_start:] for logits in fast_logits],
                            fast[:, active_start:],
                            fast_weights,
                        )
                        + _head_loss(
                            [logits[:, active_start:] for logits in medium_logits],
                            medium[:, active_start:],
                            medium_weights,
                        )
                        + _head_loss(
                            [logits[:, active_start:] for logits in slow_logits],
                            slow[:, active_start:],
                            slow_weights,
                        )
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                state = _detach_state(next_state)


def _configure_composer_optimizer(
    model: nn.Module,
    policy: str,
    composer_lr: float,
    memory_lr_multiplier: float,
) -> torch.optim.Optimizer:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    composer: list[nn.Parameter] = []
    memory: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith("event_head") or name.startswith("event_mix"):
            parameter.requires_grad_(True)
            composer.append(parameter)
        elif name.startswith("embedding") or "memory" in name and "heads" not in name:
            if policy in {"low_lr", "joint"}:
                parameter.requires_grad_(True)
                memory.append(parameter)
    if policy == "frozen":
        return torch.optim.AdamW(composer, lr=composer_lr)
    if policy == "low_lr":
        return torch.optim.AdamW(
            [
                {"params": composer, "lr": composer_lr},
                {
                    "params": memory,
                    "lr": composer_lr * memory_lr_multiplier,
                },
            ]
        )
    if policy == "joint":
        return torch.optim.AdamW([*composer, *memory], lr=composer_lr)
    raise ValueError(f"unsupported composer policy: {policy}")


def train_composer_stage(
    model: nn.Module,
    songs: list[PreparedSong],
    *,
    policy: str,
    epochs: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
    composer_lr: float,
    memory_lr_multiplier: float,
) -> None:
    optimizer = _configure_composer_optimizer(
        model, policy, composer_lr, memory_lr_multiplier
    )
    rng = random.Random(seed + 991)
    model.train()
    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            if len(song.records) < 2:
                continue
            state = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, chunk_size):
                stop = min(final_input, start + chunk_size)
                records = song.records[start:stop].unsqueeze(0).to(device)
                next_types = song.records[start + 1 : stop + 1, 0].unsqueeze(0).to(device)
                _, _, _, event_logits, next_state = model.forward_chunk(records, state)
                loss = F.cross_entropy(
                    event_logits.reshape(-1, 10), next_types.reshape(-1)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                trainable = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                state = _detach_state(next_state)


def _recall_score(correct: list[torch.Tensor], total: list[torch.Tensor]) -> float:
    head_scores: list[float] = []
    for head_correct, head_total in zip(correct, total):
        used = head_total > 0
        if used.any():
            head_scores.append(float((head_correct[used] / head_total[used]).mean()))
    return sum(head_scores) / len(head_scores) if head_scores else 0.0


def evaluate(
    model: nn.Module,
    songs: list[PreparedSong],
    *,
    chunk_size: int,
    warmup_events: int,
    device: torch.device,
) -> Metrics:
    groups = (
        (FAST_CARDS, "fast"),
        (MEDIUM_CARDS, "medium"),
        (SLOW_CARDS, "slow"),
    )
    correct = {
        name: [torch.zeros(card, dtype=torch.float64) for card in cards]
        for cards, name in groups
    }
    total = {
        name: [torch.zeros(card, dtype=torch.float64) for card in cards]
        for cards, name in groups
    }
    event_correct = 0
    event_total = 0
    evaluated = 0
    model.eval()
    with torch.no_grad():
        for song in songs:
            state = None
            for start in range(0, len(song.records), chunk_size):
                stop = min(len(song.records), start + chunk_size)
                records = song.records[start:stop].unsqueeze(0).to(device)
                fast_logits, medium_logits, slow_logits, event_logits, next_state = model.forward_chunk(
                    records, state
                )
                state = _detach_state(next_state)
                active_start = max(0, warmup_events - start)
                if active_start >= stop - start:
                    continue
                tier_data = (
                    ("fast", fast_logits, song.fast[start:stop]),
                    ("medium", medium_logits, song.medium[start:stop]),
                    ("slow", slow_logits, song.slow[start:stop]),
                )
                for name, logits_list, labels_cpu in tier_data:
                    labels = labels_cpu[active_start:].to(device)
                    for index, logits in enumerate(logits_list):
                        prediction = logits[0, active_start:].argmax(-1)
                        truth = labels[:, index]
                        for cls in truth.unique():
                            cls_int = int(cls)
                            mask = truth == cls
                            total[name][index][cls_int] += int(mask.sum())
                            correct[name][index][cls_int] += int(
                                (prediction[mask] == truth[mask]).sum()
                            )
                evaluated += stop - start - active_start

                global_start = start + active_start
                event_stop = min(stop, len(song.records) - 1)
                if global_start < event_stop:
                    local_start = global_start - start
                    local_stop = event_stop - start
                    prediction = event_logits[0, local_start:local_stop].argmax(-1).cpu()
                    truth = song.records[global_start + 1 : event_stop + 1, 0]
                    event_correct += int((prediction == truth).sum())
                    event_total += len(truth)
    return Metrics(
        fast_macro_recall=_recall_score(correct["fast"], total["fast"]),
        medium_macro_recall=_recall_score(correct["medium"], total["medium"]),
        slow_macro_recall=_recall_score(correct["slow"], total["slow"]),
        next_event_type_accuracy=event_correct / event_total if event_total else 0.0,
        events_evaluated=evaluated,
    )


def _corpus_summary(songs: Iterable[PreparedSong]) -> dict[str, int]:
    songs = list(songs)
    return {
        "songs": len(songs),
        "events": sum(len(song.records) for song in songs),
    }


def run(args) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train, validation = load_splits(
        args.train_jsonl,
        args.validation_jsonl,
        max_train_songs=args.max_train_songs,
        max_validation_songs=args.max_validation_songs,
    )
    if not any(len(song.records) > args.warmup_events for song in train):
        raise ValueError("no training song has events beyond --warmup-events")
    if not any(len(song.records) > args.warmup_events for song in validation):
        raise ValueError("no validation song has events beyond --warmup-events")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = MODELS[args.mode]().to(device)
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model_type().parameters())
        for name, model_type in MODELS.items()
    }
    if abs(parameter_counts["shared_matched"] - parameter_counts["multibank_routed"]) > 2:
        raise AssertionError("shared/routed proxy models are not parameter matched")

    train_memory_stage(
        model,
        train,
        epochs=args.memory_epochs,
        chunk_size=args.chunk_size,
        warmup_events=args.warmup_events,
        seed=args.seed,
        device=device,
        learning_rate=args.memory_lr,
    )
    before = evaluate(
        model,
        validation,
        chunk_size=args.chunk_size,
        warmup_events=args.warmup_events,
        device=device,
    )
    consolidated = copy.deepcopy(model)
    train_composer_stage(
        model,
        train,
        policy=args.composer_policy,
        epochs=args.composer_epochs,
        chunk_size=args.chunk_size,
        seed=args.seed,
        device=device,
        composer_lr=args.composer_lr,
        memory_lr_multiplier=args.memory_lr_multiplier,
    )
    after = evaluate(
        model,
        validation,
        chunk_size=args.chunk_size,
        warmup_events=args.warmup_events,
        device=device,
    )

    if args.checkpoint_out:
        checkpoint = Path(args.checkpoint_out)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model": args.mode,
                "composer_policy": args.composer_policy,
                "consolidated_memory_state_dict": consolidated.state_dict(),
                "final_state_dict": model.state_dict(),
            },
            checkpoint,
        )

    return {
        "schema_version": 1,
        "device": str(device),
        "seed": args.seed,
        "model": args.mode,
        "composer_policy": args.composer_policy,
        "parameter_counts": parameter_counts,
        "split": {
            "train_jsonl_sha256": _file_sha256(args.train_jsonl),
            "validation_jsonl_sha256": _file_sha256(args.validation_jsonl),
            "train": _corpus_summary(train),
            "validation": _corpus_summary(validation),
            "split_leakage_check": "exact MIDI SHA-256 disjoint",
            "composition_family_near_dedup": "not yet implemented; required before production claims",
        },
        "target_profile": {
            "train": target_profile(train, args.warmup_events),
            "validation": target_profile(validation, args.warmup_events),
        },
        "training": {
            "memory_epochs": args.memory_epochs,
            "composer_epochs": args.composer_epochs,
            "chunk_size": args.chunk_size,
            "warmup_events": args.warmup_events,
            "memory_lr": args.memory_lr,
            "composer_lr": args.composer_lr,
            "memory_lr_multiplier": args.memory_lr_multiplier,
            "state_carry": "composition-local fixed-size state; detach at chunk boundaries",
        },
        "validation_before_composer": asdict(before),
        "validation_after_composer": asdict(after),
        "memory_delta": {
            "fast_macro_recall": after.fast_macro_recall - before.fast_macro_recall,
            "medium_macro_recall": after.medium_macro_recall - before.medium_macro_recall,
            "slow_macro_recall": after.slow_macro_recall - before.slow_macro_recall,
        },
        "scope": "real Compound JSONL experiment harness; corpus rights and composition-family dedup remain external gates",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="State-carry recurrent-memory experiment on real Compound JSONL"
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--mode", choices=MODELS, required=True)
    parser.add_argument(
        "--composer-policy", choices=["frozen", "low_lr", "joint"], default="frozen"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--memory-epochs", type=int, default=1)
    parser.add_argument("--composer-epochs", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmup-events", type=int, default=32)
    parser.add_argument("--memory-lr", type=float, default=3e-3)
    parser.add_argument("--composer-lr", type=float, default=3e-3)
    parser.add_argument("--memory-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--max-train-songs", type=int, default=0)
    parser.add_argument("--max-validation-songs", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.warmup_events < 0:
        raise SystemExit("--chunk-size must be positive and --warmup-events non-negative")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = run(args)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
