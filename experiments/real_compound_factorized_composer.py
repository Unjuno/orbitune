from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import real_compound_bounded_composer as bounded  # noqa: E402
import real_compound_memory_experiment as memory_base  # noqa: E402
import real_compound_memory_experiment_matched as matched  # noqa: E402


D_MODEL = memory_base.D_MODEL
FIELD_CARDS = memory_base.FIELD_CARDINALITIES
PREDICTED_FIELDS = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11)
FIELD_NAMES = {
    0: "event_type",
    1: "channel",
    2: "delta_coarse",
    3: "delta_residual",
    4: "a1",
    5: "a2",
    6: "a3",
    8: "duration_coarse",
    9: "duration_residual",
    10: "continuous_coarse",
    11: "continuous_residual",
}
ACTIVE_FIELDS = {
    0: {0, 1, 2, 3, 4, 6, 8, 9},
    1: {0, 1, 2, 3, 4, 10, 11},
    2: {0, 1, 2, 3, 4},
    3: {0, 1, 2, 3, 4, 5},
    4: {0, 2, 3, 4},
    5: {0, 1, 2, 3, 4},
    6: {0, 1, 2, 3, 10, 11},
    7: {0, 1, 2, 3, 10, 11},
    8: {0, 1, 2, 3, 4, 10, 11},
    9: {0, 2, 3, 4, 5},
}


def active_mask(targets: torch.Tensor, field_index: int) -> torch.Tensor:
    event_type = targets[..., 0]
    mask = torch.zeros_like(event_type, dtype=torch.bool)
    for event, fields in ACTIVE_FIELDS.items():
        if field_index in fields:
            mask |= event_type.eq(event)
    return mask


class FactorizedHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                str(index): nn.Linear(D_MODEL, FIELD_CARDS[index])
                for index in PREDICTED_FIELDS
            }
        )

    def forward(self, hidden: torch.Tensor) -> dict[int, torch.Tensor]:
        return {int(key): head(hidden) for key, head in self.heads.items()}


class _MemoryConditionedBase(nn.Module):
    def __init__(self, *, heads: int = 4) -> None:
        super().__init__()
        self.embedding = memory_base.FactorEmbedding()
        self.memory_norm = nn.LayerNorm(D_MODEL)
        self.memory_attention = nn.MultiheadAttention(
            D_MODEL, heads, dropout=0.0, batch_first=True
        )
        self.post_norm = nn.LayerNorm(D_MODEL)
        self.post_ff = nn.Sequential(
            nn.Linear(D_MODEL, 3 * D_MODEL),
            nn.GELU(),
            nn.Linear(3 * D_MODEL, D_MODEL),
        )
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.factorized_heads = FactorizedHeads()

    def condition_memory(
        self, hidden: torch.Tensor, memory_tokens: torch.Tensor
    ) -> torch.Tensor:
        batch, steps, slots, width = memory_tokens.shape
        query = self.memory_norm(hidden).reshape(batch * steps, 1, width)
        memory = memory_tokens.reshape(batch * steps, slots, width)
        read, _ = self.memory_attention(query, memory, memory, need_weights=False)
        hidden = hidden + read.reshape(batch, steps, width)
        hidden = hidden + self.post_ff(self.post_norm(hidden))
        return hidden


class NoLocalAttentionComposer(_MemoryConditionedBase):
    def forward_chunk(
        self,
        records: torch.Tensor,
        memory_tokens: torch.Tensor,
        history_records: torch.Tensor | None = None,
        *,
        start_index: int = 0,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        del history_records, start_index
        hidden = self.condition_memory(self.embedding(records), memory_tokens)
        return self.factorized_heads(self.output_norm(hidden)), records[:, :0]


class CapacityMatchedNoLocalComposer(NoLocalAttentionComposer):
    """No-local baseline with the same parameter count as window-16 Transformer."""

    def __init__(self, *, heads: int = 4) -> None:
        super().__init__(heads=heads)
        self.capacity_norm = nn.LayerNorm(D_MODEL)
        self.capacity_ff = nn.Sequential(
            nn.Linear(D_MODEL, 249),
            nn.GELU(),
            nn.Linear(249, D_MODEL),
        )
        self.capacity_calibration = nn.Parameter(torch.zeros(87))

    def forward_chunk(
        self,
        records: torch.Tensor,
        memory_tokens: torch.Tensor,
        history_records: torch.Tensor | None = None,
        *,
        start_index: int = 0,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        del history_records, start_index
        hidden = self.condition_memory(self.embedding(records), memory_tokens)
        hidden = hidden + self.capacity_ff(self.capacity_norm(hidden))
        hidden = hidden + self.capacity_calibration.mean() * torch.tanh(hidden)
        return self.factorized_heads(self.output_norm(hidden)), records[:, :0]


class BoundedFactorizedTransformerComposer(_MemoryConditionedBase):
    def __init__(self, *, local_window: int = 16, heads: int = 4) -> None:
        super().__init__(heads=heads)
        self.local_window = local_window
        self.local_position = nn.Embedding(local_window, D_MODEL)
        self.local_norm = nn.LayerNorm(D_MODEL)
        self.local_attention = nn.MultiheadAttention(
            D_MODEL, heads, dropout=0.0, batch_first=True
        )
        self.local_ff_norm = nn.LayerNorm(D_MODEL)
        self.local_ff = nn.Sequential(
            nn.Linear(D_MODEL, 3 * D_MODEL),
            nn.GELU(),
            nn.Linear(3 * D_MODEL, D_MODEL),
        )

    def forward_chunk(
        self,
        records: torch.Tensor,
        memory_tokens: torch.Tensor,
        history_records: torch.Tensor | None,
        *,
        start_index: int,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        if history_records is None:
            full_records = records
            history_length = 0
        else:
            full_records = torch.cat([history_records, records], dim=1)
            history_length = history_records.shape[1]
        total = full_records.shape[1]
        global_first = start_index - history_length
        positions = (
            torch.arange(global_first, global_first + total, device=records.device)
            % self.local_window
        )
        hidden = self.embedding(full_records) + self.local_position(positions)[None]
        normalized = self.local_norm(hidden)
        mask = bounded._local_causal_mask(total, self.local_window, records.device)
        attended, _ = self.local_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.local_ff(self.local_ff_norm(hidden))
        current = hidden[:, history_length:]
        current = self.condition_memory(current, memory_tokens)
        logits = self.factorized_heads(self.output_norm(current))
        keep = min(self.local_window - 1, full_records.shape[1])
        history = full_records[:, -keep:].detach() if keep else full_records[:, :0]
        return logits, history


def factorized_loss(
    logits: dict[int, torch.Tensor], targets: torch.Tensor
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for index in PREDICTED_FIELDS:
        mask = active_mask(targets, index)
        if not mask.any():
            continue
        values = targets[..., index][mask]
        card = FIELD_CARDS[index]
        if int(values.max()) >= card:
            raise ValueError(
                f"active target field {index} exceeds experimental cardinality {card}"
            )
        losses.append(F.cross_entropy(logits[index][mask], values))
    if not losses:
        raise ValueError("batch has no active Compound fields")
    return torch.stack(losses).mean()


@dataclass(frozen=True, slots=True)
class FactorizedMetrics:
    active_field_nll: float
    active_field_accuracy: float
    exact_event_accuracy: float
    event_type_accuracy: float
    note_pitch_accuracy: float
    note_velocity_accuracy: float
    note_duration_pair_accuracy: float
    delta_pair_accuracy: float
    fields_scored: int
    events_scored: int


def _accumulate_metrics(
    logits: dict[int, torch.Tensor],
    targets: torch.Tensor,
    accumulator: dict[str, float],
) -> None:
    predictions = {index: value.argmax(-1) for index, value in logits.items()}
    exact = torch.ones_like(targets[..., 0], dtype=torch.bool)
    for index in PREDICTED_FIELDS:
        mask = active_mask(targets, index)
        if not mask.any():
            continue
        values = targets[..., index]
        card = FIELD_CARDS[index]
        active_values = values[mask]
        if int(active_values.max()) >= card:
            raise ValueError(
                f"active target field {index} exceeds experimental cardinality {card}"
            )
        log_probs = F.log_softmax(logits[index], dim=-1)
        chosen = log_probs.gather(
            -1, values.clamp_max(card - 1).unsqueeze(-1)
        ).squeeze(-1)
        accumulator["nll_sum"] += float((-chosen[mask]).sum())
        accumulator["field_correct"] += int((predictions[index][mask] == values[mask]).sum())
        accumulator["field_total"] += int(mask.sum())
        exact &= (~mask) | predictions[index].eq(values)
    accumulator["exact_correct"] += int(exact.sum())
    accumulator["events"] += targets.shape[0] * targets.shape[1]
    event_pred = predictions[0]
    accumulator["event_correct"] += int(event_pred.eq(targets[..., 0]).sum())

    note = targets[..., 0].eq(0)
    if note.any():
        accumulator["note_pitch_correct"] += int(
            predictions[4][note].eq(targets[..., 4][note]).sum()
        )
        accumulator["note_velocity_correct"] += int(
            predictions[6][note].eq(targets[..., 6][note]).sum()
        )
        accumulator["note_duration_correct"] += int(
            (
                predictions[8][note].eq(targets[..., 8][note])
                & predictions[9][note].eq(targets[..., 9][note])
            ).sum()
        )
        accumulator["notes"] += int(note.sum())
    accumulator["delta_correct"] += int(
        (predictions[2].eq(targets[..., 2]) & predictions[3].eq(targets[..., 3])).sum()
    )


def train_factorized_composer(
    memory_model: matched.RoutedMultiBank,
    composer: nn.Module,
    songs: list[memory_base.PreparedSong],
    *,
    epochs: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
    learning_rate: float,
) -> None:
    for parameter in memory_model.parameters():
        parameter.requires_grad_(False)
    memory_model.eval()
    composer.to(device).train()
    optimizer = torch.optim.AdamW(composer.parameters(), lr=learning_rate)
    rng = random.Random(seed + 2903)
    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            if len(song.records) < 2:
                continue
            state = None
            history = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, chunk_size):
                stop = min(final_input, start + chunk_size)
                records = song.records[start:stop].unsqueeze(0).to(device)
                targets = song.records[start + 1 : stop + 1].unsqueeze(0).to(device)
                with torch.no_grad():
                    memory_tokens, next_state = bounded.routed_memory_reads(
                        memory_model, records, state
                    )
                logits, history = composer.forward_chunk(
                    records,
                    memory_tokens,
                    history,
                    start_index=start,
                )
                loss = factorized_loss(logits, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(composer.parameters(), 1.0)
                optimizer.step()
                state = memory_base._detach_state(next_state)


def evaluate_factorized_composer(
    memory_model: matched.RoutedMultiBank,
    composer: nn.Module,
    songs: list[memory_base.PreparedSong],
    *,
    chunk_size: int,
    device: torch.device,
) -> FactorizedMetrics:
    memory_model.eval()
    composer.eval()
    accumulator = {
        "nll_sum": 0.0,
        "field_correct": 0.0,
        "field_total": 0.0,
        "exact_correct": 0.0,
        "events": 0.0,
        "event_correct": 0.0,
        "note_pitch_correct": 0.0,
        "note_velocity_correct": 0.0,
        "note_duration_correct": 0.0,
        "notes": 0.0,
        "delta_correct": 0.0,
    }
    with torch.no_grad():
        for song in songs:
            if len(song.records) < 2:
                continue
            state = None
            history = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, chunk_size):
                stop = min(final_input, start + chunk_size)
                records = song.records[start:stop].unsqueeze(0).to(device)
                targets = song.records[start + 1 : stop + 1].unsqueeze(0).to(device)
                memory_tokens, next_state = bounded.routed_memory_reads(
                    memory_model, records, state
                )
                logits, history = composer.forward_chunk(
                    records,
                    memory_tokens,
                    history,
                    start_index=start,
                )
                _accumulate_metrics(logits, targets, accumulator)
                state = memory_base._detach_state(next_state)
    field_total = max(1.0, accumulator["field_total"])
    events = max(1.0, accumulator["events"])
    notes = max(1.0, accumulator["notes"])
    return FactorizedMetrics(
        active_field_nll=accumulator["nll_sum"] / field_total,
        active_field_accuracy=accumulator["field_correct"] / field_total,
        exact_event_accuracy=accumulator["exact_correct"] / events,
        event_type_accuracy=accumulator["event_correct"] / events,
        note_pitch_accuracy=accumulator["note_pitch_correct"] / notes,
        note_velocity_accuracy=accumulator["note_velocity_correct"] / notes,
        note_duration_pair_accuracy=accumulator["note_duration_correct"] / notes,
        delta_pair_accuracy=accumulator["delta_correct"] / events,
        fields_scored=int(accumulator["field_total"]),
        events_scored=int(accumulator["events"]),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
