from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import real_compound_memory_experiment as memory_base  # noqa: E402
import real_compound_memory_experiment_matched as matched  # noqa: E402


D_MODEL = memory_base.D_MODEL


def _detach_state(state):  # type: ignore[no-untyped-def]
    return memory_base._detach_state(state)


def routed_memory_reads(
    model: matched.RoutedMultiBank,
    records: torch.Tensor,
    state,
):  # type: ignore[no-untyped-def]
    """Read fast/medium/slow memories while preserving streaming state."""

    if state is None:
        state = (None, None, None)
    hidden = model.embedding(records)
    fast, fast_state = model.fast_memory.forward_chunk(hidden, state[0])
    medium, medium_state = model.medium_memory.forward_chunk(hidden, state[1])
    slow, slow_state = model.slow_memory.forward_chunk(hidden, state[2])
    tokens = torch.stack([fast, medium, slow], dim=2)
    return tokens, (fast_state, medium_state, slow_state)


def _local_causal_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    if window <= 0:
        raise ValueError("local window must be positive")
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    allowed = (key <= query) & ((query - key) < window)
    return ~allowed


class BoundedLocalTransformerComposer(nn.Module):
    """Tiny bounded causal composer with explicit recurrent-memory cross-read.

    Long history never enters self-attention. Only the last ``local_window``
    Compound events are visible. Three fixed-size recurrent memory readouts
    (fast/medium/slow) are supplied as per-step cross-attention tokens.
    """

    def __init__(self, *, local_window: int = 16, heads: int = 4) -> None:
        super().__init__()
        if D_MODEL % heads:
            raise ValueError("D_MODEL must be divisible by heads")
        self.local_window = local_window
        self.embedding = memory_base.FactorEmbedding()
        self.local_position = nn.Embedding(local_window, D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.self_attention = nn.MultiheadAttention(
            D_MODEL, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, 3 * D_MODEL),
            nn.GELU(),
            nn.Linear(3 * D_MODEL, D_MODEL),
        )
        self.norm3 = nn.LayerNorm(D_MODEL)
        self.memory_attention = nn.MultiheadAttention(
            D_MODEL, heads, dropout=0.0, batch_first=True
        )
        self.norm4 = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 10)

    def forward_chunk(
        self,
        records: torch.Tensor,
        memory_tokens: torch.Tensor,
        history_records: torch.Tensor | None,
        *,
        start_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if records.ndim != 3:
            raise ValueError("records must have shape [batch,time,fields]")
        if memory_tokens.shape[:3] != (*records.shape[:2], 3):
            raise ValueError("memory_tokens must have shape [batch,time,3,d_model]")
        if history_records is None:
            full_records = records
            history_length = 0
        else:
            if history_records.shape[0] != records.shape[0]:
                raise ValueError("history batch mismatch")
            full_records = torch.cat([history_records, records], dim=1)
            history_length = history_records.shape[1]

        total = full_records.shape[1]
        global_first = start_index - history_length
        positions = (
            torch.arange(global_first, global_first + total, device=records.device)
            % self.local_window
        )
        hidden = self.embedding(full_records) + self.local_position(positions)[None]
        normalized = self.norm1(hidden)
        mask = _local_causal_mask(total, self.local_window, records.device)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.ff(self.norm2(hidden))
        current = hidden[:, history_length:]

        batch, steps, slots, width = memory_tokens.shape
        query = self.norm3(current).reshape(batch * steps, 1, width)
        memory = memory_tokens.reshape(batch * steps, slots, width)
        memory_read, _ = self.memory_attention(
            query, memory, memory, need_weights=False
        )
        current = current + memory_read.reshape(batch, steps, width)
        logits = self.head(self.norm4(current))

        keep = min(self.local_window - 1, full_records.shape[1])
        next_history = full_records[:, -keep:].detach() if keep else full_records[:, :0]
        return logits, next_history


def train_bounded_composer(
    memory_model: matched.RoutedMultiBank,
    composer: BoundedLocalTransformerComposer,
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
    rng = random.Random(seed + 1701)

    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            if len(song.records) < 2:
                continue
            memory_state = None
            history = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, chunk_size):
                stop = min(final_input, start + chunk_size)
                records = song.records[start:stop].unsqueeze(0).to(device)
                targets = song.records[start + 1 : stop + 1, 0].unsqueeze(0).to(device)
                with torch.no_grad():
                    memory_tokens, next_state = routed_memory_reads(
                        memory_model, records, memory_state
                    )
                logits, history = composer.forward_chunk(
                    records,
                    memory_tokens,
                    history,
                    start_index=start,
                )
                loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(composer.parameters(), 1.0)
                optimizer.step()
                memory_state = _detach_state(next_state)


def evaluate_bounded_composer(
    memory_model: matched.RoutedMultiBank,
    composer: BoundedLocalTransformerComposer,
    songs: list[memory_base.PreparedSong],
    *,
    chunk_size: int,
    device: torch.device,
) -> float:
    memory_model.eval()
    composer.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for song in songs:
            if len(song.records) < 2:
                continue
            memory_state = None
            history = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, chunk_size):
                stop = min(final_input, start + chunk_size)
                records = song.records[start:stop].unsqueeze(0).to(device)
                targets = song.records[start + 1 : stop + 1, 0].to(device)
                memory_tokens, next_state = routed_memory_reads(
                    memory_model, records, memory_state
                )
                logits, history = composer.forward_chunk(
                    records,
                    memory_tokens,
                    history,
                    start_index=start,
                )
                prediction = logits[0].argmax(-1)
                correct += int((prediction == targets).sum())
                total += targets.numel()
                memory_state = _detach_state(next_state)
    return correct / total if total else 0.0


def composer_parameter_count(composer: nn.Module) -> int:
    return sum(parameter.numel() for parameter in composer.parameters())
