from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from orbitune.compound import (
    COMPOUND_TOKENIZER_ABI,
    CompoundEvent,
    CompoundEventType,
    FactorizedTime,
    TEMPORAL_RESOLUTION,
    canonicalize_events,
    quantize_time,
)
from orbitune.compound_midi import read_compound_midi
from orbitune.compound_training import CompoundSong, load_compound_jsonl, sample_compound_batch
from orbitune.quantization import FactorizedValue, quantize_unsigned
from orbitune.tokenizer.compound_event import CompoundEventTokenizer, CompoundRecord


COMPOUND_BASE_ABI = "orbitune-compound-hierarchical-gpt-v1"
FIELD_CARDINALITIES = (10, 16, 7, 16, 1024, 1024, 128, 256, 7, 16, 8, 8)
EVENT_SLOT_COUNT = 8


@dataclass(slots=True)
class CompoundBaseConfig:
    d_model: int = 224
    n_head: int = 8
    local_layers: int = 4
    medium_layers: int = 2
    global_layers: int = 2
    intra_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.1
    local_window: int = 64
    medium_stride: int = 8
    medium_window: int = 64
    global_stride: int = 4
    global_window: int = 64
    fast_decay: float = 0.90
    medium_decay: float = 0.97
    slow_decay: float = 0.997

    def validate(self) -> None:
        integer_fields = (
            "d_model", "n_head", "local_layers", "medium_layers", "global_layers",
            "intra_layers", "ff_mult", "local_window", "medium_stride", "medium_window",
            "global_stride", "global_window",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2:
            raise ValueError("attention head dimension must be even for rotary positions")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("fast_decay", "medium_decay", "slow_decay"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")


@dataclass
class StreamState:
    local_records: list[torch.Tensor]
    medium_buffer: list[torch.Tensor]
    medium_history: list[torch.Tensor]
    global_buffer: list[torch.Tensor]
    global_history: list[torch.Tensor]
    memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    steps: int = 0


class FactorizedEventEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, d_model) for cardinality in FIELD_CARDINALITIES
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, records: torch.Tensor) -> torch.Tensor:
        if records.ndim != 3 or records.shape[-1] != len(FIELD_CARDINALITIES):
            raise ValueError("records must have shape [batch, time, 12]")
        hidden = torch.zeros(
            *records.shape[:2], self.embeddings[0].embedding_dim,
            device=records.device, dtype=self.embeddings[0].weight.dtype,
        )
        for index, embedding in enumerate(self.embeddings):
            values = records[..., index].clamp(0, FIELD_CARDINALITIES[index] - 1)
            hidden = hidden + embedding(values)
        return self.norm(hidden)


def _apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = q.shape[-1] // 2
    positions = torch.arange(q.shape[-2], device=q.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=q.device, dtype=torch.float32)
        / max(1, half)
    )
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos().to(dtype=q.dtype)[None, None]
    sin = angles.sin().to(dtype=q.dtype)[None, None]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        left, right = x[..., :half], x[..., half:]
        return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)

    return rotate(q), rotate(k)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor | None) -> torch.Tensor:
        batch, steps, width = x.shape
        q = self.q_proj(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(y.transpose(1, 2).contiguous().view(batch, steps, width))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: CompoundBaseConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiheadSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_mult * cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_mult * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_bias)
        return x + self.ff(self.norm2(x))


class TransformerStack(nn.Module):
    def __init__(self, cfg: CompoundBaseConfig, layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor | None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attention_bias)
        return self.norm(x)


def _causal_bias(length: int, device: torch.device, *, window: int | None = None) -> torch.Tensor:
    row = torch.arange(length, device=device)[:, None]
    col = torch.arange(length, device=device)[None, :]
    allowed = col <= row
    if window is not None:
        allowed &= (row - col) < window
    bias = torch.zeros(length, length, device=device, dtype=torch.float32)
    return bias.masked_fill(~allowed, float("-inf"))


def _pool_groups(x: torch.Tensor, stride: int) -> torch.Tensor:
    groups = [x[:, start : min(x.shape[1], start + stride)].mean(dim=1)
              for start in range(0, x.shape[1], stride)]
    return torch.stack(groups, dim=1) if groups else x[:, :0]


def _broadcast_completed(summary_hidden: torch.Tensor, event_steps: int, stride: int) -> torch.Tensor:
    if summary_hidden.shape[1] == 0:
        return summary_hidden.new_zeros(summary_hidden.shape[0], event_steps, summary_hidden.shape[-1])
    positions = torch.arange(event_steps, device=summary_hidden.device)
    completed = (positions + 1) // stride
    index = (completed - 1).clamp_min(0).clamp_max(summary_hidden.shape[1] - 1)
    output = summary_hidden.index_select(1, index)
    return output.masked_fill(completed.eq(0)[None, :, None], 0.0)


class DecayedGRUMemoryBank(nn.Module):
    def __init__(self, d_model: int, decay: float) -> None:
        super().__init__()
        self.decay = float(decay)
        self.norm = nn.LayerNorm(d_model)
        self.cell = nn.GRUCell(d_model, d_model)

    def step(self, x: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if state is None:
            state = x.new_zeros(x.shape[0], x.shape[-1])
        proposal = self.cell(self.norm(x), state)
        return self.decay * state + (1.0 - self.decay) * proposal


class RoutedRecurrentMemory(nn.Module):
    def __init__(self, cfg: CompoundBaseConfig) -> None:
        super().__init__()
        self.fast = DecayedGRUMemoryBank(cfg.d_model, cfg.fast_decay)
        self.medium = DecayedGRUMemoryBank(cfg.d_model, cfg.medium_decay)
        self.slow = DecayedGRUMemoryBank(cfg.d_model, cfg.slow_decay)
        self.fuse = nn.Sequential(
            nn.LayerNorm(3 * cfg.d_model), nn.Linear(3 * cfg.d_model, cfg.d_model), nn.GELU()
        )

    def step(self, x: torch.Tensor,
             state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
             ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        fast_state, medium_state, slow_state = (None, None, None) if state is None else state
        fast_state = self.fast.step(x, fast_state)
        medium_state = self.medium.step(x, medium_state)
        slow_state = self.slow.step(x, slow_state)
        read = self.fuse(torch.cat([fast_state, medium_state, slow_state], dim=-1))
        return read, (fast_state, medium_state, slow_state)

    def forward_sequence(self, x: torch.Tensor,
                         state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                         ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        outputs: list[torch.Tensor] = []
        next_state = state
        for index in range(x.shape[1]):
            read, next_state = self.step(x[:, index], next_state)
            outputs.append(read)
        if not outputs:
            raise ValueError("memory requires at least one event")
        return torch.stack(outputs, dim=1), next_state  # type: ignore[arg-type]


class GaussianHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.proj(x)
        return torch.sigmoid(raw[..., 0]), raw[..., 1].clamp(-5.0, 1.0)

    @staticmethod
    def loss(mean: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (0.5 * (target - mean).square() * torch.exp(-2.0 * log_scale) + log_scale).mean()


class MixedEventDecoder(nn.Module):
    SLOT_EVENT_TYPE = 0
    SLOT_CHANNEL = 1
    SLOT_DELTA = 2
    SLOT_A1 = 3
    SLOT_A2 = 4
    SLOT_VELOCITY = 5
    SLOT_DURATION = 6
    SLOT_CONTROL = 7

    def __init__(self, cfg: CompoundBaseConfig) -> None:
        super().__init__()
        d_model = cfg.d_model
        self.slot_pos = nn.Embedding(EVENT_SLOT_COUNT, d_model)
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.event_type_emb = nn.Embedding(10, d_model)
        self.channel_emb = nn.Embedding(16, d_model)
        self.a1_emb = nn.Embedding(1024, d_model)
        self.a2_emb = nn.Embedding(1024, d_model)
        self.scalar_emb = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
        self.stack = TransformerStack(cfg, cfg.intra_layers)
        self.event_type_head = nn.Linear(d_model, 10)
        self.channel_head = nn.Linear(d_model, 16)
        self.a1_head = nn.Linear(d_model, 1024)
        self.a2_head = nn.Linear(d_model, 1024)
        self.delta_head = GaussianHead(d_model)
        self.velocity_head = GaussianHead(d_model)
        self.duration_head = GaussianHead(d_model)
        self.control_head = GaussianHead(d_model)

    def _time_norm(self, coarse: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor((0, 24, 48, 96, 192, 384, 768, 1536), device=coarse.device, dtype=torch.float32)
        coarse = coarse.clamp(0, 6)
        lo, hi = edges[coarse], edges[coarse + 1]
        value = lo + residual.clamp(0, 15).float() / 15.0 * (hi - lo)
        return (value / 1536.0).clamp(0.0, 1.0)

    @staticmethod
    def _control_norm(coarse: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return (coarse.float().clamp(0, 7) / 8.0 + residual.float().clamp(0, 7) / 56.0).clamp(0.0, 1.0)

    def _targets(self, records: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "event_type": records[..., 0], "channel": records[..., 1],
            "delta": self._time_norm(records[..., 2], records[..., 3]),
            "a1": records[..., 4], "a2": records[..., 5],
            "velocity": (records[..., 6].float() / 127.0).clamp(0.0, 1.0),
            "duration": self._time_norm(records[..., 8], records[..., 9]),
            "control": self._control_norm(records[..., 10], records[..., 11]),
        }

    def _teacher_inputs(self, context: torch.Tensor, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        batch, steps, width = context.shape
        flat = batch * steps
        pieces = [
            self.bos[None].expand(flat, -1),
            self.event_type_emb(targets["event_type"].reshape(-1).clamp(0, 9)),
            self.channel_emb(targets["channel"].reshape(-1).clamp(0, 15)),
            self.scalar_emb(targets["delta"].reshape(-1, 1)),
            self.a1_emb(targets["a1"].reshape(-1).clamp(0, 1023)),
            self.a2_emb(targets["a2"].reshape(-1).clamp(0, 1023)),
            self.scalar_emb(targets["velocity"].reshape(-1, 1)),
            self.scalar_emb(targets["duration"].reshape(-1, 1)),
        ]
        return torch.stack(pieces, dim=1) + context.reshape(flat, width)[:, None, :] + self.slot_pos.weight[None]

    def forward_teacher(self, context: torch.Tensor, records: torch.Tensor) -> dict[str, Any]:
        targets = self._targets(records)
        decoder_input = self._teacher_inputs(context, targets)
        hidden = self.stack(decoder_input, _causal_bias(EVENT_SLOT_COUNT, decoder_input.device))
        batch, steps, width = context.shape
        hidden = hidden.view(batch, steps, EVENT_SLOT_COUNT, width)
        return {
            "targets": targets,
            "event_type": self.event_type_head(hidden[:, :, self.SLOT_EVENT_TYPE]),
            "channel": self.channel_head(hidden[:, :, self.SLOT_CHANNEL]),
            "delta": self.delta_head(hidden[:, :, self.SLOT_DELTA]),
            "a1": self.a1_head(hidden[:, :, self.SLOT_A1]),
            "a2": self.a2_head(hidden[:, :, self.SLOT_A2]),
            "velocity": self.velocity_head(hidden[:, :, self.SLOT_VELOCITY]),
            "duration": self.duration_head(hidden[:, :, self.SLOT_DURATION]),
            "control": self.control_head(hidden[:, :, self.SLOT_CONTROL]),
        }

    def loss(
        self,
        context: torch.Tensor,
        records: torch.Tensor,
        event_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Per-event composite loss for the compound event decoder.

        ``context`` and ``records`` are the standard teacher-forced inputs,
        shapes ``(B, S, d_model)`` and ``(B, S, 12)`` respectively. When
        ``event_weight`` is ``None`` the reduction is the legacy unweighted
        mean (one number per head, then ``.mean()`` across the active
        heads) — this path is bit-identical to the pre-event-weight loss.
        When ``event_weight`` is provided, it must broadcast to ``(B, S)``
        (typically a float tensor of weights in ``[0, 1+]``); each head is
        then reduced as
        ``sum(per_event_loss * head_active_mask * event_weight) / sum(head_active_mask * event_weight)``.
        Setting ``event_weight = 0`` on a position is the correct way to
        mask padding or idle lanes (the head sees no contribution and the
        loss / gradient are unaffected).
        """
        if event_weight is None:
            return self._loss_legacy(context, records)
        return self._loss_weighted(context, records, event_weight)

    def _loss_legacy(
        self,
        context: torch.Tensor,
        records: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        output = self.forward_teacher(context, records)
        target = output["targets"]
        event_type = target["event_type"]
        losses: dict[str, torch.Tensor] = {
            "event_type": F.cross_entropy(output["event_type"].reshape(-1, 10), event_type.reshape(-1)),
            "channel": F.cross_entropy(output["channel"].reshape(-1, 16), target["channel"].reshape(-1)),
            "delta": GaussianHead.loss(*output["delta"], target["delta"]),
        }
        a1_active = torch.zeros_like(event_type, dtype=torch.bool)
        for kind in (0, 1, 2, 3, 4, 5, 8, 9):
            a1_active |= event_type.eq(kind)
        if a1_active.any():
            losses["a1"] = F.cross_entropy(output["a1"][a1_active], target["a1"][a1_active].clamp_max(1023))
        a2_active = event_type.eq(int(CompoundEventType.BANK)) | event_type.eq(int(CompoundEventType.TIME_SIGNATURE))
        if a2_active.any():
            losses["a2"] = F.cross_entropy(output["a2"][a2_active], target["a2"][a2_active].clamp_max(1023))
        note = event_type.eq(int(CompoundEventType.NOTE))
        if note.any():
            losses["velocity"] = GaussianHead.loss(output["velocity"][0][note], output["velocity"][1][note], target["velocity"][note])
            losses["duration"] = GaussianHead.loss(output["duration"][0][note], output["duration"][1][note], target["duration"][note])
        control = event_type.eq(int(CompoundEventType.CC)) | event_type.eq(int(CompoundEventType.PITCH_BEND)) | event_type.eq(int(CompoundEventType.CHANNEL_PRESSURE)) | event_type.eq(int(CompoundEventType.POLY_PRESSURE))
        if control.any():
            losses["control"] = GaussianHead.loss(output["control"][0][control], output["control"][1][control], target["control"][control])
        total = torch.stack(tuple(losses.values())).mean()
        return total, {name: float(value.detach()) for name, value in losses.items()}

    def _loss_weighted(
        self,
        context: torch.Tensor,
        records: torch.Tensor,
        event_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if event_weight.shape != context.shape[:2]:
            raise ValueError(
                f"event_weight shape {tuple(event_weight.shape)} must match context[:2] {tuple(context.shape[:2])}"
            )
        output = self.forward_teacher(context, records)
        target = output["targets"]
        event_type = target["event_type"]
        B, S = event_type.shape
        ew = event_weight.to(dtype=context.dtype)

        def _reduce_event_type(per_event: torch.Tensor) -> torch.Tensor:
            w = ew.reshape(-1)
            denom = w.sum().clamp_min(1e-12)
            return (per_event.reshape(-1) * w).sum() / denom

        def _reduce_full(per_event: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            m_flat = mask.to(dtype=ew.dtype)
            w = ew * m_flat
            denom = w.sum().clamp_min(1e-12)
            return (per_event * w).sum() / denom

        losses: dict[str, torch.Tensor] = {
            "event_type": _reduce_event_type(
                F.cross_entropy(output["event_type"].reshape(-1, 10), event_type.reshape(-1), reduction="none")
            ),
            "channel": _reduce_event_type(
                F.cross_entropy(output["channel"].reshape(-1, 16), target["channel"].reshape(-1), reduction="none")
            ),
            "delta": _reduce_event_type(
                0.5 * (target["delta"] - output["delta"][0]).square() * torch.exp(-2.0 * output["delta"][1]) + output["delta"][1]
            ),
        }
        a1_active = torch.zeros_like(event_type, dtype=torch.bool)
        for kind in (0, 1, 2, 3, 4, 5, 8, 9):
            a1_active |= event_type.eq(kind)
        if a1_active.any():
            losses["a1"] = _reduce_full(
                F.cross_entropy(
                    output["a1"].reshape(-1, 1024),
                    target["a1"].clamp_max(1023).reshape(-1),
                    reduction="none",
                ).reshape(B, S),
                a1_active,
            )
        a2_active = event_type.eq(int(CompoundEventType.BANK)) | event_type.eq(int(CompoundEventType.TIME_SIGNATURE))
        if a2_active.any():
            losses["a2"] = _reduce_full(
                F.cross_entropy(
                    output["a2"].reshape(-1, 1024),
                    target["a2"].clamp_max(1023).reshape(-1),
                    reduction="none",
                ).reshape(B, S),
                a2_active,
            )
        note = event_type.eq(int(CompoundEventType.NOTE))
        if note.any():
            losses["velocity"] = _reduce_full(
                0.5 * (target["velocity"] - output["velocity"][0]).square() * torch.exp(-2.0 * output["velocity"][1]) + output["velocity"][1],
                note,
            )
            losses["duration"] = _reduce_full(
                0.5 * (target["duration"] - output["duration"][0]).square() * torch.exp(-2.0 * output["duration"][1]) + output["duration"][1],
                note,
            )
        control = event_type.eq(int(CompoundEventType.CC)) | event_type.eq(int(CompoundEventType.PITCH_BEND)) | event_type.eq(int(CompoundEventType.CHANNEL_PRESSURE)) | event_type.eq(int(CompoundEventType.POLY_PRESSURE))
        if control.any():
            losses["control"] = _reduce_full(
                0.5 * (target["control"] - output["control"][0]).square() * torch.exp(-2.0 * output["control"][1]) + output["control"][1],
                control,
            )
        total = torch.stack(tuple(losses.values())).mean()
        return total, {name: float(value.detach()) for name, value in losses.items()}

    @staticmethod
    def _sample_categorical(logits: torch.Tensor, temperature: float, top_p: float) -> int:
        probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            sorted_probs = sorted_probs.masked_fill(cumulative - sorted_probs > top_p, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
            return int(sorted_idx[torch.multinomial(sorted_probs, 1)].item())
        return int(torch.multinomial(probs, 1).item())

    @staticmethod
    def _sample_gaussian(mean: torch.Tensor, log_scale: torch.Tensor, temperature: float) -> float:
        if temperature <= 0.0:
            return float(mean.item())
        return float((mean + torch.randn_like(mean) * log_scale.exp() * temperature).clamp(0.0, 1.0).item())

    def _decode_hidden(self, context: torch.Tensor, previous: list[torch.Tensor]) -> torch.Tensor:
        if context.shape[0] != 1:
            raise ValueError("generation context must have batch size 1")
        x = torch.stack([self.bos, *previous], dim=0)
        positions = torch.arange(x.shape[0], device=x.device)
        x = x + context[0][None, :] + self.slot_pos(positions)
        return self.stack(x[None], _causal_bias(x.shape[0], x.device))[0, -1]

    def sample(self, context: torch.Tensor, *, temperature: float, top_p: float) -> CompoundRecord:
        previous: list[torch.Tensor] = []
        hidden = self._decode_hidden(context, previous)
        event_type = self._sample_categorical(self.event_type_head(hidden), temperature, top_p)
        previous.append(self.event_type_emb(torch.tensor(event_type, device=context.device)))
        hidden = self._decode_hidden(context, previous)
        channel = 0 if event_type in (int(CompoundEventType.TEMPO), int(CompoundEventType.TIME_SIGNATURE)) else self._sample_categorical(self.channel_head(hidden), temperature, top_p)
        previous.append(self.channel_emb(torch.tensor(channel, device=context.device)))
        hidden = self._decode_hidden(context, previous)
        delta_value = self._sample_gaussian(*self.delta_head(hidden), temperature)
        delta = quantize_time(max(0, int(round(delta_value * 1536.0))))
        previous.append(self.scalar_emb(torch.tensor([[delta_value]], device=context.device))[0])
        hidden = self._decode_hidden(context, previous)
        a1_logits = self.a1_head(hidden)
        if event_type in (int(CompoundEventType.NOTE), int(CompoundEventType.CC), int(CompoundEventType.PROGRAM), int(CompoundEventType.BANK), int(CompoundEventType.POLY_PRESSURE)):
            a1_logits[128:] = float("-inf")
        elif event_type == int(CompoundEventType.TEMPO):
            a1_logits[0] = float("-inf"); a1_logits[1000:] = float("-inf")
        elif event_type == int(CompoundEventType.PEDAL):
            a1_logits[2:] = float("-inf")
        elif event_type == int(CompoundEventType.TIME_SIGNATURE):
            a1_logits[0] = float("-inf"); a1_logits[256:] = float("-inf")
        else:
            a1_logits[1:] = float("-inf")
        a1 = self._sample_categorical(a1_logits, temperature, top_p)
        previous.append(self.a1_emb(torch.tensor(a1, device=context.device)))
        hidden = self._decode_hidden(context, previous)
        if event_type == int(CompoundEventType.BANK):
            a2_logits = self.a2_head(hidden); a2_logits[128:] = float("-inf")
            a2 = self._sample_categorical(a2_logits, temperature, top_p)
        elif event_type == int(CompoundEventType.TIME_SIGNATURE):
            a2_logits = self.a2_head(hidden)
            mask = torch.ones_like(a2_logits, dtype=torch.bool)
            for value in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                mask[value] = False
            a2_logits[mask] = float("-inf")
            a2 = self._sample_categorical(a2_logits, temperature, top_p)
        else:
            a2 = 0
        previous.append(self.a2_emb(torch.tensor(a2, device=context.device)))
        hidden = self._decode_hidden(context, previous)
        velocity = 0; velocity_norm = 0.0
        if event_type == int(CompoundEventType.NOTE):
            velocity_norm = self._sample_gaussian(*self.velocity_head(hidden), temperature)
            velocity = max(1, min(127, int(round(velocity_norm * 127.0))))
        previous.append(self.scalar_emb(torch.tensor([[velocity_norm]], device=context.device))[0])
        hidden = self._decode_hidden(context, previous)
        duration = FactorizedTime(0, 0); duration_norm = 0.0
        if event_type == int(CompoundEventType.NOTE):
            duration_norm = self._sample_gaussian(*self.duration_head(hidden), temperature)
            duration = quantize_time(max(1, int(round(duration_norm * 1536.0))))
        previous.append(self.scalar_emb(torch.tensor([[duration_norm]], device=context.device))[0])
        hidden = self._decode_hidden(context, previous)
        control = FactorizedValue(0, 0)
        if event_type in (int(CompoundEventType.CC), int(CompoundEventType.PITCH_BEND), int(CompoundEventType.CHANNEL_PRESSURE), int(CompoundEventType.POLY_PRESSURE)):
            control_norm = self._sample_gaussian(*self.control_head(hidden), temperature)
            control = quantize_unsigned(int(round(control_norm * 16383.0)), maximum=16383)
        return self._build_record(event_type, channel, delta, a1, a2, velocity, duration, control)

    @staticmethod
    def _build_record(
        event_type: int,
        channel: int,
        delta: "FactorizedTime",
        a1: int,
        a2: int,
        velocity: int,
        duration: "FactorizedTime",
        control: "FactorizedValue",
    ) -> "CompoundRecord":
        """Assemble a CompoundRecord and zero out slots that are unused for
        the sampled event type so the output satisfies the CompoundEvent
        ABI validation rules (see ``orbitune.compound.CompoundEvent.validate``).
        Without this normalisation, a CC / TEMPO / PROGRAM event produced
        by the model can carry leftover values in ``a3``/``a4`` (which the
        heads above may have set for NOTE events) and fail the MIDI
        write-path validation in ``tokenizer.decode_records``.
        """
        v = velocity if event_type == int(CompoundEventType.NOTE) else 0
        a4 = 0  # currently unused for every event type in the ABI
        a3 = v
        d_coarse, d_resid = (duration.coarse, duration.residual) if event_type == int(CompoundEventType.NOTE) else (0, 0)
        c_coarse = control.coarse if event_type in (
            int(CompoundEventType.CC),
            int(CompoundEventType.PITCH_BEND),
            int(CompoundEventType.CHANNEL_PRESSURE),
            int(CompoundEventType.POLY_PRESSURE),
        ) else 0
        c_resid = control.residual if event_type in (
            int(CompoundEventType.CC),
            int(CompoundEventType.PITCH_BEND),
            int(CompoundEventType.CHANNEL_PRESSURE),
            int(CompoundEventType.POLY_PRESSURE),
        ) else 0
        # Slots a2, a3, a4 are unused for some event types; zero them so
        # the record passes CompoundEvent.validate() and MIDI round-trip
        # works end-to-end. NOTE / BANK / TIME_SIGNATURE / POLY_PRESSURE
        # keep a2; everything else forces a2=0.
        keep_a2 = event_type in (
            int(CompoundEventType.NOTE),  # a2 used for duration coarse
            int(CompoundEventType.BANK),
            int(CompoundEventType.TIME_SIGNATURE),
            int(CompoundEventType.POLY_PRESSURE),
        )
        a2_final = a2 if keep_a2 else 0
        return CompoundRecord(
            event_type=event_type,
            channel=channel,
            delta_coarse=delta.coarse,
            delta_residual=delta.residual,
            a1=a1,
            a2=a2_final,
            a3=a3,
            a4=a4,
            duration_coarse=d_coarse,
            duration_residual=d_resid,
            continuous_coarse=c_coarse,
            continuous_residual=c_resid,
        )


class CompoundHierarchicalGPT(nn.Module):
    architecture = COMPOUND_BASE_ABI
    tokenizer = COMPOUND_TOKENIZER_ABI

    def __init__(self, cfg: CompoundBaseConfig | None = None) -> None:
        super().__init__()
        self.config = cfg or CompoundBaseConfig()
        self.config.validate()
        cfg = self.config
        self.embedding = FactorizedEventEmbedding(cfg.d_model)
        self.local = TransformerStack(cfg, cfg.local_layers)
        self.medium = TransformerStack(cfg, cfg.medium_layers)
        self.global_stack = TransformerStack(cfg, cfg.global_layers)
        self.memory = RoutedRecurrentMemory(cfg)
        self.fusion = nn.Sequential(nn.LayerNorm(4 * cfg.d_model), nn.Linear(4 * cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.decoder = MixedEventDecoder(cfg)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(self, records: torch.Tensor) -> torch.Tensor:
        event = self.embedding(records)
        local = self.local(event, _causal_bias(records.shape[1], records.device, window=self.config.local_window))
        medium_summary = _pool_groups(local, self.config.medium_stride)
        medium_hidden = self.medium(medium_summary, _causal_bias(medium_summary.shape[1], records.device))
        medium_context = _broadcast_completed(medium_hidden, records.shape[1], self.config.medium_stride)
        global_summary = _pool_groups(medium_hidden, self.config.global_stride)
        global_hidden = self.global_stack(global_summary, _causal_bias(global_summary.shape[1], records.device))
        global_context = _broadcast_completed(global_hidden, records.shape[1], self.config.medium_stride * self.config.global_stride)
        memory_context, _ = self.memory.forward_sequence(event)
        return self.fusion(torch.cat([local, medium_context, global_context, memory_context], dim=-1))

    def forward(self, records: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        if targets is None:
            raise ValueError("training forward requires targets")
        return self.decoder.loss(self.encode(records), targets)

    def initial_stream_state(self) -> StreamState:
        return StreamState([], [], [], [], [], None, 0)

    @torch.no_grad()
    def advance_stream(self, record: CompoundRecord | torch.Tensor, state: StreamState) -> torch.Tensor:
        device = next(self.parameters()).device
        raw = torch.tensor(record.as_tuple(), dtype=torch.long, device=device) if isinstance(record, CompoundRecord) else record.to(device=device, dtype=torch.long).reshape(12)
        state.local_records.append(raw.detach())
        if len(state.local_records) > self.config.local_window:
            state.local_records.pop(0)
        local_records = torch.stack(state.local_records)[None]
        local_event = self.embedding(local_records)
        local_hidden = self.local(local_event, _causal_bias(local_event.shape[1], device, window=self.config.local_window))[0, -1]
        event_emb = self.embedding(raw[None, None])[0, 0]
        memory_read, state.memory = self.memory.step(event_emb[None], state.memory)
        memory_read = memory_read[0]
        state.medium_buffer.append(local_hidden.detach())
        if len(state.medium_buffer) >= self.config.medium_stride:
            summary = torch.stack(state.medium_buffer).mean(dim=0)
            state.medium_buffer.clear(); state.medium_history.append(summary.detach())
            if len(state.medium_history) > self.config.medium_window:
                state.medium_history.pop(0)
            medium_sequence = torch.stack(state.medium_history)[None]
            medium_out = self.medium(medium_sequence, _causal_bias(medium_sequence.shape[1], device))[0, -1]
            state.global_buffer.append(medium_out.detach())
            if len(state.global_buffer) >= self.config.global_stride:
                global_summary = torch.stack(state.global_buffer).mean(dim=0)
                state.global_buffer.clear(); state.global_history.append(global_summary.detach())
                if len(state.global_history) > self.config.global_window:
                    state.global_history.pop(0)
        medium_context = local_hidden.new_zeros(local_hidden.shape)
        if state.medium_history:
            sequence = torch.stack(state.medium_history)[None]
            medium_context = self.medium(sequence, _causal_bias(sequence.shape[1], device))[0, -1]
        global_context = local_hidden.new_zeros(local_hidden.shape)
        if state.global_history:
            sequence = torch.stack(state.global_history)[None]
            global_context = self.global_stack(sequence, _causal_bias(sequence.shape[1], device))[0, -1]
        state.steps += 1
        return self.fusion(torch.cat([local_hidden, medium_context, global_context, memory_read], dim=-1))[None]

    @torch.no_grad()
    def generate_records(self, primer: Iterable[CompoundRecord], *, max_new_events: int,
                         temperature: float = 0.85, top_p: float = 0.92) -> list[CompoundRecord]:
        if max_new_events < 0:
            raise ValueError("max_new_events must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        state = self.initial_stream_state()
        output = list(primer)
        if not output:
            output = [CompoundRecord(event_type=int(CompoundEventType.TEMPO), channel=0,
                                     delta_coarse=0, delta_residual=0, a1=120, a2=0, a3=0, a4=0)]
        context = None
        for record in output:
            context = self.advance_stream(record, state)
        assert context is not None
        for _ in range(max_new_events):
            record = self.decoder.sample(context, temperature=temperature, top_p=top_p)
            output.append(record)
            context = self.advance_stream(record, state)
        return output

    def checkpoint_payload(self, *, optimizer: torch.optim.Optimizer | None = None, step: int = 0,
                           source_commit: str | None = None, sampler_rng_state: object | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1, "architecture": self.architecture, "tokenizer": self.tokenizer,
            "config": asdict(self.config), "model_state_dict": self.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(), "step": int(step),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(), "sampler_rng_state": sampler_rng_state,
            "source_commit": source_commit,
        }

    def save_checkpoint(self, path: str | Path, *, optimizer: torch.optim.Optimizer | None = None,
                        step: int = 0, source_commit: str | None = None,
                        sampler_rng_state: object | None = None) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(optimizer=optimizer, step=step, source_commit=source_commit,
                                           sampler_rng_state=sampler_rng_state), target)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple["CompoundHierarchicalGPT", dict[str, Any]]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must contain a mapping")
        if payload.get("architecture") != cls.architecture:
            raise ValueError("checkpoint architecture mismatch")
        if payload.get("tokenizer") != cls.tokenizer:
            raise ValueError("checkpoint tokenizer mismatch")
        raw_config = payload.get("config")
        if not isinstance(raw_config, dict):
            raise ValueError("checkpoint config missing")
        model = cls(CompoundBaseConfig(**raw_config)); model.load_state_dict(payload["model_state_dict"])
        return model, payload


def _write_vlq(value: int) -> bytes:
    if value < 0:
        raise ValueError("VLQ value must be non-negative")
    buffer = [value & 0x7F]; value >>= 7
    while value:
        buffer.append(0x80 | (value & 0x7F)); value >>= 7
    return bytes(reversed(buffer))


def write_compound_midi(path: str | Path, events: Iterable[CompoundEvent], *, division: int = TEMPORAL_RESOLUTION) -> None:
    timeline: list[tuple[int, int, bytes]] = []
    for event in canonicalize_events(events):
        tick = round(event.step * division / TEMPORAL_RESOLUTION)
        if event.type is CompoundEventType.NOTE:
            timeline.append((tick, 20, bytes([0x90 | event.channel, event.a1, event.a3])))
            off_tick = round((event.step + event.a2) * division / TEMPORAL_RESOLUTION)
            timeline.append((off_tick, 10, bytes([0x80 | event.channel, event.a1, 0])))
        elif event.type is CompoundEventType.CC:
            timeline.append((tick, 5, bytes([0xB0 | event.channel, event.a1, event.a2])))
        elif event.type is CompoundEventType.PROGRAM:
            timeline.append((tick, 4, bytes([0xC0 | event.channel, event.a1])))
        elif event.type is CompoundEventType.BANK:
            timeline.append((tick, 2, bytes([0xB0 | event.channel, 0, event.a1])))
            timeline.append((tick, 3, bytes([0xB0 | event.channel, 32, event.a2])))
        elif event.type is CompoundEventType.TEMPO:
            micros = max(1, round(60_000_000 / event.a1))
            timeline.append((tick, 0, b"\xff\x51\x03" + micros.to_bytes(3, "big")))
        elif event.type is CompoundEventType.PEDAL:
            timeline.append((tick, 5, bytes([0xB0 | event.channel, 64, 127 if event.a1 else 0])))
        elif event.type is CompoundEventType.PITCH_BEND:
            timeline.append((tick, 6, bytes([0xE0 | event.channel, event.a1 & 0x7F, (event.a1 >> 7) & 0x7F])))
        elif event.type is CompoundEventType.CHANNEL_PRESSURE:
            timeline.append((tick, 6, bytes([0xD0 | event.channel, event.a1])))
        elif event.type is CompoundEventType.POLY_PRESSURE:
            timeline.append((tick, 6, bytes([0xA0 | event.channel, event.a1, event.a2])))
        elif event.type is CompoundEventType.TIME_SIGNATURE:
            denominator_power = int(round(math.log2(event.a2)))
            timeline.append((tick, 0, bytes([0xFF, 0x58, 0x04, event.a1, denominator_power, 24, 8])))
    timeline.sort(key=lambda item: (item[0], item[1], item[2]))
    track = bytearray(); previous = 0
    for tick, _, message in timeline:
        track += _write_vlq(max(0, tick - previous)) + message; previous = tick
    track += b"\x00\xff\x2f\x00"
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + division.to_bytes(2, "big")
    data = header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    return torch.device(name)


def _load_config(path: str | None) -> CompoundBaseConfig:
    if not path:
        return CompoundBaseConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config JSON must contain an object")
    return CompoundBaseConfig(**raw)


def _validation_loss(model: CompoundHierarchicalGPT, songs: list[CompoundSong], *, batch_size: int,
                     seq_len: int, device: torch.device, seed: int, batches: int = 4) -> float:
    rng = random.Random(seed); model.eval(); values: list[float] = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = sample_compound_batch(songs, batch_size=batch_size, seq_len=seq_len, rng=rng, device=device)
            loss, _ = model(inputs, targets); values.append(float(loss))
    model.train()
    return sum(values) / len(values)


def train_command(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)
    train_songs = load_compound_jsonl(args.train_jsonl)
    validation_songs = load_compound_jsonl(args.validation_jsonl) if args.validation_jsonl else None
    start_step = 0; payload: dict[str, Any] = {}
    if args.resume:
        model, payload = CompoundHierarchicalGPT.load_checkpoint(args.resume, map_location=device); model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload.get("step", 0))
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if payload.get("python_rng_state") is not None:
            random.setstate(payload["python_rng_state"])
    else:
        model = CompoundHierarchicalGPT(_load_config(args.config)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed + 7919)
    if args.resume and payload.get("sampler_rng_state") is not None:
        rng.setstate(payload["sampler_rng_state"])
    if args.resume and payload.get("cuda_rng_state_all") is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    model.train(); checkpoint = Path(args.checkpoint)
    source_commit = os.environ.get("GITHUB_SHA") or os.environ.get("ORBITUNE_SOURCE_COMMIT")
    for step in range(start_step + 1, args.steps + 1):
        inputs, targets = sample_compound_batch(train_songs, batch_size=args.batch_size, seq_len=args.seq_len, rng=rng, device=device)
        loss, components = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
        if step == 1 or step % args.log_every == 0:
            message: dict[str, Any] = {"step": step, "loss": float(loss.detach()), "parameters": model.parameter_count(), "device": str(device), "components": components}
            if validation_songs and step % args.eval_every == 0:
                message["validation_loss"] = _validation_loss(model, validation_songs, batch_size=max(1, min(args.batch_size, 4)), seq_len=args.seq_len, device=device, seed=args.seed + step)
            print(json.dumps(message, sort_keys=True))
        if step % args.checkpoint_every == 0 or step == args.steps:
            model.save_checkpoint(checkpoint, optimizer=optimizer, step=step, source_commit=source_commit, sampler_rng_state=rng.getstate())


def generate_command(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    model, payload = CompoundHierarchicalGPT.load_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval(); tokenizer = CompoundEventTokenizer(); primer: list[CompoundRecord] = []
    if args.primer_midi:
        primer = tokenizer.encode_events(read_compound_midi(args.primer_midi))
    records = model.generate_records(primer, max_new_events=args.events, temperature=args.temperature, top_p=args.top_p)
    write_compound_midi(args.out, tokenizer.decode_records(records))
    print(json.dumps({"checkpoint_step": int(payload.get("step", 0)), "generated_events": len(records), "out": str(Path(args.out)), "device": str(device)}, sort_keys=True))


def inspect_command(args: argparse.Namespace) -> None:
    if args.checkpoint:
        model, payload = CompoundHierarchicalGPT.load_checkpoint(args.checkpoint, map_location="cpu")
        result = {"architecture": model.architecture, "tokenizer": model.tokenizer, "parameters": model.parameter_count(), "config": asdict(model.config), "step": int(payload.get("step", 0))}
    else:
        model = CompoundHierarchicalGPT(_load_config(args.config))
        result = {"architecture": model.architecture, "tokenizer": model.tokenizer, "parameters": model.parameter_count(), "config": asdict(model.config)}
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orbitune hierarchical Compound MIDI GPT")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect"); inspect.add_argument("--config"); inspect.add_argument("--checkpoint"); inspect.set_defaults(func=inspect_command)
    train = sub.add_parser("train")
    train.add_argument("--train-jsonl", required=True); train.add_argument("--validation-jsonl"); train.add_argument("--config")
    train.add_argument("--checkpoint", required=True); train.add_argument("--resume"); train.add_argument("--steps", type=int, default=10_000)
    train.add_argument("--batch-size", type=int, default=8); train.add_argument("--seq-len", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4); train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--grad-clip", type=float, default=1.0); train.add_argument("--checkpoint-every", type=int, default=250)
    train.add_argument("--log-every", type=int, default=25); train.add_argument("--eval-every", type=int, default=250)
    train.add_argument("--seed", type=int, default=1); train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.set_defaults(func=train_command)
    generate = sub.add_parser("generate")
    generate.add_argument("--checkpoint", required=True); generate.add_argument("--out", required=True); generate.add_argument("--primer-midi")
    generate.add_argument("--events", type=int, default=512); generate.add_argument("--temperature", type=float, default=0.85); generate.add_argument("--top-p", type=float, default=0.92)
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); generate.set_defaults(func=generate_command)
    return parser


def main() -> None:
    args = build_parser().parse_args(); args.func(args)


if __name__ == "__main__":
    main()
