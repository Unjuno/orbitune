from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from orbitune.compound import CompoundEventType, FactorizedTime, quantize_time
from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.quantization import FactorizedValue, quantize_unsigned
from orbitune.tokenizer.compound_event import CompoundRecord


DEFAULT_SEED = CompoundRecord(
    event_type=int(CompoundEventType.TEMPO), channel=0,
    delta_coarse=0, delta_residual=0, a1=120, a2=0, a3=0, a4=0,
)


def _argmax_allowed(logits: torch.Tensor, allowed: range | tuple[int, ...] | list[int]) -> int:
    values = logits.reshape(-1)
    iterator = iter(allowed)
    best = next(iterator)
    best_value = float(values[best])
    for index in iterator:
        value = float(values[index])
        if value > best_value:
            best = index
            best_value = value
    return int(best)


def _a1_allowed(event_type: int):
    if event_type in (
        int(CompoundEventType.NOTE), int(CompoundEventType.CC), int(CompoundEventType.PROGRAM),
        int(CompoundEventType.BANK), int(CompoundEventType.POLY_PRESSURE),
    ):
        return range(128)
    if event_type == int(CompoundEventType.TEMPO):
        return range(1, 1000)
    if event_type == int(CompoundEventType.PEDAL):
        return (0, 1)
    if event_type == int(CompoundEventType.TIME_SIGNATURE):
        return range(1, 256)
    return (0,)


def _a2_allowed(event_type: int):
    if event_type == int(CompoundEventType.BANK):
        return range(128)
    if event_type == int(CompoundEventType.TIME_SIGNATURE):
        return (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    return (0,)


def greedy_next_record(model: CompoundHierarchicalGPT, context: torch.Tensor) -> CompoundRecord:
    decoder = model.decoder
    previous: list[torch.Tensor] = []

    hidden = decoder._decode_hidden(context, previous)
    event_type = _argmax_allowed(decoder.event_type_head(hidden), range(10))
    previous.append(decoder.event_type_emb(torch.tensor(event_type, device=context.device)))

    hidden = decoder._decode_hidden(context, previous)
    if event_type in (int(CompoundEventType.TEMPO), int(CompoundEventType.TIME_SIGNATURE)):
        channel = 0
    else:
        channel = _argmax_allowed(decoder.channel_head(hidden), range(16))
    previous.append(decoder.channel_emb(torch.tensor(channel, device=context.device)))

    hidden = decoder._decode_hidden(context, previous)
    delta_norm = float(decoder.delta_head(hidden)[0].item())
    delta = quantize_time(max(0, int(round(delta_norm * 1536.0))))
    previous.append(decoder.scalar_emb(torch.tensor([[delta_norm]], device=context.device))[0])

    hidden = decoder._decode_hidden(context, previous)
    a1 = _argmax_allowed(decoder.a1_head(hidden), _a1_allowed(event_type))
    previous.append(decoder.a1_emb(torch.tensor(a1, device=context.device)))

    hidden = decoder._decode_hidden(context, previous)
    a2 = _argmax_allowed(decoder.a2_head(hidden), _a2_allowed(event_type))
    previous.append(decoder.a2_emb(torch.tensor(a2, device=context.device)))

    hidden = decoder._decode_hidden(context, previous)
    velocity = 0
    velocity_norm = 0.0
    if event_type == int(CompoundEventType.NOTE):
        velocity_norm = float(decoder.velocity_head(hidden)[0].item())
        velocity = max(1, min(127, int(round(velocity_norm * 127.0))))
    previous.append(decoder.scalar_emb(torch.tensor([[velocity_norm]], device=context.device))[0])

    hidden = decoder._decode_hidden(context, previous)
    duration = FactorizedTime(0, 0)
    duration_norm = 0.0
    if event_type == int(CompoundEventType.NOTE):
        duration_norm = float(decoder.duration_head(hidden)[0].item())
        duration = quantize_time(max(1, int(round(duration_norm * 1536.0))))
    previous.append(decoder.scalar_emb(torch.tensor([[duration_norm]], device=context.device))[0])

    hidden = decoder._decode_hidden(context, previous)
    control = FactorizedValue(0, 0)
    if event_type in (
        int(CompoundEventType.CC), int(CompoundEventType.PITCH_BEND),
        int(CompoundEventType.CHANNEL_PRESSURE), int(CompoundEventType.POLY_PRESSURE),
    ):
        control_norm = float(decoder.control_head(hidden)[0].item())
        control = quantize_unsigned(int(round(control_norm * 16383.0)), maximum=16383)

    return decoder._build_record(event_type, channel, delta, a1, a2, velocity, duration, control)


def native_greedy_rollout(model: CompoundHierarchicalGPT, *, max_new_events: int = 48) -> list[list[int]]:
    if max_new_events < 0:
        raise ValueError("max_new_events must be non-negative")
    model.eval()
    state = model.initial_stream_state()
    seed = DEFAULT_SEED
    records = [list(seed.as_tuple())]
    with torch.no_grad():
        context = model.advance_stream(seed, state)
        for _ in range(max_new_events):
            record = greedy_next_record(model, context)
            records.append(list(record.as_tuple()))
            context = model.advance_stream(record, state)
    return records


def golden_payload(model: CompoundHierarchicalGPT, *, checkpoint_sha256: str, max_new_events: int = 48) -> dict:
    records = native_greedy_rollout(model, max_new_events=max_new_events)
    canonical = json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {
        "schema": "orbitune-compound-native-greedy-golden-v1",
        "runtime_abi": "native-stream-state+decoder-prefix-v2",
        "checkpoint_sha256": checkpoint_sha256,
        "temperature": 0,
        "top_p": 1,
        "new_events": max_new_events,
        "records_sha256": hashlib.sha256(canonical).hexdigest(),
        "records": records,
    }


def write_golden(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
