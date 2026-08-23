from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence


class CompoundEventType(IntEnum):
    NOTE = 0
    CC = 1
    PROGRAM = 2
    BANK = 3
    TEMPO = 4
    PEDAL = 5
    PITCH_BEND = 6
    CHANNEL_PRESSURE = 7
    POLY_PRESSURE = 8
    TIME_SIGNATURE = 9


TIME_COARSE_EDGES: tuple[int, ...] = (0, 24, 48, 96, 192, 384, 768, 1536)
TIME_RESIDUAL_LEVELS = 16
COMPOUND_TOKENIZER_ABI = "orbitune-compound-v0-experimental"
TEMPORAL_RESOLUTION = 96


@dataclass(frozen=True, slots=True)
class CompoundEvent:
    """Canonical event used by the experimental production tokenizer.

    ``step`` is absolute musical time in units of 1/96 quarter note. Attribute
    meanings are event-type dependent; unused slots must be zero. The schema is
    deliberately factorized so one event remains one Transformer step.
    """

    type: CompoundEventType
    step: int
    channel: int = 0
    a1: int = 0
    a2: int = 0
    a3: int = 0
    a4: int = 0

    def validate(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not 0 <= self.channel <= 15:
            raise ValueError("channel must be in 0..15")
        if any(value < 0 for value in (self.a1, self.a2, self.a3, self.a4)):
            raise ValueError("attributes must be non-negative")

        if self.type is CompoundEventType.NOTE:
            if not 0 <= self.a1 <= 127:
                raise ValueError("NOTE pitch must be in 0..127")
            if self.a2 <= 0:
                raise ValueError("NOTE duration must be positive")
            if not 1 <= self.a3 <= 127:
                raise ValueError("NOTE velocity must be in 1..127")
        elif self.type is CompoundEventType.CC:
            if not 0 <= self.a1 <= 127 or not 0 <= self.a2 <= 127:
                raise ValueError("CC id/value must be in 0..127")
        elif self.type is CompoundEventType.PROGRAM:
            if not 0 <= self.a1 <= 127:
                raise ValueError("program must be in 0..127")
        elif self.type is CompoundEventType.BANK:
            if not 0 <= self.a1 <= 127 or not 0 <= self.a2 <= 127:
                raise ValueError("bank MSB/LSB must be in 0..127")
        elif self.type is CompoundEventType.PEDAL:
            if self.a1 not in (0, 1):
                raise ValueError("pedal state must be 0 or 1")
        elif self.type is CompoundEventType.PITCH_BEND:
            if not 0 <= self.a1 <= 16383:
                raise ValueError("pitch bend must be in 0..16383")
        elif self.type is CompoundEventType.CHANNEL_PRESSURE:
            if not 0 <= self.a1 <= 127:
                raise ValueError("channel pressure must be in 0..127")
        elif self.type is CompoundEventType.POLY_PRESSURE:
            if not 0 <= self.a1 <= 127 or not 0 <= self.a2 <= 127:
                raise ValueError("poly pressure pitch/value must be in 0..127")
        elif self.type is CompoundEventType.TIME_SIGNATURE:
            if self.a1 <= 0 or self.a2 <= 0:
                raise ValueError("time signature numerator/denominator must be positive")


@dataclass(frozen=True, slots=True)
class FactorizedTime:
    coarse: int
    residual: int


def quantize_time(value: int) -> FactorizedTime:
    """Quantize a non-negative 96/qn timing value into 7 coarse + 16 residual.

    Values beyond the current reference range are clipped at 1536 steps. This
    function is an ABI primitive; long values may later be represented through
    repeated/extended events without changing the model's Transformer step.
    """

    if value < 0:
        raise ValueError("time value must be non-negative")
    clipped = min(value, TIME_COARSE_EDGES[-1])
    coarse = len(TIME_COARSE_EDGES) - 2
    for index, upper in enumerate(TIME_COARSE_EDGES[1:]):
        if clipped <= upper:
            coarse = index
            break
    lo = TIME_COARSE_EDGES[coarse]
    hi = TIME_COARSE_EDGES[coarse + 1]
    width = max(1, hi - lo)
    residual = round((clipped - lo) / width * (TIME_RESIDUAL_LEVELS - 1))
    residual = max(0, min(TIME_RESIDUAL_LEVELS - 1, residual))
    return FactorizedTime(coarse=coarse, residual=residual)


def dequantize_time(value: FactorizedTime) -> int:
    if not 0 <= value.coarse < len(TIME_COARSE_EDGES) - 1:
        raise ValueError("coarse index is outside the reference range")
    if not 0 <= value.residual < TIME_RESIDUAL_LEVELS:
        raise ValueError("residual index is outside the reference range")
    lo = TIME_COARSE_EDGES[value.coarse]
    hi = TIME_COARSE_EDGES[value.coarse + 1]
    return round(lo + value.residual / (TIME_RESIDUAL_LEVELS - 1) * (hi - lo))


def canonicalize_events(events: Iterable[CompoundEvent]) -> list[CompoundEvent]:
    """Return a deterministic MIDI-1-compatible canonical event sequence.

    Same onset/channel/pitch NOTE duplicates are merged. If the same channel and
    pitch is retriggered while an earlier note is still active, the earlier note
    is truncated at the retrigger. This removes note-instance ambiguity before
    the sequence is presented to the model.
    """

    checked = list(events)
    for event in checked:
        event.validate()

    merged: dict[tuple[int, int, int], CompoundEvent] = {}
    others: list[CompoundEvent] = []
    for event in sorted(checked, key=_sort_key):
        if event.type is not CompoundEventType.NOTE:
            others.append(event)
            continue
        key = (event.step, event.channel, event.a1)
        previous = merged.get(key)
        if previous is None:
            merged[key] = event
        else:
            merged[key] = CompoundEvent(
                type=CompoundEventType.NOTE,
                step=event.step,
                channel=event.channel,
                a1=event.a1,
                a2=max(previous.a2, event.a2),
                a3=max(previous.a3, event.a3),
            )

    notes = sorted(merged.values(), key=lambda event: (event.step, event.channel, event.a1))
    fixed: list[CompoundEvent] = []
    active: dict[tuple[int, int], int] = {}
    for event in notes:
        key = (event.channel, event.a1)
        previous_index = active.get(key)
        if previous_index is not None:
            previous = fixed[previous_index]
            previous_end = previous.step + previous.a2
            if previous_end > event.step:
                fixed[previous_index] = CompoundEvent(
                    type=CompoundEventType.NOTE,
                    step=previous.step,
                    channel=previous.channel,
                    a1=previous.a1,
                    a2=max(1, event.step - previous.step),
                    a3=previous.a3,
                )
        active[key] = len(fixed)
        fixed.append(event)

    return sorted([*fixed, *others], key=_sort_key)


def to_delta_events(events: Sequence[CompoundEvent]) -> list[tuple[CompoundEvent, int]]:
    canonical = canonicalize_events(events)
    output: list[tuple[CompoundEvent, int]] = []
    previous_step = 0
    for event in canonical:
        output.append((event, event.step - previous_step))
        previous_step = event.step
    return output


def _sort_key(event: CompoundEvent) -> tuple[int, int, int, int, int, int, int]:
    return (
        event.step,
        event.channel,
        int(event.type),
        event.a1,
        event.a2,
        event.a3,
        event.a4,
    )
