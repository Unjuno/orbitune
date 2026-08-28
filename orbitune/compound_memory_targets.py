from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from orbitune.compound import CompoundEventType, FactorizedTime, TEMPORAL_RESOLUTION, dequantize_time
from orbitune.tokenizer.compound_event import CompoundRecord


MEMORY_TARGET_SCHEMA = "orbitune-compound-memory-targets-v0-experimental"
FAST_HORIZON_STEPS = 4 * TEMPORAL_RESOLUTION
MEDIUM_HORIZON_STEPS = 16 * TEMPORAL_RESOLUTION
UNKNOWN_REGISTER_BIN = 8
UNKNOWN_PROGRAM_FAMILY = 16
UNKNOWN_PITCH_CLASS = 12


@dataclass(frozen=True, slots=True)
class FastMemoryTarget:
    note_density_bin: int
    mean_velocity_bin: int
    note_gap_bin: int


@dataclass(frozen=True, slots=True)
class MediumMemoryTarget:
    mean_register_bin: int
    dominant_program_family: int
    channel_diversity_bin: int
    pedal_any: int
    tempo_bin: int


@dataclass(frozen=True, slots=True)
class SlowMemoryTarget:
    dominant_pitch_class: int
    pitch_class_entropy_bin: int
    program_family_diversity_bin: int


@dataclass(frozen=True, slots=True)
class CompoundMemoryTarget:
    step: int
    fast: FastMemoryTarget
    medium: MediumMemoryTarget
    slow: SlowMemoryTarget

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _NoteObservation:
    step: int
    pitch: int
    velocity: int
    channel: int
    program_family: int


def target_cardinalities() -> dict[str, dict[str, int]]:
    """Return exact categorical cardinalities for every experimental target head."""

    return {
        "fast": {
            "note_density_bin": 7,
            "mean_velocity_bin": 9,
            "note_gap_bin": 8,
        },
        "medium": {
            "mean_register_bin": 9,
            "dominant_program_family": 17,
            "channel_diversity_bin": 6,
            "pedal_any": 2,
            "tempo_bin": 16,
        },
        "slow": {
            "dominant_pitch_class": 13,
            "pitch_class_entropy_bin": 8,
            "program_family_diversity_bin": 6,
        },
    }


def _bucket_upper(value: float, upper_bounds: Sequence[float]) -> int:
    for index, upper in enumerate(upper_bounds):
        if value <= upper:
            return index
    return len(upper_bounds)


def _density_bin(count: int) -> int:
    return _bucket_upper(float(count), (0, 2, 4, 8, 16, 32))


def _velocity_bin(notes: Sequence[_NoteObservation]) -> int:
    if not notes:
        return 8
    mean = sum(note.velocity for note in notes) / len(notes)
    return min(7, max(0, int((mean - 1) // 16)))


def _register_bin(notes: Sequence[_NoteObservation]) -> int:
    if not notes:
        return UNKNOWN_REGISTER_BIN
    mean = sum(note.pitch for note in notes) / len(notes)
    return min(7, max(0, int(mean // 16)))


def _note_gap_bin(step: int, last_note_step: int | None) -> int:
    if last_note_step is None:
        return 7
    return _bucket_upper(float(step - last_note_step), (0, 12, 24, 48, 96, 192, 384))


def _channel_diversity_bin(notes: Sequence[_NoteObservation]) -> int:
    count = len({note.channel for note in notes})
    return _bucket_upper(float(count), (0, 1, 2, 4, 8))


def _program_family(notes: Sequence[_NoteObservation]) -> int:
    if not notes:
        return UNKNOWN_PROGRAM_FAMILY
    counts = Counter(note.program_family for note in notes)
    highest = max(counts.values())
    return min(family for family, count in counts.items() if count == highest)


def _tempo_bin(bpm: int) -> int:
    # Compound MIDI semantics use the MIDI default of 120 BPM until a TEMPO
    # event changes it. Sixteen broad bins are enough for an auxiliary state
    # target without pretending this is the final tempo representation.
    return min(15, max(0, (bpm - 1) * 16 // 999))


def _dominant_pitch_class(histogram: Sequence[int]) -> int:
    total = sum(histogram)
    if total == 0:
        return UNKNOWN_PITCH_CLASS
    highest = max(histogram)
    return min(index for index, count in enumerate(histogram) if count == highest)


def _pitch_entropy_bin(histogram: Sequence[int]) -> int:
    total = sum(histogram)
    if total <= 1:
        return 0
    entropy = 0.0
    for count in histogram:
        if count:
            probability = count / total
            entropy -= probability * math.log(probability)
    normalized = entropy / math.log(12.0)
    return min(7, max(0, int(normalized * 8.0)))


def _program_diversity_bin(families: set[int]) -> int:
    return _bucket_upper(float(len(families)), (0, 1, 2, 4, 8))


def _validate_record(record: CompoundRecord) -> CompoundEventType:
    try:
        event_type = CompoundEventType(record.event_type)
    except ValueError as exc:
        raise ValueError(f"invalid Compound event type {record.event_type}") from exc
    if not 0 <= record.channel <= 15:
        raise ValueError("Compound channel must be in 0..15")
    if not 0 <= record.delta_coarse < 7 or not 0 <= record.delta_residual < 16:
        raise ValueError("Compound delta factor is outside the experimental ABI")
    return event_type


def derive_compound_memory_targets(
    records: Iterable[CompoundRecord],
) -> list[CompoundMemoryTarget]:
    """Derive deterministic *causal* memory targets aligned one-to-one with records.

    ``fast`` summarizes the preceding four quarter-notes, ``medium`` summarizes
    the preceding sixteen quarter-notes plus current channel/global state, and
    ``slow`` is a prefix summary from the beginning of the composition. The
    current record is incorporated before its aligned target is emitted, so the
    target describes the memory state after consuming that event and never uses
    future records.
    """

    fast_notes: deque[_NoteObservation] = deque()
    medium_notes: deque[_NoteObservation] = deque()
    programs = [0] * 16
    pedals = [0] * 16
    tempo_bpm = 120
    pitch_class_histogram = [0] * 12
    prefix_program_families: set[int] = set()
    last_note_step: int | None = None
    absolute_step = 0
    output: list[CompoundMemoryTarget] = []

    for record in records:
        event_type = _validate_record(record)
        absolute_step += dequantize_time(
            FactorizedTime(record.delta_coarse, record.delta_residual)
        )

        fast_cutoff = absolute_step - FAST_HORIZON_STEPS
        while fast_notes and fast_notes[0].step < fast_cutoff:
            fast_notes.popleft()
        medium_cutoff = absolute_step - MEDIUM_HORIZON_STEPS
        while medium_notes and medium_notes[0].step < medium_cutoff:
            medium_notes.popleft()

        if event_type is CompoundEventType.PROGRAM:
            if not 0 <= record.a1 <= 127:
                raise ValueError("PROGRAM value must be in 0..127")
            programs[record.channel] = record.a1
        elif event_type is CompoundEventType.PEDAL:
            if record.a1 not in (0, 1):
                raise ValueError("PEDAL state must be 0 or 1")
            pedals[record.channel] = record.a1
        elif event_type is CompoundEventType.TEMPO:
            if not 1 <= record.a1 <= 999:
                raise ValueError("TEMPO BPM must be in 1..999")
            tempo_bpm = record.a1
        elif event_type is CompoundEventType.NOTE:
            if not 0 <= record.a1 <= 127 or not 1 <= record.a3 <= 127:
                raise ValueError("NOTE pitch/velocity is outside MIDI range")
            family = programs[record.channel] // 8
            note = _NoteObservation(
                step=absolute_step,
                pitch=record.a1,
                velocity=record.a3,
                channel=record.channel,
                program_family=family,
            )
            fast_notes.append(note)
            medium_notes.append(note)
            last_note_step = absolute_step
            pitch_class_histogram[record.a1 % 12] += 1
            prefix_program_families.add(family)

        fast_list = list(fast_notes)
        medium_list = list(medium_notes)
        output.append(
            CompoundMemoryTarget(
                step=absolute_step,
                fast=FastMemoryTarget(
                    note_density_bin=_density_bin(len(fast_list)),
                    mean_velocity_bin=_velocity_bin(fast_list),
                    note_gap_bin=_note_gap_bin(absolute_step, last_note_step),
                ),
                medium=MediumMemoryTarget(
                    mean_register_bin=_register_bin(medium_list),
                    dominant_program_family=_program_family(medium_list),
                    channel_diversity_bin=_channel_diversity_bin(medium_list),
                    pedal_any=int(any(pedals)),
                    tempo_bin=_tempo_bin(tempo_bpm),
                ),
                slow=SlowMemoryTarget(
                    dominant_pitch_class=_dominant_pitch_class(pitch_class_histogram),
                    pitch_class_entropy_bin=_pitch_entropy_bin(pitch_class_histogram),
                    program_family_diversity_bin=_program_diversity_bin(
                        prefix_program_families
                    ),
                ),
            )
        )

    return output
