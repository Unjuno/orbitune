from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from orbitune.compound import (
    COMPOUND_TOKENIZER_ABI,
    CompoundEvent,
    CompoundEventType,
    FactorizedTime,
    canonicalize_events,
    dequantize_time,
    quantize_time,
)


@dataclass(frozen=True, slots=True)
class CompoundRecord:
    """One model step for the experimental Compound tokenizer.

    Timing is factorized into small coarse/residual heads. NOTE duration uses a
    dedicated factorized pair; non-NOTE events leave duration fields at zero.
    The generic attribute slots remain event-type dependent.
    """

    event_type: int
    channel: int
    delta_coarse: int
    delta_residual: int
    a1: int
    a2: int
    a3: int
    a4: int
    duration_coarse: int = 0
    duration_residual: int = 0

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.event_type,
            self.channel,
            self.delta_coarse,
            self.delta_residual,
            self.a1,
            self.a2,
            self.a3,
            self.a4,
            self.duration_coarse,
            self.duration_residual,
        )


class CompoundEventTokenizer:
    """Experimental one-event-per-step tokenizer.

    This class intentionally does not replace `TheoryRemiTokenizer` yet. Its
    ABI is experimental until external real-MIDI validation is complete.
    """

    abi = COMPOUND_TOKENIZER_ABI

    def encode_events(self, events: Iterable[CompoundEvent]) -> list[CompoundRecord]:
        output: list[CompoundRecord] = []
        previous_step = 0
        for event in canonicalize_events(events):
            delta = quantize_time(event.step - previous_step)
            previous_step = event.step
            duration = FactorizedTime(0, 0)
            a1, a2, a3, a4 = event.a1, event.a2, event.a3, event.a4
            if event.type is CompoundEventType.NOTE:
                duration = quantize_time(event.a2)
                a2 = 0
            output.append(
                CompoundRecord(
                    event_type=int(event.type),
                    channel=event.channel,
                    delta_coarse=delta.coarse,
                    delta_residual=delta.residual,
                    a1=a1,
                    a2=a2,
                    a3=a3,
                    a4=a4,
                    duration_coarse=duration.coarse,
                    duration_residual=duration.residual,
                )
            )
        return output

    def decode_records(self, records: Iterable[CompoundRecord]) -> list[CompoundEvent]:
        output: list[CompoundEvent] = []
        step = 0
        for record in records:
            step += dequantize_time(FactorizedTime(record.delta_coarse, record.delta_residual))
            event_type = CompoundEventType(record.event_type)
            a2 = record.a2
            if event_type is CompoundEventType.NOTE:
                a2 = max(
                    1,
                    dequantize_time(
                        FactorizedTime(record.duration_coarse, record.duration_residual)
                    ),
                )
            output.append(
                CompoundEvent(
                    type=event_type,
                    step=step,
                    channel=record.channel,
                    a1=record.a1,
                    a2=a2,
                    a3=record.a3,
                    a4=record.a4,
                )
            )
        return canonicalize_events(output)
