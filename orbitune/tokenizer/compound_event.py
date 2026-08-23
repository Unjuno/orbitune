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
from orbitune.quantization import FactorizedValue, dequantize_unsigned, quantize_unsigned


@dataclass(frozen=True, slots=True)
class CompoundRecord:
    """One model step for the experimental Compound tokenizer.

    Timing and continuous controls are factorized into small categorical heads.
    NOTE duration uses a dedicated pair. CC value, pitch bend and pressure use
    the shared continuous pair. Unused fields are zero.
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
    continuous_coarse: int = 0
    continuous_residual: int = 0

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
            self.continuous_coarse,
            self.continuous_residual,
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
            continuous = FactorizedValue(0, 0)
            a1, a2, a3, a4 = event.a1, event.a2, event.a3, event.a4

            if event.type is CompoundEventType.NOTE:
                duration = quantize_time(event.a2)
                a2 = 0
            elif event.type is CompoundEventType.CC:
                continuous = quantize_unsigned(event.a2, maximum=127)
                a2 = 0
            elif event.type is CompoundEventType.PITCH_BEND:
                continuous = quantize_unsigned(event.a1, maximum=16383)
                a1 = 0
            elif event.type is CompoundEventType.CHANNEL_PRESSURE:
                continuous = quantize_unsigned(event.a1, maximum=127)
                a1 = 0
            elif event.type is CompoundEventType.POLY_PRESSURE:
                continuous = quantize_unsigned(event.a2, maximum=127)
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
                    continuous_coarse=continuous.coarse,
                    continuous_residual=continuous.residual,
                )
            )
        return output

    def decode_records(self, records: Iterable[CompoundRecord]) -> list[CompoundEvent]:
        output: list[CompoundEvent] = []
        step = 0
        for record in records:
            step += dequantize_time(FactorizedTime(record.delta_coarse, record.delta_residual))
            event_type = CompoundEventType(record.event_type)
            a1, a2 = record.a1, record.a2
            continuous = FactorizedValue(record.continuous_coarse, record.continuous_residual)

            if event_type is CompoundEventType.NOTE:
                a2 = max(1, dequantize_time(FactorizedTime(record.duration_coarse, record.duration_residual)))
            elif event_type is CompoundEventType.CC:
                a2 = dequantize_unsigned(continuous, maximum=127)
            elif event_type is CompoundEventType.PITCH_BEND:
                a1 = dequantize_unsigned(continuous, maximum=16383)
            elif event_type is CompoundEventType.CHANNEL_PRESSURE:
                a1 = dequantize_unsigned(continuous, maximum=127)
            elif event_type is CompoundEventType.POLY_PRESSURE:
                a2 = dequantize_unsigned(continuous, maximum=127)

            output.append(
                CompoundEvent(
                    type=event_type,
                    step=step,
                    channel=record.channel,
                    a1=a1,
                    a2=a2,
                    a3=record.a3,
                    a4=record.a4,
                )
            )
        return canonicalize_events(output)
