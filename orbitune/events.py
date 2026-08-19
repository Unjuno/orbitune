from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NoteEvent:
    """A quantized symbolic note event used by Orbitune v0.

    bar and position are discrete musical locations. pitch is MIDI note number.
    duration is measured in 1/16-bar units when positions_per_bar is 16.
    velocity is raw MIDI velocity in the range 1..127.
    """

    bar: int
    position: int
    pitch: int
    duration: int
    velocity: int

    def validate(self, *, positions_per_bar: int = 16) -> None:
        if self.bar < 0:
            raise ValueError("bar must be non-negative")
        if not 0 <= self.position < positions_per_bar:
            raise ValueError("position is outside the bar")
        if not 0 <= self.pitch <= 127:
            raise ValueError("pitch must be a MIDI note number")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if not 1 <= self.velocity <= 127:
            raise ValueError("velocity must be in 1..127")
