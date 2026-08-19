from __future__ import annotations

from orbitune.events import NoteEvent


def make_demo_events(*, bars: int = 4, bpm: int = 84) -> list[NoteEvent]:
    """Generate a deterministic piano-like demo pattern.

    bpm is accepted for CLI symmetry; it does not change the note pattern.
    """

    if bars <= 0:
        raise ValueError("bars must be positive")
    root_cycle = [48, 53, 55, 50]
    melody = [60, 64, 67, 71, 69, 67, 64, 62]
    events: list[NoteEvent] = []
    for bar in range(bars):
        root = root_cycle[bar % len(root_cycle)]
        events.append(NoteEvent(bar=bar, position=0, pitch=root, duration=8, velocity=50))
        events.append(NoteEvent(bar=bar, position=8, pitch=root + 7, duration=8, velocity=42))
        for idx, pos in enumerate((0, 4, 8, 12)):
            pitch = melody[(bar * 2 + idx) % len(melody)]
            events.append(NoteEvent(bar=bar, position=pos, pitch=pitch, duration=2, velocity=36))
    return events
