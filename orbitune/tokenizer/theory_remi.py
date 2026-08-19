from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orbitune.events import NoteEvent


@dataclass(slots=True)
class TheoryRemiTokenizer:
    """Theory-REMI v0 tokenizer for small MIDI-BGM experiments.

    Grammar emitted by encode_events:
      BAR POSITION_i NOTE_PITCH_p NOTE_DURATION_d VELOCITY_v ...

    The tokenizer is intentionally conservative. It uses absolute MIDI pitch and
    does not infer key, chord, or function harmony in v0.
    """

    positions_per_bar: int = 16
    min_pitch: int = 21
    max_pitch: int = 108
    max_duration: int = 64
    velocity_bins: int = 32

    def encode_events(self, events: list[NoteEvent]) -> list[str]:
        tokens: list[str] = []
        current_bar = -1
        for event in sorted(events, key=lambda e: (e.bar, e.position, e.pitch, e.duration, e.velocity)):
            event.validate(positions_per_bar=self.positions_per_bar)
            if not self.min_pitch <= event.pitch <= self.max_pitch:
                continue
            while current_bar < event.bar:
                tokens.append("BAR")
                current_bar += 1
            velocity_bin = self.velocity_to_bin(event.velocity)
            duration = max(1, min(self.max_duration, int(event.duration)))
            tokens.extend(
                [
                    f"POSITION_{event.position}",
                    f"NOTE_PITCH_{event.pitch}",
                    f"NOTE_DURATION_{duration}",
                    f"VELOCITY_{velocity_bin}",
                ]
            )
        return tokens

    def decode_events(self, tokens: list[str]) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        current_bar = -1
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "BAR":
                current_bar += 1
                i += 1
                continue
            if not token.startswith("POSITION_"):
                i += 1
                continue
            if i + 3 >= len(tokens):
                break
            position_token, pitch_token, duration_token, velocity_token = tokens[i : i + 4]
            try:
                position = int(position_token.removeprefix("POSITION_"))
                pitch = int(pitch_token.removeprefix("NOTE_PITCH_"))
                duration = int(duration_token.removeprefix("NOTE_DURATION_"))
                velocity_bin = int(velocity_token.removeprefix("VELOCITY_"))
            except ValueError:
                i += 1
                continue
            if not pitch_token.startswith("NOTE_PITCH_"):
                i += 1
                continue
            if not duration_token.startswith("NOTE_DURATION_"):
                i += 1
                continue
            if not velocity_token.startswith("VELOCITY_"):
                i += 1
                continue
            bar = max(0, current_bar)
            event = NoteEvent(
                bar=bar,
                position=position,
                pitch=pitch,
                duration=max(1, min(self.max_duration, duration)),
                velocity=self.bin_to_velocity(velocity_bin),
            )
            event.validate(positions_per_bar=self.positions_per_bar)
            events.append(event)
            i += 4
        return events

    def velocity_to_bin(self, velocity: int) -> int:
        velocity = max(1, min(127, int(velocity)))
        return max(1, min(self.velocity_bins, round(velocity / 127 * self.velocity_bins)))

    def bin_to_velocity(self, velocity_bin: int) -> int:
        velocity_bin = max(1, min(self.velocity_bins, int(velocity_bin)))
        return max(1, min(127, round(velocity_bin / self.velocity_bins * 127)))

    def write_tokens(self, tokens: list[str], path: str | Path) -> None:
        Path(path).write_text("\n".join(tokens) + "\n", encoding="utf-8")

    def read_tokens(self, path: str | Path) -> list[str]:
        return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
