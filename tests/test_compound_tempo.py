from __future__ import annotations

import pytest

from orbitune.compound import CompoundEvent, CompoundEventType
from orbitune.compound_base import write_compound_midi


def _tempo(bpm: int) -> CompoundEvent:
    return CompoundEvent(CompoundEventType.TEMPO, step=0, channel=0, a1=bpm)


@pytest.mark.parametrize("bpm", [1, 2, 3])
def test_standard_midi_rejects_unrepresentable_compound_tempo(tmp_path, bpm: int) -> None:
    # The event remains valid in the checkpoint ABI; only Standard MIDI export
    # rejects values whose microseconds-per-quarter cannot fit in 24 bits.
    _tempo(bpm).validate()
    with pytest.raises(ValueError, match="exceeds the three-byte MIDI tempo field"):
        write_compound_midi(tmp_path / "tempo.mid", [_tempo(bpm)])


def test_standard_midi_accepts_first_representable_compound_tempo(tmp_path) -> None:
    target = tmp_path / "tempo.mid"
    write_compound_midi(target, [_tempo(4)])
    assert b"\xff\x51\x03\xe4\xe1\xc0" in target.read_bytes()
