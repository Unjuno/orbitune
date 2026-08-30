from __future__ import annotations

from orbitune.compound import CompoundEvent, CompoundEventType
from orbitune.compound_memory_targets import (
    FAST_HORIZON_STEPS,
    MEDIUM_HORIZON_STEPS,
    MEMORY_TARGET_SCHEMA,
    UNKNOWN_PITCH_CLASS,
    UNKNOWN_PROGRAM_FAMILY,
    UNKNOWN_REGISTER_BIN,
    derive_compound_memory_targets,
    target_cardinalities,
)
from orbitune.tokenizer.compound_event import CompoundEventTokenizer


def _records():  # type: ignore[no-untyped-def]
    events = [
        CompoundEvent(type=CompoundEventType.TEMPO, step=0, a1=120),
        CompoundEvent(type=CompoundEventType.PROGRAM, step=0, channel=0, a1=40),
        CompoundEvent(type=CompoundEventType.NOTE, step=0, channel=0, a1=60, a2=96, a3=80),
        CompoundEvent(type=CompoundEventType.NOTE, step=96, channel=0, a1=64, a2=96, a3=96),
        CompoundEvent(type=CompoundEventType.PEDAL, step=192, channel=0, a1=1),
        CompoundEvent(type=CompoundEventType.NOTE, step=384, channel=0, a1=67, a2=48, a3=64),
        CompoundEvent(type=CompoundEventType.PROGRAM, step=768, channel=1, a1=72),
        CompoundEvent(type=CompoundEventType.NOTE, step=768, channel=1, a1=69, a2=96, a3=100),
    ]
    return CompoundEventTokenizer().encode_events(events)


def test_target_schema_and_horizons_are_explicit() -> None:
    assert MEMORY_TARGET_SCHEMA == "orbitune-compound-memory-targets-v0-experimental"
    assert FAST_HORIZON_STEPS == 384
    assert MEDIUM_HORIZON_STEPS == 1536
    assert target_cardinalities() == {
        "fast": {"note_density_bin": 7, "mean_velocity_bin": 9, "note_gap_bin": 8},
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


def test_targets_are_one_to_one_and_strictly_prefix_causal() -> None:
    records = _records()
    full = derive_compound_memory_targets(records)
    assert len(full) == len(records)
    assert [item.step for item in full] == sorted(item.step for item in full)

    for stop in range(1, len(records) + 1):
        prefix = derive_compound_memory_targets(records[:stop])
        assert prefix == full[:stop]


def test_midi_state_is_reflected_after_it_is_consumed() -> None:
    records = _records()
    targets = derive_compound_memory_targets(records)

    first_note = next(
        index for index, record in enumerate(records) if record.event_type == int(CompoundEventType.NOTE)
    )
    assert targets[first_note].slow.dominant_pitch_class == 0  # MIDI 60 = C
    assert targets[first_note].medium.dominant_program_family == 5  # program 40 // 8
    assert targets[first_note].medium.mean_register_bin == 3
    assert targets[first_note].fast.note_gap_bin == 0

    pedal = next(
        index for index, record in enumerate(records) if record.event_type == int(CompoundEventType.PEDAL)
    )
    assert targets[pedal].medium.pedal_any == 1

    channel_one_note = max(
        index for index, record in enumerate(records) if record.event_type == int(CompoundEventType.NOTE)
    )
    assert targets[channel_one_note].medium.channel_diversity_bin >= 2
    assert targets[channel_one_note].slow.program_family_diversity_bin >= 2


def test_empty_prefix_state_uses_explicit_unknown_categories() -> None:
    tokenizer = CompoundEventTokenizer()
    records = tokenizer.encode_events(
        [CompoundEvent(type=CompoundEventType.TEMPO, step=0, a1=100)]
    )
    target = derive_compound_memory_targets(records)[0]
    assert target.medium.mean_register_bin == UNKNOWN_REGISTER_BIN
    assert target.medium.dominant_program_family == UNKNOWN_PROGRAM_FAMILY
    assert target.slow.dominant_pitch_class == UNKNOWN_PITCH_CLASS


def test_every_emitted_category_stays_within_declared_cardinality() -> None:
    cardinalities = target_cardinalities()
    for target in derive_compound_memory_targets(_records()):
        payload = target.as_dict()
        for scale in ("fast", "medium", "slow"):
            values = payload[scale]
            assert isinstance(values, dict)
            for name, value in values.items():
                assert isinstance(value, int)
                assert 0 <= value < cardinalities[scale][name]
