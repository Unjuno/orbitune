import pytest

from orbitune.compound import (
    COMPOUND_TOKENIZER_ABI,
    CompoundEvent,
    CompoundEventType,
    FactorizedTime,
    canonicalize_events,
    dequantize_time,
    quantize_time,
    to_delta_events,
)


def note(step: int, duration: int, velocity: int = 80) -> CompoundEvent:
    return CompoundEvent(
        type=CompoundEventType.NOTE,
        step=step,
        channel=0,
        a1=60,
        a2=duration,
        a3=velocity,
    )


def test_compound_abi_is_explicitly_experimental():
    assert COMPOUND_TOKENIZER_ABI == "orbitune-compound-v0-experimental"


def test_same_pitch_duplicates_merge_deterministically():
    events = [note(12, 24, 70), note(12, 48, 90)]
    canonical = canonicalize_events(events)
    assert canonical == [note(12, 48, 90)]


def test_same_pitch_retrigger_truncates_previous_note():
    events = [note(10, 40, 70), note(25, 12, 90)]
    canonical = canonicalize_events(events)
    assert canonical[0] == note(10, 15, 70)
    assert canonical[1] == note(25, 12, 90)


def test_other_event_types_survive_canonicalization():
    cc = CompoundEvent(CompoundEventType.CC, step=8, channel=2, a1=74, a2=100)
    program = CompoundEvent(CompoundEventType.PROGRAM, step=0, channel=2, a1=41)
    assert canonicalize_events([cc, program]) == [program, cc]


@pytest.mark.parametrize("raw", [0, 1, 6, 12, 24, 48, 96, 192, 384, 768, 1536])
def test_time_factorization_is_bounded(raw: int):
    encoded = quantize_time(raw)
    reconstructed = dequantize_time(encoded)
    assert 0 <= encoded.coarse < 7
    assert 0 <= encoded.residual < 16
    assert 0 <= reconstructed <= 1536


def test_time_factorization_rejects_negative_values():
    with pytest.raises(ValueError):
        quantize_time(-1)


def test_time_factorization_clips_long_values():
    assert dequantize_time(quantize_time(99999)) == 1536


def test_delta_events_use_one_event_record_per_compound_event():
    events = [
        CompoundEvent(CompoundEventType.PROGRAM, step=0, channel=0, a1=5),
        note(12, 24),
        CompoundEvent(CompoundEventType.PEDAL, step=24, channel=0, a1=1),
    ]
    delta = to_delta_events(events)
    assert len(delta) == len(events)
    assert [value for _, value in delta] == [0, 12, 12]


def test_validation_catches_invalid_note():
    with pytest.raises(ValueError):
        CompoundEvent(
            CompoundEventType.NOTE,
            step=0,
            channel=0,
            a1=200,
            a2=24,
            a3=64,
        ).validate()


def test_invalid_factorized_time_is_rejected():
    with pytest.raises(ValueError):
        dequantize_time(FactorizedTime(coarse=7, residual=0))
