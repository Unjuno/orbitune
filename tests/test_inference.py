from orbitune.inference import MAX_NOTES_PER_POSITION, allowed_next_tokens
from orbitune.tokenizer.vocab import TheoryRemiVocab


def _ids(vocab: TheoryRemiVocab, *tokens: str) -> list[int]:
    return [vocab.token_to_id[token] for token in tokens]


def _note(vocab: TheoryRemiVocab, position: int, pitch: int) -> list[int]:
    return _ids(
        vocab,
        f"POSITION_{position}",
        f"NOTE_PITCH_{pitch}",
        "NOTE_DURATION_4",
        "VELOCITY_16",
    )


def test_generation_grammar_preserves_chords_and_requires_late_bar_position_before_closing():
    vocab = TheoryRemiVocab()
    ids = _ids(vocab, "BOS", "BAR") + _note(vocab, 0, 60)
    allowed = allowed_next_tokens(ids, vocab, requested_bars=1)
    assert "EOS" not in allowed
    assert "POSITION_0" in allowed  # another note in the same chord is legal
    assert "POSITION_1" in allowed

    chord_pitch_step = ids + _ids(vocab, "POSITION_0")
    pitch_choices = allowed_next_tokens(chord_pitch_step, vocab, requested_bars=1)
    assert "NOTE_PITCH_60" not in pitch_choices  # duplicate pitch at same timestamp is blocked
    assert "NOTE_PITCH_64" in pitch_choices

    ids += _note(vocab, 12, 64)
    allowed = allowed_next_tokens(ids, vocab, requested_bars=1)
    assert "EOS" in allowed
    assert "POSITION_12" in allowed
    assert "POSITION_13" in allowed


def test_generation_grammar_caps_notes_per_position():
    vocab = TheoryRemiVocab()
    ids = _ids(vocab, "BOS", "BAR")
    for index in range(MAX_NOTES_PER_POSITION):
        ids += _note(vocab, 12, 60 + index)
    allowed = allowed_next_tokens(ids, vocab, requested_bars=1)
    assert "POSITION_12" not in allowed
    assert "POSITION_13" in allowed
    assert "EOS" in allowed


def test_generation_grammar_requires_requested_number_of_bars():
    vocab = TheoryRemiVocab()
    first_bar = _ids(vocab, "BOS", "BAR") + _note(vocab, 12, 60)
    allowed = allowed_next_tokens(first_bar, vocab, requested_bars=2)
    assert "BAR" in allowed
    assert "EOS" not in allowed

    second_bar = first_bar + _ids(vocab, "BAR") + _note(vocab, 12, 64)
    allowed = allowed_next_tokens(second_bar, vocab, requested_bars=2)
    assert "EOS" in allowed
