from orbitune.inference import allowed_next_tokens
from orbitune.tokenizer.vocab import TheoryRemiVocab


def _ids(vocab: TheoryRemiVocab, *tokens: str) -> list[int]:
    return [vocab.token_to_id[token] for token in tokens]


def test_generation_grammar_requires_late_bar_position_before_closing():
    vocab = TheoryRemiVocab()
    ids = _ids(
        vocab,
        "BOS",
        "BAR",
        "POSITION_0",
        "NOTE_PITCH_60",
        "NOTE_DURATION_4",
        "VELOCITY_16",
    )
    allowed = allowed_next_tokens(ids, vocab, requested_bars=1)
    assert "EOS" not in allowed
    assert "POSITION_0" not in allowed
    assert "POSITION_1" in allowed

    ids.extend(
        _ids(
            vocab,
            "POSITION_12",
            "NOTE_PITCH_64",
            "NOTE_DURATION_4",
            "VELOCITY_16",
        )
    )
    allowed = allowed_next_tokens(ids, vocab, requested_bars=1)
    assert "EOS" in allowed
    assert "POSITION_13" in allowed


def test_generation_grammar_requires_requested_number_of_bars():
    vocab = TheoryRemiVocab()
    first_bar = _ids(
        vocab,
        "BOS",
        "BAR",
        "POSITION_12",
        "NOTE_PITCH_60",
        "NOTE_DURATION_4",
        "VELOCITY_16",
    )
    allowed = allowed_next_tokens(first_bar, vocab, requested_bars=2)
    assert "BAR" in allowed
    assert "EOS" not in allowed

    second_bar = first_bar + _ids(
        vocab,
        "BAR",
        "POSITION_12",
        "NOTE_PITCH_64",
        "NOTE_DURATION_4",
        "VELOCITY_16",
    )
    allowed = allowed_next_tokens(second_bar, vocab, requested_bars=2)
    assert "EOS" in allowed
