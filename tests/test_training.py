from pathlib import Path

from orbitune.model import OrbituneConfig
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import TrainConfig, train_base


def test_one_step_base_training(tmp_path: Path):
    vocab = TheoryRemiVocab()
    token_file = tmp_path / "tiny.tokens"
    token_file.write_text(
        "\n".join([
            "BAR", "POSITION_0", "NOTE_PITCH_60", "NOTE_DURATION_4", "VELOCITY_16",
            "POSITION_4", "NOTE_PITCH_64", "NOTE_DURATION_4", "VELOCITY_16",
            "BAR", "POSITION_0", "NOTE_PITCH_67", "NOTE_DURATION_8", "VELOCITY_14",
        ]) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "base.pt"
    report = train_base(
        [token_file],
        out,
        model_cfg=OrbituneConfig(vocab_size=len(vocab), max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0),
        train_cfg=TrainConfig(steps=1, batch_size=1, seq_len=8, learning_rate=1e-3, device="cpu"),
    )
    assert out.exists()
    assert report["steps"] == 1
    assert report["parameters"] > 0
    assert report["processed_tokens"] == 8
    assert report["elapsed_seconds"] > 0
    assert report["tokens_per_second"] > 0
