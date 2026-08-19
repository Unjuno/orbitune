from pathlib import Path

from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import TrainConfig, evaluate_token_loss, read_token_ids, train_base


def _write_tokens(path: Path, pitches: tuple[int, int, int]) -> None:
    path.write_text(
        "\n".join([
            "BAR", "POSITION_0", f"NOTE_PITCH_{pitches[0]}", "NOTE_DURATION_4", "VELOCITY_16",
            "POSITION_4", f"NOTE_PITCH_{pitches[1]}", "NOTE_DURATION_4", "VELOCITY_16",
            "BAR", "POSITION_0", f"NOTE_PITCH_{pitches[2]}", "NOTE_DURATION_8", "VELOCITY_14",
        ]) + "\n",
        encoding="utf-8",
    )


def test_one_step_base_training_with_validation(tmp_path: Path):
    vocab = TheoryRemiVocab()
    token_file = tmp_path / "tiny.tokens"
    validation_file = tmp_path / "validation.tokens"
    _write_tokens(token_file, (60, 64, 67))
    _write_tokens(validation_file, (62, 65, 69))

    out = tmp_path / "base.pt"
    report = train_base(
        [token_file],
        out,
        model_cfg=OrbituneConfig(vocab_size=len(vocab), max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0),
        train_cfg=TrainConfig(steps=1, batch_size=1, seq_len=8, learning_rate=1e-3, device="cpu"),
        validation_token_paths=[validation_file],
    )
    assert out.exists()
    assert report["steps"] == 1
    assert report["parameters"] > 0
    assert report["processed_tokens"] == 8
    assert report["elapsed_seconds"] > 0
    assert report["tokens_per_second"] > 0
    assert report["validation_tokens"] > 0
    assert report["validation_loss"] > 0
    assert report["best_validation_step"] == 1
    assert report["checkpoint_selection"] == "minimum_validation_loss"


def test_periodic_validation_restores_minimum_loss_checkpoint(tmp_path: Path):
    vocab = TheoryRemiVocab()
    token_file = tmp_path / "train.tokens"
    validation_file = tmp_path / "validation.tokens"
    _write_tokens(token_file, (60, 64, 67))
    _write_tokens(validation_file, (62, 65, 69))

    out = tmp_path / "best.pt"
    cfg = TrainConfig(
        steps=4,
        batch_size=1,
        seq_len=8,
        learning_rate=2e-3,
        device="cpu",
        seed=7,
        validation_interval=1,
    )
    report = train_base(
        [token_file],
        out,
        model_cfg=OrbituneConfig(vocab_size=len(vocab), max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0),
        train_cfg=cfg,
        validation_token_paths=[validation_file],
    )

    history = report["validation_history"]
    assert len(history) == 4
    best = min(history, key=lambda item: item["validation_loss"])
    assert report["best_validation_step"] == best["step"]
    assert report["validation_loss"] == best["validation_loss"]

    restored = OrbituneGPT.load_checkpoint(out).eval()
    validation_ids = read_token_ids([validation_file], vocab)
    restored_loss = evaluate_token_loss(restored, validation_ids, seq_len=8, device="cpu")
    assert abs(restored_loss - report["validation_loss"]) < 1e-6
