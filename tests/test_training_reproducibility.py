from pathlib import Path

import torch

from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import TrainConfig, train_base


def test_train_base_seed_covers_parameter_initialization(tmp_path: Path) -> None:
    tokens = tmp_path / "train.tokens"
    pattern = [
        "BAR",
        "POSITION_0",
        "NOTE_PITCH_60",
        "NOTE_DURATION_4",
        "VELOCITY_16",
        "POSITION_12",
        "NOTE_PITCH_67",
        "NOTE_DURATION_4",
        "VELOCITY_16",
    ]
    tokens.write_text("\n".join(pattern * 8) + "\n", encoding="utf-8")
    model_cfg = OrbituneConfig(
        vocab_size=len(TheoryRemiVocab()),
        max_seq_len=16,
        n_layer=1,
        n_embd=16,
        n_head=4,
        dropout=0.1,
    )
    train_cfg = TrainConfig(steps=2, batch_size=2, seq_len=8, seed=777)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"

    train_base([tokens], first, model_cfg=model_cfg, train_cfg=train_cfg)
    train_base([tokens], second, model_cfg=model_cfg, train_cfg=train_cfg)

    model_a = OrbituneGPT.load_checkpoint(first)
    model_b = OrbituneGPT.load_checkpoint(second)
    assert model_a.state_dict().keys() == model_b.state_dict().keys()
    for name, tensor_a in model_a.state_dict().items():
        assert torch.equal(tensor_a, model_b.state_dict()[name]), name
