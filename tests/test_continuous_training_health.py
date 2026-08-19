from __future__ import annotations

import importlib.util
import random
from collections import deque
from pathlib import Path

import torch

from orbitune.model import OrbituneConfig, OrbituneGPT


SCRIPT = Path("scripts/continuous_train.py")
spec = importlib.util.spec_from_file_location("orbitune_continuous_train", SCRIPT)
assert spec is not None and spec.loader is not None
continuous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuous)


def test_snapshot_boundaries_are_token_based():
    assert continuous._snapshot_boundary(0, 16, 16) == [16]
    assert continuous._snapshot_boundary(16, 32, 16) == [32]
    assert continuous._snapshot_boundary(9, 35, 10) == [10, 20, 30]
    assert continuous._snapshot_boundary(35, 35, 10) == []


def test_loss_zscore_waits_for_warmup_and_detects_outlier():
    history = deque([1.0, 1.1, 0.9, 1.05], maxlen=10)
    assert continuous._loss_zscore(history, 20.0, min_samples=5) is None
    history.append(0.95)
    z = continuous._loss_zscore(history, 20.0, min_samples=5)
    assert z is not None and z > 5


def test_capture_restore_preserves_training_progress(tmp_path: Path):
    cfg = OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0)
    config = {
        "vocab_size": 32,
        "max_seq_len": 16,
        "n_layer": 1,
        "n_embd": 32,
        "n_head": 4,
        "dropout": 0.0,
    }
    model = OrbituneGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = random.Random(123)
    original_random_state = rng.getstate()

    state = continuous._capture_state(
        model,
        optimizer,
        config=config,
        global_step=7,
        tokens_seen=4096,
        best_validation_loss=1.25,
        best_step=5,
        rng=rng,
    )
    path = tmp_path / "healthy.pt"
    continuous._atomic_torch_save(state, path)

    for parameter in model.parameters():
        parameter.data.zero_()
    rng.random()

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    step, tokens, best_loss, best_step = continuous._restore_state(
        loaded, model, optimizer, rng, expected_config=config
    )
    assert step == 7
    assert tokens == 4096
    assert best_loss == 1.25
    assert best_step == 5
    assert rng.getstate() == original_random_state
    assert any(torch.count_nonzero(parameter) for parameter in model.parameters())
