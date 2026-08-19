from pathlib import Path

import torch

from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab


def test_orbitune_tiny_parameter_budget():
    vocab = TheoryRemiVocab()
    model = OrbituneGPT(OrbituneConfig(vocab_size=len(vocab)))
    assert 3_000_000 <= model.parameter_count() <= 3_600_000


def test_model_forward_and_checkpoint(tmp_path: Path):
    cfg = OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0)
    model = OrbituneGPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    checkpoint = tmp_path / "base.pt"
    model.save_checkpoint(checkpoint)
    loaded = OrbituneGPT.load_checkpoint(checkpoint).eval()
    with torch.no_grad():
        a, _ = model(x)
        b, _ = loaded(x)
    assert torch.allclose(a, b)
