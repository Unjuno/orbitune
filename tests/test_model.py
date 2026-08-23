from pathlib import Path

import pytest
import torch

from orbitune.compat import REFERENCE_PARAMETER_COUNT, TOKENIZER_ABI
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab


def test_orbitune_reference_parameter_budget():
    vocab = TheoryRemiVocab()
    model = OrbituneGPT(OrbituneConfig(vocab_size=len(vocab)))
    assert model.parameter_count() == REFERENCE_PARAMETER_COUNT == 10_200_960
    assert model.config.n_embd == 448
    assert model.config.n_head == 7
    assert model.config.max_seq_len == 1024


def test_model_forward_and_checkpoint(tmp_path: Path):
    cfg = OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0)
    model = OrbituneGPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    checkpoint = tmp_path / "base.pt"
    model.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["tokenizer"] == TOKENIZER_ABI
    loaded = OrbituneGPT.load_checkpoint(checkpoint).eval()
    with torch.no_grad():
        a, _ = model(x)
        b, _ = loaded(x)
    assert torch.allclose(a, b)


def test_checkpoint_rejects_wrong_tokenizer_but_accepts_pre_metadata_v0(tmp_path: Path):
    cfg = OrbituneConfig(vocab_size=16, max_seq_len=8, n_layer=1, n_embd=16, n_head=4, dropout=0.0)
    model = OrbituneGPT(cfg)
    checkpoint = tmp_path / "base.pt"
    model.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    payload["tokenizer"] = "wrong-tokenizer"
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        OrbituneGPT.load_checkpoint(bad)

    payload.pop("tokenizer")
    old = tmp_path / "old-v0.pt"
    torch.save(payload, old)
    loaded = OrbituneGPT.load_checkpoint(old)
    assert loaded.tokenizer == TOKENIZER_ABI
