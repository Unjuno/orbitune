from __future__ import annotations

import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT


def _model() -> CompoundHierarchicalGPT:
    return CompoundHierarchicalGPT(CompoundBaseConfig(
        d_model=32, n_head=2, local_layers=2, medium_layers=2,
        global_layers=2, intra_layers=1, ff_mult=2, dropout=0.0,
        local_window=8, medium_stride=2, medium_window=3,
        global_stride=2, global_window=2,
    )).eval()


def test_window_capped_encode_has_no_future_leak():
    torch.manual_seed(503)
    model = _model()
    records = torch.zeros(2, 24, 12, dtype=torch.long)
    records[..., 4] = torch.arange(24)[None] % 80
    changed = records.clone()
    changed[:, 17:, 4] = (changed[:, 17:, 4] + 11) % 80
    with torch.no_grad():
        before = model.encode_window_capped(records)
        after = model.encode_window_capped(changed)
    torch.testing.assert_close(after[:, :17], before[:, :17], rtol=0, atol=0)


def test_window_capped_preserves_completion_timing_zero_prefixes():
    torch.manual_seed(509)
    model = _model()
    records = torch.zeros(1, 7, 12, dtype=torch.long)
    event = model.embedding(records)
    local = model.local(event, None)
    assert local.shape == (1, 7, 32)
    encoded = model.encode_window_capped(records)
    assert encoded.shape == (1, 7, 32)
    assert torch.isfinite(encoded).all()


def test_forward_selects_window_capped_training_semantics():
    torch.manual_seed(521)
    model = _model()
    records = torch.zeros(1, 24, 12, dtype=torch.long)
    model.training_encode_semantics = "a2"
    with torch.no_grad():
        expected = model.decoder.loss(model.encode_window_capped(records), records)[0]
        actual = model(records, records)[0]
    torch.testing.assert_close(actual, expected)
