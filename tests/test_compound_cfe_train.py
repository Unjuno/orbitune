from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

import orbitune.compound_base as compound_base


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compound_cfe_train.py"
SPEC = importlib.util.spec_from_file_location("compound_cfe_train_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cfe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cfe)


def test_candidate_head_counts_filters_invalid_geometry() -> None:
    assert cfe.candidate_head_counts(224, [8, 7, 14, 9, 7]) == [8, 7, 14]
    assert 224 // 8 == 28
    assert 224 // 7 == 32
    assert 224 // 14 == 16


def test_causal_fastpath_matches_materialized_causal_mask_on_cpu() -> None:
    torch.manual_seed(4)
    attn = compound_base.MultiheadSelfAttention(d_model=32, n_head=4, dropout=0.0).eval()
    x = torch.randn(2, 12, 32)
    mask = cfe._ORIGINAL_CAUSAL_BIAS(12, x.device)
    expected = cfe._ORIGINAL_ATTN_FORWARD(attn, x, mask)
    try:
        cfe.install_causal_fastpath()
        fast_mask = compound_base._causal_bias(12, x.device)
        assert fast_mask is None
        actual = attn(x, fast_mask)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        cfe.uninstall_causal_fastpath()


def test_local_window_keeps_explicit_mask_when_window_is_smaller() -> None:
    try:
        cfe.install_causal_fastpath()
        mask = compound_base._causal_bias(16, torch.device("cpu"), window=4)
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (16, 16)
        assert torch.isneginf(mask[8, 0])
        assert mask[8, 8].item() == 0.0
    finally:
        cfe.uninstall_causal_fastpath()
