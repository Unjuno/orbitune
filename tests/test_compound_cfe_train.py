from __future__ import annotations

import importlib.util
import inspect
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


def test_train_loop_does_not_clip_gradients_after_base_train_step() -> None:
    source = inspect.getsource(cfe.train)
    assert "clip_grad_norm_" not in source
    assert 'parts.get("grad_norm")' in source


def test_flush_health_queue_replays_order_and_nonfinite_counts() -> None:
    pending = [
        (10, torch.tensor(1.0), torch.tensor(2.0)),
        (11, torch.tensor(float("inf")), torch.tensor(float("nan"))),
        (12, torch.tensor(3.0), torch.tensor(4.0)),
    ]
    loss_history: list[float] = []
    grad_history: list[float] = []
    spike_events: list[dict[str, object]] = []

    loss_count, grad_count, last_loss, last_grad = cfe._flush_health_queue(
        pending,
        loss_history=loss_history,
        grad_history=grad_history,
        spike_events=spike_events,
        non_finite_loss_count=0,
        non_finite_grad_count=0,
        health_history_len=2,
        spike_min_samples=99,
        spike_z_threshold=5.0,
    )

    assert pending == []
    assert loss_count == 1
    assert grad_count == 1
    assert last_loss == 3.0
    assert last_grad == 4.0
    assert len(loss_history) == 2 and torch.isinf(torch.tensor(loss_history[0]))
    assert loss_history[1] == 3.0
    assert len(grad_history) == 2 and torch.isnan(torch.tensor(grad_history[0]))
    assert grad_history[1] == 4.0
    assert spike_events == []


def test_flush_health_queue_preserves_spike_detection_order() -> None:
    pending = [
        (1, torch.tensor(1.0), torch.tensor(1.0)),
        (2, torch.tensor(1.0), torch.tensor(1.0)),
        (3, torch.tensor(10.0), torch.tensor(3.5)),
    ]
    loss_history: list[float] = []
    grad_history: list[float] = []
    spike_events: list[dict[str, object]] = []

    _, _, last_loss, last_grad = cfe._flush_health_queue(
        pending,
        loss_history=loss_history,
        grad_history=grad_history,
        spike_events=spike_events,
        non_finite_loss_count=0,
        non_finite_grad_count=0,
        health_history_len=10,
        spike_min_samples=3,
        spike_z_threshold=1.0,
    )

    assert last_loss == 10.0
    assert last_grad == 3.5
    assert len(spike_events) == 1
    assert spike_events[0]["step"] == 3
    assert spike_events[0]["gradient_norm"] == 3.5
