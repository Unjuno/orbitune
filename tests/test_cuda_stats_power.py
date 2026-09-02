"""Unit test for the ``power_draw_watts`` mW→W fix in ``cuda_stats``.

Before the fix, ``cuda_stats()['power_draw_watts']`` was actually in
milliwatts (the value reported by ``torch.cuda.power_draw()``). On the
RTX 3080 laptop the raw value is typically ~30,000-60,000, which is
30-60 W. The fix divides by 1000.0 so the JSON key matches the unit.

This test loads the trainer module, monkey-patches
``torch.cuda.power_draw`` to a fixed raw value, and asserts the
post-fix output is in a sane W range (< 200 W on the local hardware).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compound_cuda_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("orbitune_compound_cuda_train_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for cuda_stats()")
def test_power_draw_watts_is_in_watts_not_milliwatts(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    # Simulate 35,000 mW as the raw power_draw() return value (= 35 W)
    monkeypatch.setattr(torch.cuda, "power_draw", lambda: 35_000.0)
    stats = module.cuda_stats()
    assert "power_draw_watts" in stats, "cuda_stats() must always include power_draw_watts"
    value = stats["power_draw_watts"]
    assert value == pytest.approx(35.0), f"expected 35.0 W, got {value} (mW→W conversion broken)"
    # Sanity: must be in a sane W range on this hardware (< 200 W)
    assert 0.0 <= value < 200.0


def test_cuda_stats_handles_missing_power_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """If torch.cuda exposes no power_draw, the key is simply absent (no crash)."""
    module = _load_module()
    # Drop power_draw to simulate an environment without it
    monkeypatch.setattr(torch.cuda, "power_draw", None, raising=False)
    stats = module.cuda_stats()
    # power_draw_watts may be absent; if present it must already be in W
    if "power_draw_watts" in stats:
        assert stats["power_draw_watts"] < 200.0
