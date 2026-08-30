from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "experiments" / "compound_memory_target_routing_proxy.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("compound_memory_target_routing_proxy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_balanced_proxy_exercises_memory_target_classes() -> None:
    module = _load_module()
    records, fast, medium, slow = module.make_batch(64, torch.device("cpu"), seed=12345)
    assert records.shape == (64, module.SEQ_LEN, 12)
    assert fast.shape == (64, module.SEQ_LEN, 3)
    assert medium.shape == (64, module.SEQ_LEN, 5)
    assert slow.shape == (64, module.SEQ_LEN, 3)
    late = torch.arange(module.SEQ_LEN) >= module.LATE_START
    assert fast[:, late, 0].unique().numel() >= 5
    assert fast[:, late, 1].unique().numel() >= 5
    assert medium[:, late, 1].unique().numel() >= 8
    assert medium[:, late, 4].unique().numel() >= 4
    assert slow[:, late, 0].unique().numel() >= 8
    assert slow[:, late, 1].unique().numel() >= 5


def test_parameter_matched_target_models_differ_by_at_most_two_parameters() -> None:
    module = _load_module()
    shared = module.SharedMatched()
    routed = module.RoutedMultiBank()
    shared_params = sum(parameter.numel() for parameter in shared.parameters())
    routed_params = sum(parameter.numel() for parameter in routed.parameters())
    assert shared_params == 43984
    assert routed_params == 43986
    assert abs(shared_params - routed_params) <= 2


def test_memory_only_target_heads_and_event_head_backpropagate() -> None:
    module = _load_module()
    records, fast, medium, slow = module.make_batch(2, torch.device("cpu"), seed=7)
    active = torch.arange(module.SEQ_LEN) >= module.ACTIVE_START
    for model_type in (module.SharedMatched, module.RoutedMultiBank):
        torch.manual_seed(5)
        model = model_type()
        fast_logits, medium_logits, slow_logits, event_logits = model(records)
        assert [logits.shape[-1] for logits in fast_logits] == list(module.FAST_CARDS)
        assert [logits.shape[-1] for logits in medium_logits] == list(module.MEDIUM_CARDS)
        assert [logits.shape[-1] for logits in slow_logits] == list(module.SLOW_CARDS)
        assert event_logits.shape == (2, module.SEQ_LEN, 10)
        loss = (
            module._balanced_loss(fast_logits, fast, module.FAST_CARDS, active)
            + module._balanced_loss(medium_logits, medium, module.MEDIUM_CARDS, active)
            + module._balanced_loss(slow_logits, slow, module.SLOW_CARDS, active)
        )
        loss.backward()
        memory_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "memory" in name
        ]
        assert memory_parameters
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in memory_parameters
        )
