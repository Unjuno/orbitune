from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "experiments" / "compound_field_memory_routing_proxy.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("compound_field_memory_routing_proxy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_proxy_uses_compound_width_and_causal_state_changes() -> None:
    module = _load_module()
    records, slow, medium, fast = module.make_batch(2, torch.device("cpu"), seed=7)
    assert records.shape == (2, module.SEQ_LEN, 12)
    assert slow.shape == medium.shape == fast.shape == (2, module.SEQ_LEN)
    assert set(records[:, :, 0].unique().tolist()).issubset(
        {module.NOTE, module.PROGRAM, module.TEMPO, module.PEDAL}
    )
    assert torch.equal(slow[:, 0], slow[:, -1])
    assert not torch.equal(medium[:, 0], medium[:, 64]) or not torch.equal(
        medium[:, 1], medium[:, 65]
    )


def test_parameter_matched_models_are_effectively_equal_size() -> None:
    module = _load_module()
    shared = module.SharedMatched()
    routed = module.RoutedMultiBank()
    shared_params = sum(parameter.numel() for parameter in shared.parameters())
    routed_params = sum(parameter.numel() for parameter in routed.parameters())
    assert shared_params == 30327
    assert routed_params == 30329
    assert abs(shared_params - routed_params) <= 2


def test_models_have_expected_output_contract_and_backpropagate() -> None:
    module = _load_module()
    records, slow, medium, fast = module.make_batch(2, torch.device("cpu"), seed=11)
    active = torch.arange(module.SEQ_LEN) >= module.ACTIVE_START
    for model_type in (module.SharedMatched, module.RoutedMultiBank):
        torch.manual_seed(3)
        model = model_type()
        slow_logits, medium_logits, fast_logits, event_logits = model(records)
        assert slow_logits.shape == medium_logits.shape == fast_logits.shape == (
            2,
            module.SEQ_LEN,
            module.N_STATE,
        )
        assert event_logits.shape == (2, module.SEQ_LEN, 10)
        loss = (
            torch.nn.functional.cross_entropy(
                slow_logits[:, active].reshape(-1, module.N_STATE),
                slow[:, active].reshape(-1),
            )
            + torch.nn.functional.cross_entropy(
                medium_logits[:, active].reshape(-1, module.N_STATE),
                medium[:, active].reshape(-1),
            )
            + torch.nn.functional.cross_entropy(
                fast_logits[:, active].reshape(-1, module.N_STATE),
                fast[:, active].reshape(-1),
            )
        )
        loss.backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
