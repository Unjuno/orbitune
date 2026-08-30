from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "experiments" / "compound_memory_freeze_policy_proxy.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("compound_memory_freeze_policy_proxy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_policy_only_trains_composer_path() -> None:
    module = _load_module()
    model = module.RoutedMultiBank()
    module._composer_optimizer(model, "frozen")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith("event_head") or name.startswith("event_mix")
        for name in trainable
    )
    assert not any("_memory" in name for name in trainable)
    assert not any(name.startswith("embedding") for name in trainable)


def test_low_lr_policy_keeps_target_heads_out_of_stage2_optimizer() -> None:
    module = _load_module()
    model = module.RoutedMultiBank()
    optimizer = module._composer_optimizer(model, "low_lr")
    assert sorted(group["lr"] for group in optimizer.param_groups) == [3e-4, 3e-3]
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert any(name.startswith("fast_heads") for name in frozen)
    assert any(name.startswith("medium_heads") for name in frozen)
    assert any(name.startswith("slow_heads") for name in frozen)
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("embedding") or "_memory" in name
    )


def test_joint_policy_leaves_memory_trainable() -> None:
    module = _load_module()
    model = module.RoutedMultiBank()
    optimizer = module._composer_optimizer(model, "joint")
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 3e-3
    assert all(parameter.requires_grad for parameter in model.parameters())
