from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SCRIPT = Path(__file__).parents[1] / "experiments" / "recurrent_memory_two_stage_sweep.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("orbitune_recurrent_memory_two_stage", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_distances_are_outside_local_window() -> None:
    module = _load_module()
    assert module.LOCAL_WINDOW == 16
    assert module.DISTANCES == (32, 64, 128, 256)
    assert all(distance > module.LOCAL_WINDOW for distance in module.DISTANCES)


def test_downstream_freezes_pretrained_memory_and_preserves_shapes() -> None:
    module = _load_module()
    torch.manual_seed(1)
    memory = module.MemoryEncoder()
    model = module.FixedResidual(memory)
    assert all(not parameter.requires_grad for parameter in model.memory.parameters())
    assert any(parameter.requires_grad for parameter in model.local.parameters())

    x, _, query_mask, _ = module.train_batch(2, 256, torch.device("cpu"), seed=3)
    logits = model(x)
    assert logits.shape == (2, 257, module.VOCAB)
    assert int(query_mask.sum()) == 2


def test_vectorized_linear_memory_matches_recurrent_update() -> None:
    module = _load_module()
    torch.manual_seed(4)
    layer = module.LinearMemory().eval()
    h = torch.randn(2, 12, module.D)

    actual, actual_slots = layer(h)

    x = layer.norm(h)
    q = F.elu(layer.q(x)) + 1
    k = F.elu(layer.k(x)) + 1
    v = layer.v(x)
    write = torch.sigmoid(layer.write(x))
    decay = torch.sigmoid(layer.logit_decay).clamp(0.9, 0.9999)
    state = h.new_zeros((h.shape[0], module.K, module.D))
    normalizer = h.new_zeros((h.shape[0], module.K))
    outputs = []
    slots = []
    for index in range(h.shape[1]):
        state = decay * state + write[:, index, :, None] * torch.einsum(
            "bk,bd->bkd", k[:, index], v[:, index]
        )
        normalizer = decay * normalizer + write[:, index] * k[:, index]
        slot = state / (normalizer[:, :, None] + 1e-5)
        read = torch.einsum("bk,bkd->bd", q[:, index], state) / (
            torch.einsum("bk,bk->b", q[:, index], normalizer)[:, None] + 1e-5
        )
        outputs.append(layer.mix(torch.cat([h[:, index], read], dim=-1)))
        slots.append(slot)

    expected = torch.stack(outputs, dim=1)
    expected_slots = torch.stack(slots, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual_slots, expected_slots, rtol=2e-4, atol=2e-5)


def test_all_conditioning_arms_accept_same_frozen_memory_contract() -> None:
    module = _load_module()
    x, _, _, _ = module.train_batch(1, 256, torch.device("cpu"), seed=5)
    for name, model_type in module.MODELS.items():
        torch.manual_seed(6)
        model = model_type(module.MemoryEncoder())
        assert all(not parameter.requires_grad for parameter in model.memory.parameters()), name
        logits = model(x)
        assert logits.shape == (1, 257, module.VOCAB), name
