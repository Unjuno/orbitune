from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "experiments" / "recurrent_memory_arch_proxy.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("orbitune_recurrent_memory_proxy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proxy_query_is_outside_local_window() -> None:
    module = _load_module()
    inputs, targets, query_mask = module.make_batch(2, 48, torch.device("cpu"), seed=1)

    assert inputs.shape == targets.shape == query_mask.shape == (2, 47)
    assert query_mask[:, 32].all()
    assert module.local_mask(47, 16, torch.device("cpu"))[32, 0].isneginf()
    assert module.N_STATE == 8


def test_all_proxy_arms_have_expected_output_contract() -> None:
    module = _load_module()
    inputs, _, _ = module.make_batch(2, 48, torch.device("cpu"), seed=2)

    for name, model_type in module.MODELS.items():
        torch.manual_seed(3)
        model = model_type(max_len=48)
        logits, memory_logits = model(inputs)
        assert logits.shape == (2, 47, module.VOCAB), name
        if name == "D_consolidated":
            assert memory_logits is not None
            assert memory_logits.shape == logits.shape
        else:
            assert memory_logits is None


def test_consolidated_proxy_backpropagates_through_memory_objective() -> None:
    module = _load_module()
    torch.manual_seed(4)
    model = module.ConsolidatedMemoryThenTransformer(max_len=48)
    inputs, targets, query_mask = module.make_batch(2, 48, torch.device("cpu"), seed=4)
    logits, memory_logits = model(inputs)
    assert memory_logits is not None

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, module.VOCAB),
        targets.reshape(-1),
    ) + torch.nn.functional.cross_entropy(memory_logits[query_mask], targets[query_mask])
    loss.backward()

    assert model.memory.memory.q.weight.grad is not None
    assert torch.isfinite(model.memory.memory.q.weight.grad).all()
