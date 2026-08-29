from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"
EXPERIMENT_DIR = SCRIPT.parent


def _load(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_matched_wrapper_has_no_import_order_side_effects() -> None:
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    base = importlib.import_module("real_compound_memory_experiment")
    original_shared = base.SharedMatched
    original_models = dict(base.MODELS)

    first = _load("orbitune_matched_import_repro_first")
    second = _load("orbitune_matched_import_repro_second")

    assert base.SharedMatched is original_shared
    assert base.MODELS == original_models
    assert first.SharedMatched is not second.SharedMatched
    assert issubclass(first.SharedMatched, original_shared)
    assert issubclass(second.SharedMatched, original_shared)

    routed_parameters = sum(p.numel() for p in base.RoutedMultiBank().parameters())
    first_parameters = sum(p.numel() for p in first.SharedMatched().parameters())
    second_parameters = sum(p.numel() for p in second.SharedMatched().parameters())
    assert first_parameters == second_parameters == routed_parameters == 157650

    torch.manual_seed(20260829)
    model_a = first.SharedMatched().eval()
    torch.manual_seed(20260829)
    model_b = second.SharedMatched().eval()
    for (name_a, value_a), (name_b, value_b) in zip(
        model_a.state_dict().items(), model_b.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(value_a, value_b), name_a

    records = torch.zeros((1, 6, 12), dtype=torch.long)
    # NOTE requires positive velocity/duration values only when interpreted by
    # the MIDI target extractor; the model itself consumes factor indices.
    records[:, :, 0] = 0
    records[:, :, 1] = 0
    records[:, :, 2] = 1
    records[:, :, 3] = 0
    records[:, :, 4] = 60
    records[:, :, 6] = 80
    records[:, :, 8] = 1
    with torch.no_grad():
        outputs_a = model_a.forward_chunk(records, None)
        outputs_b = model_b.forward_chunk(records, None)

    def flatten(value):  # type: ignore[no-untyped-def]
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(flatten(item))
            return out
        raise TypeError(type(value))

    tensors_a = flatten(outputs_a)
    tensors_b = flatten(outputs_b)
    assert len(tensors_a) == len(tensors_b)
    for left, right in zip(tensors_a, tensors_b, strict=True):
        assert torch.equal(left, right)
