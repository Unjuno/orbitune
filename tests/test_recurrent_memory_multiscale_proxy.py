from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "experiments" / "recurrent_memory_multiscale_proxy.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("orbitune_recurrent_memory_multiscale", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multiscale_task_has_distinct_update_periods() -> None:
    module = _load_module()
    ids, slow, medium, fast = module.make_batch(2, torch.device("cpu"), seed=1)
    assert ids.shape == slow.shape == medium.shape == fast.shape == (2, 256)
    assert torch.equal(slow[:, 0], ids[:, 0])
    assert not torch.equal(medium[:, 95], medium[:, 96]) or not torch.equal(
        medium[:, 191], medium[:, 192]
    )


def test_all_memory_modes_forward_with_scalar_and_vector_decay() -> None:
    module = _load_module()
    ids, _, _, _ = module.make_batch(2, torch.device("cpu"), seed=2)
    for name, factory in module.MODELS.items():
        torch.manual_seed(3)
        model = factory()
        slow, medium, fast, reconstruction = model(ids)
        assert slow.shape == medium.shape == fast.shape == (2, 256, module.N_STATE), name
        assert reconstruction.shape == (2, 256, module.VOCAB), name
        assert all(torch.isfinite(value).all() for value in (slow, medium, fast, reconstruction)), name


def test_multibank_has_independent_write_read_paths_and_decay_bands() -> None:
    module = _load_module()
    model = module.IndependentMultiBank()
    assert len(model.banks) == 3
    assert model.banks[0].q.weight.data_ptr() != model.banks[1].q.weight.data_ptr()
    assert model.banks[0].write.weight.data_ptr() != model.banks[2].write.weight.data_ptr()
    decays = [float(bank.decay()[0]) for bank in model.banks]
    assert decays == [pytest_approx(0.90), pytest_approx(0.97), pytest_approx(0.995)]


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-6)
