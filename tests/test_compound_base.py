from __future__ import annotations

import random
from pathlib import Path

import torch

from orbitune.compound import CompoundEvent, CompoundEventType
from orbitune.compound_base import (
    CompoundBaseConfig,
    CompoundHierarchicalGPT,
    write_compound_midi,
)
from orbitune.compound_midi import read_compound_midi


def _tiny() -> CompoundBaseConfig:
    return CompoundBaseConfig(
        d_model=32,
        n_head=4,
        local_layers=1,
        medium_layers=1,
        global_layers=1,
        intra_layers=1,
        ff_mult=2,
        dropout=0.1,
        local_window=8,
        medium_stride=2,
        medium_window=8,
        global_stride=2,
        global_window=8,
    )


def _records() -> torch.Tensor:
    rows = [
        (4, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 2, 60, 0, 82, 0, 1, 4, 0, 0),
        (0, 0, 1, 3, 64, 0, 76, 0, 1, 5, 0, 0),
        (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 3),
        (0, 0, 1, 4, 67, 0, 91, 0, 1, 6, 0, 0),
        (5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    ]
    return torch.tensor([rows], dtype=torch.long)


def test_compound_hierarchical_forward_backward() -> None:
    torch.manual_seed(1)
    model = CompoundHierarchicalGPT(_tiny())
    inputs = _records()[:, :-1]
    targets = _records()[:, 1:]
    loss, parts = model(inputs, targets)
    assert torch.isfinite(loss)
    assert "event_type" in parts
    assert "delta" in parts
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_compound_checkpoint_resume_is_exact(tmp_path: Path) -> None:
    torch.manual_seed(11)
    random.seed(11)
    model = CompoundHierarchicalGPT(_tiny())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = _records()[:, :-1]
    targets = _records()[:, 1:]

    loss, _ = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    checkpoint = tmp_path / "compound.pt"
    sampler = random.Random(99)
    model.save_checkpoint(
        checkpoint,
        optimizer=optimizer,
        step=1,
        source_commit="test",
        sampler_rng_state=sampler.getstate(),
    )

    direct_loss, _ = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    direct_loss.backward()
    optimizer.step()
    direct_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    resumed, payload = CompoundHierarchicalGPT.load_checkpoint(checkpoint)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"])
    random.setstate(payload["python_rng_state"])
    resumed_loss, _ = resumed(inputs, targets)
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_loss.backward()
    resumed_optimizer.step()

    assert torch.equal(direct_loss.detach(), resumed_loss.detach())
    for name, value in resumed.state_dict().items():
        assert torch.equal(direct_state[name], value)


def test_stream_state_is_bounded() -> None:
    model = CompoundHierarchicalGPT(_tiny()).eval()
    state = model.initial_stream_state()
    record = _records()[0, 1]
    for _ in range(80):
        context = model.advance_stream(record, state)
    assert context.shape == (1, _tiny().d_model)
    assert len(state.local_records) <= _tiny().local_window
    assert len(state.medium_history) <= _tiny().medium_window
    assert len(state.global_history) <= _tiny().global_window
    assert state.memory is not None


def test_compound_midi_writer_roundtrip(tmp_path: Path) -> None:
    source = [
        CompoundEvent(CompoundEventType.TEMPO, 0, 0, 120),
        CompoundEvent(CompoundEventType.PROGRAM, 0, 0, 5),
        CompoundEvent(CompoundEventType.NOTE, 0, 0, 60, 96, 80),
        CompoundEvent(CompoundEventType.NOTE, 96, 0, 64, 48, 90),
    ]
    target = tmp_path / "generated.mid"
    write_compound_midi(target, source)
    decoded = read_compound_midi(target)
    notes = [event for event in decoded if event.type is CompoundEventType.NOTE]
    assert [(event.step, event.a1, event.a2, event.a3) for event in notes] == [
        (0, 60, 96, 80),
        (96, 64, 48, 90),
    ]


def test_default_model_is_old_base_scale_not_proxy_scale() -> None:
    model = CompoundHierarchicalGPT()
    assert 8_000_000 <= model.parameter_count() <= 12_000_000
