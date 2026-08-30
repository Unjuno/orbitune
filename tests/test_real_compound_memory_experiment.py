from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.tokenizer.compound_event import COMPOUND_RECORD_WIDTH


SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"


def _load_module():  # type: ignore[no-untyped-def]
    name = "orbitune_real_compound_memory_experiment_matched"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _records(offset: int, count: int = 24) -> list[list[int]]:
    rows: list[list[int]] = []
    for index in range(count):
        if index % 8 == 0:
            row = [2, index % 2, 1, 0, (offset + index * 7) % 128, 0, 0, 0, 0, 0, 0, 0]
        elif index % 11 == 0:
            row = [4, 0, 1, 0, 180 + offset + index, 0, 0, 0, 0, 0, 0, 0]
        elif index % 13 == 0:
            row = [5, index % 2, 1, 0, (index // 13) % 2, 0, 0, 0, 0, 0, 0, 0]
        else:
            pitch = 36 + ((offset + index * 5) % 60)
            velocity = 32 + ((offset + index * 9) % 80)
            row = [0, index % 2, 1, index % 16, pitch, 0, velocity, 0, 1, index % 16, 0, 0]
        assert len(row) == COMPOUND_RECORD_WIDTH
        rows.append(row)
    return rows


def _write_jsonl(path: Path, *, offset: int, digest: str) -> None:
    payload = {
        "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
        "record_width": COMPOUND_RECORD_WIDTH,
        "path": f"fixture-{offset}.mid",
        "sha256": digest,
        "events": 24,
        "records": _records(offset),
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_real_harness_loads_disjoint_compound_splits_and_profiles_targets(tmp_path: Path) -> None:
    module = _load_module()
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(train, offset=1, digest="1" * 64)
    _write_jsonl(validation, offset=17, digest="2" * 64)

    train_songs, validation_songs = module.load_splits(train, validation)
    assert len(train_songs) == len(validation_songs) == 1
    assert train_songs[0].sha256 != validation_songs[0].sha256
    profile = module.target_profile(train_songs, warmup_events=2)
    assert profile["fast"]["events"] == 22
    assert profile["medium"]["events"] == 22
    assert profile["slow"]["events"] == 22


def test_real_harness_models_are_parameter_matched_and_state_is_fixed_size() -> None:
    module = _load_module()
    shared = module.SharedMatched()
    routed = module.RoutedMultiBank()
    shared_parameters = sum(parameter.numel() for parameter in shared.parameters())
    routed_parameters = sum(parameter.numel() for parameter in routed.parameters())
    assert shared_parameters == routed_parameters

    records = torch.tensor([_records(3, count=12)], dtype=torch.long)
    *_, shared_state = shared.forward_chunk(records, None)
    *_, routed_state = routed.forward_chunk(records, None)
    assert shared_state[0].shape == (1, 6, module.D_MODEL)
    assert shared_state[1].shape == (1, 6)
    assert len(routed_state) == 3
    for state in routed_state:
        assert state[0].shape == (1, 2, module.D_MODEL)
        assert state[1].shape == (1, 2)


def test_frozen_composer_policy_keeps_memory_parameters_closed() -> None:
    module = _load_module()
    model = module.RoutedMultiBank()
    optimizer = module._configure_composer_optimizer(
        model,
        "frozen",
        composer_lr=3e-3,
        memory_lr_multiplier=0.1,
    )
    del optimizer
    for name, parameter in model.named_parameters():
        if name.startswith("event_head") or name.startswith("event_mix"):
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_real_harness_rejects_exact_sha_leakage(tmp_path: Path) -> None:
    module = _load_module()
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(train, offset=1, digest="a" * 64)
    _write_jsonl(validation, offset=9, digest="a" * 64)
    try:
        module.load_splits(train, validation)
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:
        raise AssertionError("exact SHA leakage must be rejected")
