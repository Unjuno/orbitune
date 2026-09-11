from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_lora import (
    CompoundLoRAConfig,
    assert_base_unchanged,
    assert_only_lora_trainable,
    base_parameter_digests,
    inject_lora,
    load_adapter,
    save_adapter,
    sha256_file,
    trainable_parameter_names,
)


def _tiny() -> CompoundBaseConfig:
    return CompoundBaseConfig(
        d_model=32,
        n_head=4,
        local_layers=1,
        medium_layers=1,
        global_layers=1,
        intra_layers=1,
        ff_mult=2,
        dropout=0.0,
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


def _config() -> CompoundLoRAConfig:
    return CompoundLoRAConfig(
        target_patterns=("decoder.stack.blocks.*.attn.q_proj",),
        rank=4,
        alpha=8.0,
    )


def _write_dataset(path: Path) -> None:
    records = _records()[0].tolist() * 3
    payload = {
        "path": "fixture.mid",
        "sha256": "fixture",
        "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
        "record_width": 12,
        "records": records,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_injection_is_initially_noop_and_only_lora_is_trainable() -> None:
    torch.manual_seed(3)
    model = CompoundHierarchicalGPT(_tiny()).eval()
    inputs = _records()[:, :-1]
    targets = _records()[:, 1:]
    before, _ = model(inputs, targets)

    resolved = inject_lora(model, _config())
    after, _ = model(inputs, targets)

    assert resolved == ["decoder.stack.blocks.0.attn.q_proj"]
    assert torch.equal(before, after)
    assert_only_lora_trainable(model)
    trainable = trainable_parameter_names(model)
    assert trainable == [
        "decoder.stack.blocks.0.attn.q_proj.lora_A",
        "decoder.stack.blocks.0.attn.q_proj.lora_B",
    ]


def test_optimizer_updates_adapter_but_not_base() -> None:
    torch.manual_seed(4)
    model = CompoundHierarchicalGPT(_tiny())
    inject_lora(model, _config())
    assert_only_lora_trainable(model)
    base_before = base_parameter_digests(model)
    adapter_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    inputs = _records()[:, :-1]
    targets = _records()[:, 1:]
    loss, _ = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert_base_unchanged(model, base_before)
    adapter_after = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert any(
        not torch.equal(adapter_before[name], adapter_after[name])
        for name in adapter_before
    )


def test_adapter_save_reload_and_base_binding(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = CompoundHierarchicalGPT(_tiny())
    inject_lora(model, _config())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.fill_(0.125)

    adapter_dir = tmp_path / "adapter"
    save_adapter(
        model,
        adapter_dir,
        base_model_id="fixture-base",
        base_sha256="a" * 64,
        architecture_abi=model.architecture,
        tokenizer_abi=model.tokenizer,
        source_commit="test",
    )

    clean = CompoundHierarchicalGPT(_tiny())
    manifest = load_adapter(clean, adapter_dir, base_sha256="a" * 64)
    assert manifest["status"] == "experimental"
    assert manifest["public_adapter_abi"] is False

    source = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    restored = {
        name: parameter.detach().clone()
        for name, parameter in clean.named_parameters()
        if parameter.requires_grad
    }
    assert source.keys() == restored.keys()
    for name in source:
        assert torch.equal(source[name], restored[name])

    wrong_base = CompoundHierarchicalGPT(_tiny())
    with pytest.raises(ValueError, match="Base SHA mismatch"):
        load_adapter(wrong_base, adapter_dir, base_sha256="b" * 64)


def test_experimental_sft_script_end_to_end(tmp_path: Path) -> None:
    torch.manual_seed(6)
    model = CompoundHierarchicalGPT(_tiny())
    checkpoint = tmp_path / "base.pt"
    model.save_checkpoint(checkpoint, step=7, source_commit="fixture-source")
    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    _write_dataset(train_jsonl)
    _write_dataset(validation_jsonl)
    output = tmp_path / "adapter-out"

    command = [
        sys.executable,
        "scripts/compound_lora_sft.py",
        "--base-checkpoint",
        str(checkpoint),
        "--base-id",
        "fixture-base",
        "--train-jsonl",
        str(train_jsonl),
        "--validation-jsonl",
        str(validation_jsonl),
        "--output-dir",
        str(output),
        "--target-module",
        "decoder.stack.blocks.*.attn.q_proj",
        "--rank",
        "4",
        "--alpha",
        "8",
        "--steps",
        "2",
        "--batch-size",
        "1",
        "--seq-len",
        "3",
        "--validation-batches",
        "1",
        "--device",
        "cpu",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (output / "adapter.safetensors").is_file()
    assert (output / "adapter.json").is_file()
    assert (output / "training.json").is_file()
    assert (output / "metrics.json").is_file()

    manifest = json.loads((output / "adapter.json").read_text(encoding="utf-8"))
    assert manifest["base_sha256"] == sha256_file(checkpoint)
    assert manifest["base_model_id"] == "fixture-base"
    assert manifest["target_modules"] == ["decoder.stack.blocks.0.attn.q_proj"]
    assert manifest["public_adapter_abi"] is False
