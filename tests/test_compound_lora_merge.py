from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_lora import CompoundLoRAConfig, LoRALinear, inject_lora, save_adapter, sha256_file
from orbitune.compound_lora_merge import merge_lora_inplace, merge_lora_linear, verify_lora_merge_parity


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
    ]
    return torch.tensor([rows], dtype=torch.long)


def _lora_config(*, dropout: float = 0.0) -> CompoundLoRAConfig:
    return CompoundLoRAConfig(
        target_patterns=("decoder.stack.blocks.*.attn.q_proj",),
        rank=4,
        alpha=8.0,
        dropout=dropout,
    )


def _adapted_model(*, dropout: float = 0.0) -> CompoundHierarchicalGPT:
    torch.manual_seed(31)
    model = CompoundHierarchicalGPT(_tiny())
    inject_lora(model, _lora_config(dropout=dropout))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.copy_(torch.linspace(-0.02, 0.02, parameter.numel()).reshape_as(parameter))
    return model


def test_merge_lora_linear_matches_eval_wrapper_with_dropout_configured() -> None:
    model = _adapted_model(dropout=0.25).eval()
    module = model.get_submodule("decoder.stack.blocks.0.attn.q_proj")
    assert isinstance(module, LoRALinear)
    x = torch.randn(2, 3, module.base.in_features)
    expected = module(x)
    merged = merge_lora_linear(module)
    actual = merged(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in merged.parameters())


def test_bounded_merge_parity_probe_reports_target_error() -> None:
    model = _adapted_model(dropout=0.25).eval()
    report = verify_lora_merge_parity(model)
    assert set(report["modules"]) == {"decoder.stack.blocks.0.attn.q_proj"}
    assert report["max_abs"] >= 0.0
    assert report["modules"]["decoder.stack.blocks.0.attn.q_proj"]["dtype"] == "float32"


def test_merge_lora_inplace_preserves_model_outputs_and_restores_base_state_keys() -> None:
    model = _adapted_model().eval()
    inputs = _records()[:, :-1]
    targets = _records()[:, 1:]
    expected_loss, expected_parts = model(inputs, targets)

    names = merge_lora_inplace(model)
    assert names == ["decoder.stack.blocks.0.attn.q_proj"]
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    actual_loss, actual_parts = model(inputs, targets)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    assert actual_parts.keys() == expected_parts.keys()
    for key in actual_parts:
        torch.testing.assert_close(actual_parts[key], expected_parts[key], rtol=1e-5, atol=1e-6)

    clean = CompoundHierarchicalGPT(_tiny())
    clean.load_state_dict(model.state_dict(), strict=True)


def test_merge_refuses_training_mode() -> None:
    model = _adapted_model()
    with pytest.raises(ValueError, match="eval mode"):
        merge_lora_inplace(model)
    with pytest.raises(ValueError, match="eval mode"):
        verify_lora_merge_parity(model)


def test_premerge_cli_binds_exact_base_and_emits_reloadable_derivative(tmp_path: Path) -> None:
    torch.manual_seed(37)
    base = CompoundHierarchicalGPT(_tiny())
    base_path = tmp_path / "base.pt"
    base.save_checkpoint(base_path, step=19, source_commit="fixture-source")
    base_sha = sha256_file(base_path)

    adapted, _ = CompoundHierarchicalGPT.load_checkpoint(base_path)
    inject_lora(adapted, _lora_config())
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if name.endswith(".lora_B"):
                parameter.fill_(0.01)
    adapter_dir = tmp_path / "adapter"
    save_adapter(
        adapted,
        adapter_dir,
        base_model_id="fixture-a2",
        base_sha256=base_sha,
        architecture_abi=adapted.architecture,
        tokenizer_abi=adapted.tokenizer,
        source_commit="adapter-source",
    )
    adapted.eval()
    inputs = _records()[:, :-1]
    targets = _records()[:, 1:]
    expected_loss, _ = adapted(inputs, targets)

    merged_path = tmp_path / "merged.pt"
    manifest_path = tmp_path / "merged.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/merge_compound_lora.py",
            "--base-checkpoint", str(base_path),
            "--base-sha256", base_sha,
            "--base-model-id", "fixture-a2",
            "--adapter-dir", str(adapter_dir),
            "--adapter-id", "fixture-style-v1",
            "--output-checkpoint", str(merged_path),
            "--output-manifest", str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    merged, payload = CompoundHierarchicalGPT.load_checkpoint(merged_path)
    merged.eval()
    actual_loss, _ = merged(inputs, targets)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    assert payload["optimizer_state_dict"] is None
    assert payload["derived_artifact"]["kind"] == "lora-premerged"
    assert payload["derived_artifact"]["base_checkpoint_sha256"] == base_sha
    assert payload["derived_artifact"]["adapter_id"] == "fixture-style-v1"
    assert payload["derived_artifact"]["merge_parity"]["max_abs"] >= 0.0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "experimental_premerged_candidate"
    assert manifest["publication_eligible"] is False
    assert manifest["public_adapter_abi"] is False
    assert manifest["base"]["checkpoint_sha256"] == base_sha
    assert manifest["adapter"]["id"] == "fixture-style-v1"
    assert manifest["merge_parity"]["max_abs"] >= 0.0
    assert manifest["merged_checkpoint"]["sha256"] == sha256_file(merged_path)
