from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from .compound_lora import LoRALinear, iter_lora_modules


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if not child_name:
        raise ValueError("cannot replace the root module")
    setattr(parent, child_name, replacement)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_lora_linear(module: LoRALinear) -> nn.Linear:
    """Materialize one eval-mode LoRA wrapper as a plain ``nn.Linear``.

    For inference, dropout is disabled. The low-rank update is folded into the
    Base weight once, and the Base bias is unchanged. Returning a normal
    ``nn.Linear`` keeps the existing Compound checkpoint and Web-export paths
    independent of a dynamic LoRA runtime ABI.
    """

    if module.training:
        raise ValueError("LoRA must be in eval mode before deterministic merge")
    base = module.base
    merged = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    with torch.no_grad():
        delta = torch.matmul(module.lora_B, module.lora_A) * module.scaling
        merged.weight.copy_(base.weight + delta.to(device=base.weight.device, dtype=base.weight.dtype))
        if base.bias is not None:
            assert merged.bias is not None
            merged.bias.copy_(base.bias)
    for parameter in merged.parameters():
        parameter.requires_grad_(False)
    merged.eval()
    return merged


def _default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 5e-3, 5e-3
    return 1e-5, 1e-6


def verify_lora_merge_parity(model: nn.Module) -> dict[str, object]:
    """Numerically compare each eval-mode wrapper with its merged linear.

    The probe is deterministic and bounded: two synthetic rows per target
    module. This is a merge-implementation check, not a musical-quality or
    end-to-end model-equivalence claim. Later V2 native/ORT/WASM parity remains
    required for the exact exported derivative.
    """

    if model.training:
        raise ValueError("model must be in eval mode before LoRA merge parity verification")
    modules = sorted(iter_lora_modules(model), key=lambda item: item[0])
    if not modules:
        raise ValueError("model contains no experimental Compound LoRA modules")
    per_module: dict[str, dict[str, float | str]] = {}
    maximum = 0.0
    with torch.no_grad():
        for name, module in modules:
            merged = merge_lora_linear(module)
            count = max(2, 2 * module.base.in_features)
            probe = torch.linspace(
                -1.0,
                1.0,
                steps=count,
                device=module.base.weight.device,
                dtype=module.base.weight.dtype,
            )[: 2 * module.base.in_features].reshape(2, module.base.in_features)
            expected = module(probe)
            actual = merged(probe)
            diff = (actual - expected).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            rtol, atol = _default_tolerances(module.base.weight.dtype)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            maximum = max(maximum, max_abs)
            per_module[name] = {
                "dtype": str(module.base.weight.dtype).removeprefix("torch."),
                "max_abs": max_abs,
                "rtol": rtol,
                "atol": atol,
            }
    return {"max_abs": maximum, "modules": per_module}


def merge_lora_inplace(model: nn.Module) -> list[str]:
    """Replace every experimental Compound LoRA wrapper with merged linears.

    The model must be in evaluation mode. A sorted snapshot of module names is
    taken before mutation so nested replacement cannot change iteration
    semantics and provenance order stays deterministic for any layer count.
    """

    if model.training:
        raise ValueError("model must be in eval mode before deterministic LoRA merge")
    modules = sorted(iter_lora_modules(model), key=lambda item: item[0])
    if not modules:
        raise ValueError("model contains no experimental Compound LoRA modules")
    names = [name for name, _ in modules]
    for name, module in modules:
        _replace_module(model, name, merge_lora_linear(module))
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("LoRA merge left wrapper modules in the model")
    return names


def adapter_artifact_digests(adapter_dir: str | Path) -> dict[str, str]:
    target = Path(adapter_dir)
    manifest = target / "adapter.json"
    tensors = target / "adapter.safetensors"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not tensors.is_file():
        raise FileNotFoundError(tensors)
    return {
        "manifest_sha256": sha256_file(manifest),
        "tensor_sha256": sha256_file(tensors),
    }
