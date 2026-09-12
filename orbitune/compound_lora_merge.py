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

    For inference, dropout is disabled and the wrapper computes

        base(x) + scaling * B(A(x)).

    Therefore the exact merged weight is ``W + scaling * (B @ A)`` while
    the Base bias is unchanged. This helper deliberately returns a normal
    ``nn.Linear`` so the existing Compound checkpoint and Web-export paths do
    not need a LoRA-specific runtime ABI.
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
