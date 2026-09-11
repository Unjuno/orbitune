from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors


EXPERIMENTAL_COMPOUND_LORA_SCHEMA = "orbitune-compound-lora-experimental-v0"
ADAPTER_TENSOR_FILE = "adapter.safetensors"
ADAPTER_MANIFEST_FILE = "adapter.json"


@dataclass(frozen=True, slots=True)
class CompoundLoRAConfig:
    target_patterns: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float = 0.0

    def validate(self) -> None:
        if not self.target_patterns:
            raise ValueError("at least one LoRA target pattern is required")
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")


class LoRALinear(nn.Module):
    """Experimental, non-merged LoRA wrapper for one frozen ``nn.Linear``.

    The wrapped Base module is kept as-is and frozen. ``lora_B`` starts at
    zero, so injecting the Adapter is initially a no-op. This module is an
    implementation primitive only; its tensor layout is not the public
    Compound Adapter ABI.
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0.0 or not math.isfinite(alpha):
            raise ValueError("alpha must be finite and positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapted = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + adapted * self.scaling


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return normalized


def freeze_base(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if not child_name:
        raise ValueError("cannot replace the root module")
    setattr(parent, child_name, replacement)


def resolve_target_modules(model: nn.Module, patterns: Iterable[str]) -> list[str]:
    requested = tuple(str(pattern) for pattern in patterns)
    if not requested:
        raise ValueError("at least one target pattern is required")

    matches: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in requested):
            matches.append(name)

    if not matches:
        raise ValueError(f"LoRA target patterns matched no nn.Linear modules: {requested!r}")
    return sorted(set(matches))


def inject_lora(model: nn.Module, config: CompoundLoRAConfig) -> list[str]:
    """Freeze ``model`` and inject trainable LoRA matrices into matched linears."""

    config.validate()
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("model already contains experimental LoRA modules")

    freeze_base(model)
    targets = resolve_target_modules(model, config.target_patterns)
    originals = {name: model.get_submodule(name) for name in targets}
    replacements: dict[str, LoRALinear] = {}
    for name in targets:
        base = originals[name]
        if not isinstance(base, nn.Linear):
            raise TypeError(f"resolved target is no longer nn.Linear: {name}")
        replacements[name] = LoRALinear(
            base,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
    for name in targets:
        _replace_module(model, name, replacements[name])
    return targets


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def assert_only_lora_trainable(model: nn.Module) -> None:
    trainable = trainable_parameter_names(model)
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters found")
    invalid = [
        name for name in trainable
        if not (name.endswith(".lora_A") or name.endswith(".lora_B"))
    ]
    if invalid:
        raise RuntimeError(f"non-LoRA parameters are trainable: {invalid}")


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, module in iter_lora_modules(model):
        tensors[f"{name}.lora_A"] = module.lora_A.detach().cpu().contiguous()
        tensors[f"{name}.lora_B"] = module.lora_B.detach().cpu().contiguous()
    if not tensors:
        raise RuntimeError("model contains no experimental LoRA modules")
    return tensors


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def base_parameter_digests(model: nn.Module) -> dict[str, str]:
    """Hash all non-LoRA parameters to prove Base immutability across training."""

    digests: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            continue
        digests[name] = _tensor_digest(parameter)
    return digests


def assert_base_unchanged(model: nn.Module, before: dict[str, str]) -> None:
    after = base_parameter_digests(model)
    if before.keys() != after.keys():
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        raise RuntimeError(f"Base parameter set changed; missing={missing}, added={added}")
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise RuntimeError(f"Base parameters changed during Adapter training: {changed}")


def save_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model_id: str,
    base_sha256: str,
    architecture_abi: str,
    tokenizer_abi: str,
    source_commit: str | None,
    training_config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Save an explicitly experimental Adapter artifact.

    This format is intentionally not named as a stable Compound Adapter ABI.
    It exists to exercise the post-Base LoRA path before the public ABI is
    selected and frozen from measurements on the final Base.
    """

    base_sha256 = _validate_sha256(base_sha256, field="base_sha256")
    modules = list(iter_lora_modules(model))
    if not modules:
        raise RuntimeError("model contains no experimental LoRA modules")
    ranks = {module.rank for _, module in modules}
    alphas = {module.alpha for _, module in modules}
    dropouts = {
        float(module.dropout.p) if isinstance(module.dropout, nn.Dropout) else 0.0
        for _, module in modules
    }
    if len(ranks) != 1 or len(alphas) != 1 or len(dropouts) != 1:
        raise RuntimeError("experimental writer currently requires uniform rank/alpha/dropout")

    rank = next(iter(ranks))
    alpha = next(iter(alphas))
    dropout = next(iter(dropouts))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tensor_path = target / ADAPTER_TENSOR_FILE
    manifest_path = target / ADAPTER_MANIFEST_FILE

    manifest: dict[str, object] = {
        "schema": EXPERIMENTAL_COMPOUND_LORA_SCHEMA,
        "status": "experimental",
        "public_adapter_abi": False,
        "base_model_id": str(base_model_id),
        "base_sha256": base_sha256,
        "architecture_abi": str(architecture_abi),
        "tokenizer_abi": str(tokenizer_abi),
        "target_modules": [name for name, _ in modules],
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "scaling": alpha / rank,
        "tensor_file": ADAPTER_TENSOR_FILE,
        "source_commit": source_commit,
        "training_config": training_config or {},
    }
    save_safetensors(
        adapter_state_dict(model),
        str(tensor_path),
        metadata={
            "schema": EXPERIMENTAL_COMPOUND_LORA_SCHEMA,
            "base_sha256": base_sha256,
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    *,
    base_sha256: str,
    strict_base_binding: bool = True,
) -> dict[str, object]:
    """Preflight an Adapter completely, then mutate ``model`` only after validation."""

    target = Path(adapter_dir)
    manifest = json.loads((target / ADAPTER_MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPERIMENTAL_COMPOUND_LORA_SCHEMA:
        raise ValueError("unsupported experimental Compound LoRA schema")

    requested_base = _validate_sha256(base_sha256, field="base_sha256")
    expected_base = _validate_sha256(str(manifest.get("base_sha256", "")), field="adapter base_sha256")
    if strict_base_binding and expected_base != requested_base:
        raise ValueError(
            f"Adapter Base SHA mismatch: adapter={expected_base!r}, requested={requested_base!r}"
        )

    model_architecture = getattr(model, "architecture", None)
    model_tokenizer = getattr(model, "tokenizer", None)
    if model_architecture != manifest.get("architecture_abi"):
        raise ValueError("Adapter architecture ABI does not match model")
    if model_tokenizer != manifest.get("tokenizer_abi"):
        raise ValueError("Adapter tokenizer ABI does not match model")

    config = CompoundLoRAConfig(
        target_patterns=tuple(str(name) for name in manifest.get("target_modules", [])),
        rank=int(manifest["rank"]),
        alpha=float(manifest["alpha"]),
        dropout=float(manifest.get("dropout", 0.0)),
    )
    config.validate()
    targets = resolve_target_modules(model, config.target_patterns)
    if targets != sorted(config.target_patterns):
        raise ValueError("Adapter target modules do not resolve exactly on this Base")

    tensors = load_safetensors(str(target / str(manifest.get("tensor_file", ADAPTER_TENSOR_FILE))))
    expected_keys = {
        key
        for name in targets
        for key in (f"{name}.lora_A", f"{name}.lora_B")
    }
    missing = sorted(expected_keys - set(tensors))
    unexpected = sorted(set(tensors) - expected_keys)
    if missing:
        raise ValueError(f"Adapter tensors missing: {missing}")
    if unexpected:
        raise ValueError(f"unexpected Adapter tensors: {unexpected}")

    for name in targets:
        base = model.get_submodule(name)
        if not isinstance(base, nn.Linear):
            raise ValueError(f"Adapter target is not nn.Linear on this Base: {name}")
        expected_a = (config.rank, base.in_features)
        expected_b = (base.out_features, config.rank)
        if tuple(tensors[f"{name}.lora_A"].shape) != expected_a:
            raise ValueError(f"Adapter lora_A shape mismatch for {name}")
        if tuple(tensors[f"{name}.lora_B"].shape) != expected_b:
            raise ValueError(f"Adapter lora_B shape mismatch for {name}")

    injected = inject_lora(model, config)
    if injected != targets:
        raise RuntimeError("LoRA injection target set changed after preflight")
    for name, module in iter_lora_modules(model):
        module.lora_A.data.copy_(tensors[f"{name}.lora_A"].to(module.lora_A))
        module.lora_B.data.copy_(tensors[f"{name}.lora_B"].to(module.lora_B))

    assert_only_lora_trainable(model)
    return manifest
