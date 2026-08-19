from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from orbitune.model import OrbituneGPT
from orbitune.web_model import ExternalLoRALogitsModel, empty_web_lora


class LogitsOnlyModel(nn.Module):
    """Deployment wrapper that removes the training-only optional loss output."""

    def __init__(self, model: OrbituneGPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits


def _sequence_dim(model: OrbituneGPT) -> torch.export.Dim:
    return torch.export.Dim("sequence", min=1, max=model.config.max_seq_len)


def capture_exported_program(model: OrbituneGPT, *, example_seq_len: int = 64) -> torch.export.ExportedProgram:
    model = model.cpu().eval()
    if not 1 <= example_seq_len <= model.config.max_seq_len:
        raise ValueError("example_seq_len must be within the model context length")
    wrapper = LogitsOnlyModel(model).eval()
    sample = torch.zeros((1, example_seq_len), dtype=torch.long)
    return torch.export.export(
        wrapper,
        (sample,),
        dynamic_shapes={"input_ids": {1: _sequence_dim(model)}},
        strict=False,
    )


def capture_web_exported_program(model: OrbituneGPT, *, example_seq_len: int = 64) -> torch.export.ExportedProgram:
    """Capture the Base + external-LoRA-input graph used by the browser runtime."""

    model = model.cpu().eval()
    if not 1 <= example_seq_len <= model.config.max_seq_len:
        raise ValueError("example_seq_len must be within the model context length")
    wrapper = ExternalLoRALogitsModel(model).eval()
    sample = torch.zeros((1, example_seq_len), dtype=torch.long)
    lora_a, lora_b, lora_scale = empty_web_lora(model)
    return torch.export.export(
        wrapper,
        (sample, lora_a, lora_b, lora_scale),
        dynamic_shapes={
            "input_ids": {1: _sequence_dim(model)},
            "lora_a": {},
            "lora_b": {},
            "lora_scale": {},
        },
        strict=False,
    )


def _require_onnx_dependencies() -> None:
    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ONNX export requires: pip install -e '.[export]'") from exc


def export_onnx(base: str | Path, out: str | Path, *, example_seq_len: int = 64) -> Path:
    """Export a Base-only dynamic-sequence ONNX graph."""

    _require_onnx_dependencies()
    model = OrbituneGPT.load_checkpoint(base, map_location="cpu").eval()
    wrapper = LogitsOnlyModel(model).eval()
    sample = torch.zeros((1, example_seq_len), dtype=torch.long)
    program = torch.onnx.export(
        wrapper,
        (sample,),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_shapes={"input_ids": {1: _sequence_dim(model)}},
        dynamo=True,
        external_data=False,
        optimize=True,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(out), external_data=False)
    return out


def export_web_onnx(base: str | Path, out: str | Path, *, example_seq_len: int = 64) -> Path:
    """Export one browser graph that accepts compatible LoRA tensors as inputs.

    The Base weights remain inside one ONNX file. Community adapters remain
    small Safetensors files and are packed into `lora_a`, `lora_b`, and
    `lora_scale` runtime inputs by the browser.
    """

    _require_onnx_dependencies()
    model = OrbituneGPT.load_checkpoint(base, map_location="cpu").eval()
    wrapper = ExternalLoRALogitsModel(model).eval()
    sample = torch.zeros((1, example_seq_len), dtype=torch.long)
    lora_a, lora_b, lora_scale = empty_web_lora(model)
    program = torch.onnx.export(
        wrapper,
        (sample, lora_a, lora_b, lora_scale),
        input_names=["input_ids", "lora_a", "lora_b", "lora_scale"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {1: _sequence_dim(model)},
            "lora_a": {},
            "lora_b": {},
            "lora_scale": {},
        },
        dynamo=True,
        external_data=False,
        optimize=True,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(out), external_data=False)
    return out
