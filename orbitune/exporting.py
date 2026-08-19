from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from orbitune.model import OrbituneGPT


class LogitsOnlyModel(nn.Module):
    """Deployment wrapper that removes the training-only optional loss output."""

    def __init__(self, model: OrbituneGPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits


def capture_exported_program(model: OrbituneGPT, *, example_seq_len: int = 64) -> torch.export.ExportedProgram:
    model = model.cpu().eval()
    if not 1 <= example_seq_len <= model.config.max_seq_len:
        raise ValueError("example_seq_len must be within the model context length")
    wrapper = LogitsOnlyModel(model).eval()
    sample = torch.zeros((1, example_seq_len), dtype=torch.long)
    sequence = torch.export.Dim("sequence", min=1, max=model.config.max_seq_len)
    return torch.export.export(
        wrapper,
        (sample,),
        dynamic_shapes={"input_ids": {1: sequence}},
        strict=False,
    )


def export_onnx(base: str | Path, out: str | Path, *, example_seq_len: int = 64) -> Path:
    """Export a base checkpoint as a single-file dynamic-sequence ONNX graph.

    ONNX conversion dependencies are optional because training and adapter
    creation do not require them. Install Orbitune with the `export` extra.
    """

    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ONNX export requires: pip install -e '.[export]'") from exc

    model = OrbituneGPT.load_checkpoint(base, map_location="cpu").eval()
    wrapper = LogitsOnlyModel(model).eval()
    sample = torch.zeros((1, example_seq_len), dtype=torch.long)
    sequence = torch.export.Dim("sequence", min=1, max=model.config.max_seq_len)
    program = torch.onnx.export(
        wrapper,
        (sample,),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_shapes={"input_ids": {1: sequence}},
        dynamo=True,
        external_data=False,
        optimize=True,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(out), external_data=False)
    return out
