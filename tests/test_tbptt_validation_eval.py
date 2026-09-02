from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from tbptt_validation_eval import evaluate_streaming_validation


def _config() -> CompoundBaseConfig:
    cfg = CompoundBaseConfig(
        d_model=32,
        n_head=2,
        local_layers=1,
        medium_layers=1,
        global_layers=1,
        intra_layers=1,
        ff_mult=2,
        dropout=0.0,
        local_window=8,
        medium_stride=2,
        medium_window=4,
        global_stride=2,
        global_window=4,
    )
    cfg.validate()
    return cfg


def _records(count: int) -> list[list[int]]:
    return [
        [0, 0, 0, i % 16, 48 + i % 24, 0, 70, 0, 0, 2, 0, 0]
        for i in range(count)
    ]


def test_streaming_validation_is_chunk_partition_invariant(tmp_path: Path):
    torch.manual_seed(41)
    model = CompoundHierarchicalGPT(_config()).eval()
    checkpoint = tmp_path / "model.pt"
    model.save_checkpoint(checkpoint, step=1900)

    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        json.dumps(
            {
                "path": "tiny.mid",
                "sha256": "tiny",
                "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
                "record_width": 12,
                "records": _records(17),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    by_four = evaluate_streaming_validation(
        checkpoint,
        validation,
        seq_len=4,
        device="cpu",
        precision="fp32",
    )
    by_eight = evaluate_streaming_validation(
        checkpoint,
        validation,
        seq_len=8,
        device="cpu",
        precision="fp32",
    )

    assert by_four["total_events"] == 16
    assert by_eight["total_events"] == 16
    assert abs(by_four["trainer_loss_event_weighted"] - by_eight["trainer_loss_event_weighted"]) < 1e-6
    assert by_four["per_component"]["duration"]["active_events"] == 16
    assert by_four["per_component"]["event_type"]["active_events"] == 16
