from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT


ROWS = [
    (4, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 2, 60, 0, 82, 0, 1, 4, 0, 0),
    (0, 0, 1, 3, 64, 0, 76, 0, 1, 5, 0, 0),
    (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 3),
    (0, 0, 1, 4, 67, 0, 91, 0, 1, 6, 0, 0),
    (5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
]


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


def _indexed(path: Path, split: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(ROWS * 3, dtype="<i4")
    matrix.tofile(path / "records.i32")
    (path / "songs.jsonl").write_text(
        json.dumps({"path": "fixture.mid", "sha256": "fixture", "offset": 0, "length": len(matrix)}) + "\n",
        encoding="utf-8",
    )
    (path / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "orbitune-compound-indexed-v1",
                "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
                "record_width": 12,
                "dtype": "int32-le",
                "split": split,
                "manifest_sha256": "2" * 64,
                "records_file": "records.i32",
                "songs_file": "songs.jsonl",
                "songs": 1,
                "events": len(matrix),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_sweep_runs_directly_on_indexed_train_and_validation_sources(tmp_path: Path) -> None:
    torch.manual_seed(7)
    checkpoint = tmp_path / "base.pt"
    CompoundHierarchicalGPT(_tiny()).save_checkpoint(checkpoint, step=3, source_commit="fixture")
    train = _indexed(tmp_path / "train", "train")
    validation = _indexed(tmp_path / "validation", "validation")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "id": "q-r2",
                        "target_modules": ["decoder.stack.blocks.*.attn.q_proj"],
                        "rank": 2,
                        "alpha": 4,
                        "dropout": 0.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compound_lora_sweep.py",
            "--base-checkpoint", str(checkpoint),
            "--base-id", "fixture-base",
            "--train-source", str(train),
            "--validation-source", str(validation),
            "--spec", str(spec),
            "--output-dir", str(output),
            "--steps", "1",
            "--batch-size", "1",
            "--seq-len", "3",
            "--validation-batches", "1",
            "--seed", "17",
            "--device", "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "sweep.json").read_text(encoding="utf-8"))
    assert summary["common"]["train_source_identity"]["kind"] == "indexed"
    assert summary["common"]["train_source_identity"]["split"] == "train"
    assert summary["common"]["validation_source_identity"]["split"] == "validation"
    training = json.loads((output / "q-r2" / "training.json").read_text(encoding="utf-8"))
    assert training["config"]["train_source_identity"] == summary["common"]["train_source_identity"]
    assert training["config"]["validation_source_identity"] == summary["common"]["validation_source_identity"]
