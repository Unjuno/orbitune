from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_lora import sha256_file


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


def _write_dataset(path: Path) -> None:
    rows = [
        (4, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 2, 60, 0, 82, 0, 1, 4, 0, 0),
        (0, 0, 1, 3, 64, 0, 76, 0, 1, 5, 0, 0),
        (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 3),
        (0, 0, 1, 4, 67, 0, 91, 0, 1, 6, 0, 0),
        (5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    ]
    payload = {
        "path": "fixture.mid",
        "sha256": "fixture",
        "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
        "record_width": 12,
        "records": [list(row) for row in rows] * 3,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_sweep_runs_candidates_under_identical_protocol_and_ranks_ml_screen(tmp_path: Path) -> None:
    torch.manual_seed(41)
    model = CompoundHierarchicalGPT(_tiny())
    checkpoint = tmp_path / "base.pt"
    model.save_checkpoint(checkpoint, step=11, source_commit="fixture-source")
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_dataset(train)
    _write_dataset(validation)
    spec = tmp_path / "sweep.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "id": "q-r2",
                        "target_modules": ["decoder.stack.blocks.*.attn.q_proj"],
                        "rank": 2,
                        "alpha": 2,
                    },
                    {
                        "id": "q-r4",
                        "target_modules": ["decoder.stack.blocks.*.attn.q_proj"],
                        "rank": 4,
                        "alpha": 4,
                    },
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
            "--train-jsonl", str(train),
            "--validation-jsonl", str(validation),
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
    assert summary["schema"] == "orbitune-compound-lora-sweep-v1"
    assert summary["status"] == "experimental_ml_screen"
    assert summary["preference_claim"] is False
    assert summary["common"]["base_checkpoint_sha256"] == sha256_file(checkpoint)
    assert summary["common"]["train_jsonl_sha256"] == sha256_file(train)
    assert summary["common"]["validation_jsonl_sha256"] == sha256_file(validation)
    assert {row["id"] for row in summary["candidates"]} == {"q-r2", "q-r4"}
    assert [row["ml_rank"] for row in summary["candidates"]] == [1, 2]
    assert len({row["initial_validation_loss"] for row in summary["candidates"]}) == 1

    by_id = {row["id"]: row for row in summary["candidates"]}
    assert by_id["q-r2"]["trainable_parameters"] < by_id["q-r4"]["trainable_parameters"]
    for candidate_id in ("q-r2", "q-r4"):
        candidate_dir = output / candidate_id
        assert (candidate_dir / "adapter.safetensors").is_file()
        assert (candidate_dir / "metrics.json").is_file()
        assert by_id[candidate_id]["adapter_sha256"] == sha256_file(candidate_dir / "adapter.safetensors")


def test_sweep_rejects_duplicate_candidate_ids_before_training(tmp_path: Path) -> None:
    torch.manual_seed(42)
    checkpoint = tmp_path / "base.pt"
    CompoundHierarchicalGPT(_tiny()).save_checkpoint(checkpoint, step=1)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_dataset(train)
    _write_dataset(validation)
    spec = tmp_path / "bad.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {"id": "same", "target_modules": ["decoder.stack.blocks.*.attn.q_proj"], "rank": 2, "alpha": 2},
                    {"id": "same", "target_modules": ["decoder.stack.blocks.*.attn.v_proj"], "rank": 4, "alpha": 4},
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compound_lora_sweep.py",
            "--base-checkpoint", str(checkpoint),
            "--base-id", "fixture-base",
            "--train-jsonl", str(train),
            "--validation-jsonl", str(validation),
            "--spec", str(spec),
            "--output-dir", str(tmp_path / "out"),
            "--device", "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "duplicate candidate id" in completed.stderr
