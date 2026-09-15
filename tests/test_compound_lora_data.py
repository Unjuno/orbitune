from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.compound_lora_data import load_lora_data_source
from orbitune.compound_training import sample_compound_batch


ROWS = [
    (4, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 2, 60, 0, 82, 0, 1, 4, 0, 0),
    (0, 0, 1, 3, 64, 0, 76, 0, 1, 5, 0, 0),
    (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 3),
    (0, 0, 1, 4, 67, 0, 91, 0, 1, 6, 0, 0),
    (5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
]


def write_indexed_fixture(path: Path, *, split: str = "train") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(ROWS * 3, dtype="<i4")
    matrix.tofile(path / "records.i32")
    (path / "songs.jsonl").write_text(
        json.dumps(
            {
                "path": "fixture.mid",
                "sha256": "fixture",
                "offset": 0,
                "length": len(matrix),
                "quality_weight": 1.0,
                "sampling_weight": 1.0,
                "tracks": 1,
                "composition_fingerprint": "fixture-composition",
                "source_id": "fixture",
                "license": "CC0-1.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = {
        "schema_version": 1,
        "format": "orbitune-compound-indexed-v1",
        "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
        "record_width": 12,
        "dtype": "int32-le",
        "split": split,
        "manifest": "fixture-manifest.jsonl",
        "manifest_sha256": "1" * 64,
        "records_file": "records.i32",
        "songs_file": "songs.jsonl",
        "songs": 1,
        "events": len(matrix),
    }
    (path / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return path


def test_indexed_lora_source_loads_memmap_without_jsonl_materialization(tmp_path: Path) -> None:
    source_dir = write_indexed_fixture(tmp_path / "train")
    source = load_lora_data_source(source_dir)
    assert source.kind == "indexed"
    assert source.identity["split"] == "train"
    assert source.identity["songs"] == 1
    assert source.identity["events"] == 18
    assert len(str(source.identity["identity_sha256"])) == 64
    assert len(source.songs) == 1
    inputs, targets = sample_compound_batch(
        source.songs,
        batch_size=1,
        seq_len=4,
        rng=random.Random(7),
        device="cpu",
    )
    assert tuple(inputs.shape) == (1, 4, 12)
    assert tuple(targets.shape) == (1, 4, 12)


def test_indexed_lora_source_accepts_explicit_index_path(tmp_path: Path) -> None:
    source_dir = write_indexed_fixture(tmp_path / "validation", split="validation")
    source = load_lora_data_source(source_dir / "index.json")
    assert source.kind == "indexed"
    assert source.path == source_dir
    assert source.identity["split"] == "validation"
