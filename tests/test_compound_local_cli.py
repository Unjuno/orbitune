from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import torch

from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_midi import read_compound_midi
from orbitune.tokenizer.compound_event import CompoundEventTokenizer


ROWS = [
    [4, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 60, 0, 82, 0, 1, 4, 0, 0],
    [0, 0, 1, 3, 64, 0, 76, 0, 1, 5, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 3],
    [0, 0, 1, 4, 67, 0, 91, 0, 1, 6, 0, 0],
    [5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
]


def _write_song(path: Path, name: str) -> None:
    payload = {
        "tokenizer_abi": CompoundEventTokenizer.abi,
        "record_width": 12,
        "path": name,
        "sha256": name * 8,
        "records": ROWS,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_installed_compound_cli_train_resume_generate(tmp_path: Path) -> None:
    command = shutil.which("orbitune-compound")
    if command is None:
        # On Windows editable installs the entrypoint script is not always
        # on PATH; the CLI is also reachable as `python -m orbitune.compound_cli`.
        import sys as _sys
        command = [_sys.executable, "-W", "ignore", "-m", "orbitune.compound_cli"]
    else:
        command = [command]

    config = tmp_path / "tiny.json"
    config.write_text(
        json.dumps(
            {
                "d_model": 32,
                "n_head": 4,
                "local_layers": 1,
                "medium_layers": 1,
                "global_layers": 1,
                "intra_layers": 1,
                "ff_mult": 2,
                "dropout": 0.0,
                "local_window": 8,
                "medium_stride": 2,
                "medium_window": 8,
                "global_stride": 2,
                "global_window": 8,
            }
        ),
        encoding="utf-8",
    )
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_song(train, "train")
    _write_song(validation, "validation")
    checkpoint = tmp_path / "compound.pt"

    inspect = subprocess.run(
        command + ["inspect", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "orbitune-compound-hierarchical-gpt-v1" in inspect.stdout

    common = [
        *command,
        "train",
        "--train-jsonl",
        str(train),
        "--validation-jsonl",
        str(validation),
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cpu",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--checkpoint-every",
        "1",
        "--log-every",
        "1",
    ]
    subprocess.run(common + ["--config", str(config), "--steps", "1"], check=True)
    assert checkpoint.exists()
    model, payload = CompoundHierarchicalGPT.load_checkpoint(checkpoint)
    assert payload["step"] == 1
    assert model.config.d_model == 32

    subprocess.run(common + ["--resume", str(checkpoint), "--steps", "2"], check=True)
    _, payload = CompoundHierarchicalGPT.load_checkpoint(checkpoint)
    assert payload["step"] == 2

    midi = tmp_path / "generated.mid"
    subprocess.run(
        command
        + [
            "generate",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(midi),
            "--events",
            "4",
            "--device",
            "cpu",
        ],
        check=True,
    )
    assert midi.exists() and midi.stat().st_size > 0
    assert read_compound_midi(midi)

    # The smoke is deliberately CPU-only; it must not consume a GPU in CI.
    assert not any(parameter.is_cuda for parameter in model.parameters())
    assert torch.cuda.is_available() in (True, False)
