from __future__ import annotations

import json
from pathlib import Path

from orbitune.compound import CompoundEvent, CompoundEventType
from orbitune.compound_base import CompoundHierarchicalGPT, write_compound_midi
from orbitune.compound_cli import main
from orbitune.compound_midi import read_compound_midi


def _song(root: Path, name: str, transpose: int) -> Path:
    events = [CompoundEvent(CompoundEventType.TEMPO, 0, 0, 96 + transpose)]
    for index in range(12):
        events.append(
            CompoundEvent(
                CompoundEventType.NOTE,
                index * 24,
                0,
                55 + transpose + (index % 5),
                18 + (index % 3) * 6,
                64 + (index % 8),
            )
        )
    path = root / name
    write_compound_midi(path, events)
    return path


def _tiny_config(path: Path) -> None:
    path.write_text(
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
                "fast_decay": 0.9,
                "medium_decay": 0.97,
                "slow_decay": 0.997,
            }
        ),
        encoding="utf-8",
    )


def test_local_cli_prepare_train_resume_generate(tmp_path: Path, capsys) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    _song(midi_dir, "a.mid", 0)
    _song(midi_dir, "b.mid", 5)

    train = tmp_path / "data" / "train.jsonl"
    validation = tmp_path / "data" / "validation.jsonl"
    report = tmp_path / "data" / "report.json"
    main(
        [
            "prepare",
            str(midi_dir),
            "--train-out",
            str(train),
            "--validation-out",
            str(validation),
            "--report",
            str(report),
            "--validation-fraction",
            "0.5",
            "--min-events",
            "4",
        ]
    )
    assert train.is_file() and validation.is_file() and report.is_file()

    config = tmp_path / "tiny.json"
    _tiny_config(config)
    checkpoint = tmp_path / "compound.pt"
    main(
        [
            "train",
            "--train-jsonl",
            str(train),
            "--validation-jsonl",
            str(validation),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--seq-len",
            "4",
            "--checkpoint-every",
            "1",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert checkpoint.is_file()
    _, first = CompoundHierarchicalGPT.load_checkpoint(checkpoint)
    assert first["step"] == 1

    main(
        [
            "resume",
            "--train-jsonl",
            str(train),
            "--validation-jsonl",
            str(validation),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--seq-len",
            "4",
            "--checkpoint-every",
            "1",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--device",
            "cpu",
        ]
    )
    _, resumed = CompoundHierarchicalGPT.load_checkpoint(checkpoint)
    assert resumed["step"] == 2

    main(["info", "--checkpoint", str(checkpoint)])
    info = capsys.readouterr().out
    assert "orbitune-compound-hierarchical-gpt-v1" in info

    generated = tmp_path / "generated.mid"
    main(
        [
            "generate",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(generated),
            "--events",
            "2",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert generated.read_bytes().startswith(b"MThd")
    assert read_compound_midi(generated)
