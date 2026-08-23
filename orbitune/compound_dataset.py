from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from orbitune.compound_midi import read_compound_midi
from orbitune.dataset import find_midi_files
from orbitune.midi_metadata import inspect_midi_metadata
from orbitune.tokenizer import CompoundEventTokenizer


COMPOUND_RECORD_WIDTH = 12


def _collect(source: str | Path, *, min_events: int) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    tokenizer = CompoundEventTokenizer()
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for path in find_midi_files(source):
        try:
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            events = read_compound_midi(path)
            if len(events) < min_events:
                rejected.append({"path": str(path), "reason": "too_few_events"})
                continue
            records = tokenizer.encode_events(events)
            metadata = inspect_midi_metadata(path)
            accepted.append(
                {
                    "tokenizer_abi": tokenizer.abi,
                    "record_width": COMPOUND_RECORD_WIDTH,
                    "path": str(path),
                    "sha256": digest,
                    "events": len(events),
                    "records": [list(record.as_tuple()) for record in records],
                    "midi_format": metadata.midi_format,
                    "tracks": metadata.track_count,
                    "time_signatures": [list(item) for item in metadata.time_signatures],
                }
            )
        except (ValueError, IndexError, OSError) as exc:
            rejected.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    return accepted, rejected


def _write_jsonl(records: list[dict[str, object]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def prepare_compound_split_corpus(
    source: str | Path,
    out_train: str | Path,
    out_validation: str | Path,
    out_report: str | Path,
    *,
    validation_fraction: float = 0.1,
    split_seed: str = "orbitune-compound-v0-experimental",
    min_events: int = 1,
) -> dict[str, object]:
    """Prepare song-preserving Compound Event train/validation JSONL files.

    Exact MIDI byte duplicates are grouped by SHA-256 and cannot cross the
    split. Each JSONL row remains one source composition/file and carries the
    exact experimental tokenizer ABI plus record width, preventing silent
    mixing of incompatible record layouts.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    files = find_midi_files(source)
    if not files:
        raise ValueError(f"no MIDI files found under {source}")
    accepted, rejected = _collect(source, min_events=min_events)
    if len(accepted) < 2:
        raise ValueError("at least two usable MIDI files are required")

    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in accepted:
        by_hash[str(record["sha256"])].append(record)
    if len(by_hash) < 2:
        raise ValueError("at least two unique MIDI contents are required")

    def group_key(item: tuple[str, list[dict[str, object]]]) -> str:
        digest, _ = item
        return hashlib.sha256(f"{split_seed}\0{digest}".encode()).hexdigest()

    groups = sorted(by_hash.items(), key=group_key)
    target = max(1, round(len(accepted) * validation_fraction))
    validation_hashes: set[str] = set()
    count = 0
    for digest, group in groups[:-1]:
        if validation_hashes and count >= target:
            break
        validation_hashes.add(digest)
        count += len(group)

    validation = [record for record in accepted if str(record["sha256"]) in validation_hashes]
    train = [record for record in accepted if str(record["sha256"]) not in validation_hashes]
    if not train or not validation:
        raise ValueError("unable to construct non-empty split")

    _write_jsonl(train, out_train)
    _write_jsonl(validation, out_validation)

    train_hashes = {str(record["sha256"]) for record in train}
    val_hashes = {str(record["sha256"]) for record in validation}
    if train_hashes & val_hashes:
        raise AssertionError("duplicate content leaked across splits")

    report = {
        "representation": "hybrid-compound",
        "tokenizer_abi": CompoundEventTokenizer.abi,
        "source": str(source),
        "split_seed": split_seed,
        "split_unit": "MIDI content SHA-256 group",
        "files_seen": len(files),
        "files_accepted": len(accepted),
        "files_rejected": len(rejected),
        "train_files": len(train),
        "validation_files": len(validation),
        "train_events": sum(int(record["events"]) for record in train),
        "validation_events": sum(int(record["events"]) for record in validation),
        "record_width": COMPOUND_RECORD_WIDTH,
        "rejected": rejected,
    }
    target_report = Path(out_report)
    target_report.parent.mkdir(parents=True, exist_ok=True)
    target_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
