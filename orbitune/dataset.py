from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from orbitune.midi import read_midi
from orbitune.midi_metadata import inspect_midi_metadata, is_4_4_compatible
from orbitune.tokenizer import TheoryRemiTokenizer


def find_midi_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted({*root.rglob("*.mid"), *root.rglob("*.midi")})


def _collect_records(
    files: list[Path],
    *,
    min_events: int,
    require_4_4: bool,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    tokenizer = TheoryRemiTokenizer()
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for midi_path in files:
        try:
            payload = midi_path.read_bytes()
            content_sha256 = hashlib.sha256(payload).hexdigest()
            metadata = inspect_midi_metadata(midi_path)
            if require_4_4 and not is_4_4_compatible(metadata):
                rejected.append({"path": str(midi_path), "reason": f"unsupported_time_signature:{metadata.time_signatures}"})
                continue
            events = read_midi(midi_path)
            if len(events) < min_events:
                rejected.append({"path": str(midi_path), "reason": "too_few_events"})
                continue
            tokens = tokenizer.encode_events(events)
            if not tokens:
                rejected.append({"path": str(midi_path), "reason": "empty_token_sequence"})
                continue
            accepted.append(
                {
                    "path": str(midi_path),
                    "sha256": content_sha256,
                    "events": len(events),
                    "tokens": tokens,
                    "midi_format": metadata.midi_format,
                    "tracks": metadata.track_count,
                    "time_signatures": [list(item) for item in metadata.time_signatures],
                }
            )
        except (ValueError, IndexError, OSError) as exc:
            rejected.append({"path": str(midi_path), "reason": f"{type(exc).__name__}: {exc}"})
    return accepted, rejected


def _merge_records(records: list[dict[str, object]]) -> list[str]:
    merged: list[str] = []
    for index, record in enumerate(records):
        if index:
            merged.extend(["EOS", "BOS"])
        merged.extend(record["tokens"])
    return merged


def _public_record(record: dict[str, object]) -> dict[str, object]:
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "events": record["events"],
        "tokens": len(record["tokens"]),
        "midi_format": record["midi_format"],
        "tracks": record["tracks"],
        "time_signatures": record["time_signatures"],
    }


def prepare_corpus(
    source: str | Path,
    out_tokens: str | Path,
    out_report: str | Path,
    *,
    min_events: int = 1,
    require_4_4: bool = True,
) -> dict[str, object]:
    files = find_midi_files(source)
    if not files:
        raise ValueError(f"no MIDI files found under {source}")
    accepted, rejected = _collect_records(files, min_events=min_events, require_4_4=require_4_4)
    if not accepted:
        raise ValueError("no usable MIDI files found")

    tokenizer = TheoryRemiTokenizer()
    merged = _merge_records(accepted)
    out_tokens = Path(out_tokens)
    out_report = Path(out_report)
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_tokens(merged, out_tokens)

    unique_hashes = {str(item["sha256"]) for item in accepted}
    report: dict[str, object] = {
        "source": str(source),
        "require_4_4": require_4_4,
        "files_seen": len(files),
        "files_accepted": len(accepted),
        "files_rejected": len(rejected),
        "unique_content_files": len(unique_hashes),
        "duplicate_files": len(accepted) - len(unique_hashes),
        "song_boundaries": max(0, len(accepted) - 1),
        "total_events": sum(int(item["events"]) for item in accepted),
        "total_tokens": len(merged),
        "accepted": [_public_record(item) for item in accepted],
        "rejected": rejected,
    }
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def prepare_split_corpus(
    source: str | Path,
    out_train: str | Path,
    out_validation: str | Path,
    out_report: str | Path,
    *,
    validation_fraction: float = 0.1,
    split_seed: str = "orbitune-v0",
    min_events: int = 1,
    require_4_4: bool = True,
) -> dict[str, object]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    files = find_midi_files(source)
    if not files:
        raise ValueError(f"no MIDI files found under {source}")
    accepted, rejected = _collect_records(files, min_events=min_events, require_4_4=require_4_4)
    if len(accepted) < 2:
        raise ValueError("at least two usable MIDI files are required for a train/validation split")

    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in accepted:
        by_hash[str(record["sha256"])].append(record)
    if len(by_hash) < 2:
        raise ValueError("at least two unique MIDI contents are required for a train/validation split")

    def group_key(item: tuple[str, list[dict[str, object]]]) -> str:
        content_hash, _ = item
        return hashlib.sha256(f"{split_seed}\0{content_hash}".encode("utf-8")).hexdigest()

    groups = sorted(by_hash.items(), key=group_key)
    target_validation_files = max(1, round(len(accepted) * validation_fraction))
    validation_groups: list[list[dict[str, object]]] = []
    validation_file_count = 0
    # Keep at least one unique content group in training. Whole duplicate groups
    # move together, preventing exact-copy leakage between train and validation.
    for _, group in groups[:-1]:
        if validation_groups and validation_file_count >= target_validation_files:
            break
        validation_groups.append(group)
        validation_file_count += len(group)

    validation_ids = {str(record["sha256"]) for group in validation_groups for record in group}
    validation_records = [record for record in accepted if str(record["sha256"]) in validation_ids]
    train_records = [record for record in accepted if str(record["sha256"]) not in validation_ids]
    if not validation_records or not train_records:
        raise ValueError("unable to construct non-empty train and validation splits")

    tokenizer = TheoryRemiTokenizer()
    train_tokens = _merge_records(train_records)
    validation_tokens = _merge_records(validation_records)
    out_train = Path(out_train)
    out_validation = Path(out_validation)
    out_report = Path(out_report)
    for path in (out_train, out_validation, out_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_tokens(train_tokens, out_train)
    tokenizer.write_tokens(validation_tokens, out_validation)

    train_hashes = {str(item["sha256"]) for item in train_records}
    validation_hashes = {str(item["sha256"]) for item in validation_records}
    if train_hashes & validation_hashes:
        raise AssertionError("duplicate content leaked across train and validation splits")

    report: dict[str, object] = {
        "source": str(source),
        "require_4_4": require_4_4,
        "split_seed": split_seed,
        "split_unit": "MIDI content SHA-256 group",
        "validation_fraction_requested": validation_fraction,
        "files_seen": len(files),
        "files_accepted": len(accepted),
        "files_rejected": len(rejected),
        "unique_content_files": len(by_hash),
        "duplicate_files": len(accepted) - len(by_hash),
        "train_files": len(train_records),
        "validation_files": len(validation_records),
        "train_unique_contents": len(train_hashes),
        "validation_unique_contents": len(validation_hashes),
        "train_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "train": [_public_record(item) for item in train_records],
        "validation": [_public_record(item) for item in validation_records],
        "rejected": rejected,
    }
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
