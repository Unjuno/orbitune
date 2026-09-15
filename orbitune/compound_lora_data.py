from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbitune.compound_indexed import load_indexed_compound_corpus
from orbitune.compound_training import load_compound_jsonl


@dataclass(frozen=True, slots=True)
class CompoundLoRADataSource:
    kind: str
    path: Path
    songs: list[Any]
    identity: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indexed_paths(path: Path) -> tuple[Path, Path, Path]:
    index_path = path / "index.json" if path.is_dir() else path
    if index_path.name != "index.json":
        raise ValueError(f"indexed Compound source must be a directory or index.json: {path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records_path = index_path.parent / str(payload.get("records_file", "records.i32"))
    songs_path = index_path.parent / str(payload.get("songs_file", "songs.jsonl"))
    if not records_path.is_file() or not songs_path.is_file():
        raise ValueError(f"indexed Compound source is incomplete: {index_path.parent}")
    return index_path, records_path, songs_path


def load_lora_data_source(path: str | Path) -> CompoundLoRADataSource:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        songs = load_compound_jsonl(source)
        return CompoundLoRADataSource(
            kind="jsonl",
            path=source,
            songs=songs,
            identity={
                "kind": "jsonl",
                "path": str(source),
                "sha256": _sha256_file(source),
                "songs": len(songs),
            },
        )

    index_path, records_path, songs_path = _indexed_paths(source)
    corpus = load_indexed_compound_corpus(index_path)
    metadata = corpus.metadata
    identity_payload = {
        "kind": "indexed",
        "format": metadata.get("format"),
        "schema_version": metadata.get("schema_version"),
        "tokenizer_abi": metadata.get("tokenizer_abi"),
        "record_width": metadata.get("record_width"),
        "split": metadata.get("split"),
        "manifest_sha256": metadata.get("manifest_sha256"),
        "index_sha256": _sha256_file(index_path),
        "records_bytes": records_path.stat().st_size,
        "songs_bytes": songs_path.stat().st_size,
        "songs": int(metadata.get("songs", len(corpus.songs))),
        "events": int(metadata.get("events", 0)),
    }
    digest = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    identity_payload["identity_sha256"] = digest
    return CompoundLoRADataSource(
        kind="indexed",
        path=index_path.parent,
        songs=corpus.songs,
        identity=identity_payload,
    )
