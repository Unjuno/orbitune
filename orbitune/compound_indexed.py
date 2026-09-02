from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from orbitune.compound_midi import read_compound_midi
from orbitune.tokenizer.compound_event import COMPOUND_RECORD_WIDTH, CompoundEventTokenizer


INDEX_FORMAT = "orbitune-compound-indexed-v1"
INDEX_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IndexedRecords(Sequence[Sequence[int]]):
    """Song-local view into one shared read-only Compound record memmap."""

    __slots__ = ("_records", "offset", "length")

    def __init__(self, records: np.memmap, offset: int, length: int) -> None:
        self._records = records
        self.offset = int(offset)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item):  # type: ignore[no-untyped-def]
        if isinstance(item, slice):
            start, stop, step = item.indices(self.length)
            return self._records[self.offset + start : self.offset + stop : step]
        index = int(item)
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self._records[self.offset + index]


@dataclass(frozen=True, slots=True)
class IndexedCompoundSong:
    path: str
    sha256: str
    tokenizer_abi: str
    records: IndexedRecords
    quality_weight: float = 1.0
    sampling_weight: float = 1.0
    tracks: int = 0
    composition_fingerprint: str = ""
    source_id: str = ""
    license: str = ""


@dataclass(slots=True)
class IndexedCompoundCorpus:
    index_path: Path
    records: np.memmap
    songs: list[IndexedCompoundSong]
    metadata: dict[str, object]


def _read_manifest_rows(path: Path, split: str) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("split") == split:
                if not payload.get("path"):
                    raise ValueError(f"{path}:{line_number}: missing MIDI path")
                yield payload


def build_indexed_compound_dataset(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    split: str,
) -> dict[str, object]:
    """Encode one manifest split into a flat int32 record store plus song index.

    This format keeps hundreds of millions of Compound records out of Python
    object memory. Training processes memory-map the one flat array and only
    copy the sampled song windows needed for the current optimizer step.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.i32"
    songs_path = out_dir / "songs.jsonl"
    index_path = out_dir / "index.json"
    tokenizer = CompoundEventTokenizer()
    manifest_sha256 = _sha256_file(manifest_path)

    song_count = 0
    event_count = 0
    source_counts: dict[str, int] = {}
    license_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    tmp_records = records_path.with_suffix(records_path.suffix + ".tmp")
    tmp_songs = songs_path.with_suffix(songs_path.suffix + ".tmp")

    try:
        with tmp_records.open("wb") as record_handle, tmp_songs.open("w", encoding="utf-8") as song_handle:
            for row in _read_manifest_rows(manifest_path, split):
                midi_path = Path(str(row["path"]))
                events = read_compound_midi(midi_path)
                records = tokenizer.encode_events(events)
                if not records:
                    raise ValueError(f"{midi_path}: no Compound records")
                matrix = np.asarray([record.as_tuple() for record in records], dtype="<i4")
                if matrix.ndim != 2 or matrix.shape[1] != COMPOUND_RECORD_WIDTH:
                    raise AssertionError("Compound tokenizer emitted invalid record matrix")
                offset = event_count
                matrix.tofile(record_handle)
                length = int(matrix.shape[0])
                payload = {
                    "path": str(midi_path),
                    "sha256": str(row.get("raw_sha256", row.get("sha256", ""))),
                    "composition_fingerprint": str(row.get("composition_fingerprint", "")),
                    "source_id": str(row.get("source_id", "")),
                    "license": str(row.get("license", "")),
                    "quality_weight": float(row.get("quality_weight", 1.0)),
                    "sampling_weight": float(row.get("sampling_weight", row.get("quality_weight", 1.0))),
                    "tracks": int(row.get("tracks", 0)),
                    "track_bucket": str(row.get("track_bucket", "")),
                    "offset": offset,
                    "length": length,
                }
                song_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                song_count += 1
                event_count += length
                source = payload["source_id"]
                license_id = payload["license"]
                bucket = payload["track_bucket"]
                source_counts[source] = source_counts.get(source, 0) + 1
                license_counts[license_id] = license_counts.get(license_id, 0) + 1
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if song_count == 0:
            raise ValueError(f"manifest contains no {split} songs")
        tmp_records.replace(records_path)
        tmp_songs.replace(songs_path)
    except Exception:
        tmp_records.unlink(missing_ok=True)
        tmp_songs.unlink(missing_ok=True)
        raise

    metadata: dict[str, object] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "format": INDEX_FORMAT,
        "tokenizer_abi": tokenizer.abi,
        "record_width": COMPOUND_RECORD_WIDTH,
        "dtype": "int32-le",
        "split": split,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "records_file": records_path.name,
        "songs_file": songs_path.name,
        "songs": song_count,
        "events": event_count,
        "source_counts": source_counts,
        "license_counts": license_counts,
        "track_bucket_counts": bucket_counts,
    }
    tmp_index = index_path.with_suffix(".json.tmp")
    tmp_index.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    tmp_index.replace(index_path)
    return metadata


def load_indexed_compound_corpus(index_path: str | Path) -> IndexedCompoundCorpus:
    index_path = Path(index_path)
    metadata = json.loads(index_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", 0)) != INDEX_SCHEMA_VERSION or metadata.get("format") != INDEX_FORMAT:
        raise ValueError(f"unsupported indexed Compound corpus: {index_path}")
    if int(metadata.get("record_width", -1)) != COMPOUND_RECORD_WIDTH:
        raise ValueError("indexed corpus record width does not match current Compound ABI")
    if str(metadata.get("tokenizer_abi", "")) != CompoundEventTokenizer.abi:
        raise ValueError("indexed corpus tokenizer ABI does not match current Compound tokenizer")
    manifest_sha256 = str(metadata.get("manifest_sha256", ""))
    if len(manifest_sha256) != 64:
        raise ValueError("indexed corpus is missing a valid manifest_sha256 identity")
    event_count = int(metadata["events"])
    records_path = index_path.parent / str(metadata["records_file"])
    songs_path = index_path.parent / str(metadata["songs_file"])
    records = np.memmap(records_path, dtype="<i4", mode="r", shape=(event_count, COMPOUND_RECORD_WIDTH))
    songs: list[IndexedCompoundSong] = []
    with songs_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            offset = int(row["offset"])
            length = int(row["length"])
            if offset < 0 or length <= 0 or offset + length > event_count:
                raise ValueError(f"{songs_path}:{line_number}: invalid indexed song range")
            songs.append(
                IndexedCompoundSong(
                    path=str(row.get("path", "")),
                    sha256=str(row.get("sha256", "")),
                    tokenizer_abi=CompoundEventTokenizer.abi,
                    records=IndexedRecords(records, offset, length),
                    quality_weight=float(row.get("quality_weight", 1.0)),
                    sampling_weight=float(row.get("sampling_weight", row.get("quality_weight", 1.0))),
                    tracks=int(row.get("tracks", 0)),
                    composition_fingerprint=str(row.get("composition_fingerprint", "")),
                    source_id=str(row.get("source_id", "")),
                    license=str(row.get("license", "")),
                )
            )
    if len(songs) != int(metadata.get("songs", -1)):
        raise ValueError("indexed corpus song count does not match index metadata")
    return IndexedCompoundCorpus(index_path=index_path, records=records, songs=songs, metadata=metadata)
