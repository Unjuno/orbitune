"""Source loading, exposure budgets and resume guards for state-carry training.

No model/state-transition math lives here. Source identities bind ordered
metadata and song tables, not a new full-byte audit of existing record stores.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbitune.compound_indexed import (
    load_indexed_compound_corpus,
    load_sharded_indexed_corpus,
)
from orbitune.compound_training import load_compound_jsonl
from orbitune.tokenizer.compound_event import COMPOUND_RECORD_WIDTH, CompoundEventTokenizer


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class TrainingSources:
    songs: list[Any]
    corpora: list[Any]
    indexed: bool
    identity: str
    descriptors: list[dict[str, Any]]

    @property
    def source_format(self) -> str:
        return "indexed_memmap" if self.indexed else "jsonl_memory"


def _source_paths(value: str | Path) -> list[Path]:
    parts = str(value).split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError("source list contains an empty path")
    return [Path(part.strip()).expanduser() for part in parts]


def _resolve_source(path: Path) -> tuple[Path, str]:
    if path.is_dir():
        for filename, kind in (("index_manifest.json", "sharded"), ("index.json", "indexed")):
            candidate = path / filename
            if candidate.is_file():
                return candidate.resolve(), kind
        raise ValueError(f"no index.json or index_manifest.json in {path}")
    if path.name == "songs.jsonl" and (path.parent / "index.json").is_file():
        return (path.parent / "index.json").resolve(), "indexed"
    if not path.is_file():
        raise FileNotFoundError(path)
    kind = {"index_manifest.json": "sharded", "index.json": "indexed"}.get(path.name, "jsonl")
    return path.resolve(), kind


def _indexed_descriptor(corpus: Any) -> dict[str, Any]:
    """Check metadata/ranges without copying or rescanning every record."""
    meta = corpus.metadata
    if meta.get("tokenizer_abi", CompoundEventTokenizer.abi) != CompoundEventTokenizer.abi:
        raise ValueError("TBPTT indexed tokenizer ABI mismatch")
    if int(meta.get("record_width", COMPOUND_RECORD_WIDTH)) != COMPOUND_RECORD_WIDTH:
        raise ValueError("TBPTT indexed record width mismatch")
    count = int(meta["events"])
    records_path = Path(corpus.records.filename)
    expected_bytes = count * COMPOUND_RECORD_WIDTH * corpus.records.dtype.itemsize
    if records_path.stat().st_size != expected_bytes:
        raise ValueError(f"indexed record byte size mismatch: {records_path}")
    if len(corpus.songs) != int(meta["songs"]):
        raise ValueError("TBPTT indexed song count mismatch")
    next_offset = 0
    for song in corpus.songs:
        if song.tokenizer_abi != CompoundEventTokenizer.abi:
            raise ValueError("TBPTT song tokenizer ABI mismatch")
        if song.records.offset != next_offset or len(song.records) <= 0:
            raise ValueError("TBPTT indexed song ranges must be contiguous and nonempty")
        next_offset += len(song.records)
        if next_offset > count:
            raise ValueError("TBPTT indexed song range exceeds record store")
        if not all(math.isfinite(float(w)) and float(w) >= 0 for w in
                   (song.quality_weight, song.sampling_weight)):
            raise ValueError("TBPTT indexed weights must be finite and nonnegative")
    if next_offset != count:
        raise ValueError("TBPTT indexed records are not fully accounted for by song ranges")
    songs_path = corpus.index_path.parent / str(meta["songs_file"])
    return {
        "index_sha256": _file_sha256(corpus.index_path),
        "songs_sha256": _file_sha256(songs_path),
        "record_bytes": expected_bytes,
        "record_dtype": corpus.records.dtype.str,
        "songs": len(corpus.songs),
        "records": count,
    }


def load_training_sources(value: str | Path) -> TrainingSources:
    """Load one or comma-separated flat/sharded indexes, in caller order.

    Existing Aria int32 and GigaMIDI uint8 stores stay read-only memmaps. No
    concatenation, MIDI parsing or narrowing cast is performed. A mix of
    indexed data and in-memory record JSONL is deliberately rejected rather
    than dispatching the wrong sampler. Multiple homogeneous JSONLs work.
    """
    resolved = [_resolve_source(path) for path in _source_paths(value)]
    paths = [path for path, _ in resolved]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate training source path")
    kinds = {kind != "jsonl" for _, kind in resolved}
    if len(kinds) != 1:
        raise ValueError("cannot mix indexed sources and in-memory record JSONL")
    indexed = kinds.pop()
    songs: list[Any] = []
    corpora: list[Any] = []
    descriptors: list[dict[str, Any]] = []
    seen_stores: set[Path] = set()
    for path, kind in resolved:
        if kind == "jsonl":
            loaded = load_compound_jsonl(path)
            songs.extend(loaded)
            descriptors.append({"kind": kind, "source_sha256": _file_sha256(path), "songs": len(loaded)})
            continue
        corpus = (load_sharded_indexed_corpus(path) if kind == "sharded"
                  else load_indexed_compound_corpus(path))
        shards = corpus.shards if kind == "sharded" else [corpus]
        if not shards:
            raise ValueError("empty sharded training source")
        shard_descriptors = []
        for shard in shards:
            store = Path(shard.records.filename).resolve()
            if store in seen_stores:
                raise ValueError("duplicate indexed record store across training sources")
            seen_stores.add(store)
            shard_descriptors.append(_indexed_descriptor(shard))
        corpora.append(corpus)  # retain ownership of every read-only memmap
        songs.extend(corpus.songs)
        descriptors.append({"kind": kind, "source_sha256": _file_sha256(path), "shards": shard_descriptors})
    if not songs:
        raise ValueError("training source contains no songs")
    return TrainingSources(songs, corpora, indexed,
                           _identity({"schema": "tbptt-sources-v1", "sources": descriptors}), descriptors)


def validation_identity(sources: TrainingSources, *, seq_len: int, max_songs: int) -> str:
    return _identity({"schema": "tbptt-stream-validation-v1", "sources": sources.identity,
                      "seq_len": seq_len, "max_songs": max_songs,
                      "selection": "ordered-prefix", "tails": "drop-incomplete-chunk"})


def target_final_step(*, start_step: int, start_events: int, batch_size: int,
                      seq_len: int, steps: int | None, target_events: int | None) -> int:
    """Convert a cumulative sampled-position target to full optimizer steps.

    Round UP to the next complete batch; no partial-batch trajectory change.
    The overshoot is less than batch_size * seq_len event positions. Python
    integers avoid 32-bit overflow for the 4,096,000,000-position target.
    """
    if start_step < 0 or start_events < 0 or batch_size <= 0 or seq_len <= 0:
        raise ValueError("invalid start counters or batch geometry")
    if (steps is None) == (target_events is None):
        raise ValueError("provide exactly one of --steps or --target-events")
    if steps is not None:
        if steps <= 0:
            raise ValueError("--steps must be positive")
        return max(start_step, steps)
    if target_events is None or target_events <= 0:
        raise ValueError("--target-events must be positive")
    remaining = max(0, target_events - start_events)
    per_step = batch_size * seq_len
    return start_step + (remaining + per_step - 1) // per_step


def paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and os.path.samefile(left, right)


def guard_transition_outputs(resume: str | Path, checkpoint: str | Path) -> None:
    """A fixed-window -> TBPTT transition cannot overwrite any parent alias."""
    parent, out = Path(resume), Path(checkpoint)
    for target in (out, out.with_name(out.stem + ".best.pt"), out.with_name(out.stem + ".healthy.pt")):
        if paths_alias(parent, target):
            raise ValueError("TBPTT transition output aliases the fixed-window parent checkpoint")


def validate_resume_contract(payload: dict[str, Any], runtime: dict[str, Any]) -> None:
    """Same-regime resume is fail-closed; no silent partial state reset."""
    required = ("optimizer_state_dict", "torch_rng_state", "python_rng_state", "sampler_rng_state",
                "tbptt_sampler_state", "tbptt_stream_states")
    for name in required:
        if payload.get(name) is None:
            raise ValueError(f"TBPTT resume missing {name}")
    if not isinstance(payload["tbptt_sampler_state"], dict):
        raise ValueError("invalid TBPTT sampler state")
    states = payload["tbptt_stream_states"]
    if not isinstance(states, list) or len(states) != runtime["batch_size"]:
        raise ValueError("TBPTT resume stream-state lane count mismatch")
    previous = payload.get("runtime") or {}
    keys = ("training_mode", "device_type", "batch_size", "seq_len", "precision", "n_head", "causal_fastpath",
            "training_source_format", "weighted_corpus_sampling", "training_corpus_identity",
            "validation_corpus_identity", "validation_protocol_identity", "grad_clip", "fused_adamw")
    for name in keys:
        if name not in previous:
            raise ValueError(f"TBPTT resume missing runtime identity {name}; legacy resume needs explicit migration")
        if previous[name] != runtime[name]:
            raise ValueError(f"TBPTT resume runtime mismatch for {name}")
    if runtime["precision"] == "fp16" and payload.get("amp_scaler_state_dict") is None:
        raise ValueError("TBPTT fp16 resume missing GradScaler state")
    if runtime.get("device_type", "cuda") == "cuda" and payload.get("cuda_rng_state_all") is None:
        raise ValueError("TBPTT CUDA resume missing CUDA RNG state")
    lane_state = payload["tbptt_sampler_state"]
    offsets, indices = lane_state.get("offsets"), lane_state.get("song_indices")
    if not isinstance(offsets, list) or not isinstance(indices, list) or not (len(offsets) == len(indices) == len(states)):
        raise ValueError("TBPTT resume sampler lane count mismatch")
    for state, offset, index in zip(states, offsets, indices):
        required_state = ("local_records", "medium_buffer", "medium_history", "global_buffer", "global_history", "memory", "steps")
        if not isinstance(state, dict) or any(name not in state for name in required_state):
            raise ValueError("TBPTT resume incomplete stream state")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0 or offset % runtime["seq_len"]:
            raise ValueError("TBPTT resume invalid chunk offset")
        if int(state["steps"]) != offset:
            raise ValueError("TBPTT resume stream steps do not match sampler offset")
        if index == -1 and offset != 0:
            raise ValueError("TBPTT uninitialized lane has a nonzero offset")
        if offset > 0 and (state["memory"] is None or not state["local_records"]):
            raise ValueError("TBPTT active lane is missing its carried context")
