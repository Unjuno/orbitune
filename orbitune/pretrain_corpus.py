from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from orbitune.compound import CompoundEventType
from orbitune.compound_midi import read_compound_midi


CONFIG_DEFAULT = Path("configs/pretrain_corpus_commercial_v1.json")
_MIDI_SUFFIXES = {".mid", ".midi"}
_SCORE_SUFFIXES = {".mscz", ".mscx", ".musicxml", ".mxl", ".ly"}


@dataclass(frozen=True, slots=True)
class CorpusSource:
    id: str
    tier: str
    kind: str
    license: str
    commercial_safe: bool
    raw: dict[str, object]


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    source_id: str
    path: str
    license: str
    tier: str
    quality_weight: float
    raw_sha256: str
    normalized_fingerprint: str
    composition_fingerprint: str
    events: int
    tracks: int
    rating: float | None = None
    quality_flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "license": self.license,
            "tier": self.tier,
            "quality_weight": self.quality_weight,
            "raw_sha256": self.raw_sha256,
            "normalized_fingerprint": self.normalized_fingerprint,
            "composition_fingerprint": self.composition_fingerprint,
            "events": self.events,
            "tracks": self.tracks,
            "rating": self.rating,
            "quality_flags": list(self.quality_flags),
        }


def load_registry(path: str | Path = CONFIG_DEFAULT) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported pretrain corpus registry schema")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("corpus registry requires non-empty sources")
    return payload


def registry_sources(payload: dict[str, object]) -> list[CorpusSource]:
    result: list[CorpusSource] = []
    for item in payload["sources"]:  # type: ignore[index]
        if not isinstance(item, dict):
            raise ValueError("source entry must be an object")
        result.append(
            CorpusSource(
                id=str(item["id"]),
                tier=str(item["tier"]),
                kind=str(item["kind"]),
                license=str(item["license"]),
                commercial_safe=bool(item.get("commercial_safe", False)),
                raw=item,
            )
        )
    return result


def commercial_safe_sources(payload: dict[str, object]) -> list[CorpusSource]:
    sources = registry_sources(payload)
    unsafe = [source.id for source in sources if not source.commercial_safe]
    if unsafe:
        raise ValueError(f"commercial corpus registry contains unsafe sources: {unsafe}")
    return sources


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: object) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def iter_pdmx_midi(root: str | Path) -> Iterator[tuple[Path, dict[str, object]]]:
    """Yield the conservative PDMX Base-pretraining subset.

    Requires the upstream license-conflict flag and PDMX's own best-unique-
    arrangement flag.  The latter preserves distinct instrumentation/arrangement
    variants while removing near-identical exports identified by PDMX.
    """

    root = Path(root)
    csv_path = root / "PDMX.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            no_conflict = _truthy(row.get("subset:no_license_conflict", row.get("no_license_conflict", "")))
            dedup = _truthy(row.get("subset:deduplicated", row.get("is_best_unique_arrangement", "")))
            raw_mid = str(row.get("mid", "")).strip()
            if not no_conflict or not dedup or not raw_mid or raw_mid.lower() in {"n/a", "nan", "none"}:
                continue
            raw_mid = raw_mid.replace("\\", "/")
            if raw_mid.startswith("./"):
                raw_mid = raw_mid[2:]
            midi_path = root / raw_mid
            if midi_path.exists():
                yield midi_path, {
                    "rating": _float_or_none(row.get("rating")),
                    "n_tracks": row.get("n_tracks"),
                    "license": row.get("license") or "public-domain",
                }


def _mutopia_license_from_text(text: str) -> str | None:
    lower = text.lower()
    if "public domain" in lower or "public domain mark" in lower:
        return "public-domain"
    if re.search(r"creative commons attribution(?![- ]sharealike)(?![- ]noncommercial)", lower):
        if "4.0" in lower:
            return "cc-by-4.0"
        if "3.0" in lower:
            return "cc-by-3.0"
    return None


def mutopia_license_for_midi(path: str | Path) -> str | None:
    midi = Path(path)
    candidates = list(midi.parent.glob("*.ly"))
    for candidate in candidates:
        try:
            license_id = _mutopia_license_from_text(candidate.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if license_id:
            return license_id
    return None


def _hash_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def midi_fingerprints(path: str | Path) -> tuple[str, str, str, int, int]:
    """Return raw, normalized and transposition-invariant composition hashes.

    The composition hash intentionally ignores MIDI programs/channels/velocity
    and expresses note pitch relative to the first NOTE.  It is used to keep
    obvious transpositions/exports of one composition in one split. It is not a
    copyright or plagiarism detector.
    """

    path = Path(path)
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    events = read_compound_midi(path)
    if not events:
        raise ValueError("empty MIDI event sequence")
    first_step = events[0].step
    normalized: list[str] = []
    notes = [event for event in events if event.type is CompoundEventType.NOTE]
    first_pitch = notes[0].a1 if notes else 0
    composition: list[str] = []
    channels: set[int] = set()
    for event in events:
        channels.add(event.channel)
        normalized.append(
            f"{int(event.type)}:{event.step-first_step}:{event.channel}:{event.a1}:{event.a2}:{event.a3}:{event.a4}"
        )
        if event.type is CompoundEventType.NOTE:
            composition.append(f"{event.step-first_step}:{event.a1-first_pitch}:{event.a2}")
        elif event.type is CompoundEventType.TIME_SIGNATURE:
            composition.append(f"ts:{event.step-first_step}:{event.a1}:{event.a2}")
    normalized_hash = _hash_parts(normalized)
    composition_hash = _hash_parts(composition or normalized)
    return raw_sha, normalized_hash, composition_hash, len(events), len(channels)


def collect_entries(
    source: CorpusSource,
    root: str | Path,
    *,
    converted_root: str | Path | None = None,
) -> tuple[list[CorpusEntry], list[dict[str, str]]]:
    root = Path(root)
    converted_root = Path(converted_root) if converted_root is not None else None
    candidates: list[tuple[Path, dict[str, object]]] = []
    if source.kind == "zenodo_pdmx":
        candidates.extend(iter_pdmx_midi(root))
    else:
        seen: set[Path] = set()
        for suffix in _MIDI_SUFFIXES:
            for path in root.rglob(f"*{suffix}"):
                if path not in seen:
                    seen.add(path)
                    candidates.append((path, {}))
        if converted_root is not None and converted_root.exists():
            for suffix in _MIDI_SUFFIXES:
                for path in converted_root.rglob(f"*{suffix}"):
                    if path not in seen:
                        seen.add(path)
                        candidates.append((path, {"converted": True}))

    accepted: list[CorpusEntry] = []
    rejected: list[dict[str, str]] = []
    for path, meta in candidates:
        license_id = source.license
        if source.id == "mutopia":
            detected = mutopia_license_for_midi(path)
            if detected is None:
                rejected.append({"path": str(path), "reason": "mutopia_license_not_in_pd_cc0_ccby_allowlist"})
                continue
            license_id = detected
        try:
            raw_sha, norm_hash, comp_hash, event_count, tracks = midi_fingerprints(path)
        except (ValueError, IndexError, OSError) as exc:
            rejected.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        rating = meta.get("rating") if isinstance(meta, dict) else None
        flags: list[str] = []
        weight = float(source.raw.get("quality_weight", 1.0))
        if isinstance(rating, (int, float)):
            flags.append("rated")
            weight *= 1.5
            if rating >= 4.74:
                flags.append("high_rated")
                weight *= 4.0 / 3.0
        if source.tier == "quality-anchor":
            flags.append("openscore_verified")
        accepted.append(
            CorpusEntry(
                source_id=source.id,
                path=str(path),
                license=license_id,
                tier=source.tier,
                quality_weight=weight,
                raw_sha256=raw_sha,
                normalized_fingerprint=norm_hash,
                composition_fingerprint=comp_hash,
                events=event_count,
                tracks=tracks,
                rating=float(rating) if isinstance(rating, (int, float)) else None,
                quality_flags=tuple(flags),
            )
        )
    return accepted, rejected


def deduplicate_entries(entries: Iterable[CorpusEntry]) -> list[CorpusEntry]:
    """Cross-source dedup, preferring quality anchors and then larger files."""

    tier_rank = {"quality-anchor": 0, "primary": 1, "direct-midi-supplement": 2, "supplement": 3}
    by_normalized: dict[str, CorpusEntry] = {}
    for entry in entries:
        previous = by_normalized.get(entry.normalized_fingerprint)
        if previous is None:
            by_normalized[entry.normalized_fingerprint] = entry
            continue
        candidate_key = (tier_rank.get(entry.tier, 99), -entry.quality_weight, -entry.events, entry.path)
        previous_key = (tier_rank.get(previous.tier, 99), -previous.quality_weight, -previous.events, previous.path)
        if candidate_key < previous_key:
            by_normalized[entry.normalized_fingerprint] = entry
    return sorted(by_normalized.values(), key=lambda item: (item.composition_fingerprint, item.source_id, item.path))


def split_for_composition(composition_fingerprint: str, *, seed: str, validation_fraction: float, test_fraction: float) -> str:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid validation/test fractions")
    value = int(hashlib.sha256(f"{seed}\0{composition_fingerprint}".encode("utf-8")).hexdigest()[:16], 16) / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def write_manifest(entries: Iterable[CorpusEntry], path: str | Path, *, split_config: dict[str, object]) -> dict[str, object]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    seed = str(split_config["seed"])
    validation_fraction = float(split_config["validation_fraction"])
    test_fraction = float(split_config["test_fraction"])
    counts = {"train": 0, "validation": 0, "test": 0}
    events = {"train": 0, "validation": 0, "test": 0}
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            split = split_for_composition(
                entry.composition_fingerprint,
                seed=seed,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
            )
            payload = entry.as_dict()
            payload["split"] = split
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            counts[split] += 1
            events[split] += entry.events
    return {"files": counts, "events": events}
