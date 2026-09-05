from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from orbitune.compound_indexed import build_indexed_compound_dataset
from orbitune.pretrain_corpus import (
    _mutopia_license_from_text,
    collect_entries,
    commercial_safe_sources,
    deduplicate_entries,
    load_registry,
    write_manifest,
)


_MUSESCORE_SUFFIXES = {".mscz", ".mscx", ".musicxml", ".mxl"}
_HUMDRUM_SUFFIXES = {".krn"}
_MUTOPIA_DENY_MARKERS = (
    "noncommercial",
    "non-commercial",
    "sharealike",
    "share-alike",
    "cc-by-sa",
    "cc by-sa",
    "by-nc",
    "-nc",
)


def _find_binary(explicit: str | None, candidates: tuple[str, ...]) -> str | None:
    if explicit:
        return explicit
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _score_candidates(root: Path, source_raw: dict[str, object]) -> list[Path]:
    globs = source_raw.get("score_globs", [])
    if not isinstance(globs, list):
        return []
    result: set[Path] = set()
    for pattern in globs:
        for path in root.glob(str(pattern)):
            if path.is_file() and path.suffix.lower() in _MUSESCORE_SUFFIXES:
                result.add(path)
    return sorted(result)


def _humdrum_candidates(root: Path, source_raw: dict[str, object]) -> list[Path]:
    globs = source_raw.get("humdrum_globs", [])
    if not isinstance(globs, list):
        return []
    result: set[Path] = set()
    for pattern in globs:
        for path in root.glob(str(pattern)):
            if path.is_file() and path.suffix.lower() in _HUMDRUM_SUFFIXES:
                result.add(path)
    return sorted(result)


def convert_scores_to_midi(
    source_root: Path,
    converted_root: Path,
    source_raw: dict[str, object],
    *,
    musescore_bin: str | None,
) -> dict[str, object]:
    candidates = _score_candidates(source_root, source_raw)
    if not candidates:
        return {"score_candidates": 0, "converted": 0, "cached": 0, "failed": []}
    if musescore_bin is None:
        return {
            "score_candidates": len(candidates),
            "converted": 0,
            "cached": 0,
            "failed": [],
            "blocked": "MuseScore CLI not found; pass --musescore-bin to include score-only sources",
        }

    converted = 0
    cached = 0
    failed: list[dict[str, str]] = []
    for source_path in candidates:
        relative = source_path.relative_to(source_root)
        output = (converted_root / relative).with_suffix(".mid")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.stat().st_size > 0 and output.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
            cached += 1
            continue
        process = subprocess.run(
            [musescore_bin, "-o", str(output), str(source_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if process.returncode != 0 or not output.exists() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            failed.append({"path": str(source_path), "output": process.stdout[-1000:]})
            continue
        converted += 1
    return {
        "score_candidates": len(candidates),
        "converted": converted,
        "cached": cached,
        "failed": failed,
    }


def convert_humdrum_to_midi(
    source_root: Path,
    converted_root: Path,
    source_raw: dict[str, object],
    *,
    hum2mid_bin: str | None,
) -> dict[str, object]:
    """Convert allowlisted pinned Humdrum **kern files through hum2mid.

    NIFC's own repositories use the same ``hum2mid <input> -o <output>``
    contract for their MIDI build target. Orbitune keeps conversion output in a
    separate derived-data tree so the pinned source checkout remains immutable.
    """
    candidates = _humdrum_candidates(source_root, source_raw)
    if not candidates:
        return {"score_candidates": 0, "converted": 0, "cached": 0, "failed": []}
    if hum2mid_bin is None:
        return {
            "score_candidates": len(candidates),
            "converted": 0,
            "cached": 0,
            "failed": [],
            "blocked": "hum2mid CLI not found; pass --hum2mid-bin to include Humdrum score sources",
        }

    converted = 0
    cached = 0
    failed: list[dict[str, str]] = []
    for source_path in candidates:
        relative = source_path.relative_to(source_root)
        output = (converted_root / relative).with_suffix(".mid")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.stat().st_size > 0 and output.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
            cached += 1
            continue
        process = subprocess.run(
            [hum2mid_bin, str(source_path), "-o", str(output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if process.returncode != 0 or not output.exists() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            failed.append({"path": str(source_path), "output": process.stdout[-1000:]})
            continue
        converted += 1
    return {
        "score_candidates": len(candidates),
        "converted": converted,
        "cached": cached,
        "failed": failed,
    }


def _mutopia_primary_scores(source_root: Path) -> list[tuple[Path, str]]:
    """Select only primary LilyPond scores with an allowlisted local license.

    License inference intentionally does not borrow metadata from sibling score
    files. A directory may contain several works or support files with different
    terms; using a neighboring Public Domain marker to license this score would
    be unsafe. Ambiguous or mixed NC/SA text fails closed.
    """
    result: list[tuple[Path, str]] = []
    for path in source_root.glob("ftp/**/*.ly"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lower = text.lower()
        # Support/include files are numerous. Mutopia's primary score sources
        # carry title metadata and a MIDI score block.
        if "mutopiatitle" not in lower or "\\midi" not in text:
            continue
        if any(marker in lower for marker in _MUTOPIA_DENY_MARKERS):
            continue
        license_id = _mutopia_license_from_text(text)
        if license_id is not None:
            result.append((path, license_id))
    return result


def convert_mutopia_to_midi(
    source_root: Path,
    converted_root: Path,
    *,
    lilypond_bin: str | None,
) -> dict[str, object]:
    candidates = _mutopia_primary_scores(source_root)
    if not candidates:
        return {"score_candidates": 0, "converted": 0, "cached": 0, "failed": []}
    if lilypond_bin is None:
        return {
            "score_candidates": len(candidates),
            "converted": 0,
            "cached": 0,
            "failed": [],
            "blocked": "LilyPond CLI not found; pass --lilypond-bin to include Mutopia source scores",
        }

    converted = 0
    cached = 0
    failed: list[dict[str, str]] = []
    for source_path, license_id in candidates:
        relative = source_path.relative_to(source_root)
        output = (converted_root / relative).with_suffix(".mid")
        output.parent.mkdir(parents=True, exist_ok=True)
        sidecar = output.with_suffix(".ly")
        if output.exists() and output.stat().st_size > 0 and output.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
            if not sidecar.exists():
                sidecar.write_text(f'license = "{license_id}"\n', encoding="utf-8")
            cached += 1
            continue

        prefix = output.with_suffix("")
        for stale in output.parent.glob(prefix.name + "*.midi"):
            stale.unlink(missing_ok=True)
        process = subprocess.run(
            [lilypond_bin, "-dno-print-pages", "-o", str(prefix), str(source_path)],
            cwd=source_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        generated = sorted(output.parent.glob(prefix.name + "*.midi"))
        if process.returncode != 0 or not generated:
            failed.append({"path": str(source_path), "output": process.stdout[-1000:]})
            continue
        # Preserve every MIDI book/output. The first keeps the canonical stem;
        # additional outputs retain LilyPond's generated suffix.
        for index, generated_path in enumerate(generated):
            if index == 0:
                target = output
            else:
                suffix = generated_path.stem[len(prefix.name) :]
                target = output.with_name(prefix.name + suffix + ".mid")
            generated_path.replace(target)
            target.with_suffix(".ly").write_text(f'license = "{license_id}"\n', encoding="utf-8")
            converted += 1
    return {
        "score_candidates": len(candidates),
        "converted": converted,
        "cached": cached,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the commercial-safe Orbitune Base corpus manifest and memory-mapped Compound datasets."
    )
    parser.add_argument("--config", default="configs/pretrain_corpus_commercial_v1.json")
    parser.add_argument("--root", default="data/corpora/commercial_v1")
    parser.add_argument("--musescore-bin", default=None)
    parser.add_argument("--lilypond-bin", default=None)
    parser.add_argument("--hum2mid-bin", default=None)
    parser.add_argument("--skip-score-conversion", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--index-root", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    registry = load_registry(args.config)
    sources = commercial_safe_sources(registry)
    converted_base = root / "converted"
    musescore = None if args.skip_score_conversion else _find_binary(
        args.musescore_bin, ("MuseScore4", "MuseScore4.exe", "mscore", "musescore")
    )
    lilypond = None if args.skip_score_conversion else _find_binary(args.lilypond_bin, ("lilypond", "lilypond.exe"))
    hum2mid = None if args.skip_score_conversion else _find_binary(args.hum2mid_bin, ("hum2mid", "hum2mid.exe"))

    all_entries = []
    source_reports: dict[str, object] = {}
    for source in sources:
        source_root = root / source.id
        if not source_root.exists():
            raise SystemExit(
                f"missing installed source {source.id}: {source_root}. "
                "Run scripts/install_pretrain_corpora.py first."
            )
        t0 = time.time()
        print(f"[build] start source={source.id} kind={source.kind} root={source_root}", flush=True)
        conversion: dict[str, object] = {"score_candidates": 0, "converted": 0, "cached": 0, "failed": []}
        converted_root: Path | None = None
        if source.kind in {"git_scores", "huggingface_score_snapshot"}:
            converted_root = converted_base / source.id
            if not args.skip_score_conversion:
                if source.id == "mutopia":
                    conversion = convert_mutopia_to_midi(
                        source_root,
                        converted_root,
                        lilypond_bin=lilypond,
                    )
                elif source.raw.get("converter") == "hum2mid":
                    conversion = convert_humdrum_to_midi(
                        source_root,
                        converted_root,
                        source.raw,
                        hum2mid_bin=hum2mid,
                    )
                else:
                    conversion = convert_scores_to_midi(
                        source_root,
                        converted_root,
                        source.raw,
                        musescore_bin=musescore,
                    )

        if source.id == "mutopia":
            # Train only on files produced by the fail-closed converter above.
            # Source-tree MIDI files without an exact validated sidecar are not
            # admitted merely because a neighboring LilyPond file is permissive.
            if converted_root is None:
                raise AssertionError("Mutopia converted root was not initialized")
            entries, rejected = collect_entries(source, converted_root, converted_root=converted_root)
        else:
            entries, rejected = collect_entries(source, source_root, converted_root=converted_root)
        if not entries:
            blocked = conversion.get("blocked")
            detail = f" ({blocked})" if blocked else ""
            raise SystemExit(f"source {source.id} produced zero usable MIDI files{detail}")
        all_entries.extend(entries)
        elapsed = time.time() - t0
        print(f"[build] end source={source.id} accepted={len(entries)} events={sum(e.events for e in entries)} rejected={len(rejected)} elapsed={elapsed:.1f}s", flush=True)
        source_reports[source.id] = {
            "accepted_before_cross_dedup": len(entries),
            "events_before_cross_dedup": sum(entry.events for entry in entries),
            "rejected": len(rejected),
            "rejected_examples": rejected[:100],
            "conversion": conversion,
        }

    deduped = deduplicate_entries(all_entries)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.jsonl"
    sampling = registry.get("sampling", {})
    bucket_targets = sampling.get("track_buckets", {}) if isinstance(sampling, dict) else {}
    split_config = registry.get("split")
    if not isinstance(split_config, dict):
        raise SystemExit("registry split configuration is missing")
    manifest_report = write_manifest(
        deduped,
        manifest,
        split_config=split_config,
        track_bucket_targets=bucket_targets if isinstance(bucket_targets, dict) else None,
    )

    index_root = Path(args.index_root) if args.index_root else root / "compound_indexed"
    indexes: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        indexes[split] = build_indexed_compound_dataset(manifest, index_root / split, split=split)

    report = {
        "registry": str(args.config),
        "registry_name": registry.get("name"),
        "musescore_bin": musescore,
        "lilypond_bin": lilypond,
        "hum2mid_bin": hum2mid,
        "sources": source_reports,
        "accepted_before_cross_dedup": len(all_entries),
        "accepted_after_cross_dedup": len(deduped),
        "cross_source_duplicates_removed": len(all_entries) - len(deduped),
        "manifest": str(manifest),
        "manifest_report": manifest_report,
        "indexes": indexes,
    }
    report_path = root / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
