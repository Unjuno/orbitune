from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from orbitune.compound_indexed import build_indexed_compound_dataset
from orbitune.pretrain_corpus import (
    collect_entries,
    commercial_safe_sources,
    deduplicate_entries,
    load_registry,
    write_manifest,
)


_MUSESCORE_SUFFIXES = {".mscz", ".mscx", ".musicxml", ".mxl"}


def _musescore_binary(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for candidate in ("MuseScore4", "MuseScore4.exe", "mscore", "musescore"):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the commercial-safe Orbitune Base corpus manifest and memory-mapped Compound datasets."
    )
    parser.add_argument("--config", default="configs/pretrain_corpus_commercial_v1.json")
    parser.add_argument("--root", default="data/corpora/commercial_v1")
    parser.add_argument("--musescore-bin", default=None)
    parser.add_argument("--skip-score-conversion", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--index-root", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    registry = load_registry(args.config)
    sources = commercial_safe_sources(registry)
    converted_base = root / "converted"
    musescore = None if args.skip_score_conversion else _musescore_binary(args.musescore_bin)

    all_entries = []
    source_reports: dict[str, object] = {}
    for source in sources:
        source_root = root / source.id
        if not source_root.exists():
            raise SystemExit(
                f"missing installed source {source.id}: {source_root}. "
                "Run scripts/install_pretrain_corpora.py first."
            )
        conversion: dict[str, object] = {"score_candidates": 0, "converted": 0, "cached": 0, "failed": []}
        converted_root: Path | None = None
        if source.kind == "git_scores" and source.id != "mutopia":
            converted_root = converted_base / source.id
            if not args.skip_score_conversion:
                conversion = convert_scores_to_midi(
                    source_root,
                    converted_root,
                    source.raw,
                    musescore_bin=musescore,
                )
        entries, rejected = collect_entries(source, source_root, converted_root=converted_root)
        if not entries:
            blocked = conversion.get("blocked")
            detail = f" ({blocked})" if blocked else ""
            raise SystemExit(f"source {source.id} produced zero usable MIDI files{detail}")
        all_entries.extend(entries)
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
