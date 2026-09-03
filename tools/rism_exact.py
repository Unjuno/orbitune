from __future__ import annotations

import argparse
import atexit
import base64
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import rism_census as census  # noqa: E402

EXPECTED_VEROVIO_VERSION = "6.3.0"
_HEX64 = set("0123456789abcdef")

_WORKER_TOOLKIT: Any = None
_WORKER_TOKENIZER: Any = None
_WORKER_MIDI_PATH: Path | None = None
_WORKER_VEROVIO_VERSION = "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in _HEX64 for ch in text)


def load_baseline_normalized(manifest: Path) -> tuple[set[str], dict[str, object]]:
    fingerprints: set[str] = set()
    rows = 0
    missing = 0
    invalid = 0
    digest = hashlib.sha256()
    with manifest.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            rows += 1
            row = json.loads(raw_line)
            value = row.get("normalized_fingerprint", "")
            if not value:
                missing += 1
                continue
            if not _valid_sha256(value):
                invalid += 1
                continue
            fingerprints.add(str(value).lower())
    if missing or invalid:
        raise ValueError(
            f"baseline manifest is not suitable for exact normalized dedup: "
            f"missing={missing}, invalid={invalid}"
        )
    return fingerprints, {
        "path": str(manifest),
        "rows": rows,
        "unique_normalized_fingerprints": len(fingerprints),
        "sha256": digest.hexdigest(),
    }


def iter_admitted_unique(
    archive: Path,
    *,
    pd_death_cutoff: int,
    counters: Counter[str],
    limit: int | None = None,
) -> Iterator[dict[str, str]]:
    seen_pae: set[str] = set()
    yielded = 0
    with gzip.open(archive, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            if elem.tag != f"{census.MARC}record":
                continue
            counters["source_records"] += 1
            record_id = census._record_id(elem)
            composer, person_dates = census._record_composer(elem)
            source_date = census._record_source_date(elem)
            safe_person = census._person_date_is_pd_safe(person_dates, cutoff=pd_death_cutoff)
            incipits = list(census._iter_incipits(elem))
            if incipits:
                counters["records_with_musical_incipit"] += 1
            for incipit in incipits:
                counters["musical_incipits"] += 1
                if not safe_person:
                    counters["rejected_person_date_policy"] += 1
                    continue
                counters["pd_safe_incipits_pre_dedup"] += 1
                pae_fingerprint = census._incipit_fingerprint(incipit)
                if pae_fingerprint in seen_pae:
                    counters["pae_duplicates"] += 1
                    continue
                seen_pae.add(pae_fingerprint)
                counters["pae_unique_before_context_gate"] += 1
                if not incipit["clef"]:
                    counters["rejected_missing_clef"] += 1
                    continue
                counters["pae_unique"] += 1
                yield {
                    **incipit,
                    "record_id": record_id,
                    "composer": composer,
                    "person_dates": person_dates,
                    "source_date": source_date,
                    "pae_fingerprint": pae_fingerprint,
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    counters["limited_run"] = 1
                    elem.clear()
                    return
            elem.clear()


def _worker_init() -> None:
    global _WORKER_TOOLKIT, _WORKER_TOKENIZER, _WORKER_MIDI_PATH, _WORKER_VEROVIO_VERSION

    import verovio  # type: ignore[import-not-found]
    from orbitune.tokenizer.compound_event import CompoundEventTokenizer

    _WORKER_VEROVIO_VERSION = importlib.metadata.version("verovio")
    if _WORKER_VEROVIO_VERSION != EXPECTED_VEROVIO_VERSION:
        raise RuntimeError(
            f"Verovio version mismatch: expected {EXPECTED_VEROVIO_VERSION}, got {_WORKER_VEROVIO_VERSION}"
        )
    verovio.enableLogToBuffer(True)
    verovio.enableLog(verovio.LOG_WARNING)
    toolkit = verovio.toolkit()
    toolkit.setOptions({"inputFrom": "pae", "breaks": "none"})
    tokenizer = CompoundEventTokenizer()
    tmp = Path(tempfile.mkdtemp(prefix=f"orbitune-rism-{os.getpid()}-"))
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    _WORKER_TOOLKIT = toolkit
    _WORKER_TOKENIZER = tokenizer
    _WORKER_MIDI_PATH = tmp / "incipit.mid"


def _read_verovio_log() -> str:
    if _WORKER_TOOLKIT is None:
        return ""
    try:
        return str(_WORKER_TOOLKIT.getLog() or "")
    except Exception:
        return ""


def _convert_one(item: dict[str, str]) -> dict[str, object]:
    if _WORKER_TOOLKIT is None or _WORKER_TOKENIZER is None or _WORKER_MIDI_PATH is None:
        raise RuntimeError("RISM worker is not initialized")

    from orbitune.compound_midi import read_compound_midi
    from orbitune.pretrain_corpus import midi_fingerprints

    _read_verovio_log()  # clear buffered diagnostics from the preceding item
    try:
        payload = census._pae_payload(item)
        loaded = _WORKER_TOOLKIT.loadData(payload)
        if loaded is False:
            raise RuntimeError("Verovio loadData returned false")
        midi_b64 = _WORKER_TOOLKIT.renderToMIDI()
        midi_bytes = base64.b64decode(midi_b64, validate=True)
        if not midi_bytes:
            raise RuntimeError("empty MIDI")
        _WORKER_MIDI_PATH.write_bytes(midi_bytes)
        raw_sha256, normalized, composition, midi_event_count, midi_tracks = midi_fingerprints(_WORKER_MIDI_PATH)
        events = read_compound_midi(_WORKER_MIDI_PATH)
        if not events:
            raise RuntimeError("empty MIDI event sequence")
        records = _WORKER_TOKENIZER.encode_events(events)
        if len(records) < 2:
            raise RuntimeError("no active Compound next-event pairs")
        log = _read_verovio_log()
        return {
            "ok": True,
            "record_id": item.get("record_id", ""),
            "incipit_no": item.get("incipit_no", ""),
            "composer": item.get("composer", ""),
            "person_dates": item.get("person_dates", ""),
            "source_date": item.get("source_date", ""),
            "pae_fingerprint": item["pae_fingerprint"],
            "normalized_fingerprint": normalized,
            "composition_fingerprint": composition,
            "rendered_midi_sha256": raw_sha256,
            "midi_events": midi_event_count,
            "midi_tracks": midi_tracks,
            "compound_records": len(records),
            "active_events": len(records) - 1,
            "verovio_log": log[:4000],
            "verovio_log_lines": len([line for line in log.splitlines() if line.strip()]),
            "verovio_version": _WORKER_VEROVIO_VERSION,
        }
    except Exception as exc:
        log = _read_verovio_log()
        return {
            "ok": False,
            "record_id": item.get("record_id", ""),
            "pae_fingerprint": item.get("pae_fingerprint", ""),
            "error": f"{type(exc).__name__}: {exc}",
            "verovio_log": log[:4000],
            "verovio_log_lines": len([line for line in log.splitlines() if line.strip()]),
            "verovio_version": _WORKER_VEROVIO_VERSION,
        }


def classify_conversion_result(
    result: dict[str, object],
    *,
    baseline_normalized: set[str],
    seen_rism_normalized: set[str],
    counters: Counter[str],
) -> bool:
    counters["conversion_attempted"] += 1
    if not bool(result.get("ok")):
        counters["conversion_failure"] += 1
        counters["verovio_log_lines"] += int(result.get("verovio_log_lines", 0) or 0)
        if result.get("verovio_log"):
            counters["conversion_failures_with_verovio_log"] += 1
        return False

    counters["conversion_success"] += 1
    log_lines = int(result.get("verovio_log_lines", 0) or 0)
    counters["verovio_log_lines"] += log_lines
    if log_lines:
        counters["conversion_success_with_verovio_log"] += 1

    normalized = str(result["normalized_fingerprint"])
    if normalized in seen_rism_normalized:
        counters["intra_source_duplicates"] += 1
        return False
    seen_rism_normalized.add(normalized)
    counters["normalized_unique"] += 1

    if normalized in baseline_normalized:
        counters["cross_v4_duplicates"] += 1
        return False

    counters["retained_after_cross_dedup"] += 1
    counters["exact_active_events_post_dedup"] += int(result["active_events"])
    counters["exact_compound_records_post_dedup"] += int(result["compound_records"])
    counters["exact_midi_events_post_dedup"] += int(result["midi_events"])
    return True


def _retained_row(result: dict[str, object], *, source_sha1: str) -> dict[str, object]:
    return {
        "source_id": "rism",
        "record_id": result.get("record_id", ""),
        "incipit_no": result.get("incipit_no", ""),
        "composer": result.get("composer", ""),
        "person_dates": result.get("person_dates", ""),
        "source_date": result.get("source_date", ""),
        "license": "cc-by-3.0",
        "rism_export_sha1": source_sha1,
        "pae_fingerprint": result["pae_fingerprint"],
        "normalized_fingerprint": result["normalized_fingerprint"],
        "composition_fingerprint": result["composition_fingerprint"],
        "rendered_midi_sha256": result["rendered_midi_sha256"],
        "midi_events": result["midi_events"],
        "midi_tracks": result["midi_tracks"],
        "compound_records": result["compound_records"],
        "active_events": result["active_events"],
        "verovio_version": result.get("verovio_version", EXPECTED_VEROVIO_VERSION),
        "admission_evidence": {
            "policy": "bounded MARC 100$d latest year <= 1955; MARC 031$p present; clef present",
            "person_dates": result.get("person_dates", ""),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact local RISM conversion and normalized cross-dedup against a completed commercial-v4 baseline manifest."
    )
    parser.add_argument("--archive", default=".rism_census/source-2026-08-01.xml.gz")
    parser.add_argument("--sha1", default=census.DEFAULT_SHA1)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--pd-death-cutoff", type=int, default=census.DEFAULT_PD_DEATH_CUTOFF)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Development-only cap; any limited run is marked non-exact.")
    parser.add_argument("--report", default="rism_exact_report.json")
    parser.add_argument("--entries-output", default="rism_exact_retained.jsonl.gz")
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.exists():
        raise SystemExit(f"missing RISM archive: {archive}")
    actual_sha1 = census._sha1_file(archive)
    if actual_sha1 != args.sha1:
        raise SystemExit(f"RISM archive SHA1 mismatch: expected {args.sha1}, got {actual_sha1}")
    if args.workers < 1 or args.chunksize < 1:
        raise SystemExit("--workers and --chunksize must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 when supplied")

    baseline_normalized, baseline_report = load_baseline_normalized(Path(args.baseline_manifest))
    counters: Counter[str] = Counter()
    seen_rism_normalized: set[str] = set()
    failure_examples: list[dict[str, object]] = []
    warning_examples: list[dict[str, object]] = []
    retained_compositions: set[str] = set()

    candidates = iter_admitted_unique(
        archive,
        pd_death_cutoff=args.pd_death_cutoff,
        counters=counters,
        limit=args.limit,
    )

    output_path = Path(args.entries_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with gzip.open(output_path, "wt", encoding="utf-8", newline="\n") as output_handle:
        with ctx.Pool(processes=args.workers, initializer=_worker_init) as pool:
            for result in pool.imap(_convert_one, candidates, chunksize=args.chunksize):
                retained = classify_conversion_result(
                    result,
                    baseline_normalized=baseline_normalized,
                    seen_rism_normalized=seen_rism_normalized,
                    counters=counters,
                )
                if not bool(result.get("ok")) and len(failure_examples) < 50:
                    failure_examples.append(
                        {
                            "record_id": result.get("record_id", ""),
                            "pae_fingerprint": result.get("pae_fingerprint", ""),
                            "error": result.get("error", ""),
                            "verovio_log": result.get("verovio_log", ""),
                        }
                    )
                elif result.get("verovio_log") and len(warning_examples) < 50:
                    warning_examples.append(
                        {
                            "record_id": result.get("record_id", ""),
                            "pae_fingerprint": result.get("pae_fingerprint", ""),
                            "verovio_log": result.get("verovio_log", ""),
                        }
                    )
                if retained:
                    retained_compositions.add(str(result["composition_fingerprint"]))
                    output_handle.write(
                        json.dumps(_retained_row(result, source_sha1=actual_sha1), separators=(",", ":"), ensure_ascii=False)
                        + "\n"
                    )
                attempted = counters["conversion_attempted"]
                if args.progress_every > 0 and attempted % args.progress_every == 0:
                    print(
                        f"[rism-exact] attempted={attempted:,} success={counters['conversion_success']:,} "
                        f"normalized_unique={counters['normalized_unique']:,} retained={counters['retained_after_cross_dedup']:,} "
                        f"active={counters['exact_active_events_post_dedup']:,}",
                        flush=True,
                    )

    exact = args.limit is None
    report = {
        "source": {
            "name": "RISM source MARCXML export",
            "archive": str(archive),
            "sha1": actual_sha1,
            "license": "CC-BY-3.0",
            "export_date": "2026-08-01",
        },
        "baseline": baseline_report,
        "runtime": {
            "verovio_expected": EXPECTED_VEROVIO_VERSION,
            "workers": args.workers,
            "chunksize": args.chunksize,
        },
        "admission_policy": {
            "pd_death_cutoff": args.pd_death_cutoff,
            "rule": f"MARC 100$d bounded historical dates with latest year <= {args.pd_death_cutoff}; MARC 031$p present; clef present",
            "unknown_or_birth_only_dates": "reject",
            "anonymous_or_unknown_composer_expansion": "not enabled",
        },
        "counts": {
            "RISM_SOURCE_RECORDS": counters["source_records"],
            "RISM_MUSICAL_INCIPITS": counters["musical_incipits"],
            "RISM_PD_SAFE_PRE_DEDUP": counters["pd_safe_incipits_pre_dedup"],
            "RISM_PAE_UNIQUE": counters["pae_unique"],
            "RISM_PAE_DUPLICATES": counters["pae_duplicates"],
            "RISM_REJECT_MISSING_CLEF": counters["rejected_missing_clef"],
            "RISM_CONVERSION_ATTEMPTED": counters["conversion_attempted"],
            "RISM_CONVERSION_SUCCESS": counters["conversion_success"],
            "RISM_CONVERSION_FAILURE": counters["conversion_failure"],
            "RISM_NORMALIZED_UNIQUE": counters["normalized_unique"],
            "RISM_INTRA_SOURCE_DUPLICATES": counters["intra_source_duplicates"],
            "RISM_CROSS_V4_DUPLICATES": counters["cross_v4_duplicates"],
            "RISM_RETAINED_AFTER_CROSS_DEDUP": counters["retained_after_cross_dedup"],
            "RISM_COMPOSITION_UNIQUE_RETAINED": len(retained_compositions),
            "RISM_EXACT_ACTIVE_EVENTS_POST_DEDUP": counters["exact_active_events_post_dedup"],
            "RISM_VEROVIO_LOG_RECORDS": counters["conversion_success_with_verovio_log"]
            + counters["conversion_failures_with_verovio_log"],
            "RISM_VEROVIO_LOG_LINES": counters["verovio_log_lines"],
        },
        "outputs": {
            "retained_entries_jsonl_gz": str(output_path),
            "retained_entries_sha256": _sha256_file(output_path),
        },
        "diagnostics": {
            "failure_examples": failure_examples,
            "verovio_log_examples": warning_examples,
        },
        "is_exact_full_export": exact,
        "limit": args.limit,
        "decision_hint": (
            "ELIGIBLE_FOR_PRODUCTION_SOURCE_IMPLEMENTATION"
            if exact and counters["exact_active_events_post_dedup"] > 15_000_000
            else "DO_NOT_PROMOTE_FROM_THIS_RESULT"
        ),
        "training_started": False,
        "lr_calibration_started": False,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
