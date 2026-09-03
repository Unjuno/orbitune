from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import random
import re
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

MARC_NS = "http://www.loc.gov/MARC21/slim"
MARC = f"{{{MARC_NS}}}"
DEFAULT_URL = "https://rism.digital/exports/archive/source-2026-08-01.xml.gz"
DEFAULT_SHA1 = "69261a6a6d30fa28139147287b6fcc060fd78edc"
DEFAULT_PD_DEATH_CUTOFF = 1955
USER_AGENT = "Orbitune-RISM-census/1.0 (+https://github.com/Unjuno/orbitune)"
_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")
_OPEN_ENDED_BIRTH_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\s*[-–—]\s*$")
_WS_RE = re.compile(r"\s+")


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 - upstream integrity checksum, not security use
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as response, partial.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(target)


def _normalize_text(value: str | None) -> str:
    return _WS_RE.sub(" ", (value or "").strip())


def _subfields(field: ET.Element) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for child in field.findall(f"{MARC}subfield"):
        code = str(child.attrib.get("code", ""))
        text = _normalize_text(child.text)
        if code and text:
            result.setdefault(code, []).append(text)
    return result


def _first(values: dict[str, list[str]], code: str, default: str = "") -> str:
    items = values.get(code, [])
    return items[0] if items else default


def _person_date_is_pd_safe(value: str, *, cutoff: int = DEFAULT_PD_DEATH_CUTOFF) -> bool:
    """Conservative evidence test for a historical composer/person date string.

    A concrete death year or a fully bounded historical range ending on/before
    the cutoff is accepted. Birth-only open-ended forms such as ``1900-`` are
    rejected. Unknown or missing dates are rejected. ``fl. 1732-1735`` is
    accepted because the person's documented activity is wholly historical.
    """

    text = _normalize_text(value)
    if not text or _OPEN_ENDED_BIRTH_RE.search(text):
        return False
    years = [int(item) for item in _YEAR_RE.findall(text)]
    if not years:
        return False
    return max(years) <= cutoff


def _record_source_date(record: ET.Element) -> str:
    for tag in ("260", "264"):
        for field in record.findall(f"{MARC}datafield[@tag='{tag}']"):
            values = _subfields(field)
            if values.get("c"):
                return _first(values, "c")
    return ""


def _record_composer(record: ET.Element) -> tuple[str, str]:
    field = record.find(f"{MARC}datafield[@tag='100']")
    if field is None:
        return "", ""
    values = _subfields(field)
    return _first(values, "a"), _first(values, "d")


def _record_id(record: ET.Element) -> str:
    field = record.find(f"{MARC}controlfield[@tag='001']")
    return _normalize_text(field.text if field is not None else "")


def _iter_incipits(record: ET.Element) -> Iterable[dict[str, str]]:
    for field in record.findall(f"{MARC}datafield[@tag='031']"):
        values = _subfields(field)
        pae = _first(values, "p")
        if not pae:
            continue
        yield {
            "clef": _first(values, "g"),
            "keysig": _first(values, "n"),
            "timesig": _first(values, "o"),
            "pae": pae,
            "instrument": _first(values, "m"),
            "incipit_no": ".".join(
                part for part in (_first(values, "a"), _first(values, "b"), _first(values, "c")) if part
            ),
        }


def _incipit_fingerprint(incipit: dict[str, str]) -> str:
    payload = "\0".join(
        _normalize_text(incipit.get(key, "")) for key in ("clef", "keysig", "timesig", "pae")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reservoir_add(
    reservoir: list[dict[str, str]],
    item: dict[str, str],
    *,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    index = rng.randrange(seen)
    if index < limit:
        reservoir[index] = item


def scan_export(
    archive: Path,
    *,
    sample_size: int,
    pd_death_cutoff: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    counters: Counter[str] = Counter()
    unique_safe: set[str] = set()
    samples: list[dict[str, str]] = []
    rng = random.Random(seed)
    composer_examples: Counter[str] = Counter()
    source_date_examples: Counter[str] = Counter()

    with gzip.open(archive, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            if elem.tag != f"{MARC}record":
                continue
            counters["source_records"] += 1
            record_id = _record_id(elem)
            composer, person_dates = _record_composer(elem)
            source_date = _record_source_date(elem)
            safe_person = _person_date_is_pd_safe(person_dates, cutoff=pd_death_cutoff)
            incipits = list(_iter_incipits(elem))
            if incipits:
                counters["records_with_musical_incipit"] += 1
                composer_examples[composer or "<missing>"] += 1
                if source_date:
                    source_date_examples[source_date] += 1
            for incipit in incipits:
                counters["musical_incipits"] += 1
                if incipit["clef"]:
                    counters["incipits_with_clef"] += 1
                if incipit["timesig"]:
                    counters["incipits_with_timesig"] += 1
                if person_dates:
                    counters["incipits_with_person_dates"] += 1
                if not safe_person:
                    continue
                counters["pd_safe_incipits_pre_dedup"] += 1
                fingerprint = _incipit_fingerprint(incipit)
                if fingerprint in unique_safe:
                    counters["pd_safe_duplicate_incipits"] += 1
                    continue
                unique_safe.add(fingerprint)
                counters["pd_safe_unique_incipits"] += 1
                if not incipit["clef"]:
                    counters["pd_safe_unique_missing_clef"] += 1
                    continue
                candidate = {
                    **incipit,
                    "record_id": record_id,
                    "composer": composer,
                    "person_dates": person_dates,
                    "source_date": source_date,
                    "fingerprint": fingerprint,
                }
                _reservoir_add(
                    samples,
                    candidate,
                    seen=counters["pd_safe_unique_incipits"],
                    limit=sample_size,
                    rng=rng,
                )
            elem.clear()

    result: dict[str, Any] = {
        "archive": str(archive),
        "pd_death_cutoff": pd_death_cutoff,
        **dict(counters),
        "pd_safe_unique_fraction_of_all": (
            counters["pd_safe_unique_incipits"] / counters["musical_incipits"]
            if counters["musical_incipits"]
            else 0.0
        ),
        "top_composers_among_records_with_incipits": composer_examples.most_common(20),
        "source_date_examples": source_date_examples.most_common(20),
        "sample_size_selected": len(samples),
    }
    return result, samples


def _pae_payload(item: dict[str, str]) -> str:
    # Verovio accepts this compact JSON representation of PAE input.
    return json.dumps(
        {
            "clef": item.get("clef", ""),
            "keysig": item.get("keysig", ""),
            "timesig": item.get("timesig", ""),
            "data": item["pae"],
        },
        ensure_ascii=False,
    )


def parse_sample(samples: list[dict[str, str]]) -> dict[str, Any]:
    if not samples:
        return {
            "attempted": 0,
            "parsed": 0,
            "failures": 0,
            "mean_active_events": 0.0,
            "median_active_events": 0.0,
        }

    import verovio  # type: ignore[import-not-found]

    from orbitune.compound_midi import read_compound_midi
    from orbitune.tokenizer.compound_event import CompoundEventTokenizer

    toolkit = verovio.toolkit()
    toolkit.setOptions({"inputFrom": "pae", "breaks": "none", "logLevel": "off"})
    tokenizer = CompoundEventTokenizer()
    active_counts: list[int] = []
    failure_examples: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="orbitune-rism-") as tmp:
        midi_path = Path(tmp) / "incipit.mid"
        for item in samples:
            try:
                payload = _pae_payload(item)
                loaded = toolkit.loadData(payload)
                if loaded is False:
                    raise RuntimeError("Verovio loadData returned false")
                midi_b64 = toolkit.renderToMIDI()
                midi_path.write_bytes(base64.b64decode(midi_b64))
                events = read_compound_midi(midi_path)
                records = tokenizer.encode_events(events)
                if not records:
                    raise RuntimeError("no Compound records")
                active_counts.append(max(0, len(records) - 1))
            except Exception as exc:  # audit must quantify failures rather than abort
                if len(failure_examples) < 25:
                    failure_examples.append(
                        {
                            "record_id": item.get("record_id", ""),
                            "fingerprint": item.get("fingerprint", ""),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    ordered = sorted(active_counts)
    parsed = len(ordered)
    median = 0.0
    if parsed:
        mid = parsed // 2
        if parsed % 2:
            median = float(ordered[mid])
        else:
            median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "attempted": len(samples),
        "parsed": parsed,
        "failures": len(samples) - parsed,
        "parse_success_rate": parsed / len(samples),
        "mean_active_events": (sum(ordered) / parsed) if parsed else 0.0,
        "median_active_events": median,
        "min_active_events": ordered[0] if ordered else 0,
        "max_active_events": ordered[-1] if ordered else 0,
        "p90_active_events": ordered[min(parsed - 1, int(parsed * 0.90))] if parsed else 0,
        "failure_examples": failure_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming census of RISM MARCXML musical incipits.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha1", default=DEFAULT_SHA1)
    parser.add_argument("--work-dir", default=".rism_census")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--pd-death-cutoff", type=int, default=DEFAULT_PD_DEATH_CUTOFF)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="rism_census.json")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    archive = work_dir / "source-2026-08-01.xml.gz"
    if not archive.exists() or _sha1_file(archive) != args.sha1:
        archive.unlink(missing_ok=True)
        print(f"[download] {args.url}", flush=True)
        _download(args.url, archive)
    actual_sha1 = _sha1_file(archive)
    if actual_sha1 != args.sha1:
        raise SystemExit(f"RISM archive SHA1 mismatch: expected {args.sha1}, got {actual_sha1}")

    print(f"[scan] {archive} sha1={actual_sha1}", flush=True)
    census, samples = scan_export(
        archive,
        sample_size=args.sample_size,
        pd_death_cutoff=args.pd_death_cutoff,
        seed=args.seed,
    )
    print(
        "[scan] records={source_records:,} incipits={musical_incipits:,} pd_safe_unique={pd_safe_unique_incipits:,}".format(
            **census
        ),
        flush=True,
    )

    print(f"[sample] parsing {len(samples):,} deterministic reservoir entries", flush=True)
    parsed = parse_sample(samples)
    safe_unique = int(census.get("pd_safe_unique_incipits", 0))
    projected = safe_unique * float(parsed.get("mean_active_events", 0.0))
    result = {
        "source": {
            "name": "RISM source MARCXML export",
            "url": args.url,
            "sha1": actual_sha1,
            "license": "CC-BY-3.0",
            "export_date": "2026-08-01",
        },
        "admission_policy": {
            "kind": "audit-only conservative PD evidence",
            "rule": f"MARC 100$d has bounded historical person dates ending <= {args.pd_death_cutoff}; PAE 031$p present",
            "important": "This is a census gate, not final production legal admission. Unknown/birth-only person dates fail closed.",
        },
        "census": census,
        "sample_parse": parsed,
        "projection": {
            "pd_safe_unique_active_events_before_cross_source_dedup": int(round(projected)),
            "basis": "pd_safe_unique_incipits * sample mean Orbitune active Compound events",
            "is_exact": False,
        },
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
