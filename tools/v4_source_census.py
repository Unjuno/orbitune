from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from orbitune.compound import CompoundEventType
from orbitune.compound_midi import read_compound_midi
from orbitune.midi_metadata import inspect_midi_metadata
from orbitune.tokenizer.compound_event import CompoundEventTokenizer


USER_AGENT = "Orbitune-v4-source-census/1.0 (+https://github.com/Unjuno/orbitune)"
NRG_URL = "https://zenodo.org/records/15304989/files/nrgcp_midi_dataset.tar.gz?download=1"
NRG_MD5 = "7443fe30674ef149aa4c23580044f597"
CPDL_API = "https://www.cpdl.org/wiki/api.php"
HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"
SAFE_IMSLP_LICENSES = ("cc-by-3.0", "cc-by-4.0")

_TOKENIZER: CompoundEventTokenizer | None = None


def _tokenizer() -> CompoundEventTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = CompoundEventTokenizer()
    return _TOKENIZER


def _hash_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan_midi(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    try:
        raw_sha = _sha256_file(path)
        events = read_compound_midi(path)
        if not events:
            raise ValueError("empty MIDI event sequence")
        first_step = events[0].step
        notes = [event for event in events if event.type is CompoundEventType.NOTE]
        first_pitch = notes[0].a1 if notes else 0
        normalized: list[str] = []
        composition: list[str] = []
        for event in events:
            normalized.append(
                f"{int(event.type)}:{event.step-first_step}:{event.channel}:{event.a1}:{event.a2}:{event.a3}:{event.a4}"
            )
            if event.type is CompoundEventType.NOTE:
                composition.append(f"{event.step-first_step}:{event.a1-first_pitch}:{event.a2}")
            elif event.type is CompoundEventType.TIME_SIGNATURE:
                composition.append(f"ts:{event.step-first_step}:{event.a1}:{event.a2}")
        normalized_hash = _hash_parts(normalized)
        composition_hash = _hash_parts(composition or normalized)
        records = _tokenizer().encode_events(events)
        if not records:
            raise ValueError("no Compound records")
        record_count = len(records)
        track_count = inspect_midi_metadata(path).track_count
        return {
            "path": str(path),
            "ok": True,
            "raw_sha256": raw_sha,
            "normalized_fingerprint": normalized_hash,
            "composition_fingerprint": composition_hash,
            "midi_events": len(events),
            "compound_records": record_count,
            "active_events": max(0, record_count - 1),
            "tracks": int(track_count),
        }
    except Exception as exc:  # census must report failures rather than aborting the entire source
        return {
            "path": str(path),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _midi_paths(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
        ]
    )


def scan_midi_tree(root: Path, *, workers: int) -> dict[str, Any]:
    paths = _midi_paths(root)
    raw_seen: set[str] = set()
    norm_seen: dict[str, tuple[int, int]] = {}
    comp_seen: set[str] = set()
    failures: list[dict[str, str]] = []
    track_counts: Counter[int] = Counter()
    total_records = 0
    total_active = 0
    unique_norm_records = 0
    unique_norm_active = 0
    raw_duplicate_files = 0
    normalized_duplicate_files = 0
    composition_duplicate_files = 0

    if workers <= 1:
        results = map(lambda p: _scan_midi(str(p)), paths)
        pool = None
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_scan_midi, (str(path) for path in paths), chunksize=32)

    try:
        for result in results:
            if not result.get("ok"):
                failures.append({"path": str(result["path"]), "error": str(result["error"])})
                continue
            raw_sha = str(result["raw_sha256"])
            norm = str(result["normalized_fingerprint"])
            comp = str(result["composition_fingerprint"])
            records = int(result["compound_records"])
            active = int(result["active_events"])
            tracks = int(result["tracks"])
            total_records += records
            total_active += active
            track_counts[tracks] += 1

            if raw_sha in raw_seen:
                raw_duplicate_files += 1
            else:
                raw_seen.add(raw_sha)

            if norm in norm_seen:
                normalized_duplicate_files += 1
            else:
                norm_seen[norm] = (records, active)
                unique_norm_records += records
                unique_norm_active += active

            if comp in comp_seen:
                composition_duplicate_files += 1
            else:
                comp_seen.add(comp)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    return {
        "root": str(root),
        "midi_files": len(paths),
        "parsed_files": len(paths) - len(failures),
        "parse_failures": len(failures),
        "failure_examples": failures[:20],
        "raw_unique": len(raw_seen),
        "raw_duplicate_files": raw_duplicate_files,
        "normalized_unique": len(norm_seen),
        "normalized_duplicate_files": normalized_duplicate_files,
        "composition_unique": len(comp_seen),
        "composition_duplicate_files": composition_duplicate_files,
        "compound_records_all": total_records,
        "active_events_all": total_active,
        "compound_records_normalized_unique": unique_norm_records,
        "active_events_normalized_unique": unique_norm_active,
        "track_count_histogram": {str(k): v for k, v in sorted(track_counts.items())},
    }


def _request(url: str, *, timeout: int = 120) -> urllib.request.urlopen:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def _http_json(url: str, *, timeout: int = 120) -> dict[str, Any]:
    with _request(url, timeout=timeout) as response:
        return json.load(response)


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with _request(url, timeout=300) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _extract_archive(archive: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as handle:
            handle.extractall(out)
        return out
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(out)
        return out
    raise ValueError(f"unsupported archive format: {archive}")


def run_nrg(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="orbitune-nrg-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "nrgcp_midi_dataset.tar.gz"
        _download(NRG_URL, archive)
        md5 = hashlib.md5(archive.read_bytes()).hexdigest()  # noqa: S324 - upstream integrity value
        if md5 != NRG_MD5:
            raise RuntimeError(f"NRG-CP MD5 mismatch: {md5} != {NRG_MD5}")
        extracted = _extract_archive(archive, tmp_path / "extracted")
        result = scan_midi_tree(extracted, workers=args.workers)
        result.update(
            {
                "source": "nrg_cp",
                "license": "cc-by-4.0",
                "archive_bytes": archive.stat().st_size,
                "archive_md5": md5,
            }
        )
        return result


def run_scan_archive(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input).resolve()
    if source.is_dir():
        root = source
        temp = None
    else:
        temp = tempfile.TemporaryDirectory(prefix="orbitune-midi-archive-")
        root = _extract_archive(source, Path(temp.name) / "extracted")
    try:
        result = scan_midi_tree(root, workers=args.workers)
        result["source"] = args.source_name
        result["input"] = str(source)
        if source.is_file():
            result["archive_bytes"] = source.stat().st_size
            result["archive_sha256"] = _sha256_file(source)
        return result
    finally:
        if temp is not None:
            temp.cleanup()


def _cpdl_params(**kwargs: object) -> str:
    payload = {"format": "json", "formatversion": "2", **kwargs}
    return CPDL_API + "?" + urllib.parse.urlencode(payload)


def _cpdl_edition_namespace() -> int:
    data = _http_json(_cpdl_params(action="query", meta="siteinfo", siprop="namespaces"))
    namespaces = data["query"]["namespaces"]
    if isinstance(namespaces, dict):
        values = namespaces.values()
    else:
        values = namespaces
    for item in values:
        name = str(item.get("name", item.get("*", ""))).strip().lower()
        canonical = str(item.get("canonical", "")).strip().lower()
        if name == "edition" or canonical == "edition":
            return int(item["id"])
    raise RuntimeError("CPDL Edition namespace not found")


def _cpdl_all_editions(namespace: int) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    continuation: str | None = None
    while True:
        params: dict[str, object] = {
            "action": "query",
            "list": "allpages",
            "apnamespace": namespace,
            "apprefix": "CPDL ",
            "aplimit": "max",
        }
        if continuation:
            params["apcontinue"] = continuation
        data = _http_json(_cpdl_params(**params))
        pages.extend(data.get("query", {}).get("allpages", []))
        continuation = data.get("continue", {}).get("apcontinue")
        if not continuation:
            break
    return pages


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _cpdl_fetch_batch(pageids: list[int]) -> list[dict[str, Any]]:
    url = _cpdl_params(
        action="query",
        prop="revisions",
        pageids="|".join(str(v) for v in pageids),
        rvprop="content",
        rvslots="main",
    )
    data = _http_json(url)
    return list(data.get("query", {}).get("pages", []))


_PD_RE = re.compile(r"\{\{\s*Copy\s*\|\s*(?:PD|Public\s+Domain)\b", re.I)
_CC_BY_RE = re.compile(
    r"\{\{\s*CopyCC\s*\|\s*Attribution(?:\s+Only)?[^}\n]*(?P<version>3\.0|4\.0)[^}\n]*\}\}",
    re.I,
)
_UNSAFE_COPY_RE = re.compile(
    r"\{\{\s*(?:Copy|CopyCC)\s*\|[^}\n]*(?:Non[- ]?Commercial|Share[- ]?Alike|No[- ]?Derivatives|Personal|CPDL|GnuGPL)",
    re.I,
)
_WITHDRAWN_RE = re.compile(r"(?:\bStatus\s*=\s*withdrawn\b|\|\s*status\s*=\s*withdrawn\b)", re.I)
_MIDI_RE = re.compile(r"\.(?:mid|midi)\b", re.I)
_MUSICXML_RE = re.compile(r"\.(?:mxl|musicxml|xml)\b", re.I)
_MUSESCORE_RE = re.compile(r"\.(?:mscz|mscx)\b", re.I)
_LILYPOND_RE = re.compile(r"\.ly\b", re.I)


def _revision_content(page: dict[str, Any]) -> str:
    revisions = page.get("revisions") or []
    if not revisions:
        return ""
    rev = revisions[0]
    slots = rev.get("slots") or {}
    if isinstance(slots, dict) and "main" in slots:
        main = slots["main"]
        return str(main.get("content", main.get("*", "")))
    return str(rev.get("content", rev.get("*", "")))


def _classify_cpdl_page(page: dict[str, Any]) -> dict[str, Any]:
    content = _revision_content(page)
    title = str(page.get("title", ""))
    if not content:
        return {"title": title, "status": "no_content"}
    if _WITHDRAWN_RE.search(content):
        return {"title": title, "status": "withdrawn"}
    unsafe = bool(_UNSAFE_COPY_RE.search(content))
    pd = bool(_PD_RE.search(content))
    cc_match = _CC_BY_RE.search(content)
    if unsafe:
        license_id = "unsafe_or_copyleft"
    elif pd and not cc_match:
        license_id = "public-domain"
    elif cc_match and not pd:
        license_id = f"cc-by-{cc_match.group('version')}"
    elif pd and cc_match:
        license_id = "ambiguous_multiple_safe_templates"
    else:
        license_id = "unrecognized"

    formats = {
        "midi": bool(_MIDI_RE.search(content)),
        "musicxml": bool(_MUSICXML_RE.search(content)),
        "musescore": bool(_MUSESCORE_RE.search(content)),
        "lilypond": bool(_LILYPOND_RE.search(content)),
    }
    symbolic = any(formats.values())
    accepted_license = license_id in {"public-domain", "cc-by-3.0", "cc-by-4.0"}
    return {
        "title": title,
        "status": "candidate" if accepted_license and symbolic else "excluded",
        "license": license_id,
        "formats": formats,
        "symbolic": symbolic,
    }


def run_cpdl(args: argparse.Namespace) -> dict[str, Any]:
    namespace = _cpdl_edition_namespace()
    pages = _cpdl_all_editions(namespace)
    batches = [
        [int(page["pageid"]) for page in batch]
        for batch in _chunks(pages, 50)
    ]
    classified: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fetched in pool.map(_cpdl_fetch_batch, batches):
            classified.extend(_classify_cpdl_page(page) for page in fetched)

    licenses = Counter(str(row.get("license", row.get("status", "unknown"))) for row in classified)
    candidate_rows = [row for row in classified if row.get("status") == "candidate"]
    midi_rows = [row for row in candidate_rows if row["formats"]["midi"]]
    convertible_rows = [
        row
        for row in candidate_rows
        if row["formats"]["musicxml"] or row["formats"]["musescore"] or row["formats"]["lilypond"]
    ]
    format_counts = Counter()
    for row in candidate_rows:
        for key, value in row["formats"].items():
            if value:
                format_counts[key] += 1
    return {
        "source": "cpdl",
        "edition_namespace": namespace,
        "edition_pages": len(pages),
        "classified_pages": len(classified),
        "license_class_counts": dict(sorted(licenses.items())),
        "safe_symbolic_editions": len(candidate_rows),
        "safe_midi_editions": len(midi_rows),
        "safe_convertible_editions": len(convertible_rows),
        "safe_format_counts": dict(sorted(format_counts.items())),
        "candidate_examples": candidate_rows[:30],
        "policy": "fail-closed: only explicit Public Domain or CC-BY 3.0/4.0 edition templates plus symbolic file extension",
    }


def _hf_parquet_files(dataset: str) -> list[dict[str, Any]]:
    url = HF_DATASETS_SERVER + "/parquet?" + urllib.parse.urlencode({"dataset": dataset})
    data = _http_json(url)
    files = data.get("parquet_files") or []
    if not files:
        raise RuntimeError(f"no parquet export for {dataset}: {data}")
    return files


def run_imslp(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("imslp census requires duckdb") from exc

    datasets = [
        "TiMauzi/imslp-midi-by-sa",
        "TiMauzi/imslp-midi-by-nc-sa",
        "TiMauzi/imslp-midi-cc0-1.0",
    ]
    parquet: dict[str, list[str]] = {
        dataset: [str(item["url"]) for item in _hf_parquet_files(dataset)]
        for dataset in datasets
    }
    con = duckdb.connect(database=":memory:")

    def rows_for(dataset: str, licenses: tuple[str, ...] | None) -> list[tuple[str, str, str]]:
        urls = parquet[dataset]
        url_sql = "[" + ",".join("'" + url.replace("'", "''") + "'" for url in urls) + "]"
        where = ""
        if licenses:
            vals = ",".join("'" + value.replace("'", "''") + "'" for value in licenses)
            where = f" WHERE lower(license) IN ({vals})"
        query = (
            "SELECT midi_source, metadata_source, lower(license) AS license "
            f"FROM read_parquet({url_sql}){where}"
        )
        return [(str(a), str(b), str(c)) for a, b, c in con.execute(query).fetchall()]

    by_sa = rows_for(datasets[0], SAFE_IMSLP_LICENSES)
    by_nc_sa = rows_for(datasets[1], SAFE_IMSLP_LICENSES)
    cc0_all = rows_for(datasets[2], None)
    cc0_sources = {row[0] for row in cc0_all}

    combined: dict[str, tuple[str, str, str]] = {}
    source_membership: dict[str, set[str]] = {}
    for dataset, rows in ((datasets[0], by_sa), (datasets[1], by_nc_sa)):
        for row in rows:
            combined.setdefault(row[0], row)
            source_membership.setdefault(row[0], set()).add(dataset)

    overlap_cc0 = sum(1 for source in combined if source in cc0_sources)
    unique_new = [row for source, row in combined.items() if source not in cc0_sources]
    license_counts = Counter(row[2] for row in unique_new)
    wrapper_overlap = sum(1 for memberships in source_membership.values() if len(memberships) > 1)
    return {
        "source": "imslp_ccby_audit",
        "by_sa_safe_rows": len(by_sa),
        "by_nc_sa_safe_rows": len(by_nc_sa),
        "wrapper_union_unique_midi_sources": len(combined),
        "wrapper_overlap_unique_midi_sources": wrapper_overlap,
        "overlap_with_cc0_midi_sources": overlap_cc0,
        "new_ccby_unique_midi_sources": len(unique_new),
        "new_ccby_license_counts": dict(sorted(license_counts.items())),
        "new_candidate_examples": [
            {"midi_source": row[0], "metadata_source": row[1], "license": row[2]}
            for row in unique_new[:30]
        ],
        "note": "candidate count only; original IMSLP provenance/license should be revalidated before admission",
    }


def _write_result(result: dict[str, Any], output: str | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text, flush=True)
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Disposable commercial-v4 source census harness")
    sub = parser.add_subparsers(dest="command", required=True)

    nrg = sub.add_parser("nrg-cp")
    nrg.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    nrg.add_argument("--output")

    scan = sub.add_parser("scan-archive")
    scan.add_argument("--input", required=True)
    scan.add_argument("--source-name", required=True)
    scan.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    scan.add_argument("--output")

    cpdl = sub.add_parser("cpdl")
    cpdl.add_argument("--workers", type=int, default=8)
    cpdl.add_argument("--output")

    imslp = sub.add_parser("imslp")
    imslp.add_argument("--output")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "nrg-cp":
        result = run_nrg(args)
    elif args.command == "scan-archive":
        result = run_scan_archive(args)
    elif args.command == "cpdl":
        result = run_cpdl(args)
    elif args.command == "imslp":
        result = run_imslp(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _write_result(result, getattr(args, "output", None))


if __name__ == "__main__":
    main()
