from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from collections import Counter
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "v4_source_census.py"
spec = importlib.util.spec_from_file_location("v4_source_census", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATASETS = (
    "TiMauzi/imslp-midi-by-sa",
    "TiMauzi/imslp-midi-by-nc-sa",
)
CC0_DATASET = "TiMauzi/imslp-midi-cc0-1.0"
SAFE = ("cc-by-3.0", "cc-by-4.0")


def _url_sql(urls: list[str]) -> str:
    return "[" + ",".join("'" + url.replace("'", "''") + "'" for url in urls) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    con = duckdb.connect(database=":memory:")
    parquet = {
        dataset: [str(item["url"]) for item in mod._hf_parquet_files(dataset)]
        for dataset in (*DATASETS, CC0_DATASET)
    }
    cc0_sources = {
        str(row[0])
        for row in con.execute(
            f"SELECT midi_source FROM read_parquet({_url_sql(parquet[CC0_DATASET])})"
        ).fetchall()
    }

    candidates: dict[str, tuple[str, str, str, bytes]] = {}
    membership: dict[str, set[str]] = {}
    wrapper_rows: dict[str, int] = {}
    licenses = ",".join("'" + value + "'" for value in SAFE)

    for dataset in DATASETS:
        query = (
            "SELECT midi_source, metadata_source, lower(license) AS license, midi "
            f"FROM read_parquet({_url_sql(parquet[dataset])}) "
            f"WHERE lower(license) IN ({licenses})"
        )
        rows = con.execute(query).fetchall()
        wrapper_rows[dataset] = len(rows)
        for midi_source, metadata_source, license_id, midi_blob in rows:
            source = str(midi_source)
            membership.setdefault(source, set()).add(dataset)
            if midi_blob is None:
                continue
            blob = bytes(midi_blob)
            previous = candidates.get(source)
            if previous is not None and previous[3] != blob:
                raise RuntimeError(f"same IMSLP midi_source has different bytes across wrappers: {source}")
            candidates[source] = (source, str(metadata_source), str(license_id), blob)

    new_candidates = {
        source: row for source, row in candidates.items() if source not in cc0_sources
    }
    license_counts = Counter(row[2] for row in new_candidates.values())
    wrapper_overlap = sum(1 for source in new_candidates if len(membership.get(source, ())) > 1)

    with tempfile.TemporaryDirectory(prefix="orbitune-imslp-ccby-") as tmp:
        midi_root = Path(tmp)
        provenance: dict[str, dict[str, str]] = {}
        for source, row in new_candidates.items():
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            path = midi_root / f"{digest}.mid"
            path.write_bytes(row[3])
            provenance[path.name] = {
                "midi_source": row[0],
                "metadata_source": row[1],
                "license": row[2],
            }
        exact = mod.scan_midi_tree(midi_root, workers=args.workers)

    result = {
        "source": "imslp_ccby_exact_audit",
        "wrapper_safe_rows": wrapper_rows,
        "wrapper_union_unique_midi_sources": len(candidates),
        "wrapper_overlap_new_sources": wrapper_overlap,
        "overlap_with_existing_cc0_midi_sources": len(candidates) - len(new_candidates),
        "new_ccby_unique_midi_sources": len(new_candidates),
        "new_ccby_license_counts": dict(sorted(license_counts.items())),
        "exact_midi_census": exact,
        "candidate_examples": [
            {"midi_source": row[0], "metadata_source": row[1], "license": row[2]}
            for row in list(new_candidates.values())[:30]
        ],
        "admission_note": (
            "Exact bytes/event counts are measured here, but each original IMSLP file/metadata "
            "page still requires fail-closed provenance revalidation before commercial admission."
        ),
    }
    mod._write_result(result, args.output)


if __name__ == "__main__":
    main()
