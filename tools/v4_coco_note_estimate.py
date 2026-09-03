from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
import statistics
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "v4_source_census.py"
spec = importlib.util.spec_from_file_location("v4_source_census", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

BASE = "https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped"
SHARDS = {
    "train": [1, 2, 3, 25, 26, 27, 49, 50, 51, 73, 74, 75],
    "valid": [1, 4, 7, 10],
    "test": [1, 4, 7, 10],
}
SAMPLE_ZIPS = {
    "string_track001010": "https://lukewys.github.io/cocochorales/assets/sample_examples/pieces/string_track001010.zip",
    "brass_track049013": "https://lukewys.github.io/cocochorales/assets/sample_examples/pieces/brass_track049013.zip",
    "woodwind_track097010": "https://lukewys.github.io/cocochorales/assets/sample_examples/pieces/woodwind_track097010.zip",
    "random_track145011": "https://lukewys.github.io/cocochorales/assets/sample_examples/pieces/random_track145011.zip",
}
PIECE_RE = re.compile(r"(?:string|brass|woodwind|random)_track\d{6}")


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": mod.USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read()


def _csv_note_signature(raw: bytes) -> tuple[int, int, str]:
    text = io.StringIO(raw.decode("utf-8", errors="replace"))
    reader = csv.DictReader(text)
    nonzero = 0
    rests = 0
    digest = hashlib.sha256()
    for row in reader:
        pitch_text = row.get("pitch", "0")
        try:
            pitch = int(float(pitch_text or 0))
        except ValueError:
            pitch = 0
        if pitch > 0:
            nonzero += 1
            digest.update(
                (
                    f"{pitch}:"
                    f"{row.get('onset','')}:"
                    f"{row.get('offset','')}:"
                    f"{row.get('note_length','')}\n"
                ).encode("utf-8")
            )
        else:
            rests += 1
    return nonzero, rests, digest.hexdigest()


def _scan_note_expression_shards() -> dict[str, object]:
    total_bytes = 0
    csv_files = 0
    nonzero_notes = 0
    rest_rows = 0
    piece_parts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    split_pieces: dict[str, set[str]] = defaultdict(set)
    unknown_members: list[str] = []

    for split, shard_ids in SHARDS.items():
        for shard in shard_ids:
            url = f"{BASE}/note_expression/{split}/{shard}.tar.bz2"
            payload = _download(url)
            total_bytes += len(payload)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:bz2") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith(".csv"):
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    raw = handle.read()
                    count, rests, signature = _csv_note_signature(raw)
                    csv_files += 1
                    nonzero_notes += count
                    rest_rows += rests
                    match = PIECE_RE.search(member.name)
                    if match is None:
                        if len(unknown_members) < 20:
                            unknown_members.append(member.name)
                        continue
                    piece = match.group(0)
                    split_pieces[split].add(piece)
                    voice_key = member.name.rsplit("/", 1)[-1]
                    piece_parts[piece].append((voice_key, signature))
            print(
                json.dumps(
                    {
                        "event": "coco_note_shard_complete",
                        "split": split,
                        "shard": shard,
                        "download_bytes_total": total_bytes,
                        "csv_files": csv_files,
                        "pieces": sum(len(v) for v in split_pieces.values()),
                        "nonzero_notes": nonzero_notes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    piece_signatures: Counter[str] = Counter()
    voice_counts: Counter[int] = Counter()
    for piece, parts in piece_parts.items():
        voice_counts[len(parts)] += 1
        digest = hashlib.sha256()
        for key, signature in sorted(parts):
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(signature.encode("ascii"))
            digest.update(b"\0")
        piece_signatures[digest.hexdigest()] += 1

    piece_count = len(piece_parts)
    duplicate_piece_count = sum(count - 1 for count in piece_signatures.values() if count > 1)
    return {
        "download_bytes": total_bytes,
        "csv_files": csv_files,
        "piece_count": piece_count,
        "split_piece_counts": {key: len(value) for key, value in sorted(split_pieces.items())},
        "voice_csv_count_per_piece": {str(k): v for k, v in sorted(voice_counts.items())},
        "nonzero_note_rows": nonzero_notes,
        "rest_rows": rest_rows,
        "mean_nonzero_notes_per_piece": (nonzero_notes / piece_count if piece_count else 0.0),
        "note_signature_unique_pieces": len(piece_signatures),
        "note_signature_duplicate_pieces": duplicate_piece_count,
        "note_signature_duplicate_rate": (duplicate_piece_count / piece_count if piece_count else 0.0),
        "unknown_member_examples": unknown_members,
    }


def _scan_official_samples() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for piece, url in SAMPLE_ZIPS.items():
        payload = _download(url)
        with tempfile.TemporaryDirectory(prefix="orbitune-coco-example-") as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                archive.extractall(root)
            midi_candidates = [
                path
                for path in root.rglob("*.mid")
                if path.name.lower() in {"mix.mid", f"{piece}.mid"}
            ]
            if not midi_candidates:
                midi_candidates = [path for path in root.rglob("*.mid") if "stem" not in str(path).lower()]
            if not midi_candidates:
                raise RuntimeError(f"no mixture MIDI found in sample {piece}")
            midi = sorted(midi_candidates, key=lambda p: (p.name.lower() != "mix.mid", len(str(p))))[0]
            scan = mod._scan_midi(str(midi))
            if not scan.get("ok"):
                raise RuntimeError(f"sample MIDI parse failed {piece}: {scan}")

            notes = 0
            rests = 0
            csv_count = 0
            for path in root.rglob("*.csv"):
                # sample archive contains expression CSVs; ignore unrelated tables if any
                raw = path.read_bytes()
                count, rest_count, _ = _csv_note_signature(raw)
                if count or rest_count:
                    notes += count
                    rests += rest_count
                    csv_count += 1
            overhead = int(scan["active_events"]) - notes
            rows.append(
                {
                    "piece": piece,
                    "zip_bytes": len(payload),
                    "expression_csv_files": csv_count,
                    "nonzero_notes": notes,
                    "rest_rows": rests,
                    "compound_records": int(scan["compound_records"]),
                    "active_events": int(scan["active_events"]),
                    "event_overhead_vs_note_rows": overhead,
                    "tracks": int(scan["tracks"]),
                }
            )
    overheads = [int(row["event_overhead_vs_note_rows"]) for row in rows]
    return {
        "examples": rows,
        "overheads": overheads,
        "overhead_min": min(overheads),
        "overhead_max": max(overheads),
        "overhead_mean": statistics.mean(overheads),
        "overhead_consistent": len(set(overheads)) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    notes = _scan_note_expression_shards()
    samples = _scan_official_samples()
    piece_count = int(notes["piece_count"])
    note_count = int(notes["nonzero_note_rows"])
    if piece_count <= 0:
        raise RuntimeError("no CocoChorales pieces found in note-expression census")

    full_pieces = 240_000
    avg_notes = note_count / piece_count
    overhead_mean = float(samples["overhead_mean"])
    overhead_min = int(samples["overhead_min"])
    overhead_max = int(samples["overhead_max"])
    estimate = round((avg_notes + overhead_mean) * full_pieces)
    estimate_low = round((avg_notes + overhead_min) * full_pieces)
    estimate_high = round((avg_notes + overhead_max) * full_pieces)

    # A second conservative projection removes the observed duplicate-signature rate
    # from the sampled 40k subset. This is not Orbitune composition-fingerprint dedup,
    # but gives a useful lower-bound sensitivity estimate before downloading 569 GB.
    dup_rate = float(notes["note_signature_duplicate_rate"])
    dedup_sensitivity = round(estimate * (1.0 - dup_rate))

    result = {
        "source": "cocochorales_official_tiny_40k_estimate",
        "license": "cc-by-4.0",
        "full_dataset_pieces": full_pieces,
        "official_tiny_note_expression_census": notes,
        "official_example_calibration": samples,
        "projected_full_active_events": estimate,
        "projected_full_active_events_low": estimate_low,
        "projected_full_active_events_high": estimate_high,
        "projected_after_sample_note_signature_duplicate_rate": dedup_sensitivity,
        "method_note": (
            "Counts all non-rest note-expression rows in the official 40k tiny subset, then calibrates "
            "the constant/non-note Compound overhead using four official downloadable mixture-MIDI examples. "
            "This is an estimate, not the final Orbitune cross-source dedup census."
        ),
    }
    mod._write_result(result, args.output)


if __name__ == "__main__":
    main()
