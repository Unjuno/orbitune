from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import statistics
import urllib.request
from collections import Counter
from pathlib import Path


BASE = "https://storage.googleapis.com/magentadata/datasets/bach-doodle"
DEFAULT_SHARDS = (0, 24, 48, 72, 96, 120, 144, 168)
TOTAL_SHARDS = 192
OFFICIAL_TOTAL_EXAMPLES = 21_600_000
USER_AGENT = "Orbitune-v4-source-census/1.0 (+https://github.com/Unjuno/orbitune)"


def _download(shard: int) -> bytes:
    url = f"{BASE}/bach-doodle.jsonl-{shard:05d}-of-{TOTAL_SHARDS:05d}.gz"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read()


def _notes(sequence_value) -> list[dict[str, object]]:
    if not isinstance(sequence_value, list) or not sequence_value:
        return []
    payload = sequence_value[0]
    if not isinstance(payload, dict):
        return []
    notes = payload.get("notes") or []
    return [note for note in notes if isinstance(note, dict)]


def _qstep(note: dict[str, object], key: str, fallback: str) -> int:
    value = note.get(key)
    if value is None:
        value = note.get(fallback)
    try:
        return int(round(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def _fingerprints(notes: list[dict[str, object]]) -> tuple[str, str]:
    if not notes:
        return "", ""
    normalized_parts: list[str] = []
    transposed_parts: list[str] = []
    first_pitch = None
    first_start = None
    for note in notes:
        try:
            pitch = int(note.get("pitch") or 0)
        except (TypeError, ValueError):
            pitch = 0
        start = _qstep(note, "quantizedStartStep", "startTime")
        end = _qstep(note, "quantizedEndStep", "endTime")
        if first_start is None:
            first_start = start
        if first_pitch is None and pitch > 0:
            first_pitch = pitch
        base_start = first_start or 0
        base_pitch = first_pitch or 0
        normalized_parts.append(f"{start-base_start}:{end-start}:{pitch}")
        transposed_parts.append(f"{start-base_start}:{end-start}:{pitch-base_pitch}")
    return (
        hashlib.sha256("|".join(normalized_parts).encode("utf-8")).hexdigest(),
        hashlib.sha256("|".join(transposed_parts).encode("utf-8")).hexdigest(),
    )


def _feedback_value(row: dict[str, object]) -> str:
    feedback = row.get("feedback")
    if not isinstance(feedback, list) or not feedback:
        return "missing"
    value = feedback[0]
    if isinstance(value, dict):
        for key in ("rating", "value", "feedback"):
            if key in value:
                return str(value[key])
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default=",".join(str(v) for v in DEFAULT_SHARDS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = tuple(int(part) for part in args.shards.split(",") if part.strip())

    rows_total = 0
    input_note_total = 0
    output_note_total = 0
    output_note_counts: list[int] = []
    input_note_counts: list[int] = []
    compressed_bytes = 0
    output_exact = Counter()
    output_transposed = Counter()
    input_exact = Counter()
    input_transposed = Counter()
    feedback = Counter()
    backends = Counter()
    countries = Counter()
    rows_with_output = 0
    rows_with_input = 0
    rows_output_superset_input = 0
    shard_rows: dict[str, int] = {}

    for shard in shards:
        payload = _download(shard)
        compressed_bytes += len(payload)
        local_rows = 0
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
            for raw in handle:
                row = json.loads(raw)
                local_rows += 1
                rows_total += 1
                input_notes = _notes(row.get("input_sequence"))
                output_notes = _notes(row.get("output_sequence"))
                input_count = len(input_notes)
                output_count = len(output_notes)
                if input_count:
                    rows_with_input += 1
                    input_note_total += input_count
                    input_note_counts.append(input_count)
                    exact, transposed = _fingerprints(input_notes)
                    input_exact[exact] += 1
                    input_transposed[transposed] += 1
                if output_count:
                    rows_with_output += 1
                    output_note_total += output_count
                    output_note_counts.append(output_count)
                    exact, transposed = _fingerprints(output_notes)
                    output_exact[exact] += 1
                    output_transposed[transposed] += 1
                if output_count >= input_count and input_count > 0:
                    rows_output_superset_input += 1
                feedback[_feedback_value(row)] += 1
                backends[str(row.get("backend", "missing"))] += 1
                countries[str(row.get("country", "missing"))] += 1
        shard_rows[str(shard)] = local_rows
        print(
            json.dumps(
                {
                    "event": "bach_doodle_shard_complete",
                    "shard": shard,
                    "rows": local_rows,
                    "rows_total": rows_total,
                    "compressed_bytes_total": compressed_bytes,
                    "output_notes_total": output_note_total,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def dup_stats(counter: Counter[str]) -> dict[str, float | int]:
        total = sum(counter.values())
        duplicate_rows = sum(count - 1 for count in counter.values() if count > 1)
        return {
            "rows": total,
            "unique": len(counter),
            "duplicate_rows": duplicate_rows,
            "duplicate_rate": (duplicate_rows / total if total else 0.0),
        }

    mean_output = output_note_total / rows_with_output if rows_with_output else 0.0
    mean_input = input_note_total / rows_with_input if rows_with_input else 0.0
    projected_output_notes = round(mean_output * OFFICIAL_TOTAL_EXAMPLES)
    projected_input_notes = round(mean_input * OFFICIAL_TOTAL_EXAMPLES)

    output_transposed_stats = dup_stats(output_transposed)
    projected_output_after_sample_transposed_dup = round(
        projected_output_notes * (1.0 - float(output_transposed_stats["duplicate_rate"]))
    )

    result = {
        "source": "bach_doodle_sharded_sample",
        "license": "cc-by-4.0-dataset",
        "official_total_examples": OFFICIAL_TOTAL_EXAMPLES,
        "sample_shards": list(shards),
        "sample_rows": rows_total,
        "sample_compressed_bytes": compressed_bytes,
        "shard_rows": shard_rows,
        "rows_with_input": rows_with_input,
        "rows_with_output": rows_with_output,
        "mean_input_notes": mean_input,
        "median_input_notes": statistics.median(input_note_counts) if input_note_counts else 0,
        "mean_output_notes": mean_output,
        "median_output_notes": statistics.median(output_note_counts) if output_note_counts else 0,
        "projected_total_input_note_events": projected_input_notes,
        "projected_total_output_note_events": projected_output_notes,
        "projected_output_after_sample_transposition_shape_duplicate_rate": projected_output_after_sample_transposed_dup,
        "input_exact_signature": dup_stats(input_exact),
        "input_transposition_shape_signature": dup_stats(input_transposed),
        "output_exact_signature": dup_stats(output_exact),
        "output_transposition_shape_signature": output_transposed_stats,
        "rows_output_note_count_ge_input_note_count": rows_output_superset_input,
        "feedback_counts": dict(feedback.most_common()),
        "backend_counts": dict(backends.most_common()),
        "top_country_counts": dict(countries.most_common(30)),
        "rights_warning": (
            "Dataset is CC-BY-4.0, but official Bach Doodle documentation demonstrates that user-entered "
            "input melodies include recognizable third-party/popular melodies. This census measures scale only; "
            "it does not establish a clean composition-rights chain for commercial Base admission."
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text, flush=True)
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
