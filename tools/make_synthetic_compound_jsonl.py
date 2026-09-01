"""Generate a synthetic Compound Event JSONL corpus for benchmarking.

The real Orbitune Compound corpus is not shipped with the repository; this
script writes a small synthetic JSONL that satisfies the schema enforced by
``orbitune.compound_training.load_compound_jsonl``.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.tokenizer.compound_event import COMPOUND_RECORD_WIDTH


def synth_song(rng: random.Random, length: int) -> tuple[str, str, list[list[int]]]:
    """Generate a plausible compound-event sequence of ``length`` events."""
    records: list[list[int]] = []
    last_time = 0
    for _ in range(length):
        event_type = rng.randint(0, 9)
        channel = rng.randint(0, 15)
        delta_coarse = rng.randint(0, 6)
        delta_residual = rng.randint(0, 15)
        # Fields 4..7 are coarse/residual MIDI values; keep within 1024.
        a1 = rng.randint(0, 1023)
        a2 = rng.randint(0, 1023)
        velocity = rng.randint(0, 127)
        pitch = rng.randint(0, 255)
        duration_coarse = rng.randint(0, 6)
        duration_residual = rng.randint(0, 15)
        continuous_coarse = rng.randint(0, 7)
        continuous_residual = rng.randint(0, 7)
        records.append([
            event_type, channel, delta_coarse, delta_residual,
            a1, a2, velocity, pitch,
            duration_coarse, duration_residual, continuous_coarse, continuous_residual,
        ])
        last_time += 1
    payload = json.dumps(records).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()
    return f"synthetic://{length}", sha, records


def main() -> None:
    out = Path("data/continuous/synthetic_compound.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    n_songs = 32
    min_len = 600  # > 512 to allow seq_len=512 sampling
    max_len = 1500
    with out.open("w", encoding="utf-8") as fh:
        for i in range(n_songs):
            length = rng.randint(min_len, max_len)
            path, sha, records = synth_song(rng, length)
            fh.write(json.dumps({
                "path": path,
                "sha256": sha,
                "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
                "record_width": COMPOUND_RECORD_WIDTH,
                "records": records,
            }) + "\n")
    print(f"wrote {out} with {n_songs} synthetic songs")


if __name__ == "__main__":
    main()