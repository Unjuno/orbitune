"""Re-execute the full-v4 cross-dedup against the v4 manifest.

Loads the v4 manifest fingerprint index and the delta census JSON, then
performs exact cross-dedup by:
  1. normalized_fingerprint (production key)
  2. composition_fingerprint (split grouping; not production dedup)

Outputs a per-artifact cross-v4 match list and a summary to a JSON file
in the tools/ directory for permanent provenance.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mido


V4_MANIFEST = Path(r"C:\ov4\manifest.jsonl")
DELTA_CENSUS = Path(
    r"C:\Users\junny\OneDrive\Desktop\MIDI-GPT\orbitune_clone\tools\openscore_string_quartets_v5_delta_census.json"
)
OUT = Path(
    r"C:\Users\junny\OneDrive\Desktop\MIDI-GPT\orbitune_clone\tools\openscore_string_quartets_v5_cross_v4_dedup.json"
)
V4_MANIFEST_SHA256_FROZEN = "1c582a08a3087952a57a604b5652cae2bef6dd4e6acb2addc5eb49cdfbe58c72"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    actual_sha = sha256_file(V4_MANIFEST)
    assert actual_sha == V4_MANIFEST_SHA256_FROZEN, (
        f"v4 manifest SHA mismatch: expected {V4_MANIFEST_SHA256_FROZEN}, got {actual_sha}"
    )
    print(f"[cross-v4] v4_manifest_sha256 verified: {actual_sha}")

    v4_norm: dict[str, dict] = {}
    v4_comp: dict[str, dict] = {}
    with V4_MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n = r.get("normalized_fingerprint")
            c = r.get("composition_fingerprint")
            if n:
                v4_norm[n] = r
            if c:
                v4_comp[c] = r
    print(f"[cross-v4] v4 normalized_fp unique = {len(v4_norm)}")
    print(f"[cross-v4] v4 composition_fp unique = {len(v4_comp)}")

    delta = json.loads(DELTA_CENSUS.read_text(encoding="utf-8"))
    rows = delta["rows"]
    print(f"[cross-v4] delta input rows = {len(rows)}")

    matches_norm = 0
    matches_comp = 0
    details = []
    for r in rows:
        n = r.get("normalized_fingerprint")
        c = r.get("composition_fingerprint")
        if n and n in v4_norm:
            v4_hit = v4_norm[n]
            matches_norm += 1
            details.append({
                "delta_relpath": r["relpath"],
                "delta_active_event_count": r["active_event_count"],
                "match_basis": "normalized_fingerprint",
                "v4_source_id": v4_hit["source_id"],
                "v4_path": v4_hit["path"],
                "v4_events": v4_hit["events"],
                "v4_split": v4_hit["split"],
                "precedence_outcome": "DELTA_DROPPED (v4 retains canonical; delta is duplicate of an already-admitted v4 work)",
            })
        elif c and c in v4_comp:
            v4_hit = v4_comp[c]
            matches_comp += 1
            details.append({
                "delta_relpath": r["relpath"],
                "delta_active_event_count": r["active_event_count"],
                "match_basis": "composition_fingerprint",
                "v4_source_id": v4_hit["source_id"],
                "v4_path": v4_hit["path"],
                "v4_events": v4_hit["events"],
                "v4_split": v4_hit["split"],
                "precedence_outcome": "DELTA_DROPPED (composition-group duplicate; v4 retains canonical; Orbitune production dedup key is normalized_fingerprint, but composition_fingerprint is reported as supporting evidence)",
            })

    cross_dedup_removed = matches_norm + matches_comp
    accepted = len(rows) - cross_dedup_removed
    accepted_active_events = sum(
        r["active_event_count"] for r in rows if r["relpath"] not in {d["delta_relpath"] for d in details}
    )

    out = {
        "v4_manifest_sha256": V4_MANIFEST_SHA256_FROZEN,
        "v4_manifest_sha256_actual": actual_sha,
        "v4_manifest_sha256_match": True,
        "v4_normalized_fingerprint_index_size": len(v4_norm),
        "v4_composition_fingerprint_index_size": len(v4_comp),
        "delta_input_artifacts": len(rows),
        "delta_normalized_matches": matches_norm,
        "delta_composition_matches": matches_comp,
        "delta_cross_v4_dedup_removed": cross_dedup_removed,
        "delta_post_dedup_accepted": accepted,
        "delta_post_dedup_active_events": accepted_active_events,
        "production_dedup_key": "normalized_fingerprint",
        "composition_fingerprint_role": "split_grouping_only (NOT production dedup key)",
        "full_v4_dedup_gate": "PASS" if accepted == len(rows) else "FAIL",
        "match_details": details,
    }

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n[cross-v4] ==========================")
    print(f"[cross-v4] v4_manifest_sha256                  = {V4_MANIFEST_SHA256_FROZEN}")
    print(f"[cross-v4] FULL_V4_FINGERPRINTS_INDEXED        = {len(v4_norm)}")
    print(f"[cross-v4] CROSS_V4_NORMALIZED_MATCHES         = {matches_norm}")
    print(f"[cross-v4] CROSS_V4_COMPOSITION_MATCHES        = {matches_comp}")
    print(f"[cross-v4] CROSS_V4_DEDUP_REMOVED              = {cross_dedup_removed}")
    print(f"[cross-v4] DELTA_ACCEPTED                      = {accepted}")
    print(f"[cross-v4] POST_FULL_V4_DEDUP_ACTIVE_EVENTS    = {accepted_active_events}")
    print(f"[cross-v4] FULL_V4_DEDUP_GATE                  = {out['full_v4_dedup_gate']}")
    print(f"[cross-v4] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
