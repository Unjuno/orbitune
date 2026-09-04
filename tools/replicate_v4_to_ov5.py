"""Replicate C:\\ov4 source tree and converted tree to C:\\ov5, then
replace the openscore_string_quartets subtree with the v5 checkout +
the 41 delta-generated MIDI files.

This is a non-incremental sync: v4 material is copied wholesale and
v5's only change is the OpenScore StringQuartets source pin. After
this script, C:\\ov5 mirrors C:\\ov4 except for openscore_string_quartets
which is at the v5 commit a3b38134 and has +41 new converted MIDIs.

Post-replication integrity contract (hard fail if violated):
    For every non-OpenScore source:
        SOURCE_EXPECTED_FILES  = file count in C:\\ov4\\<source> - .git/**
        SOURCE_REPLICATED_FILES = file count in C:\\ov5\\<source> - .git/**
        SOURCE_MISSING_FILES  = 0
        SOURCE_EXTRA_FILES    = 0
        plus per-file byte + SHA256 match for every copied file
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

OV4 = Path(r"C:\ov4")
OV5 = Path(r"C:\ov5")
# Sources whose v5 file collection must equal v4 (excluding .git internals).
NON_DELTA_SOURCES = [
    "pdmx", "openscore_lieder", "openscore_orchestra", "mutopia",
    "imslp_midi_cc0", "florence_price_art_songs", "muse_omr_benchmark",
    "nifc_polish_scores", "nifc_chopin_first_editions", "nrg_cp",
]
# Sources that are pinned at a specific upstream ref and copied verbatim.
DELTA_PIN_SOURCES = [
    "openscore_string_quartets",  # pinned to a3b38134 (delta vs v4)
]
# Sources that are junctions/symlinks (no copy).
LINKED_SOURCES = [
    "groove_midi_dataset",  # mklink /J to C:\ov4\groove_midi_dataset
]


def _iter_files(root: Path):
    """Yield files under `root` excluding any .git/ subtree."""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] == ".git":
            continue
        yield p, rel


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_identity(sid: str) -> None:
    src_root = OV4 / sid
    dst_root = OV5 / sid
    if not src_root.exists():
        print(f"[verify] {sid}: source {src_root} missing in v4; skipping", flush=True)
        return
    if not dst_root.exists():
        raise RuntimeError(f"[verify] {sid}: destination {dst_root} missing in v5")

    src_files = {rel: p for p, rel in _iter_files(src_root)}
    dst_files = {rel: p for p, rel in _iter_files(dst_root)}

    missing = [rel for rel in src_files if rel not in dst_files]
    extra = [rel for rel in dst_files if rel not in src_files]
    if missing or extra:
        raise RuntimeError(
            f"[verify] {sid} FILESET MISMATCH: "
            f"missing={len(missing)} extra={len(extra)} "
            f"(first missing={missing[:3]}, first extra={extra[:3]})"
        )

    bad = 0
    for rel, sp in src_files.items():
        dp = dst_files[rel]
        if sp.stat().st_size != dp.stat().st_size:
            bad += 1
            print(f"[verify] {sid} SIZE MISMATCH: {rel}", flush=True)
            continue
        if _sha256(sp) != _sha256(dp):
            bad += 1
            print(f"[verify] {sid} HASH MISMATCH: {rel}", flush=True)
    if bad:
        raise RuntimeError(f"[verify] {sid}: {bad} files failed size/hash identity")

    print(
        f"[verify] {sid}: "
        f"expected={len(src_files)} replicated={len(dst_files)} "
        f"missing=0 extra=0 bad=0 IDENTITY=PASS",
        flush=True,
    )


def main() -> int:
    # Step 1: copy source roots for non-delta sources.
    for sid in NON_DELTA_SOURCES:
        src = OV4 / sid
        if not src.exists():
            print(f"[replicate] WARN: {src} missing in v4; skipping", flush=True)
            continue
        dst = OV5 / sid
        if dst.exists():
            print(f"[replicate] {sid} already present in v5; reconciling via top-up copy", flush=True)
            # Top-up: copy any v4 file missing in v5 (handles partial prior copies).
            for sp, rel in _iter_files(src):
                dp = dst / rel
                if dp.exists():
                    continue
                dp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sp, dp)
        else:
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))
            print(f"[replicate] copied source {sid}", flush=True)
        # Copy converted subtree if present.
        src_conv = OV4 / "converted" / sid
        if src_conv.exists():
            dst_conv = OV5 / "converted" / sid
            if dst_conv.exists():
                shutil.rmtree(dst_conv)
            shutil.copytree(src_conv, dst_conv)
            print(f"[replicate] copied converted/{sid}", flush=True)

    # Step 2: delta-pinned sources are already placed out-of-band (git checkout + delta MIDI).
    for sid in DELTA_PIN_SOURCES:
        dst = OV5 / sid
        if dst.exists():
            print(f"[replicate] {sid} pinned; leaving v5 checkout in place", flush=True)
        else:
            print(f"[replicate] WARN: {sid} pinned but not present in v5", flush=True)

    # Step 3: linked sources.
    for sid in LINKED_SOURCES:
        dst = OV5 / sid
        if dst.exists():
            print(f"[replicate] {sid} junction present in v5 (no copy)", flush=True)
        else:
            print(f"[replicate] WARN: {sid} junction missing in v5", flush=True)

    # Step 4: hard-fail identity verification for every non-delta, non-linked source.
    for sid in NON_DELTA_SOURCES:
        _verify_identity(sid)
    for sid in DELTA_PIN_SOURCES:
        # Openscore SQ at v5 has 38 added mscz + 3 renamed mscx vs v4; do not
        # require file-set equality with v4. Just confirm v5 has its files.
        dst = OV5 / sid
        if dst.exists():
            n = sum(1 for _ in _iter_files(dst))
            print(f"[replicate] {sid} (pinned): {n} non-.git files in v5", flush=True)

    # Step 5: emit replication report.
    report = {
        "ov4": str(OV4),
        "ov5": str(OV5),
        "non_delta_sources": NON_DELTA_SOURCES,
        "delta_pin_sources": DELTA_PIN_SOURCES,
        "linked_sources": LINKED_SOURCES,
        "replication_identity": "PASS",
    }
    out = OV5 / "replication_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[replicate] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
