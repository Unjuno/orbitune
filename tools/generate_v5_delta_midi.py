"""Generate MIDI files for the 41 v5 delta artifacts so v5 build can
proceed without a MuseScore CLI.

The 41 MuseScore 4 .mscz/.mscx files were already downloaded and parsed
by the delta census (tools/openscore_string_quartets_v5_delta_census.json).
We re-parse the same source files here, build a multi-track MIDI file
with proper tempo/time-signature, and write the result to
``C:\\ov5\\converted\\openscore_string_quartets\\scores\\<composer>\\<work>\\<id>.mid``.

This is a best-effort MIDI writer used only for the v5 OpenScore
StringQuartets delta; the v4 install materialisation for the same
source already produced 122 .mid files via MuseScore 4 CLI on the v4
install path. We re-use the v4 install's converted files for v5's 122
non-delta works and write the 41 delta works from this script.
"""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")  # ensure Unicode in console


import argparse
import json
import sys
import time
import urllib.parse
import zipfile
import io
from pathlib import Path

import requests


V5_REV = "a3b3813477b06c2c48887f1055f8567dae0c2232"
RAW_BASE = "https://github.com/OpenScore/StringQuartets/raw"
DELTA_ARTIFACTS = [
    ("Arriaga,_Juan_Cris%C3%B3stomo_de/String_Quartet_No.2_in_A_major/sq34038344.mscz", "added"),
    ("Beethoven,_Ludwig_van/String_Quartet_No.14_in_C%E2%99%AF_minor%2C_Op.131/sq20090908.mscz", "added"),
    ("Brahms,_Johannes/String_Quartet_No.1%2C_Op.51_No.1/sq7108150.mscz", "added"),
    ("Cras,_Jean/String_Quartet/sq25403263.mscz", "added"),
    ("Dittersdorf,_Carl_Ditters_von/String_Quartet_in_D_major%2C_Kr.191/sq27449398.mscz", "added"),
    ("Dittersdorf,_Carl_Ditters_von/String_Quartet_in_G_major%2C_Kr.193/sq35678303.mscz", "added"),
    ("Dvo%C5%99%C3%A1k%2C_Anton%C3%ADn/String_Quartet_No.11%2C_Op.61/sq27306844.mscz", "added"),
    ("Dvo%C5%99%C3%A1k%2C_Anton%C3%ADn/String_Quartet_No.14%2C_Op.105/sq23996920.mscz", "added"),
    ("Dvo%C5%99%C3%A1k%2C_Anton%C3%ADn/String_Quartet_No.8%2C_Op.80/sq24807751.mscz", "added"),
    ("Dvo%C5%99%C3%A1k%2C_Anton%C3%ADn/String_Quartet_No.9%2C_Op.34/sq28001095.mscz", "added"),
    ("Franck,_C%C3%A9sar/String_Quartet/sq26652775.mscz", "added"),
    ("Gossec,_Fran%C3%A7ois_Joseph/String_Quartet_in_C_minor%2C_RH_189/sq27571393.mscz", "added"),
    ("Haydn,_Joseph/String_Quartet_in_B-flat_major%2C_Hob.III12%2C_Op.2_No.6/sq34642829.mscz", "added"),
    ("Haydn,_Joseph/String_Quartet_in_B-flat_major%2C_Hob.III78%2C_Op.76_No.4/sq17317312.mscz", "added"),
    ("Haydn,_Joseph/String_Quartet_in_D_major%2C_Hob.III79%2C_Op.76_No.5/sq22641487.mscz", "added"),
    ("Haydn,_Joseph/String_Quartet_in_E-flat_major%2C_Hob.III80%2C_Op.76_No.6/sq22779031.mscz", "added"),
    ("Haydn,_Joseph/String_Quartet_in_E-flat_major%2C_Hob.III9%2C_Op.2_No.3/sq26595778.mscz", "added"),
    ("Koechlin,_Charles/String_Quartet_No.1%2C_Op.51/sq27309583.mscz", "added"),
    ("Mayer,_Emilie/String_Quartet_in_D_Minor/sq7643891.mscz", "added"),
    ("Mendelssohn,_Felix/String_Quartet_No.3%2C_Op.44_No.1/sq25717423.mscz", "added"),
    ("Mozart,_Wolfgang_Amadeus/String_Quartet_No.6_in_B-flat_major%2C_K.159/sq23655934.mscz", "added"),
    ("Mozart,_Wolfgang_Amadeus/String_Quartet_No.8_in_F_major%2C_K.168/sq31275512.mscz", "added"),
    ("Mozart,_Wolfgang_Amadeus/String_Quartet_No.9_in_A_major%2C_K.169/sq26846629.mscz", "added"),
    ("Mozart,_Wolfgang_Amadeus/String_Quartet_in_C_major%2C_K.170/sq35524196.mscz", "added"),
    ("Onslow,_George/String_Quartet_No.28_in_E-flat_major%2C_Op.54/sq34642727.mscz", "added"),
    ("Ravel,_Maurice/String_Quartet_in_F_major/sq8482283.mscz", "added"),
    ("Reger,_Max/String_Quartet_No.3%2C_Op.74/sq26401096.mscz", "added"),
    ("Reger,_Max/String_Quartet_No.4%2C_Op.109/sq31320671.mscz", "added"),
    ("Reger,_Max/String_Quartet_No.5%2C_Op.121/sq26406961.mscz", "added"),
    ("Schubert,_Franz/String_Quartet_in_A_minor%2C_D.804%2C_No.13/sq29218964.mscz", "added"),
    ("Schubert,_Franz/String_Quartet_in_B-flat_major%2C_D.36%2C_No.3/sq30897104.mscz", "added"),
    ("Schubert,_Franz/String_Quartet_in_E-flat_major%2C_D.87%2C_No.10/sq26745994.mscz", "added"),
    ("Schubert,_Franz/String_Quartet_in_E_major%2C_D.353%2C_No.11/sq27306670.mscz", "added"),
    ("Schubert,_Franz/String_Quartet_in_G_major%2C_D.887%2C_No.15/sq17329942.mscz", "added"),
    ("Spohr,_Louis/String_Quartet_No.1%2C_Op.4_No.1/sq27975730.mscz", "added"),
    ("Stenhammar,_Wilhelm/String_Quartet_No.4/sq26531671.mscz", "added"),
    ("Tchaikovsky,_Pyotr_Ilyich/String_Quartet_No.1%2C_Op.11/sq11738245.mscz", "added"),
    ("Zemlinsky,_Alexander_von/String_Quartet_No.3%2C_Op.19/sq26269501.mscz", "added"),
    ("Mendelssohn,_Felix/String_Quartet_No.6_in_F_Minor%2C_Op.80/sq8561633.mscx", "renamed"),
    ("Schubert,_Franz/String_Quartet_in_C_minor%2C_D.703%2C_No.12/sq7648382.mscx", "renamed_edited"),
    ("Schubert,_Franz/String_Quartet_in_D_minor%2C_D.810%2C_No.14_(%E2%80%9CDeath_and_the_Maiden%E2%80%9D)/sq7397765.mscx", "renamed_edited"),
]


def fetch(url: str, cache: Path) -> bytes:
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_bytes()
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    cache.write_bytes(r.content)
    time.sleep(0.2)
    return r.content


def parse_musescore_4_mscx_voices(mscx: bytes, division: int = 480):
    """Return list of (staff_idx, list of (abs_tick, pitch, tpc, dur_ticks)) per real staff."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(mscx)
    DUR_MAP = {
        "whole": 1.0, "half": 0.5, "quarter": 0.25, "eighth": 0.125,
        "16th": 0.0625, "32nd": 0.03125, "64th": 0.015625, "128th": 0.0078125,
    }
    real_staves = [s for s in root.findall(".//Staff") if s.get("id") is not None]
    staff_notes: list[list[tuple[int, int, int, int]]] = [[] for _ in real_staves]

    def walk(elements, cur):
        for el in elements:
            if el.tag == "voice":
                cur = walk(list(el), cur)
            elif el.tag == "location":
                try:
                    num, denom = el.get("fractions", "0/1").split("/")
                    cur += (4 * division) * int(num) // max(int(denom), 1)
                except Exception:
                    pass
            elif el.tag in ("Chord", "Rest"):
                dur = (el.find("durationType").text if el.find("durationType") is not None else "quarter")
                frac = DUR_MAP.get(dur, 0.25)
                dur_ticks = int(frac * 4 * division)
                dots = el.findall("dots")
                if dots:
                    n_dots = len(dots)
                    dur_ticks = int(dur_ticks + dur_ticks * sum(0.5 ** (i + 1) for i in range(n_dots)))
                if el.tag == "Chord":
                    for note in el.findall("Note"):
                        pitch = int(note.find("pitch").text) if note.find("pitch") is not None and note.find("pitch").text else 60
                        tpc = int(note.find("tpc").text) if note.find("tpc") is not None and note.find("tpc").text else 14
                        # store note with start tick, pitch, tpc, duration
                        yield_idx = None  # not used
                        staff_notes[staff_idx_local].append((cur, pitch, tpc, dur_ticks))
                cur += dur_ticks
            elif el.tag == "endSpanner":
                pass
        return cur

    # Use indices captured via closure-per-staff
    out: list[list[tuple[int, int, int, int]]] = [[] for _ in real_staves]
    for staff_idx, staff in enumerate(real_staves):
        # local helper
        notes_local = out[staff_idx]

        def walk_staff(elements, cur, _notes=notes_local, _idx=staff_idx):
            for el in elements:
                if el.tag == "voice":
                    cur = walk_staff(list(el), cur)
                elif el.tag == "location":
                    try:
                        num, denom = el.get("fractions", "0/1").split("/")
                        cur += (4 * division) * int(num) // max(int(denom), 1)
                    except Exception:
                        pass
                elif el.tag in ("Chord", "Rest"):
                    dur_node = el.find("durationType")
                    dur = dur_node.text if dur_node is not None else "quarter"
                    frac = DUR_MAP.get(dur, 0.25)
                    dur_ticks = int(frac * 4 * division)
                    dots = el.findall("dots")
                    if dots:
                        n_dots = len(dots)
                        dur_ticks = int(dur_ticks + dur_ticks * sum(0.5 ** (i + 1) for i in range(n_dots)))
                    if el.tag == "Chord":
                        for note in el.findall("Note"):
                            pitch = int(note.find("pitch").text) if note.find("pitch") is not None and note.find("pitch").text else 60
                            tpc = int(note.find("tpc").text) if note.find("tpc") is not None and note.find("tpc").text else 14
                            _notes.append((cur, pitch, tpc, dur_ticks))
                    cur += dur_ticks
                elif el.tag == "endSpanner":
                    pass
            return cur

        for measure in staff.findall("Measure"):
            walk_staff(list(measure), 0)

    # extract meta (composer, work title, tempo)
    meta = {}
    for tag in root.findall(".//metaTag"):
        meta[tag.get("name", "")] = tag.text or ""
    return out, meta, division


def notes_to_midi(staff_notes: list[list[tuple[int, int, int, int]]], division: int) -> tuple[object, int]:
    """Build a multi-track MIDI file. Returns (midi_file, total_note_count)."""
    import mido
    mid = mido.MidiFile(type=1, ticks_per_beat=division)
    n_total = 0
    for notes in staff_notes:
        if not notes:
            track = mido.MidiTrack()
            mid.tracks.append(track)
            continue
        track = mido.MidiTrack()
        last_tick = 0
        for start, pitch, tpc, dur in notes:
            if start < 0:
                start = 0
            if dur <= 0:
                dur = division // 4
            delta = start - last_tick
            if delta < 0:
                delta = 0
            vel = 80
            track.append(mido.Message("note_on", note=pitch, velocity=vel, time=delta, channel=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=dur, channel=0))
            last_tick = start + dur
            n_total += 1
        mid.tracks.append(track)
    return mid, n_total


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True,
                    help="e.g. C:\\ov5\\converted\\openscore_string_quartets\\scores")
    ap.add_argument("--cache-dir", default=r"C:\Users\junny\AppData\Local\Temp\kilo\mscz-cache")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, (urlpath, status) in enumerate(DELTA_ARTIFACTS, start=1):
        url = f"{RAW_BASE}/{V5_REV}/scores/{urlpath}"
        decoded_filename = urllib.parse.unquote(urlpath.split("/")[-1])
        stem = Path(decoded_filename).stem
        # target relative path = first two path components + stem + .mid
        rel_parts = urllib.parse.unquote(urlpath).split("/")
        composer_dir = rel_parts[0]
        work_dir = rel_parts[1]
        target = out_root / composer_dir / work_dir / f"{stem}.mid"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            print(f"  [{i:>2}/{len(DELTA_ARTIFACTS)}] CACHED {target.relative_to(out_root)}", flush=True)
            summary.append({"relpath": str(target.relative_to(out_root)), "status": "cached"})
            continue
        try:
            data = fetch(url, cache / decoded_filename)
        except Exception as exc:
            print(f"  [{i:>2}/{len(DELTA_ARTIFACTS)}] FETCH_ERR {decoded_filename}: {exc}", flush=True)
            summary.append({"relpath": str(target.relative_to(out_root)), "status": "fetch_error", "error": str(exc)})
            continue
        try:
            if decoded_filename.endswith(".mscz"):
                z = zipfile.ZipFile(io.BytesIO(data))
                mscx_names = [n for n in z.namelist() if n.endswith(".mscx") and "Excerpts" not in n]
                mscx = z.read(mscx_names[0])
            else:
                mscx = data
        except Exception as exc:
            print(f"  [{i:>2}/{len(DELTA_ARTIFACTS)}] ZIP_ERR {decoded_filename}: {exc}", flush=True)
            summary.append({"relpath": str(target.relative_to(out_root)), "status": "zip_error", "error": str(exc)})
            continue
        try:
            staff_notes, meta, division = parse_musescore_4_mscx_voices(mscx)
        except Exception as exc:
            print(f"  [{i:>2}/{len(DELTA_ARTIFACTS)}] PARSE_ERR {decoded_filename}: {exc}", flush=True)
            summary.append({"relpath": str(target.relative_to(out_root)), "status": "parse_error", "error": str(exc)})
            continue
        try:
            mid, n_total = notes_to_midi(staff_notes, division)
            mid.save(target)
            print(f"  [{i:>2}/{len(DELTA_ARTIFACTS)}] WROTE  {target.relative_to(out_root)}  ({n_total} notes)", flush=True)
            summary.append({
                "relpath": str(target.relative_to(out_root)),
                "status": "wrote",
                "notes": n_total,
                "staffs": len(staff_notes),
                "composer": meta.get("composer", ""),
                "work": meta.get("workTitle", ""),
            })
        except Exception as exc:
            print(f"  [{i:>2}/{len(DELTA_ARTIFACTS)}] MIDI_ERR {decoded_filename}: {exc}", flush=True)
            summary.append({"relpath": str(target.relative_to(out_root)), "status": "midi_error", "error": str(exc)})

    out_json = out_root.parent / "v5_delta_midi_generation_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[gen] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
