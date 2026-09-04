"""Re-summarize the v5 delta MIDI generation by scanning C:\\ov5\\converted\\openscore_string_quartets\\scores and using the source mscx metadata already downloaded into the census cache.

Updates the existing v5_delta_midi_generation_summary.json so the
status reflects actual filesystem state, not the prior run's print errors.
"""
import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

OUT_ROOT = Path(r"C:\ov5\converted\openscore_string_quartets\scores")
SUMMARY = Path(r"C:\ov5\converted\openscore_string_quartets\v5_delta_midi_generation_summary.json")
CACHE = Path(r"C:\Users\junny\AppData\Local\Temp\kilo\mscz-cache")
RAW_BASE = "https://github.com/OpenScore/StringQuartets/raw"
V5_REV = "a3b3813477b06c2c48887f1055f8567dae0c2232"

import zipfile
import io
import xml.etree.ElementTree as ET

DELTA = [
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


def fetch_meta(decoded_filename: str) -> dict:
    p = CACHE / decoded_filename
    if not p.exists():
        return {}
    try:
        if decoded_filename.endswith(".mscz"):
            z = zipfile.ZipFile(p)
            mscx_names = [n for n in z.namelist() if n.endswith(".mscx") and "Excerpts" not in n]
            data = z.read(mscx_names[0])
        else:
            data = p.read_bytes()
        root = ET.fromstring(data)
        meta = {}
        for tag in root.findall(".//metaTag"):
            meta[tag.get("name", "")] = tag.text or ""
        return meta
    except Exception:
        return {}


def main():
    import urllib.parse
    results = []
    for urlpath, status in DELTA:
        decoded_filename = urllib.parse.unquote(urlpath.split("/")[-1])
        stem = Path(decoded_filename).stem
        rel_parts = urllib.parse.unquote(urlpath).split("/")
        composer_dir = rel_parts[0]
        work_dir = rel_parts[1]
        target = OUT_ROOT / composer_dir / work_dir / f"{stem}.mid"
        if target.exists() and target.stat().st_size > 0:
            meta = fetch_meta(decoded_filename)
            results.append({
                "relpath": str(target.relative_to(OUT_ROOT)),
                "status": "wrote",
                "bytes": target.stat().st_size,
                "composer": meta.get("composer", ""),
                "work": meta.get("workTitle", ""),
                "movement_state": status,
            })
        else:
            results.append({
                "relpath": str(target.relative_to(OUT_ROOT)) if target.exists() else f"missing: {target}",
                "status": "missing",
            })
    SUMMARY.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"[resummary] wrote {SUMMARY}")
    print(f"[resummary] counts: {counts}")


if __name__ == "__main__":
    main()
