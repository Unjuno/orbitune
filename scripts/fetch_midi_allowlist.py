from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ALLOWED_LICENSES = {"Public Domain"}
DEFAULT_ALLOWED_HOSTS = {"www.mutopiaproject.org"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_allowlist(
    manifest_path: str | Path,
    output_dir: str | Path,
    provenance_out: str | Path,
    *,
    timeout_seconds: int = 30,
) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported allowlist schema")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("allowlist has no items")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("allowlist item must be an object")
        item_id = str(item.get("id", ""))
        if not item_id or item_id in seen_ids:
            raise ValueError(f"invalid or duplicate allowlist id: {item_id!r}")
        seen_ids.add(item_id)
        license_name = str(item.get("license", ""))
        if license_name not in ALLOWED_LICENSES:
            raise ValueError(f"license not allowed for smoke corpus: {license_name!r}")
        midi_url = str(item.get("midi_url", ""))
        page_url = str(item.get("piece_page", ""))
        for url in (midi_url, page_url):
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in DEFAULT_ALLOWED_HOSTS:
                raise ValueError(f"URL outside strict allowlist host: {url}")
        if not midi_url.lower().endswith(".mid"):
            raise ValueError(f"MIDI URL does not end in .mid: {midi_url}")

        request = urllib.request.Request(
            midi_url,
            headers={"User-Agent": "Orbitune-Research-Smoke/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
        if len(payload) < 14 or not payload.startswith(b"MThd"):
            raise ValueError(f"downloaded payload is not a Standard MIDI File: {item_id}")

        filename = f"{item_id}.mid"
        path = destination / filename
        path.write_bytes(payload)
        records.append(
            {
                "id": item_id,
                "title": item.get("title"),
                "composer": item.get("composer"),
                "license": license_name,
                "piece_page": page_url,
                "midi_url": midi_url,
                "mutopia_music_id": item.get("mutopia_music_id"),
                "filename": filename,
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "content_type": content_type,
            }
        )

    report = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "corpus_name": manifest.get("name"),
        "intended_use": manifest.get("intended_use"),
        "files": records,
    }
    target = Path(provenance_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a strict provenance MIDI allowlist")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provenance-out", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    report = fetch_allowlist(
        args.manifest,
        args.output_dir,
        args.provenance_out,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
