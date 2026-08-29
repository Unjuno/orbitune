from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse


MANIFEST = (
    Path(__file__).parents[1]
    / "experiments"
    / "data"
    / "manifests"
    / "mutopia_public_domain_smoke_v1.json"
)


def test_mutopia_public_domain_smoke_manifest_is_strict_allowlist() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["license_policy"].startswith("include only")
    items = payload["items"]
    assert len(items) >= 5
    ids = set()
    pages = set()
    midi_urls = set()
    for item in items:
        assert item["license"] == "Public Domain"
        assert item["id"] not in ids
        assert item["piece_page"] not in pages
        assert item["midi_url"] not in midi_urls
        ids.add(item["id"])
        pages.add(item["piece_page"])
        midi_urls.add(item["midi_url"])
        for key in ("piece_page", "midi_url"):
            parsed = urlparse(item[key])
            assert parsed.scheme == "https"
            assert parsed.hostname == "www.mutopiaproject.org"
        assert item["piece_page"].startswith(
            "https://www.mutopiaproject.org/cgibin/piece-info.cgi?id="
        )
        assert item["midi_url"].endswith(".mid")
        assert item["mutopia_music_id"].startswith("Mutopia-")
