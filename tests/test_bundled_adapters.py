from __future__ import annotations

import json
from pathlib import Path

from orbitune.adapter import load_manifest, validate_manifest


def test_bundled_adapters_are_complete_and_registered() -> None:
    registry = json.loads(Path("registry/adapters.json").read_text(encoding="utf-8"))
    registered = {item["id"] for item in registry["adapters"]}

    discovered: set[str] = set()
    for parent in (Path("adapters/official"), Path("adapters/community")):
        for directory in sorted(path for path in parent.iterdir() if path.is_dir()):
            adapter_id = directory.name
            discovered.add(adapter_id)
            required = [directory / "manifest.json", directory / "adapter.safetensors", directory / "demo.mid", directory / "README.md"]
            missing = [path.name for path in required if not path.exists()]
            assert not missing, f"{adapter_id} is missing: {missing}"

            manifest = load_manifest(directory / "manifest.json")
            errors = validate_manifest(manifest)
            assert not errors, f"{adapter_id} manifest errors: {errors}"
            assert manifest["name"] == adapter_id
            assert (directory / "adapter.safetensors").stat().st_size < 5 * 1024 * 1024

    assert discovered == registered
