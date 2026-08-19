from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from orbitune.adapter import validate_adapter_directory
from orbitune.compat import BASE_MODEL_ID, TOKENIZER_ABI


def discover_adapter_directories(root: str | Path = "adapters") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    directories: list[Path] = []
    for manifest in root.rglob("manifest.json"):
        if manifest.parent.name.startswith("."):
            continue
        directories.append(manifest.parent)
    return sorted(directories)


def build_registry(root: str | Path = "adapters") -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_hashes: set[str] = set()
    root = Path(root)
    for directory in discover_adapter_directories(root):
        manifest = validate_adapter_directory(directory)
        adapter_id = str(manifest["name"])
        if adapter_id in seen:
            raise ValueError(f"duplicate adapter id: {adapter_id}")
        seen.add(adapter_id)
        base_hashes.add(str(manifest["base_sha256"]).lower())
        relative = directory.relative_to(root)
        source = relative.parts[0] if relative.parts else "community"
        entries.append(
            {
                "id": adapter_id,
                "display_name": manifest["display_name"],
                "description": manifest.get("description", ""),
                "family": manifest.get("adapter_family", "style"),
                "source": source,
                "license": manifest["license"],
                "base_sha256": manifest["base_sha256"],
                "generation_defaults": manifest["generation_defaults"],
                "tags": manifest.get("tags", []),
                "adapter_url": f"./adapters/{relative.as_posix()}/adapter.safetensors",
                "demo_url": f"./adapters/{relative.as_posix()}/demo.mid",
            }
        )
    if len(base_hashes) > 1:
        raise ValueError("bundled adapters target multiple Base checkpoints; Orbitune registry requires one immutable Base")
    entries.sort(key=lambda item: (item["source"] != "official", item["display_name"].lower(), item["id"]))
    return {
        "schema_version": "0.1.0",
        "base_model": BASE_MODEL_ID,
        "base_sha256": next(iter(base_hashes), ""),
        "tokenizer": TOKENIZER_ABI,
        "adapters": entries,
    }


def write_registry(out: str | Path, root: str | Path = "adapters") -> dict[str, Any]:
    registry = build_registry(root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return registry


def build_web_adapter_assets(web_root: str | Path = "web", adapter_root: str | Path = "adapters") -> dict[str, Any]:
    web_root = Path(web_root)
    adapter_root = Path(adapter_root)
    registry = write_registry(web_root / "data" / "adapters.json", adapter_root)
    destination_root = web_root / "adapters"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    for directory in discover_adapter_directories(adapter_root):
        relative = directory.relative_to(adapter_root)
        destination = destination_root / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("adapter.safetensors", "demo.mid", "manifest.json"):
            shutil.copy2(directory / name, destination / name)
    return registry
