from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from orbitune.adapter import validate_adapter_directory
from orbitune.base_registry import build_base_registry, discover_base_directories


def discover_adapter_directories(root: str | Path = "adapters") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path.parent for path in root.rglob("manifest.json") if not path.parent.name.startswith("."))


def build_registry(root: str | Path = "adapters", base_root: str | Path | None = None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    root = Path(root)
    base_by_id: dict[str, dict[str, Any]] = {}
    if base_root is not None:
        base_registry = build_base_registry(base_root)
        base_by_id = {item["id"]: item for item in base_registry["bases"]}

    for directory in discover_adapter_directories(root):
        manifest = validate_adapter_directory(directory)
        adapter_id = str(manifest["name"])
        if adapter_id in seen:
            raise ValueError(f"duplicate adapter id: {adapter_id}")
        seen.add(adapter_id)
        if base_root is not None:
            base = base_by_id.get(manifest["base_model"])
            if base is None:
                raise ValueError(f"adapter {adapter_id} references unknown Base {manifest['base_model']}")
            if base["checkpoint_sha256"].lower() != manifest["base_sha256"].lower():
                raise ValueError(f"adapter {adapter_id} base_sha256 does not match Base {manifest['base_model']}")
            if base["architecture"] != manifest["architecture"] or base["tokenizer"] != manifest["tokenizer"]:
                raise ValueError(f"adapter {adapter_id} ABI does not match Base {manifest['base_model']}")
        relative = directory.relative_to(root)
        source = relative.parts[0] if relative.parts else "community"
        entries.append({
            "id": adapter_id,
            "display_name": manifest["display_name"],
            "description": manifest.get("description", ""),
            "family": manifest.get("adapter_family", "style"),
            "source": source,
            "base_model": manifest["base_model"],
            "base_sha256": manifest["base_sha256"],
            "license": manifest["license"],
            "generation_defaults": manifest["generation_defaults"],
            "tags": manifest.get("tags", []),
            "adapter_url": f"./adapters/{relative.as_posix()}/adapter.safetensors",
            "demo_url": f"./adapters/{relative.as_posix()}/demo.mid",
        })
    entries.sort(key=lambda item: (item["source"] != "official", item["display_name"].lower(), item["id"]))
    return {"schema_version": "0.2.0", "adapters": entries}


def write_registry(out: str | Path, root: str | Path = "adapters", base_root: str | Path | None = None) -> dict[str, Any]:
    registry = build_registry(root, base_root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return registry


def write_registries(adapter_out: str | Path, base_out: str | Path, adapter_root: str | Path = "adapters", base_root: str | Path = "bases") -> tuple[dict[str, Any], dict[str, Any]]:
    bases = build_base_registry(base_root)
    adapters = build_registry(adapter_root, base_root)
    for out, payload in ((Path(base_out), bases), (Path(adapter_out), adapters)):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return bases, adapters


def build_web_assets(web_root: str | Path = "web", adapter_root: str | Path = "adapters", base_root: str | Path = "bases") -> tuple[dict[str, Any], dict[str, Any]]:
    web_root = Path(web_root); adapter_root = Path(adapter_root); base_root = Path(base_root)
    bases, adapters = write_registries(web_root / "data" / "adapters.json", web_root / "data" / "bases.json", adapter_root, base_root)
    adapter_dest = web_root / "adapters"; base_dest = web_root / "bases"
    if adapter_dest.exists(): shutil.rmtree(adapter_dest)
    if base_dest.exists(): shutil.rmtree(base_dest)
    for directory in discover_adapter_directories(adapter_root):
        relative = directory.relative_to(adapter_root); dest = adapter_dest / relative; dest.mkdir(parents=True, exist_ok=True)
        for name in ("adapter.safetensors", "demo.mid", "manifest.json"):
            shutil.copy2(directory / name, dest / name)
    for directory in discover_base_directories(base_root):
        dest = base_dest / directory.name; dest.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name in ("manifest.json", manifest["web_onnx"]["filename"]):
            shutil.copy2(directory / name, dest / name)
    return bases, adapters


def build_web_adapter_assets(web_root: str | Path = "web", adapter_root: str | Path = "adapters") -> dict[str, Any]:
    # Legacy wrapper retained for tests/tools that have not migrated to Base registry generation.
    registry = write_registry(Path(web_root) / "data" / "adapters.json", adapter_root)
    destination_root = Path(web_root) / "adapters"
    if destination_root.exists(): shutil.rmtree(destination_root)
    for directory in discover_adapter_directories(adapter_root):
        relative = directory.relative_to(Path(adapter_root)); destination = destination_root / relative; destination.mkdir(parents=True, exist_ok=True)
        for name in ("adapter.safetensors", "demo.mid", "manifest.json"):
            shutil.copy2(directory / name, destination / name)
    return registry
