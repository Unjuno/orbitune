from __future__ import annotations

from pathlib import Path

from orbitune.adapter import validate_adapter_directory
from orbitune.registry import build_registry, discover_adapter_directories


def test_bundled_adapters_are_fully_valid_and_registry_builds() -> None:
    discovered = discover_adapter_directories("adapters")
    ids: set[str] = set()
    for directory in discovered:
        manifest = validate_adapter_directory(directory)
        adapter_id = manifest["name"]
        assert directory.name == adapter_id
        assert adapter_id not in ids
        ids.add(adapter_id)
        total_size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        assert total_size <= 5 * 1024 * 1024, f"{adapter_id} exceeds 5 MiB"

    registry = build_registry("adapters")
    assert {item["id"] for item in registry["adapters"]} == ids
    assert registry["base_model"] == "orbitune-tiny-v0"
    assert registry["tokenizer"] == "theory-remi-v0"


def test_only_official_and_community_adapter_roots_are_used() -> None:
    allowed = {"official", "community"}
    for directory in discover_adapter_directories("adapters"):
        relative = directory.relative_to(Path("adapters"))
        assert relative.parts and relative.parts[0] in allowed
