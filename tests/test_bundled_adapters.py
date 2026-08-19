from __future__ import annotations

from pathlib import Path

from orbitune.adapter import validate_adapter_directory
from orbitune.base_registry import build_base_registry
from orbitune.registry import build_registry, discover_adapter_directories


def test_bundled_assets_are_fully_valid_and_dependency_graph_builds() -> None:
    bases = build_base_registry("bases")
    known_bases = {item["id"]: item for item in bases["bases"]}
    discovered = discover_adapter_directories("adapters")
    ids: set[str] = set()
    for directory in discovered:
        manifest = validate_adapter_directory(directory)
        adapter_id = manifest["name"]
        assert directory.name == adapter_id
        assert adapter_id not in ids
        ids.add(adapter_id)
        assert manifest["base_model"] in known_bases
        assert manifest["base_sha256"].lower() == known_bases[manifest["base_model"]]["checkpoint_sha256"].lower()
        total_size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        assert total_size <= 5 * 1024 * 1024, f"{adapter_id} exceeds 5 MiB"

    registry = build_registry("adapters", "bases")
    assert {item["id"] for item in registry["adapters"]} == ids


def test_only_official_and_community_adapter_roots_are_used() -> None:
    allowed = {"official", "community"}
    for directory in discover_adapter_directories("adapters"):
        relative = directory.relative_to(Path("adapters"))
        assert relative.parts and relative.parts[0] in allowed
