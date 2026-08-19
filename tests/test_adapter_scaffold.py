from orbitune.adapter import create_adapter_scaffold, load_manifest, validate_manifest


def test_adapter_scaffold_requires_base_hash_and_rights_confirmation(tmp_path):
    root = create_adapter_scaffold(
        tmp_path / "chill-piano-v0",
        name="chill-piano-v0",
        display_name="Chill Piano",
    )
    manifest = load_manifest(root / "manifest.json")
    errors = validate_manifest(manifest)
    assert (root / "README.md").exists()
    assert manifest["base_model"] == "orbitune-base"
    assert manifest["rank"] == 4
    assert any("base_sha256" in error for error in errors)
    assert any("rights_confirmed" in error for error in errors)
