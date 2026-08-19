from orbitune.adapter import validate_manifest


def test_valid_adapter_manifest_passes():
    manifest = {
        "artifact_type": "orbitune_adapter",
        "name": "chill-piano-v0",
        "version": "0.1.0",
        "base_model": "orbitune-tiny-v0",
        "architecture": "orbitune-midi-gpt-v0",
        "tokenizer": "theory-remi-v0",
        "adapter_type": "lora",
        "rank": 4,
        "target_modules": ["q_proj", "v_proj"],
        "license": "Apache-2.0",
        "training_data": {"source_type": "synthetic", "license": "Apache-2.0"},
        "generation_defaults": {"bpm": 84, "bars": 8, "temperature": 0.85},
    }
    assert validate_manifest(manifest) == []


def test_wrong_base_model_fails():
    manifest = {
        "artifact_type": "orbitune_adapter",
        "name": "bad-v0",
        "version": "0.1.0",
        "base_model": "other-base",
        "architecture": "orbitune-midi-gpt-v0",
        "tokenizer": "theory-remi-v0",
        "adapter_type": "lora",
        "rank": 4,
        "target_modules": ["q_proj"],
        "license": "Apache-2.0",
        "training_data": {},
        "generation_defaults": {"temperature": 0.85, "bars": 8},
    }
    assert any("base_model" in error for error in validate_manifest(manifest))
