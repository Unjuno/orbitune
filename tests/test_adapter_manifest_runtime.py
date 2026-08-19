from orbitune.adapter import validate_manifest


def valid_manifest() -> dict:
    return {
        "artifact_type": "orbitune_adapter",
        "name": "chill-piano-v0",
        "version": "0.1.0",
        "display_name": "Chill Piano",
        "description": "Soft piano BGM tendency.",
        "adapter_family": "style",
        "base_model": "orbitune-base",
        "base_sha256": "a" * 64,
        "architecture": "orbitune-midi-gpt-v0",
        "parameter_scale": "10m",
        "tokenizer": "theory-remi-v0",
        "adapter_type": "lora",
        "rank": 4,
        "target_modules": ["q_proj", "v_proj"],
        "license": "Apache-2.0",
        "training_data": {"source_type": "synthetic", "license": "Apache-2.0", "rights_confirmed": True, "num_files": 4, "num_tokens": 1000},
        "generation_defaults": {"bpm": 84, "bars": 8, "temperature": 0.85},
        "tags": ["piano", "chill"],
    }


def test_valid_adapter_manifest_passes(): assert validate_manifest(valid_manifest()) == []

def test_other_valid_base_id_is_allowed():
    manifest=valid_manifest(); manifest["base_model"]="community-piano-base"; assert validate_manifest(manifest)==[]

def test_invalid_base_id_fails():
    manifest=valid_manifest(); manifest["base_model"]="Bad Base ID"; assert any("base_model" in error for error in validate_manifest(manifest))

def test_exact_base_hash_is_required():
    manifest=valid_manifest(); manifest["base_sha256"]="TODO"; assert any("base_sha256" in error for error in validate_manifest(manifest))

def test_adapter_abi_rejects_wrong_rank_and_target_set():
    manifest=valid_manifest(); manifest["rank"]=8; manifest["target_modules"]=["q_proj"]; errors=validate_manifest(manifest); assert any("rank must be 4" in error for error in errors); assert any("target_modules" in error for error in errors)

def test_rights_confirmation_is_required():
    manifest=valid_manifest(); manifest["training_data"]["rights_confirmed"]=False; assert any("rights_confirmed" in error for error in validate_manifest(manifest))

def test_unknown_manifest_fields_are_rejected():
    manifest=valid_manifest(); manifest["mystery"]=123; assert any("unknown manifest fields" in error for error in validate_manifest(manifest))
