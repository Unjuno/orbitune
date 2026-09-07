from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publication_check", ROOT / "scripts/check_publication.py")
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def _write(root: Path, rel: str, content: bytes = b"safe text\n") -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _schema_record():
    schema = json.loads((ROOT / "schemas/research_publication.schema.json").read_text())
    record = json.loads((ROOT / CHECK.MODEL_DIR / "publication.json").read_text())
    return schema, record


def test_publication_schema_accepts_documentation_only_record():
    schema, record = _schema_record()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(record)


@pytest.mark.parametrize("change", ["download_url", "commercial", "registry", "claims_exact_resume"])
def test_publication_schema_rejects_premature_publication(change):
    schema, record = _schema_record()
    record = copy.deepcopy(record)
    if change == "download_url":
        record["checkpoint"]["download_url"] = "https://example.invalid/model.pt"
    elif change == "commercial":
        record["lineage"]["commercial_eligible"] = True
    elif change == "registry":
        record["public_registry_eligible"] = True
    else:
        record["reported_training"]["bit_exact_continuation"] = True
    assert list(Draft202012Validator(schema).iter_errors(record))


@pytest.mark.parametrize("rel", [".env", ".env.local", ".huggingface/token", "id_rsa", "private.key"])
def test_blocks_credential_paths(tmp_path, rel):
    _write(tmp_path, rel)
    assert CHECK.scan_paths(tmp_path, [rel])


def test_allows_reviewed_env_example(tmp_path):
    _write(tmp_path, ".env.example", b"HF_TOKEN=\n")
    assert CHECK.scan_paths(tmp_path, [".env.example"]) == []


@pytest.mark.parametrize("rel", ["runs/train.json", "raw/data.mid", "indexes/a.json", "data/train.parquet", "models/x/model.pt"])
def test_blocks_local_artifacts(tmp_path, rel):
    _write(tmp_path, rel)
    assert CHECK.scan_paths(tmp_path, [rel])


def test_preserves_legacy_adapter_artifact_path(tmp_path):
    _write(tmp_path, "adapters/official/demo/adapter.safetensors", b"fixture")
    assert CHECK.scan_paths(tmp_path, ["adapters/official/demo/adapter.safetensors"]) == []


def test_credential_detection_redacts_value(tmp_path):
    token = b"hf_" + b"X" * 40
    _write(tmp_path, "accident.txt", token)
    errors = CHECK.scan_paths(tmp_path, ["accident.txt"])
    assert errors and token.decode() not in str(errors)


def test_credential_detection_crosses_buffer_boundary(tmp_path):
    token = b"ghp_" + b"X" * 40
    _write(tmp_path, "accident.txt", b" " * (1024 * 1024 - 6) + token)
    assert CHECK.scan_paths(tmp_path, ["accident.txt"])


def test_docs_detect_broken_link_and_personal_home(tmp_path):
    _write(tmp_path, "README.md", ("[missing](missing.md)\nC:" + "\\Users\\example\\repo\n").encode())
    errors = CHECK.check_doc_links(tmp_path, ("README.md",))
    assert len(errors) == 2


def test_docs_accept_existing_links_and_external_urls(tmp_path):
    _write(tmp_path, "README.md", b"[local](docs/a.md#heading) [web](https://example.invalid/)\n")
    _write(tmp_path, "docs/a.md")
    assert CHECK.check_doc_links(tmp_path, ["README.md"]) == []


def test_repository_publication_record_and_links():
    assert CHECK.validate_publication(ROOT) == []
    assert CHECK.check_doc_links(ROOT) == []


def test_repository_generation_examples_use_implemented_flags():
    tree = ast.parse((ROOT / "orbitune/compound_cli.py").read_text(encoding="utf-8"))
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id == "generate":
            flags.update(arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    required = {"--checkpoint", "--out", "--events", "--device", "--temperature", "--top-p", "--primer-midi"}
    assert required <= flags
    text = (ROOT / CHECK.MODEL_DIR / "usage.md").read_text()
    commands = [line for line in text.splitlines() if line.startswith("orbitune-compound generate ")]
    assert len(commands) == 2
    for line in commands:
        used = {part for part in line.split() if part.startswith("--")}
        assert used <= flags
        assert "--seed" not in used


def test_preserves_existing_placeholder_but_still_scans_it(tmp_path):
    _write(tmp_path, "runs/compound/.gitkeep", b"# local outputs stay ignored\n")
    assert CHECK.scan_paths(tmp_path, ["runs/compound/.gitkeep"]) == []
    _write(tmp_path, "runs/compound/.gitkeep", b"hf_" + b"Y" * 40)
    assert CHECK.scan_paths(tmp_path, ["runs/compound/.gitkeep"])
