from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/research_nc_aria_gigamidi_a2_512_v1"
CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/ROADMAP.md",
    ROOT / "docs/HANDOFF.md",
    ROOT / "docs/PUBLICATION.md",
    ROOT / "docs/COMPOUND_LORA_POLICY.md",
)


def test_a2_release_manifest_is_canonical_and_complete() -> None:
    schema = json.loads((ROOT / "schemas/a2_release.schema.json").read_text(encoding="utf-8"))
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    assert manifest["checkpoint"]["sha256"] == "e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0"
    assert manifest["lineage"]["source_commit"] == "8489870f81a1591515a98e58554e533fcac9d095"


def test_current_docs_reference_canonical_a2_release() -> None:
    canonical = "models/research_nc_aria_gigamidi_a2_512_v1/manifest.json"
    for path in CURRENT_DOCS:
        text = path.read_text(encoding="utf-8")
        relative = canonical if path.parent == ROOT else f"../{canonical}"
        assert relative in text, f"{path.relative_to(ROOT)} does not reference the canonical release manifest"


def test_current_docs_do_not_claim_a2_training_is_active() -> None:
    stale_claims = (
        "active local long-run Base continuation",
        "active local run toward step 1M",
        "continuation toward step 1,000,000",
        "ACTIVE   local long-run full-parameter Base continuation",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in CURRENT_DOCS)
    for claim in stale_claims:
        assert claim not in combined
