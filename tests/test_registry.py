import hashlib
import json
from pathlib import Path

import pytest

from orbitune.compat import REFERENCE_PARAMETER_COUNT
from orbitune.demo import make_demo_events
from orbitune.midi import write_midi
from orbitune.registry import build_registry, write_registry


def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _write_base(root: Path, base_id: str, checkpoint: bytes = b"base-checkpoint") -> str:
    directory=root/base_id; directory.mkdir(parents=True); onnx=b"base-onnx"; (directory/"model.pt").write_bytes(checkpoint); (directory/"web.onnx").write_bytes(onnx); (directory/"README.md").write_text("# test base\n",encoding="utf-8")
    manifest={"artifact_type":"orbitune_base","id":base_id,"display_name":base_id,"architecture":"orbitune-midi-gpt-v0","tokenizer":"theory-remi-v0","parameter_count":REFERENCE_PARAMETER_COUNT,"checkpoint":{"filename":"model.pt","sha256":_sha(checkpoint),"bytes":len(checkpoint)},"web_onnx":{"filename":"web.onnx","sha256":_sha(onnx),"bytes":len(onnx)},"license":"CC0-1.0","training_data":{"source_type":"synthetic","license":"CC0-1.0","rights_confirmed":True},"lineage":{"parent_checkpoint":None,"commercial_eligible":True,"distribution_scope":"commercial","license_policy":"prod-only","corpus_registry":"configs/test.json","corpus_manifest_sha256":"2"*64,"restricted_source_ids":[],"rights_summary":"PROD-only test corpus"},"tags":["test"]}
    (directory/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8"); return manifest["checkpoint"]["sha256"]


def _write_test_safetensors(path: Path, *, base_sha: str) -> None:
    metadata={"format":"orbitune-lora-v0","base_sha256":base_sha,"rank":"4","alpha":"8.0","dropout":"0.0","target_modules":json.dumps(["q_proj","v_proj"])}; header={"__metadata__":metadata}; offset=0
    for layer in range(4):
        for target in ("q_proj","v_proj"):
            for suffix,shape in (("lora_a",[4,448]),("lora_b",[448,4])):
                size=4*448*4; header[f"blocks.{layer}.attn.{target}.{suffix}"]={"dtype":"F32","shape":shape,"data_offsets":[offset,offset+size]}; offset+=size
    encoded=json.dumps(header,separators=(",",":")).encode("utf-8"); path.write_bytes(len(encoded).to_bytes(8,"little")+encoded+bytes(offset))


def _write_adapter(root: Path, name: str, source: str, *, base_id: str, base_sha: str) -> None:
    directory=root/source/name; directory.mkdir(parents=True)
    manifest={"artifact_type":"orbitune_adapter","name":name,"version":"0.1.0","display_name":name,"adapter_family":"style","base_model":base_id,"base_sha256":base_sha,"architecture":"orbitune-midi-gpt-v0","parameter_scale":"10m","tokenizer":"theory-remi-v0","adapter_type":"lora","rank":4,"target_modules":["q_proj","v_proj"],"generation_defaults":{"bpm":84,"bars":8,"temperature":0.85},"license":"CC0-1.0","training_data":{"source_type":"original","license":"CC0-1.0","rights_confirmed":True},"tags":["test"]}
    (directory/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8"); _write_test_safetensors(directory/"adapter.safetensors",base_sha=base_sha); write_midi(make_demo_events(bars=1),directory/"demo.mid",bpm=84); (directory/"README.md").write_text("# test\n",encoding="utf-8")


def _isolate_registry_linkage_from_checkpoint_abi(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests exercise cross-registry linkage, not checkpoint deserialization.

    Current-ABI checkpoint integrity is covered separately by Base registry tests.
    Keeping these tiny fake artifacts avoids allocating/writing a 10.2M model twice
    merely to test Base-id/SHA/ABI relationship logic.
    """
    monkeypatch.setattr("orbitune.base_registry._validate_current_checkpoint", lambda path, manifest: None)


def test_registry_allows_adapters_for_multiple_registered_bases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _isolate_registry_linkage_from_checkpoint_abi(monkeypatch)
    bases=tmp_path/"bases"; adapters=tmp_path/"adapters"; sha_a=_write_base(bases,"base-a",b"A"); sha_b=_write_base(bases,"base-b",b"B"); _write_adapter(adapters,"style-a-v0","community",base_id="base-a",base_sha=sha_a); _write_adapter(adapters,"style-b-v0","community",base_id="base-b",base_sha=sha_b); registry=build_registry(adapters,bases); assert {item["base_model"] for item in registry["adapters"]}=={"base-a","base-b"}


def test_registry_rejects_wrong_base_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _isolate_registry_linkage_from_checkpoint_abi(monkeypatch)
    bases=tmp_path/"bases"; adapters=tmp_path/"adapters"; _write_base(bases,"base-a",b"A"); _write_adapter(adapters,"bad-v0","community",base_id="base-a",base_sha="b"*64)
    with pytest.raises(ValueError,match="base_sha256"): build_registry(adapters,bases)


def test_legacy_write_registry_still_writes_adapter_entries_without_cross_registry(tmp_path: Path):
    adapters=tmp_path/"adapters"; _write_adapter(adapters,"community-test-v0","community",base_id="custom-base",base_sha="a"*64); out=tmp_path/"registry.json"; write_registry(out,adapters); written=json.loads(out.read_text(encoding="utf-8")); assert written["adapters"][0]["base_model"]=="custom-base"
