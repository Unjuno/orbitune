from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

SCRIPT=Path("scripts/download_base_model.py"); spec=importlib.util.spec_from_file_location("orbitune_download_base_model",SCRIPT); assert spec is not None and spec.loader is not None; downloader=importlib.util.module_from_spec(spec); spec.loader.exec_module(downloader)


def _manifest(payload:bytes)->dict[str,object]:
    digest=hashlib.sha256(payload).hexdigest()
    return {"model_id":"orbitune-base","architecture":"orbitune-midi-gpt-v0","tokenizer":"theory-remi-v0","parameters":10_200_960,"base_sha256":digest,"artifacts":{"checkpoint":{"filename":"orbitune-base.pt","bytes":len(payload),"sha256":digest,"url":"https://example.test/orbitune-base.pt"}}}


def test_validate_artifact_accepts_immutable_base_manifest():
    artifact=downloader._validate_artifact(_manifest(b"checkpoint"),"checkpoint"); assert artifact["filename"]=="orbitune-base.pt"

def test_validate_artifact_rejects_wrong_model_contract():
    manifest=_manifest(b"checkpoint"); manifest["parameters"]=123
    with pytest.raises(ValueError,match="parameter count mismatch"): downloader._validate_artifact(manifest,"checkpoint")

def test_validate_artifact_rejects_checkpoint_hash_not_equal_to_base_hash():
    manifest=_manifest(b"checkpoint"); manifest["base_sha256"]="0"*64
    with pytest.raises(ValueError,match="checkpoint artifact hash"): downloader._validate_artifact(manifest,"checkpoint")

def test_download_verified_checks_hash_before_replace(tmp_path:Path,monkeypatch:pytest.MonkeyPatch):
    payload=b"verified-orbitune-checkpoint"; artifact=downloader._validate_artifact(_manifest(payload),"checkpoint"); monkeypatch.setattr(downloader.urllib.request,"urlopen",lambda *args,**kwargs:io.BytesIO(payload)); target=tmp_path/"orbitune-base.pt"; downloader._download_verified(artifact,target); assert target.read_bytes()==payload; assert downloader.sha256_file(target)==artifact["sha256"]; assert not list(tmp_path.glob("*.part"))

def test_download_verified_removes_corrupt_partial_file(tmp_path:Path,monkeypatch:pytest.MonkeyPatch):
    expected=b"expected"; artifact=downloader._validate_artifact(_manifest(expected),"checkpoint"); monkeypatch.setattr(downloader.urllib.request,"urlopen",lambda *args,**kwargs:io.BytesIO(b"corrupt")); target=tmp_path/"orbitune-base.pt"
    with pytest.raises(ValueError): downloader._download_verified(artifact,target)
    assert not target.exists(); assert not list(tmp_path.glob("*.part"))
