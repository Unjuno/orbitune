from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path

import pytest


ENTRYPOINT = Path(__file__).resolve().parents[1] / "workloads" / "runpod-training-canary" / "entrypoint.py"


def _load_entrypoint():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("orbitune_runpod_entrypoint_v3", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values() -> dict[str, str]:
    return {
        "key_b64": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
        "key_id": "paid-runpod-v3",
        "nonce": "a" * 64,
        "execution_name": "gpu-control-111111111111-aaaaaaaaaaaa",
        "plan_fingerprint": "sha256:" + "1" * 64,
        "image_digest": "sha256:" + "2" * 64,
    }


def test_completion_v3_authenticates_process_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entrypoint()
    monkeypatch.setenv("ORBITUNE_SOURCE_SHA", "d" * 40)
    result_bytes = b'{"schema_version":1,"status":"pass"}\n'
    path, completion_bytes = module._write_completion_v3_evidence(
        tmp_path,
        result_bytes,
        _values(),
        process_exit_code=0,
    )

    assert path.name == "completion-v3.json"
    assert path.read_bytes() == completion_bytes
    evidence = json.loads(completion_bytes)
    assert evidence["schema_version"] == 3
    assert evidence["process_exit_code"] == 0
    assert evidence["result_sha256"] == "sha256:" + hashlib.sha256(result_bytes).hexdigest()

    signed = {
        key: evidence[key]
        for key in (
            "execution_name",
            "image_digest",
            "key_id",
            "nonce",
            "plan_fingerprint",
            "process_exit_code",
            "result_sha256",
            "source_sha",
        )
    }
    canonical = json.dumps(signed, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    secret = base64.b64decode("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=")
    expected = hmac.new(secret, canonical, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(evidence["mac_sha256"], expected)

    tampered = dict(signed, process_exit_code=1)
    tampered_canonical = json.dumps(
        tampered, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    tampered_mac = hmac.new(secret, tampered_canonical, hashlib.sha256).hexdigest()
    assert not hmac.compare_digest(evidence["mac_sha256"], tampered_mac)


def test_completion_v3_rejects_invalid_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entrypoint()
    monkeypatch.setenv("ORBITUNE_SOURCE_SHA", "d" * 40)
    with pytest.raises(ValueError, match="process_exit_code"):
        module._write_completion_v3_evidence(tmp_path, b"{}\n", _values(), process_exit_code=256)
