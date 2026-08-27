from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ENTRYPOINT = Path(__file__).parents[1] / "workloads" / "runpod-training-canary" / "entrypoint.py"


def _load_entrypoint():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("orbitune_runpod_entrypoint_security", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completion_environment(module) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {
        module._COMPLETION_ENV["key_b64"]: "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
        module._COMPLETION_ENV["key_id"]: "paid-runpod-v2",
        module._COMPLETION_ENV["nonce"]: "a" * 64,
        module._COMPLETION_ENV["execution_name"]: "gpu-control-111111111111-aaaaaaaaaaaa",
        module._COMPLETION_ENV["plan_fingerprint"]: "sha256:" + "1" * 64,
        module._COMPLETION_ENV["image_digest"]: "sha256:" + "2" * 64,
    }


def test_runner_environment_strips_all_completion_material(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entrypoint()
    completion_environment = _completion_environment(module)
    for name, value in completion_environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ORBITUNE_SOURCE_SHA", "d" * 40)
    monkeypatch.setenv("KEEP_FOR_RUNNER", "visible")

    values, child_environment = module._runner_environment()

    assert values is not None
    assert values["key_b64"] == completion_environment[module._COMPLETION_ENV["key_b64"]]
    assert values["execution_name"] == completion_environment[module._COMPLETION_ENV["execution_name"]]
    for name in module._COMPLETION_ENV.values():
        assert name not in child_environment
    assert child_environment["ORBITUNE_SOURCE_SHA"] == "d" * 40
    assert child_environment["KEEP_FOR_RUNNER"] == "visible"


def test_partial_completion_environment_fails_before_child_environment_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_entrypoint()
    for name in module._COMPLETION_ENV.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(module._COMPLETION_ENV["key_id"], "paid-runpod-v2")

    with pytest.raises(ValueError, match="partial gpu-control completion environment"):
        module._runner_environment()
