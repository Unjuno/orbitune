from __future__ import annotations

import importlib.util
import json
import os
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


def test_completion_capture_scrubs_parent_and_child_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entrypoint()
    completion_environment = _completion_environment(module)
    for name, value in completion_environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ORBITUNE_SOURCE_SHA", "d" * 40)
    monkeypatch.setenv("KEEP_FOR_RUNNER", "visible")

    values = module._capture_completion_values()
    child_environment = module._runner_environment()

    assert values is not None
    assert values["key_b64"] == completion_environment[module._COMPLETION_ENV["key_b64"]]
    assert values["execution_name"] == completion_environment[module._COMPLETION_ENV["execution_name"]]
    for name in module._COMPLETION_ENV.values():
        assert name not in os.environ
        assert name not in child_environment
    assert child_environment["ORBITUNE_SOURCE_SHA"] == "d" * 40
    assert child_environment["KEEP_FOR_RUNNER"] == "visible"
    assert child_environment["HOME"] == str(module._TRAINING_HOME)


def test_partial_completion_environment_fails_after_scrubbing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entrypoint()
    for name in module._COMPLETION_ENV.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(module._COMPLETION_ENV["key_id"], "paid-runpod-v2")

    with pytest.raises(ValueError, match="partial gpu-control completion environment"):
        module._capture_completion_values()
    for name in module._COMPLETION_ENV.values():
        assert name not in os.environ


def test_authenticated_runner_drops_to_dedicated_uid_gid(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entrypoint()
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)

    kwargs = module._authenticated_subprocess_kwargs({"PATH": "/usr/bin"})

    assert kwargs["user"] == 10001
    assert kwargs["group"] == 10001
    assert kwargs["extra_groups"] == ()
    assert kwargs["umask"] == 0o077
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True


def test_authenticated_runner_fails_without_root(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entrypoint()
    monkeypatch.setattr(module.os, "geteuid", lambda: 1000)

    with pytest.raises(ValueError, match="root privilege separation"):
        module._authenticated_subprocess_kwargs({})


def test_authenticated_output_is_fixed_to_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_entrypoint()
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)

    with pytest.raises(ValueError, match="--output-dir /outputs"):
        module._prepare_authenticated_paths(tmp_path)


def test_output_dir_parser_rejects_ambiguous_forms() -> None:
    module = _load_entrypoint()
    with pytest.raises(ValueError, match="requires a value"):
        module._output_dir(["--output-dir"])
    with pytest.raises(ValueError, match="at most once"):
        module._output_dir(["--output-dir", "/a", "--output-dir=/b"])


def test_result_status_must_match_effective_exit_code() -> None:
    module = _load_entrypoint()
    passed = json.dumps({"schema_version": 1, "workload_id": module.WORKLOAD_ID, "status": "pass"}).encode()
    failed = json.dumps({"schema_version": 1, "workload_id": module.WORKLOAD_ID, "status": "fail"}).encode()

    module._validate_result_status(passed, effective_exit_code=0)
    module._validate_result_status(failed, effective_exit_code=2)
    with pytest.raises(ValueError, match="does not match"):
        module._validate_result_status(passed, effective_exit_code=2)
    with pytest.raises(ValueError, match="does not match"):
        module._validate_result_status(failed, effective_exit_code=0)


def test_regular_file_reader_rejects_symlink(tmp_path: Path) -> None:
    module = _load_entrypoint()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "result.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="readable regular file"):
        module._read_regular_file(link)
