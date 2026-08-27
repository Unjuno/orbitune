from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 2
WORKLOAD_ID = "orbitune-runpod-training-canary-v1"
RESULT_LOG_MARKER = "GPU_CONTROL_RESULT_JSON_V1:"
COMPLETION_LOG_MARKER = "GPU_CONTROL_COMPLETION_JSON_V2:"
_MAX_MARKER_BYTES = 16 * 1024
_MISSING_RESULT_EXIT_CODE = 5
_COMPLETION_ERROR_EXIT_CODE = 4
_TRAINING_UID = 10001
_TRAINING_GID = 10001
_AUTHENTICATED_OUTPUT_DIR = Path("/outputs")
_TRAINING_HOME = Path("/tmp/orbitune-runner")
_COMPLETION_ENV = {
    "key_b64": "GPU_CONTROL_COMPLETION_KEY_B64",
    "key_id": "GPU_CONTROL_COMPLETION_KEY_ID",
    "nonce": "GPU_CONTROL_COMPLETION_NONCE",
    "execution_name": "GPU_CONTROL_EXECUTION_NAME",
    "plan_fingerprint": "GPU_CONTROL_PLAN_FINGERPRINT",
    "image_digest": "GPU_CONTROL_IMAGE_DIGEST",
}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXECUTION_NAME_RE = re.compile(r"^gpu-control-[0-9a-f]{12}-[0-9a-f]{12}$")


def _output_dir(argv: list[str]) -> Path:
    values: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--output-dir":
            if index + 1 >= len(argv):
                raise ValueError("--output-dir requires a value")
            values.append(argv[index + 1])
            index += 2
            continue
        if arg.startswith("--output-dir="):
            values.append(arg.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise ValueError("--output-dir may be specified at most once")
    value = values[0] if values else "/outputs"
    if not value:
        raise ValueError("--output-dir must not be empty")
    return Path(value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_failure_result(path: Path, *, exit_code: int) -> bytes:
    payload = _json_bytes({
        "schema_version": SCHEMA_VERSION,
        "workload_id": WORKLOAD_ID,
        "source_sha": os.environ.get("ORBITUNE_SOURCE_SHA", "unbaked-local").strip(),
        "status": "fail",
        "failure_kind": "runner-exited-without-result",
        "runner_exit_code": exit_code,
    })
    _atomic_bytes(path, payload)
    return payload


def _capture_completion_values() -> dict[str, str] | None:
    raw: dict[str, str | None] = {}
    for name, env_name in _COMPLETION_ENV.items():
        raw[name] = os.environ.pop(env_name, None)
    populated = {name for name, value in raw.items() if value not in {None, ""}}
    if not populated:
        return None
    if populated != set(_COMPLETION_ENV):
        missing = sorted(set(_COMPLETION_ENV) - populated)
        raise ValueError(f"partial gpu-control completion environment; missing: {', '.join(missing)}")
    values: dict[str, str] = {}
    for name, value in raw.items():
        assert value is not None
        if value != value.strip():
            raise ValueError(f"{_COMPLETION_ENV[name]} must not contain surrounding whitespace")
        values[name] = value
    return values


def _runner_environment() -> dict[str, str]:
    child_environment = os.environ.copy()
    for env_name in _COMPLETION_ENV.values():
        child_environment.pop(env_name, None)
    child_environment["HOME"] = str(_TRAINING_HOME)
    child_environment["XDG_CACHE_HOME"] = str(_TRAINING_HOME / ".cache")
    child_environment["TMPDIR"] = "/tmp"
    return child_environment


def _prepare_authenticated_paths(output_dir: Path) -> None:
    if os.geteuid() != 0:
        raise ValueError("authenticated completion requires the wrapper to run as root")
    if output_dir != _AUTHENTICATED_OUTPUT_DIR:
        raise ValueError("authenticated completion requires --output-dir /outputs")
    if output_dir.exists():
        info = os.lstat(output_dir)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("/outputs must be a real directory")
    else:
        output_dir.mkdir(mode=0o700)
    _TRAINING_HOME.mkdir(parents=True, exist_ok=True)
    os.chown(output_dir, _TRAINING_UID, _TRAINING_GID)
    os.chmod(output_dir, 0o700)
    os.chown(_TRAINING_HOME, _TRAINING_UID, _TRAINING_GID)
    os.chmod(_TRAINING_HOME, 0o700)


def _authenticated_subprocess_kwargs(environment: dict[str, str]) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise ValueError("authenticated completion requires root privilege separation")
    return {
        "check": False,
        "env": environment,
        "user": _TRAINING_UID,
        "group": _TRAINING_GID,
        "extra_groups": (),
        "umask": 0o077,
        "start_new_session": True,
        "close_fds": True,
    }


def _reclaim_authenticated_output(output_dir: Path) -> None:
    info = os.lstat(output_dir)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("/outputs changed type during training")
    os.chown(output_dir, 0, 0)
    os.chmod(output_dir, 0o700)
    for name in ("result.json", "canary-base.pt"):
        path = output_dir / name
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            os.chown(path, 0, 0, follow_symlinks=False)
            os.chmod(path, 0o400)


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{path.name} is not a readable regular file") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path.name} must be a regular file")
        if info.st_size > _MAX_MARKER_BYTES:
            raise ValueError(f"{path.name} exceeds the bounded pre-encoding size")
        chunks: list[bytes] = []
        remaining = _MAX_MARKER_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(4096, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_MARKER_BYTES:
            raise ValueError(f"{path.name} exceeds the bounded pre-encoding size")
        return raw
    finally:
        os.close(descriptor)


def _strict_json_object(raw: bytes, label: str) -> dict[str, object]:
    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            value[key] = item
        return value
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_result_status(result_bytes: bytes, *, effective_exit_code: int) -> None:
    payload = _strict_json_object(result_bytes, "result.json")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result.json schema_version is invalid")
    if payload.get("workload_id") != WORKLOAD_ID:
        raise ValueError("result.json workload_id is invalid")
    expected_status = "pass" if effective_exit_code == 0 else "fail"
    if payload.get("status") != expected_status:
        raise ValueError("result.json status does not match runner exit outcome")


def _emit_bytes_marker(prefix: str, label: str, raw: bytes) -> None:
    marker = prefix.encode("ascii") + base64.urlsafe_b64encode(raw)
    if len(marker) > _MAX_MARKER_BYTES:
        raise ValueError(f"{label} encoded log marker exceeds bounded marker size")
    print(marker.decode("ascii"), flush=True)


def _write_completion_evidence(output_dir: Path, result_bytes: bytes, values: dict[str, str]) -> tuple[Path, bytes]:
    source_sha = os.environ.get("ORBITUNE_SOURCE_SHA", "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("authenticated completion requires a baked 40-character ORBITUNE_SOURCE_SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", values["nonce"]):
        raise ValueError("GPU_CONTROL_COMPLETION_NONCE must be 32 bytes encoded as lowercase hex")
    if not _SHA256_RE.fullmatch(values["plan_fingerprint"]):
        raise ValueError("GPU_CONTROL_PLAN_FINGERPRINT must be a lowercase sha256 digest")
    if not _SHA256_RE.fullmatch(values["image_digest"]):
        raise ValueError("GPU_CONTROL_IMAGE_DIGEST must be a lowercase sha256 digest")
    if not _EXECUTION_NAME_RE.fullmatch(values["execution_name"]):
        raise ValueError("GPU_CONTROL_EXECUTION_NAME is invalid")
    expected_name = f"gpu-control-{values['plan_fingerprint'][7:19]}-{values['nonce'][:12]}"
    if values["execution_name"] != expected_name:
        raise ValueError("GPU_CONTROL_EXECUTION_NAME does not match plan fingerprint and nonce")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", values["key_id"]):
        raise ValueError("GPU_CONTROL_COMPLETION_KEY_ID is invalid")
    try:
        secret = bytearray(base64.b64decode(values["key_b64"], validate=True))
    except Exception as exc:
        raise ValueError("GPU_CONTROL_COMPLETION_KEY_B64 must be valid base64") from exc
    values["key_b64"] = ""
    if len(secret) < 32:
        for index in range(len(secret)):
            secret[index] = 0
        raise ValueError("GPU_CONTROL_COMPLETION_KEY_B64 must decode to at least 32 bytes")
    result_digest = "sha256:" + hashlib.sha256(result_bytes).hexdigest()
    signed = {
        "execution_name": values["execution_name"],
        "image_digest": values["image_digest"],
        "key_id": values["key_id"],
        "nonce": values["nonce"],
        "plan_fingerprint": values["plan_fingerprint"],
        "result_sha256": result_digest,
        "source_sha": source_sha,
    }
    canonical = json.dumps(signed, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        mac = hmac.new(secret, canonical, hashlib.sha256).hexdigest()
    finally:
        for index in range(len(secret)):
            secret[index] = 0
    evidence: dict[str, object] = {**signed, "mac_sha256": mac, "schema_version": COMPLETION_SCHEMA_VERSION}
    completion_bytes = _json_bytes(evidence)
    path = output_dir / "completion.json"
    _atomic_bytes(path, completion_bytes)
    return path, completion_bytes


def _seal_authenticated_outputs(output_dir: Path) -> None:
    for name in ("result.json", "completion.json", "canary-base.pt"):
        path = output_dir / name
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, 0o444)
    os.chown(output_dir, 0, 0)
    os.chmod(output_dir, 0o555)


def main() -> int:
    argv = sys.argv[1:]
    try:
        completion_values = _capture_completion_values()
        output_dir = _output_dir(argv)
        authenticated = completion_values is not None
        if authenticated:
            _prepare_authenticated_paths(output_dir)
        runner_environment = _runner_environment()
    except Exception as exc:
        print(f"completion environment error: {exc}", file=sys.stderr)
        return _COMPLETION_ERROR_EXIT_CODE
    result_path = output_dir / "result.json"
    command = [sys.executable, "workloads/runpod-training-canary/run.py", *argv]
    try:
        if authenticated:
            completed = subprocess.run(command, **_authenticated_subprocess_kwargs(runner_environment))
            _reclaim_authenticated_output(output_dir)
        else:
            completed = subprocess.run(command, check=False, env=runner_environment)
    except Exception as exc:
        print(f"runner launch error: {exc}", file=sys.stderr)
        return _COMPLETION_ERROR_EXIT_CODE
    result_missing = not result_path.is_file()
    effective_exit_code = completed.returncode
    if result_missing and effective_exit_code == 0:
        effective_exit_code = _MISSING_RESULT_EXIT_CODE
    try:
        if result_missing:
            result_bytes = _write_failure_result(result_path, exit_code=completed.returncode)
        else:
            result_bytes = _read_regular_file(result_path)
        _validate_result_status(result_bytes, effective_exit_code=effective_exit_code)
        completion_bytes = None
        if completion_values is not None:
            _, completion_bytes = _write_completion_evidence(output_dir, result_bytes, completion_values)
        _emit_bytes_marker(RESULT_LOG_MARKER, "result.json", result_bytes)
        if completion_bytes is not None:
            _emit_bytes_marker(COMPLETION_LOG_MARKER, "completion.json", completion_bytes)
        if authenticated:
            _seal_authenticated_outputs(output_dir)
    except Exception as exc:
        print(f"completion evidence error: {exc}", file=sys.stderr)
        return _COMPLETION_ERROR_EXIT_CODE
    return effective_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
