from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType


COMPLETION_SCHEMA_VERSION = 3
COMPLETION_LOG_MARKER = "GPU_CONTROL_COMPLETION_JSON_V3:"


def _load_base_entrypoint() -> ModuleType:
    path = Path(__file__).with_name("entrypoint.py")
    spec = importlib.util.spec_from_file_location("orbitune_runpod_entrypoint_v2_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load trusted canary entrypoint helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_entrypoint()


def _validate_process_exit_code(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not -255 <= value <= 255:
        raise ValueError("process_exit_code must be an integer between -255 and 255")
    return value


def _write_completion_evidence_v3(
    output_dir: Path,
    result_bytes: bytes,
    values: dict[str, str],
    *,
    process_exit_code: int,
) -> tuple[Path, bytes]:
    source_sha = os.environ.get("ORBITUNE_SOURCE_SHA", "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("authenticated completion requires a baked 40-character ORBITUNE_SOURCE_SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", values["nonce"]):
        raise ValueError("GPU_CONTROL_COMPLETION_NONCE must be 32 bytes encoded as lowercase hex")
    if not _BASE._SHA256_RE.fullmatch(values["plan_fingerprint"]):
        raise ValueError("GPU_CONTROL_PLAN_FINGERPRINT must be a lowercase sha256 digest")
    if not _BASE._SHA256_RE.fullmatch(values["image_digest"]):
        raise ValueError("GPU_CONTROL_IMAGE_DIGEST must be a lowercase sha256 digest")
    if not _BASE._EXECUTION_NAME_RE.fullmatch(values["execution_name"]):
        raise ValueError("GPU_CONTROL_EXECUTION_NAME is invalid")
    expected_name = f"gpu-control-{values['plan_fingerprint'][7:19]}-{values['nonce'][:12]}"
    if values["execution_name"] != expected_name:
        raise ValueError("GPU_CONTROL_EXECUTION_NAME does not match plan fingerprint and nonce")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", values["key_id"]):
        raise ValueError("GPU_CONTROL_COMPLETION_KEY_ID is invalid")
    signed_exit_code = _validate_process_exit_code(process_exit_code)

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
        "process_exit_code": signed_exit_code,
        "result_sha256": result_digest,
        "source_sha": source_sha,
    }
    canonical = json.dumps(signed, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        mac = hmac.new(secret, canonical, hashlib.sha256).hexdigest()
    finally:
        for index in range(len(secret)):
            secret[index] = 0

    evidence: dict[str, object] = {
        **signed,
        "mac_sha256": mac,
        "schema_version": COMPLETION_SCHEMA_VERSION,
    }
    completion_bytes = _BASE._json_bytes(evidence)
    path = output_dir / "completion.json"
    _BASE._atomic_bytes(path, completion_bytes)
    return path, completion_bytes


def _authenticated_main(completion_values: dict[str, str]) -> int:
    argv = sys.argv[1:]
    try:
        output_dir = _BASE._output_dir(argv)
        _BASE._prepare_authenticated_paths(output_dir)
        runner_environment = _BASE._runner_environment()
    except Exception as exc:
        print(f"completion environment error: {exc}", file=sys.stderr)
        return _BASE._COMPLETION_ERROR_EXIT_CODE

    result_path = output_dir / "result.json"
    command = [sys.executable, "workloads/runpod-training-canary/run.py", *argv]
    try:
        completed = subprocess.run(command, **_BASE._authenticated_subprocess_kwargs(runner_environment))
        _BASE._reclaim_authenticated_output(output_dir)
    except Exception as exc:
        print(f"runner launch error: {exc}", file=sys.stderr)
        return _BASE._COMPLETION_ERROR_EXIT_CODE

    result_missing = not result_path.is_file()
    effective_exit_code = completed.returncode
    if result_missing and effective_exit_code == 0:
        effective_exit_code = _BASE._MISSING_RESULT_EXIT_CODE

    try:
        if result_missing:
            result_bytes = _BASE._write_failure_result(result_path, exit_code=completed.returncode)
        else:
            result_bytes = _BASE._read_regular_file(result_path)
        _BASE._validate_result_status(result_bytes, effective_exit_code=effective_exit_code)
        _, completion_bytes = _write_completion_evidence_v3(
            output_dir,
            result_bytes,
            completion_values,
            process_exit_code=effective_exit_code,
        )
        _BASE._emit_bytes_marker(_BASE.RESULT_LOG_MARKER, "result.json", result_bytes)
        _BASE._emit_bytes_marker(COMPLETION_LOG_MARKER, "completion.json", completion_bytes)
        _BASE._seal_authenticated_outputs(output_dir)
    except Exception as exc:
        print(f"completion evidence error: {exc}", file=sys.stderr)
        return _BASE._COMPLETION_ERROR_EXIT_CODE
    return effective_exit_code


def main() -> int:
    try:
        completion_values = _BASE._capture_completion_values()
    except Exception as exc:
        print(f"completion environment error: {exc}", file=sys.stderr)
        return _BASE._COMPLETION_ERROR_EXIT_CODE

    if completion_values is None:
        # Preserve the existing unauthenticated/local contract exactly.
        return _BASE.main()
    return _authenticated_main(completion_values)


if __name__ == "__main__":
    raise SystemExit(main())
