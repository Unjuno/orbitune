from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
WORKLOAD_ID = "orbitune-runpod-training-canary-v1"
_COMPLETION_ENV = {
    "key_b64": "GPU_CONTROL_COMPLETION_KEY_B64",
    "key_id": "GPU_CONTROL_COMPLETION_KEY_ID",
    "nonce": "GPU_CONTROL_COMPLETION_NONCE",
    "plan_fingerprint": "GPU_CONTROL_PLAN_FINGERPRINT",
    "provider_job_id": "GPU_CONTROL_PROVIDER_JOB_ID",
    "image_digest": "GPU_CONTROL_IMAGE_DIGEST",
}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _output_dir(argv: list[str]) -> Path:
    for index, arg in enumerate(argv):
        if arg == "--output-dir" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if arg.startswith("--output-dir="):
            return Path(arg.split("=", 1)[1])
    return Path("/outputs")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_failure_result(path: Path, *, exit_code: int) -> None:
    _atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "workload_id": WORKLOAD_ID,
            "source_sha": os.environ.get("ORBITUNE_SOURCE_SHA", "unbaked-local").strip(),
            "status": "fail",
            "failure_kind": "runner-exited-without-result",
            "runner_exit_code": exit_code,
        },
    )


def _completion_values() -> dict[str, str] | None:
    present = {name: os.environ.get(env_name, "").strip() for name, env_name in _COMPLETION_ENV.items()}
    populated = {name for name, value in present.items() if value}
    if not populated:
        return None
    if populated != set(_COMPLETION_ENV):
        missing = sorted(set(_COMPLETION_ENV) - populated)
        raise ValueError(f"partial gpu-control completion environment; missing: {', '.join(missing)}")
    return present


def _write_completion_evidence(output_dir: Path, result_path: Path, values: dict[str, str]) -> None:
    source_sha = os.environ.get("ORBITUNE_SOURCE_SHA", "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("authenticated completion requires a baked 40-character ORBITUNE_SOURCE_SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", values["nonce"]):
        raise ValueError("GPU_CONTROL_COMPLETION_NONCE must be 32 bytes encoded as lowercase hex")
    if not _SHA256_RE.fullmatch(values["plan_fingerprint"]):
        raise ValueError("GPU_CONTROL_PLAN_FINGERPRINT must be a lowercase sha256 digest")
    if not _SHA256_RE.fullmatch(values["image_digest"]):
        raise ValueError("GPU_CONTROL_IMAGE_DIGEST must be a lowercase sha256 digest")
    if not values["provider_job_id"]:
        raise ValueError("GPU_CONTROL_PROVIDER_JOB_ID is required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", values["key_id"]):
        raise ValueError("GPU_CONTROL_COMPLETION_KEY_ID is invalid")
    try:
        secret = base64.b64decode(values["key_b64"], validate=True)
    except Exception as exc:
        raise ValueError("GPU_CONTROL_COMPLETION_KEY_B64 must be valid base64") from exc
    if len(secret) < 32:
        raise ValueError("GPU_CONTROL_COMPLETION_KEY_B64 must decode to at least 32 bytes")

    result_digest = "sha256:" + hashlib.sha256(result_path.read_bytes()).hexdigest()
    signed = {
        "image_digest": values["image_digest"],
        "key_id": values["key_id"],
        "nonce": values["nonce"],
        "plan_fingerprint": values["plan_fingerprint"],
        "provider_job_id": values["provider_job_id"],
        "result_sha256": result_digest,
        "source_sha": source_sha,
    }
    canonical = json.dumps(signed, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence: dict[str, object] = {
        **signed,
        "mac_sha256": hmac.new(secret, canonical, hashlib.sha256).hexdigest(),
        "schema_version": 1,
    }
    _atomic_json(output_dir / "completion.json", evidence)


def main() -> int:
    argv = sys.argv[1:]
    output_dir = _output_dir(argv)
    result_path = output_dir / "result.json"
    command = [sys.executable, "workloads/runpod-training-canary/run.py", *argv]
    completed = subprocess.run(command, check=False)
    if not result_path.is_file():
        _write_failure_result(result_path, exit_code=completed.returncode)
    if completed.returncode != 0:
        return completed.returncode

    try:
        completion_values = _completion_values()
        if completion_values is not None:
            _write_completion_evidence(output_dir, result_path, completion_values)
    except Exception as exc:
        print(f"completion evidence error: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
