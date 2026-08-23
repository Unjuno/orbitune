from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
WORKLOAD_ID = "orbitune-runpod-training-canary-v1"


def _output_dir(argv: list[str]) -> Path:
    for index, arg in enumerate(argv):
        if arg == "--output-dir" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if arg.startswith("--output-dir="):
            return Path(arg.split("=", 1)[1])
    return Path("/outputs")


def _write_failure_result(path: Path, *, exit_code: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workload_id": WORKLOAD_ID,
        "source_sha": os.environ.get("ORBITUNE_SOURCE_SHA", "unbaked-local").strip(),
        "status": "fail",
        "failure_kind": "runner-exited-without-result",
        "runner_exit_code": exit_code,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    argv = sys.argv[1:]
    output_dir = _output_dir(argv)
    result_path = output_dir / "result.json"
    command = [sys.executable, "workloads/runpod-training-canary/run.py", *argv]
    completed = subprocess.run(command, check=False)
    if not result_path.is_file():
        _write_failure_result(result_path, exit_code=completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
