from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "experiments" / "recurrent_memory_cuda_benchmark.py"
LAUNCHER = ROOT / "workloads" / "recurrent-memory-cuda-benchmark" / "run-local.sh"
DOCKERFILE = ROOT / "workloads" / "recurrent-memory-cuda-benchmark" / "Dockerfile"


def test_benchmark_cpu_smoke(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--lengths",
            "16,32",
            "--batch",
            "1",
            "--d-model",
            "32",
            "--heads",
            "4",
            "--slots",
            "4",
            "--chunk-size",
            "8",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["device"] == "cpu"
    assert result["gpu_name"] is None
    assert result["dtype"] == "fp32"

    rows = result["results"]
    parallel = [row for row in rows if row["kernel"] in {"linear_parallel_scan", "sdpa_full_causal"}]
    stream = [row for row in rows if row["kernel"] in {"linear_recurrent_stream", "sdpa_kv_stream"}]
    assert len(parallel) == 4
    assert len(stream) == 2
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["milliseconds"] > 0 for row in rows)
    assert all(row["tokens_per_second"] > 0 for row in rows)
    assert all(row["peak_memory_bytes"] is None for row in rows)

    linear_stream = next(row for row in stream if row["kernel"] == "linear_recurrent_stream")
    sdpa_stream = next(row for row in stream if row["kernel"] == "sdpa_kv_stream")
    assert linear_stream["state_or_cache_bytes"] == (4 * 32 + 4) * 4
    assert sdpa_stream["state_or_cache_bytes"] == 32 * 32 * 2 * 4
    assert linear_stream["state_or_cache_bytes"] < sdpa_stream["state_or_cache_bytes"]


def test_vlab16_launcher_is_gpu_only_and_hardened() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-RTX 3080}"' in text
    assert '--gpus all' in text
    assert '--device cuda' in text
    assert '--network none' in text
    assert '--read-only' in text
    assert '--cap-drop ALL' in text
    assert '--security-opt no-new-privileges' in text
    assert 'git status --porcelain --untracked-files=all' in text
    assert '256,512,1024,2048,4096,8192' in text


def test_benchmark_container_is_pinned_and_contains_scan_dependency() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "pytorch/pytorch@sha256:" in text
    assert "recurrent_memory_chunkwise_scan.py" in text
    assert "recurrent_memory_cuda_benchmark.py" in text
    assert 'ENTRYPOINT ["python", "experiments/recurrent_memory_cuda_benchmark.py"]' in text
