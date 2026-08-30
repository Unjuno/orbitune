#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-RTX 3080}"
HOST_LABEL="${HOST_LABEL:-VLab16}"
LENGTHS="${LENGTHS:-256,512,1024,2048,4096,8192}"
DTYPE="${DTYPE:-fp16}"
BATCH="${BATCH:-1}"
D_MODEL="${D_MODEL:-256}"
HEADS="${HEADS:-4}"
SLOTS="${SLOTS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
HOT_WINDOW="${HOT_WINDOW:-256}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-10}"

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 10; }
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 11; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 12; }

SOURCE_SHA="$(git rev-parse HEAD)"
if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "could not resolve exact source SHA" >&2
  exit 13
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "working tree must be clean before the local benchmark" >&2
  exit 14
fi

GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
if [[ -z "$GPU_INFO" ]]; then
  echo "nvidia-smi returned no GPU" >&2
  exit 15
fi
if ! grep -Fq "$EXPECTED_GPU_SUBSTRING" <<<"$GPU_INFO"; then
  echo "expected GPU containing '$EXPECTED_GPU_SUBSTRING', got:" >&2
  echo "$GPU_INFO" >&2
  exit 16
fi

IMAGE="orbitune-recurrent-memory-bench:${SOURCE_SHA:0:12}"
OUTPUT_DIR="$ROOT/outputs/recurrent-memory-benchmark/$HOST_LABEL/$SOURCE_SHA"
mkdir -p "$OUTPUT_DIR"

echo "host=$HOST_LABEL"
echo "source_sha=$SOURCE_SHA"
echo "gpu=$GPU_INFO"
echo "lengths=$LENGTHS"
echo "hot_window=$HOT_WINDOW"

docker build \
  --file workloads/recurrent-memory-cuda-benchmark/Dockerfile \
  --tag "$IMAGE" \
  .

docker run --rm \
  --gpus all \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g \
  --mount "type=bind,src=$OUTPUT_DIR,dst=/outputs" \
  "$IMAGE" \
  --device cuda \
  --dtype "$DTYPE" \
  --lengths "$LENGTHS" \
  --batch "$BATCH" \
  --d-model "$D_MODEL" \
  --heads "$HEADS" \
  --slots "$SLOTS" \
  --chunk-size "$CHUNK_SIZE" \
  --hot-window "$HOT_WINDOW" \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS" \
  --out /outputs/result.json

python - "$OUTPUT_DIR/result.json" "$EXPECTED_GPU_SUBSTRING" "$LENGTHS" "$HOT_WINDOW" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_gpu = sys.argv[2]
lengths = [int(v) for v in sys.argv[3].split(",")]
hot_window = int(sys.argv[4])
result = json.loads(path.read_text(encoding="utf-8"))

assert result["schema_version"] == 1
assert result["device"] == "cuda"
assert expected_gpu in result["gpu_name"]
assert result["hot_window"] == hot_window
assert result["results"]

parallel = [r for r in result["results"] if r["kernel"] in {"linear_parallel_scan", "sdpa_full_causal"}]
stream = [r for r in result["results"] if r["kernel"] in {"linear_recurrent_stream", "sdpa_kv_stream"}]
hot_cold = [r for r in result["results"] if r["kernel"] == "hot_cold_memory_first_stream"]
assert sorted({r["length"] for r in parallel}) == sorted(lengths)
assert len(parallel) == 2 * len(lengths)
assert len(stream) == 2
assert sorted(r["length"] for r in hot_cold) == sorted(lengths)
for row in result["results"]:
    assert row["status"] in {"ok", "oom"}
    if row["status"] == "ok":
        assert row["milliseconds"] > 0
        assert row["tokens_per_second"] > 0
        assert row["peak_memory_bytes"] is not None and row["peak_memory_bytes"] > 0
        assert row["state_or_cache_bytes"] is not None and row["state_or_cache_bytes"] > 0

linear_stream = next(r for r in stream if r["kernel"] == "linear_recurrent_stream")
sdpa_stream = next(r for r in stream if r["kernel"] == "sdpa_kv_stream")
assert linear_stream["state_or_cache_bytes"] < sdpa_stream["state_or_cache_bytes"]

steady = [r for r in hot_cold if r["length"] >= hot_window]
assert steady
assert len({r["state_or_cache_bytes"] for r in steady}) == 1
full_kv_by_length = {
    r["length"]: r["state_or_cache_bytes"]
    for r in parallel
    if r["kernel"] == "sdpa_full_causal"
}
for row in steady:
    if row["length"] > hot_window:
        assert row["state_or_cache_bytes"] < full_kv_by_length[row["length"]]

print(json.dumps({
    "status": "pass",
    "gpu_name": result["gpu_name"],
    "dtype": result["dtype"],
    "lengths": lengths,
    "hot_window": hot_window,
    "linear_stream_state_bytes": linear_stream["state_or_cache_bytes"],
    "sdpa_stream_cache_bytes": sdpa_stream["state_or_cache_bytes"],
    "hot_cold_steady_state_bytes": steady[-1]["state_or_cache_bytes"],
    "result": str(path),
}, indent=2, sort_keys=True))
PY
