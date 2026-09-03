#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-RTX 3080}"
STEPS="${STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-256}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-50}"
HOST_LABEL="${HOST_LABEL:-VLab16}"

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 10; }
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 11; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 12; }

SOURCE_SHA="$(git rev-parse HEAD)"
if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "could not resolve exact source SHA" >&2
  exit 13
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "working tree must be clean before the local GPU canary" >&2
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

echo "host=$HOST_LABEL"
echo "source_sha=$SOURCE_SHA"
echo "gpu=$GPU_INFO"

IMAGE="orbitune-local-gpu-canary:${SOURCE_SHA:0:12}"
OUTPUT_DIR="$ROOT/outputs/local-gpu-canary/$HOST_LABEL/$SOURCE_SHA"
mkdir -p "$OUTPUT_DIR"

docker build \
  --file workloads/local-gpu-canary/Dockerfile \
  --build-arg "ORBITUNE_SOURCE_SHA=$SOURCE_SHA" \
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
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g \
  --mount "type=bind,src=$OUTPUT_DIR,dst=/outputs" \
  "$IMAGE" \
  --output-dir /outputs \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --validation-interval "$VALIDATION_INTERVAL" \
  --device cuda \
  --require-cuda

python - "$OUTPUT_DIR/result.json" "$SOURCE_SHA" "$EXPECTED_GPU_SUBSTRING" "$STEPS" "$BATCH_SIZE" "$SEQ_LEN" <<'PY'
import json
import math
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
source_sha = sys.argv[2]
expected_gpu = sys.argv[3]
steps = int(sys.argv[4])
batch_size = int(sys.argv[5])
seq_len = int(sys.argv[6])

result = json.loads(result_path.read_text(encoding="utf-8"))
assert result["schema_version"] == 1
assert result["workload_id"] == "orbitune-local-gpu-canary-v1"
assert result["source_sha"] == source_sha
assert result["status"] == "pass"
assert result["device_type"] == "cuda"
assert result["cuda_available"] is True
assert expected_gpu in result["gpu_name"]
assert result["peak_vram_bytes"] > 0
assert result["steps"] == steps
assert result["batch_size"] == batch_size
assert result["seq_len"] == seq_len
assert result["tokens_processed"] == steps * batch_size * seq_len
assert math.isfinite(result["first_training_loss"])
assert math.isfinite(result["final_training_loss"])
assert len(result["validation_history"]) >= 1
assert result["artifacts"][0]["name"] == "canary-base.pt"
assert result["artifacts"][0]["bytes"] > 0
print(json.dumps({
    "status": "pass",
    "source_sha": source_sha,
    "gpu_name": result["gpu_name"],
    "peak_vram_bytes": result["peak_vram_bytes"],
    "tokens_processed": result["tokens_processed"],
    "tokens_per_second": result["tokens_per_second"],
    "result": str(result_path),
}, indent=2, sort_keys=True))
PY
