#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-RTX 3080}"
MEMORY_EPOCHS="${MEMORY_EPOCHS:-3}"
COMPOSER_EPOCHS="${COMPOSER_EPOCHS:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
WARMUP_EVENTS="${WARMUP_EVENTS:-32}"

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 10; }
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 11; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 12; }

SOURCE_SHA="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "working tree must be clean" >&2
  exit 13
fi
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
if ! grep -Fq "$EXPECTED_GPU_SUBSTRING" <<<"$GPU_INFO"; then
  echo "expected GPU containing '$EXPECTED_GPU_SUBSTRING', got: $GPU_INFO" >&2
  exit 14
fi

IMAGE="orbitune-mutopia-real-memory-smoke:${SOURCE_SHA:0:12}"
OUTPUT_DIR="$ROOT/outputs/mutopia-real-memory-smoke/$SOURCE_SHA"
RAW_DIR="$OUTPUT_DIR/raw-midi"
mkdir -p "$RAW_DIR" "$OUTPUT_DIR/results"

docker build \
  --file workloads/mutopia-real-memory-smoke/Dockerfile \
  --tag "$IMAGE" \
  .

# Acquisition is the only network-enabled phase. The manifest is a strict
# Public-Domain allowlist and the fetcher records SHA-256 + provenance.
docker run --rm \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=256m \
  --mount "type=bind,src=$OUTPUT_DIR,dst=/outputs" \
  "$IMAGE" \
  python scripts/fetch_midi_allowlist.py \
    --manifest experiments/data/manifests/mutopia_public_domain_smoke_v1.json \
    --output-dir /outputs/raw-midi \
    --provenance-out /outputs/provenance.json

# All remaining data processing/training is network-isolated.
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m \
  --mount "type=bind,src=$OUTPUT_DIR,dst=/outputs" \
  "$IMAGE" \
  python scripts/prepare_compound_split.py \
    --source /outputs/raw-midi \
    --train-out /outputs/train.jsonl \
    --validation-out /outputs/validation.jsonl \
    --report-out /outputs/split-report.json \
    --validation-fraction 0.2 \
    --split-seed mutopia-public-domain-smoke-v1 \
    --min-events 32

for mode in shared_matched multibank_routed; do
  for seed in 1 2 3; do
    docker run --rm \
      --gpus all \
      --network none \
      --read-only \
      --cap-drop ALL \
      --security-opt no-new-privileges \
      --pids-limit 256 \
      --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g \
      --mount "type=bind,src=$OUTPUT_DIR,dst=/outputs" \
      "$IMAGE" \
      python experiments/real_compound_memory_experiment_matched.py \
        --train-jsonl /outputs/train.jsonl \
        --validation-jsonl /outputs/validation.jsonl \
        --mode "$mode" \
        --composer-policy frozen \
        --seed "$seed" \
        --memory-epochs "$MEMORY_EPOCHS" \
        --composer-epochs "$COMPOSER_EPOCHS" \
        --chunk-size "$CHUNK_SIZE" \
        --warmup-events "$WARMUP_EVENTS" \
        --device cuda \
        --checkpoint-out "/outputs/results/${mode}-seed${seed}.pt" \
        --out "/outputs/results/${mode}-seed${seed}.json"
  done
done

python - "$OUTPUT_DIR" "$SOURCE_SHA" "$GPU_INFO" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
source_sha = sys.argv[2]
gpu = sys.argv[3]
summary = {"source_sha": source_sha, "gpu": gpu, "modes": {}}
for mode in ("shared_matched", "multibank_routed"):
    rows = []
    for seed in (1, 2, 3):
        rows.append(json.loads((root / "results" / f"{mode}-seed{seed}.json").read_text()))
    after = [row["validation_after_composer"] for row in rows]
    summary["modes"][mode] = {
        "fast_macro_recall_mean": statistics.mean(row["fast_macro_recall"] for row in after),
        "medium_macro_recall_mean": statistics.mean(row["medium_macro_recall"] for row in after),
        "slow_macro_recall_mean": statistics.mean(row["slow_macro_recall"] for row in after),
        "next_event_type_accuracy_mean": statistics.mean(row["next_event_type_accuracy"] for row in after),
        "seeds": [1, 2, 3],
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "results=$OUTPUT_DIR"
