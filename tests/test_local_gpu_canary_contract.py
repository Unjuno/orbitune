from pathlib import Path


ROOT = Path(__file__).parents[1]
LOCAL = ROOT / "workloads" / "local-gpu-canary"
RUN = ROOT / "workloads" / "runpod-training-canary" / "run.py"


def test_local_gpu_image_reuses_pinned_cuda_base_and_provider_neutral_runner() -> None:
    dockerfile = (LOCAL / "Dockerfile").read_text(encoding="utf-8")
    assert "pytorch/pytorch@sha256:" in dockerfile
    assert "ORBITUNE_SOURCE_SHA" in dockerfile
    assert "orbitune-local-gpu-canary-v1" in dockerfile
    assert "workloads/runpod-training-canary/run.py" in dockerfile
    assert '"--workload-id", "orbitune-local-gpu-canary-v1"' in dockerfile


def test_local_launcher_requires_cuda_and_bounded_container_isolation() -> None:
    launcher = (LOCAL / "run-local.sh").read_text(encoding="utf-8")
    assert "EXPECTED_GPU_SUBSTRING:-RTX 3080" in launcher
    assert "BATCH_SIZE:-4" in launcher
    assert "--gpus all" in launcher
    assert "--network none" in launcher
    assert "--read-only" in launcher
    assert "--cap-drop ALL" in launcher
    assert "--security-opt no-new-privileges" in launcher
    assert "--device cuda" in launcher
    assert "--require-cuda" in launcher
    assert "git status --porcelain --untracked-files=all" in launcher
    assert "RUNPOD_API_KEY" not in launcher


def test_training_runner_keeps_runpod_default_but_allows_local_workload_identity() -> None:
    runner = RUN.read_text(encoding="utf-8")
    assert 'WORKLOAD_ID = "orbitune-runpod-training-canary-v1"' in runner
    assert 'parser.add_argument("--workload-id", default=WORKLOAD_ID)' in runner
    assert '"workload_id": args.workload_id' in runner
