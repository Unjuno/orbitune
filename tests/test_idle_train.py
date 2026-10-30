from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from tools.idle_gate import CpuProbe, GpuProbe, MachineGate, safe_read
from tools.idle_train import past_deadline, tail_step


def _reading(util: float, free_mib: float = 15500.0, cpu: float | None = 10.0) -> dict[str, float | str]:
    """Default to a fully free GPU so only the field under test varies."""
    reading: dict[str, float | str] = {
        "util_percent": util,
        "free_mib": free_mib,
        "used_mib": 16384.0 - free_mib,
        "total_mib": 16384.0,
        "source": "test",
    }
    if cpu is not None:
        reading["cpu_percent"] = cpu
    return reading


def _gate(**overrides: object) -> MachineGate:
    settings: dict[str, object] = {
        "util_threshold": 5.0,
        "min_free_mib": 14000.0,
        "max_cpu_percent": 0.0,
        "idle_seconds": 60.0,
        "busy_seconds": 40.0,
        "poll_seconds": 20.0,
    }
    settings.update(overrides)
    return MachineGate(**settings)  # type: ignore[arg-type]


def test_gate_needs_sustained_quiet_before_starting() -> None:
    gate = _gate(idle_seconds=60.0, poll_seconds=20.0)
    assert gate.update(_reading(0.0)).idle is False
    assert gate.update(_reading(0.0)).idle is False
    assert gate.update(_reading(0.0)).idle is True


def test_gate_resets_quiet_after_one_busy_sample() -> None:
    gate = _gate(idle_seconds=60.0, poll_seconds=20.0)
    gate.update(_reading(0.0))
    gate.update(_reading(0.0))
    state = gate.update(_reading(95.0))
    assert state.idle is False
    assert state.block == "gpu_util"
    assert state.quiet_seconds == 0.0


def test_gate_requires_sustained_busy_before_yielding() -> None:
    gate = _gate(busy_seconds=40.0, poll_seconds=20.0)
    assert gate.update(_reading(95.0)).busy is False
    assert gate.update(_reading(95.0)).busy is True


def test_gate_busy_latches_closed_again_when_the_gpu_frees_up() -> None:
    gate = _gate(busy_seconds=40.0, poll_seconds=20.0)
    gate.update(_reading(95.0))
    gate.update(_reading(95.0))
    assert gate.update(_reading(0.0)).busy is False


def test_gate_blocks_on_vram_headroom() -> None:
    # A run that peaks near 15.3 GiB of 16 GiB must not start unless the GPU is
    # essentially empty, so low free memory has to count as blocked on its own.
    gate = _gate(min_free_mib=14000.0)
    assert gate.blocked_by(_reading(0.0, free_mib=6000.0)) == "vram"
    assert gate.blocked_by(_reading(0.0, free_mib=15500.0)) is None


def test_cpu_guard_is_opt_in() -> None:
    assert _gate(max_cpu_percent=0.0).blocked_by(_reading(0.0, cpu=99.0)) is None
    assert _gate(max_cpu_percent=50.0).blocked_by(_reading(0.0, cpu=95.0)) == "cpu"


def test_deadline_parses_local_hh_mm() -> None:
    assert past_deadline(None) is False
    assert past_deadline("00:00") is True
    assert past_deadline("23:59") is False


def test_tail_step_finds_the_highest_step(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        '{"event": "trainer_launch", "resume": "x"}\n'
        '{"epoch_index": 0, "event": "epoch_sampler_resumed"}\n'
        '{"event": "training_stopped", "reason": "idle_yield", "step": 273038}\n',
        encoding="utf-8")
    assert tail_step(log) == 273038


def test_tail_step_is_none_before_the_trainer_logs_a_step(tmp_path: Path) -> None:
    # Startup logs sampler_config / epoch_sampler_resumed but no step, so the
    # supervisor can tell "still loading the corpus" from "training".
    log = tmp_path / "train.log"
    log.write_text('{"event": "sampler_config", "batch_size": 64}\n', encoding="utf-8")
    assert tail_step(log) is None


def test_tail_step_handles_a_missing_log(tmp_path: Path) -> None:
    assert tail_step(tmp_path / "absent.log") is None


def test_safe_read_swallows_a_broken_probe() -> None:
    class Broken:
        def sample(self) -> dict[str, float | str]:
            raise RuntimeError("nvml went away")

    assert safe_read(Broken(), None) is None  # type: ignore[arg-type]


def test_cpu_probe_needs_two_samples() -> None:
    assert CpuProbe().sample() is None


def test_gpu_probe_reports_a_reading() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no GPU probe available on this machine")
    reading = GpuProbe(0).sample()
    assert float(reading["util_percent"]) >= 0.0
    assert float(reading["total_mib"]) > 0.0


def test_foreign_processes_excludes_our_own_pids() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no GPU probe available on this machine")
    probe = GpuProbe(0)
    resident = [pid for pid, _ in probe.compute_processes()]
    assert all(pid not in {4242, 4243} for pid in probe.foreign_processes({4242, 4243}))
    assert resident == [pid for pid, _ in probe.compute_processes()]


def test_compute_processes_tolerates_unreported_memory() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no GPU probe available on this machine")
    # Windows WDDM reports per-process memory as [N/A]; the PID must still parse.
    for pid, memory in GpuProbe(0).compute_processes():
        assert pid > 0
        assert memory is None or memory >= 0.0
def test_gate_reset_requires_a_fresh_idle_window() -> None:
    gate = _gate()
    for _ in range(4):
        gate.update(_reading(0.0))
    assert gate.update(_reading(0.0)).idle is True
    gate.reset()
    state = gate.update(_reading(0.0))
    assert state.idle is False
    assert state.quiet_seconds == gate.poll_seconds


def test_gate_reset_clears_the_busy_latch_too() -> None:
    gate = _gate()
    for _ in range(4):
        gate.update(_reading(95.0))
    assert gate.update(_reading(95.0)).busy is True
    gate.reset()
    assert gate.update(_reading(95.0)).busy is False


def test_log_tail_is_bounded_at_the_window() -> None:
    from tools.idle_train import _log_tail

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "train.log"
        log.write_bytes(b"x" * 50_000 + b"tail-marker")
        tail = _log_tail(log, 1024)
        assert tail is not None
        assert len(tail) == 1024
        assert tail.endswith(b"tail-marker")


def test_log_tail_of_a_missing_log_is_none(tmp_path: Path) -> None:
    from tools.idle_train import _log_tail

    assert _log_tail(tmp_path / "absent.log", 1024) is None


def test_descendants_of_walks_the_full_chain() -> None:
    from tools.idle_train import descendants_of

    parents = {1: 0, 2: 1, 3: 1, 4: 2, 99: 3, 100: 90}
    assert descendants_of(1, parents) == {1, 2, 3, 4, 99}
    assert descendants_of(3, parents) == {3, 99}
    assert descendants_of(100, parents) == {100}


def test_log_tail_never_earlier_than_the_floor(tmp_path: Path) -> None:
    from tools.idle_train import _log_tail

    log = tmp_path / "train.log"
    log.write_bytes(b"" * 0 + b"old-oom-text!" + b"y" * 500)
    floor = log.stat().st_size
    assert _log_tail(log, 20000, floor=floor) == b""
    with log.open("a") as handle:
        handle.write("new-slice-went-clean")
    tail = _log_tail(log, 20000, floor=floor)
    assert tail == b"new-slice-went-clean"
    tail_from_old = _log_tail(log, 20000, floor=0)
    assert b"old-oom-text!" in tail_from_old
