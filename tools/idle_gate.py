"""Machine-idle gate shared by the idle-time training supervisor.

The GPU is sampled as a signal only. pynvml is preferred and nvidia-smi is the
fallback, so a broken NVML handle degrades instead of killing the supervisor.

One `MachineGate.update()` call per poll advances both directions: how long the
machine has been continuously quiet (use it to start work) and how long it has
been continuously blocked (use it to yield).
"""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from typing import Any


class GpuProbe:
    """GPU utilization and free VRAM. Uses pynvml, falls back to nvidia-smi."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._handle: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.source = "pynvml"
        except Exception:
            self.source = "nvidia-smi"

    def sample(self) -> dict[str, float | str]:
        if self._handle is not None:
            try:
                import pynvml

                rates = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return {
                    "util_percent": float(rates.gpu),
                    "free_mib": float(memory.free) / (1024 * 1024),
                    "used_mib": float(memory.used) / (1024 * 1024),
                    "total_mib": float(memory.total) / (1024 * 1024),
                    "source": self.source,
                }
            except Exception:
                self._handle = None
                self.source = "nvidia-smi"
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", str(self.index)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.splitlines()[0]
        fields = [part.strip() for part in output.split(",")]
        used, total = float(fields[1]), float(fields[2])
        return {
            "util_percent": float(fields[0]),
            "free_mib": total - used,
            "used_mib": used,
            "total_mib": total,
            "source": self.source,
        }

    def compute_processes(self) -> list[tuple[int, float | None]]:
        """CUDA compute PIDs resident on the GPU, with VRAM where the driver reports it.

        Used to tell "our own trainer is saturating the GPU" apart from "someone
        else has arrived", which utilization alone cannot distinguish. Returns an
        empty list when the GPU is free. Windows WDDM often reports per-process
        memory as ``[N/A]``, so memory is optional and never load-bearing.
        """
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits", "-i", str(self.index)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        processes: list[tuple[int, float | None]] = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if not fields or not fields[0].isdigit():
                continue
            try:
                memory: float | None = float(fields[1])
            except (IndexError, ValueError):
                memory = None
            processes.append((int(fields[0]), memory))
        return processes

    def foreign_processes(self, own_pids: set[int]) -> list[tuple[int, float | None]]:
        return [entry for entry in self.compute_processes() if entry[0] not in own_pids]


class _FileTimes(ctypes.Structure):
    _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    def ticks(self) -> int:
        return (self.high << 32) | self.low


class CpuProbe:
    """Whole-system CPU busy percentage via GetSystemTimes. No dependencies."""

    def __init__(self) -> None:
        self._idle = _FileTimes()
        self._kernel = _FileTimes()
        self._user = _FileTimes()
        self._previous: tuple[int, int] | None = None

    def sample(self) -> float | None:
        kernel32 = getattr(ctypes, "windll", None)
        if kernel32 is None:
            return None
        if not kernel32.kernel32.GetSystemTimes(  # type: ignore[union-attr]
                ctypes.byref(self._idle), ctypes.byref(self._kernel), ctypes.byref(self._user)):
            return None
        idle = self._idle.ticks()
        total = self._kernel.ticks() + self._user.ticks()
        if self._previous is None:
            self._previous = (idle, total)
            return None
        idle_delta = idle - self._previous[0]
        total_delta = total - self._previous[1]
        self._previous = (idle, total)
        if total_delta <= 0:
            return None
        return 100.0 * (total_delta - idle_delta) / total_delta


@dataclass(frozen=True)
class GateState:
    """Result of one gate poll."""

    block: str | None
    quiet_seconds: float
    busy_seconds: float
    idle: bool
    busy: bool


class MachineGate:
    """Latches sustained quiet and sustained busy from sampled machine state.

    Both edges require persistence so a single anomalous sample cannot start or
    abandon work. A training run additionally has its own in-flight latency, so
    `busy_seconds` is normally set above `poll_seconds` to tolerate one noisy
    sample before the supervisor yields.
    """

    def __init__(self, *, util_threshold: float, min_free_mib: float, max_cpu_percent: float,
                 idle_seconds: float, busy_seconds: float, poll_seconds: float) -> None:
        self.util_threshold = util_threshold
        self.min_free_mib = min_free_mib
        self.max_cpu_percent = max_cpu_percent
        self.idle_seconds = idle_seconds
        self.busy_seconds = busy_seconds
        self.poll_seconds = poll_seconds
        self._quiet = 0.0
        self._busy = 0.0

    def blocked_by(self, reading: dict[str, float | str]) -> str | None:
        if float(reading["util_percent"]) > self.util_threshold:
            return "gpu_util"
        if float(reading["free_mib"]) < self.min_free_mib:
            return "vram"
        cpu = reading.get("cpu_percent")
        if cpu is not None and self.max_cpu_percent > 0 and float(cpu) > self.max_cpu_percent:
            return "cpu"
        return None

    def update(self, reading: dict[str, float | str]) -> GateState:
        block = self.blocked_by(reading)
        if block is None:
            self._quiet += self.poll_seconds
            self._busy = 0.0
        else:
            self._busy += self.poll_seconds
            self._quiet = 0.0
        return GateState(
            block=block,
            quiet_seconds=self._quiet,
            busy_seconds=self._busy,
            idle=self._quiet >= self.idle_seconds,
            busy=self._busy >= self.busy_seconds,
        )

    def reset(self) -> None:
        """Clear both latches so a fresh sustained window is required before launching.

        The supervisor is blocked inside a slice while the trainer runs, so the
        gate is not being fed: after a slice ends, both accumulation counters
        would still hold the values that opened the previous launch, and the next
        slice would start immediately. Resetting after every slice makes the
        next launch wait for its own sustained empty window instead.
        """

        self._quiet = 0.0
        self._busy = 0.0


def read_machine(gpu: GpuProbe, cpu: CpuProbe | None) -> dict[str, float | str]:
    reading: dict[str, float | str] = gpu.sample()
    if cpu is not None:
        reading["cpu_percent"] = cpu.sample()
    return reading


def safe_read(gpu: GpuProbe, cpu: CpuProbe | None) -> dict[str, float | str] | None:
    """Sample the machine. A transient probe failure must not kill the supervisor."""
    try:
        return read_machine(gpu, cpu)
    except Exception:
        return None