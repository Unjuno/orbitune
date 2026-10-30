"""Idle-gated training supervisor for the Compound Base.

Keeps the production trainer running whenever the GPU is free, and yields the GPU
the moment another job wants it. Unlike C:\\orbitune-production\\epoch_a2_watchdog.py,
which supervises restarts unconditionally, this supervisor refuses to start
while the GPU is occupied and asks the trainer to stand down when that changes.

Why a sentinel instead of a kill: the trainer checkpoints every 30M events, about
1076 steps and 35 minutes. `Process.terminate()` on Windows is an immediate
TerminateProcess, so stopping that way throws away the whole interval. This
supervisor passes `--yield-sentinel` to the trainer, which checks it at the step
boundary and exits through its normal post-loop path, writing a final checkpoint.
Measured on this machine: startup 33s, 2.1s per step, graceful exit 67s.

Yield attribution uses the resident CUDA process set rather than utilization,
because a running trainer saturates the GPU itself and utilization alone cannot
tell our own load from a neighbour's.

Safety policy is not reimplemented here. Checkpoint validation, the approved
resume parents, runtime compatibility and snapshot/milestone retention all come
from the production watchdog module, which is imported by path, so this file can
never drift from the rules the run was approved under. Supervisor events are
appended to the same watchdog.jsonl as a single timeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.idle_gate import CpuProbe, GpuProbe, MachineGate, safe_read

DEFAULT_WATCHDOG = Path(r"C:\orbitune-production\epoch_a2_watchdog.py")


def load_watchdog(path: Path) -> ModuleType:
    """Import the production watchdog so resume/snapshot policy stays single-sourced."""
    if not path.is_file():
        raise SystemExit(f"watchdog module not found: {path}")
    spec = importlib.util.spec_from_file_location("epoch_a2_watchdog", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import watchdog module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def past_deadline(target: str | None) -> bool:
    if not target:
        return False
    hour, minute = (int(part) for part in target.split(":"))
    now = time.localtime()
    return now.tm_hour * 60 + now.tm_min >= hour * 60 + minute


def request_yield(sentinel: Path, watch: ModuleType, reason: str, **fields: object) -> None:
    """Drop the sentinel and record the ask. The trainer exits within one step."""
    already = sentinel.exists()
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(reason, encoding="utf-8")
    watch.record("yield_requested", reason=reason, sentinel=str(sentinel),
                 already_pending=already, **fields)


def clear_yield(sentinel: Path, watch: ModuleType) -> None:
    try:
        sentinel.unlink()
    except FileNotFoundError:
        return
    watch.record("yield_cleared", sentinel=str(sentinel))


def wait_for_exit(proc: subprocess.Popen, watch: ModuleType, timeout: float,
                  *, label: str) -> bool:
    """Wait for a graceful exit. Escalates to terminate() only on timeout."""
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if time.monotonic() > deadline:
            watch.record("graceful_exit_timed_out", label=label, pid=proc.pid,
                         timeout_seconds=timeout)
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            return False
        time.sleep(2)
    return True


_STEP_PATTERN = re.compile(rb'"step":\s*(\d+)')


def tail_step(log: Path, window: int = 200_000) -> int | None:
    """Highest step number visible in the tail of the trainer log, or None.

    Used to tell "the trainer is still loading the corpus and building the
    optimizer state" apart from "the trainer is training". Yield accounting must
    not start during startup: measured startup on this machine is 33s, which is
    longer than a whole yield budget, so a busy timer armed at launch would
    trigger before the first step and the run would never advance at all.
    """
    tail = _log_tail(log, window)
    if tail is None:
        return None
    steps = [int(match) for match in _STEP_PATTERN.findall(tail)]
    return max(steps) if steps else None


def _log_tail(log: Path, window: int, floor: int = 0) -> bytes | None:
    """Bounded tail read of a log that may have grown to gigabytes.

    `floor` marks bytes written before the slice this check belongs to: the log
    is shared and append-only across slices, so detection text from an earlier
    slice must not be attributed to this one. Returns b"" when nothing was
    appended past the floor yet.
    """
    try:
        with log.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            start = max(floor, size - window)
            if size <= start:
                return b""
            handle.seek(start)
            return handle.read()
    except OSError:
        return None


def _process_parents() -> dict[int, int] | None:
    """One pid -> parent-pid snapshot of the Windows process tree, or None.

    Used to find the real CUDA worker when the venv launcher stub re-spawns the
    interpreter: the trainer's actual GPU process is a grandchild pid that
    neither `proc.pid` nor `os.getpid()` knows about.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             'Get-CimInstance Win32_Process | ForEach-Object { "{0},{1}" -f $_.ProcessId,$_.ParentProcessId }'],
            capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    parents: dict[int, int] = {}
    for line in result.stdout.splitlines():
        left, comma, right = line.strip().partition(",")
        if comma and left.isdigit() and right.isdigit():
            parents[int(left)] = int(right)
    return parents or None


def descendants_of(root: int, parents: dict[int, int]) -> set[int]:
    """Every pid reachable from `root` by following child links down."""
    found = {root}
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        for pid, parent_pid in parents.items():
            if parent_pid == parent and pid not in found:
                found.add(pid)
                frontier.append(pid)
    return found


def _own_family() -> set[int]:
    """Our process family (us plus every descendant), for CUDA ownership tests.

    An empty result on probe failure means the caller falls back to the direct
    pids it already knows about, which under-detects but does not over-count
    foreign load.
    """
    parents = _process_parents()
    if parents is None:
        return set()
    return descendants_of(os.getpid(), parents)


def supervise_slice(*, watch: ModuleType, proc: subprocess.Popen, args: argparse.Namespace,
                    sentinel: Path, gpu: GpuProbe, cpu: CpuProbe, stopping: list[bool],
                    resume_step: int) -> tuple[str, bool]:
    """Watch a running trainer.

    Returns (why it stopped, whether the trainer got past its first step). The
    second element drives the relaunch backoff: a slice that never started must
    not be retried hot, or the supervisor would spin launch/timeout forever.
    """
    own = {proc.pid, os.getpid()}
    busy_seconds = 0.0
    last_mtime: float | None = None
    started_at = time.monotonic()
    trained = False
    stall_logged = False
    while proc.poll() is None:
        # The venv interpreter on this machine is a launcher stub that re-spawns
        # the real worker under the base Python, so CUDA runs in a grandchild
        # whose pid is neither proc.pid nor our own. Resolve the full process
        # family fresh on every poll: the worker may not exist yet at launch.
        own = {os.getpid(), proc.pid} | _own_family()
        source = watch.HEALTHY if watch.HEALTHY.is_file() else watch.CHECKPOINT
        if source.is_file():
            mtime = source.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                payload = watch.load_valid(source)
                if payload is not None:
                    watch.preserve_snapshots(payload)

        if not trained:
            step_now = tail_step(watch.LOG)
            if step_now is not None and step_now > resume_step:
                trained = True
                watch.record("trainer_training", pid=proc.pid, first_logged_step=step_now,
                             startup_seconds=round(time.monotonic() - started_at, 1))
            elif time.monotonic() - started_at > args.startup_timeout_seconds:
                if not stall_logged:
                    stall_logged = True
                    watch.record("trainer_startup_stalled", pid=proc.pid,
                                 startup_seconds=round(time.monotonic() - started_at, 1))

        if watch.STOP.exists():
            reason = "stop_requested"
        elif past_deadline(args.until):
            reason = "until_deadline"
        elif stopping[0]:
            reason = "signal"
        else:
            reason = ""

        if reason:
            request_yield(sentinel, watch, reason, pid=proc.pid, trained=trained)
            wait_for_exit(proc, watch, args.graceful_timeout, label=reason)
            clear_yield(sentinel, watch)
            return reason, trained

        try:
            foreign = gpu.foreign_processes(own)
        except Exception as error:
            foreign = []
            watch.record("foreign_process_probe_failed", error=repr(error))
        if foreign and trained:
            busy_seconds += args.poll_seconds
            if busy_seconds >= args.yield_after_busy_seconds:
                request_yield(sentinel, watch, "foreign_gpu_process", pid=proc.pid,
                              busy_seconds=busy_seconds, foreign=foreign,
                              grace_seconds=args.graceful_timeout)
                wait_for_exit(proc, watch, args.graceful_timeout, label="foreign_gpu_process")
                clear_yield(sentinel, watch)
                return "foreign_gpu_process", trained
        else:
            busy_seconds = 0.0
        time.sleep(args.poll_seconds)
    return "exited", trained


def run(args: argparse.Namespace, watch: ModuleType, stopping: list[bool]) -> int:
    watch.RUN.mkdir(parents=True, exist_ok=True)
    sentinel = watch.RUN / "YIELD"
    clear_yield(sentinel, watch)

    gpu = GpuProbe(args.gpu_index)
    cpu = CpuProbe()
    gate = MachineGate(
        util_threshold=args.util_threshold,
        min_free_mib=args.min_free_mib,
        max_cpu_percent=args.max_cpu_percent,
        idle_seconds=args.idle_seconds,
        busy_seconds=args.yield_after_busy_seconds,
        poll_seconds=args.poll_seconds,
    )

    watch.record(
        "idle_supervisor_started", pid=os.getpid(), source=str(watch.SOURCE),
        commit=watch.LOCAL_COMMIT, run=str(watch.RUN), yield_sentinel=str(sentinel),
        gpu_probe=gpu.source, util_threshold=args.util_threshold,
        min_free_mib=args.min_free_mib, max_cpu_percent=args.max_cpu_percent,
        idle_seconds=args.idle_seconds, poll_seconds=args.poll_seconds,
        yield_after_busy_seconds=args.yield_after_busy_seconds,
        graceful_timeout=args.graceful_timeout, max_failures=args.max_failures,
        until=args.until, heartbeat_seconds=args.heartbeat_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        no_progress_backoff=args.no_progress_backoff,
        allow_foreign_processes=args.allow_foreign_processes,
        launch_requires_empty_gpu=bool(getattr(args, "launch_requires_empty_gpu", False)),
    )

    failures = 0
    last_failed_step: int | None = None
    waiting_since: float | None = None
    last_heartbeat = time.monotonic()
    slices = 0
    yielded = 0

    while not stopping[0]:
        if watch.STOP.exists():
            watch.record("supervisor_stop_requested", slices=slices, yielded=yielded)
            break
        if past_deadline(args.until):
            watch.record("supervisor_deadline_reached", until=args.until,
                         slices=slices, yielded=yielded)
            break

        if args.dry_run:
            # Reported ahead of the idle gate so it answers immediately even
            # while the GPU is occupied.
            resume, payload, from_parent = watch.latest_resume()
            command = watch.trainer_command(resume, from_parent) + ["--yield-sentinel", str(sentinel)]
            reading = safe_read(gpu, cpu) or {}
            watch.record("dry_run", resume=str(resume), resume_step=int(payload["step"]),
                         resume_events=int(payload["events_seen"]),
                         epoch=watch.epoch_progress(payload),
                         running_now=bool(gpu.foreign_processes({os.getpid()})),
                         command=command, **reading)
            print(json.dumps({"resume": str(resume), "resume_step": int(payload["step"]),
                              "resume_events": int(payload["events_seen"]),
                              "command": command, "machine": reading}, indent=2))
            return 0

        reading = safe_read(gpu, cpu)
        if reading is None:
            watch.record("probe_failed")
            time.sleep(args.poll_seconds)
            continue
        try:
            foreign = gpu.foreign_processes({os.getpid()} | _own_family())
        except Exception as error:
            foreign = None
            watch.record("foreign_process_probe_failed", error=repr(error))
        # A resident neighbour is not the same thing as a busy GPU: browsers and
        # other apps park a CUDA context with a few MiB and 0% utilization for
        # hours, and a presence-only gate then blocks every slice (observed 7h
        # and 14h stalls with a free card). Before a slice starts the trainer is
        # not running, so the utilization sample belongs to the neighbours: util
        # plus free VRAM is the honest "is the card spare" test. The VRAM floor
        # still protects a neighbour that actually holds memory, and
        # --launch-requires-empty-gpu restores the old all-or-nothing rule.
        # foreign=None means the probe failed, in which case only the utilization
        # and VRAM rules apply.
        needs_empty = bool(getattr(args, "launch_requires_empty_gpu", False))
        occupied = bool(foreign) and needs_empty and not args.allow_foreign_processes
        state = gate.update(reading)
        block = "foreign_gpu_process" if occupied else state.block
        can_start = state.idle and not occupied
        if not can_start:
            if waiting_since is None:
                waiting_since = time.monotonic()
                watch.record("waiting_for_idle", block=block, foreign=foreign, **reading)
            if time.monotonic() - last_heartbeat >= args.heartbeat_seconds:
                last_heartbeat = time.monotonic()
                watch.record("heartbeat_waiting", waited_seconds=int(
                    time.monotonic() - (waiting_since or time.monotonic())),
                    block=block, foreign=foreign,
                    util_percent=reading["util_percent"], free_mib=reading["free_mib"],
                    last_step=last_failed_step)
            time.sleep(args.poll_seconds)
            continue
        if waiting_since is not None:
            watch.record("machine_idle", waited_seconds=int(time.monotonic() - waiting_since),
                         foreign=foreign, **reading)
            waiting_since = None

        resume, payload, from_parent = watch.latest_resume()
        step = int(payload["step"])
        watch.preserve_snapshots(payload)
        launch_offset = watch.LOG.stat().st_size
        command = watch.trainer_command(resume, from_parent) + ["--yield-sentinel", str(sentinel)]
        clear_yield(sentinel, watch)
        env = os.environ.copy()
        env["ORBITUNE_SOURCE_COMMIT"] = watch.LOCAL_COMMIT
        env["PYTHONPATH"] = str(watch.SOURCE)

        with watch.LOG.open("a", encoding="utf-8", buffering=1) as output:
            output.write("\n")
            output.write(json.dumps({"event": "trainer_launch", "supervisor": "idle",
                                     "resume": str(resume)}) + "\n")
            proc = subprocess.Popen(command, cwd=watch.SOURCE, env=env, stdout=output,
                                    stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        slices += 1
        watch.record("trainer_started", pid=proc.pid, resume=str(resume), resume_step=step,
                     resume_events=int(payload["events_seen"]), slice=slices, command=command)
        outcome, trained = supervise_slice(
            watch=watch, proc=proc, args=args, sentinel=sentinel,
            gpu=gpu, cpu=cpu, stopping=stopping, resume_step=step)
        # The gate went un-fed while the slice ran, so its latches still carry
        # the state that opened this launch. Clearing them here keeps the next
        # launch from starting until the card has been quiet again for a whole
        # idle window; hot relaunching was what lost the race against a foreign
        # job attaching during the trainer's ~33 s startup and OOM'd at startup.
        gate.reset()
        exit_code = int(proc.returncode if proc.returncode is not None else -1)
        if outcome != "exited":
            yielded += 1

        latest_path, latest, _ = watch.latest_resume()
        latest_step = int(latest["step"])
        watch.record("trainer_exited", outcome=outcome, exit_code=exit_code,
                     resume_step=step, latest_step=latest_step, trained=trained,
                     latest_events=int(latest["events_seen"]),
                     checkpoint=str(latest_path), slices=slices, yielded=yielded)

        if outcome in ("stop_requested", "until_deadline", "signal"):
            return 0

        # The log has grown to several gigabytes across a run; read only its
        # tail. `_log_tail` mirrors `tail_step`'s bounded read.
        tail_bytes = _log_tail(watch.LOG, 20000, floor=launch_offset)
        tail = tail_bytes.decode("utf-8", errors="replace").lower() if tail_bytes is not None else ""
        corrupt = ("non-finite" in tail or '"loss": nan' in tail
                   or '"grad_norm": nan' in tail)
        oomed = "out of memory" in tail and not corrupt
        if oomed:
            # A foreign job can attach during the trainer's startup, after the
            # pre-launch checks have already run; the trainer then dies before
            # reaching a step boundary, writing nothing. Transient, and harmless
            # to the checkpoint because nothing was written. Back off and retry
            # like any other no-progress failure; only persistent OOM at the
            # same state (`repeated`, or the failure budget) ends the run.
            failures += 1
            last_failed_step = latest_step
            watch.record("trainer_oom_stop", exit_code=exit_code, latest_step=latest_step,
                         failures=failures)
            if failures >= args.max_failures:
                return 4
            time.sleep(min(60, failures * 20))
            continue
        if corrupt:
            watch.record("trainer_unsafe_stop", exit_code=exit_code, latest_step=latest_step)
            return exit_code or 4
        if outcome == "foreign_gpu_process":
            if trained:
                failures = 0
                last_failed_step = None
                continue
            # A foreign job was already resident when the slice launched, so the
            # pre-launch gate let us start and startup never got a clean window.
            # Back off instead of retrying hot.
            watch.record("slice_yielded_before_first_step", latest_step=latest_step,
                         backoff_seconds=args.no_progress_backoff)
            time.sleep(args.no_progress_backoff)
            continue
        repeated = last_failed_step == latest_step
        if repeated:
            watch.record("trainer_repeated_step", latest_step=latest_step,
                         failures=failures + 1)
        if repeated or failures >= args.max_failures:
            return exit_code or 4
        failures += 1
        last_failed_step = latest_step
        time.sleep(min(60, failures * 20))

    watch.record("supervisor_exit", slices=slices, yielded=yielded)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Compound Base only while the GPU is free, yielding when it is not")
    parser.add_argument("--watchdog", type=Path, default=DEFAULT_WATCHDOG,
                        help="production watchdog module that owns resume and snapshot policy")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--util-threshold", type=float, default=5.0)
    # This run peaks at ~15.3 GiB reserved of 16 GiB, so the GPU has to be
    # essentially empty before a slice starts or the trainer will OOM.
    parser.add_argument("--min-free-mib", type=float, default=14000.0)
    parser.add_argument("--max-cpu-percent", type=float, default=0.0,
                        help="0 disables the CPU guard")
    parser.add_argument("--allow-foreign-processes", action="store_true",
                        help="Start a slice even while another CUDA process is resident. "
                             "Off by default: this geometry reserves ~15.3 GiB of 16 GiB, "
                             "so a co-resident job can be starved.")
    parser.add_argument("--launch-requires-empty-gpu", action="store_true",
                        help="Refuse to start a slice while any other CUDA process is "
                             "resident, even at 0%% utilization. Off by default: a neighbour "
                             "that parks a CUDA context for hours would otherwise block "
                             "training indefinitely. Yielding to a neighbour that arrives "
                             "mid-slice is unaffected.")
    parser.add_argument("--idle-seconds", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--yield-after-busy-seconds", type=float, default=40.0)
    parser.add_argument("--graceful-timeout", type=float, default=900.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--no-progress-backoff", type=float, default=120.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=600.0)
    parser.add_argument("--max-failures", type=int, default=5)
    parser.add_argument("--until", help="local HH:MM after which the supervisor exits")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the resume point and command, then exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0 or args.idle_seconds <= 0:
        raise SystemExit("--poll-seconds and --idle-seconds must be positive")
    watch = load_watchdog(args.watchdog)

    stopping = [False]

    def handle(signum: int, _frame: object) -> None:
        stopping[0] = True
        watch.record("supervisor_signal", signal=signum)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, handle)

    try:
        lock_fd = os.open(watch.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        watch.record("duplicate_supervisor_blocked", lock=str(watch.LOCK))
        return 3
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    try:
        return run(args, watch, stopping)
    finally:
        try:
            watch.LOCK.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())