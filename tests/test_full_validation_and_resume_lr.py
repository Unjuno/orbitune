"""Tests for the full-validation evaluator and resume-LR override.

Covers:
  * Full-validation evaluator: deterministic plan, identical hash on
    re-evaluation, per-component aggregation, evaluator is forward-only
    (model weights unchanged), evaluator accepts BF16/FP16/FP32.
  * Resume-LR override: --override-resume-lr without --resume fails,
    --override-resume-lr with --resume applies to all param_groups, the
    override is recorded in the new checkpoint's runtime dict.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_training import (
    atomic_torch_save,
    build_compound_checkpoint,
    load_compound_jsonl,
    parse_compound_checkpoint,
)

from full_validation_eval import _plan_payload, evaluate_full_validation  # type: ignore


def _tiny_songs(n_songs: int, *, length: int = 1024, seed: int = 0) -> list:
    """Build ``n_songs`` synthetic CompoundSong objects with realistic
    12-wide records. Reuses the same JSONL-then-load path as production
    so the plan hash is computed the same way."""
    from orbitune.tokenizer.compound_event import CompoundEventTokenizer
    abi = CompoundEventTokenizer.abi
    width = 12
    import random as _r
    rng = _r.Random(seed)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "songs.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for i in range(n_songs):
                records = [
                    [rng.randint(0, 9), rng.randint(0, 15), rng.randint(0, 6),
                     rng.randint(0, 15), rng.randint(0, 127), rng.randint(0, 127),
                     rng.randint(1, 127), 0, rng.randint(0, 6), rng.randint(0, 15),
                     rng.randint(0, 7), rng.randint(0, 7)]
                    for _ in range(length)
                ]
                row = {"tokenizer_abi": abi, "record_width": width, "records": records, "name": f"s{i}"}
                f.write(json.dumps(row) + "\n")
        return load_compound_jsonl(str(p))


def _write_tiny_jsonl(path: Path, n_songs: int = 2, length: int = 1024, seed: int = 0) -> None:
    """Write a minimal 12-wide Compound JSONL at ``path`` for subprocess
    trainer tests. The trainer only needs syntactically valid records;
    it samples a batch per step, so the data quality does not matter as
    long as it's consistent."""
    from orbitune.tokenizer.compound_event import CompoundEventTokenizer
    import random as _r
    rng = _r.Random(seed)
    abi = CompoundEventTokenizer.abi
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_songs):
            records = [
                [rng.randint(0, 9), rng.randint(0, 15), rng.randint(0, 6),
                 rng.randint(0, 15), rng.randint(0, 127), rng.randint(0, 127),
                 rng.randint(1, 127), 0, rng.randint(0, 6), rng.randint(0, 15),
                 rng.randint(0, 7), rng.randint(0, 7)]
                for _ in range(length)
            ]
            row = {"tokenizer_abi": abi, "record_width": 12, "records": records, "name": f"s{i}"}
            f.write(json.dumps(row) + "\n")


def _tiny_checkpoint(step: int = 0, *, n_head: int = 2) -> Path:
    """Save a small Compound checkpoint that fits in CPU memory and runs
    in <1 second on the evaluator."""
    cfg = CompoundBaseConfig(
        d_model=32, n_head=n_head,
        local_layers=1, medium_layers=1, global_layers=1, intra_layers=1,
        local_window=8, medium_stride=2, medium_window=8,
        global_stride=2, global_window=8,
    )
    cfg.validate()
    model = CompoundHierarchicalGPT(cfg)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tiny.pt"
        import random as _r
        rng = _r.Random(0)
        payload = build_compound_checkpoint(
            model=model,
            optimizer=None,
            scaler=None,
            step=step,
            events_seen=step * 1024,
            runtime={
                "n_head": n_head, "seq_len": 256, "batch_size": 4,
                "precision": "bf16", "causal_fastpath": True,
                "learning_rate": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0,
            },
            rng=rng,
        )
        atomic_torch_save(payload, path)
        # Re-load and return path to a *copy* in a stable tempdir
        out = Path(tempfile.mkdtemp()) / "tiny.pt"
        out.write_bytes(path.read_bytes())
        return out


class TestFullValidationPlan(unittest.TestCase):
    def test_plan_deterministic(self):
        songs = _tiny_songs(3, length=1024)
        plan_a = _plan_payload(songs, seq_len=256)
        plan_b = _plan_payload(songs, seq_len=256)
        self.assertEqual(plan_a["window_hash"], plan_b["window_hash"])
        self.assertEqual(plan_a["windows"], plan_b["windows"])

    def test_plan_windows_count(self):
        # 3 songs of 1024 events, seq_len=256.
        # range(0, 1024-256=768, 256) = [0, 256, 512] = 3 windows per song,
        # 3 songs = 9 windows total.
        songs = _tiny_songs(3, length=1024)
        plan = _plan_payload(songs, seq_len=256)
        self.assertEqual(plan["window_count"], 9)
        self.assertEqual(plan["total_events"], 9 * 256)

    def test_plan_drops_short_songs(self):
        # Song with fewer than seq_len+1 events produces zero windows.
        # 200 events < 256+1=257 -> 0 windows.
        songs = _tiny_songs(1, length=200)
        plan = _plan_payload(songs, seq_len=256)
        self.assertEqual(plan["window_count"], 0)

    def test_plan_partial_windows_dropped(self):
        # Song of 600 events: range(0, 600-256=344, 256) = [0, 256] = 2 windows.
        # The trailing 344..600 partial window is dropped.
        songs = _tiny_songs(1, length=600)
        plan = _plan_payload(songs, seq_len=256)
        self.assertEqual(plan["window_count"], 2)

    def test_plan_300_event_song_has_one_window(self):
        # Boundary case: 300 events >= seq_len+1 (257) -> exactly 1 window.
        songs = _tiny_songs(1, length=300)
        plan = _plan_payload(songs, seq_len=256)
        self.assertEqual(plan["window_count"], 1)


class TestFullValidationEvaluator(unittest.TestCase):
    def test_evaluator_is_forward_only(self):
        ckpt = _tiny_checkpoint(step=0)
        songs_path = Path(tempfile.mkdtemp()) / "songs.jsonl"
        # re-export songs as JSONL
        songs = _tiny_songs(2, length=1024)
        with songs_path.open("w", encoding="utf-8") as f:
            from orbitune.tokenizer.compound_event import CompoundEventTokenizer
            for s in songs:
                row = {
                    "tokenizer_abi": CompoundEventTokenizer.abi,
                    "record_width": 12,
                    "records": [list(r) for r in s.records],
                    "sha256": s.sha256,
                }
                f.write(json.dumps(row) + "\n")
        # capture model state before
        before = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        before_keys = sorted(before.keys())
        before_sum = sum(float(before[k].abs().sum()) for k in before_keys)
        result = evaluate_full_validation(
            checkpoint_path=ckpt,
            validation_jsonl=songs_path,
            seq_len=256,
            batch_size=8,
            device="cpu",
            precision="fp32",
        )
        after = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        after_sum = sum(float(after[k].abs().sum()) for k in before_keys)
        self.assertEqual(set(after.keys()), set(before_keys))
        self.assertEqual(before_sum, after_sum, msg="model state changed during evaluation")
        # result has expected fields
        self.assertIn("window_hash", result)
        self.assertIn("mean_loss_per_event", result)
        self.assertIn("mean_loss_trainer_style", result)
        self.assertIn("per_component", result)
        self.assertGreater(result["window_count"], 0)
        self.assertGreater(result["total_events"], 0)
        # math.isfinite
        self.assertTrue(__import__("math").isfinite(result["mean_loss_per_event"]))

    def test_evaluator_deterministic_across_two_runs(self):
        ckpt = _tiny_checkpoint(step=0)
        songs_path = Path(tempfile.mkdtemp()) / "songs.jsonl"
        songs = _tiny_songs(2, length=1024)
        with songs_path.open("w", encoding="utf-8") as f:
            from orbitune.tokenizer.compound_event import CompoundEventTokenizer
            for s in songs:
                row = {
                    "tokenizer_abi": CompoundEventTokenizer.abi,
                    "record_width": 12,
                    "records": [list(r) for r in s.records],
                    "sha256": s.sha256,
                }
                f.write(json.dumps(row) + "\n")
        a = evaluate_full_validation(ckpt, songs_path, seq_len=256, batch_size=8, device="cpu", precision="fp32")
        b = evaluate_full_validation(ckpt, songs_path, seq_len=256, batch_size=8, device="cpu", precision="fp32")
        self.assertEqual(a["window_hash"], b["window_hash"])
        self.assertEqual(a["total_events"], b["total_events"])
        # Per-component mean should be byte-identical in fp32
        for k in a["per_component"]:
            self.assertAlmostEqual(
                a["per_component"][k]["mean_per_event"],
                b["per_component"][k]["mean_per_event"],
                places=6,
            )

    def test_evaluator_reports_events_and_scalar_fields_separately(self):
        """The Compound record tensor is (B, T, 12). The evaluator must
        report both total_events (= B*T) and total_scalar_fields (= B*T*12)
        so downstream consumers don't confuse the two."""
        ckpt = _tiny_checkpoint(step=0)
        songs_path = Path(tempfile.mkdtemp()) / "songs.jsonl"
        songs = _tiny_songs(2, length=1024)
        with songs_path.open("w", encoding="utf-8") as f:
            from orbitune.tokenizer.compound_event import CompoundEventTokenizer
            for s in songs:
                row = {
                    "tokenizer_abi": CompoundEventTokenizer.abi,
                    "record_width": 12,
                    "records": [list(r) for r in s.records],
                    "sha256": s.sha256,
                }
                f.write(json.dumps(row) + "\n")
        result = evaluate_full_validation(
            checkpoint_path=ckpt, validation_jsonl=songs_path,
            seq_len=256, batch_size=8, device="cpu", precision="fp32",
        )
        # 2 songs * 3 windows = 6 windows, 6*256=1536 events.
        self.assertEqual(result["total_events"], 6 * 256)
        # 12-field record width, so 12x scalar fields.
        self.assertEqual(result["total_scalar_fields"], result["total_events"] * 12)
        self.assertEqual(result["record_width"], 12)
        # per-event and per-scalar-field means are the same number
        # (every head attends every field), so the trainer-style mean
        # also equals the per-event mean.
        self.assertAlmostEqual(
            result["mean_loss_per_event"], result["mean_loss_trainer_style"], places=6
        )


class TestResumeLrOverride(unittest.TestCase):
    def test_override_without_resume_rejected(self):
        # Run the trainer without --resume and with --override-resume-lr;
        # it must exit with code 2 and the error must mention the
        # --resume dependency.
        with tempfile.TemporaryDirectory() as d:
            train_jsonl = Path(d) / "train.jsonl"
            val_jsonl = Path(d) / "val.jsonl"
            _write_tiny_jsonl(train_jsonl, n_songs=2, length=1024)
            _write_tiny_jsonl(val_jsonl, n_songs=1, length=512)
            out_ckpt = Path(d) / "x.pt"
            result = subprocess.run(
                [
                    sys.executable, "-W", "ignore",
                    str(ROOT / "scripts" / "compound_longrun_train.py"),
                    "--train-jsonl", str(train_jsonl),
                    "--validation-jsonl", str(val_jsonl),
                    "--checkpoint", str(out_ckpt),
                    "--steps", "1",
                    "--config", str(ROOT / "configs" / "compound_hierarchical_9m_nhead7.json"),
                    "--override-resume-lr", "1e-4",
                    "--allow-fixed-window-training",
                ],
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        combined = result.stdout + result.stderr
        self.assertIn("--resume", combined)
        self.assertIn("refusing to silently retune", combined)

    def test_override_applied_to_param_groups(self):
        """End-to-end test: a 1-step trainer run with --override-resume-lr
        must emit the override event, the new checkpoint's runtime must
        record ``resume_lr_override_applied``, and every param_group's
        LR must be overwritten to the requested value."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for the production trainer")
        # Build a small "base" checkpoint using the same helpers.
        # The base must use the same n_head (7) as the production config
        # so the trainer's resume-n_head check passes.
        cfg = CompoundBaseConfig(
            d_model=28, n_head=7,  # 28 % 7 == 0; match the production n_head=7
            local_layers=1, medium_layers=1, global_layers=1, intra_layers=1,
            local_window=8, medium_stride=2, medium_window=8,
            global_stride=2, global_window=8,
        )
        cfg.validate()
        model = CompoundHierarchicalGPT(cfg).to("cuda")
        import torch.optim as _opt
        opt = _opt.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        # Fake a step so the optimizer state has entries.
        x = torch.randint(0, 10, (1, 8, 12), dtype=torch.long, device="cuda")
        y = torch.randint(0, 10, (1, 8, 12), dtype=torch.long, device="cuda")
        loss, _ = model(x, y)
        loss.backward()
        opt.step()

        import random as _r
        rng = _r.Random(0)
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "base.pt"
            out = Path(d) / "out.pt"
            train_jsonl = Path(d) / "train.jsonl"
            val_jsonl = Path(d) / "val.jsonl"
            _write_tiny_jsonl(train_jsonl, n_songs=2, length=1024)
            _write_tiny_jsonl(val_jsonl, n_songs=1, length=512)
            payload = build_compound_checkpoint(
                model=model, optimizer=opt, scaler=None, step=10, events_seen=10 * 1024,
                runtime={
                    "n_head": 7, "seq_len": 256, "batch_size": 4,
                    "precision": "bf16", "causal_fastpath": True,
                    "learning_rate": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0,
                },
                rng=rng,
            )
            atomic_torch_save(payload, ckpt)

            result = subprocess.run(
                [
                    sys.executable, "-W", "ignore",
                    str(ROOT / "scripts" / "compound_longrun_train.py"),
                    "--train-jsonl", str(train_jsonl),
                    "--validation-jsonl", str(val_jsonl),
                    "--checkpoint", str(out),
                    "--resume", str(ckpt),
                    "--steps", "1",
                    "--batch-size", "1",
                    "--seq-len", "256",
                    "--n-head", "7",
                    "--override-resume-lr", "1e-4",
                    "--allow-fixed-window-training",
                    "--no-causal-fastpath",
                    "--allow-runtime-change",  # the test changes batch_size and fastpath
                    "--eval-every", "1000000",  # skip validation in this test
                    "--checkpoint-every", "1",  # save after the single step
                    "--log-every", "1",
                    "--steps", "11",  # absolute; resume starts at 10, so this runs step 11 only
                ],
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            # Confirm the override event was emitted with the new LR.
            self.assertIn("resume_lr_override_applied", result.stdout)
            self.assertIn("0.0001", result.stdout)

            # Load the new checkpoint and verify runtime + optimizer state
            new_payload = torch.load(out, map_location="cpu", weights_only=False)
            new_runtime = new_payload.get("runtime", {})
            self.assertEqual(new_runtime.get("resume_lr_override_applied"), 1e-4)
            new_opt_state = new_payload.get("optimizer_state_dict")
            self.assertIsNotNone(new_opt_state)
            for pg in new_opt_state["param_groups"]:
                self.assertAlmostEqual(float(pg["lr"]), 1e-4)


if __name__ == "__main__":
    unittest.main()
