"""Tests for the long-run production-readiness audit (P0/P1 fixes).

These tests cover:
  * atomic checkpoint save
  * CUDA RNG state normalize/restore (CPU safe)
  * validation window plan determinism
  * assert_runtime_compatible (strict and override)
  * build_compound_checkpoint / parse_compound_checkpoint schema v1 + v2
  * synthetic-guard in compound_cfe_train
  * state-semantics equivalence for CompoundHierarchicalGPT at window=1
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

from dataclasses import asdict as _asdict

from orbitune import compound_training as ct
from orbitune.compound_base import (
    CompoundBaseConfig,
    CompoundHierarchicalGPT,
)
from orbitune.compound_training import load_compound_jsonl, CompoundSong


# ---------------------------------------------------------------------------
# P0: atomic_torch_save
# ---------------------------------------------------------------------------

class TestAtomicSave(unittest.TestCase):
    def test_atomic_save_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pt"
            payload = {"a": 1, "b": [1, 2, 3]}
            ct.atomic_torch_save(payload, path)
            self.assertTrue(path.exists())
            loaded = torch.load(path, weights_only=False, map_location="cpu")
            self.assertEqual(loaded, payload)
            # No leftover tmp file.
            self.assertFalse((path.parent / (path.name + ".tmp")).exists())

    def test_atomic_save_does_not_overwrite_original_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pt"
            original = {"x": "original"}
            ct.atomic_torch_save(original, path)
            before = torch.load(path, weights_only=False, map_location="cpu")
            self.assertEqual(before, original)
            # Simulate a failure mid-write by making the payload unserialisable.
            class _Boom:
                def __reduce__(self):
                    raise RuntimeError("nope")
            with self.assertRaises(Exception):
                ct.atomic_torch_save({"boom": _Boom()}, path)
            after = torch.load(path, weights_only=False, map_location="cpu")
            self.assertEqual(after, original)


# ---------------------------------------------------------------------------
# P0: normalize_cuda_rng_state / restore_cuda_rng_state
# ---------------------------------------------------------------------------

class TestCudaRngState(unittest.TestCase):
    def test_normalize_returns_byte_tensor_list(self):
        # Real CUDA get_rng_state_all returns a list of CUDA uint8 tensors.
        # We simulate that here with CPU uint8.
        state = torch.zeros(65536, dtype=torch.uint8)
        normalized = ct.normalize_cuda_rng_state([state])
        self.assertIsInstance(normalized, list)
        self.assertIsInstance(normalized[0], torch.Tensor)
        self.assertEqual(normalized[0].dtype, torch.uint8)

    def test_normalize_handles_none(self):
        # normalize returns an empty list for None (sentinel: "no state").
        self.assertEqual(ct.normalize_cuda_rng_state(None), [])

    def test_restore_cuda_rng_state_byte_only(self):
        # On CPU-only machines, the function must not raise; on CUDA, it
        # must accept the real RNG state size. We just exercise the code
        # path with the actual current state to confirm the function
        # round-trips without corruption.
        from orbitune.compound_training import COMPOUND_TOKENIZER_ABI  # noqa
        from orbitune.tokenizer.compound_event import COMPOUND_RECORD_WIDTH  # noqa
        if torch.cuda.is_available():
            real = torch.cuda.get_rng_state_all()
            ct.restore_cuda_rng_state(real)
            after = torch.cuda.get_rng_state_all()
            for a, b in zip(real, after):
                self.assertTrue(torch.equal(a, b))
        else:
            # CPU-only: the function is a no-op.
            ct.restore_cuda_rng_state(None)
            ct.restore_cuda_rng_state([torch.zeros(8, dtype=torch.uint8)])  # not called, but no-op


# ---------------------------------------------------------------------------
# P0: validation window plan determinism
# ---------------------------------------------------------------------------

class TestValidationPlanDeterminism(unittest.TestCase):
    def _songs(self, n=4, length=128):
        # Build a tiny JSONL on disk and load via the real loader to get
        # proper CompoundSong objects with .records attribute and sha256.
        from orbitune.compound_training import COMPOUND_TOKENIZER_ABI
        from orbitune.tokenizer.compound_event import COMPOUND_RECORD_WIDTH
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "songs.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for _ in range(n):
                    import random as _r
                    rng = _r.Random(0)
                    records = [
                        [rng.randint(0, 9), rng.randint(0, 15), rng.randint(0, 6), rng.randint(0, 15),
                         rng.randint(0, 127), rng.randint(0, 127), rng.randint(1, 127), 0,
                         rng.randint(0, 6), rng.randint(0, 15), rng.randint(0, 7), rng.randint(0, 7)]
                        for _ in range(length)
                    ]
                    row = {
                        "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
                        "record_width": COMPOUND_RECORD_WIDTH,
                        "records": records,
                        "name": f"song{_}",
                    }
                    f.write(json.dumps(row) + "\n")
            return load_compound_jsonl(str(path))

    def test_same_seed_same_hash(self):
        songs = self._songs()
        plan_a = ct.capture_validation_window_plan(songs, validation_seed=1234, batches=3, batch_size=2, seq_len=64)
        plan_b = ct.capture_validation_window_plan(songs, validation_seed=1234, batches=3, batch_size=2, seq_len=64)
        self.assertEqual(plan_a["window_hash"], plan_b["window_hash"])
        # The actual windows should also be byte-identical.
        self.assertEqual(plan_a["plan"], plan_b["plan"])

    def test_different_seed_different_hash(self):
        songs = self._songs()
        plan_a = ct.capture_validation_window_plan(songs, validation_seed=1234, batches=3, batch_size=2, seq_len=64)
        plan_b = ct.capture_validation_window_plan(songs, validation_seed=1235, batches=3, batch_size=2, seq_len=64)
        self.assertNotEqual(plan_a["window_hash"], plan_b["window_hash"])


# ---------------------------------------------------------------------------
# P1: assert_runtime_compatible
# ---------------------------------------------------------------------------

class TestRuntimeCompatibility(unittest.TestCase):
    def test_identical_runtime_passes(self):
        runtime = {"d_model": 224, "n_head": 7, "local_window": 64, "precision": "bf16"}
        ct.assert_runtime_compatible(runtime, cli_runtime=dict(runtime), allow_runtime_change=False)

    def test_changed_runtime_fails_strict(self):
        runtime = {"d_model": 224, "n_head": 7, "local_window": 64, "precision": "bf16"}
        with self.assertRaises(SystemExit):
            ct.assert_runtime_compatible(runtime, cli_runtime=dict(runtime, n_head=8), allow_runtime_change=False)

    def test_changed_runtime_passes_with_override(self):
        runtime = {"d_model": 224, "n_head": 7, "local_window": 64, "precision": "bf16"}
        # Must not raise.
        ct.assert_runtime_compatible(runtime, cli_runtime=dict(runtime, n_head=8), allow_runtime_change=True)


# ---------------------------------------------------------------------------
# P1: build / parse compound checkpoint schema
# ---------------------------------------------------------------------------

class TestCheckpointSchema(unittest.TestCase):
    def _small_model(self):
        cfg = CompoundBaseConfig(d_model=32, n_head=2, local_layers=1, medium_layers=1, global_layers=1, intra_layers=1)
        return CompoundHierarchicalGPT(cfg)

    def test_schema_v2_roundtrip(self):
        import random as _r
        model = self._small_model()
        rng = _r.Random(0)
        payload = ct.build_compound_checkpoint(
            model=model,
            optimizer=None,
            scaler=None,
            step=42,
            events_seen=42 * 64,
            runtime={"n_head": 2, "seq_len": 64, "batch_size": 4, "precision": "bf16", "causal_fastpath": True},
            rng=rng,
            validation_history=[{"step": 10, "val_loss": 1.5}],
        )
        self.assertEqual(payload["schema_version"], ct.COMPOUND_CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(payload["step"], 42)
        self.assertEqual(payload["validation_history"], [{"step": 10, "val_loss": 1.5}])
        loaded = ct.parse_compound_checkpoint(payload)
        self.assertEqual(loaded["step"], 42)

    def test_legacy_v1_loads_with_warning(self):
        # Build a v1-shaped payload (no schema_version, no extra fields).
        model = self._small_model()
        v1 = {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "config": _asdict(model.config),
            "step": 0,
        }
        loaded = ct.parse_compound_checkpoint(v1)
        self.assertEqual(loaded["step"], 0)


# ---------------------------------------------------------------------------
# P0: synthetic guard
# ---------------------------------------------------------------------------

class TestSyntheticGuard(unittest.TestCase):
    def test_synthetic_path_rejected(self):
        from compound_cfe_train import _looks_like_synthetic
        # Path is rejected by the caller when _looks_like_synthetic returns True
        # and the caller has not passed --allow-synthetic. We verify the matcher.
        self.assertTrue(_looks_like_synthetic("data/continuous/synthetic_compound.jsonl"))
        self.assertTrue(_looks_like_synthetic("benchmarks/fixtures/cfe/synthetic_compound.jsonl"))
        self.assertTrue(_looks_like_synthetic("data/cfe/foo.jsonl"))
        # Real path must not match.
        self.assertFalse(_looks_like_synthetic("data/v1/train.jsonl"))
        self.assertFalse(_looks_like_synthetic("data/real_midi/validation.jsonl"))


# ---------------------------------------------------------------------------
# P1: training vs generation state semantics equivalence
# ---------------------------------------------------------------------------

class TestStateSemanticsEquivalence(unittest.TestCase):
    def _build(self):
        cfg = CompoundBaseConfig(
            d_model=32, n_head=2,
            local_layers=1, medium_layers=1, global_layers=1, intra_layers=1,
            local_window=8, medium_stride=2, medium_window=4,
            global_stride=2, global_window=4,
        )
        cfg.validate()
        return CompoundHierarchicalGPT(cfg).eval()

    def test_window1_advance_stream_matches_encode(self):
        """For the very first 16 events, encode() and advance_stream() must
        produce the same per-position context vector (up to FP tolerance)."""
        torch.manual_seed(7)
        model = self._build()
        records = torch.randint(0, 30, (1, 16, 12), dtype=torch.long)
        with torch.no_grad():
            enc_context = model.encode(records)[0]
            state = model.initial_stream_state()
            streamed_contexts = []
            for t in range(records.shape[1]):
                ctx = model.advance_stream(records[0, t], state)[0]
                streamed_contexts.append(ctx)
            streamed = torch.stack(streamed_contexts)
        diff = (enc_context - streamed).abs().max().item()
        self.assertLess(diff, 1e-4, msg=f"window-1 equivalence broken: max abs diff {diff}")


if __name__ == "__main__":
    unittest.main()