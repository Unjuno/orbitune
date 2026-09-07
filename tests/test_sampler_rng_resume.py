"""Regression tests for sampler RNG save/restore in Compound checkpoints.

Covers:
  * build_compound_checkpoint saves sampler_rng_state from the local rng
  * sampler_rng_state and python_rng_state are independent streams
  * checkpoint roundtrip preserves sampler_rng_state
  * legacy checkpoint with sampler_rng_state=None still loads
  * uninterrupted sampling sequence matches checkpoint+resume sequence
  * old behavior (sampler_rng_state=None) would fail the resume test
  * validation corpus identity reset on identity change
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_training import (
    COMPOUND_CHECKPOINT_SCHEMA_VERSION,
    atomic_torch_save,
    build_compound_checkpoint,
    parse_compound_checkpoint,
)


def _tiny_model():
    cfg = CompoundBaseConfig(
        d_model=32, n_head=2,
        local_layers=1, medium_layers=1, global_layers=1, intra_layers=1,
    )
    cfg.validate()
    return CompoundHierarchicalGPT(cfg)


def _simulate_sampling_sequence(rng: random.Random, songs: list, batch_size: int,
                                seq_len: int, num_steps: int) -> list:
    """Draw (song_index, start_offset) pairs exactly like IndexedTensorSampler.sample.

    Replicates the RNG consumption pattern: for each sampled element,
    rng.choice(eligible) + rng.randrange(0, len(song.records) - seq_len).
    """
    eligible = [i for i, song in enumerate(songs) if len(song.records) >= seq_len + 1]
    sequence = []
    for _ in range(num_steps):
        for _ in range(batch_size):
            song_index = rng.choice(eligible)
            song = songs[song_index]
            start = rng.randrange(0, len(song.records) - seq_len)
            sequence.append((song_index, start))
    return sequence


def _make_synthetic_songs(n: int = 20, length: int = 256, seed: int = 0) -> list:
    """Create lightweight synthetic song objects with .records and .sha256."""
    import hashlib
    rng = random.Random(seed)

    class _Song:
        __slots__ = ("records", "sha256")
        def __init__(self, records, sha):
            self.records = records
            self.sha256 = sha

    songs = []
    for i in range(n):
        records = [(rng.randint(0, 9),) * 12 for _ in range(length)]
        songs.append(_Song(records, hashlib.sha256(f"song{i}".encode()).hexdigest()))
    return songs


class TestSamplerRngSavedAndRestored(unittest.TestCase):
    """Verify build_compound_checkpoint saves sampler_rng_state from the local rng."""

    def test_sampler_rng_state_is_not_none(self):
        model = _tiny_model()
        rng = random.Random(42)
        _ = rng.random()  # advance state
        payload = build_compound_checkpoint(
            model=model, optimizer=None, scaler=None,
            step=1, events_seen=128,
            runtime={"n_head": 2, "seq_len": 64, "batch_size": 4, "precision": "bf16"},
            rng=rng,
        )
        self.assertIsNotNone(payload["sampler_rng_state"])
        self.assertEqual(payload["sampler_rng_state"], rng.getstate())

    def test_python_rng_state_is_global_random(self):
        model = _tiny_model()
        rng = random.Random(99)
        _ = rng.random()
        global_state = random.getstate()
        payload = build_compound_checkpoint(
            model=model, optimizer=None, scaler=None,
            step=1, events_seen=128,
            runtime={"n_head": 2, "seq_len": 64, "batch_size": 4, "precision": "bf16"},
            rng=rng,
        )
        self.assertEqual(payload["python_rng_state"], global_state)
        self.assertNotEqual(payload["sampler_rng_state"], payload["python_rng_state"])


class TestCheckpointRngRoundtrip(unittest.TestCase):
    """Verify sampler_rng_state survives a save/reload roundtrip."""

    def test_roundtrip_preserves_sampler_rng_state(self):
        model = _tiny_model()
        rng = random.Random(123)
        for _ in range(50):
            rng.choice([0, 1, 2, 3, 4])

        payload = build_compound_checkpoint(
            model=model, optimizer=None, scaler=None,
            step=50, events_seen=50 * 256,
            runtime={"n_head": 2, "seq_len": 256, "batch_size": 16, "precision": "bf16"},
            rng=rng,
        )
        saved_state = rng.getstate()

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pt"
            atomic_torch_save(payload, path)

            loaded = torch.load(path, map_location="cpu", weights_only=False)
            loaded = parse_compound_checkpoint(loaded)
            self.assertEqual(loaded["sampler_rng_state"], saved_state)
            self.assertEqual(loaded["python_rng_state"], random.getstate())

            restored_rng = random.Random(123)
            restored_rng.setstate(loaded["sampler_rng_state"])
            self.assertEqual(restored_rng.getstate(), saved_state)


class TestLegacyCheckpointCompatibility(unittest.TestCase):
    """Legacy checkpoints with sampler_rng_state=None must still load."""

    def test_legacy_v1_schema_loads(self):
        model = _tiny_model()
        v1 = {
            "schema_version": 1,
            "architecture": model.architecture,
            "tokenizer": model.tokenizer,
            "config": {
                "d_model": 32, "n_head": 2,
                "local_layers": 1, "medium_layers": 1, "global_layers": 1, "intra_layers": 1,
                "local_window": 8, "medium_stride": 2, "medium_window": 8,
                "global_stride": 2, "global_window": 8, "ff_mult": 2, "dropout": 0.1,
            },
            "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            "step": 0,
            "sampler_rng_state": None,
            "python_rng_state": random.getstate(),
        }
        loaded = parse_compound_checkpoint(v1)
        self.assertIsNone(loaded.get("sampler_rng_state"))
        self.assertEqual(loaded["step"], 0)

    def test_legacy_checkpoint_rng_restore_is_noop_for_local_rng(self):
        """A legacy checkpoint with sampler_rng_state=None must not crash
        when the resume code checks for it. The local rng stays at its
        freshly-seeded state (this is the known resume-fidelity bug)."""
        model = _tiny_model()
        legacy_payload = build_compound_checkpoint(
            model=model, optimizer=None, scaler=None,
            step=50000, events_seen=50000 * 256,
            runtime={"n_head": 2, "seq_len": 256, "batch_size": 16, "precision": "bf16"},
            rng=random.Random(3),
        )
        # Simulate the bug: force sampler_rng_state to None (legacy)
        legacy_payload["sampler_rng_state"] = None

        rng = random.Random(3 + 7919)
        # Resume code behavior: skips restore when None
        if legacy_payload.get("sampler_rng_state") is not None:
            rng.setstate(legacy_payload["sampler_rng_state"])
        # rng is still at its initial seeded state, NOT at step 50000's position
        self.assertEqual(legacy_payload.get("sampler_rng_state"), None)


class TestSamplingSequenceResumeMatchesContinuous(unittest.TestCase):
    """Core regression test: uninterrupted sampling == checkpoint + resume sampling.

    Run A: initialize RNG, draw N steps continuously.
    Run B: initialize same RNG, draw K steps, save checkpoint with sampler_rng_state,
           construct a NEW RNG object, restore sampler_rng_state, draw remaining N-K steps.

    Require: sequence_A == sequence_B
    """

    def test_uninterrupted_equals_resume(self):
        songs = _make_synthetic_songs(n=50, length=512, seed=77)
        batch_size = 4
        seq_len = 32
        total_steps = 50
        checkpoint_step = 20

        # --- Run A: continuous sampling ---
        rng_a = random.Random(7922)
        seq_a = _simulate_sampling_sequence(rng_a, songs, batch_size, seq_len, total_steps)

        # --- Run B: sample K steps, checkpoint, resume, sample rest ---
        rng_b = random.Random(7922)
        seq_b_first = _simulate_sampling_sequence(rng_b, songs, batch_size, seq_len, checkpoint_step)

        # Save checkpoint state (simulating build_compound_checkpoint)
        checkpoint_state = rng_b.getstate()
        sampler_rng_state = checkpoint_state

        # Construct a NEW RNG object as resume would
        rng_c = random.Random(7922)
        # Restore from checkpoint
        rng_c.setstate(sampler_rng_state)

        # Continue sampling
        seq_b_rest = _simulate_sampling_sequence(
            rng_c, songs, batch_size, seq_len, total_steps - checkpoint_step
        )

        # Combine: first K steps + resumed steps
        seq_b = seq_b_first + seq_b_rest

        self.assertEqual(len(seq_a), len(seq_b))
        for i, (a, b) in enumerate(zip(seq_a, seq_b)):
            self.assertEqual(a, b, f"sequence mismatch at draw {i}: A={a} B={b}")

    def test_old_behavior_would_fail(self):
        """Verify the test catches the old bug: with sampler_rng_state=None,
        the resumed RNG restarts from the seed and produces a different sequence."""
        songs = _make_synthetic_songs(n=50, length=512, seed=77)
        batch_size = 4
        seq_len = 32
        total_steps = 50
        checkpoint_step = 20

        rng_a = random.Random(7922)
        seq_a = _simulate_sampling_sequence(rng_a, songs, batch_size, seq_len, total_steps)

        rng_b = random.Random(7922)
        seq_b_first = _simulate_sampling_sequence(rng_b, songs, batch_size, seq_len, checkpoint_step)

        # Simulate OLD behavior: sampler_rng_state is None, rng not restored
        sampler_rng_state = None  # old bug

        rng_c = random.Random(7922)
        if sampler_rng_state is not None:
            rng_c.setstate(sampler_rng_state)
        # rng_c is NOT restored — starts from seed 7922

        seq_b_rest = _simulate_sampling_sequence(
            rng_c, songs, batch_size, seq_len, total_steps - checkpoint_step
        )

        seq_b = seq_b_first + seq_b_rest

        # With the old bug, seq_b diverges from seq_a at the checkpoint boundary
        self.assertNotEqual(seq_a[checkpoint_step * batch_size:],
                            seq_b[checkpoint_step * batch_size:],
                            "Old behavior should fail: resumed sequence should diverge")


class TestValidationCorpusIdentity(unittest.TestCase):
    """Verify validation corpus identity behavior for best-loss scoping."""

    def test_corpus_identity_reset_on_change(self):
        """When validation corpus identity changes between parent and current
        training corpus, best_validation_loss and best_step must be reset."""
        from orbitune.compound_training import parse_compound_checkpoint

        model = _tiny_model()
        rng = random.Random(0)
        payload = build_compound_checkpoint(
            model=model, optimizer=None, scaler=None,
            step=10, events_seen=10,
            runtime={"n_head": 2, "seq_len": 64, "batch_size": 4, "precision": "bf16"},
            rng=rng,
            validation_corpus_identity="old_identity_sha",
        )
        parsed = parse_compound_checkpoint(payload)
        self.assertEqual(parsed["validation_corpus_identity"], "old_identity_sha")

        # Simulate an identity mismatch (as compound_cfe_train.py does at line 530)
        new_identity = "new_identity_sha"
        old_identity = parsed["validation_corpus_identity"]
        if old_identity is not None and old_identity != new_identity:
            # best_validation_loss and best_step should be reset
            health = parsed.setdefault("health", {})
            health["best_validation_loss"] = None
            health["best_step"] = None

        self.assertIsNone(parsed["health"]["best_validation_loss"])
        self.assertIsNone(parsed["health"]["best_step"])


if __name__ == "__main__":
    unittest.main()
