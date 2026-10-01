from __future__ import annotations

import copy
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_indexed import INDEX_FORMAT, SHARDED_INDEX_FORMAT
from orbitune.compound_longrun import build_longrun_checkpoint
from orbitune.indexed_sampling import IndexedSequentialSongChunkSampler
from orbitune.tbptt_run_support import (
    guard_transition_outputs,
    load_training_sources,
    target_final_step,
    validate_resume_contract,
    validation_identity,
)
from orbitune.tokenizer.compound_event import CompoundEventTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _store(path: Path, *, dtype="int32-le", count=13, sha="a" * 64, report=False):
    path.mkdir(parents=True)
    rows = np.zeros((count, 12), dtype=np.uint8 if dtype == "uint8" else "<i4")
    rows[:, 4] = 60 + np.arange(count) % 12
    rows[:, 6] = 80
    rows[:, 3] = 2
    rows[:, 9] = 3
    rows.tofile(path / "records.i32")
    (path / "songs.jsonl").write_text(json.dumps({"path": "not-needed.mid", "sha256": sha,
        "offset": 0, "length": count, "source_id": "test", "sampling_weight": 1.0}) + "\n")
    metadata = {"schema_version": 1, "format": INDEX_FORMAT,
        "tokenizer_abi": CompoundEventTokenizer.abi, "record_width": 12,
        "dtype": dtype, "split": "train", "manifest_sha256": sha,
        "records_file": "records.i32", "songs_file": "songs.jsonl", "songs": 1, "events": count}
    (path / ("report.json" if report else "index.json")).write_text(json.dumps(metadata))
    if report:
        (path / "DONE").touch()
    return path


def _sharded(path: Path, *, dtype="uint8", report=False):
    part = _store(path / "part-000", dtype=dtype, sha="b" * 64, report=report)
    (path / "index_manifest.json").write_text(json.dumps({"schema_version": 1,
        "format": SHARDED_INDEX_FORMAT, "total_songs": 1,
        "shards": [{"path": str(part.resolve())}]}))
    return path


@pytest.mark.parametrize("report", [False, True])
def test_mixed_flat_int32_and_sharded_uint8_keep_memmaps_and_order(tmp_path, report):
    aria = _store(tmp_path / "aria")
    giga = _sharded(tmp_path / "giga", report=report)
    sources = load_training_sources(f"{aria},{giga}")
    assert sources.indexed and len(sources.songs) == 2
    assert [s.sha256 for s in sources.songs] == ["a" * 64, "b" * 64]
    assert sources.songs[0].records._records.dtype == np.dtype("<i4")
    assert sources.songs[1].records._records.dtype == np.dtype("uint8")
    assert all(isinstance(song.records._records, np.memmap) for song in sources.songs)
    assert all(not song.records._records.flags.writeable for song in sources.songs)
    explicit = load_training_sources(f"{aria / 'index.json'},{giga / 'index_manifest.json'}")
    assert sources.identity == explicit.identity
    assert sources.identity != load_training_sources(f"{giga},{aria}").identity
    assert load_training_sources(aria / "songs.jsonl").indexed


def test_no_narrowing_of_int32_record_fields(tmp_path):
    path = _store(tmp_path / "aria")
    records = np.memmap(path / "records.i32", mode="r+", dtype="<i4", shape=(13, 12))
    records[0, 0], records[0, 4] = 4, 999
    records.flush()
    sources = load_training_sources(path)
    sampler = IndexedSequentialSongChunkSampler(sources.songs, batch_size=1, seq_len=4, rng=random.Random(1))
    batch = sampler.sample("cpu")
    assert batch.inputs.dtype == torch.int64
    assert batch.inputs[0, 0, 4] == 999


def test_mixed_source_sampler_resume_preserves_next_chunk(tmp_path):
    aria = _store(tmp_path / "aria")
    giga = _sharded(tmp_path / "giga")
    sources = load_training_sources(f"{aria},{giga}")
    rng = random.Random(7)
    a = IndexedSequentialSongChunkSampler(sources.songs, batch_size=2, seq_len=4, rng=rng)
    a.sample("cpu")
    state, rng_state = a.state_dict(), rng.getstate()
    expected = a.sample("cpu")
    restored_rng = random.Random()
    restored_rng.setstate(rng_state)
    b = IndexedSequentialSongChunkSampler(sources.songs, batch_size=2, seq_len=4, rng=restored_rng)
    b.load_state_dict(state)
    actual = b.sample("cpu")
    assert actual.offsets == expected.offsets
    assert actual.song_indices == expected.song_indices
    torch.testing.assert_close(actual.inputs, expected.inputs, rtol=0, atol=0)
    torch.testing.assert_close(actual.reset_mask, expected.reset_mask, rtol=0, atol=0)
    reordered = IndexedSequentialSongChunkSampler(list(reversed(sources.songs)), batch_size=2,
                                                 seq_len=4, rng=random.Random(7))
    with pytest.raises(ValueError, match="corpus identity"):
        reordered.load_state_dict(state)


@pytest.mark.parametrize("corruption", ["range", "size", "count", "abi"])
def test_report_only_shard_fails_closed_on_bad_metadata(tmp_path, corruption):
    root = _sharded(tmp_path / "giga", report=True)
    part = root / "part-000"
    meta = json.loads((part / "report.json").read_text())
    if corruption == "range":
        row = json.loads((part / "songs.jsonl").read_text())
        row["offset"] = 1
        (part / "songs.jsonl").write_text(json.dumps(row) + "\n")
    elif corruption == "size":
        with (part / "records.i32").open("ab") as f:
            f.write(b"x")
    elif corruption == "count":
        meta["songs"] = 2
    else:
        meta["tokenizer_abi"] = "wrong"
    (part / "report.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_training_sources(root)


def test_source_duplicates_empty_entries_and_raw_mix_rejected(tmp_path):
    flat = _store(tmp_path / "aria")
    with pytest.raises(ValueError, match="duplicate"):
        load_training_sources(f"{flat},{flat / 'index.json'}")
    with pytest.raises(ValueError, match="empty path"):
        load_training_sources(f"{flat},")
    raw = tmp_path / "raw.jsonl"
    raw.write_text("{}\n")
    with pytest.raises(ValueError, match="cannot mix"):
        load_training_sources(f"{flat},{raw}")
    sharded = tmp_path / "overlap"
    sharded.mkdir()
    (sharded / "index_manifest.json").write_text(json.dumps({"schema_version": 1,
        "format": SHARDED_INDEX_FORMAT, "total_songs": 1, "shards": [{"path": str(flat)}]}))
    with pytest.raises(ValueError, match="duplicate indexed record store"):
        load_training_sources(f"{flat},{sharded}")


def test_validation_identity_binds_song_tables_geometry_and_selection(tmp_path):
    path = _store(tmp_path / "val")
    source = load_training_sources(path)
    identity = validation_identity(source, seq_len=4, max_songs=2)
    assert identity != validation_identity(source, seq_len=8, max_songs=2)
    assert identity != validation_identity(source, seq_len=4, max_songs=0)
    row = json.loads((path / "songs.jsonl").read_text())
    row["sha256"] = "c" * 64
    (path / "songs.jsonl").write_text(json.dumps(row) + "\n")
    assert source.identity != load_training_sources(path).identity


@pytest.mark.parametrize("batch,seq", [(16, 256), (4, 64), (1, 32)])
def test_four_billion_target_inherits_exposure_not_global_step_geometry(batch, seq):
    start_step, start_events, target = 104750, 429056000, 4096000000
    final = target_final_step(start_step=start_step, start_events=start_events,
                             batch_size=batch, seq_len=seq, steps=None, target_events=target)
    actual = start_events + (final - start_step) * batch * seq
    assert target <= actual < target + batch * seq
    if batch == 16:
        assert final == 1000000
    else:
        assert final != 1000000


def test_event_budget_rounds_up_without_partial_optimizer_batch():
    assert target_final_step(start_step=2, start_events=8, batch_size=2, seq_len=2,
                             steps=None, target_events=17) == 5
    assert target_final_step(start_step=5, start_events=20, batch_size=2, seq_len=2,
                             steps=None, target_events=17) == 5
    assert target_final_step(start_step=5, start_events=20, batch_size=2, seq_len=2,
                             steps=10, target_events=None) == 10
    with pytest.raises(ValueError):
        target_final_step(start_step=0, start_events=0, batch_size=1, seq_len=1, steps=2, target_events=2)
    with pytest.raises(ValueError):
        target_final_step(start_step=0, start_events=0, batch_size=0, seq_len=1, steps=None, target_events=2)


def test_transition_rejects_parent_output_including_healthy_alias(tmp_path):
    parent = tmp_path / "model.healthy.pt"
    parent.write_bytes(b"immutable")
    with pytest.raises(ValueError, match="aliases"):
        guard_transition_outputs(parent, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="aliases"):
        guard_transition_outputs(parent, parent)
    guard_transition_outputs(parent, tmp_path / "new" / "model.pt")
    assert parent.read_bytes() == b"immutable"


def _trainer():
    spec = importlib.util.spec_from_file_location("tbptt_runtime_test", ROOT / "scripts" / "compound_tbptt_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def test_real_tbptt_loop_event_stop_and_exact_cpu_resume(tmp_path, monkeypatch):
    """Exercise the actual trainer, AdamW, RNG, sampler and carried-state save.

    CPU fp32 test evidence is deliberately not a CUDA/bf16 performance claim.
    """
    trainer = _trainer()
    monkeypatch.setattr(trainer.base, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(trainer.base, "optimizer_for", lambda m, lr, wd:
                        (torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd), False))
    monkeypatch.setattr(trainer.base, "cuda_stats", lambda: {})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    sources = _store(tmp_path / "data", count=17)
    cfg = CompoundBaseConfig(d_model=16, n_head=2, local_layers=1, medium_layers=1,
        global_layers=1, intra_layers=1, ff_mult=2, dropout=0.1,
        local_window=4, medium_stride=2, medium_window=2, global_stride=2, global_window=2)
    torch.manual_seed(9)
    model = CompoundHierarchicalGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    parent = tmp_path / "parent.pt"
    torch.save(build_longrun_checkpoint(model=model, optimizer=optimizer, scaler=None,
        step=100, events_seen=400, runtime={"precision": "fp32"}, sampler_rng=random.Random(17)), parent)
    parent_bytes = parent.read_bytes()

    def run(out, resume, target, transition=False):
        argv = ["--train-source", str(sources), "--validation-source", str(sources),
            "--checkpoint", str(out), "--resume", str(resume), "--target-events", str(target),
            "--batch-size", "1", "--seq-len", "2", "--precision", "fp32", "--no-causal-fastpath",
            "--checkpoint-every", "2", "--log-every", "2", "--eval-every", "0", "--allow-synthetic"]
        if transition:
            argv += ["--override-resume-lr", "0.0001"]
        trainer.train(trainer.build_parser().parse_args(argv))
        return torch.load(out, map_location="cpu", weights_only=False)

    full = run(tmp_path / "full" / "model.pt", parent, 410, True)
    split_path = tmp_path / "split" / "model.pt"
    intermediate = run(split_path, parent, 404, True)
    assert intermediate["step"] == 102 and intermediate["events_seen"] == 404
    split = run(split_path, split_path, 410)
    assert full["step"] == split["step"] == 105  # final save even off cadence
    assert full["events_seen"] == split["events_seen"] == 410  # not 12 times larger
    assert full["runtime"]["tbptt_origin_events_seen"] == split["runtime"]["tbptt_origin_events_seen"] == 400
    for key in ("model_state_dict", "optimizer_state_dict", "torch_rng_state", "python_rng_state",
                "sampler_rng_state", "tbptt_sampler_state", "tbptt_stream_states"):
        _assert_tree_equal(full[key], split[key])
    assert parent.read_bytes() == parent_bytes
    for key in ("sampler_rng_state", "tbptt_stream_states", "optimizer_state_dict"):
        damaged = copy.deepcopy(split)
        damaged[key] = None
        with pytest.raises(ValueError, match="missing"):
            validate_resume_contract(damaged, split["runtime"])
    drift = dict(split["runtime"], validation_protocol_identity="changed")
    with pytest.raises(ValueError, match="validation_protocol_identity"):
        validate_resume_contract(split, drift)
    damaged = copy.deepcopy(split)
    damaged["tbptt_stream_states"][0]["steps"] += 1
    with pytest.raises(ValueError, match="sampler offset"):
        validate_resume_contract(damaged, split["runtime"])


def test_cli_target_modes_are_mutually_exclusive():
    parser = _trainer().build_parser()
    common = ["--train-source", "train", "--validation-source", "val", "--checkpoint", "new.pt"]
    args = parser.parse_args(common + ["--target-events", "4096000000"])
    assert args.target_events == 4096000000 and args.steps is None and args.n_head is None
    with pytest.raises(SystemExit):
        parser.parse_args(common + ["--target-events", "10", "--steps", "1"])
