from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import torch

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_tbptt import (
    SequentialSongChunkSampler,
    batch_stream_states_from_cpu,
    batch_stream_states_to_cpu,
    detach_batch_stream_states,
    encode_tbptt_chunk,
    initial_batch_stream_states,
    tbptt_loss,
)
from orbitune.compound_training import CompoundSong

ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> CompoundBaseConfig:
    cfg = CompoundBaseConfig(
        d_model=32,
        n_head=2,
        local_layers=1,
        medium_layers=1,
        global_layers=1,
        intra_layers=1,
        ff_mult=2,
        dropout=0.0,
        local_window=8,
        medium_stride=2,
        medium_window=4,
        global_stride=2,
        global_window=4,
    )
    cfg.validate()
    return cfg


def _records(count: int, pitch_offset: int = 0) -> tuple[tuple[int, ...], ...]:
    rows = []
    for i in range(count):
        rows.append(
            (
                0,  # NOTE
                0,
                0,
                min(15, i % 16),
                48 + (i + pitch_offset) % 24,
                0,
                64 + i % 32,
                0,
                0,
                1 + i % 15,
                0,
                0,
            )
        )
    return tuple(rows)


def _song(name: str, count: int, pitch_offset: int = 0) -> CompoundSong:
    return CompoundSong(
        path=name,
        sha256=name,
        tokenizer_abi=COMPOUND_TOKENIZER_ABI,
        records=_records(count, pitch_offset),
    )


def test_tbptt_values_match_generation_advance_stream_in_eval():
    torch.manual_seed(17)
    model = CompoundHierarchicalGPT(_cfg()).eval()
    records = torch.tensor(_records(16), dtype=torch.long)[None]

    states = initial_batch_stream_states(model, 1)
    with torch.no_grad():
        chunk_context, _ = encode_tbptt_chunk(model, records, states)

        generation_state = model.initial_stream_state()
        generated = []
        for t in range(records.shape[1]):
            generated.append(model.advance_stream(records[0, t], generation_state)[0])
        generation_context = torch.stack(generated, dim=0)[None]

    torch.testing.assert_close(chunk_context, generation_context, rtol=1e-5, atol=1e-6)


def test_arbitrary_chunk_partition_is_value_equivalent_with_carried_state():
    torch.manual_seed(23)
    model = CompoundHierarchicalGPT(_cfg()).eval()
    records = torch.tensor(_records(19), dtype=torch.long)[None]

    with torch.no_grad():
        full_states = initial_batch_stream_states(model, 1)
        full, _ = encode_tbptt_chunk(model, records, full_states)

        split_states = initial_batch_stream_states(model, 1)
        pieces = []
        for start, end in ((0, 3), (3, 10), (10, 11), (11, 19)):
            value, split_states = encode_tbptt_chunk(model, records[:, start:end], split_states)
            split_states = detach_batch_stream_states(split_states)
            pieces.append(value)
        split = torch.cat(pieces, dim=1)

    torch.testing.assert_close(full, split, rtol=1e-5, atol=1e-6)


def test_reset_mask_resets_only_selected_lane():
    torch.manual_seed(29)
    model = CompoundHierarchicalGPT(_cfg()).eval()
    first = torch.tensor([_records(6), _records(6, 7)], dtype=torch.long)
    second = torch.tensor([_records(4, 2), _records(4, 9)], dtype=torch.long)

    states = initial_batch_stream_states(model, 2)
    with torch.no_grad():
        _, states = encode_tbptt_chunk(model, first, states)
        assert states[0].steps == 6 and states[1].steps == 6
        _, states = encode_tbptt_chunk(
            model,
            second,
            states,
            reset_mask=torch.tensor([True, False]),
        )
    assert states[0].steps == 4
    assert states[1].steps == 10


def test_tbptt_backward_and_boundary_detach():
    torch.manual_seed(31)
    model = CompoundHierarchicalGPT(_cfg()).train()
    records = torch.tensor(_records(9), dtype=torch.long)[None]
    states = initial_batch_stream_states(model, 1)
    loss, _, states = tbptt_loss(model, records[:, :-1], records[:, 1:], states)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)

    detached = detach_batch_stream_states(states)
    hidden = (
        detached[0].medium_buffer
        + detached[0].medium_history
        + detached[0].global_buffer
        + detached[0].global_history
    )
    assert all(tensor.grad_fn is None for tensor in hidden)
    if detached[0].memory is not None:
        assert all(tensor.grad_fn is None for tensor in detached[0].memory)


def test_sequential_sampler_never_crosses_song_boundary_and_marks_reset():
    songs = [_song("a", 13), _song("b", 17, 3)]
    rng = random.Random(5)
    sampler = SequentialSongChunkSampler(songs, batch_size=1, seq_len=4, rng=rng)

    first = sampler.sample("cpu")
    assert first.reset_mask.tolist() == [True]
    assert first.offsets == (0,)
    assert first.inputs.shape == (1, 4, 12)

    second = sampler.sample("cpu")
    assert second.reset_mask.tolist() == [False]
    assert second.offsets == (4,)

    saw_reset = False
    for _ in range(10):
        batch = sampler.sample("cpu")
        if bool(batch.reset_mask[0]):
            saw_reset = True
            assert batch.offsets == (0,)
            break
    assert saw_reset


def test_sampler_state_and_rng_roundtrip_produce_same_next_chunk():
    songs = [_song("a", 25), _song("b", 29, 4), _song("c", 31, 8)]
    rng_a = random.Random(11)
    sampler_a = SequentialSongChunkSampler(songs, batch_size=2, seq_len=4, rng=rng_a)
    sampler_a.sample("cpu")

    sampler_state = sampler_a.state_dict()
    rng_state = rng_a.getstate()
    expected = sampler_a.sample("cpu")

    rng_b = random.Random()
    rng_b.setstate(rng_state)
    sampler_b = SequentialSongChunkSampler(songs, batch_size=2, seq_len=4, rng=rng_b)
    sampler_b.load_state_dict(sampler_state)
    actual = sampler_b.sample("cpu")

    assert actual.song_indices == expected.song_indices
    assert actual.offsets == expected.offsets
    assert actual.reset_mask.tolist() == expected.reset_mask.tolist()
    torch.testing.assert_close(actual.inputs, expected.inputs)
    torch.testing.assert_close(actual.targets, expected.targets)


def test_stream_state_cpu_payload_roundtrip_preserves_values():
    torch.manual_seed(37)
    model = CompoundHierarchicalGPT(_cfg()).eval()
    records = torch.tensor(_records(9), dtype=torch.long)[None]
    states = initial_batch_stream_states(model, 1)
    with torch.no_grad():
        _, states = encode_tbptt_chunk(model, records, states)
    payload = batch_stream_states_to_cpu(states)
    restored = batch_stream_states_from_cpu(payload, "cpu")

    assert restored[0].steps == states[0].steps
    assert len(restored[0].local_records) == len(states[0].local_records)
    assert len(restored[0].medium_history) == len(states[0].medium_history)
    for left, right in zip(restored[0].medium_history, states[0].medium_history):
        torch.testing.assert_close(left, right.cpu())
    if states[0].memory is not None:
        assert restored[0].memory is not None
        for left, right in zip(restored[0].memory, states[0].memory):
            torch.testing.assert_close(left, right.cpu())


def test_tbptt_trainer_help_imports_without_cuda():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compound_tbptt_train.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "State-carry TBPTT trainer" in result.stdout
    assert "--override-resume-lr" in result.stdout
