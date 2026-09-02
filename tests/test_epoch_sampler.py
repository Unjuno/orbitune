"""Tests for the epoch-aware, no-replacement TBPTT sampler.

These tests pin the production commercial-base pretrain contract:
(A) no-replacement song-visit count = batch_size per epoch boundary,
(B) deterministic seed => same shuffle, different seed => different,
(C) exact resume of song_indices / offsets / inputs / targets /
    event weights / epoch completion,
(D) partial final chunks: every active target pair appears once with
    weight > 0,
(E) epoch tail with batch_size > remaining_songs: idle lanes have
    event_weight = 0 and there is no next-epoch prefetch,
(F) weighted loss: all-weights = 1 matches time-vectorized loss to fp
    tolerance; padding weight = 0 contributes nothing.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_indexed import IndexedCompoundSong, IndexedRecords
from orbitune.compound_tbptt import initial_batch_stream_states
from orbitune.compound_tbptt_time_vectorized import encode_tbptt_chunk
from orbitune.epoch_sampler import EpochAwareNoReplacementSampler


def _records(count: int, pitch_offset: int = 0) -> np.ndarray:
    matrix = np.zeros((count, 12), dtype="<i4")
    matrix[:, 0] = 0  # NOTE
    matrix[:, 1] = 0  # channel
    matrix[:, 3] = np.arange(count) % 16
    matrix[:, 4] = 48 + (np.arange(count) + pitch_offset) % 24
    matrix[:, 6] = 64 + np.arange(count) % 32
    matrix[:, 9] = 1 + np.arange(count) % 15
    return matrix


def _song(idx: int, count: int) -> IndexedCompoundSong:
    matrix = _records(count, pitch_offset=idx)
    records = IndexedRecords(matrix, offset=0, length=count)
    return IndexedCompoundSong(
        path=f"synthetic://song-{idx}.mid",
        sha256=f"sha-{idx:064d}"[-64:],
        tokenizer_abi="synthetic-1",
        records=records,
        quality_weight=1.0,
        sampling_weight=1.0,
        tracks=1,
        composition_fingerprint=f"comp-{idx}",
        source_id="synthetic",
        license="public-domain",
    )


def _small_songs(n: int, length: int = 40) -> list[IndexedCompoundSong]:
    return [_song(i, length) for i in range(n)]


# --------------------------------------------------------------------- A --


def test_no_replacement_song_visit_count() -> None:
    """Every song is drawn exactly once (no replacement) per epoch boundary."""
    songs = _small_songs(7, length=24)
    sampler = EpochAwareNoReplacementSampler(songs, batch_size=2, seq_len=4, epoch_seed=11)
    song_starts: list[int] = []
    while not sampler.is_epoch_complete:
        sample = sampler.sample("cpu")
        # A "song start" is a chunk with reset_mask=True and a real song index
        for lane, reset in enumerate(sample.batch.reset_mask.tolist()):
            song_index = sample.batch.song_indices[lane]
            if reset and song_index >= 0 and sample.batch.offsets[lane] == 0:
                song_starts.append(song_index)
    # Exactly 7 song starts, one per song, no replacement
    assert sorted(song_starts) == [0, 1, 2, 3, 4, 5, 6]
    assert sampler.epoch_events_seen == sampler.epoch_events_total
    # Second epoch: independent shuffle, no carry-over, all 7 songs again
    sampler.advance_epoch()
    second_starts: list[int] = []
    while not sampler.is_epoch_complete:
        sample = sampler.sample("cpu")
        for lane, reset in enumerate(sample.batch.reset_mask.tolist()):
            song_index = sample.batch.song_indices[lane]
            if reset and song_index >= 0 and sample.batch.offsets[lane] == 0:
                second_starts.append(song_index)
    assert sorted(second_starts) == [0, 1, 2, 3, 4, 5, 6]


# --------------------------------------------------------------------- B --


def test_deterministic_seed_reproducible_shuffle() -> None:
    songs_a = _small_songs(6, length=20)
    songs_b = _small_songs(6, length=20)
    sa = EpochAwareNoReplacementSampler(songs_a, batch_size=2, seq_len=4, epoch_seed=42)
    sb = EpochAwareNoReplacementSampler(songs_b, batch_size=2, seq_len=4, epoch_seed=42)
    assert sa.shuffled_song_order == sb.shuffled_song_order
    sc = EpochAwareNoReplacementSampler(_small_songs(6, length=20), batch_size=2, seq_len=4, epoch_seed=43)
    assert sa.shuffled_song_order != sc.shuffled_song_order


# --------------------------------------------------------------------- C --


def test_exact_resume_round_trip() -> None:
    songs = _small_songs(5, length=24)
    sampler_a = EpochAwareNoReplacementSampler(songs, batch_size=2, seq_len=4, epoch_seed=99)
    # Warm up the lanes so offsets/order_cursor are non-trivial
    for _ in range(3):
        sampler_a.sample("cpu")
    state = sampler_a.state_dict()
    events_before = sampler_a.epoch_events_seen
    cursor_before = sampler_a.order_cursor
    # Capture the next 3 chunks (these are the "resume" data)
    emitted: list[tuple[tuple[int, ...], tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(3):
        sample = sampler_a.sample("cpu")
        emitted.append(
            (
                sample.batch.song_indices,
                sample.batch.offsets,
                sample.batch.inputs.clone(),
                sample.batch.targets.clone(),
                sample.event_weight.clone(),
            )
        )

    sampler_b = EpochAwareNoReplacementSampler(_small_songs(5, length=24), batch_size=2, seq_len=4, epoch_seed=99)
    sampler_b.load_state_dict(state)
    assert sampler_b.epoch_events_seen == events_before
    assert sampler_b.order_cursor == cursor_before
    # The resumed chunks must match exactly
    for expected in emitted:
        resumed = sampler_b.sample("cpu")
        assert resumed.batch.song_indices == expected[0]
        assert resumed.batch.offsets == expected[1]
        assert torch.equal(resumed.batch.inputs, expected[2])
        assert torch.equal(resumed.batch.targets, expected[3])
        assert torch.equal(resumed.event_weight, expected[4])


# --------------------------------------------------------------------- D --


def test_partial_final_chunks_all_targets_covered() -> None:
    """For song length L with seq_len S, all L-1 target pairs are emitted once with weight > 0."""
    length = 13
    seq_len = 5
    songs = [_song(0, length)]
    sampler = EpochAwareNoReplacementSampler(songs, batch_size=1, seq_len=seq_len, epoch_seed=7)
    seen_pairs: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    active_count = 0
    while not sampler.is_epoch_complete:
        sample = sampler.sample("cpu")
        # Inputs/targets are (1, S, 12) record vectors; event_weight is (1, S)
        assert sample.batch.inputs.shape == (1, seq_len, 12)
        assert sample.batch.targets.shape == (1, seq_len, 12)
        assert sample.event_weight.shape == (1, seq_len)
        # Per-position target pair: a pair (in, out) is "active" iff weight>0
        for s in range(seq_len):
            w = float(sample.event_weight[0, s].item())
            if w > 0:
                in_t = tuple(sample.batch.inputs[0, s].tolist())
                out_t = tuple(sample.batch.targets[0, s].tolist())
                seen_pairs.add((sample.batch.song_indices[0], in_t, out_t))
                active_count += 1
    # All L-1=12 target pairs seen, each once
    assert len(seen_pairs) == length - 1
    # The total active count matches the sampler total
    assert active_count == length - 1
    assert sampler.epoch_events_seen == length - 1
    assert sampler.epoch_events_total == length - 1


# --------------------------------------------------------------------- E --


def test_epoch_tail_idle_lanes_no_prefetch() -> None:
    """batch_size > remaining_songs: idle lanes are weight=0 and no next-epoch prefetch."""
    songs = _small_songs(2, length=24)  # only 2 songs
    sampler = EpochAwareNoReplacementSampler(songs, batch_size=4, seq_len=4, epoch_seed=3)
    # Step 1: lanes 0,1 get songs; lanes 2,3 are idle because only 2 songs exist
    sample = sampler.sample("cpu")
    assert sample.batch.song_indices[0] >= 0
    assert sample.batch.song_indices[1] >= 0
    assert sample.batch.song_indices[2] == -1
    assert sample.batch.song_indices[3] == -1
    assert (sample.event_weight[2] == 0).all()
    assert (sample.event_weight[3] == 0).all()
    # No next-epoch prefetch: order_cursor must not exceed the number of songs
    assert sampler.order_cursor == 2
    assert len(sampler.shuffled_song_order) == 2
    # Advance epoch: a fresh shuffle appears, with no carry-over from the previous epoch
    sampler.advance_epoch()
    assert sampler.epoch_events_seen == 0
    assert sampler.order_cursor == 0
    assert set(sampler.shuffled_song_order) == {0, 1}


# --------------------------------------------------------------------- F --


def test_weighted_loss_equivalence_and_padding_isolation() -> None:
    """all-weights=1 matches time-vectorized loss; weight=0 contributes nothing."""
    songs = _small_songs(3, length=24)
    seq_len = 4
    batch_size = 2
    sampler = EpochAwareNoReplacementSampler(songs, batch_size=batch_size, seq_len=seq_len, epoch_seed=5)
    sample = sampler.sample("cpu")
    cfg = CompoundBaseConfig(
        d_model=32, n_head=2, local_layers=1, medium_layers=1, global_layers=1,
        intra_layers=1, ff_mult=2, dropout=0.0, local_window=8,
        medium_stride=2, medium_window=4, global_stride=2, global_window=4,
    )
    cfg.validate()
    model = CompoundHierarchicalGPT(cfg).eval()
    states = initial_batch_stream_states(model, batch_size)
    contexts, _ = encode_tbptt_chunk(model, sample.batch.inputs, states, reset_mask=sample.batch.reset_mask)
    decoder = model.decoder
    # Unweighted (event_weight=None) baseline
    loss_unweighted, _ = decoder.loss(contexts, sample.batch.targets)
    # Weighted with all 1s — must match within fp tolerance
    ones = torch.ones_like(sample.event_weight)
    loss_ones, _ = decoder.loss(contexts, sample.batch.targets, event_weight=ones)
    assert torch.allclose(loss_unweighted, loss_ones, atol=1e-5, rtol=1e-5)
    # Weighted with weight=0 for the entire batch — loss is 0 and no gradient
    zeros = torch.zeros_like(sample.event_weight)
    loss_zero, _ = decoder.loss(contexts, sample.batch.targets, event_weight=zeros)
    assert float(loss_zero.item()) == 0.0
    # Backward with all-zero weight must produce zero gradient on every
    # parameter — this is the production isolation contract for padding /
    # idle lanes.
    decoder.zero_grad(set_to_none=True)
    loss_zero_for_grad = decoder.loss(contexts, sample.batch.targets, event_weight=zeros)[0]
    loss_zero_for_grad.backward()
    for name, p in decoder.named_parameters():
        if p.grad is None:
            continue
        assert float(p.grad.detach().abs().max().item()) == 0.0, f"weight=0 isolation failed for {name}"


# ---------------------------------------------------------------------


def test_state_dict_corpus_identity_fail_closed(tmp_path: Path) -> None:
    songs_a = _small_songs(4, length=16)
    songs_b = _small_songs(4, length=20)  # different length => different corpus identity
    sampler_a = EpochAwareNoReplacementSampler(songs_a, batch_size=2, seq_len=4, epoch_seed=1)
    state = sampler_a.state_dict()
    sampler_b = EpochAwareNoReplacementSampler(songs_b, batch_size=2, seq_len=4, epoch_seed=1)
    with pytest.raises(ValueError, match="corpus identity"):
        sampler_b.load_state_dict(state)


def test_state_dict_seq_len_mismatch_fail_closed() -> None:
    songs = _small_songs(4, length=20)
    sampler_a = EpochAwareNoReplacementSampler(songs, batch_size=2, seq_len=4, epoch_seed=1)
    state = sampler_a.state_dict()
    sampler_b = EpochAwareNoReplacementSampler(_small_songs(4, length=20), batch_size=2, seq_len=8, epoch_seed=1)
    with pytest.raises(ValueError, match="seq_len"):
        sampler_b.load_state_dict(state)


def test_state_dict_batch_size_mismatch_fail_closed() -> None:
    songs = _small_songs(4, length=20)
    sampler_a = EpochAwareNoReplacementSampler(songs, batch_size=2, seq_len=4, epoch_seed=1)
    state = sampler_a.state_dict()
    sampler_b = EpochAwareNoReplacementSampler(_small_songs(4, length=20), batch_size=3, seq_len=4, epoch_seed=1)
    with pytest.raises(ValueError, match="batch_size"):
        sampler_b.load_state_dict(state)
