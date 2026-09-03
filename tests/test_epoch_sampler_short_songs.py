from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from orbitune.epoch_sampler import EpochAwareNoReplacementSampler


def _song(length: int, *, weight: float = 1.0, tag: str = "song"):
    records = np.arange(length * 12, dtype=np.int64).reshape(length, 12)
    return SimpleNamespace(
        records=records,
        sha256=f"sha-{tag}",
        composition_fingerprint=f"comp-{tag}",
        quality_weight=1.0,
        sampling_weight=weight,
        tracks=1,
        source_id="test",
        license="cc0-1.0",
    )


def test_song_shorter_than_seq_len_is_not_dropped() -> None:
    sampler = EpochAwareNoReplacementSampler(
        [_song(13)], batch_size=1, seq_len=64, epoch_seed=7, weighted=False
    )
    assert sampler.epoch_events_total == 12

    sample = sampler.sample("cpu")
    assert sample.events_counted == 12
    assert sample.batch.song_indices == (0,)
    assert sample.batch.offsets == (0,)
    assert sample.batch.reset_mask.tolist() == [True]
    assert sample.event_weight.shape == (1, 64)
    assert torch.equal(sample.event_weight[0, :12], torch.ones(12))
    assert torch.equal(sample.event_weight[0, 12:], torch.zeros(52))
    assert sampler.is_epoch_complete


def test_epoch_event_total_is_independent_of_seq_len_and_batch_size() -> None:
    songs = [
        _song(2, tag="a"),
        _song(13, tag="b"),
        _song(65, tag="c"),
        _song(130, tag="d"),
    ]
    expected = sum(len(song.records) - 1 for song in songs)

    small = EpochAwareNoReplacementSampler(
        songs, batch_size=1, seq_len=4, epoch_seed=11, weighted=False
    )
    large = EpochAwareNoReplacementSampler(
        songs, batch_size=4, seq_len=64, epoch_seed=11, weighted=False
    )
    assert small.epoch_events_total == expected
    assert large.epoch_events_total == expected


def test_weighted_mode_emits_manifest_weight_but_counts_real_events() -> None:
    sampler = EpochAwareNoReplacementSampler(
        [_song(70, weight=2.5)], batch_size=1, seq_len=64, epoch_seed=3, weighted=True
    )
    assert sampler.epoch_events_total == 69

    first = sampler.sample("cpu")
    assert first.events_counted == 64
    assert torch.all(first.event_weight == 2.5)
    assert first.batch.reset_mask.tolist() == [True]

    tail = sampler.sample("cpu")
    assert tail.events_counted == 5
    assert torch.equal(tail.event_weight[0, :5], torch.full((5,), 2.5))
    assert torch.equal(tail.event_weight[0, 5:], torch.zeros(59))
    assert tail.batch.reset_mask.tolist() == [False]
    assert sampler.epoch_events_seen == 69
    assert sampler.is_epoch_complete


def test_resume_in_later_epoch_keeps_future_shuffle_sequence() -> None:
    songs = [_song(10, tag=str(i)) for i in range(7)]
    live = EpochAwareNoReplacementSampler(
        songs, batch_size=2, seq_len=4, epoch_seed=123, weighted=False
    )

    # Finish epoch 0, enter epoch 1, and save from a non-trivial point there.
    while not live.is_epoch_complete:
        live.sample("cpu")
    live.advance_epoch()
    live.sample("cpu")
    state = live.state_dict()

    resumed = EpochAwareNoReplacementSampler(
        songs, batch_size=2, seq_len=4, epoch_seed=999, weighted=False
    )
    resumed.load_state_dict(state)

    # Finish epoch 1 in lockstep.
    while not live.is_epoch_complete:
        a = live.sample("cpu")
        b = resumed.sample("cpu")
        assert a.batch.song_indices == b.batch.song_indices
        assert a.batch.offsets == b.batch.offsets
        assert torch.equal(a.event_weight, b.event_weight)

    # Advancing after resume must produce the same epoch-2 shuffle even though
    # the constructor used a different seed before state restoration.
    live.advance_epoch()
    resumed.advance_epoch()
    assert live.epoch_index == resumed.epoch_index == 2
    assert live.shuffled_song_order == resumed.shuffled_song_order


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_weighted_mode_rejects_invalid_manifest_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="invalid sampling_weight"):
        EpochAwareNoReplacementSampler(
            [_song(8, weight=weight)],
            batch_size=1,
            seq_len=4,
            epoch_seed=1,
            weighted=True,
        )
