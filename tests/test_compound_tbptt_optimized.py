from __future__ import annotations

import copy

import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_tbptt import (
    detach_batch_stream_states,
    encode_tbptt_chunk as legacy_encode,
    initial_batch_stream_states,
    tbptt_loss as legacy_loss,
)
from orbitune.compound_tbptt_optimized import (
    encode_tbptt_chunk as optimized_encode,
    tbptt_loss as optimized_loss,
)


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


def _records(batch: int, steps: int) -> torch.Tensor:
    rows = []
    for lane in range(batch):
        lane_rows = []
        for i in range(steps):
            lane_rows.append(
                (
                    0,
                    lane % 4,
                    0,
                    min(15, i % 16),
                    48 + (i + lane * 5) % 24,
                    0,
                    64 + (i + lane) % 32,
                    0,
                    0,
                    1 + i % 15,
                    0,
                    0,
                )
            )
        rows.append(lane_rows)
    return torch.tensor(rows, dtype=torch.long)


def _assert_state_close(left, right) -> None:
    assert left.steps == right.steps
    for name in (
        "local_records",
        "medium_buffer",
        "medium_history",
        "global_buffer",
        "global_history",
    ):
        left_values = getattr(left, name)
        right_values = getattr(right, name)
        assert len(left_values) == len(right_values), name
        for left_value, right_value in zip(left_values, right_values):
            torch.testing.assert_close(left_value, right_value, rtol=1e-5, atol=1e-6)
    assert (left.memory is None) == (right.memory is None)
    if left.memory is not None and right.memory is not None:
        for left_value, right_value in zip(left.memory, right.memory):
            torch.testing.assert_close(left_value, right_value, rtol=1e-5, atol=1e-6)


def test_optimized_multi_lane_matches_legacy_with_carry_and_resets():
    torch.manual_seed(101)
    model = CompoundHierarchicalGPT(_cfg()).eval()
    first = _records(3, 7)
    second = _records(3, 6)

    legacy_states = initial_batch_stream_states(model, 3)
    optimized_states = initial_batch_stream_states(model, 3)
    with torch.no_grad():
        legacy_first, legacy_states = legacy_encode(model, first, legacy_states)
        optimized_first, optimized_states = optimized_encode(model, first, optimized_states)
        torch.testing.assert_close(optimized_first, legacy_first, rtol=1e-5, atol=1e-6)
        for left, right in zip(optimized_states, legacy_states):
            _assert_state_close(left, right)

        reset = torch.tensor([False, True, False])
        legacy_second, legacy_states = legacy_encode(
            model, second, legacy_states, reset_mask=reset
        )
        optimized_second, optimized_states = optimized_encode(
            model, second, optimized_states, reset_mask=reset
        )

    torch.testing.assert_close(optimized_second, legacy_second, rtol=1e-5, atol=1e-6)
    for left, right in zip(optimized_states, legacy_states):
        _assert_state_close(left, right)


def test_optimized_chunk_partition_matches_legacy_partition():
    torch.manual_seed(103)
    model = CompoundHierarchicalGPT(_cfg()).eval()
    records = _records(2, 17)
    legacy_states = initial_batch_stream_states(model, 2)
    optimized_states = initial_batch_stream_states(model, 2)
    legacy_pieces = []
    optimized_pieces = []

    with torch.no_grad():
        for start, end in ((0, 3), (3, 9), (9, 10), (10, 17)):
            legacy_value, legacy_states = legacy_encode(
                model, records[:, start:end], legacy_states
            )
            optimized_value, optimized_states = optimized_encode(
                model, records[:, start:end], optimized_states
            )
            legacy_states = detach_batch_stream_states(legacy_states)
            optimized_states = detach_batch_stream_states(optimized_states)
            legacy_pieces.append(legacy_value)
            optimized_pieces.append(optimized_value)

    torch.testing.assert_close(
        torch.cat(optimized_pieces, dim=1),
        torch.cat(legacy_pieces, dim=1),
        rtol=1e-5,
        atol=1e-6,
    )
    for left, right in zip(optimized_states, legacy_states):
        _assert_state_close(left, right)


def test_optimized_loss_and_gradients_match_legacy_without_dropout():
    torch.manual_seed(107)
    legacy_model = CompoundHierarchicalGPT(_cfg()).train()
    optimized_model = copy.deepcopy(legacy_model).train()
    records = _records(2, 8)
    inputs = records[:, :-1]
    targets = records[:, 1:]

    legacy_states = initial_batch_stream_states(legacy_model, 2)
    optimized_states = initial_batch_stream_states(optimized_model, 2)
    legacy_value, _, legacy_states = legacy_loss(
        legacy_model, inputs, targets, legacy_states
    )
    optimized_value, _, optimized_states = optimized_loss(
        optimized_model, inputs, targets, optimized_states
    )

    torch.testing.assert_close(optimized_value, legacy_value, rtol=1e-5, atol=1e-6)
    legacy_value.backward()
    optimized_value.backward()

    legacy_grads = dict(legacy_model.named_parameters())
    optimized_grads = dict(optimized_model.named_parameters())
    assert legacy_grads.keys() == optimized_grads.keys()
    for name in legacy_grads:
        left = legacy_grads[name].grad
        right = optimized_grads[name].grad
        assert (left is None) == (right is None), name
        if left is not None and right is not None:
            torch.testing.assert_close(right, left, rtol=2e-4, atol=2e-6, msg=name)

    for left, right in zip(optimized_states, legacy_states):
        _assert_state_close(left, right)
