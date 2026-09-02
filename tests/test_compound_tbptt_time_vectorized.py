from __future__ import annotations

import copy

import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_tbptt import (
    detach_batch_stream_states,
    initial_batch_stream_states,
)
from orbitune.compound_tbptt_optimized import (
    encode_tbptt_chunk as lane_batched_encode,
    tbptt_loss as lane_batched_loss,
)
from orbitune.compound_tbptt_time_vectorized import (
    encode_tbptt_chunk as time_vectorized_encode,
    tbptt_loss as time_vectorized_loss,
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
            torch.testing.assert_close(left_value, right_value, rtol=2e-5, atol=2e-6)
    assert (left.memory is None) == (right.memory is None)
    if left.memory is not None and right.memory is not None:
        for left_value, right_value in zip(left.memory, right.memory):
            torch.testing.assert_close(left_value, right_value, rtol=2e-5, atol=2e-6)


def test_time_vectorized_matches_lane_batched_with_carry_and_selective_reset():
    torch.manual_seed(211)
    model = CompoundHierarchicalGPT(_cfg()).eval()
    first = _records(3, 11)
    second = _records(3, 9)

    lane_states = initial_batch_stream_states(model, 3)
    vector_states = initial_batch_stream_states(model, 3)
    with torch.no_grad():
        lane_first, lane_states = lane_batched_encode(model, first, lane_states)
        vector_first, vector_states = time_vectorized_encode(model, first, vector_states)
        torch.testing.assert_close(vector_first, lane_first, rtol=2e-5, atol=2e-6)
        for left, right in zip(vector_states, lane_states):
            _assert_state_close(left, right)

        reset = torch.tensor([False, True, False])
        lane_second, lane_states = lane_batched_encode(
            model, second, lane_states, reset_mask=reset
        )
        vector_second, vector_states = time_vectorized_encode(
            model, second, vector_states, reset_mask=reset
        )

    torch.testing.assert_close(vector_second, lane_second, rtol=2e-5, atol=2e-6)
    for left, right in zip(vector_states, lane_states):
        _assert_state_close(left, right)


def test_time_vectorized_matches_arbitrary_chunk_partition():
    torch.manual_seed(223)
    model = CompoundHierarchicalGPT(_cfg()).eval()
    records = _records(2, 21)
    lane_states = initial_batch_stream_states(model, 2)
    vector_states = initial_batch_stream_states(model, 2)
    lane_pieces = []
    vector_pieces = []

    with torch.no_grad():
        for start, end in ((0, 1), (1, 6), (6, 13), (13, 15), (15, 21)):
            lane_value, lane_states = lane_batched_encode(
                model, records[:, start:end], lane_states
            )
            vector_value, vector_states = time_vectorized_encode(
                model, records[:, start:end], vector_states
            )
            lane_states = detach_batch_stream_states(lane_states)
            vector_states = detach_batch_stream_states(vector_states)
            lane_pieces.append(lane_value)
            vector_pieces.append(vector_value)

    torch.testing.assert_close(
        torch.cat(vector_pieces, dim=1),
        torch.cat(lane_pieces, dim=1),
        rtol=2e-5,
        atol=2e-6,
    )
    for left, right in zip(vector_states, lane_states):
        _assert_state_close(left, right)


def test_time_vectorized_loss_and_gradients_match_without_dropout():
    torch.manual_seed(227)
    lane_model = CompoundHierarchicalGPT(_cfg()).train()
    vector_model = copy.deepcopy(lane_model).train()
    records = _records(2, 10)
    inputs = records[:, :-1]
    targets = records[:, 1:]

    lane_states = initial_batch_stream_states(lane_model, 2)
    vector_states = initial_batch_stream_states(vector_model, 2)
    lane_value, _, lane_states = lane_batched_loss(
        lane_model, inputs, targets, lane_states
    )
    vector_value, _, vector_states = time_vectorized_loss(
        vector_model, inputs, targets, vector_states
    )

    torch.testing.assert_close(vector_value, lane_value, rtol=2e-5, atol=2e-6)
    lane_value.backward()
    vector_value.backward()

    lane_params = dict(lane_model.named_parameters())
    vector_params = dict(vector_model.named_parameters())
    assert lane_params.keys() == vector_params.keys()
    for name in lane_params:
        left = lane_params[name].grad
        right = vector_params[name].grad
        assert (left is None) == (right is None), name
        if left is not None and right is not None:
            torch.testing.assert_close(right, left, rtol=5e-4, atol=5e-6, msg=name)

    for left, right in zip(vector_states, lane_states):
        _assert_state_close(left, right)


def test_time_vectorized_reduces_transformer_forward_calls_after_warmup():
    torch.manual_seed(229)
    lane_model = CompoundHierarchicalGPT(_cfg()).eval()
    vector_model = copy.deepcopy(lane_model).eval()
    warm = _records(2, 16)
    measured = _records(2, 8)
    lane_states = initial_batch_stream_states(lane_model, 2)
    vector_states = initial_batch_stream_states(vector_model, 2)

    with torch.no_grad():
        _, lane_states = lane_batched_encode(lane_model, warm, lane_states)
        _, vector_states = time_vectorized_encode(vector_model, warm, vector_states)
        lane_states = detach_batch_stream_states(lane_states)
        vector_states = detach_batch_stream_states(vector_states)

    lane_calls = {"local": 0, "medium": 0, "global": 0}
    vector_calls = {"local": 0, "medium": 0, "global": 0}

    lane_hooks = [
        lane_model.local.register_forward_hook(lambda *_: lane_calls.__setitem__("local", lane_calls["local"] + 1)),
        lane_model.medium.register_forward_hook(lambda *_: lane_calls.__setitem__("medium", lane_calls["medium"] + 1)),
        lane_model.global_stack.register_forward_hook(lambda *_: lane_calls.__setitem__("global", lane_calls["global"] + 1)),
    ]
    vector_hooks = [
        vector_model.local.register_forward_hook(lambda *_: vector_calls.__setitem__("local", vector_calls["local"] + 1)),
        vector_model.medium.register_forward_hook(lambda *_: vector_calls.__setitem__("medium", vector_calls["medium"] + 1)),
        vector_model.global_stack.register_forward_hook(lambda *_: vector_calls.__setitem__("global", vector_calls["global"] + 1)),
    ]

    try:
        with torch.no_grad():
            lane_value, _ = lane_batched_encode(lane_model, measured, lane_states)
            vector_value, _ = time_vectorized_encode(vector_model, measured, vector_states)
    finally:
        for hook in lane_hooks + vector_hooks:
            hook.remove()

    torch.testing.assert_close(vector_value, lane_value, rtol=2e-5, atol=2e-6)
    assert vector_calls["local"] < lane_calls["local"]
    assert vector_calls["medium"] < lane_calls["medium"]
    assert vector_calls["global"] <= lane_calls["global"]
