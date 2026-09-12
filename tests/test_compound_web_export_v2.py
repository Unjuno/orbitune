from __future__ import annotations

import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_web_export import (
    CompoundDecoderPrefixV2,
    CompoundStreamAdvanceV2,
    initial_tensor_stream_state,
    synthetic_stream_records,
    verify_native_decoder_parity,
    verify_native_stream_parity,
)


def _tiny() -> CompoundHierarchicalGPT:
    torch.manual_seed(29)
    cfg = CompoundBaseConfig(
        d_model=32,
        n_head=4,
        local_layers=1,
        medium_layers=1,
        global_layers=1,
        intra_layers=1,
        ff_mult=2,
        dropout=0.0,
        local_window=4,
        medium_stride=2,
        medium_window=3,
        global_stride=2,
        global_window=3,
    )
    return CompoundHierarchicalGPT(cfg).eval()


def test_tensor_stream_wrapper_matches_native_across_boundaries() -> None:
    model = _tiny()
    evidence = verify_native_stream_parity(model, steps=18, atol=2e-5, rtol=2e-5)
    assert evidence["steps"] == 18
    assert evidence["max_abs"] < 2e-5


def test_decoder_prefix_wrapper_matches_native_stage_semantics() -> None:
    model = _tiny()
    evidence = verify_native_decoder_parity(model, atol=2e-6, rtol=2e-6)
    assert evidence["max_abs"] < 2e-6


def test_stream_state_shapes_follow_model_configuration() -> None:
    model = _tiny()
    state = initial_tensor_stream_state(model)
    assert state[0].shape == (4, 12)
    assert state[2].shape == (2, 32)
    assert state[4].shape == (3, 32)
    assert state[6].shape == (2, 32)
    assert state[8].shape == (3, 32)
    assert state[10].shape == state[11].shape == state[12].shape == (1, 32)


def test_stream_wrapper_does_not_retain_unbounded_python_history() -> None:
    model = _tiny()
    wrapper = CompoundStreamAdvanceV2(model).eval()
    state = initial_tensor_stream_state(model)
    for record in synthetic_stream_records(40):
        output = wrapper(record, *state)
        state = output[1:]
    assert state[0].shape == (4, 12)
    assert int(state[1]) == 4
    assert state[4].shape == (3, 32)
    assert state[8].shape == (3, 32)


def test_decoder_graph_has_fixed_eight_stage_surface() -> None:
    model = _tiny()
    wrapper = CompoundDecoderPrefixV2(model).eval()
    output = wrapper(
        torch.zeros(1, 32), torch.tensor(0), torch.tensor(0), torch.tensor(0.0),
        torch.tensor(60), torch.tensor(0), torch.tensor(0.5), torch.tensor(0.25),
    )
    assert output[0].shape == (1, 8, 10)
    assert output[1].shape == (1, 8, 16)
    assert output[4].shape == (1, 8, 1024)
    assert output[5].shape == (1, 8, 1024)
