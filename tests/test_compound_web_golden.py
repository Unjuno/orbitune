from __future__ import annotations

import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_web_golden import native_greedy_rollout


def _tiny() -> CompoundHierarchicalGPT:
    torch.manual_seed(41)
    return CompoundHierarchicalGPT(CompoundBaseConfig(
        d_model=32, n_head=4, local_layers=1, medium_layers=1, global_layers=1,
        intra_layers=1, ff_mult=2, dropout=0.0, local_window=4,
        medium_stride=2, medium_window=3, global_stride=2, global_window=3,
    )).eval()


def test_native_greedy_rollout_is_rng_independent_and_deterministic() -> None:
    model = _tiny()
    before = torch.get_rng_state().clone()
    first = native_greedy_rollout(model, max_new_events=6)
    after = torch.get_rng_state().clone()
    second = native_greedy_rollout(model, max_new_events=6)
    assert torch.equal(before, after)
    assert first == second
    assert len(first) == 7
    assert all(len(record) == 12 for record in first)
