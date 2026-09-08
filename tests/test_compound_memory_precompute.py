from __future__ import annotations

import copy

import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_tbptt import initial_batch_stream_states
from orbitune.compound_tbptt_hybrid import tbptt_loss_hybrid


def _model() -> CompoundHierarchicalGPT:
    return CompoundHierarchicalGPT(CompoundBaseConfig(
        d_model=32, n_head=2, local_layers=1, medium_layers=1,
        global_layers=1, intra_layers=1, ff_mult=2, dropout=0.0,
        local_window=8, medium_stride=2, medium_window=4,
        global_stride=2, global_window=4,
    )).train()


def test_precomputed_memory_matches_loss_states_gradients_and_update():
    torch.manual_seed(431)
    eager = _model()
    candidate = copy.deepcopy(eager)
    records = torch.zeros(3, 8, 12, dtype=torch.long)
    records[..., 1] = torch.arange(3)[:, None]
    records[..., 4] = torch.arange(8)[None, :] + 48
    x, y = records[:, :-1], records[:, 1:]
    eager_states = initial_batch_stream_states(eager, 3)
    candidate_states = initial_batch_stream_states(candidate, 3)
    eager_loss, _, eager_states = tbptt_loss_hybrid(eager, x, y, eager_states)
    candidate_loss, _, candidate_states = tbptt_loss_hybrid(
        candidate, x, y, candidate_states, memory_input_precompute=True)
    torch.testing.assert_close(candidate_loss, eager_loss, rtol=2e-5, atol=2e-6)
    eager_loss.backward(); candidate_loss.backward()
    for (left_name, left), (right_name, right) in zip(
        eager.named_parameters(), candidate.named_parameters()
    ):
        assert left_name == right_name
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=5e-4, atol=5e-6, msg=left_name)
    for candidate_state, eager_state in zip(candidate_states, eager_states):
        assert candidate_state.memory is not None and eager_state.memory is not None
        for candidate_bank, eager_bank in zip(candidate_state.memory, eager_state.memory):
            torch.testing.assert_close(candidate_bank, eager_bank, rtol=2e-5, atol=2e-6)
    eager_opt = torch.optim.SGD(eager.parameters(), lr=1e-3)
    candidate_opt = torch.optim.SGD(candidate.parameters(), lr=1e-3)
    eager_opt.step(); candidate_opt.step()
    for left, right in zip(eager.parameters(), candidate.parameters()):
        torch.testing.assert_close(right, left, rtol=5e-4, atol=5e-6)


def test_precomputed_memory_preserves_selective_reset_isolation():
    torch.manual_seed(433)
    eager = _model().eval()
    candidate = copy.deepcopy(eager)
    records = torch.zeros(3, 5, 12, dtype=torch.long)
    eager_states = initial_batch_stream_states(eager, 3)
    candidate_states = initial_batch_stream_states(candidate, 3)
    with torch.no_grad():
        tbptt_loss_hybrid(eager, records, records, eager_states)
        tbptt_loss_hybrid(candidate, records, records, candidate_states,
                          memory_input_precompute=True)
        reset = torch.tensor([False, True, False])
        left, _, eager_states = tbptt_loss_hybrid(
            eager, records, records, eager_states, reset_mask=reset)
        right, _, candidate_states = tbptt_loss_hybrid(
            candidate, records, records, candidate_states, reset_mask=reset,
            memory_input_precompute=True)
    torch.testing.assert_close(right, left, rtol=2e-5, atol=2e-6)


def test_threebank_matches_each_independent_memory_state_loss_and_gradients():
    torch.manual_seed(439)
    eager = _model()
    candidate = copy.deepcopy(eager)
    records = torch.zeros(3, 8, 12, dtype=torch.long)
    records[..., 1] = torch.arange(3)[:, None]
    records[..., 4] = torch.arange(8)[None, :] + 48
    x, y = records[:, :-1], records[:, 1:]
    eager_states = initial_batch_stream_states(eager, 3)
    candidate_states = initial_batch_stream_states(candidate, 3)
    eager_loss, _, eager_states = tbptt_loss_hybrid(
        eager, x, y, eager_states, memory_input_precompute=True)
    candidate_loss, _, candidate_states = tbptt_loss_hybrid(
        candidate, x, y, candidate_states, memory_threebank=True)
    torch.testing.assert_close(candidate_loss, eager_loss, rtol=2e-5, atol=2e-6)
    for candidate_state, eager_state in zip(candidate_states, eager_states):
        assert candidate_state.memory is not None and eager_state.memory is not None
        for candidate_bank, eager_bank in zip(candidate_state.memory, eager_state.memory):
            torch.testing.assert_close(candidate_bank, eager_bank, rtol=2e-5, atol=2e-6)
    eager_loss.backward(); candidate_loss.backward()
    for (left_name, left), (right_name, right) in zip(
        eager.named_parameters(), candidate.named_parameters()
    ):
        assert left_name == right_name
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=7e-4, atol=7e-6, msg=left_name)
