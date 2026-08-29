from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


COMMITTED_SCRIPT = Path(__file__).parents[1] / "experiments" / "recurrent_memory_committed_buffer.py"
BOUNDED_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_bounded_composer.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _records(cards: tuple[int, ...], *, length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    fields = [
        torch.randint(card, (1, length, 1), generator=generator)
        for card in cards
    ]
    return torch.cat(fields, dim=-1)


def _assert_state_close(left, right) -> None:  # type: ignore[no-untyped-def]
    if left is None or right is None:
        assert left is right
        return
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
        return
    assert isinstance(left, tuple) and isinstance(right, tuple)
    assert len(left) == len(right)
    for lvalue, rvalue in zip(left, right):
        _assert_state_close(lvalue, rvalue)


def test_committed_memory_is_exact_nonoverlapping_stream_partition() -> None:
    committed = _load(COMMITTED_SCRIPT, "orbitune_committed_stream")
    bounded = _load(BOUNDED_SCRIPT, "orbitune_committed_bounded")
    torch.manual_seed(91)
    model = committed.memory.RoutedMultiBank().eval()
    records = _records(
        tuple(committed.memory.base.FIELD_CARDINALITIES), length=48, seed=92
    )
    hot_window = 7
    state = committed.empty_state()

    with torch.no_grad():
        for index in range(records.shape[1]):
            state = committed.push_event(
                model,
                records[:, index : index + 1],
                state,
                hot_window=hot_window,
            )
            expected_committed = max(0, index + 1 - hot_window)
            expected_hot = records[:, expected_committed : index + 1]
            assert state.committed_events == expected_committed
            assert state.hot_records is not None
            assert torch.equal(state.hot_records, expected_hot)
            assert state.hot_records.shape[1] <= hot_window
            assert state.committed_events + state.hot_records.shape[1] == index + 1

            if expected_committed == 0:
                assert state.cold_tokens is None
                assert state.bank_state is None
            else:
                prefix = records[:, :expected_committed]
                expected_tokens, expected_bank_state = bounded.routed_memory_reads(
                    model, prefix, None
                )
                assert state.cold_tokens is not None
                torch.testing.assert_close(
                    state.cold_tokens,
                    expected_tokens[:, -1],
                    atol=2e-5,
                    rtol=2e-5,
                )
                _assert_state_close(state.bank_state, expected_bank_state)


def test_committed_memory_chunking_matches_eventwise_stream() -> None:
    committed = _load(COMMITTED_SCRIPT, "orbitune_committed_chunk")
    torch.manual_seed(101)
    model = committed.memory.RoutedMultiBank().eval()
    records = _records(
        tuple(committed.memory.base.FIELD_CARDINALITIES), length=53, seed=102
    )
    hot_window = 9

    eventwise = committed.empty_state()
    with torch.no_grad():
        for index in range(records.shape[1]):
            eventwise = committed.push_event(
                model,
                records[:, index : index + 1],
                eventwise,
                hot_window=hot_window,
            )

        chunked = committed.empty_state()
        start = 0
        for width in (3, 11, 1, 17, 5, 16):
            stop = min(records.shape[1], start + width)
            if stop <= start:
                break
            chunked = committed.push_chunk(
                model,
                records[:, start:stop],
                chunked,
                hot_window=hot_window,
            )
            start = stop
        if start < records.shape[1]:
            chunked = committed.push_chunk(
                model,
                records[:, start:],
                chunked,
                hot_window=hot_window,
            )

    assert eventwise.committed_events == chunked.committed_events
    assert eventwise.hot_records is not None and chunked.hot_records is not None
    assert torch.equal(eventwise.hot_records, chunked.hot_records)
    assert eventwise.cold_tokens is not None and chunked.cold_tokens is not None
    torch.testing.assert_close(
        eventwise.cold_tokens, chunked.cold_tokens, atol=2e-5, rtol=2e-5
    )
    _assert_state_close(eventwise.bank_state, chunked.bank_state)
    assert chunked.hot_records.shape[1] == hot_window
    assert chunked.committed_events == records.shape[1] - hot_window
