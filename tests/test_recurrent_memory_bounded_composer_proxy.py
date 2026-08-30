from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "experiments" / "recurrent_memory_bounded_composer_proxy.py"


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("orbitune_bounded_composer_proxy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_long_memory_has_no_absolute_position_embedding() -> None:
    module = _load()
    memory = module.ConsolidatedMemory()
    assert not any("pos" in name for name, _ in memory.named_parameters())
    local = module.WindowTransformer()
    assert local.pos.num_embeddings == module.LOCAL_WINDOW + module.MEMORY_SLOTS


def test_streaming_state_is_fixed_size_and_matches_parallel_scan() -> None:
    module = _load()
    torch.manual_seed(20260829)
    memory = module.ConsolidatedMemory().eval()
    ids = torch.randint(0, module.VOCAB, (2, 64))
    with torch.no_grad():
        parallel = memory.encode(ids)
        streamed = memory.stream_encode(ids, chunk_size=16)
    assert parallel.shape == streamed.shape == (2, 64, module.D_MODEL)
    assert torch.allclose(parallel, streamed, atol=2e-5, rtol=2e-5)

    embedded = memory.emb(ids[:, :1])
    state = memory.memory.initial_state(2, embedded)
    assert state.state.shape == (2, module.MEMORY_SLOTS, module.D_MODEL)
    assert state.normalizer.shape == (2, module.MEMORY_SLOTS)


def test_memory_slot_composer_only_attends_over_bounded_recent_window() -> None:
    module = _load()
    memory = module.ConsolidatedMemory()
    composer = module.ConditionedComposer(memory, "slots").eval()
    ids, _, positions = module.make_batch(2, torch.device("cpu"), seed=4, distance=32)
    with torch.no_grad():
        hidden = memory.encode(ids)
        logits = composer.predict_at(ids, positions, hidden)
    assert logits.shape == (2, module.VOCAB)
    windows = module.gather_windows(ids, positions)
    assert windows.shape == (2, module.LOCAL_WINDOW)
