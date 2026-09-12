from __future__ import annotations

import pytest

from scripts.compound_lora_sweep import _assert_each_pattern_resolved


def test_each_requested_target_pattern_must_resolve() -> None:
    resolved = ["decoder.stack.blocks.0.attn.q_proj"]
    _assert_each_pattern_resolved(
        "candidate",
        ["decoder.stack.blocks.*.attn.q_proj"],
        resolved,
    )
    with pytest.raises(RuntimeError, match="matched no resolved module"):
        _assert_each_pattern_resolved(
            "candidate",
            [
                "decoder.stack.blocks.*.attn.q_proj",
                "decoder.stack.blocks.*.attn.not_a_real_projection",
            ],
            resolved,
        )
