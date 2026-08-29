from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import real_compound_memory_experiment as base  # noqa: E402


class SharedMatchedExact(base.SharedMatched):
    """Shared-memory control with exactly the routed model's parameter count.

    The original real-data harness was 194 parameters smaller than the routed
    multibank model (157,456 vs 157,650).  This adds a 48->4 bias-free
    calibration probe (192 parameters) plus two learned gates.  The probe is
    applied as per-tier logit temperature modulation, so all 194 parameters are
    on the loss path rather than being inert bookkeeping parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.capacity_probe = nn.Linear(base.D_MODEL, 4, bias=False)
        self.capacity_gates = nn.Parameter(torch.tensor([0.1, 0.1]))

    def forward_chunk(self, records: torch.Tensor, state):  # type: ignore[no-untyped-def]
        fast, medium, slow, event, next_state = super().forward_chunk(records, state)
        hidden = self.embedding(records)
        probe = torch.tanh(self.capacity_probe(hidden))
        gain = self.capacity_gates[0]
        offset = self.capacity_gates[1]

        def scale(logits: torch.Tensor, channel: int) -> torch.Tensor:
            # Multiplicative temperature and class-index-sensitive slope both
            # affect cross-entropy, avoiding an inert same-offset correction.
            temperature = 1.0 + gain * probe[:, :, channel : channel + 1]
            classes = torch.linspace(
                -1.0,
                1.0,
                logits.shape[-1],
                device=logits.device,
                dtype=logits.dtype,
            ).view(1, 1, -1)
            slope = offset * probe[:, :, channel : channel + 1] * classes
            return logits * temperature + slope

        fast = [scale(logits, 0) for logits in fast]
        medium = [scale(logits, 1) for logits in medium]
        slow = [scale(logits, 2) for logits in slow]
        event = scale(event, 3)
        return fast, medium, slow, event, next_state


# Patch the base module's globals because base.run/base.parse_args resolve MODELS
# in their own module namespace.
base.SharedMatched = SharedMatchedExact
base.MODELS = {
    "shared_matched": SharedMatchedExact,
    "multibank_routed": base.RoutedMultiBank,
}

SharedMatched = SharedMatchedExact
RoutedMultiBank = base.RoutedMultiBank
MODELS = base.MODELS

# Re-export the public experiment helpers used by tests and callers.
load_splits = base.load_splits
target_profile = base.target_profile
train_memory_stage = base.train_memory_stage
train_composer_stage = base.train_composer_stage
evaluate = base.evaluate
_configure_composer_optimizer = base._configure_composer_optimizer
run = base.run
D_MODEL = base.D_MODEL


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
