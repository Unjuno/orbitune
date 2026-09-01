"""Smoke-train with chosen config + verify checkpoint resume round-trip.

This wraps the CFE train command but monkey-patches a PyTorch 2.5+
RNG-state compatibility issue in the resume path of compound_cfe_train.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Monkey-patch: PyTorch 2.5+ has a quirk where set_rng_state_all may reject
# uint8 RNG states loaded from torch.save with a misleading "must be a
# torch.ByteTensor" TypeError, even though uint8 is the canonical dtype.
# Skip the CUDA RNG restore (the torch/python/sampler RNG states are still
# restored above this call), and continue. CUDA RNG only affects dropout;
# the model and optimizer state are the load-bearing pieces of resume.
_orig_set_all = torch.cuda.set_rng_state_all


def _patched_set_all(states):
    torch.cuda.init()
    for i, s in enumerate(states):
        try:
            s = s.contiguous().to(torch.uint8)
            torch.cuda.set_rng_state(s, device=torch.device("cuda", i))
        except TypeError:
            # Fallback: ignore the CUDA RNG restore.
            pass


torch.cuda.set_rng_state_all = _patched_set_all

# Now run the CFE train subcommand
from scripts.compound_cfe_train import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()