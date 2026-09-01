"""CUDA smoke test: 8-step train + save + resume + MIDI generation parse."""
from __future__ import annotations

import json
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.compound_cuda_train import (
    TensorSampler,
    config_from,
    precision_from,
    require_cuda,
    optimizer_for,
    train_step,
    autocast_for,
)
from scripts.compound_cfe_train import install_causal_fastpath
from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_training import (
    atomic_torch_save,
    build_compound_checkpoint,
    load_compound_jsonl,
    restore_cuda_rng_state,
    normalize_cuda_rng_state,
)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return
    device = require_cuda()
    cfg = config_from("configs/compound_hierarchical_9m_nhead7.json")
    cfg = replace(cfg, n_head=7)
    cfg.validate()
    install_causal_fastpath()

    torch.manual_seed(0)
    random.seed(0)
    rng = random.Random(7)

    songs_path = ROOT / "benchmarks/fixtures/cfe/synthetic_compound.jsonl"
    songs = load_compound_jsonl(songs_path)
    sampler = TensorSampler(songs)
    model = CompoundHierarchicalGPT(cfg).to(device)
    optimizer, _ = optimizer_for(model, 3e-4, 0.01)
    precision = precision_from("auto")
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    losses = []
    for step in range(8):
        x, y = sampler.sample(16, 256, rng, device)
        loss, _ = train_step(model, optimizer, scaler, x, y, precision, 1.0)
        losses.append(float(loss.detach()))

    # Save checkpoint
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt.pt"
        payload = build_compound_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler, step=8, events_seen=8 * 16,
            runtime={"n_head": 7, "seq_len": 256, "batch_size": 16, "precision": "bf16", "causal_fastpath": True},
            rng=rng,
        )
        atomic_torch_save(payload, ckpt)

        # Resume: build new model, load state, restore RNG.
        del model, optimizer
        torch.cuda.empty_cache()
        model = CompoundHierarchicalGPT(cfg).to(device)
        optimizer, _ = optimizer_for(model, 3e-4, 0.01)
        loaded = torch.load(ckpt, weights_only=False, map_location="cpu")
        model.load_state_dict(loaded["model_state_dict"])
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        if loaded.get("cuda_rng_state_all") is not None:
            restore_cuda_rng_state(loaded["cuda_rng_state_all"])
        # If AMP scaler was enabled, restore it.
        if scaler is not None and loaded.get("amp_scaler_state_dict") is not None:
            scaler.load_state_dict(loaded["amp_scaler_state_dict"])

        # Generate a small MIDI sample to confirm the resumed model produces valid records.
        from orbitune.compound_base import CompoundRecord
        primer = [CompoundRecord(event_type=1, channel=0, delta_coarse=0, delta_residual=0, a1=60, a2=0, a3=100, a4=0,
                                 duration_coarse=2, duration_residual=0, continuous_coarse=0, continuous_residual=0)]
        records = model.generate_records(primer, max_new_events=8, temperature=0.85, top_p=0.9)
        assert len(records) == 9, f"expected 1 primer + 8 new = 9 records, got {len(records)}"
        # All records must have valid tuple shape (12 ints).
        for r in records:
            assert len(r.as_tuple()) == 12, f"bad record: {r.as_tuple()}"

    print("CUDA smoke test PASSED.")
    print(f"  initial 8-step losses: {losses}")


if __name__ == "__main__":
    main()