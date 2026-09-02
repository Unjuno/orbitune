"""CUDA acceptance smoke: exact resume + MIDI write/read round-trip."""
from __future__ import annotations

import importlib.util
import random
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitune.compound_base import CompoundHierarchicalGPT, CompoundRecord, write_compound_midi
from orbitune.compound_longrun import build_longrun_checkpoint, restore_longrun_rng, safe_backward_step
from orbitune.compound_midi import read_compound_midi
from orbitune.compound_training import atomic_torch_save, load_compound_jsonl, parse_compound_checkpoint
from orbitune.tokenizer.compound_event import CompoundEventTokenizer


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_script("orbitune_cuda_smoke_base", ROOT / "scripts" / "compound_cuda_train.py")
cfe = _load_script("orbitune_cuda_smoke_cfe", ROOT / "scripts" / "compound_cfe_train.py")


def _one_step(model, optimizer, scaler, x, y, precision):
    optimizer.zero_grad(set_to_none=True)
    with base.autocast_for(precision):
        loss, _ = base.fast_loss(model, x, y)
    result = safe_backward_step(
        loss=loss,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        grad_clip=1.0,
    )
    assert result.stepped, result
    return result.loss_value


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    device = base.require_cuda()
    precision = base.precision_from("auto")
    cfg = base.config_from("configs/compound_hierarchical_9m_nhead7.json")
    cfe.install_causal_fastpath()
    songs = load_compound_jsonl(ROOT / "benchmarks" / "fixtures" / "cfe" / "synthetic_compound.jsonl")
    sampler = base.TensorSampler(songs)

    torch.manual_seed(11)
    random.seed(13)
    sampler_rng = random.Random(17)
    model = CompoundHierarchicalGPT(cfg).to(device)
    optimizer, _ = base.optimizer_for(model, 3e-4, 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    for _ in range(2):
        x, y = sampler.sample(8, 256, sampler_rng, device)
        _one_step(model, optimizer, scaler, x, y, precision)

    runtime = {
        "n_head": 7,
        "seq_len": 256,
        "batch_size": 8,
        "precision": precision,
        "causal_fastpath": True,
    }

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "resume.pt"
        payload = build_longrun_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=2,
            events_seen=2 * 8 * 256,
            runtime=runtime,
            sampler_rng=sampler_rng,
            source_commit="cuda-smoke",
        )
        atomic_torch_save(payload, checkpoint)

        # Continuous reference: capture the exact next sampled window and update.
        x_ref, y_ref = sampler.sample(8, 256, sampler_rng, device)
        x_ref_cpu = x_ref.cpu()
        y_ref_cpu = y_ref.cpu()
        reference_loss = _one_step(model, optimizer, scaler, x_ref, y_ref, precision)
        reference_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        # Fresh process-equivalent objects, then restore every persisted stream.
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        model, raw = CompoundHierarchicalGPT.load_checkpoint(checkpoint, map_location="cpu")
        loaded = parse_compound_checkpoint(raw)
        model.to(device)
        optimizer, _ = base.optimizer_for(model, 3e-4, 0.01)
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        if scaler.is_enabled() and loaded.get("amp_scaler_state_dict"):
            scaler.load_state_dict(loaded["amp_scaler_state_dict"])
        resumed_rng = random.Random(999)
        restore_longrun_rng(loaded, resumed_rng)

        x_resume, y_resume = sampler.sample(8, 256, resumed_rng, device)
        assert torch.equal(x_ref_cpu, x_resume.cpu()), "sampler RNG did not resume exactly"
        assert torch.equal(y_ref_cpu, y_resume.cpu()), "target window did not resume exactly"
        resumed_loss = _one_step(model, optimizer, scaler, x_resume, y_resume, precision)
        assert abs(reference_loss - resumed_loss) < 2e-4, (reference_loss, resumed_loss)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(), reference_state[key], rtol=2e-3, atol=2e-4)

        # Real MIDI round-trip from the resumed checkpoint.
        model.eval()
        primer = [
            CompoundRecord(
                event_type=1,
                channel=0,
                delta_coarse=0,
                delta_residual=0,
                a1=60,
                a2=0,
                a3=100,
                a4=0,
                duration_coarse=2,
                duration_residual=0,
                continuous_coarse=0,
                continuous_residual=0,
            )
        ]
        records = model.generate_records(primer, max_new_events=8, temperature=0.85, top_p=0.9)
        tokenizer = CompoundEventTokenizer()
        midi_path = root / "generated.mid"
        write_compound_midi(midi_path, tokenizer.decode_records(records))
        decoded = read_compound_midi(midi_path)
        assert midi_path.stat().st_size > 0
        assert decoded, "generated MIDI parsed to zero events"

    print("CUDA smoke test PASSED")
    print(f"precision={precision} reference_loss={reference_loss:.6f} resumed_loss={resumed_loss:.6f}")
    print("exact sampler RNG resume: PASS")
    print("optimizer/model continuation: PASS")
    print("MIDI write/read round-trip: PASS")


if __name__ == "__main__":
    main()
