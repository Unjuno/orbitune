from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from orbitune.compound_base import CompoundHierarchicalGPT, StreamState, _causal_bias


STREAM_INPUT_NAMES = (
    "rec", "loc", "lloc", "mbuf", "mblen", "mhist", "mhlen", "gbuf", "gblen",
    "ghist", "ghlen", "memf", "memm", "mems", "steps",
)
STREAM_OUTPUT_NAMES = (
    "ctx", "loc_o", "lloc_o", "mbuf_o", "mblen_o", "mhist_o", "mhlen_o",
    "gbuf_o", "gblen_o", "ghist_o", "ghlen_o", "memf_o", "memm_o", "mems_o", "steps_o",
)
DECODER_INPUT_NAMES = ("ctx", "et", "ch", "dn", "a1", "a2", "vn", "dur")
DECODER_OUTPUT_NAMES = (
    "event_type_logits", "channel_logits", "delta_mean", "delta_log_scale",
    "a1_logits", "a2_logits", "velocity_mean", "velocity_log_scale",
    "duration_mean", "duration_log_scale", "control_mean", "control_log_scale",
)


class CompoundStreamAdvanceV2(nn.Module):
    """Tensor-only equivalent of ``CompoundHierarchicalGPT.advance_stream``.

    State is represented by fixed-capacity tensors plus scalar lengths so the
    exact native stream semantics can cross the ONNX/Web boundary without an
    ever-growing KV cache or record history.
    """

    def __init__(self, model: CompoundHierarchicalGPT) -> None:
        super().__init__()
        cfg = model.config
        self.embedding = model.embedding
        self.local = model.local
        self.medium = model.medium
        self.global_stack = model.global_stack
        self.memory = model.memory
        self.fusion = model.fusion
        self.d_model = cfg.d_model
        self.local_window = cfg.local_window
        self.medium_stride = cfg.medium_stride
        self.medium_window = cfg.medium_window
        self.global_stride = cfg.global_stride
        self.global_window = cfg.global_window
        self.register_buffer("local_bias", _causal_bias(cfg.local_window, torch.device("cpu"), window=cfg.local_window), persistent=False)
        self.register_buffer("medium_bias", _causal_bias(cfg.medium_window, torch.device("cpu")), persistent=False)
        self.register_buffer("global_bias", _causal_bias(cfg.global_window, torch.device("cpu")), persistent=False)

    @staticmethod
    def _append_fixed(buffer: torch.Tensor, length: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        capacity = buffer.shape[0]
        index = length.clamp(0, capacity - 1).reshape(1, 1).expand(1, buffer.shape[1])
        inserted = buffer.scatter(0, index, value.reshape(1, -1))
        shifted = torch.cat((buffer[1:], value.reshape(1, -1)), dim=0)
        full = length >= capacity
        output = torch.where(full, shifted, inserted)
        next_length = torch.minimum(length + 1, length.new_tensor(capacity))
        return output, next_length

    @staticmethod
    def _last_hidden(stack: nn.Module, values: torch.Tensor, length: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        hidden = stack(values[None], bias)[0]
        width = hidden.shape[-1]
        index = (length - 1).clamp(0, hidden.shape[0] - 1).reshape(1, 1).expand(1, width)
        selected = hidden.gather(0, index)[0]
        return torch.where(length > 0, selected, torch.zeros_like(selected))

    def forward(
        self,
        rec: torch.Tensor,
        loc: torch.Tensor,
        lloc: torch.Tensor,
        mbuf: torch.Tensor,
        mblen: torch.Tensor,
        mhist: torch.Tensor,
        mhlen: torch.Tensor,
        gbuf: torch.Tensor,
        gblen: torch.Tensor,
        ghist: torch.Tensor,
        ghlen: torch.Tensor,
        memf: torch.Tensor,
        memm: torch.Tensor,
        mems: torch.Tensor,
        steps: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        rec = rec.reshape(12).to(dtype=torch.long)
        loc_o, lloc_o = self._append_fixed(loc, lloc, rec)
        local_event = self.embedding(loc_o[None])
        local_all = self.local(local_event, self.local_bias)[0]
        local_index = (lloc_o - 1).clamp(0, self.local_window - 1).reshape(1, 1).expand(1, self.d_model)
        local_hidden = local_all.gather(0, local_index)[0]

        event_emb = self.embedding(rec[None, None])[0, 0]
        memory_read, memory_o = self.memory.step(event_emb[None], (memf, memm, mems))
        memf_o, memm_o, mems_o = memory_o

        mbuf_inserted, mblen_inserted = self._append_fixed(mbuf, mblen, local_hidden)
        medium_complete = (mblen + 1) >= self.medium_stride
        medium_summary = mbuf_inserted.mean(dim=0)
        mbuf_o = torch.where(medium_complete, torch.zeros_like(mbuf_inserted), mbuf_inserted)
        mblen_o = torch.where(medium_complete, torch.zeros_like(mblen_inserted), mblen_inserted)

        mhist_candidate, mhlen_candidate = self._append_fixed(mhist, mhlen, medium_summary)
        mhist_o = torch.where(medium_complete, mhist_candidate, mhist)
        mhlen_o = torch.where(medium_complete, mhlen_candidate, mhlen)
        medium_context = self._last_hidden(self.medium, mhist_o, mhlen_o, self.medium_bias)

        gbuf_candidate, gblen_candidate = self._append_fixed(gbuf, gblen, medium_context)
        gbuf_after_medium = torch.where(medium_complete, gbuf_candidate, gbuf)
        gblen_after_medium = torch.where(medium_complete, gblen_candidate, gblen)
        global_complete = medium_complete & ((gblen + 1) >= self.global_stride)
        global_summary = gbuf_after_medium.mean(dim=0)
        gbuf_o = torch.where(global_complete, torch.zeros_like(gbuf_after_medium), gbuf_after_medium)
        gblen_o = torch.where(global_complete, torch.zeros_like(gblen_after_medium), gblen_after_medium)

        ghist_candidate, ghlen_candidate = self._append_fixed(ghist, ghlen, global_summary)
        ghist_o = torch.where(global_complete, ghist_candidate, ghist)
        ghlen_o = torch.where(global_complete, ghlen_candidate, ghlen)
        global_context = self._last_hidden(self.global_stack, ghist_o, ghlen_o, self.global_bias)

        ctx = self.fusion(torch.cat((local_hidden, medium_context, global_context, memory_read[0]), dim=-1))[None]
        steps_o = steps + 1
        return (
            ctx, loc_o, lloc_o, mbuf_o, mblen_o, mhist_o, mhlen_o, gbuf_o, gblen_o,
            ghist_o, ghlen_o, memf_o, memm_o, mems_o, steps_o,
        )


class CompoundDecoderPrefixV2(nn.Module):
    """Export all eight causal decoder stages from one prefix graph call."""

    def __init__(self, model: CompoundHierarchicalGPT) -> None:
        super().__init__()
        self.decoder = model.decoder
        self.d_model = model.config.d_model
        self.register_buffer("bias", _causal_bias(8, torch.device("cpu")), persistent=False)

    def forward(
        self,
        ctx: torch.Tensor,
        et: torch.Tensor,
        ch: torch.Tensor,
        dn: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        vn: torch.Tensor,
        dur: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        decoder = self.decoder
        ctx = ctx.reshape(1, self.d_model)
        pieces = (
            decoder.bos,
            decoder.event_type_emb(et.to(torch.long).clamp(0, 9)),
            decoder.channel_emb(ch.to(torch.long).clamp(0, 15)),
            decoder.scalar_emb(dn.to(torch.float32).reshape(1, 1))[0],
            decoder.a1_emb(a1.to(torch.long).clamp(0, 1023)),
            decoder.a2_emb(a2.to(torch.long).clamp(0, 1023)),
            decoder.scalar_emb(vn.to(torch.float32).reshape(1, 1))[0],
            decoder.scalar_emb(dur.to(torch.float32).reshape(1, 1))[0],
        )
        x = torch.stack(pieces, dim=0)[None] + ctx[:, None, :] + decoder.slot_pos.weight[None]
        hidden = decoder.stack(x, self.bias)
        delta_mean, delta_log_scale = decoder.delta_head(hidden)
        velocity_mean, velocity_log_scale = decoder.velocity_head(hidden)
        duration_mean, duration_log_scale = decoder.duration_head(hidden)
        control_mean, control_log_scale = decoder.control_head(hidden)
        return (
            decoder.event_type_head(hidden),
            decoder.channel_head(hidden),
            delta_mean,
            delta_log_scale,
            decoder.a1_head(hidden),
            decoder.a2_head(hidden),
            velocity_mean,
            velocity_log_scale,
            duration_mean,
            duration_log_scale,
            control_mean,
            control_log_scale,
        )


def initial_tensor_stream_state(model: CompoundHierarchicalGPT, *, device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    cfg = model.config
    device = torch.device(device)
    dtype = next(model.parameters()).dtype
    i64 = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=device)
    f32 = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)
    return (
        i64(cfg.local_window, 12), i64(),
        f32(cfg.medium_stride, cfg.d_model), i64(),
        f32(cfg.medium_window, cfg.d_model), i64(),
        f32(cfg.global_stride, cfg.d_model), i64(),
        f32(cfg.global_window, cfg.d_model), i64(),
        f32(1, cfg.d_model), f32(1, cfg.d_model), f32(1, cfg.d_model), i64(),
    )


def synthetic_stream_records(count: int, *, device: torch.device | str = "cpu") -> list[torch.Tensor]:
    records: list[torch.Tensor] = []
    device = torch.device(device)
    for index in range(count):
        if index % 17 == 0:
            raw = [4, 0, 0, 0, 80 + index % 80, 0, 0, 0, 0, 0, 0, 0]
        else:
            raw = [0, index % 4, index % 7, index % 16, 48 + index % 36, 0, 50 + index % 70, 0, (index // 2) % 7, index % 16, 0, 0]
        records.append(torch.tensor(raw, dtype=torch.long, device=device))
    return records


def verify_native_stream_parity(model: CompoundHierarchicalGPT, *, steps: int = 96, atol: float = 3e-5, rtol: float = 3e-5) -> dict[str, float | int]:
    model.eval()
    wrapper = CompoundStreamAdvanceV2(model).eval()
    device = next(model.parameters()).device
    native_state = model.initial_stream_state()
    tensor_state = initial_tensor_stream_state(model, device=device)
    max_abs = 0.0
    for record in synthetic_stream_records(steps, device=device):
        with torch.no_grad():
            native_ctx = model.advance_stream(record, native_state)
            outputs = wrapper(record, *tensor_state)
        exported_ctx, tensor_state = outputs[0], outputs[1:]
        delta = float((native_ctx - exported_ctx).abs().max().item())
        max_abs = max(max_abs, delta)
        if not torch.allclose(native_ctx, exported_ctx, atol=atol, rtol=rtol):
            raise RuntimeError(f"stream tensor wrapper diverged from native advance_stream at step {native_state.steps}: max_abs={delta}")
    return {"steps": steps, "max_abs": max_abs}


def verify_native_decoder_parity(model: CompoundHierarchicalGPT, *, atol: float = 2e-6, rtol: float = 2e-6) -> dict[str, float]:
    model.eval()
    decoder = model.decoder
    wrapper = CompoundDecoderPrefixV2(model).eval()
    device = next(model.parameters()).device
    state = model.initial_stream_state()
    ctx = None
    for record in synthetic_stream_records(19, device=device):
        ctx = model.advance_stream(record, state)
    assert ctx is not None
    et = torch.tensor(0, dtype=torch.long, device=device)
    ch = torch.tensor(2, dtype=torch.long, device=device)
    dn = torch.tensor(0.37, dtype=torch.float32, device=device)
    a1 = torch.tensor(64, dtype=torch.long, device=device)
    a2 = torch.tensor(0, dtype=torch.long, device=device)
    vn = torch.tensor(0.71, dtype=torch.float32, device=device)
    dur = torch.tensor(0.22, dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = wrapper(ctx, et, ch, dn, a1, a2, vn, dur)
        previous: list[torch.Tensor] = []
        reference: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        hidden = decoder._decode_hidden(ctx, previous); reference.append(decoder.event_type_head(hidden))
        previous.append(decoder.event_type_emb(et)); hidden = decoder._decode_hidden(ctx, previous); reference.append(decoder.channel_head(hidden))
        previous.append(decoder.channel_emb(ch)); hidden = decoder._decode_hidden(ctx, previous); reference.append(decoder.delta_head(hidden))
        previous.append(decoder.scalar_emb(dn.reshape(1, 1))[0]); hidden = decoder._decode_hidden(ctx, previous); reference.append(decoder.a1_head(hidden))
        previous.append(decoder.a1_emb(a1)); hidden = decoder._decode_hidden(ctx, previous); reference.append(decoder.a2_head(hidden))
        previous.append(decoder.a2_emb(a2)); hidden = decoder._decode_hidden(ctx, previous); reference.append(decoder.velocity_head(hidden))
        previous.append(decoder.scalar_emb(vn.reshape(1, 1))[0]); hidden = decoder._decode_hidden(ctx, previous); reference.append(decoder.duration_head(hidden))
        previous.append(decoder.scalar_emb(dur.reshape(1, 1))[0]); hidden = decoder._decode_hidden(ctx, previous); reference.append(decoder.control_head(hidden))
    checks = (
        (outputs[0][0, 0], reference[0]),
        (outputs[1][0, 1], reference[1]),
        (outputs[2][0, 2], reference[2][0]), (outputs[3][0, 2], reference[2][1]),
        (outputs[4][0, 3], reference[3]), (outputs[5][0, 4], reference[4]),
        (outputs[6][0, 5], reference[5][0]), (outputs[7][0, 5], reference[5][1]),
        (outputs[8][0, 6], reference[6][0]), (outputs[9][0, 6], reference[6][1]),
        (outputs[10][0, 7], reference[7][0]), (outputs[11][0, 7], reference[7][1]),
    )
    max_abs = 0.0
    for actual, expected in checks:
        delta = float((actual - expected).abs().max().item())
        max_abs = max(max_abs, delta)
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            raise RuntimeError(f"decoder prefix wrapper diverged from native decoder: max_abs={delta}")
    return {"max_abs": max_abs}


def production_contract(model: CompoundHierarchicalGPT) -> None:
    cfg = model.config
    expected = {
        "d_model": 224, "local_window": 64, "medium_stride": 8, "medium_window": 64,
        "global_stride": 4, "global_window": 64,
    }
    actual = {name: getattr(cfg, name) for name in expected}
    if actual != expected:
        raise ValueError(f"checkpoint does not match native-stream V2 Web contract: {actual} != {expected}")


def export_compound_web_v2(model: CompoundHierarchicalGPT, out_dir: str | Path, *, opset: int = 18) -> tuple[Path, Path]:
    production_contract(model)
    model.eval().cpu()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stream_path = out / "stream.onnx"
    decoder_path = out / "decoder_prefix.onnx"
    stream = CompoundStreamAdvanceV2(model).eval()
    decoder = CompoundDecoderPrefixV2(model).eval()
    seed = torch.tensor([4, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    state = initial_tensor_stream_state(model)
    torch.onnx.export(
        stream, (seed, *state), stream_path,
        input_names=list(STREAM_INPUT_NAMES), output_names=list(STREAM_OUTPUT_NAMES),
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    decoder_inputs = (
        torch.zeros(1, 224, dtype=torch.float32), torch.tensor(0, dtype=torch.long),
        torch.tensor(0, dtype=torch.long), torch.tensor(0.0), torch.tensor(0, dtype=torch.long),
        torch.tensor(0, dtype=torch.long), torch.tensor(0.0), torch.tensor(0.0),
    )
    torch.onnx.export(
        decoder, decoder_inputs, decoder_path,
        input_names=list(DECODER_INPUT_NAMES), output_names=list(DECODER_OUTPUT_NAMES),
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    return stream_path, decoder_path


def _numpy_input(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def verify_onnxruntime_parity(model: CompoundHierarchicalGPT, stream_path: str | Path, decoder_path: str | Path, *, steps: int = 24, atol: float = 1e-4, rtol: float = 1e-4) -> dict[str, float | int]:
    import numpy as np
    import onnxruntime as ort

    model.eval().cpu()
    stream_wrapper = CompoundStreamAdvanceV2(model).eval()
    stream_session = ort.InferenceSession(str(stream_path), providers=["CPUExecutionProvider"])
    pt_state = initial_tensor_stream_state(model)
    ort_state = tuple(t.clone() for t in pt_state)
    max_stream_abs = 0.0
    for record in synthetic_stream_records(steps):
        with torch.no_grad(): pt_outputs = stream_wrapper(record, *pt_state)
        ort_inputs = {name: _numpy_input(value) for name, value in zip(STREAM_INPUT_NAMES, (record, *ort_state))}
        ort_raw = stream_session.run(list(STREAM_OUTPUT_NAMES), ort_inputs)
        ort_outputs = tuple(torch.from_numpy(value) for value in ort_raw)
        for index, (pt_value, ort_value) in enumerate(zip(pt_outputs, ort_outputs)):
            if pt_value.is_floating_point():
                delta = float((pt_value.cpu() - ort_value).abs().max().item())
                max_stream_abs = max(max_stream_abs, delta)
                if not torch.allclose(pt_value.cpu(), ort_value, atol=atol, rtol=rtol):
                    raise RuntimeError(f"ONNX stream mismatch output={STREAM_OUTPUT_NAMES[index]} max_abs={delta}")
            elif not torch.equal(pt_value.cpu(), ort_value):
                raise RuntimeError(f"ONNX stream integer mismatch output={STREAM_OUTPUT_NAMES[index]}")
        pt_state = pt_outputs[1:]
        ort_state = ort_outputs[1:]

    decoder = CompoundDecoderPrefixV2(model).eval()
    decoder_session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    decoder_inputs = (
        torch.linspace(-0.3, 0.3, 224).reshape(1, 224), torch.tensor(0, dtype=torch.long),
        torch.tensor(2, dtype=torch.long), torch.tensor(0.37), torch.tensor(64, dtype=torch.long),
        torch.tensor(0, dtype=torch.long), torch.tensor(0.71), torch.tensor(0.22),
    )
    with torch.no_grad(): pt_decoder = decoder(*decoder_inputs)
    ort_decoder_raw = decoder_session.run(list(DECODER_OUTPUT_NAMES), {name: _numpy_input(value) for name, value in zip(DECODER_INPUT_NAMES, decoder_inputs)})
    max_decoder_abs = 0.0
    for name, pt_value, raw in zip(DECODER_OUTPUT_NAMES, pt_decoder, ort_decoder_raw):
        ort_value = torch.from_numpy(np.asarray(raw))
        delta = float((pt_value.cpu() - ort_value).abs().max().item())
        max_decoder_abs = max(max_decoder_abs, delta)
        if not torch.allclose(pt_value.cpu(), ort_value, atol=atol, rtol=rtol):
            raise RuntimeError(f"ONNX decoder mismatch output={name} max_abs={delta}")
    return {"steps": steps, "stream_max_abs": max_stream_abs, "decoder_max_abs": max_decoder_abs}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def export_report(model: CompoundHierarchicalGPT, stream_path: str | Path, decoder_path: str | Path, *, checkpoint_sha256: str, checkpoint_source_commit: str | None, native_stream: dict, native_decoder: dict, ort_parity: dict | None) -> dict:
    stream = Path(stream_path); decoder = Path(decoder_path)
    return {
        "schema": "orbitune-compound-web-v2-export-report-v1",
        "architecture": model.architecture,
        "tokenizer": model.tokenizer,
        "runtime_abi": "native-stream-state+decoder-prefix-v2",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_source_commit": checkpoint_source_commit,
        "config": asdict(model.config),
        "stream": {"filename": stream.name, "sha256": sha256_file(stream), "bytes": stream.stat().st_size},
        "decoder": {"filename": decoder.name, "sha256": sha256_file(decoder), "bytes": decoder.stat().st_size},
        "parity": {"native_stream": native_stream, "native_decoder": native_decoder, "onnxruntime": ort_parity},
    }


def write_report(path: str | Path, report: dict) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
