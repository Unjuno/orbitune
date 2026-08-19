from __future__ import annotations

from pathlib import Path

import torch

from orbitune.lora import load_adapter
from orbitune.midi import write_midi
from orbitune.model import OrbituneGPT
from orbitune.tokenizer import TheoryRemiTokenizer
from orbitune.tokenizer.vocab import TheoryRemiVocab


def _allowed_tokens(last_token: str | None, vocab: TheoryRemiVocab) -> list[str]:
    if last_token in {None, "BOS", "VELOCITY_1"} or (last_token and last_token.startswith("VELOCITY_")):
        return ["BAR", "EOS", *[f"POSITION_{i}" for i in range(vocab.positions_per_bar)]]
    if last_token == "BAR":
        return [f"POSITION_{i}" for i in range(vocab.positions_per_bar)]
    if last_token.startswith("POSITION_"):
        return [f"NOTE_PITCH_{p}" for p in range(vocab.min_pitch, vocab.max_pitch + 1)]
    if last_token.startswith("NOTE_PITCH_"):
        return [f"NOTE_DURATION_{d}" for d in range(1, vocab.max_duration + 1)]
    if last_token.startswith("NOTE_DURATION_"):
        return [f"VELOCITY_{v}" for v in range(1, vocab.velocity_bins + 1)]
    return ["BAR", "EOS"]


def _sample(logits: torch.Tensor, allowed_ids: list[int], *, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    masked = torch.full_like(logits, float("-inf"))
    masked[allowed_ids] = logits[allowed_ids]
    probs = torch.softmax(masked / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_token_ids(
    model: OrbituneGPT,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.85,
    top_p: float = 0.92,
    device: str = "cpu",
) -> list[int]:
    vocab = TheoryRemiVocab()
    if model.config.vocab_size != len(vocab):
        raise ValueError("model/tokenizer vocabulary mismatch")
    model = model.to(device).eval()
    ids = [vocab.bos_id]
    last_token = "BOS"
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.config.max_seq_len :]], dtype=torch.long, device=device)
        logits, _ = model(context)
        allowed = _allowed_tokens(last_token, vocab)
        allowed_ids = [vocab.token_to_id[token] for token in allowed]
        next_id = _sample(logits[0, -1], allowed_ids, temperature=temperature, top_p=top_p)
        next_token = vocab.tokens[next_id]
        ids.append(next_id)
        last_token = next_token
        if next_token == "EOS":
            break
    return ids


def generate_midi(
    base: str | Path,
    out: str | Path,
    *,
    adapter: str | Path | None = None,
    bpm: int = 84,
    temperature: float = 0.85,
    top_p: float = 0.92,
    max_new_tokens: int = 256,
    device: str = "cpu",
) -> int:
    vocab = TheoryRemiVocab()
    tokenizer = TheoryRemiTokenizer()
    model = OrbituneGPT.load_checkpoint(base, map_location=device)
    if adapter is not None:
        load_adapter(model, adapter, map_location=device)
    ids = generate_token_ids(
        model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    tokens = vocab.decode(ids, strip_special_tokens=True)
    events = tokenizer.decode_events(tokens)
    write_midi(events, out, bpm=bpm)
    return len(events)
