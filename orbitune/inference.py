from __future__ import annotations

from pathlib import Path

import torch

from orbitune.lora import load_adapter
from orbitune.midi import write_midi
from orbitune.model import OrbituneGPT
from orbitune.tokenizer import TheoryRemiTokenizer
from orbitune.tokenizer.vocab import TheoryRemiVocab


def _generation_state(token_ids: list[int], vocab: TheoryRemiVocab) -> tuple[int, int | None]:
    bar_count = 0
    last_position: int | None = None
    for token_id in token_ids:
        token = vocab.tokens[token_id]
        if token == "BAR":
            bar_count += 1
            last_position = None
        elif token.startswith("POSITION_"):
            last_position = int(token.removeprefix("POSITION_"))
    return bar_count, last_position


def allowed_next_tokens(token_ids: list[int], vocab: TheoryRemiVocab, *, requested_bars: int) -> list[str]:
    if requested_bars <= 0:
        raise ValueError("requested_bars must be positive")
    last_token = vocab.tokens[token_ids[-1]] if token_ids else "BOS"
    bar_count, last_position = _generation_state(token_ids, vocab)

    if last_token == "EOS":
        return []
    if last_token == "BOS":
        return ["BAR"]
    if last_token == "BAR":
        return [f"POSITION_{i}" for i in range(vocab.positions_per_bar)]
    if last_token.startswith("POSITION_"):
        return [f"NOTE_PITCH_{p}" for p in range(vocab.min_pitch, vocab.max_pitch + 1)]
    if last_token.startswith("NOTE_PITCH_"):
        return [f"NOTE_DURATION_{d}" for d in range(1, vocab.max_duration + 1)]
    if last_token.startswith("NOTE_DURATION_"):
        return [f"VELOCITY_{v}" for v in range(1, vocab.velocity_bins + 1)]
    if last_token.startswith("VELOCITY_"):
        if last_position is None:
            raise ValueError("velocity token encountered before a position token")
        higher_positions = [
            f"POSITION_{i}"
            for i in range(last_position + 1, vocab.positions_per_bar)
        ]
        # Require every completed bar to reach the final quarter of the bar.
        # This prevents a requested 8-bar output from degenerating into eight
        # one-note bars while still allowing sparse material inside each bar.
        can_close_bar = last_position >= 12
        if bar_count >= requested_bars:
            return [*higher_positions, "EOS"] if can_close_bar else higher_positions
        return [*higher_positions, "BAR"] if can_close_bar else higher_positions
    return ["BAR"]


def _sample(logits: torch.Tensor, allowed_ids: list[int], *, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not allowed_ids:
        raise ValueError("generation grammar has no allowed next token")
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
    bars: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.85,
    top_p: float = 0.92,
    device: str = "cpu",
) -> list[int]:
    vocab = TheoryRemiVocab()
    if model.config.vocab_size != len(vocab):
        raise ValueError("model/tokenizer vocabulary mismatch")
    if bars <= 0:
        raise ValueError("bars must be positive")
    model = model.to(device).eval()
    ids = [vocab.bos_id]
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.config.max_seq_len :]], dtype=torch.long, device=device)
        logits, _ = model(context)
        allowed = allowed_next_tokens(ids, vocab, requested_bars=bars)
        allowed_ids = [vocab.token_to_id[token] for token in allowed]
        next_id = _sample(logits[0, -1], allowed_ids, temperature=temperature, top_p=top_p)
        ids.append(next_id)
        if vocab.tokens[next_id] == "EOS":
            break
    if vocab.tokens[ids[-1]] != "EOS":
        raise RuntimeError("generation hit max_new_tokens before completing the requested bars")
    return ids


def generate_midi(
    base: str | Path,
    out: str | Path,
    *,
    adapter: str | Path | None = None,
    bpm: int = 84,
    bars: int = 8,
    temperature: float = 0.85,
    top_p: float = 0.92,
    max_new_tokens: int = 2048,
    device: str = "cpu",
) -> int:
    vocab = TheoryRemiVocab()
    tokenizer = TheoryRemiTokenizer()
    model = OrbituneGPT.load_checkpoint(base, map_location=device)
    if adapter is not None:
        load_adapter(model, adapter, map_location=device)
    ids = generate_token_ids(
        model,
        bars=bars,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    tokens = vocab.decode(ids, strip_special_tokens=True)
    events = tokenizer.decode_events(tokens)
    write_midi(events, out, bpm=bpm)
    return len(events)
