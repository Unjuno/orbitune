from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TheoryRemiVocab:
    """Fixed v0 vocabulary used by orbitune-tiny-v0."""

    positions_per_bar: int = 16
    min_pitch: int = 21
    max_pitch: int = 108
    max_duration: int = 64
    velocity_bins: int = 32
    tokens: list[str] = field(init=False)
    token_to_id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        tokens = ["PAD", "BOS", "EOS", "BAR"]
        tokens += [f"POSITION_{i}" for i in range(self.positions_per_bar)]
        tokens += [f"NOTE_PITCH_{p}" for p in range(self.min_pitch, self.max_pitch + 1)]
        tokens += [f"NOTE_DURATION_{d}" for d in range(1, self.max_duration + 1)]
        tokens += [f"VELOCITY_{v}" for v in range(1, self.velocity_bins + 1)]
        self.tokens = tokens
        self.token_to_id = {token: i for i, token in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: list[str], *, add_special_tokens: bool = True) -> list[int]:
        seq = ["BOS", *tokens, "EOS"] if add_special_tokens else tokens
        try:
            return [self.token_to_id[token] for token in seq]
        except KeyError as exc:
            raise ValueError(f"unknown Theory-REMI token: {exc.args[0]}") from exc

    def decode(self, ids: list[int], *, strip_special_tokens: bool = True) -> list[str]:
        out: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.tokens):
                raise ValueError(f"token id out of range: {idx}")
            token = self.tokens[idx]
            if strip_special_tokens and token in {"PAD", "BOS", "EOS"}:
                continue
            out.append(token)
        return out

    @property
    def pad_id(self) -> int:
        return self.token_to_id["PAD"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["BOS"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["EOS"]
