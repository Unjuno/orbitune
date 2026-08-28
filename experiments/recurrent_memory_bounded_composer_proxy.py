from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from recurrent_memory_chunkwise_scan import chunkwise_discounted_scan  # noqa: E402

N_STATE = 8
N_NOTE = 24
NOTE_BASE = N_STATE
QUERY = NOTE_BASE + N_NOTE
ANSWER_BASE = QUERY + 1
VOCAB = ANSWER_BASE + N_STATE
DISTANCES = (32, 64, 128, 256, 512, 1024)
LOCAL_WINDOW = 16
D_MODEL = 16
MEMORY_SLOTS = 4
SCAN_CHUNK = 128


def make_batch(batch: int, device: torch.device, *, seed: int, distance: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    states = torch.randint(0, N_STATE, (batch,), generator=generator)
    motifs = torch.randint(0, 8, (batch,), generator=generator)
    distances = torch.full((batch,), distance, dtype=torch.long)
    sequence = torch.empty((batch, distance + 2), dtype=torch.long)
    sequence[:, 0] = states
    for row in range(batch):
        query = int(distances[row])
        for step in range(1, sequence.shape[1]):
            if step == query:
                sequence[row, step] = QUERY
            elif step == query + 1:
                sequence[row, step] = ANSWER_BASE + states[row]
            else:
                sequence[row, step] = NOTE_BASE + (
                    (motifs[row] + step + 3 * ((step // 4) % 4)) % N_NOTE
                )
    return (
        sequence[:, :-1].to(device),
        sequence[:, 1:].to(device),
        distances.to(device),
    )


def gather_windows(ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch = ids.shape[0]
    offsets = torch.arange(LOCAL_WINDOW, device=ids.device) - LOCAL_WINDOW + 1
    indices = positions[:, None] + offsets[None, :]
    if bool((indices < 0).any()):
        raise ValueError("positions must be at least LOCAL_WINDOW - 1")
    rows = torch.arange(batch, device=ids.device)[:, None]
    return ids[rows, indices]


def local_positions(distances: torch.Tensor, *, seed: int, maximum: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = torch.randint(
        LOCAL_WINDOW, maximum, (distances.shape[0],), generator=generator
    ).to(distances.device)
    invalid = (positions == distances) | (positions == distances + 1)
    return torch.where(invalid, positions + 2, positions).clamp_max(maximum)


@dataclass(frozen=True, slots=True)
class RecurrentState:
    state: torch.Tensor
    normalizer: torch.Tensor


class LinearMemoryLayer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, key_dim: int = MEMORY_SLOTS) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.q = nn.Linear(d_model, key_dim, bias=False)
        self.k = nn.Linear(d_model, key_dim, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.write = nn.Linear(d_model, 1)
        self.mix = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.logit_decay = nn.Parameter(torch.tensor(5.3))
        nn.init.constant_(self.write.bias, -1.5)

    def initial_state(self, batch: int, template: torch.Tensor) -> RecurrentState:
        return RecurrentState(
            template.new_zeros((batch, self.key_dim, template.shape[-1])),
            template.new_zeros((batch, self.key_dim)),
        )

    def _features(self, h: torch.Tensor):
        x = self.norm(h)
        return (
            F.elu(self.q(x)) + 1,
            F.elu(self.k(x)) + 1,
            self.v(x),
            torch.sigmoid(self.write(x)),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q, k, v, write = self._features(h)
        decay = torch.sigmoid(self.logit_decay).clamp(0.9, 0.9999)
        contributions = write.unsqueeze(-1) * torch.einsum("btk,btd->btkd", k, v)
        normalizers = write * k
        scan = chunkwise_discounted_scan(
            contributions,
            normalizers,
            decay,
            chunk_size=SCAN_CHUNK,
        )
        read = torch.einsum("btk,btkd->btd", q, scan.states) / (
            torch.einsum("btk,btk->bt", q, scan.normalizers).unsqueeze(-1) + 1e-5
        )
        return self.mix(torch.cat([h, read], dim=-1))

    def step(self, h: torch.Tensor, recurrent: RecurrentState) -> tuple[torch.Tensor, RecurrentState]:
        if h.ndim != 2:
            raise ValueError("streaming h must have shape [batch, d_model]")
        q, k, v, write = self._features(h)
        decay = torch.sigmoid(self.logit_decay).clamp(0.9, 0.9999)
        state = decay * recurrent.state + write.unsqueeze(-1) * torch.einsum(
            "bk,bd->bkd", k, v
        )
        normalizer = decay * recurrent.normalizer + write * k
        read = torch.einsum("bk,bkd->bd", q, state) / (
            torch.einsum("bk,bk->b", q, normalizer).unsqueeze(-1) + 1e-5
        )
        return self.mix(torch.cat([h, read], dim=-1)), RecurrentState(state, normalizer)


class ConsolidatedMemory(nn.Module):
    """Long memory intentionally has no absolute sequence-position embedding."""

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        self.memory = LinearMemoryLayer(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.state_head = nn.Linear(d_model, N_STATE)
        self.token_head = nn.Linear(d_model, VOCAB, bias=False)
        self.token_head.weight = self.emb.weight

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(ids)
        return self.norm(h + self.memory(h))

    def stream_encode(self, ids: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        embeddings = self.emb(ids)
        recurrent = self.memory.initial_state(ids.shape[0], embeddings)
        outputs: list[torch.Tensor] = []
        for start in range(0, ids.shape[1], chunk_size):
            stop = min(ids.shape[1], start + chunk_size)
            for index in range(start, stop):
                raw = embeddings[:, index]
                memory_out, recurrent = self.memory.step(raw, recurrent)
                outputs.append(self.norm(raw + memory_out))
        return torch.stack(outputs, dim=1)


class WindowTransformer(nn.Module):
    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(LOCAL_WINDOW + MEMORY_SLOTS, d_model)
        block = nn.TransformerEncoderLayer(
            d_model,
            2,
            4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(block, 1)
        self.norm = nn.LayerNorm(d_model)

    def encode(self, windows: torch.Tensor, prefix: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.emb(windows)
        if prefix is not None:
            hidden = torch.cat([prefix, hidden], dim=1)
        length = hidden.shape[1]
        hidden = hidden + self.pos(torch.arange(length, device=hidden.device))[None]
        mask = torch.triu(
            torch.full((length, length), float("-inf"), device=hidden.device), diagonal=1
        )
        return self.norm(self.net(hidden, mask=mask, is_causal=True)[:, -1])


class ConditionedComposer(nn.Module):
    def __init__(self, memory: ConsolidatedMemory, mode: str, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.memory = memory
        self.mode = mode
        self.local = WindowTransformer(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.head.weight = self.local.emb.weight
        if mode == "gated":
            self.gate = nn.Linear(2 * d_model, d_model)
        elif mode == "film":
            self.film = nn.Linear(d_model, 2 * d_model)
        elif mode == "slots":
            self.slot_proj = nn.Linear(d_model, MEMORY_SLOTS * 2 * d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
        elif mode == "prefix":
            self.prefix_proj = nn.Linear(d_model, MEMORY_SLOTS * d_model)
        elif mode != "residual":
            raise ValueError(mode)
        for parameter in self.memory.parameters():
            parameter.requires_grad = False
        self.memory.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.memory.eval()
        return self

    def compose_hidden(self, windows: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if self.mode == "prefix":
            prefix = self.prefix_proj(memory).view(
                memory.shape[0], MEMORY_SLOTS, D_MODEL
            )
            return self.local.encode(windows, prefix)
        local = self.local.encode(windows)
        if self.mode == "residual":
            return local + memory
        if self.mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([local, memory], dim=-1)))
            return local + gate * memory
        if self.mode == "film":
            scale, shift = self.film(memory).chunk(2, dim=-1)
            return local * (1 + 0.5 * torch.tanh(scale)) + shift
        key_value = self.slot_proj(memory).view(
            memory.shape[0], MEMORY_SLOTS, 2, D_MODEL
        )
        keys, values = key_value[:, :, 0], key_value[:, :, 1]
        query = self.q_proj(local).unsqueeze(1)
        attention = torch.softmax(
            (query * keys).sum(-1) / math.sqrt(D_MODEL), dim=-1
        )
        read = (attention.unsqueeze(-1) * values).sum(1)
        return local + self.out_proj(read)

    def predict_at(
        self,
        ids: torch.Tensor,
        positions: torch.Tensor,
        memory_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_hidden is None:
            with torch.no_grad():
                memory_hidden = self.memory.encode(ids)
        rows = torch.arange(ids.shape[0], device=ids.device)
        memory = memory_hidden[rows, positions]
        return self.head(self.compose_hidden(gather_windows(ids, positions), memory))


@dataclass(slots=True)
class Metrics:
    distance: int
    query_accuracy: float
    local_accuracy: float
    memory_state_accuracy: float


def train_memory(seed: int, steps_per_stage: int, batch: int, device: torch.device):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ConsolidatedMemory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=0.01)
    started = time.perf_counter()
    for stage_index, distance in enumerate(DISTANCES):
        for inner in range(steps_per_stage):
            step = stage_index * steps_per_stage + inner
            inputs, targets, _ = make_batch(
                batch, device, seed=seed * 100000 + step, distance=distance
            )
            hidden = model.encode(inputs)
            states = inputs[:, 0]
            state_logits = model.state_head(hidden[:, distance])
            token_logits = model.token_head(hidden[:, distance])
            loss = F.cross_entropy(state_logits, states) + 0.25 * F.cross_entropy(
                token_logits, targets[:, distance]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model, time.perf_counter() - started


def train_composer(
    memory: ConsolidatedMemory,
    mode: str,
    seed: int,
    steps: int,
    batch: int,
    device: torch.device,
):
    torch.manual_seed(10000 + seed)
    random.seed(10000 + seed)
    model = ConditionedComposer(memory, mode).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-3, weight_decay=0.01)
    started = time.perf_counter()
    for step in range(steps):
        distance = DISTANCES[step % len(DISTANCES)]
        inputs, targets, query_positions = make_batch(
            batch,
            device,
            seed=900000 + seed * 100000 + step,
            distance=distance,
        )
        local_pos = local_positions(
            query_positions,
            seed=800000 + seed * 100000 + step,
            maximum=max(LOCAL_WINDOW, distance - 2),
        )
        with torch.no_grad():
            memory_hidden = model.memory.encode(inputs)
        rows = torch.arange(batch, device=device)
        query_logits = model.predict_at(inputs, query_positions, memory_hidden)
        local_logits = model.predict_at(inputs, local_pos, memory_hidden)
        loss = 4.0 * F.cross_entropy(
            query_logits, targets[rows, query_positions]
        ) + F.cross_entropy(local_logits, targets[rows, local_pos])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    return model, time.perf_counter() - started


def evaluate(model: ConditionedComposer, device: torch.device, distance: int, batch: int = 128):
    model.eval()
    inputs, targets, query_positions = make_batch(
        batch, device, seed=70000 + distance, distance=distance
    )
    local_pos = local_positions(
        query_positions,
        seed=71000 + distance,
        maximum=max(LOCAL_WINDOW, distance - 2),
    )
    rows = torch.arange(batch, device=device)
    with torch.no_grad():
        memory_hidden = model.memory.encode(inputs)
        query = model.predict_at(inputs, query_positions, memory_hidden).argmax(-1)
        local = model.predict_at(inputs, local_pos, memory_hidden).argmax(-1)
        state = model.memory.state_head(memory_hidden[rows, query_positions]).argmax(-1)
    return Metrics(
        distance,
        float((query == targets[rows, query_positions]).float().mean()),
        float((local == targets[rows, local_pos]).float().mean()),
        float((state == inputs[:, 0]).float().mean()),
    )


def run(mode: str, seed: int, memory_steps_per_stage: int, composer_steps: int, batch: int, device: torch.device):
    memory, memory_seconds = train_memory(seed, memory_steps_per_stage, batch, device)
    composer, composer_seconds = train_composer(
        memory, mode, seed, composer_steps, batch, device
    )
    return {
        "mode": mode,
        "seed": seed,
        "memory_train_seconds": memory_seconds,
        "composer_train_seconds": composer_seconds,
        "memory_parameters": sum(p.numel() for p in memory.parameters()),
        "composer_trainable_parameters": sum(
            p.numel() for p in composer.parameters() if p.requires_grad
        ),
        "eval": [asdict(evaluate(composer, device, distance)) for distance in DISTANCES],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["prefix", "residual", "gated", "film", "slots"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--memory-steps-per-stage", type=int, default=80)
    parser.add_argument("--composer-steps", type=int, default=240)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run(
        args.mode,
        args.seed,
        args.memory_steps_per_stage,
        args.composer_steps,
        args.batch,
        device,
    )
    payload = {
        "schema_version": 1,
        "device": str(device),
        "torch_version": torch.__version__,
        "distances": DISTANCES,
        "local_window": LOCAL_WINDOW,
        "query_chance": 1 / N_STATE,
        "long_memory_absolute_position_embedding": False,
        "result": result,
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
