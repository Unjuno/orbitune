from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_STATE = 8
N_NOTE = 32
NOTE_BASE = N_STATE
QUERY = NOTE_BASE + N_NOTE
ANSWER_BASE = QUERY + 1
VOCAB = ANSWER_BASE + N_STATE


def make_batch(
    batch: int,
    seq_len: int,
    device: torch.device,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a local-pattern plus long-state-recall synthetic sequence.

    The state token appears only at position zero. QUERY positions occur after
    the 16-token local-attention window has lost access to that token. The
    answer is therefore impossible for a bounded local model unless some
    persistent state carries the information forward.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    states = torch.randint(0, N_STATE, (batch,), generator=generator)
    motifs = torch.randint(0, 8, (batch,), generator=generator)
    sequence = torch.empty((batch, seq_len), dtype=torch.long)
    sequence[:, 0] = states
    query_positions = set(range(32, seq_len - 1, 32))

    for position in range(1, seq_len):
        if position in query_positions:
            sequence[:, position] = QUERY
        elif position - 1 in query_positions:
            sequence[:, position] = ANSWER_BASE + states
        else:
            note = (motifs + position + 3 * ((position // 4) % 4)) % N_NOTE
            sequence[:, position] = NOTE_BASE + note

    inputs = sequence[:, :-1].to(device)
    targets = sequence[:, 1:].to(device)
    return inputs, targets, inputs.eq(QUERY)


def local_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(length, device=device)[:, None]
    columns = torch.arange(length, device=device)[None, :]
    blocked = (columns > rows) | ((rows - columns) >= window)
    return torch.zeros((length, length), device=device).masked_fill(blocked, float("-inf"))


class LinearMemoryLayer(nn.Module):
    """Selective recurrent linear-attention memory with fixed-size state.

    This is intentionally a transparent proxy, not a production kernel. The
    recurrence is written as a Python loop so the memory semantics are easy to
    inspect. CUDA throughput must be measured with a fused/scan implementation
    before drawing performance conclusions.
    """

    def __init__(self, width: int = 32, key_width: int = 8) -> None:
        super().__init__()
        self.key_width = key_width
        self.q = nn.Linear(width, key_width, bias=False)
        self.k = nn.Linear(width, key_width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.write = nn.Linear(width, 1)
        self.mix = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)
        self.logit_decay = nn.Parameter(torch.tensor(5.3))
        nn.init.constant_(self.write.bias, -1.5)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        state = hidden.new_zeros((batch, self.key_width, width))
        normalizer = hidden.new_zeros((batch, self.key_width))
        outputs: list[torch.Tensor] = []
        decay = torch.sigmoid(self.logit_decay)

        for position in range(length):
            current = self.norm(hidden[:, position])
            query = F.elu(self.q(current)) + 1.0
            key = F.elu(self.k(current)) + 1.0
            value = self.v(current)
            write = torch.sigmoid(self.write(current))
            state = decay * state + write[:, :, None] * torch.einsum("bk,bd->bkd", key, value)
            normalizer = decay * normalizer + write * key
            readout = torch.einsum("bk,bkd->bd", query, state) / (
                torch.einsum("bk,bk->b", query, normalizer)[:, None] + 1e-5
            )
            outputs.append(self.mix(torch.cat([hidden[:, position], readout], dim=-1)))
        return torch.stack(outputs, dim=1)


class MemoryEncoder(nn.Module):
    def __init__(self, width: int = 32, max_len: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(max_len, width)
        self.memory = LinearMemoryLayer(width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB, bias=False)
        self.head.weight = self.embedding.weight

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.shape[1]
        hidden = self.embedding(token_ids) + self.position(
            torch.arange(length, device=token_ids.device)
        )[None]
        return self.norm(hidden + self.memory(hidden))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(token_ids))


class LocalTransformer(nn.Module):
    def __init__(self, width: int = 32, window: int = 16, max_len: int = 64) -> None:
        super().__init__()
        self.window = window
        self.embedding = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(max_len, width)
        block = nn.TransformerEncoderLayer(
            width,
            4,
            4 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, 2)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        length = token_ids.shape[1]
        hidden = self.embedding(token_ids) + self.position(
            torch.arange(length, device=token_ids.device)
        )[None]
        hidden = self.transformer(
            hidden,
            mask=local_mask(length, self.window, token_ids.device),
            is_causal=True,
        )
        return self.head(self.norm(hidden)), None


class LinearOnly(nn.Module):
    def __init__(self, width: int = 32, max_len: int = 64) -> None:
        super().__init__()
        self.memory = MemoryEncoder(width, max_len)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.memory(token_ids), None


class NaiveMemoryThenTransformer(nn.Module):
    def __init__(self, width: int = 32, window: int = 16, max_len: int = 64) -> None:
        super().__init__()
        self.window = window
        self.memory = MemoryEncoder(width, max_len)
        block = nn.TransformerEncoderLayer(
            width,
            4,
            4 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, 2)
        self.norm = nn.LayerNorm(width)
        self.head = self.memory.head

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        memory = self.memory.encode(token_ids)
        length = token_ids.shape[1]
        hidden = self.transformer(
            memory,
            mask=local_mask(length, self.window, token_ids.device),
            is_causal=True,
        )
        return self.head(self.norm(hidden)), None


class ConsolidatedMemoryThenTransformer(nn.Module):
    """Memory is trained explicitly and cannot be dropped by the local stack."""

    def __init__(self, width: int = 32, window: int = 16, max_len: int = 64) -> None:
        super().__init__()
        self.window = window
        self.memory = MemoryEncoder(width, max_len)
        block = nn.TransformerEncoderLayer(
            width,
            4,
            4 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, 2)
        self.norm = nn.LayerNorm(width)
        self.head = self.memory.head

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.memory.encode(token_ids)
        memory_logits = self.head(memory)
        length = token_ids.shape[1]
        local = self.transformer(
            memory,
            mask=local_mask(length, self.window, token_ids.device),
            is_causal=True,
        )
        # Keep the consolidated memory as a non-optional conditioning path.
        return self.head(self.norm(local + memory)), memory_logits


MODELS: dict[str, type[nn.Module]] = {
    "A_local": LocalTransformer,
    "B_linear": LinearOnly,
    "C_naive_hybrid": NaiveMemoryThenTransformer,
    "D_consolidated": ConsolidatedMemoryThenTransformer,
}


@dataclass
class Result:
    model: str
    seed: int
    parameters: int
    train_seconds: float
    ms_per_step: float
    val_loss: float
    query_accuracy: float
    local_accuracy: float
    memory_aux_query_accuracy: float | None


def evaluate(
    model: nn.Module,
    device: torch.device,
    seq_len: int,
) -> tuple[float, float, float, float | None]:
    model.eval()
    losses: list[float] = []
    query_correct = query_count = local_correct = local_count = 0
    memory_correct = memory_count = 0

    with torch.no_grad():
        for batch_index in range(4):
            inputs, targets, query_mask = make_batch(
                32,
                seq_len,
                device,
                seed=50_000 + batch_index,
            )
            logits, memory_logits = model(inputs)
            losses.append(
                float(F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1)))
            )
            predicted = logits.argmax(dim=-1)
            local_mask_value = ~query_mask
            query_correct += int(((predicted == targets) & query_mask).sum())
            query_count += int(query_mask.sum())
            local_correct += int(((predicted == targets) & local_mask_value).sum())
            local_count += int(local_mask_value.sum())
            if memory_logits is not None:
                memory_predicted = memory_logits.argmax(dim=-1)
                memory_correct += int(((memory_predicted == targets) & query_mask).sum())
                memory_count += int(query_mask.sum())

    memory_accuracy = memory_correct / memory_count if memory_count else None
    return (
        sum(losses) / len(losses),
        query_correct / max(query_count, 1),
        local_correct / max(local_count, 1),
        memory_accuracy,
    )


def run(
    model_name: str,
    seed: int,
    steps: int,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> Result:
    # The seed is applied before model construction so initialization is reproducible.
    torch.manual_seed(seed)
    random.seed(seed)
    model = MODELS[model_name](max_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    model.train()
    started = time.perf_counter()

    for step in range(steps):
        inputs, targets, query_mask = make_batch(
            batch,
            seq_len,
            device,
            seed=seed * 100_000 + step,
        )
        logits, memory_logits = model(inputs)
        raw_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        weights = torch.ones_like(raw_loss)
        weights[query_mask] = 32.0
        loss = (raw_loss * weights).sum() / weights.sum()
        if memory_logits is not None:
            # Explicitly require the memory block itself to retain the long state.
            loss = loss + F.cross_entropy(memory_logits[query_mask], targets[query_mask])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    elapsed = time.perf_counter() - started
    val_loss, query_accuracy, local_accuracy, memory_accuracy = evaluate(
        model,
        device,
        seq_len,
    )
    return Result(
        model=model_name,
        seed=seed,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        train_seconds=elapsed,
        ms_per_step=1000 * elapsed / steps,
        val_loss=val_loss,
        query_accuracy=query_accuracy,
        local_accuracy=local_accuracy,
        memory_aux_query_accuracy=memory_accuracy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run(
        args.model,
        args.seed,
        args.steps,
        args.batch,
        args.seq_len,
        device,
    )
    payload = {
        "schema_version": 2,
        "device": str(device),
        "torch_version": torch.__version__,
        "task": {
            "seq_len": args.seq_len,
            "local_window": 16,
            "query_positions": list(range(32, args.seq_len - 1, 32)),
            "query_chance": 1 / N_STATE,
            "query_loss_weight": 32.0,
        },
        "result": asdict(result),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
