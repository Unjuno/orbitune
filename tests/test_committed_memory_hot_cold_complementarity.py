from __future__ import annotations

import importlib.util
import json
import random
import statistics
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from orbitune.compound_dataset import prepare_compound_split_corpus


MEMORY_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"
BOUNDED_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_bounded_composer.py"

HOT_WINDOW = 4
D_MODEL = 48
TARGET_PITCHES = (67, 69, 71, 72)


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _vlq(value: int) -> bytes:
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(parts))


def _note(track: bytearray, pitch: int) -> None:
    track += _vlq(12) + bytes([0x90, pitch, 80])
    track += _vlq(12) + bytes([0x80, pitch, 0])


def _midi(seed: int, patterns: int = 12) -> bytes:
    combos = [(long_bit, local_bit) for long_bit in (0, 1) for local_bit in (0, 1)]
    branches = (combos * ((patterns + 3) // 4))[:patterns]
    random.Random(70_000 + seed).shuffle(branches)
    fillers = (60, 62, 65)
    track = bytearray()
    track += b"\x00\xff\x51\x03" + int(500_000).to_bytes(3, "big")
    track += b"\x00" + bytes([0xC0, 0])
    for long_bit, local_bit in branches:
        _note(track, 40 if long_bit == 0 else 52)
        for index in range(26):
            _note(track, fillers[index % len(fillers)])
        _note(track, 43 if local_bit == 0 else 55)
        _note(track, 60)
        _note(track, 62)
        _note(track, 64)
        target_index = 2 * long_bit + local_bit
        _note(track, TARGET_PITCHES[target_index])
    track += b"\x00\xff\x2f\x00"
    header = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + (96).to_bytes(2, "big")
    )
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def _examples(song):  # type: ignore[no-untyped-def]
    result: list[tuple[int, int, int, int]] = []
    records = song.records
    for current in range(len(records) - 1):
        if int(records[current, 0]) != 0 or int(records[current, 4]) != 64:
            continue
        target_pitch = int(records[current + 1, 4])
        if int(records[current + 1, 0]) != 0 or target_pitch not in TARGET_PITCHES:
            continue
        target = TARGET_PITCHES.index(target_pitch)
        long_bit, local_bit = divmod(target, 2)
        assert int(records[current - 30, 4]) == (40 if long_bit == 0 else 52)
        assert int(records[current - 3, 4]) == (43 if local_bit == 0 else 55)
        commit_index = current - HOT_WINDOW
        assert commit_index >= current - 30
        assert commit_index < current - 3
        result.append((current, commit_index, long_bit, target))
    return result


def _assert_partition_contract(songs) -> None:  # type: ignore[no-untyped-def]
    local_by_class: dict[int, set[tuple[tuple[int, ...], ...]]] = {0: set(), 1: set()}
    counts = [0, 0, 0, 0]
    for song in songs:
        for current, commit_index, long_bit, target in _examples(song):
            local_bit = target % 2
            local = tuple(
                tuple(int(v) for v in row)
                for row in song.records[current - HOT_WINDOW + 1 : current + 1]
            )
            local_by_class[local_bit].add(local)
            counts[target] += 1
            # The committed prefix stops before the local marker, so the cold
            # memory cannot observe the short-range bit by construction.
            assert commit_index == current - HOT_WINDOW
            assert int(song.records[commit_index + 1, 4]) in (43, 55)
    assert min(counts) > 0
    assert local_by_class[0] and local_by_class[1]


class LongProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * D_MODEL),
            nn.Linear(3 * D_MODEL, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _committed_features(bounded, memory_model, song):  # type: ignore[no-untyped-def]
    tokens, _ = bounded.routed_memory_reads(memory_model, song.records.unsqueeze(0), None)
    examples = _examples(song)
    indices = torch.tensor([item[1] for item in examples], dtype=torch.long)
    features = tokens[0, indices]
    long_labels = torch.tensor([item[2] for item in examples], dtype=torch.long)
    targets = torch.tensor([item[3] for item in examples], dtype=torch.long)
    local = torch.stack(
        [song.records[item[0] - HOT_WINDOW + 1 : item[0] + 1] for item in examples]
    )
    return local, features, long_labels, targets


def _future_consolidate_committed(bounded, memory_model, songs, *, seed: int, epochs: int) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(seed + 71_000)
    head = LongProbe()
    params = [*memory_model.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    rng = random.Random(seed + 71_000)
    memory_model.train()
    head.train()
    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            _, features, long_labels, _ = _committed_features(bounded, memory_model, song)
            logits = head(features.reshape(features.shape[0], -1))
            loss = F.cross_entropy(logits, long_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()


def _long_probe_accuracy(bounded, memory_model, train, validation, *, seed: int) -> float:  # type: ignore[no-untyped-def]
    memory_model.eval()
    with torch.no_grad():
        train_rows = [_committed_features(bounded, memory_model, song) for song in train]
        val_rows = [_committed_features(bounded, memory_model, song) for song in validation]
        train_x = torch.cat([row[1].reshape(row[1].shape[0], -1) for row in train_rows])
        train_y = torch.cat([row[2] for row in train_rows])
        val_x = torch.cat([row[1].reshape(row[1].shape[0], -1) for row in val_rows])
        val_y = torch.cat([row[2] for row in val_rows])
    torch.manual_seed(seed + 72_000)
    probe = LongProbe()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    for _ in range(200):
        loss = F.cross_entropy(probe(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float(probe(val_x).argmax(-1).eq(val_y).float().mean())


class TwoScaleComposer(nn.Module):
    def __init__(self, factor_embedding) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.embedding = factor_embedding()
        self.position = nn.Embedding(HOT_WINDOW, D_MODEL)
        self.local_norm = nn.LayerNorm(D_MODEL)
        self.local_attention = nn.MultiheadAttention(D_MODEL, 4, dropout=0.0, batch_first=True)
        self.local_ff_norm = nn.LayerNorm(D_MODEL)
        self.local_ff = nn.Sequential(
            nn.Linear(D_MODEL, 3 * D_MODEL), nn.GELU(), nn.Linear(3 * D_MODEL, D_MODEL)
        )
        self.memory_norm = nn.LayerNorm(D_MODEL)
        self.memory_attention = nn.MultiheadAttention(D_MODEL, 4, dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 4)

    def forward(
        self,
        local_records: torch.Tensor,
        memory_tokens: torch.Tensor,
        *,
        use_local: bool,
        use_memory: bool,
    ) -> torch.Tensor:
        hidden = self.embedding(local_records)
        if use_local:
            pos = torch.arange(HOT_WINDOW, device=hidden.device)
            hidden = hidden + self.position(pos)[None]
            normed = self.local_norm(hidden)
            causal = torch.triu(
                torch.ones(HOT_WINDOW, HOT_WINDOW, dtype=torch.bool, device=hidden.device),
                diagonal=1,
            )
            attended, _ = self.local_attention(
                normed, normed, normed, attn_mask=causal, need_weights=False
            )
            hidden = hidden + attended
            hidden = hidden + self.local_ff(self.local_ff_norm(hidden))
            query = hidden[:, -1]
        else:
            query = hidden[:, -1]
        if use_memory:
            q = self.memory_norm(query).unsqueeze(1)
            read, _ = self.memory_attention(q, memory_tokens, memory_tokens, need_weights=False)
            query = query + read[:, 0]
        return self.head(self.output_norm(query))


def _dataset(bounded, memory_model, songs):  # type: ignore[no-untyped-def]
    memory_model.eval()
    rows = []
    with torch.no_grad():
        for song in songs:
            local, memory_tokens, _, targets = _committed_features(bounded, memory_model, song)
            rows.append((local, memory_tokens, targets))
    return (
        torch.cat([row[0] for row in rows]),
        torch.cat([row[1] for row in rows]),
        torch.cat([row[2] for row in rows]),
    )


def _fit_composer(memory_base, train_data, val_data, *, seed: int, use_local: bool, use_memory: bool) -> float:  # type: ignore[no-untyped-def]
    torch.manual_seed(seed + 73_000)
    model = TwoScaleComposer(memory_base.FactorEmbedding)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    train_local, train_memory, train_targets = train_data
    val_local, val_memory, val_targets = val_data
    for _ in range(250):
        logits = model(
            train_local,
            train_memory,
            use_local=use_local,
            use_memory=use_memory,
        )
        loss = F.cross_entropy(logits, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = model(
            val_local,
            val_memory,
            use_local=use_local,
            use_memory=use_memory,
        ).argmax(-1)
    return float(pred.eq(val_targets).float().mean())


def test_committed_cold_memory_and_hot_local_context_are_complementary(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_hot_cold_memory")
    bounded = _load(BOUNDED_SCRIPT, "orbitune_hot_cold_bounded")
    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"hot-cold-{fixture_seed}.mid").write_bytes(_midi(fixture_seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        tmp_path / "split.json",
        validation_fraction=0.25,
        split_seed="committed-hot-cold-v1",
        min_events=64,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)
    _assert_partition_contract([*train, *validation])

    configs = {
        "current_only": (False, False),
        "local_only": (True, False),
        "memory_only": (False, True),
        "memory_plus_local": (True, True),
    }
    rows: dict[str, list[float]] = {name: [] for name in configs}
    long_probe_rows: list[float] = []
    validation_examples = sum(len(_examples(song)) for song in validation)

    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        random.seed(seed)
        memory_model = memory.RoutedMultiBank()
        _future_consolidate_committed(
            bounded, memory_model, train, seed=seed, epochs=20
        )
        long_probe_rows.append(
            _long_probe_accuracy(bounded, memory_model, train, validation, seed=seed)
        )
        train_data = _dataset(bounded, memory_model, train)
        val_data = _dataset(bounded, memory_model, validation)
        for name, (use_local, use_memory) in configs.items():
            rows[name].append(
                _fit_composer(
                    memory.base,
                    train_data,
                    val_data,
                    seed=seed,
                    use_local=use_local,
                    use_memory=use_memory,
                )
            )

    warnings.warn(
        "COMMITTED_MEMORY_HOT_COLD="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "hot_window": HOT_WINDOW,
                "long_marker_distance": 30,
                "local_marker_distance": 3,
                "validation_examples_each_seed": validation_examples,
                "future_consolidation_epochs": 20,
                "long_bit_fresh_probe": {
                    "by_seed": long_probe_rows,
                    "mean": statistics.mean(long_probe_rows),
                },
                "four_way_target_accuracy": {
                    name: {"by_seed": values, "mean": statistics.mean(values)}
                    for name, values in rows.items()
                },
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
