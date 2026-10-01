# Compound TBPTT continuation over existing indexed corpora

## Scope and status

This is an implementation/migration guide, **not evidence that the local production run has switched to TBPTT**, reached its target, or improved musical quality. The existing fixed-window CFE trainer, model architecture, tokenizer, checkpoints, indexes and Web ABI are not changed by this path. Keep a currently running trainer and its watchdog untouched while reviewing this implementation.

The existing `orbitune/compound_tbptt.py` supplies the differentiable native stream path. Local/medium/global histories and fast/medium/slow recurrent memory carry across successive chunks of a song; a song boundary resets only its batch lane. At chunk boundaries, tensors are detached. State values can carry older context, but gradients do **not** propagate through the detached history. This is truncated BPTT, not full-song backpropagation or proof of learned long-range musical structure.

## Sources: reuse Aria and GigaMIDI, do not rebuild

`scripts/compound_tbptt_train.py` accepts one or comma-separated sources, in the supplied order:

- a flat index directory or `index.json` (including an adjacent `songs.jsonl` alias);
- a sharded index directory or `index_manifest.json` using the repository's existing shard path convention;
- multiple flat/sharded indexed corpora, including Aria int32 and GigaMIDI uint8;
- one or multiple in-memory **record** JSONLs, for bounded tests.

A raw MIDI ingest manifest is not a record JSONL. Indexed and in-memory sources cannot be mixed in the same argument. Empty entries, repeated sources and repeated resolved record-store paths are rejected. Literal commas in source paths are not supported by this comma-separated interface.

Memmaps remain read-only. Only selected windows are promoted to int64; no concatenation, MIDI reparse, record-store rewrite, or narrowing cast is introduced. Existing uint8 encodings are read as stored; this change does not prove that earlier encoding conversions preserved all values.

The loader checks per-shard metadata, record byte size, tokenizer/record-width declarations, song counts, nonnegative finite weights and contiguous song ranges. Ordered source/index metadata and song-table hashes form the resume identity. This reads metadata/song tables, **not all record bytes**. It is not a replacement for the prior immutable index/hash audit. Use completed, verified index artifacts; do not train from directories still being written.

## Cumulative event target

Choose exactly one target option:

```text
--steps 1000000
```

or:

```text
--target-events 4096000000
```

`--target-events-seen` is an alias for the latter. This target includes `events_seen` inherited from the parent. It counts sampled next-event positions used in successful optimizer updates, not the 12 individual integer fields of a record. It is neither a count of unique examples nor an exact epoch.

| Counter / setting | Meaning | Unit / type | Preconditions |
| --- | --- | --- | --- |
| `step` | Cumulative completed optimizer updates | Count, integer | Nonnegative; inherited at transition |
| `events_seen` | Cumulative sampled next-event positions | Count, integer | Nonnegative, inherited from actual parent metadata |
| `batch_size` | Independent song lanes per update | Count, integer | Positive and fixed within a resumed TBPTT run |
| `seq_len` | Input positions per lane/chunk | Count, integer | Positive and fixed within a resumed TBPTT run |
| `target_events` | Cumulative requested exposure threshold | Count, integer | Positive; mutually exclusive with `--steps` |
| `tbptt_events_seen` | Exposure added since the first TBPTT transition | Count, integer | Excludes all inherited fixed-window exposure |

All are dimensionless counts, not seconds or audio samples. Each full update contributes `batch_size * seq_len` positions. A target between batch boundaries rounds **up** to a complete update, with overshoot strictly smaller than one full batch. The final checkpoint is saved even when the stopping step is not on the ordinary checkpoint cadence. An already-met target causes no further update or save.

For example, a parent reporting step 104,750 and 429,056,000 positions still reaches the 4,096,000,000 target at step 1,000,000 if the new geometry is 16 by 256. With 4 by 64, the stopping step is different; inherited positions are not recomputed using the new geometry. No 32-bit counter is used for the four-billion-position target.

Runtime records `tbptt_origin_step` and `tbptt_origin_events_seen`; logs distinguish total `events_seen` from `tbptt_events_seen`. Earlier fixed-window exposure must never be described as state-carry training.

## Migration and parent protection

Use a copy of the **latest verified resumable checkpoint**, not an automatic rollback to the historical 100k parent. Hash and preserve the selected fork point before the local experiment. Use a new output directory. On fixed-window-to-TBPTT transition the trainer rejects any output path, including `.best.pt` and `.healthy.pt`, that aliases the input parent (including resolvable symlinks/hardlinks).

The transition requires an explicit `--override-resume-lr`. This acknowledges a different training regime; it does not require a particular numeric LR. Choose the trial LR from measured stability, rather than treating example values as an optimum. Checkpoint model configuration is authoritative; `--n-head` verifies it rather than reshaping a resumed model. The default fresh configuration is the checked-in 8-head Compound family.

Model and optimizer states are retained. New TBPTT lanes start at song boundaries, not at a fictional continuation of random fixed windows. Legacy missing sampler RNG is reported, never reconstructed by assertion. New checkpoints preserve the separate sampler RNG, lane positions, stream state, optimizer and runtime identities.

A measured parent `events_seen` field is required; a legacy parser's fallback from `step` is not accepted as exposure accounting. Fixed-window validation/best-loss history is reset at transition because streaming-state validation is a different protocol.

## Same-mode resume: reject silent resets

A TBPTT resume requires optimizer state, sampler and global RNG states, lane state and stream state. CUDA runs also require saved CUDA RNG, and fp16 requires GradScaler state. The sampler restores the next song/chunk positions; stream steps must agree with the saved offsets.

Resume binds batch/chunk geometry, precision, model heads, causal path, weighting mode, training source identity, validation source identity and validation protocol identity. Source order is significant. Changing validation chunk length or ordered song selection is a protocol change, not continuation of the same best-loss series.

Older experimental TBPTT checkpoints without these identities are rejected and require a separate explicit migration audit. Do not invent missing identities to force them through the guard.

## Sampling limitations remain explicit

Song starts still use the existing replacement sampler. A selected song advances sequentially, but another lane or later draw may choose it again. Incomplete final chunks are dropped and songs shorter than the chunk requirement are ineligible. This path does **not** implement no-replacement coverage, visit every event, or remove every repeated example. Weighted and unweighted song-continuous sampling also have different exposure distributions from random-window training; retain and report the selected policy.

## Validation and local GPU gate

The default first-two-song validation is a smoke-size check only. For a multi-source quality comparison, choose an explicit, fixed, representative held-out selection. Both the fixed-window control checkpoint and the TBPTT candidate must be evaluated with the same streaming/prefix protocol; compare fixed-window metrics separately. A lower loss from a changed protocol is not evidence of improvement by itself.

Before switching the production process, the local training agent must:

1. verify actual source descriptors and fork-point checkpoint on disk;
2. run a short GPU forward/backward and save/resume trial on a copy;
3. verify chunk state carry, per-song reset, finite updates and actual CUDA/BF16 resume behavior;
4. measure warmed steady-state throughput after medium/global histories have filled, not only the empty-state startup;
5. compare an equal-exposure control and candidate with held-out streaming loss and generated MIDI/listening checks;
6. only then make an explicit production transition, keeping the old lineage recoverable and avoiding two trainers on the same GPU.

The CUDA trainer remains CUDA-only. CPU test monkeypatches exercise the actual orchestration with a tiny model but are not GPU performance evidence. Existing CFE PATCH A+B performance numbers do not transfer automatically: the TBPTT path retains `safe_backward_step`, including its non-finite checks, and is not the buffered CFE hot loop.

A recurrence/history carry implementation is necessary for this experiment, not a guarantee of better music. No new GPU throughput or model-quality result is claimed here.

## Tests

```bash
python -m pytest -q tests/test_tbptt_run_support.py tests/test_compound_tbptt.py
```

New tests cover mixed int32/uint8 memmaps, report-only shards, metadata corruption, ordered identities, replacement-sampler resume, billion-position arithmetic, final off-cadence save, parent-output protection and actual uninterrupted-versus-resumed TBPTT training with AdamW/dropout on CPU. Existing tests cover native stream value equivalence, per-lane reset and graph detach.

No checkpoint, ONNX graph, corpus or MIDI output is published by these changes. Existing rights and publication gates remain unchanged.
