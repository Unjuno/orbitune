# Recurrent memory multiscale routing experiment

This proxy asks whether one recurrent linear-attention memory should carry short-, medium-, and long-horizon state, or whether Orbitune should separate those roles into independent memory banks.

It is an architecture proxy only. The latent variables are synthetic and the result does not freeze the production Base ABI.

## Task

A 256-step stream contains three hidden state processes:

- `slow`: written once at position 0 and never repeated;
- `medium`: updated every 96 steps;
- `fast`: updated every 24 steps.

Filler events dominate the sequence. The memory representation must reconstruct the current event and classify the current slow, medium, and fast states at every position. Evaluation uses the late region beginning at position 160.

This deliberately creates interference: a useful memory has to preserve the original slow state while repeatedly accepting medium and fast updates.

## Compared memory layouts

### Shared decay

One six-slot linear memory with one shared learned decay.

### Per-slot learned decay

One six-slot memory with a separately learned decay for every slot.

### Fixed multiband decay

One six-slot memory with forced decay values:

```text
0.90, 0.90, 0.97, 0.97, 0.995, 0.995
```

The write/read projections are still shared across all six slots.

### Independent multibank

Three independent two-slot banks:

```text
fast bank    decay 0.90
medium bank  decay 0.97
slow bank    decay 0.995
```

Each bank has separate query, key, value, and write projections. Their readouts are concatenated before the latent-state heads.

## Initial result

120 optimization steps, batch 8, sequence length 256.

Single-memory seed 1:

| Layout | Slow late | Medium late | Fast late | Event reconstruction |
| --- | ---: | ---: | ---: | ---: |
| shared decay | 26.5% | 59.4% | 92.8% | 100.0% |
| per-slot learned decay | 26.0% | 60.8% | 90.4% | 100.0% |
| fixed multiband decay | 27.4% | 53.7% | 95.8% | 99.95% |

The per-slot learned decays clustered around roughly 0.968-0.976. They did not spontaneously create a sufficiently slow memory path. Forcing slow decay values without separating the routing projections also did not materially improve slow-state retention.

Independent multibank:

| Seed | Slow late | Medium late | Fast late | Event reconstruction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 64.5% | 98.6% | 99.5% | 99.86% |
| 2 | 79.4% | 79.2% | 99.5% | 100.0% |
| 3 | 94.0% | 91.0% | 100.0% | 99.97% |

The relevant change is not merely the decay constants. The independent-bank layout also prevents the slow path from sharing the same write/read routing parameters with rapidly changing events.

## Current interpretation

The evidence supports carrying this structure forward as the multiscale baseline:

```text
Compound Event embedding
        │
        ├─ fast recurrent bank
        ├─ medium recurrent bank
        └─ slow recurrent bank
              ↓
       consolidated memory
              ↓
       bounded local Transformer
```

This is stronger than assigning different decay constants to slots inside one shared memory. In the current proxy, shared routing appears to be the main source of interference.

The result does **not** prove that the final model needs exactly three banks or these exact decay values. Real Compound-event experiments should determine the appropriate number of timescales and whether the banks need fixed, learned, or data-dependent forgetting.

## Next gate

The next useful experiment is no longer another synthetic state classification task. Replace the three hidden variables with statistics derived from real Compound MIDI, for example:

- fast: recent note density / velocity / local rhythmic state;
- medium: instrumentation, register, pedal and phrase-level state;
- slow: tonal distribution, section/motif identity and long-horizon structural summaries.

Then compare:

1. one shared memory;
2. independent multibank memory;
3. multibank with learned decay inside each bank.

Only after this survives real-MIDI proxy data should the architecture move to the 5M/10M/20M Base scale sweep.

## Reproduction

Build:

```bash
docker build -f workloads/recurrent-memory-multiscale-proxy/Dockerfile -t orbitune-memory-multiscale .
```

Run one arm:

```bash
docker run --rm orbitune-memory-multiscale \
  --mode independent_multibank \
  --seed 1 \
  --steps 120 \
  --batch 8 \
  --out /tmp/result.json
```

On VLab16, add `--gpus all`. CPU timing from this proxy is not a CUDA throughput result.
