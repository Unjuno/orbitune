# Compound-field recurrent-memory routing proxy

This experiment narrows the recurrent-memory architecture before real-MIDI training. It uses the current 12-field Compound record shape directly, but the records are synthetic. It is not evidence of musical quality and does not replace the real Compound/MIDI gate.

## Question

Does explicit fast / medium / slow memory routing help because of the routing itself, or only because the multibank model has more parameters?

The proxy encodes three causal latent states through Compound-like fields:

- slow: composition-level root class reflected through NOTE pitch;
- medium: program-family state changed every 64 events;
- fast: velocity class changed every 16 events;
- generation auxiliary: next Compound `event_type`.

Sequence length is 128 events. Late-state accuracy is measured from event 80 onward.

## Parameter-matched comparison

Both models use six total memory slots.

### Shared matched

One six-slot recurrent memory, then separate downstream mix paths. An adapter is added so the model has essentially the same parameter count as the routed model.

### Multibank routed

Three independent two-slot banks:

```text
fast bank   decay 0.90  -> fast head
medium bank decay 0.97  -> medium head
slow bank   decay 0.997 -> slow head

all three banks -> next-event head
```

Parameter counts differ by only two parameters:

- shared matched: 30,327;
- multibank routed: 30,329.

## Three-seed result

70 optimization steps, batch 4, seeds 1/2/3:

| Model | Slow late | Medium late | Fast late | Next event type |
| --- | ---: | ---: | ---: | ---: |
| shared matched | 41.48% | 47.89% | 93.68% | 93.40% |
| multibank routed | **66.75%** | **50.58%** | 93.62% | 93.20% |

The important result is the slow-state gap. Because the parameter counts are matched within two parameters, the improvement cannot be explained by raw parameter count alone in this proxy.

Fast-state retention is effectively unchanged. Medium-state retention improves slightly. Next-event-type prediction remains essentially unchanged, so the long-state gain is not bought by a major collapse in the local generation auxiliary.

## Interpretation

The current architecture hypothesis becomes more specific:

```text
Compound Event embedding
  ├─ fast recurrent bank   -> fast latent targets
  ├─ medium recurrent bank -> medium latent targets
  └─ slow recurrent bank   -> slow latent targets
        ↓
explicit consolidated memory interface
        ↓
bounded local Transformer / composer
        ↓
next Compound Event
        ↓
state update
        ↺
```

The evidence favors **separate write/read routing** over one shared recurrent memory that is merely made wider or given extra downstream capacity.

## What this does not prove

This proxy still does not establish:

- real-MIDI musical quality;
- that the fixed decay values are optimal;
- that two slots per timescale are sufficient;
- that the three timescales are the final decomposition;
- superiority to a parameter-matched full-attention production model;
- that final Base pretraining should keep the banks frozen;
- CUDA efficiency.

The synthetic latent labels are deliberately clean. Real Compound-derived targets are noisier and correlated, so they are the next architecture gate.

## Next gate

1. run the parameter-matched shared-vs-routed comparison on `derive_compound_memory_targets(...)` outputs from actual Compound records;
2. measure per-target head accuracy rather than one aggregate timescale label;
3. compare two, three and four memory banks only if the real targets justify them;
4. benchmark the same recurrent scan/recurrent-step path on VLab16 RTX 3080 against PyTorch SDPA;
5. only after real-target and CUDA gates, tune model scale.

## Reproduction

```bash
docker build \
  -f workloads/compound-field-memory-routing-proxy/Dockerfile \
  -t orbitune-compound-field-memory-routing .

for mode in shared_matched multibank_routed; do
  for seed in 1 2 3; do
    docker run --rm orbitune-compound-field-memory-routing \
      --mode "$mode" \
      --seed "$seed" \
      --steps 70 \
      --batch 4 \
      --device cpu \
      --out "/tmp/${mode}-${seed}.json"
  done
done
```

Use `--device cuda` only as a training-device smoke; performance claims must come from the dedicated VLab16 benchmark harness.
