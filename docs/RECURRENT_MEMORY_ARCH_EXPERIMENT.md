# Recurrent memory before local Transformer experiment

This experiment tests the architecture direction for an indefinitely looping symbolic-music generator. It is a proxy, not a Base-training benchmark and not a frozen production ABI.

## Question

Can a fixed-size recurrent linear-attention memory retain information outside a bounded local-attention window, and can a downstream local Transformer use that memory without destroying it?

The proxy sequence contains:

- a state token at position 0;
- locally predictable motif tokens;
- a QUERY outside the 16-token local-attention window;
- an answer token determined only by the original state.

The long-state query has eight possible answers, so chance accuracy is 12.5%.

## Experiment 1: naive versus consolidated memory

Arms:

- `A_local`: bounded local Transformer only.
- `B_linear`: fixed-size selective recurrent linear-attention memory only.
- `C_naive_hybrid`: recurrent memory followed directly by the local Transformer.
- `D_consolidated`: recurrent memory with an explicit auxiliary long-state objective, followed by the local Transformer, with the memory state kept as a non-optional residual conditioning path.

300 optimization steps, batch 12, sequence length 48, local window 16:

| Arm | Params | Mean long-query accuracy | Mean local accuracy | Mean validation loss |
| --- | ---: | ---: | ---: | ---: |
| A local | 28,576 | 14.1% | 95.3% | 0.363 |
| B linear | 6,882 | 100.0% | 96.6% | 0.154 |
| C naive hybrid | 32,354 | 14.1% | 96.5% | 0.326 |
| D consolidated | 32,354 | 82.0% | 94.7% | 0.201 |

For `D_consolidated`, the memory block's own auxiliary query accuracy was 100% for all three seeds.

The important result was that simple serial composition was insufficient:

```text
linear memory -> local Transformer
```

could still let the downstream stack ignore or erase the long-memory signal.

## Experiment 2: explicitly consolidate, freeze, then condition

The follow-up tests the stronger interpretation of the architecture boundary:

```text
stage 1
recurrent linear memory
  -> latent-state classification
  -> current-event reconstruction
  -> consolidate
  -> freeze

stage 2
frozen memory representation
  -> bounded local Transformer
  -> explicit conditioning
  -> next event
```

The memory stage is trained first, then its embedding, recurrent memory and consolidation layers are frozen before downstream training. This makes the experiment test the conditioning interface rather than allowing end-to-end training to repurpose the memory block.

### Memory-retention result

With 300 stage-1 steps and batch 12, all three tested seeds reached 100% latent-state classification accuracy through query distance 256. Event reconstruction was approximately 100% as well.

The local window remains 16, so all tested query distances are outside direct local attention:

```text
32 / 64 / 128 / 256 events
```

### Conditioning sweep

With the same frozen seed-1 memory and 60 downstream optimization steps:

| Conditioning | Trainable params | Query @ 32 | @ 64 | @ 128 | @ 256 | Local acc @ 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed residual | 7,321 | 100% | 100% | 100% | 100% | 74.38% |
| gated residual | 9,073 | 100% | 100% | 100% | 100% | 74.62% |
| cross-attention | 9,721 | 100% | 100% | 100% | 100% | 74.62% |
| memory tokens | 13,369 | 100% | 100% | 100% | 100% | 74.65% |

There is no meaningful accuracy advantage in this proxy for the more complex conditioning paths. `fixed residual` is therefore the reference baseline and the other three remain ablations.

### Seed reproducibility

For `fixed residual`, stage-1 memory accuracy at distance 256 was 100% in seeds 1, 2 and 3.

At 60 downstream steps:

- seed 1: 100% query accuracy at distance 256;
- seed 2: 90.625% at distance 256;
- seed 3: 100% at distance 256.

Extending only seed 2 from 60 to 100 downstream steps raised distance-256 query accuracy to 100% while the frozen memory remained at 100% throughout. This points to downstream convergence variance rather than loss of long-term state.

## Recurrent/parallel equivalence

The two-stage proxy uses a discounted cumulative scan during training so CPU experiments do not spend most of their time in a Python recurrence loop. The represented state update is still recurrent:

```text
S_t = decay * S_(t-1) + write_t * k_t v_t^T
Z_t = decay * Z_(t-1) + write_t * k_t
```

CI compares the vectorized training implementation with an explicit sequential recurrence and requires numerical agreement. This preserves the intended fixed-size streaming inference semantics.

## Current architecture candidate

The current reference hypothesis is now narrower:

```text
Compound Event embedding
        ↓
recurrent linear memory
        ↓
explicit consolidation objectives
        ↓
fixed-size memory state
        ↓
freeze/separate memory boundary during architecture validation
        ↓
local Transformer over bounded recent context
        ↓
fixed residual memory conditioning  <- reference baseline
        ↓
next Compound Event
        ↓
state update
        ↺
```

This is not yet a decision to freeze the production training stack. Freezing the memory in this proxy is an experimental instrument that proves the interface can preserve long-horizon information. Later real-MIDI experiments must compare permanently frozen memory against staged unfreezing or joint fine-tuning.

## What the result does not prove

The proxy does not establish:

- CUDA throughput or memory efficiency;
- superiority over a production-grade full-attention baseline;
- real-MIDI musical quality;
- the correct latent targets for music;
- the correct number or timescales of memory states;
- that the memory should remain frozen in final Base pretraining;
- the final 5M/10M/20M parameter scale.

CPU timings are implementation diagnostics only. They are not evidence that the current Python/PyTorch linear-attention kernel is faster than SDPA.

## Next experiments

Do not run the parameter-scale sweep yet. The next architecture experiments are:

1. implement CUDA-suitable scan/recurrent execution and benchmark it on VLab16 RTX 3080;
2. replace the synthetic latent state with Compound-event-derived targets;
3. test fast / medium / slow recurrent memory states rather than one decay timescale;
4. test chunkwise state carry across much longer streams with bounded recent context;
5. compare frozen memory, staged unfreezing and low-LR joint fine-tuning on the same real-MIDI proxy;
6. verify that fixed residual remains sufficient once the latent target is no longer trivial;
7. only then run 5M / 10M / 20M scale experiments.

## Containers

Initial four-arm experiment:

```bash
docker build -f workloads/recurrent-memory-arch-proxy/Dockerfile -t orbitune-recurrent-proxy .

docker run --rm orbitune-recurrent-proxy \
  --model D_consolidated \
  --seed 1 \
  --steps 300 \
  --batch 12 \
  --seq-len 48 \
  --out /tmp/result.json
```

Two-stage conditioning sweep:

```bash
docker build -f workloads/recurrent-memory-two-stage-sweep/Dockerfile -t orbitune-recurrent-two-stage .

docker run --rm orbitune-recurrent-two-stage \
  --model fixed_residual \
  --seed 1 \
  --memory-steps 300 \
  --down-steps 100 \
  --batch 12 \
  --max-distance 256 \
  --out /tmp/result.json
```

For VLab16 GPU execution, add `--gpus all` and mount an output directory. CPU/GPU numerical results and CUDA throughput results must be recorded separately.
