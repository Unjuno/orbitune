# Recurrent memory before local Transformer experiment

This experiment tests the architecture direction for an indefinitely looping symbolic-music generator. It is a proxy, not a Base-training benchmark and not a frozen production ABI.

## Question

Can a fixed-size recurrent linear-attention memory retain information outside a bounded local-attention window, and can a downstream local Transformer use that memory without destroying it?

The proxy sequence contains:

- a state token at position 0;
- locally predictable motif tokens;
- a QUERY after the 16-token local window has lost access to the state token;
- an answer token determined only by the original state.

The long-state query has eight possible answers, so chance accuracy is 12.5%.

## Arms

- `A_local`: bounded local Transformer only.
- `B_linear`: fixed-size selective recurrent linear-attention memory only.
- `C_naive_hybrid`: recurrent memory followed directly by the local Transformer.
- `D_consolidated`: recurrent memory with an explicit auxiliary long-state objective, followed by the local Transformer, with the memory state kept as a non-optional residual conditioning path.

The memory recurrence keeps a fixed key/value state. The implementation is deliberately written as a Python loop for inspectability. Its CPU timing is not a performance claim. A fused/scan CUDA implementation must be benchmarked separately on VLab16.

## Initial three-seed result

300 optimization steps, batch 12, sequence length 48, local window 16:

| Arm | Params | Mean long-query accuracy | Mean local accuracy | Mean validation loss |
| --- | ---: | ---: | ---: | ---: |
| A local | 28,576 | 14.1% | 95.3% | 0.363 |
| B linear | 6,882 | 100.0% | 96.6% | 0.154 |
| C naive hybrid | 32,354 | 14.1% | 96.5% | 0.326 |
| D consolidated | 32,354 | 82.0% | 94.7% | 0.201 |

For `D_consolidated`, the memory block's own auxiliary query accuracy was 100% for all three seeds.

## Interpretation

The important result is not that this tiny memory implementation is already the final architecture. It is that simple serial composition is insufficient:

```text
linear memory -> local Transformer
```

can still let the downstream stack ignore or erase the long-memory signal.

The current candidate therefore has a stronger boundary:

```text
Compound Event embedding
        ↓
recurrent linear memory
        ↓
explicit memory objective / consolidation
        ↓
fixed-size memory representation
        ↓
local Transformer over bounded recent context
        ↓
non-optional memory conditioning
        ↓
next Compound Event
```

This matches the product requirement better than an ever-growing GPT KV cache: recurrent state remains bounded while the local Transformer handles precise short-horizon musical structure.

## Next experiments

Do not scale parameter count yet. First validate the memory boundary itself:

1. replace the transparent Python recurrence with a CUDA-suitable scan/fused implementation;
2. compare one versus multiple memory timescales;
3. replace the synthetic state token with Compound-event-derived long-horizon targets;
4. compare memory conditioning mechanisms: fixed residual, gated residual, global memory tokens, and local cross-attention;
5. test chunkwise state carry with gradients detached at chunk boundaries;
6. measure memory retention as the query distance grows well beyond the local window;
7. only after the architecture survives real-MIDI proxy tasks, run the 5M/10M/20M scale sweep.

## Container

Build from the repository root:

```bash
docker build -f workloads/recurrent-memory-arch-proxy/Dockerfile -t orbitune-recurrent-proxy .
```

Run one arm:

```bash
docker run --rm orbitune-recurrent-proxy \
  --model D_consolidated \
  --seed 1 \
  --steps 300 \
  --batch 12 \
  --seq-len 48 \
  --out /tmp/result.json
```

For VLab16 GPU execution, add `--gpus all` and mount an output directory. CPU/GPU numerical results should be treated separately from kernel-throughput benchmarking.
