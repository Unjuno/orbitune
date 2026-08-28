# Staged memory freeze-policy experiment

This experiment tests the architecture idea that recurrent memory should first be consolidated, then protected while the downstream composer learns.

It uses the balanced synthetic Compound target setup from `RECURRENT_MEMORY_TARGET_ROUTING_EXPERIMENT.md`. The recurrent model is the routed fast / medium / slow multibank candidate.

## Procedure

Stage 1 trains only the causal memory targets. The next-event composer path is not part of the loss.

Stage 2 trains only next `event_type` prediction under three policies:

- `frozen`: embedding + recurrent memory are frozen; only event mixing/head layers train;
- `low_lr`: composer trains at 3e-3 while embedding + recurrent memory train at 3e-4;
- `joint`: the full model trains at 3e-3.

This is a gradient-interference proxy. The final bounded Local Transformer is not modeled here, so the absolute next-event accuracy is not a production quality metric.

## Three-seed result

Stage 1: 60 steps. Stage 2: 80 steps. Batch 4.

Stage-1 memory mean:

- fast macro recall: 47.47%;
- medium macro recall: 31.26%;
- slow macro recall: 18.91%.

| Stage-2 policy | Fast after | Medium after | Slow after | Next event type |
| --- | ---: | ---: | ---: | ---: |
| frozen | **47.47%** | **31.26%** | **18.91%** | 89.05% |
| low LR | 47.84% | 31.35% | 17.10% | 89.05% |
| joint | 42.94% | 28.24% | 17.10% | **89.39%** |

Memory delta versus stage 1:

- frozen: 0 / 0 / 0 pt;
- low LR: +0.37 / +0.09 / -1.81 pt;
- joint: **-4.53 / -3.02 / -1.81 pt**.

Full joint optimization gains only about **+0.34 percentage point** on the next-event auxiliary over frozen memory, while materially degrading fast and medium memory representation.

## Interpretation

The current evidence supports the user's proposed separation:

```text
stage 1
recurrent memory
  -> explicit consolidation objectives
  -> stabilize / checkpoint

stage 2
bounded composer
  -> reads consolidated memory
  -> memory frozen initially
  -> optional low-LR unfreeze only after validation
```

The result does **not** prove that production Base pretraining should permanently freeze memory. It does establish a useful default for the next experiments: do not start with unrestricted same-LR joint training.

## Next real-data test

On real Compound MIDI, compare:

1. permanently frozen memory;
2. frozen warmup then staged unfreeze;
3. memory LR at 0.1x composer LR;
4. fully joint training.

Track both generation loss and the memory target metrics. Any composer gain that destroys the memory representation should be treated as a regression rather than a free improvement.
