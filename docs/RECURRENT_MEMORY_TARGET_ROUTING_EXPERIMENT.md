# Balanced Compound memory-target routing experiment

This experiment is the closest architecture proxy in PR #15 to the intended recurrent-memory objective. It generates valid synthetic Compound records, derives the repository's actual experimental causal memory targets with `derive_compound_memory_targets(...)`, and compares a parameter-matched shared memory against explicitly routed fast/medium/slow banks.

It is still synthetic and does not replace real-MIDI validation.

## Why the corpus was balanced

An earlier synthetic corpus made several targets nearly constant (for example note-density and pitch-entropy), so raw accuracy could be inflated by majority prediction. That run is not used for the architecture decision.

The replacement corpus varies:

- musical delta / event density;
- velocity;
- register;
- pitch-set size and entropy;
- channel count;
- program-family count and changes;
- pedal state;
- tempo over a broad range;
- explicit non-note rest stretches.

Training uses class-balanced cross entropy per target head. Evaluation uses macro recall per target head, averaged within fast / medium / slow tiers.

## Memory-only consolidation boundary

The memory target heads do not see the current event embedding directly. They only see a readout from recurrent memory. The next-event auxiliary can see the current event plus memory.

This prevents the memory objective from being solved through a current-event shortcut and directly tests the intended consolidation boundary.

## Parameter-matched models

Both models use six total associative slots.

- `shared_matched`: one six-slot bank at decay 0.97 plus an adapter for parameter matching;
- `multibank_routed`: three independent two-slot banks at decays 0.90 / 0.97 / 0.997, routed to fast / medium / slow target heads.

Parameter counts:

- shared: 43,984;
- routed multibank: 43,986.

## Three-seed result

60 steps, batch 4, sequence length 128:

| Model | Fast macro recall | Medium macro recall | Slow macro recall | Next event type |
| --- | ---: | ---: | ---: | ---: |
| shared matched | 26.82% | 28.71% | 17.13% | 88.95% |
| routed multibank | **47.40%** | **30.93%** | **18.58%** | 88.97% |
| delta | **+20.57 pt** | **+2.21 pt** | **+1.45 pt** | +0.02 pt |

The strongest evidence is for the fast bank. Medium improves modestly. Slow improves only slightly and remains an open design problem.

## Slow-bank ablations

Two follow-up tests were run on the same balanced target setup.

### Additive accumulator

The slow Q/K associative bank was replaced by a decay-free learned cumulative weighted average. Mean slow macro recall fell from **18.58% to 17.06%**. Fast and medium metrics also fell slightly. A simple prefix accumulator is therefore not a better slow-memory default in this proxy.

### Learned decay

The slow associative bank's decay was made learnable, initialized at 0.997. Across three seeds it converged only to approximately **0.99688–0.99696**, and the output metrics were effectively unchanged from the fixed-decay baseline. The weak slow result therefore is not explained by choosing 0.997 instead of a nearby decay.

These negative results narrow the open question: the remaining slow-memory issue is more likely the target representation, routing/state structure, or optimization horizon than a trivial decay-value choice.

## Current architecture interpretation

The result supports a staged architecture with an explicit memory subsystem rather than a generic Transformer with interleaved linear layers:

```text
Compound Event
    ↓
factorized event embedding
    ↓
fast recurrent memory   ──> fast consolidation objectives
medium recurrent memory ──> medium consolidation objectives
slow recurrent memory   ──> slow consolidation objectives (still experimental)
    ↓
consolidated memory interface
    ↓
bounded local Transformer / composer
    ↓
next Compound Event
    ↓
fixed-size state update
    ↺
```

The fast / medium / slow bank count and fixed decay values are not frozen. Current evidence supports explicit routing most strongly for fast memory, modestly for medium memory, and only weakly for slow memory.

## Required next gates

1. replace balanced synthetic records with provenance-approved real MIDI converted to Compound records;
2. profile real target distributions and choose class balancing from those distributions;
3. compare shared vs routed memory on the same immutable real-data split;
4. redesign slow objectives/representation only if real-data results reproduce the weak slow signal;
5. run the VLab16 RTX 3080 benchmark harness and compare recurrent-state memory / throughput with PyTorch SDPA;
6. only after real-data and GPU gates, choose Base parameter scale.

## Reproduction

```bash
docker build \
  -f workloads/compound-memory-target-routing-proxy/Dockerfile \
  -t orbitune-compound-memory-target-routing .

for mode in shared_matched multibank_routed; do
  for seed in 1 2 3; do
    docker run --rm orbitune-compound-memory-target-routing \
      --mode "$mode" \
      --seed "$seed" \
      --steps 60 \
      --batch 4 \
      --device cpu \
      --out "/tmp/${mode}-${seed}.json"
  done
done
```
