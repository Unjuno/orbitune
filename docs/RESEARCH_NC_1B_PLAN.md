# Research-NC 1B event plan

## Objective

Build a research-only Orbitune lineage that can reach roughly **1 billion post-dedup active next-event pairs** without weakening the commercial-safe lineage.

The commercial checkpoint remains the clean ancestor. Research/non-commercial data may only enter descendants that are explicitly marked non-commercial.

```text
commercial-safe corpus
        |
        v
commercial-clean Base
        |
        +--------------------------+
        |                          |
        v                          v
commercial descendants       research-NC descendants
PROD sources only             PROD + RESEARCH_NC sources
commercial_eligible=true      commercial_eligible=false
```

A checkpoint that has any RESEARCH_NC or more restrictive ancestor must never be promoted, merged, distilled, or otherwise represented as a commercial-eligible checkpoint.

## Frozen starting point

The production v4 baseline remains immutable:

- train songs: 215,963
- train records: 234,904,281
- 1x active events: 234,688,318
- manifest SHA-256: `1c582a08a3087952a57a604b5652cae2bef6dd4e6acb2addc5eb49cdfbe58c72`

Commercial v5 source admission continues independently. The OpenScore StringQuartets delta is handled by its own source-specific integration flow and is not part of this planning PR.

The commercial-safe corpus does **not** need to reach 1B before training. Its purpose is to provide a clean, auditable ancestor checkpoint.

## 1B metric

The 1B target refers to:

```text
POST_DEDUP_ACTIVE_NEXT_EVENT_PAIRS
```

Do not substitute any of the following:

- raw MIDI note count;
- source-reported note events;
- file count;
- `TRAIN_RECORDS`;
- pre-dedup Compound-event count.

Every source must be converted through the production Orbitune representation and measured after the applicable quality gates and normalized-event deduplication.

## Source classes

Use four source classes:

| Class | Training allowed | Commercial lineage | Meaning |
| --- | --- | --- | --- |
| `PROD` | yes | yes | commercial-safe source with closed rights/provenance gates |
| `RESEARCH_NC` | yes | no | source explicitly usable for non-commercial research/training |
| `HOLD` | no | no | rights, acquisition, provenance, or model-use permission remains unresolved |
| `REJECT` | no | no | incompatible terms or prohibited use |

`NC` is not synonymous with `HOLD`: a clearly licensed non-commercial research dataset may be used in a research-NC lineage. Conversely, an unknown or ambiguous license is not made usable by merely labeling the resulting checkpoint non-commercial.

## Checkpoint rights contract

Every new Base manifest must record its lineage and rights state, including:

- parent checkpoint identity, if any;
- `commercial_eligible`;
- `distribution_scope`;
- `license_policy`;
- exact corpus registry;
- exact corpus manifest SHA-256;
- restricted/non-commercial source ids;
- a concise rights summary.

Allowed policies:

- `prod-only`: only PROD ancestry; commercial eligibility may be true;
- `research-nc`: contains RESEARCH_NC ancestry; commercial eligibility must be false;
- `restricted`: internal/restricted use only; commercial eligibility must be false.

No research-NC or restricted checkpoint may become the parent of a `commercial_eligible=true` Base.

## Phase 0 — freeze the commercial ancestor

Complete the current commercial-v5 source-specific work and produce an exact commercial-clean Base checkpoint.

Required artifacts before branching:

- exact corpus registry;
- corpus manifest SHA-256;
- train/validation/test counts;
- 1x sampler active-event invariant;
- checkpoint SHA-256;
- checkpoint manifest with `license_policy=prod-only` and `commercial_eligible=true`.

Training and source expansion are separate decisions. Additional PROD sources may continue to improve a later commercial Base even after a research branch exists.

## Phase 1 — GigaMIDI feasibility census

GigaMIDI is the first research-NC volume candidate. Do not ingest the entire corpus before measuring it in Orbitune.

Pin an exact dataset revision/release and record the access terms in evidence. Then run deterministic sample censuses, preferably at 10k and 50k files, measuring:

- parse success rate;
- rejected/corrupt rate;
- active events per file distribution;
- track/instrument distribution;
- duration and event-count outliers;
- intra-source normalized-event duplicates;
- cross-commercial normalized-event overlap;
- retained active events after quality filters;
- estimated full-corpus active events with uncertainty bounds.

A source-reported MIDI note count is planning evidence only. It is not the Orbitune 1B metric.

## Phase 2 — exact GigaMIDI research corpus

If the feasibility census supports proceeding and the research-use evidence remains valid:

1. create a dedicated `research-nc` corpus registry;
2. pin the exact GigaMIDI revision;
3. convert the complete admitted subset;
4. run intra-source normalized-event dedup;
5. run cross-commercial normalized-event dedup;
6. freeze a manifest and corpus identity;
7. measure exact post-dedup active events.

The first target is approximately:

```text
commercial-clean contribution + retained research-NC contribution >= 1,000,000,000 active events
```

Do not ingest extra data merely to exceed the number if quality measurements deteriorate.

## Phase 3 — Aria-MIDI only if needed or useful

Aria-MIDI is a second research-NC candidate, primarily for expressive solo-piano coverage.

Use the official pruned/deduplicated generative-model subset rather than the unfiltered full set unless measurements justify otherwise. Before admission, measure cross-overlap with both the commercial corpus and GigaMIDI.

Aria-MIDI is not required if GigaMIDI already supplies sufficient high-quality volume and the desired instrumentation balance. It may still be useful as a controlled expressive-piano mixture.

## Phase 4 — mixture and quality gate

Do not assume that a 1B corpus should be sampled uniformly.

Evaluate at least:

- commercial-clean retention in the training mixture;
- multi-instrument balance;
- piano dominance;
- duplicate/composition repetition;
- long-form versus loop/fragment proportions;
- validation loss by source class;
- long-form generation quality;
- instrumentation and density distributions.

Treat mixture weights as empirical hyperparameters, not rights classifications.

## Phase 5 — research-NC Base training

The research Base must descend from the frozen commercial-clean checkpoint and carry:

```text
commercial_eligible=false
license_policy=research-nc
```

Its manifest must pin the parent checkpoint SHA-256 and the exact research corpus manifest SHA-256.

Do not copy, merge, distill, backport, or otherwise transfer research-NC weights into a commercial-eligible lineage.

## Distribution rule

Dataset permission to train does not automatically establish permission to distribute a trained checkpoint commercially or without conditions.

`distribution_scope` must therefore be set independently of `commercial_eligible`:

- `commercial`: checkpoint distribution is compatible with the commercial lineage policy;
- `noncommercial`: distribution, if performed, is explicitly non-commercial;
- `internal-only`: no public checkpoint distribution until the relevant rights/terms are resolved.

For an NC-trained checkpoint, default conservatively to `noncommercial` or `internal-only` according to the source-specific evidence. Do not infer commercial checkpoint rights from third-party model releases trained on similar data.

## Stop lines

This planning change does not authorize data ingestion or training by itself.

Until the relevant source-specific task authorizes them:

```text
GIGAMIDI_FULL_INGEST_STARTED=NO
ARIA_FULL_INGEST_STARTED=NO
RESEARCH_NC_TRAINING_STARTED=NO
COMMERCIAL_LINEAGE_CONTAMINATED=NO
```

## Next executable task

After the commercial v5 integration PR is closed, the next research task is:

```text
GIGAMIDI_EXACT_REVISION_AND_10K_50K_ORBITUNE_CENSUS
```

Output must include the measured active-event retention rate and a projected post-dedup contribution to the 1B target before any full GigaMIDI build is authorized.
