# Orbitune roadmap

This roadmap tracks the **current project direction**. Historical proxy/audit documents remain in the repository, but their old `NEXT`/`OPEN` lists are not the current schedule.

## Milestone A — repository and compatibility foundation — DONE

- Apache-2.0 source repository
- test and publication-safety CI
- immutable checkpoint-hash identity rules
- legacy Base/Adapter manifests and registries
- GitHub Pages deployment pipeline
- source/model/publication boundaries documented
- deployment workflow split into unprivileged build/test and main-only deploy

## Milestone B — Compound representation, model and data path — DONE FOR V1

Implemented and used by the documented research model:

- one 12-field Compound record per musical event step
- deterministic MIDI canonicalization
- NOTE / CC / PROGRAM / BANK / TEMPO / PEDAL / PITCH_BEND / CHANNEL_PRESSURE / POLY_PRESSURE / TIME_SIGNATURE scope
- factorized timing and continuous controls
- hierarchical local/medium/global Transformer context
- routed fast/medium/slow recurrent memory
- intra-event Transformer decoder with discrete + continuous heads
- checkpoint/resume with optimizer and RNG state
- large indexed Aria + GigaMIDI research corpus path

The tokenizer identifier remains `orbitune-compound-v0-experimental` because it is part of the trained checkpoint ABI; changing the identifier would not retroactively change the model.

## Milestone C — frozen Research-NC Aria+GigaMIDI V1 — DONE

Documented frozen checkpoint:

```text
model id       research-nc-aria-gigamidi-v1
architecture   orbitune-compound-hierarchical-gpt-v1
step           100,000
parameters     8,857,250
events_seen    409,600,000
```

The indexed train corpus contains 4,069,137,373 active next-event pairs. That is corpus capacity, not unique training exposure. The sampler draws with replacement.

The checkpoint remains documentation-only in the public repository while redistribution review is incomplete.

## Milestone D — native Compound Web ABI and Pages runtime — DONE FOR SOURCE

Completed:

- rejected the incorrect stateless/teacher-forced V1 Web export as a production ABI
- defined native V2 `stream` + `decoder_prefix` graph contract
- native stream-state parity validation
- sequential intra-event decoder parity validation
- exact greedy native-vs-Web rollout validation in local export work
- `onnxruntime-web`/WASM smoke in local export work
- browser-side Python-compatible rounding, masks and quantization
- Compound → Standard MIDI conversion
- Play / Stop WebAudio preview
- SHA-bound fail-closed model variant loader
- pre-merged LoRA variant **concept**
- GitHub Pages source deployment

The source runtime is public. The actual Compound ONNX model files are not public yet. The Base V2 parity evidence does not validate a future LoRA-merged artifact; any merged variant must repeat parity on its own exact bytes.

## Milestone E — long-run full-parameter Base continuation — ACTIVE LOCAL RUN

A normal full-parameter continuation from the immutable 100k parent is now reported active locally under run id `research_nc_aria_gigamidi_v2_1m`. The continuation checkpoint is mutable training state and is **not** a public model release or Adapter compatibility target.

Reported local-run evidence snapshot, 2026-09-08:

```text
parent model id              research-nc-aria-gigamidi-v1
parent step                  100,000
parent SHA-256               8bf20a1198c4f5ee086ca13fd89521af6cfaa1fb28dd6602b1eb8f0b104629dc
parent unchanged             yes
continuation run id          research_nc_aria_gigamidi_v2_1m
reported continuation step   102,000
reported events_seen         417,792,000
batch × sequence             16 × 256
precision                    bf16
learning rate                3e-4
weight decay                 0.01
optimizer                    fused AdamW
sampler RNG state            present in new continuation checkpoints
validation window hash       db2b3903dbdae6e9c3821494009b5adbd1b0ba19499adcca84b790c66a5c9814
non-finite loss / grad       0 / 0
```

These values are **reported local run evidence**. The continuation checkpoint itself is not distributed by this repository, so the snapshot is not a public-byte verification claim. The immutable V1 publication record remains unchanged.

The clean round target remains 1,000,000 global steps under the active 16 × 256 batch geometry:

```text
4,096 event positions / step
× 1,000,000 steps
= 4,096,000,000 cumulative sampled event positions
```

This is approximately one corpus-equivalent amount of sampled exposure relative to the 4.069B active-pair corpus, **not** a claim that every corpus pair is visited exactly once. Replacement sampling permits repeats and unvisited records.

This milestone is not complete until an actual final checkpoint is saved, loaded, hashed, evaluated and documented. A later checkpoint must receive a new model identity; it must not overwrite the 100k historical V1 record.

## Milestone F — final Base selection and evaluation — AFTER A LONG-RUN CANDIDATE EXISTS

For a later Base candidate:

- verify exact checkpoint identity and loadability
- compare held-out loss against earlier milestones
- run fixed generated-MIDI regression batches
- separate parse validity from musical-quality claims
- retain useful intermediate milestones for comparison
- decide which exact checkpoint becomes the next immutable Adapter target
- repeat native Web export/parity validation for the selected checkpoint

Do not silently repoint an existing Base id or Adapter dependency to newer bytes.

## Milestone G — Compound LoRA / Adapter ABI — AFTER TARGET BASE FREEZE

Policy is defined in [COMPOUND_LORA_POLICY.md](COMPOUND_LORA_POLICY.md).

Required work:

- keep Base pretraining full-parameter; LoRA is post-Base adaptation
- enumerate candidate Compound target modules on the exact frozen Base
- measure rank/target tradeoffs rather than copying legacy rank-4 `q_proj`/`v_proj`
- freeze a new Compound Adapter ABI identifier
- define strict Safetensors tensor/metadata layout
- bind every Adapter to exact Base id + checkpoint SHA-256
- train with Base parameters frozen
- establish held-out and generated-MIDI Adapter evaluation
- add concrete Compound contribution schema/CLI only after the ABI is frozen

Community Compound Adapter binaries remain gated until then.

## Milestone H — Compound LoRA Web deployment — AFTER ADAPTER VALIDATION

Planned initial deployment strategy:

```text
frozen Base + validated Adapter
→ local merge
→ export matched V2 stream + decoder graphs
→ repeat native/Web parity on the exact merged bytes
→ only then publish as lora-premerged variant
```

No LoRA-specific merged Compound Web artifact has been validated or published yet. Dynamic browser LoRA is optional later work and requires its own validated packing/numerical/runtime contract.

## Milestone I — model / ONNX publication — RIGHTS-GATED

Before exposing a downloadable model or browser variant:

- complete redistribution review for every restricted source lineage
- verify exact Base and exported artifact hashes/sizes
- provide real versioned artifact URLs with browser-compatible CORS
- attach model card/provenance/evaluation evidence
- keep research-NC restrictions intact
- run clean-environment generation and native/Web parity against the exact release bytes

A model host such as Hugging Face may be used after these gates; repository source publication alone is not weight-publication permission.

## Milestone J — post-training necessity gate — OPTIONAL

Policy learning is not mandatory by default.

```text
Base pretraining
→ held-out rollout evaluation
→ high-quality SFT comparison if needed
→ DPO only if a reliable preference/selection gap remains
→ reward-based RL only after reward validity + anti-collapse tests
```

The branch is allowed to terminate at `NOT REQUIRED`.

See `docs/POST_TRAINING_RESEARCH.md` for the research rationale.

## Ongoing engineering issues

Tracked separately from Base training:

- Standard MIDI 3-byte TEMPO behavior for generated 1–3 BPM values (issue #48)
- import the exact local native Compound Web golden fixture into source CI without reconstructing omitted values (issue #49)
- safe CFE telemetry/synchronization fixes from issue #52 are merged; deeper recurrent-memory, sampler/H2D and boundary-overhead profiling remains separate from the active production lineage
- corpus accounting/provenance limitations already recorded in the V1 model documentation
- future public artifact/Adapter release review

## Current critical-path summary

```text
DONE     repository/publication safety foundation
DONE     Compound hierarchical Base implementation
DONE     Aria+GigaMIDI V1 frozen at step 100k
DONE     native Compound V2 Web runtime source + Pages deployment
ACTIVE   local long-run full-parameter Base continuation toward step 1M
THEN     evaluate/freeze next immutable Base candidate if produced
THEN     freeze and validate Compound LoRA ABI
THEN     optional Adapter/pre-merged variants after their own parity validation
GATE     redistribution review before public model/ONNX release
OPTIONAL SFT/DPO/RL only if Base evaluation demonstrates a need
```
