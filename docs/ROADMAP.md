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

## Milestone E — A2-512 full-parameter Base continuation — DONE

The long-run continuation from the immutable historical 100k checkpoint is complete. The selected result is frozen as `orbitune-a2-512-research-nc`, externally published, and recorded by the canonical [A2-512 release manifest](../models/research_nc_aria_gigamidi_a2_512_v1/manifest.json).

```text
model id                     orbitune-a2-512-research-nc
final step                   220,813
final events_seen            4,096,016,384
checkpoint SHA-256           e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0
checkpoint frozen            yes
strict load                  PASS
external publication         Hugging Face: Unjuno/orbitune-a2-512
training source commit       8489870f81a1591515a98e58554e533fcac9d095
source provenance            reachable public commit
```

The final event count is cumulative sampled exposure, not exact unique corpus coverage. The historical `research-nc-aria-gigamidi-v1` record remains unchanged.

## Milestone F — final Base selection and evaluation — DONE FOR A2-512

Completed for the A2-512 release:

- verify exact checkpoint identity and loadability
- compare held-out loss against earlier milestones
- run fixed generated-MIDI regression batches
- separate parse validity from musical-quality claims
- retain useful intermediate milestones for comparison
- selected and published an exact immutable checkpoint
- retained Web ONNX export/parity as separate follow-up work

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

## Milestone I — model / ONNX publication — PYTORCH BASE DONE; ONNX GATED

The PyTorch A2-512 Base is published with its research-NC restriction, exact hash, source provenance and evaluation metadata. Before exposing a browser variant:

- complete redistribution review for every restricted source lineage
- verify exact Base and exported artifact hashes/sizes
- provide real versioned artifact URLs with browser-compatible CORS
- attach model card/provenance/evaluation evidence
- keep research-NC restrictions intact
- run clean-environment generation and native/Web parity against the exact release bytes

The A2-512 Hugging Face release does not imply that an ONNX browser variant or training corpus has been released.

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

- issue #48 is resolved in source by preserving the checkpoint ABI domain `1..999 BPM` while explicitly rejecting unrepresentable `1..3 BPM` at both Python and Web Standard MIDI serialization boundaries
- issue #49 remains a non-blocking Web-release follow-up: import the exact native Compound Web golden fixture into CI
- issue #52 remains future performance research; it is not an A2-512 Base release blocker
- PR #55 is a separate TBPTT training experiment and must not be mixed into this frozen release baseline
- corpus accounting/provenance limitations already recorded in the V1 model documentation
- future public artifact/Adapter release review

## Current critical-path summary

```text
DONE     repository/publication safety foundation
DONE     Compound hierarchical Base implementation
DONE     Aria+GigaMIDI V1 frozen at step 100k
DONE     native Compound V2 Web runtime source + Pages deployment
DONE     A2-512 training, evaluation, immutable freeze and external publication
DONE     exact training-source commit preserved on a public provenance branch
THEN     freeze and validate Compound LoRA ABI
THEN     optional Adapter/pre-merged variants after their own parity validation
GATE     exact ONNX review/parity before browser model release
OPTIONAL SFT/DPO/RL only if Base evaluation demonstrates a need
```
