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

The tokenizer identifier remains `orbitune-compound-v0-experimental` because it is part of the trained checkpoint ABI.

## Milestone C — historical Research-NC Aria+GigaMIDI V1 — DONE

```text
model id       research-nc-aria-gigamidi-v1
step           100,000
parameters     8,857,250
events_seen    409,600,000
```

The indexed train corpus contains 4,069,137,373 active next-event pairs. That is corpus capacity, not unique training exposure. The historical 100k checkpoint remains a separate documentation-only identity.

## Milestone D — native Compound Web ABI, PWA and A2 browser Base — DONE

Completed:

- rejected the incorrect stateless/teacher-forced Web export
- defined native V2 `stream` + `decoder_prefix` contract
- fixed-capacity tensorized stream-state export
- native stream-state parity regression
- sequential eight-stage decoder-prefix parity
- deterministic Python ONNX Runtime parity
- exact native-vs-`onnxruntime-web`/WASM greedy record parity
- reproducible A2 graph bytes across two independent exports
- browser-side Python-compatible rounding, masks and quantization
- Compound → Standard MIDI conversion
- bounded generate-while-listening WebAudio stream
- finite MIDI preview/export
- installable PWA shell
- explicit SHA-verified offline graph caching
- Base/`lora-premerged` variant selector contract
- GitHub Pages publication of the immutable A2 Base

Public application:

```text
https://unjuno.github.io/orbitune/
```

Reviewed A2 graph identities:

```text
stream.onnx
  SHA256  27be3d6a4db726f52d7fc7e8df2e7a24be04ab5bd76da243bd8dd92712f2e107
  bytes   27,241,782

decoder_prefix.onnx
  SHA256  abb8326680222aff32d6b2fcb45356c748c523216cb0d75d6a755e615f419225
  bytes   8,620,888
```

Large ONNX binaries remain outside Git. The main Pages build reproduces them from the pinned checkpoint and deploys only if exact hashes and parity pass.

## Milestone E — A2-512 full-parameter Base continuation — DONE

The continuation from the immutable historical 100k checkpoint is complete. The selected result is frozen as `orbitune-a2-512-research-nc` and published.

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
browser Base                 published through GitHub Pages
```

The final event count is cumulative sampled exposure, not exact unique corpus coverage. The historical V1 identity remains unchanged.

## Milestone F — final Base selection and evaluation — DONE FOR A2-512

Completed:

- exact checkpoint identity/loadability
- held-out validation protocol
- fixed generated-MIDI regressions
- immutable A2 model identity
- source provenance recovery
- native V2 Web export
- exact graph identity freeze
- Python ORT parity
- exact WASM greedy parity
- public PWA Base release

A later model must receive a new identity rather than silently repointing A2.

## Milestone G — Compound LoRA target/rank measurement and Adapter ABI — NEXT

Policy is defined in [COMPOUND_LORA_POLICY.md](COMPOUND_LORA_POLICY.md).

Required work:

- use the immutable A2 checkpoint as the comparison Base
- enumerate candidate target modules
- measure target-module × rank × scaling tradeoffs rather than copying legacy rank-4 `q_proj`/`v_proj`
- compare Base versus SFT-LoRA under held-out and generated-MIDI evaluation
- freeze a new Compound Adapter ABI only after measured selection
- define strict Safetensors tensor/metadata layout
- bind every Adapter to exact A2 Base id + checkpoint SHA-256
- keep Base parameters frozen during Adapter training
- define concrete contribution schema/CLI after ABI freeze

Community production Compound Adapter binaries remain gated until then.

## Milestone H — Compound LoRA Web deployment — AFTER ADAPTER VALIDATION

Initial deployment strategy:

```text
immutable A2 Base + validated Adapter
→ merge locally
→ export matched V2 stream + decoder graphs
→ freeze exact merged graph hashes
→ rerun native/Python ORT/WASM parity on those bytes
→ publish as kind = "lora-premerged" with adapter_id
```

The PWA already supports the variant shape, but no production Compound LoRA Web artifact is published yet. Dynamic browser-side LoRA remains optional later work.

## Milestone I — Base model / ONNX publication — DONE FOR A2

The A2 PyTorch checkpoint and its reviewed V2 browser graph pair are published under the same research-NC/noncommercial lineage. Training data is not distributed.

The deployment gate is reproducible and fail-closed: any checkpoint SHA, graph SHA/size, native parity, Python ORT parity, WebAssembly parity, rights-state or runtime-config mismatch prevents Pages deployment.

## Milestone J — human-preference post-training necessity gate — NEXT AFTER SFT BASELINE

Policy learning is not mandatory by default.

```text
immutable A2 Base
→ measured LoRA/SFT baseline
→ human listening/preference dataset
→ verify a real preference/selection gap
→ DPO only if pairwise preference learning is justified
→ reward-based RL only after reward validity + anti-collapse tests
```

The branch may terminate at `NOT REQUIRED` if SFT/selection is sufficient.

## Ongoing engineering issues

- issue #48: resolved; checkpoint TEMPO ABI remains `1..999 BPM`, Python/Web Standard MIDI explicitly reject unrepresentable `1..3 BPM`
- issue #49: historical golden-fixture follow-up is superseded in practical release coverage by the reproducible A2 native greedy fixture generated during the current Web export workflow; the issue itself may still be triaged/closed separately
- issue #52: future performance research, not a release blocker
- PR #55: separate TBPTT Base-training experiment; do not mix into the frozen A2/post-training baseline
- branch protection / always-running required-check aggregation remains separate repository hardening
- real mobile/tablet/browser performance measurement remains required before making device-wide real-time guarantees

## Current critical path

```text
DONE     repository/publication safety foundation
DONE     Compound hierarchical Base implementation
DONE     historical V1 frozen at step 100k
DONE     A2-512 training/evaluation/freeze/publication
DONE     exact A2 source provenance
DONE     native V2 ONNX export + deterministic graph identities
DONE     exact native/Python ORT/WASM Base parity
DONE     public installable A2 Compound PWA + bounded continuous stream
NEXT     Compound LoRA target/rank sweep and Base-vs-SFT evaluation
THEN     freeze Compound Adapter ABI only from measured evidence
THEN     human-preference dataset + DPO/RL necessity gate
THEN     publish validated pre-merged LoRA variants through the same Web gate
```
