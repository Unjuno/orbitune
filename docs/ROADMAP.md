# Orbitune Roadmap

## Milestone 0: repository foundation

- README and project scope
- Apache-2.0 license
- adapter contribution policy
- adapter manifest schema
- empty adapter registry
- GitHub Actions test workflow
- GitHub Pages UI shell

## Milestone 1: local SDK scaffold

- Theory-REMI v0 tokenizer
- MIDI roundtrip tests
- tiny decoder-only GPT implementation
- LoRA wrapper
- adapter manifest validation CLI
- adapter packaging CLI

## Milestone 2: 3M base training

- train `orbitune-tiny-v0` on license-confirmed MIDI data
- publish base model as a release asset
- keep base model weights out of Git history

## Milestone 3: first adapters

- `chill-piano-v0`
- `dark-minimal-v0`
- `battle-loop-v0`

## Milestone 4: browser inference

- export 3M model to ONNX or another browser-compatible format
- test WASM inference first
- treat WebGPU as optional acceleration
- benchmark 4 and 8 bar generation on phones

## Milestone 5: community adapter flow

- accept small compatible adapters directly in the repo
- validate manifests in CI
- display bundled adapters in the web UI

## Later research

- quasi-periodic sliding-window generation
- reranking / repetition control
- texture-control events
- procedural ambience control
