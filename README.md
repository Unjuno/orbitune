# Orbitune

Orbitune is a lightweight MIDI generation framework for local BGM creation.

It uses a shared small MIDI base model and small LoRA adapters to generate style-specific background music. The project focuses on symbolic MIDI generation, adapter training, adapter validation, and a simple web UI for selecting adapters and generating MIDI.

Orbitune v0 is MIDI-only. It does not generate raw audio waveforms, audio-codec tokens, vocals, mixes, or environmental sound directly.

## Project goals

- Provide a fixed 3M-parameter MIDI base model target: `orbitune-tiny-v0`.
- Support small LoRA adapters as style, genre, mood, or texture-like BGM packs.
- Let contributors train, validate, and commit compatible adapters.
- Keep the base model weights out of the repository.
- Provide a smartphone-oriented web UI with adapter selection, BPM, length, and temperature.

## v0 scope

- Symbolic MIDI generation only
- Theory-REMI v0 token format
- Piano/BGM-oriented generation
- Base + one LoRA adapter at a time
- 4 to 8 bar generation as the default UX
- GitHub-first distribution

## Non-goals

- Raw audio generation
- Audio codec token generation
- Vocal generation
- Full DAW-quality mixing
- Multi-instrument orchestration as the v0 target
- Multiple LoRA composition in v0
- WebGPU-only runtime assumptions

## Repository policy

Base model weights are not committed to this repository. Use `scripts/download_base_model.py` once a release asset is available.

Small compatible adapters may be committed directly under `adapters/official/` or `adapters/community/`, provided that they include a manifest, README, demo MIDI, license declaration, and pass validation.

## Planned command surface

```bash
orbitune inspect
orbitune tokenize
orbitune detokenize
orbitune train-base
orbitune train-adapter
orbitune generate
orbitune validate-adapter
orbitune package-adapter
```

## License

The Orbitune source code is licensed under Apache-2.0. Community adapters may use their own compatible licenses, but every adapter must declare its license and training-data status.
