# Orbitune Architecture

Orbitune v0 is a MIDI-only generation system with a fixed small base model and small LoRA adapters.

## Core idea

```text
Base Model = shared MIDI grammar
LoRA Adapter = style / mood / genre / texture-like generation tendency
```

The system distributes generation capability rather than finished audio tracks.

## v0 pipeline

```text
MIDI corpus
  -> Theory-REMI v0 tokenizer
  -> token dataset
  -> orbitune-tiny-v0 base training
  -> LoRA adapter training
  -> Base + Adapter MIDI generation
  -> MIDI playback / export
```

## Runtime target

The default v0 runtime target is smartphone-friendly symbolic generation:

- 3M-parameter target base model
- context length: 512 tokens
- output: MIDI event tokens
- one adapter at a time
- generate-then-play UX for 4 to 8 bars

## Event format

Theory-REMI v0 uses only music events:

- `BAR`
- `POSITION_0..15`
- `NOTE_PITCH_21..108`
- `NOTE_DURATION_1..64`
- `VELOCITY_1..32`

Future versions may add control and texture-control events, but v0 does not generate audio waveforms.

## Repository policy

- Source code and small adapters may be committed.
- Base model weights are not committed under `models/`.
- Community adapters must validate and include metadata.
