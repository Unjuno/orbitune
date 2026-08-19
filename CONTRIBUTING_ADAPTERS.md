# Contributing Orbitune Adapters

Orbitune accepts small LoRA adapters for the fixed v0 base target: `orbitune-tiny-v0`.

Adapters may be committed directly to this repository when they are small, compatible, documented, and validated.

## Required directory layout

```text
adapters/community/<adapter-id>/
  manifest.json
  adapter.safetensors
  demo.mid
  README.md
```

Official adapters use the same layout under `adapters/official/`.

## Compatibility requirements

Every adapter must declare:

- `base_model`: `orbitune-tiny-v0`
- `architecture`: `orbitune-midi-gpt-v0`
- `parameter_scale`: `3m`
- `tokenizer`: `theory-remi-v0`
- `adapter_type`: `lora`

Adapters for other base models are out of scope for v0.

## Size policy

Recommended limit:

```text
one adapter directory <= 1 MB
```

Hard review threshold:

```text
5 MB or larger requires maintainer approval
```

The base model checkpoint must not be committed to this repository.

## Required metadata

Each `manifest.json` must include:

- adapter identity and version
- base model and tokenizer compatibility
- LoRA rank and target modules
- generation defaults
- tags
- license
- training data declaration

## Training data declaration

Adapters must disclose the training-data status. This does not need to expose private file names, but it must state whether the contributor has rights to use and publish the resulting adapter.

Minimum example:

```json
{
  "training_data": {
    "source_type": "user_provided_midi",
    "license": "original",
    "num_files": 32,
    "num_tokens": 120000,
    "rights_confirmed": true
  }
}
```

Adapters with unclear rights may be rejected.

## Quality requirements

Each adapter must include at least one demo MIDI file. The generated MIDI should be playable, non-empty, and appropriate for background-music use.

Minimum checks:

- demo MIDI opens in a standard MIDI player or DAW
- no long accidental silence unless intended
- no extreme note density spikes
- manifest validates against the schema
- adapter is listed in `registry/adapters.json`

## Pull request checklist

- [ ] Adapter directory added under `adapters/community/<adapter-id>/`
- [ ] `manifest.json` validates
- [ ] `adapter.safetensors` is present
- [ ] `demo.mid` is present
- [ ] `README.md` describes the adapter
- [ ] `registry/adapters.json` includes the adapter
- [ ] License and training-data declaration are included
