# Base model weights

Base model weights are intentionally not committed to this repository.

The v0 base target is:

```text
orbitune-tiny-v0
architecture: orbitune-midi-gpt-v0
parameter scale: 3m
tokenizer: theory-remi-v0
```

When release assets are available, download them with:

```bash
python scripts/download_base_model.py --model orbitune-tiny-v0
```

Adapter weights under `adapters/` may be committed when they satisfy the adapter contribution policy. Base checkpoints under `models/` must remain local or release-managed.
