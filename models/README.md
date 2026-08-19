# Base model weights

Base model weights are intentionally not committed to this repository.

The public Base identity is `orbitune-base`. It is an immutable checkpoint: once published, the checkpoint bytes and SHA-256 compatibility key must never be replaced under that identity.

```text
Base:             orbitune-base
Parameters:       2,945,760
Architecture ABI: orbitune-midi-gpt-v0
Tokenizer ABI:    theory-remi-v0
Compatibility:    exact checkpoint SHA-256
```

After release, download and verify it with:

```bash
python scripts/download_base_model.py --out models
```

Adapters under `adapters/` may be committed only when their manifest and Safetensors metadata both target the exact published Base SHA-256. A future different Base must use a separate compatibility lineage rather than replacing this one.
