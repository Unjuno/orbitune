# Orbitune A2-512 Research-NC

This is the immutable release record for the completed A2-512 continuation. The checkpoint is hosted on [Hugging Face](https://huggingface.co/Unjuno/orbitune-a2-512); model weights are intentionally not tracked in Git.

| Field | Value |
| --- | --- |
| Model ID | `orbitune-a2-512-research-nc` |
| Checkpoint | `model.pt` |
| Parameters | 8,857,250 |
| Final global step | 220,813 |
| Cumulative events seen | 4,096,016,384 |
| SHA-256 | `e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0` |
| Bytes | 106,635,500 |
| Strict load | PASS |
| License | CC-BY-NC-SA-4.0 |

The final fixed `a2-seq512-v1` validation loss was `-1.5476370453834534` over 98,304 events. The release contains the full training checkpoint, including optimizer and RNG state, and is not a Transformers-format model or a browser-ready ONNX export.

Download and verify:

```bash
hf download Unjuno/orbitune-a2-512 model.pt --local-dir orbitune-a2-512
python -c "import hashlib, pathlib; p=pathlib.Path('orbitune-a2-512/model.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

The recorded training-source identity is `8489870f81a1591515a98e58554e533fcac9d095`; it is provenance metadata and is not currently a reachable commit in the public repository. Training sources include Aria-MIDI and GigaMIDI, so the checkpoint is restricted to research and non-commercial use. The corpus itself is not distributed.

Machine-readable metadata is in [manifest.json](manifest.json). The public Hub revision is [`45579e28a32d5847f3121142aca9382b4e4aaedc`](https://huggingface.co/Unjuno/orbitune-a2-512/commit/45579e28a32d5847f3121142aca9382b4e4aaedc).
