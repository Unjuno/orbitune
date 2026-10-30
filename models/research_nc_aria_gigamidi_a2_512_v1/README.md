# Orbitune A2-512 Research-NC

This is the immutable release record for the completed A2-512 continuation. The checkpoint is hosted on [Hugging Face](https://huggingface.co/Unjuno/orbitune-a2-512); model weights are intentionally not tracked in Git.

| Field | Value |
| --- | --- |
| Model ID | `orbitune-a2-512-research-nc` |
| Checkpoint | `model.pt` |
| Parameters | 8,857,250 |
| Final global step | 586,076 |
| Cumulative events seen | 14,268,978,580 |
| SHA-256 | `25d43bd663501b5daf1b9708efe31706fd86a344f036ebe7be7a66df683eedda` |
| Bytes | 117,969,708 |
| Strict load | PASS |
| License | CC-BY-NC-SA-4.0 |

This release continues the previous public checkpoint (step 220,813) for 365,263 additional steps — two full passes plus 50.0% of the third pass of the 4,069,137,373-event research-NC epoch — and was stopped deliberately at that point.

Full validation over every validation window of both sources (42,323 windows, 21,669,376 events, `a2-fullval-seq512-v1`) gives `-1.553983`, against `-1.430447` for the previously published checkpoint measured on the same windows. The lighter in-training screening series (`a2-seq512-v1`, 98,304 events) reported `-1.576682` at step 585,754. The release contains the full training checkpoint, including optimizer and RNG state, and is not a Transformers-format model or a browser-ready ONNX export.

Download and verify:

```bash
hf download Unjuno/orbitune-a2-512 model.pt --local-dir orbitune-a2-512
python -c "import hashlib, pathlib; p=pathlib.Path('orbitune-a2-512/model.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

The exact training-source commit is [`8489870f81a1591515a98e58554e533fcac9d095`](https://github.com/Unjuno/orbitune/tree/8489870f81a1591515a98e58554e533fcac9d095), preserved on `provenance/a2-512-training-source`. The only local source deviation is a float32 rewrite of the Gaussian NLL terms and the sampling softmax; parameters, modules, and configuration are unchanged. Training sources include Aria-MIDI and GigaMIDI, so the checkpoint is restricted to research and non-commercial use. The corpus itself is not distributed.

Machine-readable metadata is in [manifest.json](manifest.json). The public Hub revision is [`8a362fb907f96176929e6df941885bc87fbabf3d`](https://huggingface.co/Unjuno/orbitune-a2-512/commit/8a362fb907f96176929e6df941885bc87fbabf3d). The browser graph contract is in [web_release.json](web_release.json).
