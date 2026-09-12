# Publication status and boundaries

## Scope

Orbitune publishes source code, the immutable A2-512 research checkpoint, and a reviewed browser serialization of that same Base. Source and model licenses remain separate.

- Source code: Apache-2.0.
- Historical 100k checkpoint: documentation-only.
- A2-512 checkpoint: published on Hugging Face under CC-BY-NC-SA-4.0, research/non-commercial scope.
- A2 native-stream V2 ONNX graph pair: distributed through the GitHub Pages artifact under the same research/non-commercial lineage.
- Training data / memmap indexes: not redistributed.
- Production Compound LoRA/Adapter artifact: not published yet.

The current application is deployed at `https://unjuno.github.io/orbitune/`.

## Public artifact contract

| Artifact | Status |
| --- | --- |
| Python/runtime source | Public in Git under the source-code license |
| A2-512 PyTorch checkpoint | Published on Hugging Face; exact SHA and pinned Hub revision recorded in Git |
| Historical 100k checkpoint | Documentation-only |
| Compound Web/PWA source | Public; native-stream V2 runtime + install/offline/continuous-stream support |
| A2 `stream.onnx` | Published inside the generated GitHub Pages artifact; not tracked in Git |
| A2 `decoder_prefix.onnx` | Published inside the generated GitHub Pages artifact; not tracked in Git |
| A2 Web release record | `models/research_nc_aria_gigamidi_a2_512_v1/web_release.json` |
| Public Pages runtime config | Generated at deploy time only after exact artifact verification |
| Checked-in runtime config | Intentionally fail-closed: A2-bound, `variants: []`, review pending |
| Compound LoRA policy / experimental SFT primitive | Public source; **not** a frozen production Compound Adapter ABI |
| Production Compound LoRA binaries | Not published yet |
| Training data | Not redistributed |

## A2 identities

Canonical Base release:

```text
model id        orbitune-a2-512-research-nc
checkpoint SHA  e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0
Hub revision    45579e28a32d5847f3121142aca9382b4e4aaedc
source commit   8489870f81a1591515a98e58554e533fcac9d095
```

Reviewed Web serialization:

```text
stream.onnx
  bytes   27,241,782
  SHA256  27be3d6a4db726f52d7fc7e8df2e7a24be04ab5bd76da243bd8dd92712f2e107

decoder_prefix.onnx
  bytes   8,620,888
  SHA256  abb8326680222aff32d6b2fcb45356c748c523216cb0d75d6a755e615f419225
```

The graph pair is a deterministic derived serialization of the same frozen A2 weights. It does not broaden the Base license or commercial eligibility.

## Why the checked-in runtime config stays unpublished

`web/compound-runtime-config.json` in Git intentionally remains fail-closed. Pull requests and source checkouts therefore do not claim that an artifact is published merely because source metadata was edited.

The main Pages build performs the publication transition in the generated artifact only after all of the following pass:

1. download the checkpoint from the pinned Hub revision;
2. verify the exact Base checkpoint SHA-256;
3. reproduce the V2 `stream` + `decoder_prefix` graphs;
4. run native tensor-wrapper parity;
5. run Python ONNX Runtime parity;
6. require exact graph bytes/sizes to match the reviewed Web release record;
7. generate the fixed native greedy reference;
8. require exact `onnxruntime-web`/WASM record equality;
9. generate a `runtime_model_published` config with exactly the reviewed A2 Base variant;
10. deploy the Pages artifact.

If any gate fails, deployment is skipped.

## A2 Web validation evidence

Two independent exports on GitHub-hosted runners produced identical graph SHA-256 values. The deterministic seed + 48-new-event reference contains 49 records with record-list SHA-256 `668972140fb8acaea2955453a882be7512209774b2daa2074649f1c8104b22a1`. The WASM rollout matched the native record list exactly.

Bounded CI throughput evidence using Node 22.23.2, `onnxruntime-web` 1.29.0, WASM and one thread observed 19.317–22.978 generated events/s and 605–844 ms graph load time. These are CI-runner measurements, not device-fleet guarantees.

## Base pretraining and Adapter publication

For Compound, Base and Adapter stages remain separate:

```text
full-parameter Base training
→ immutable A2 Base id + SHA-256
→ Base Web release (now published)
→ measure Compound LoRA target/rank choices
→ freeze a versioned Compound Adapter ABI
→ frozen-Base LoRA training
→ Adapter / merged-variant evaluation
→ Adapter rights review
→ exact merged V2 export/parity
→ optional LoRA Web variant publication
```

The legacy Theory-REMI `orbitune-lora-v0` ABI is not a shortcut around this gate.

An Adapter cannot broaden the permissions of its Base. A research-NC A2 Base yields research-NC/noncommercial Adapter and merged descendants at most.

## Before a Compound LoRA / Adapter release

1. Bind to the immutable A2 model id and exact checkpoint SHA-256.
2. Freeze a new Compound Adapter ABI only after measured target-module/rank/scaling experiments.
3. Freeze tensor names/shapes, rank/alpha/scaling and Safetensors metadata layout.
4. Train with Base weights frozen and preserve reproducibility/provenance.
5. Reject wrong-Base, wrong-ABI, missing and duplicate tensors.
6. Record held-out and generated-MIDI Base-versus-Adapter evaluation.
7. Review Adapter training-data rights independently.
8. For Web publication, pre-merge against the exact Base, export the matched V2 graph pair, and rerun exact native/Python ORT/WASM parity on those exact bytes.
9. Publish immutable hashes/URLs and an explicit Adapter identity.

Until those conditions are met, Compound LoRA remains an experimental/local adaptation path. The deployed selector is LoRA-ready but currently exposes the reviewed A2 Base only.

## Historical and source-hygiene boundaries

The historical 100k release records remain unchanged. Missing historical evidence is not renamed to PASS. `check_publication.py` validates tracked publication/source contracts but is not a comprehensive secret scanner, malware analysis, legal review, or Git-history audit.

The Pages release build is an artifact gate in addition to source CI; passing source tests alone is not sufficient to publish different model bytes.

## Local checks

```bash
python -m pip install -e ".[dev]"
python scripts/check_publication.py
python -m pytest -q
node --test web/*.test.mjs
```

Model-dependent A2 export/WASM validation is exercised by the dedicated `compound-web-export` workflow and repeated by the main Pages deployment workflow.
