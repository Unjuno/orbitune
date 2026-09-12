# Orbitune

Local-first symbolic MIDI generation with a hierarchical Compound Transformer and a separate legacy Theory-REMI / LoRA runtime.

## Start here

| Goal | Entry point |
| --- | --- |
| Use the public A2-512 Web/PWA app | [Orbitune Infinite MIDI](https://unjuno.github.io/orbitune/) |
| Install and inspect the code | Quick start below |
| Download the completed A2-512 research model | [A2-512 model card](models/research_nc_aria_gigamidi_a2_512_v1/README.md) |
| Inspect the reviewed A2 Web release | [A2 Web release record](models/research_nc_aria_gigamidi_a2_512_v1/web_release.json) |
| Understand the historical 100k model | [Historical model card](models/research_nc_aria_gigamidi_v1/README.md) |
| Understand the Compound architecture | [Compound Base](docs/COMPOUND_BASE.md) |
| Understand browser/PWA streaming | [Compound PWA](docs/COMPOUND_PWA.md) |
| Inspect the native browser ABI | [Compound Web runtime](docs/COMPOUND_WEB_RUNTIME.md) |
| Understand how Compound LoRA will be applied | [Compound LoRA policy](docs/COMPOUND_LORA_POLICY.md) |
| Understand release availability and limitations | [Publication status](docs/PUBLICATION.md) |
| Contribute code, Bases, or Adapters | [Contributing](CONTRIBUTING.md) |
| Browse architecture, roadmap and historical work | [Documentation index](docs/README.md) |

## What is available

The completed `orbitune-a2-512-research-nc` checkpoint is publicly available from [Hugging Face](https://huggingface.co/Unjuno/orbitune-a2-512). Its immutable SHA-256 is `e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0`. Weights remain outside Git, and a source clone alone is not a pretrained installation.

The same immutable A2-512 Base is now available through the public GitHub Pages PWA. The Pages build deterministically derives the reviewed native-stream V2 ONNX graph pair from the pinned checkpoint, verifies exact graph SHA-256 identities, reruns native/Python ONNX Runtime/WebAssembly parity, and only then publishes the browser variant. The large ONNX binaries are generated into the Pages artifact rather than tracked in Git.

The public app supports finite MIDI generation/export and continuous generate-while-listening mode. Infinite-stream mode carries bounded model state and bounded playback lookahead instead of retaining an ever-growing musical history. Stopping/resetting discards the compressed generation state, so earlier music is not reconstructable unless separately recorded.

The historical frozen research checkpoint `research-nc-aria-gigamidi-v1` remains at global step 100,000. Its recorded indexed train corpus contains 4,069,137,373 active next-event pairs. **Corpus capacity is not the number of unique events consumed by training.** Its checkpoint record reports cumulative `events_seen = 409,600,000`; replacement sampling means this is not an exact epoch/coverage claim.

A2-512 completed at global step 220,813 and 4,096,016,384 cumulative sampled event positions. It is frozen under a separate identity and does not overwrite the historical 100k checkpoint. Its [machine-readable release manifest](models/research_nc_aria_gigamidi_a2_512_v1/manifest.json) is the canonical Base record.

## Public Web/PWA release

```text
App: https://unjuno.github.io/orbitune/

stream.onnx
  bytes   27,241,782
  SHA256  27be3d6a4db726f52d7fc7e8df2e7a24be04ab5bd76da243bd8dd92712f2e107

decoder_prefix.onnx
  bytes   8,620,888
  SHA256  abb8326680222aff32d6b2fcb45356c748c523216cb0d75d6a755e615f419225
```

The reviewed 48-new-event greedy reference matches exactly under `onnxruntime-web`/WASM. Two independent exports produced the same graph hashes. Bounded GitHub-runner measurements observed 19.317–22.978 generated events/s under the documented single-threaded WASM test; this is not a mobile-device performance guarantee.

The PWA can explicitly download and SHA-verify the selected model graph pair for offline reuse. Browser storage may still be evicted by platform policy.

The deployed selector is ready for reviewed `lora-premerged` variants, but **no production Compound LoRA artifact is published yet**. Future LoRA variants must bind to the exact A2 Base and repeat the same exact-byte V2 export/parity gate.

## Quick start: source checkout

Python 3.10 or newer is required by the package. Existing Python CI uses Python 3.11; CUDA is not needed to inspect the code.

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install and inspect without downloading datasets or starting training:

```bash
python -m pip install -e ".[dev]"
orbitune-compound --help
orbitune-compound info --config configs/compound_hierarchical_9m.json
```

A checkpoint can be downloaded with `hf download Unjuno/orbitune-a2-512 model.pt --local-dir orbitune-a2-512`. Verify its SHA-256 against the A2 release record before loading it.

## Runtime and Adapter boundaries

**Compound Transformer:** factorized Compound event embeddings, local/medium/global attention, routed fast/medium/slow recurrent memory, an intra-event Transformer and mixed discrete/continuous heads. A serialized Compound record has 12 fields.

**Compound Web:** native stream-state V2 uses a matched `stream` graph plus `decoder_prefix` graph. Browser sampling/quantization mirrors Python semantics; the PWA provides bounded continuous generation, finite MIDI export, WebAudio preview, installability, and explicit SHA-verified offline model caching.

**Compound LoRA:** Base pretraining remains full-parameter training. LoRA is a post-Base adaptation stage against the immutable A2 checkpoint. The public Compound Adapter ABI is not frozen yet, so target modules/rank must not be copied from the legacy stack. The initial Web strategy is a separately validated pre-merged variant.

**Theory-REMI reference:** retained at `web/legacy.html` and through the legacy CLI/Base/Adapter registry. Its `orbitune-lora-v0` ABI is not interchangeable with Compound.

## Repository map

| Location | Purpose |
| --- | --- |
| `orbitune/` | Model, MIDI representation, sampler, runtime and Web-export code |
| `configs/` | Model and corpus configurations |
| `models/` | Research model cards and immutable release metadata; large weights/graphs are not tracked in Git |
| `bases/`, `adapters/`, `registry/` | Separate legacy Theory-REMI Base/Adapter contribution path |
| `scripts/`, `tools/` | Training, export, corpus, maintenance and audit utilities |
| `tests/`, `benchmarks/fixtures/` | Tests and bounded synthetic fixtures |
| `docs/` | Architecture, Web/PWA, publication and historical records |
| `experiments/`, `workloads/` | Research history and optional compute tooling |
| `web/` | Primary Compound PWA plus preserved legacy Theory-REMI runtime |

Raw corpora, downloaded archives, indexes, credentials and run outputs belong outside tracked source.

## Validation

```bash
python scripts/check_publication.py
python -m pytest -q
node --test web/*.test.mjs
```

The dedicated Compound Web export workflow downloads the pinned A2 checkpoint, verifies the checkpoint SHA, exports the reviewed V2 graphs, checks Python ONNX Runtime parity, and requires exact native-vs-WASM greedy record parity. The main Pages workflow repeats the artifact gate before deployment.

A passing source CI does not by itself certify different model bytes, musical quality, device-specific real-time performance, or rights beyond the documented release scope.

## Licenses and distribution

Source code is [Apache-2.0](LICENSE). The A2-512 checkpoint and derived reviewed Web graphs are CC-BY-NC-SA-4.0 / research-noncommercial. Dataset terms, Base terms, Adapter terms and rights in generated output remain distinct.

An Adapter or merged derivative cannot broaden the permissions of its Base. Commercial and research-NC lineages must remain separate.

See [publication boundaries](docs/PUBLICATION.md), [Compound PWA](docs/COMPOUND_PWA.md), [Compound Web runtime](docs/COMPOUND_WEB_RUNTIME.md), [the A2-512 model card](models/research_nc_aria_gigamidi_a2_512_v1/README.md), [Compound LoRA policy](docs/COMPOUND_LORA_POLICY.md), and [security guidance](SECURITY.md).
