# Orbitune

Local-first symbolic MIDI generation with a hierarchical Compound Transformer and a separate legacy Theory-REMI / LoRA runtime.

## Start here

| Goal | Entry point |
| --- | --- |
| Install and inspect the code | Quick start below |
| Understand the trained research model | [Research model card](models/research_nc_aria_gigamidi_v1/README.md) |
| Understand the Compound architecture | [Compound Base](docs/COMPOUND_BASE.md) |
| Inspect the Compound browser runtime | [Compound Web runtime](docs/COMPOUND_WEB_RUNTIME.md) |
| Understand how Compound LoRA will be applied | [Compound LoRA policy](docs/COMPOUND_LORA_POLICY.md) |
| Use an independently obtained checkpoint | [Checkpoint verification and generation](models/research_nc_aria_gigamidi_v1/usage.md) |
| Understand release availability and limitations | [Publication status](docs/PUBLICATION.md) |
| Contribute code, Bases, or Adapters | [Contributing](CONTRIBUTING.md) |
| Browse architecture, roadmap and historical work | [Documentation index](docs/README.md) |

## What is available

**This repository is public source code, browser-runtime source, documentation and CI — not a downloadable pretrained model release.** The documented Aria+GigaMIDI checkpoint is not tracked in Git, has no download URL in the publication record, and its Compound ONNX graphs are not distributed here. A clone is not a pretrained installation.

The Compound browser runtime is implemented and deployed in fail-closed form: generation remains disabled until a separately reviewed release supplies exact model URLs and SHA-256 values. The [publication record](models/research_nc_aria_gigamidi_v1/publication.json) records the current documentation-only model availability.

The reported frozen research checkpoint is `research-nc-aria-gigamidi-v1`, at global step 100,000. Its recorded indexed train corpus contains 4,069,137,373 active next-event pairs. **Corpus capacity is not the number of unique events consumed by training.** The checkpoint record reports cumulative `events_seen = 409,600,000`; replacement sampling means this is not an exact epoch/coverage claim. See the [data card](models/research_nc_aria_gigamidi_v1/training_data.md) and [reproducibility notes](models/research_nc_aria_gigamidi_v1/reproducibility.md).

A later full-parameter continuation may become a new Base candidate only after it is completed, evaluated, frozen and documented under a new immutable identity. It does not overwrite the historical 100k checkpoint.

## Quick start: source checkout

Python 3.10 or newer is required by the package. The existing Python CI uses Python 3.11; CUDA is not needed to inspect the code. Use a virtual environment and install the PyTorch build appropriate to your machine when GPU execution is required.

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

Then install and inspect, without downloading datasets or starting training:

```bash
python -m pip install -e ".[dev]"
orbitune-compound --help
orbitune-compound info --config configs/compound_hierarchical_9m.json
```

A local checkpoint from a trusted source is required for Python generation. The exact implemented flags are documented in [usage.md](models/research_nc_aria_gigamidi_v1/usage.md). Do not load an untrusted `.pt` file merely because its filename looks correct.

## Runtime and Adapter boundaries

**Compound Transformer:** factorized Compound event embeddings, local/medium/global attention, routed fast/medium/slow recurrent memory, an intra-event Transformer and mixed discrete/continuous heads. A serialized Compound record has 12 fields. Architecture details are in [COMPOUND_BASE.md](docs/COMPOUND_BASE.md); the separate browser ABI is in [COMPOUND_WEB_RUNTIME.md](docs/COMPOUND_WEB_RUNTIME.md).

**Compound LoRA:** Base pretraining remains full-parameter training. LoRA is a post-Base adaptation stage against an immutable checkpoint. The public Compound Adapter ABI is not frozen yet, so target modules/rank must not be copied from the legacy stack. The planned initial Web strategy is a pre-merged variant, but each future merged artifact must pass native/Web parity on its own exact bytes before publication. See [COMPOUND_LORA_POLICY.md](docs/COMPOUND_LORA_POLICY.md).

**Theory-REMI reference:** a separate legacy path for the `orbitune` CLI, Base/Adapter registry, rank-4 LoRA and its original ONNX/browser tooling. Its `orbitune-lora-v0` ABI is not interchangeable with Compound.

## Repository map

| Location | Purpose |
| --- | --- |
| `orbitune/` | Model, MIDI representation, sampler and runtime code |
| `configs/` | Model and corpus configurations; existing source pins are preserved |
| `models/` | Research model cards and publication metadata; documented weights are not tracked |
| `bases/`, `adapters/`, `registry/` | Separately validated legacy Theory-REMI Base/Adapter contribution path |
| `scripts/`, `tools/` | Training, corpus, maintenance and audit utilities |
| `tests/`, `benchmarks/fixtures/` | Tests and bounded synthetic fixtures |
| `docs/` | Current architecture/publication policies plus clearly scoped historical research records |
| `experiments/`, `workloads/` | Research history and optional compute tooling |
| `web/` | Legacy Theory-REMI page plus an isolated Compound V2 browser runtime; model binaries are not tracked |

Raw corpora, downloaded archives, indexes, credentials and run outputs belong outside tracked source. Historical reports remain in place so their references and evidence identities are not broken.

## Validation

Run the publication checks without training or network access:

```bash
python scripts/check_publication.py
python -m pytest -q tests/test_publication.py
```

Browser-runtime unit tests run under Node 22 in CI:

```bash
node --test web/*.test.mjs
```

The broader Python test suite can be run with `python -m pytest -q`. A passing source CI does not certify an unpublished checkpoint, corpus completeness, musical quality or redistribution rights.

## Licenses and distribution

Source code is [Apache-2.0](LICENSE). The research model's recorded checkpoint license is CC-BY-NC-SA-4.0 and its project policy remains noncommercial; public runtime/LoRA policy documentation does not grant additional rights or publish its weights. Dataset terms, Base terms, Adapter terms and rights in generated output are distinct.

An Adapter or merged derivative cannot broaden the permissions of its Base. Commercial and research-NC lineages must remain separate.

See [publication boundaries](docs/PUBLICATION.md), [the model card](models/research_nc_aria_gigamidi_v1/README.md), [Compound LoRA policy](docs/COMPOUND_LORA_POLICY.md), and [security guidance](SECURITY.md).
