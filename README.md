# Orbitune

Local-first symbolic MIDI generation with a hierarchical Compound Transformer and a separate legacy Theory-REMI / LoRA runtime.

## Start here

| Goal | Entry point |
| --- | --- |
| Install and inspect the code | Quick start below |
| Understand the trained research model | [Research model card](models/research_nc_aria_gigamidi_v1/README.md) |
| Use an independently obtained checkpoint | [Checkpoint verification and generation](models/research_nc_aria_gigamidi_v1/usage.md) |
| Understand release availability and limitations | [Publication status](docs/PUBLICATION.md) |
| Contribute code, Bases, or Adapters | [Contributing](CONTRIBUTING.md) |
| Browse architecture and historical work | [Documentation index](docs/README.md) |

## What is available

**This repository publishes source code and research-model documentation, not a downloadable pretrained model release.** The documented Aria+GigaMIDI checkpoint is not tracked in Git, has no download URL in the publication record, and has no published Compound ONNX/Web artifact. A clone is not a pretrained installation.

The [publication record](models/research_nc_aria_gigamidi_v1/publication.json) explicitly records this distinction. The older model `manifest.json` is retained as a historical training report; it is **not** a validated public Base-registry entry.

The reported final research checkpoint is `research-nc-aria-gigamidi-v1`, at global step 100,000. Its recorded indexed train corpus contains 4,069,137,373 active next-event pairs. **Corpus capacity is not the number of unique events consumed by training.** See the [data card](models/research_nc_aria_gigamidi_v1/training_data.md) for accounting limitations and the [reproducibility notes](models/research_nc_aria_gigamidi_v1/reproducibility.md) for the historical sampler-resume bug.

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

A local checkpoint from a trusted source is required for generation. The exact implemented flags are documented in [usage.md](models/research_nc_aria_gigamidi_v1/usage.md). Do not load an untrusted `.pt` file merely because its filename looks correct.

## Runtime boundaries

**Compound Transformer:** factorized event embeddings, local/medium/global attention, recurrent memory, and discrete/continuous event heads. A serialized Compound record has 12 fields. Architecture and runtime details are in [COMPOUND_BASE.md](docs/COMPOUND_BASE.md).

**Theory-REMI reference:** a separate legacy path for the `orbitune` CLI, Base/Adapter registry, LoRA, and ONNX/browser tooling. Its ABI is not interchangeable with the experimental Compound checkpoint. Do not put the documented research manifest into `bases/` or claim browser compatibility without the required export and validation.

## Repository map

| Location | Purpose |
| --- | --- |
| `orbitune/` | Model, MIDI representation, sampler, and runtime code |
| `configs/` | Model and corpus configurations; existing source pins are preserved |
| `models/` | Research model cards and metadata; candidate weights are local-only |
| `bases/`, `adapters/`, `registry/` | Separately validated legacy Base/Adapter contribution path |
| `scripts/`, `tools/` | Training, corpus, maintenance, and audit utilities |
| `tests/`, `benchmarks/fixtures/` | Tests and bounded synthetic fixtures |
| `docs/`, `experiments/`, `workloads/` | Documentation, research history, and optional compute tooling |
| `web/` | Legacy browser runtime, not a Compound model release |

Raw corpora, downloaded archives, indexes, credentials, and run outputs belong outside tracked source. Historical reports remain in place so their references and evidence identities are not broken.

## Validation

Run the publication checks without training or network access:

```bash
python scripts/check_publication.py
python -m pytest -q tests/test_publication.py
```

The broader test suite contains CPU model/CLI fixtures and can be run with `python -m pytest -q`. A passing source CI does not certify an unpublished checkpoint, corpus completeness, or musical quality.

## Licenses and distribution

Source code is [Apache-2.0](LICENSE). The research model's recorded checkpoint license is CC-BY-NC-SA-4.0 and its project policy remains noncommercial; this cleanup does not grant additional rights or publish its weights. Dataset terms, checkpoint terms, and rights in generated output are distinct. Do not infer a universal generated-output license from the code license or a dataset label.

See [publication boundaries](docs/PUBLICATION.md), [the model card](models/research_nc_aria_gigamidi_v1/README.md), and [security guidance](SECURITY.md).
