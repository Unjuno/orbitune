# Commercial-safe Base corpus v3

`configs/pretrain_corpus_commercial_v3.json` is an additive successor to the
frozen `commercial_v2` registry. v1 and v2 remain unchanged so their source
sets and measured identities stay reproducible.

## Admission policy

v3 continues the conservative Base policy:

1. Prefer Public Domain and CC0 material.
2. Admit high-value CC-BY 3.0/4.0 material only when the exact upstream source
   revision is pinned and release-time attribution can be generated.
3. Reject NC, SA/copyleft, unknown, unclear, or moving/unpinned source material.
4. Build a new manifest/index/corpus identity before production training. Do
   not append data to an already-started production sampler lineage.

## Added source: NIFC Polish Music Heritage in Open Access

```text
repository: https://github.com/pl-wnifc/humdrum-polish-scores
revision:   13ac964e0dd8bcd5fffd837169cbf653242c12e8
license:    CC-BY-4.0
```

The pinned upstream repository describes roughly 8.9k Humdrum scores and
13.4M sounding notes. Its `LICENSE.txt` explicitly allows sharing and
adaptation for any purpose, including commercial use, subject to attribution.

Orbitune admits only the pinned `**/kern/*.krn` score files and converts them
through `hum2mid`. Derived MIDI remains outside the git checkout. The manifest
retains `source_id=nifc_polish_scores` and `license=cc-by-4.0` so attribution
can be generated at release time.

## Added source: NIFC Chopin First Editions

```text
repository: https://github.com/pl-wnifc/humdrum-chopin-first-editions
revision:   95dfb105c1669c72d10b04088566154f12d3dc1c
license:    CC-BY-4.0
```

The pinned repository contains Humdrum encodings of Chopin first editions. Its
own `bin/makeMidi` uses the same conversion contract adopted by Orbitune:

```text
hum2mid <input.krn> -o <output.mid>
```

Orbitune admits only `kern/*.krn` at the pinned revision. Attribution is
required for this source as well.

## Humdrum conversion dependency

`scripts/build_pretrain_corpus.py` now accepts:

```text
--hum2mid-bin <path-to-hum2mid-or-shim>
```

If a configured Humdrum source has score candidates but `hum2mid` is missing,
the build fails closed instead of silently omitting that source.

On the current Windows/WSL2 corpus host it is acceptable to expose a small
Windows `.bat` shim that calls a working WSL `hum2mid`, just as MuseScore and
LilyPond are already exposed through shims. Do not modify the pinned NIFC
checkouts in-place.

## Muse OMR Benchmark status

Muse OMR remains a **high-priority CC0 source**, but it is deliberately not in
the production v3 registry yet.

Verified upstream facts:

```text
repo:       musegroup/omr_benchmark (Hugging Face dataset)
size:       1,077 symbolic-score/PDF pairs
score form: MuseScore Studio score files
license:    CC0-1.0
works:      Public Domain according to the upstream dataset card
code repo:  https://github.com/musescore/omr_benchmark
```

The data bytes live on Hugging Face. The currently available connector metadata
confirms the CC0/PD policy but does not expose the exact immutable 40-character
Hub revision. Orbitune therefore refuses to put a moving `main` reference into
the production registry.

Before Muse OMR is admitted, resolve the exact Hub SHA with a normal
`huggingface_hub` client on the corpus host, for example:

```powershell
.\venv_cuda\Scripts\python.exe -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('musegroup/omr_benchmark').sha)"
```

Then add that exact SHA to a registry source and download only the symbolic
score payload needed for MuseScore-to-MIDI conversion. PDFs are not required
for Base pretraining. The installer must record the resolved revision and the
build must retain the CC0 provenance.

This is a provenance blocker, not a license blocker.

## Local v3 build outline

Use a fresh root, for example `C:\ov3`. Reuse the already-installed v2 sources
with junctions/symlinks, then install only the two NIFC sources.

```powershell
$V2 = "C:\ov2"
$V3 = "C:\ov3"
New-Item -ItemType Directory -Force $V3 | Out-Null

foreach ($name in @(
  "pdmx",
  "openscore_lieder",
  "openscore_string_quartets",
  "openscore_orchestra",
  "mutopia",
  "imslp_midi_cc0",
  "florence_price_art_songs"
)) {
  if (-not (Test-Path "$V3\$name")) {
    New-Item -ItemType Junction -Path "$V3\$name" -Target "$V2\$name" | Out-Null
  }
}

.\venv_cuda\Scripts\python.exe scripts\install_pretrain_corpora.py `
  --config configs\pretrain_corpus_commercial_v3.json `
  --root $V3 `
  --sources nifc_polish_scores,nifc_chopin_first_editions
```

Build with the existing MuseScore/LilyPond binaries plus hum2mid:

```powershell
.\venv_cuda\Scripts\python.exe scripts\build_pretrain_corpus.py `
  --config configs\pretrain_corpus_commercial_v3.json `
  --root C:\ov3 `
  --musescore-bin "<working MuseScore shim/exe>" `
  --lilypond-bin "<working LilyPond shim/exe>" `
  --hum2mid-bin "<working hum2mid shim/exe>"
```

After build, record the new train/validation/test census, per-source event
counts, dedup removals, manifest SHA, indexed corpus identity and exact
`EpochAwareNoReplacementSampler.epoch_events_total` invariant across at least
B2/S32, B4/S64 and B1/S128.

Do not resume a v1/v2 production sampler checkpoint against v3. Production
training starts fresh only after the final corpus identity is locked.
