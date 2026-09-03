# Commercial-safe Base corpus v3

`configs/pretrain_corpus_commercial_v3.json` is an additive successor to the
frozen `commercial_v2` registry. v1 and v2 remain unchanged so their source
sets and measured identities stay reproducible.

## Admission policy

v3 keeps the conservative Base policy:

1. Prefer Public Domain and CC0 material.
2. Admit high-value CC-BY 3.0/4.0 material only when provenance is explicit and
   release-time attribution can be generated.
3. Reject NC, SA/copyleft, unknown, unclear, or unprovenanced source material.
4. Network sources must be reduced to an immutable revision before their bytes
   are admitted into a corpus root.
5. Build a new manifest/index/corpus identity before production training. Do
   not append data to an already-started production sampler lineage.

## Added source: Muse OMR Benchmark

```text
repository: musegroup/omr_benchmark (Hugging Face dataset)
size:       1,077 symbolic-score/PDF pairs upstream
score form: MuseScore Studio score files
license:    CC0-1.0
works:      Public Domain according to the upstream dataset card
code repo:  https://github.com/musescore/omr_benchmark
```

Orbitune downloads only the symbolic MuseScore payload plus
`benchmark_dataset.json`; PDF payloads are explicitly excluded from the Base
corpus installer.

The registry intentionally uses:

```text
revision_policy = resolve-exact-at-install
```

This does **not** mean training follows a moving Hub branch. On the first local
install, `scripts/install_pretrain_corpora.py` asks the Hugging Face API for the
current dataset SHA, requires an exact 40-character hexadecimal revision, and
then passes that exact SHA to `snapshot_download`.

The successful snapshot writes:

```text
<source-root>/.orbitune_source_lock.json
```

and the exact resolved revision is also returned into `install_manifest.json`.
Every later install against that corpus root reuses the locked SHA instead of
resolving Hub state again. A non-empty source directory without the lock fails
closed, preventing accidental reuse of an unprovenanced or mixed snapshot.

The lock also binds the permitted download patterns. A registry change that
would alter those patterns fails instead of silently changing the materialized
source. Download patterns containing PDF payloads are rejected.

After installation, MuseScore conversion follows the same derived-data path as
other score-only sources; downloaded source score files remain untouched.

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

`scripts/build_pretrain_corpus.py` accepts:

```text
--hum2mid-bin <path-to-hum2mid-or-shim>
```

If a configured Humdrum source has score candidates but `hum2mid` is missing,
the build fails closed instead of silently omitting that source.

On the current Windows/WSL2 corpus host it is acceptable to expose a small
Windows `.bat` shim that calls a working WSL `hum2mid`, just as MuseScore and
LilyPond are already exposed through shims. Do not modify the pinned NIFC
checkouts in-place.

## Local v3 build outline

Use a fresh root, for example `C:\ov3`. Reuse the already-installed v2 sources
with junctions/symlinks, then install only the three v3 additions.

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

.\venv_cuda\Scripts\python.exe -m pip install -e ".[corpus]"

.\venv_cuda\Scripts\python.exe scripts\install_pretrain_corpora.py `
  --config configs\pretrain_corpus_commercial_v3.json `
  --root $V3 `
  --sources muse_omr_benchmark,nifc_polish_scores,nifc_chopin_first_editions
```

Before building, inspect the exact Muse OMR lock and install manifest entry:

```powershell
Get-Content C:\ov3\muse_omr_benchmark\.orbitune_source_lock.json
Get-Content C:\ov3\install_manifest.json
```

Both must report the same exact 40-character Hub SHA.

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
training starts fresh only after the final v3 corpus identity is locked.
