# Commercial-safe Base corpus v2

`configs/pretrain_corpus_commercial_v2.json` is an additive successor to the
frozen `commercial_v1` registry. v1 remains available so its measured corpus
identity and 184,787,415-event one-pass census stay reproducible.

## Added source: Florence Price Art Song Dataset

v2 adds `florence_price_art_songs` from the pinned upstream commit:

```text
TT515/Florence_Price_Art_Song_Dataset
fa25c98d495e4ab86217dcc341a4b2d9fb714cfa
```

The pinned upstream README states that the scores are released under CC0 and
explicitly lists 17 songs omitted because their copyright renewal/status could
not be confirmed. Orbitune does not attempt to reconstruct or re-admit those
omitted works. The repository generation is pinned so later upstream changes
cannot silently alter the admitted set.

The complete-song tree already contains MIDI for almost every admitted score.
Orbitune consumes those source MIDI files directly. The only explicit score
conversion candidate is `price_songs_main/Thou'rt My Loved One/*.mscz`, which
upstream documents as a complete song supplied without a MIDI file. The
`price_incomplete/` tree is not a conversion candidate.

The source is a small high-quality direct-MIDI supplement with
`quality_weight = 2.0`, not a scale replacement for PDMX. It deliberately does
not use the `quality-anchor` tier because the current v1 telemetry maps that
tier to the OpenScore-specific `openscore_verified` flag.

## Split compatibility

v2 intentionally preserves the v1 composition split seed:

```text
orbitune-commercial-safe-v1
```

Because split assignment is a deterministic function of composition
fingerprint plus seed, this keeps every existing v1 composition in the same
train/validation/test split. Only newly admitted compositions receive new
assignments.

## Local migration without redownloading v1

Use a new corpus root, for example `C:\ov2`, rather than overwriting the
already-built v1 indexed corpus. Existing v1 source directories may be reused
through junctions/symlinks, then only the new Florence Price source needs to be
installed.

Example outline on Windows:

```powershell
$V1 = "C:\ov1"
$V2 = "C:\ov2"
New-Item -ItemType Directory -Force $V2 | Out-Null

foreach ($name in @(
  "pdmx",
  "openscore_lieder",
  "openscore_string_quartets",
  "openscore_orchestra",
  "mutopia",
  "imslp_midi_cc0"
)) {
  if (-not (Test-Path "$V2\$name")) {
    New-Item -ItemType Junction -Path "$V2\$name" -Target "$V1\$name" | Out-Null
  }
}

.\venv_cuda\Scripts\python.exe scripts\install_pretrain_corpora.py `
  --config configs\pretrain_corpus_commercial_v2.json `
  --root $V2 `
  --sources florence_price_art_songs
```

Then build v2 with the same MuseScore/LilyPond binaries used for v1:

```powershell
.\venv_cuda\Scripts\python.exe scripts\build_pretrain_corpus.py `
  --config configs\pretrain_corpus_commercial_v2.json `
  --root $V2 `
  --musescore-bin "<working MuseScore shim/exe>" `
  --lilypond-bin "<working LilyPond shim/exe>"
```

Record the new manifest SHA, train corpus identity, song/event census, and
`EpochAwareNoReplacementSampler.epoch_events_total` before production training.

## Training-lineage rule

Changing from v1 to v2 changes the corpus identity. This is harmless while data
is still being acquired/built and no production Base lineage has started.

Once a production checkpoint has trained on v1, do **not** append v2 data and
resume that checkpoint as though it were the same run. The production sampler
binds its checkpoint state to corpus identity and should fail closed on such a
mismatch. If training has already started, preserve that run as a v1 experiment
and start a fresh v2 lineage from the same model initialization policy.

## Remaining expansion candidates

The following remain candidates until their exact source revision and
fail-closed admission policy are implemented and tested:

- Muse OMR Benchmark — upstream advertises CC0 / Public Domain works, but the
  Hugging Face dataset revision still needs to be pinned in the Orbitune
  registry before admission.
- Humdrum / KernScores — a meta-corpus of many repositories with heterogeneous
  repository/per-file licensing; admission must be per-source, not blanket.
- CPDL — only edition-level Public Domain / CC0 / CC-BY material is suitable for
  the conservative proprietary Base path; CPDL-license/copy-left editions stay
  excluded.
- Wikimedia Commons MIDI — requires per-file revision and license provenance;
  no blanket Commons admission.

The goal is to maximize commercially usable data without weakening provenance
or silently changing an already-started training distribution.
