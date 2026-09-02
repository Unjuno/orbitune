# Orbitune Commercial-Safe Base Pretraining Corpus v1

## Goal

Orbitune's Base objective is a small (~10M parameter) but broadly pretrained symbolic-music model. The Base is intentionally not specialized for retro RPG, battle, town, jazz, piano, or another narrow style. Those distributions belong in LoRA adapters. Base pretraining should instead learn general composition, instrumentation, rhythm, voice-leading, accompaniment, multi-part interaction, and long-form state.

The corpus policy is conservative: v1 uses only sources whose selected material is Public Domain, CC0, or CC-BY and whose use is compatible with commercial use according to the source metadata. This policy is engineering provenance, not legal advice.

## Included sources

The authoritative machine-readable registry is `configs/pretrain_corpus_commercial_v1.json`.

| Source | Role | v1 hard filter |
| --- | --- | --- |
| PDMX v9 | Primary scale corpus | `no_license_conflict` + upstream `deduplicated` + MIDI available/parseable |
| OpenScore Lieder | Quality anchor | CC0 canonical score tree only |
| OpenScore String Quartets | Quality anchor | CC0 canonical score tree only |
| OpenScore / Hauptstimme Orchestra | Quality anchor | CC0 canonical full-score files only; annotation-derived melody/CSV material is excluded |
| Mutopia | Supplement | Per-score Public Domain / CC0 / CC-BY 3.0/4.0 only, inferred from that score itself and converted fail-closed |
| IMSLP MIDI CC0 | Direct-MIDI supplement | Public Domain / CC0-only dataset variant, pinned to the exact data revision |

Git score sources are pinned to exact source commits in the registry. The Hugging Face source is likewise pinned to a full immutable dataset revision. The installer records resolved source identity again in `install_manifest.json`.

### PDMX

PDMX v9 reports 254,077 source scores / about 6,250 hours. Its maintainers found public-facing vs embedded license conflicts for 31,221 scores and recommend the `no_license_conflict` subset. PDMX's own arrangement-level deduplication reduces repeated exports while preserving materially different instrumentation/arrangements. Orbitune requires both flags.

The Orbitune installer intentionally does **not** download the full PDMX PDF/MXL/MusicRender distribution. Base pretraining currently needs only:

- `PDMX.csv`
- `mid.tar.gz`
- `subset_paths.tar.gz`

These are about 469 MB compressed/downloaded in total, versus many GB for the complete release. Sizes and upstream MD5 checksums are pinned and verified by `scripts/install_pretrain_corpora.py`.

## Explicitly excluded from v1

The v1 Base does not ingest MAESTRO, Aria-MIDI, GigaMIDI, Discover/Godzilla, ComMU, NES-MDB, Lakh/Slakh, MetaMIDI, SymphonyNet, Distant Listening Corpus, or other NC/unclear-license material. This is not a claim that all excluded datasets are unusable; it is a deliberate conservative Base policy.

NES-style or other retro specialization should be evaluated as LoRA data rather than silently changing the Base corpus license/style distribution.

## Candidate commercial-safe expansion sources

The following are **candidates only** and are not part of `commercial_v1` until a pinned installer, fail-closed provenance filter and regression tests are merged. A Hugging Face or repository-level license tag is not sufficient by itself when upstream score or composition rights are ambiguous.

| Candidate | Intended admission rule | Status |
| --- | --- | --- |
| CPDL / Choral Public Domain Library | Edition-level Public Domain / CC0 / CC-BY only; reject CPDL License, SA, NC and ambiguous external editions | Candidate |
| Humdrum / KernScores | Allowlisted upstream repository license plus no conflicting per-file copyright/license metadata | Candidate |
| Muse OMR Benchmark | CC0 dataset with Public Domain underlying works; pin exact dataset revision | Candidate |
| Florence Price Art Song Dataset | Use only the publisher-provided CC0 / copyright-cleared training subset | Candidate |
| Wikimedia Commons MIDI | Per-file Public Domain / CC0 / CC-BY only, with revision and license provenance retained | Candidate |

Large wrapper datasets assembled from Lakh, MetaMIDI, FreeMIDI, Bread MIDI or other unclear upstream collections remain excluded even when the wrapper repository advertises a permissive license. Upstream provenance must survive the entire rights chain.

## Current Base pretraining course

This section is the current project course for the commercial Base and should be treated as the planning source of truth until replaced by measured corpus/build results.

1. Build the complete commercial-safe corpus and record the **post-filter, post-dedup train Compound-event count**. Do not infer the final size from raw source file counts.
2. The present planning estimate is roughly **200M-300M unique-ish train Compound events**, with ~250M as a budgetary center only. This is **not** a measured corpus size and must be replaced by `build_report.json` / indexed-corpus census once the full build is complete.
3. Prefer a true one-pass first epoch over repeatedly sampling a small corpus. One epoch means consuming each admitted training composition once in a deterministic shuffled order while carrying TBPTT state within a song and resetting at song boundaries.
4. Save staged evaluation snapshots during the first pass. Initial planning gates are approximately `50M`, `100M`, `150M`, `200M`, then the measured **1.0x corpus pass** checkpoint. If the measured corpus is larger, add intermediate gates rather than skipping directly to the end.
5. Evaluate every promotion candidate with the same frozen validation/evaluation suite. The final Base is **the best validated checkpoint, not automatically the last checkpoint**.
6. If the 1.0x checkpoint is still improving materially, continue the same run in controlled additional exposure stages, for example `1.5x`, `2.0x`, `3.0x`, and at most about **1B total training-event exposure** unless evidence justifies a different ceiling.
7. `1B` is therefore an **exposure ceiling / scaling probe**, not a requirement to fabricate or admit 1B unique events. Corpus quality, provenance and deduplication take precedence over reaching a round number.
8. Keep release metadata, corpus identity, source commit, metrics and checkpoint hashes in git. Large checkpoint binaries should live in release/artifact storage rather than ordinary git history.
9. Promote the selected Base to the product/model registry only after its model card records why that checkpoint beat the alternatives (training exposure, validation, generation checks and downstream adaptation evidence).

The existing indexed samplers currently use random song selection. That is appropriate for staged CFE experiments but it does **not** constitute an exact one-pass epoch. Before full Base pretraining, add and test an epoch-aware deterministic sequential sampler so the `1.0x` checkpoint has a literal corpus-pass meaning and exact resume preserves the remaining shuffled order.

## Install

Install the optional Hugging Face dataset dependency:

```powershell
.\venv_cuda\Scripts\python.exe -m pip install -e ".[corpus]"
```

Install all registered sources locally:

```powershell
.\venv_cuda\Scripts\python.exe scripts\install_pretrain_corpora.py `
  --root data\corpora\commercial_v1
```

Large dataset bytes are ignored by git under `data/corpora/`.

The installer is restartable. PDMX files are size/checksum verified. Git sources use pinned revisions. Hugging Face direct MIDI is loaded from its pinned data revision and all upstream splits are materialized as raw `.mid` plus scalar metadata JSON before Orbitune performs its own composition-level split.

## Normalize, deduplicate and build indexed Compound data

OpenScore repositories are notation-first. Install MuseScore 4. Mutopia source-score conversion additionally requires LilyPond. Put both CLIs on `PATH` or pass them explicitly:

```powershell
.\venv_cuda\Scripts\python.exe scripts\build_pretrain_corpus.py `
  --root data\corpora\commercial_v1 `
  --musescore-bin "C:\Program Files\MuseScore 4\bin\MuseScore4.exe" `
  --lilypond-bin "C:\Program Files\LilyPond\usr\bin\lilypond.exe"
```

The build performs:

1. Score-to-MIDI conversion where required.
2. MIDI parser/canonicalization validation through Orbitune's current Compound MIDI path.
3. Raw SHA-256 calculation.
4. Canonical event fingerprinting.
5. Transposition-invariant composition fingerprinting for split leakage prevention.
6. PDMX upstream dedup filtering plus cross-source exact-normalized dedup.
7. Composition-level train/validation/test assignment.
8. Instrumentation-bucket and quality sampling weights.
9. Direct Compound Event tokenization.
10. Flat int32 memory-mapped record stores and song indexes.

OpenScore source selection is deliberately narrow. Lieder and String Quartets are read only from their canonical `scores/` trees. OpenScore Orchestra is converted only from canonical `data/**/*.mscz` full scores; `_melody.mxl`, annotation CSVs and other annotation-derived artifacts are not candidates. Mutopia does not admit arbitrary source-tree MIDI: only LilyPond primary scores whose **own file** has an allowlisted license are converted, and the converted MIDI carries a local normalized license sidecar.

Outputs are under:

```text
data/corpora/commercial_v1/
  install_manifest.json
  manifest.jsonl
  build_report.json
  converted/
  compound_indexed/
    train/
      index.json
      records.i32
      songs.jsonl
    validation/
      index.json
      records.i32
      songs.jsonl
    test/
      index.json
      records.i32
      songs.jsonl
```

The indexed format is intentional. The previous JSONL loader materializes all song records as Python objects and the fixed-window CFE sampler then copies all songs to Torch tensors. That is acceptable for the small MAESTRO smoke corpus but not for a 100k-song / hundreds-of-millions-event Base corpus. The indexed backend memory-maps one flat int32 matrix and copies only current training windows.

`index.json` contains the SHA-256 of the source manifest. Rebuild publication is fail-closed: the index is the commit marker and is published last, so a crash cannot leave an old index advertising partially replaced record/song files.

## Dedup and split semantics

Cross-source exact-normalized duplicates prefer quality-anchor editions, then PDMX, then direct-MIDI and supplement sources.

The `composition_fingerprint_v1` additionally ignores instrument/channel/velocity and represents note pitches relative to the first note. Its purpose is conservative split grouping, including obvious transpositions. It is **not** a copyright or plagiarism detector.

Every variant with the same composition fingerprint is assigned to one split. Train/validation/test composition leakage is asserted during manifest creation.

## Sampling policy

PDMX is strongly solo-piano-heavy. Orbitune is a MIDI model, not a piano-only model. The v1 training distribution therefore targets:

```text
solo / one-track             40%
small ensemble / 2-5 tracks  50%
large ensemble / 6+ tracks   10%
```

Manifest `sampling_weight` combines this bucket correction with quality bias:

```text
ordinary              1.0x
rated                  1.5x
high-rated             2.0x total
OpenScore quality      2.0x source prior
```

Enable it explicitly during training with `--weighted-corpus-sampling`. Validation remains deterministic and unweighted. Sequential TBPTT compensates song-start probability by the number of complete chunks, so long songs do not silently receive extra probability mass merely because a selected lane remains on them longer.

For the planned literal one-pass epoch, weighting must not be implemented by sampling admitted compositions with replacement. Preserve one composition visit per epoch and express desired distribution control through deterministic ordering/interleaving, selective corpus admission/downsampling, or explicitly documented loss weighting. The exact policy must remain auditable in the manifest and run metadata.

## Fixed-window indexed training

The legacy fixed-window production path remains available for controlled experiments:

```powershell
.\venv_cuda\Scripts\python.exe scripts\compound_longrun_train.py `
  --train-source data\corpora\commercial_v1\compound_indexed\train `
  --validation-source data\corpora\commercial_v1\compound_indexed\validation `
  --checkpoint runs\compound\commercial-base.pt `
  --steps 2000 `
  --batch-size 144 `
  --seq-len 256 `
  --weighted-corpus-sampling `
  --allow-fixed-window-training
```

This does not change the documented fixed-window state semantics.

## State-carry TBPTT indexed training

The intended long-form Base path is state-carry TBPTT once its RTX 3080 context-fit A/B is accepted:

```powershell
.\venv_cuda\Scripts\python.exe scripts\compound_tbptt_train.py `
  --train-source data\corpora\commercial_v1\compound_indexed\train `
  --validation-source data\corpora\commercial_v1\compound_indexed\validation `
  --checkpoint runs\compound\commercial-base-tbptt.pt `
  --steps <STAGED_TARGET> `
  --batch-size <TBPTT_CFE_BATCH> `
  --seq-len <TBPTT_CFE_SEQ> `
  --validation-songs 64 `
  --weighted-corpus-sampling
```

Do not reuse the fixed-window `batch=144, seq=256` assumption for TBPTT without the dedicated GPU context-fit experiment. The trainer's low-cost default validation count is not sufficient for selecting a large-corpus Base checkpoint: use a materially larger deterministic set such as 64 songs during staged training, and use `--validation-songs 0` for full-validation stage gates before increasing the training budget.

Indexed TBPTT checkpoint sampler state is bound to an ordered corpus identity including song hashes, composition fingerprints, lengths and sampling weights. Exact resume rejects a changed indexed training corpus instead of silently carrying lane offsets into different songs.

## Base scaling and checkpoint promotion

Do not run independent `10M -> 30M -> 100M -> 300M` corpus experiments by default. The preferred experiment is one reproducible long run over the measured commercial-safe corpus, with staged checkpoints that expose the scaling curve without changing the underlying data definition.

During the first corpus pass, checkpoint by cumulative training exposure at the documented gates and at `1.0x`. Continue beyond one pass only when held-out metrics and generation checks still improve materially. When continuation is justified, measure the same run at additional corpus-pass/exposure gates up to the planned 1B exposure ceiling.

Track at least streaming validation, per-head loss, instrumentation/generalization, generation diversity, nearest-training-piece similarity, and downstream LoRA adaptation efficiency. A strong Base is defined partly by how quickly small LoRA datasets can move it to the target musical distribution without destroying general musical competence.

Before distributing a trained Base, generate a source/attribution report from the retained manifest for any CC-BY material and include it with the model release metadata. The corpus manifest is provenance input; it is not itself a substitute for release-time attribution review.
