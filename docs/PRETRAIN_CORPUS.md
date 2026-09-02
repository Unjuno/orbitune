# Orbitune Commercial-Safe Base Pretraining Corpus v1

## Goal

Orbitune's Base objective is a small (~10M parameter) but broadly pretrained symbolic-music model. The Base is intentionally not specialized for retro RPG, battle, town, jazz, piano, or another narrow style. Those distributions belong in LoRA adapters. Base pretraining should instead learn general composition, instrumentation, rhythm, voice-leading, accompaniment, multi-part interaction, and long-form state.

The corpus policy is conservative: v1 uses only sources whose selected material is Public Domain, CC0, or CC-BY and whose use is compatible with commercial use according to the source metadata. This policy is engineering provenance, not legal advice.

## Included sources

The authoritative machine-readable registry is `configs/pretrain_corpus_commercial_v1.json`.

| Source | Role | v1 hard filter |
| --- | --- | --- |
| PDMX v9 | Primary scale corpus | `no_license_conflict` + upstream `deduplicated` + MIDI available/parseable |
| OpenScore Lieder | Quality anchor | CC0 score corpus |
| OpenScore String Quartets | Quality anchor | CC0 score corpus |
| OpenScore / Hauptstimme Orchestra | Quality anchor | CC0 score content only; annotations are not training data |
| Mutopia | Supplement | Per-item Public Domain / CC0 / CC-BY 3.0/4.0 only |
| IMSLP MIDI CC0 | Direct-MIDI supplement | Public Domain / CC0-only dataset variant |

Git score sources are pinned to exact source commits in the registry. The installer records the resolved revisions again in `install_manifest.json`.

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

The installer is restartable. PDMX files are size/checksum verified. Git sources use pinned revisions. Hugging Face direct MIDI is materialized as raw `.mid` plus scalar metadata JSON.

## Normalize, deduplicate and build indexed Compound data

OpenScore repositories are notation-first. Install MuseScore 4 and either place its CLI on `PATH` or pass it explicitly:

```powershell
.\venv_cuda\Scripts\python.exe scripts\build_pretrain_corpus.py `
  --root data\corpora\commercial_v1 `
  --musescore-bin "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
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

Enable it explicitly during training with `--weighted-corpus-sampling`. Validation should remain deterministic and unweighted.

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
  --weighted-corpus-sampling
```

Do not reuse the fixed-window `batch=144, seq=256` assumption for TBPTT without the dedicated GPU context-fit experiment.

## Base scaling plan

Do not assume the full corpus must be repeated for a fixed epoch count. Measure the 10M-class model's data scaling curve using unique Compound-event caps such as:

```text
10M -> 30M -> 100M -> 300M -> full commercial-safe corpus
```

Track at least streaming validation, per-head loss, instrumentation/generalization, generation diversity, nearest-training-piece similarity, and downstream LoRA adaptation efficiency. A strong Base is defined partly by how quickly small LoRA datasets can move it to the target musical distribution without destroying general musical competence.
