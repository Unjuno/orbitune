# Commercial v4 source status

This file is the hand-off marker for local corpus integration work. It is intentionally conservative.

Status semantics:

- **GREEN / INTEGRATE**: license/provenance is sufficiently clear to implement the downloader/converter/registry path locally. Final admission still requires the normal immutable pin, provenance manifest, cross-source dedup, split/index build, and exact final census.
- **YELLOW / HOLD**: mechanically useful, but a provenance or access gate is still unresolved. Do not add to the production corpus registry yet.
- **RED / REJECT**: do not admit to the commercial Base corpus under the current policy.

## Repository integration precondition

Before using partial `--sources` installs for v4, fix the current installer manifest semantics on `main`.

`install_pretrain_corpora.py` currently builds `installed` only from the selected source subset and then replaces `root/install_manifest.json` with that partial dictionary. A partial install can therefore erase provenance entries for already-installed sources. This is a production provenance blocker for incremental v4 assembly.

Required behavior before relying on partial installs:

- preserve compatible existing source entries when installing a subset;
- fail closed if an existing manifest belongs to a different registry/corpus identity unless an explicit migration path is implemented;
- update/replace only the source entries actually reinstalled;
- regression-test full install -> partial install -> full manifest retention.

A one-off local manifest merge script is not a substitute for fixing the production installer semantics.

## GREEN / INTEGRATE

### `nrg_cp`

**Status: GREEN / INTEGRATE**

- Upstream: WaivOps NRG-CP, Zenodo record `15304989`.
- License: CC BY 4.0.
- Upstream states that the source material was obtained from verified composers/providers for copyright clearance.
- Exact CI census on the official archive succeeded:
  - archive MD5: `7443fe30674ef149aa4c23580044f597`
  - MIDI files: `30,943`
  - parsed: `30,943 / 30,943`
  - parse failures: `0`
  - normalized unique MIDI: `30,943`
  - exact active Compound events: `4,714,362`
- Local integration must retain attribution/provenance and use the immutable archive identity above.

### `pdmx_relaxed_safe`

**Status: GREEN / INTEGRATE, BUT DO NOT WIDEN THE LICENSE PREDICATE**

- PDMX is published specifically as a public-domain symbolic-music dataset.
- Upstream explicitly recommends the `subset:no_license_conflict` population after discovering public/internal license discrepancies elsewhere in the corpus.
- Relaxing only the upstream `subset:deduplicated` gate does not relax the licensing predicate.
- Local relaxed-safe census:
  - total PDMX rows: `254,077`
  - `no_license_conflict`: `222,856`
  - current v3 deduplicated admission population: `77,321`
  - additional relaxed candidates: `145,499`
  - materially distinct additional candidates after local fingerprint audit: `101,743`
  - additional post-dedup active Compound events: `24,864,284`
  - 36 referenced MIDI files were missing on disk and must remain fail-closed rather than synthesized/reconstructed implicitly.
- Preserve the exact fail-closed selector and exclude rows with any license conflict/unclear provenance. Do not reinterpret this marker as permission to ingest arbitrary MuseScore material.
- The local agent must record the selector, source revision/archive identity, retained-row counts, missing-file handling, and delta versus the current v3 PDMX subset in the build report.

### `groove_midi_dataset`

**Status: GREEN / INTEGRATE**

- Upstream: Google Magenta Groove MIDI Dataset (GMD) v1.0.0.
- License: CC BY 4.0.
- Provenance is direct: Google hired drummers, mostly professionals, to perform/improvise the released drum material on a Roland TD-11; the MIDI is the primary recorded performance data rather than a transcription scraped from third-party recordings.
- Official MIDI-only archive:
  - file: `groove-v1.0.0-midionly.zip`
  - size: about 3.11 MB
  - SHA256: `651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e`
  - 1,150 MIDI files / 22,214 measures / 445,494 drum hits / 13.6 hours
- Prefer original GMD for symbolic training. Do **not** separately bulk-ingest E-GMD as additional symbolic volume: E-GMD repeatedly re-records the same GMD sequences across many kits and is intended mainly for audio/transcription work.

## YELLOW / HOLD

### `cocochorales`

**Status: YELLOW / HOLD FOR GENERATOR-TRAINING PROVENANCE CONSISTENCY**

- Upstream CocoChorales is explicitly CC BY 4.0 and contains 240,000 generated four-part chorale examples.
- The released symbolic material is generated rather than scraped directly from third-party compositions, and the audit measured the official tiny symbolic subset:
  - 40,000 pieces sampled
  - non-rest note rows: `3,777,260`
  - note-signature duplicates in that sample: `0`
  - projected full 240k active-event contribution: approximately `23,803,560` before Orbitune cross-source dedup
- However, the official Magenta description states that CocoChorales is dataset amplification from a Coconet model trained on the 382-example J.S. Bach Chorales Dataset.
- Orbitune already applies a stricter generator-training provenance check to `js_fake_chorales`; applying that policy consistently means the exact digital-score provenance/license of the Coconet training corpus should be documented before production admission.
- This is a provenance-policy hold, not an assertion that CocoChorales is illegally distributed. Do not download the multi-terabyte audio payload in any case; only symbolic/MIDI data is relevant to Base.

### `js_fake_chorales`

**Status: YELLOW / HOLD FOR GENERATOR-TRAINING PROVENANCE REVIEW**

- Upstream: `omarperacha/js-fakes`; repository `LICENSE` is CC BY 4.0 and the released corpus contains 500 generated chorales plus 700 JSF-Extended chorales.
- The pieces are generated by KS_Chorus, but the paper states KS_Chorus was trained on Bach chorales from the `music21` toolkit.
- Confirm the exact digital-score provenance/license of that generator-training corpus before promotion to GREEN.
- If cleared, use only as a low-weight synthetic supplement/regularizer; it is not meaningful bulk diversity.

### `imslp_midi_ccby_3_4_expansion`

**Status: YELLOW / HOLD FOR PER-FILE PROVENANCE REVALIDATION**

- Mechanical census is strong:
  - candidate MIDI sources: `3,724`
  - CC BY 4.0 metadata rows: `2,494`
  - CC BY 3.0 metadata rows: `1,230`
  - parsed: `3,724 / 3,724`
  - parse failures: `0`
  - normalized unique: `3,720`
  - exact normalized-unique active Compound events: `10,009,270`
- However, wrapper metadata alone is not the production provenance authority.
- Before GREEN status, each retained item must be fail-closed revalidated against its original IMSLP file/metadata page and attribution metadata must be retained.
- Do not add these 3,724 rows to the production registry merely because the wrapper dataset labels them CC BY.

### `cpdl_filtered_editions`

**Status: YELLOW / HOLD**

- Blanket CPDL admission is not allowed.
- Only edition-level explicit PD / CC0 / CC BY 3.0 / CC BY 4.0 candidates are potentially admissible.
- The CI census is currently blocked by CPDL/MediaWiki access returning 403 from hosted runner IPs.
- Continue only with a fail-closed edition-level audit from an allowed access path; never fall back to the default CPDL license or ambiguous editions.

### `musicnet_reference_midi`

**Status: YELLOW / HOLD FOR PER-MIDI LICENSE CHAIN**

- Official MusicNet release contains 330 freely-licensed classical recordings and a 2.6 MB archive of reference MIDI used to construct the note labels.
- The official metadata includes source and transcriber fields and says that provenance of specific recordings and MIDI is described there.
- That metadata does **not** itself provide an explicit per-MIDI license field. Many MIDI files are credited to named modern transcribers/sites, so public-domain composition status alone does not establish rights to the digital transcription.
- Before GREEN status, build a fail-closed allowlist that resolves each retained reference MIDI to an upstream source/transcriber license permitting commercial reuse, or obtain an authoritative MusicNet statement that the MIDI archive itself is covered by a suitable reusable license.
- Do not substitute MusicNet-EM for this purpose without a separate license check; derived/refined label releases have different terms in the ecosystem.

### `strudel_synth`

**Status: YELLOW / HOLD FOR CONDITIONING-SEED PROVENANCE REVIEW**

- Upstream `haiyewon/Strudel-Synth` is explicitly CC BY 4.0 and contains 21,174 synthetic MIDI/Strudel pairs generated by Claude and rendered through Strudel.
- The construction paper states that musical conditioning seeds expose attributes derived from EveryNoise, TheoryTab, and MidiCaps, including genre/key/tempo/meter/instrumentation/chord-progression metadata.
- Because those seeds can encode song-specific analysis/metadata from third-party corpora, confirm that using those conditioning attributes to generate the symbolic corpus satisfies Orbitune's strict provenance standard before promotion.
- Scale is modest relative to the remaining v4 gap, so this should not block v4 if cleaner sources are available.

## RED / REJECT OR DO NOT USE FOR COMMERCIAL BASE

### `bach_doodle`

**Status: RED / DO NOT ADMIT**

- Dataset wrapper license is CC BY 4.0, and the scale is enormous.
- Eight-shard audit sample: `354,666` rows, mean output notes `40.68`, projected output note events about `878.6M`.
- But the source contains user-entered melodies, and the official dataset description does not establish a clean composition-rights chain for every submitted melody.
- The dataset license therefore does not satisfy Orbitune's current commercial-safe provenance requirement for the underlying musical compositions.

### `dmelodies`

**Status: RED / DO NOT ADMIT**

- The repository README explicitly marks the dataset **CC BY-NC-SA 4.0**.
- The dataset is synthetic and large (`1,354,752` simple two-bar melodies), but the NC and SA terms are both denied by the current commercial Base policy.

### `nes_mdb`

**Status: RED / DO NOT ADMIT**

- NES-MDB contains 5,278 symbolic songs extracted from the assembly/VGM data of 397 commercial NES games.
- The paper/code may carry open licenses, but those licenses do not establish commercial training rights to the underlying game soundtrack compositions.
- Do not infer composition rights from the dataset or repository software license.

### Restricted/NC Humdrum-family material

**Status: RED / DO NOT ADMIT**

- MuseData material carrying personal/academic/research-only restrictions is excluded.
- CC BY-NC, CC BY-NC-SA, CC BY-SA, or other policy-denied sources remain excluded even when technically convertible to Humdrum/MIDI.

## Local-agent hand-off rule

The local agent may immediately implement/download/build only the **GREEN** sources above. It must not silently promote YELLOW sources. If a source-specific audit changes a status, update this file and preserve the evidence before changing the production registry.

As of this audit pass, the immediate GREEN integration queue is therefore:

1. `pdmx_relaxed_safe`
2. `nrg_cp`
3. `groove_midi_dataset`

`cocochorales` has been moved to YELLOW pending the same generator-training provenance standard already applied to `js_fake_chorales`.
