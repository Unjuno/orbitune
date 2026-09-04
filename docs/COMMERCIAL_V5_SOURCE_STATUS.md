# Commercial v5 source status

This file is the v5 successor to `COMMERCIAL_V4_SOURCE_STATUS.md`. It is intentionally conservative. The v5 status semantics are identical to the v4 status semantics; see `COMMERCIAL_V4_SOURCE_STATUS.md` for the full definition of GREEN / YELLOW / RED.

## v4 outcomes and the v5 starting state

v4 was the v3 identity plus three additions, with the PDMX upstream `deduplicated` gate relaxed. v4 was stopped at the 300M-event gate:

- `V4_TRAIN_SONGS = 215,963`
- `V4_TRAIN_RECORDS = 234,904,281`
- `V4_ONE_X_ACTIVE_EVENTS = 234,688,318`
- `V4_MANIFEST_SHA256 = 1c582a08a3087952a57a604b5652cae2bef6dd4e6acb2addc5eb49cdfbe58c72`
- `V4_TRAIN_CORPUS_IDENTITY = 61dec03cb00dad05d80f80e01c113f14d7abdc77d52e1d2004ba55f90eee35c6`
- v3 identity preserved: `V3_TRAIN_SONGS = 84,993`, `V3_TRAIN_RECORDS = 204,683,325`, `V3_MANIFEST_SHA256 = 72ca3ee809063eae140f893826957c62ea1eaba6c8988dd08585690ac1eb6fdc`, `V3_TRAIN_CORPUS_IDENTITY = dffd3d55a665c113fb22ccb1b114d295bce7db628c6e144d63b323b7c7c7242a`
- `DEFICIT_TO_300M = 65,095,719` events (v3 had a `95,316,675` deficit)
- Sampler invariant B=2/S=32 = B=4/S=64 = B=1/S=128 = `234,688,318` (PASS)

The v5 question is the same as the v4 question: which held source can be admitted without lowering license quality, with reproducible provenance, fail-closed dedup, and a documented contribution to the deficit? v4 added only ~30.2M events. v5 must add at least `65,095,719` events post-dedup to clear the 300M gate; with the v4 admission plan unchanged the headroom from v4 to v5 is the same.

v5 is documented here on this audit branch only. No production code, no install, no build, no commit on `main` will be made from this file.

## GREEN candidates (under v5 conditions)

No v5 GREEN promotions are made by this file. The three v4 GREEN sources remain GREEN for v5:

- `pdmx` (relaxed policy)
- `nrg_cp`
- `groove_midi_dataset`

Their status and census values from `COMMERCIAL_V4_SOURCE_STATUS.md` are not re-opened here.

## YELLOW / HOLD candidates under v5 review

### `imslp_midi_ccby_3_4_expansion`

**Status: YELLOW / HOLD FOR PER-FILE PROVENANCE REVALIDATION (unchanged from v4)**

The mechanical census is strong and was previously measured end-to-end:

- candidate MIDI sources: `3,724`
- CC BY 4.0 metadata rows: `2,494`
- CC BY 3.0 metadata rows: `1,230`
- parsed: `3,724 / 3,724`
- parse failures: `0`
- normalized unique: `3,720`
- exact normalized-unique active Compound events: `10,009,270`

Wrapper metadata alone is not the production provenance authority. The hand-off into v5 requires the following before any GREEN status is granted:

1. **Per-file IMSLP revalidation** for each of the 3,724 wrapper rows, against its original IMSLP file/metadata page. The wrapper dataset is not authoritative for license claims.
2. **Attribution retention** for every admitted row. The full composer / work / edition / license / source URL must persist in the per-row provenance manifest. CC BY 3.0 and CC BY 4.0 differ in attribution wording; both are admissible but the manifest must record which clause applies.
3. **Filepath integrity check**. The previous v3 install history shows that the IMSPD / IMSLP corpus contains filenames that are Windows-illegal (e.g. `Thou'rt My Loved One`). The expansion corpus is larger and is expected to contain more such files. The local admission path must fail closed on Windows-illegal filenames or pre-filter them with a documented allowlist glob.
4. **Projected contribution at most ~10M events post-dedup** even under the optimistic case. The expansion alone is too small to clear the v4-to-v5 deficit of `65,095,719` events, so v5 cannot rely on this single source for the headroom. It would be admitted as a quality-anchor supplement, not as the deficit-bridger.
5. **ND / IMNSF / not-public-domain exclusion**. Any row in the expansion that resolves to a non-free or unclear-license IMSLP file must be excluded even when the wrapper says CC BY. The wrapper is not the authority.

Until those five conditions are met, this source remains YELLOW / HOLD. No registry or install step will be created for it on `main` from this branch.

### `cocochorales`

**Status: YELLOW / HOLD FOR GENERATOR-TRAINING PROVENANCE CONSISTENCY (under v5 re-evaluation)**

CocoChorales is CC BY 4.0 and contains 240,000 generated four-part chorale examples. The released symbolic material is generated rather than scraped directly from third-party compositions, but the Coconet model that produced it was trained on the J.S. Bach Chorales Dataset. Orbitune's stricter generator-training provenance check (the same one applied to `js_fake_chorales`) is therefore owed to CocoChorales before promotion.

The previous measurement on a 40,000-piece sample projected approximately `23,803,560` active events for the full 240k. v5 does not have a new measurement to override that figure.

v5 re-evaluation tool: `tools/cocochorales_train_corpus_provenance.py` (this audit branch only). It applies the 9-condition v5 GREEN gate against the documented upstream chain:

- `G1` license_is_cc_by_3_or_4: **PASS** — wrapper is CC BY 4.0 (Yusong Wu, Magenta official).
- `G2` underlying_composition_admissible: **PASS** — the J.S. Bach four-part chorales are public-domain in the US, EU, and Canada (Bach d. 1750).
- `G3` edition_encoding_rights_admissible: **PASS** — CocoChorales is Coconet-generated, not a transcription of any specific third-party recording or score; v4 audit measured 0 note-signature duplicates in a 40k sample.
- `G4` not_nd: **PASS** — wrapper is CC BY 4.0 only, no ND.
- `G5` not_imnsf: **PASS** — the underlying work is in the public domain, not in IMNSF, not in-copyright.
- `G6` not_non_public_domain: **PASS** — underlying work is PD; wrapper is CC BY 4.0.
- `G7` attribution_recoverable: **PASS** — five attribution lines recorded: Yusong Wu (CC BY 4.0), Coconet (Apache-2.0, Magenta team), J.S. Bach (public-domain), JSB Chorales Dataset (czhuang, no LICENSE file but underlying music is PD), music21 corpus (BSD-3-Clause code with Margaret Greentree attribution).
- `G8` pathname_policy: **PASS_PENDING_INSTALL_CHECK** — CocoChorales tiny subset uses standard filenames; the actual Windows-illegal-character check is performed at install time, not at audit time.
- `G9` parse_conversion_succeeds: **PASS** — v4 audit: 0 parse failures in 40k sample, 3,777,260 non-rest note rows, 0 note-signature duplicates.

v5 re-evaluation result: the static chain passes all 9 conditions, with G8 deferred to install time. The 40k signature-duplicate-zero measurement is the empirical evidence that condition G3 (edition/encoding rights admissible) holds: the generated chorales are not verbatim transcriptions of any specific Bach chorale.

A production GREEN promotion for CocoChorales is still **not** granted by this audit commit. The next step is the installer-side per-row census against the official Magenta tiny symbolic subset, which produces a real per-row verdicts file rather than the 2-row synthetic fixture used by the audit tool. Only after that real per-row census passes the gate is CocoChorales eligible to write a v5 registry entry on `main`.

### `js_fake_chorales`

**Status: YELLOW / HOLD FOR GENERATOR-TRAINING PROVENANCE REVIEW (unchanged from v4)**

The 500 / 700 generated chorales were produced by KS_Chorus, which was trained on Bach chorales from the `music21` toolkit. Confirm the exact digital-score provenance / license of that generator-training corpus before promotion.

### `cpdl_filtered_editions`

**Status: YELLOW / HOLD (unchanged from v4)**

Blanket CPDL admission is not allowed. Edition-level explicit PD / CC0 / CC BY 3.0 / CC BY 4.0 candidates are potentially admissible, but the CI census is blocked by CPDL/MediaWiki access returning 403 from hosted runner IPs. v5 still has no allowed access path.

### `musicnet_reference_midi`

**Status: YELLOW / HOLD FOR PER-MIDI LICENSE CHAIN (unchanged from v4)**

The official MusicNet release does not provide an explicit per-MIDI license field. Many reference MIDI files are credited to named modern transcribers, so public-domain composition status alone does not establish rights to the digital transcription. v5 has no new information to upgrade the status.

### `strudel_synth`

**Status: YELLOW / HOLD FOR CONDITIONING-SEED PROVENANCE REVIEW (unchanged from v4)**

The construction paper states that musical conditioning seeds expose attributes derived from EveryNoise, TheoryTab, and MidiCaps. Because those seeds can encode song-specific analysis / metadata from third-party corpora, the conditioning attribute origin must be confirmed before promotion. Scale is small relative to the remaining v4-to-v5 deficit, so this should not block v5 if cleaner sources are available.

## RED / REJECT

The following sources remain RED. v5 does not re-litigate them.

- `bach_doodle` — user-entered melodies, no clean composition-rights chain.
- `dmelodies` — CC BY-NC-SA 4.0.
- `nes_mdb` — derived from commercial NES game soundtracks; no commercial training rights.
- Restricted / NC Humdrum-family material — policy-denied regardless of convertibility.

## Cross-source v5 admission gate

A v5 admission must satisfy all of the following before it is allowed to write a registry entry on `main`:

1. **License and provenance** are explicit at the per-file level (wrapper metadata is not the authority).
2. **Filepath integrity** is verified (no Windows-illegal filenames; illegal-named files are excluded with documented reason).
3. **Composition-fingerprint cross-source dedup** is applied at the build layer, not silently at the source layer. The audit's 36 missing-on-disk PDMX files and the 50 / 2,268 / 30,301 cross-source collision counts from the v4 audit remain the reference for v4 source-vs-v4 source collisions; v5 collisions are reported at the same level of detail.
4. **Attribution requirement** is recorded in the per-row provenance manifest and surface-level summary for any CC BY 3.0 / CC BY 4.0 source.
5. **The local agent never mutates `C:\ov3`** and never overwrites a v3 / v4 build identity. v3 and v4 identities are immutable.

## v5 stop-line

A v5 build stops at the same 300M-event gate as v4. If v5 is below 300M after all available GREEN sources are admitted, the deficit is reported honestly and the local agent does **not** promote YELLOW sources to GREEN to close the gap. CocoChorales, the IMSLP CC-BY 3/4 expansion, CPDL, js_fake_chorales, musicnet_reference_midi, strudel_synth, Bach Doodle and the restricted Humdrum material remain HELD unless their YELLOW / RED status is independently re-litigated in this audit file.

## Local-agent hand-off rule

The local agent may implement / download / build only the sources explicitly marked GREEN above. The IMSLP CC-BY 3/4 expansion, CocoChorales, js_fake_chorales, CPDL, musicnet_reference_midi, and strudel_synth are not yet GREEN and may not be added to a production registry by this branch.

If a source-specific audit changes a status, update this file and preserve the evidence before changing the production registry. This file lives on the `audit/commercial-v5-source-census` branch and is never merged to `main`.
