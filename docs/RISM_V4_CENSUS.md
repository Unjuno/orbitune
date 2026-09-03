# RISM commercial-v4 census

Status: **PROMISING / LOCAL EXACT CENSUS REQUIRED**

This audit evaluates the RISM source MARCXML export as a short-form melodic-diversity supplement for the commercial Base corpus. It does **not** add RISM to the production corpus registry by itself.

## Immutable source used

- Export: `archive/source-2026-08-01.xml.gz`
- SHA1: `69261a6a6d30fa28139147287b6fcc060fd78edc`
- Export license: CC BY 3.0
- Musical incipit field: MARC `031$p` (Plaine & Easie), with `031$g/$n/$o` supplying clef/key-signature/time-signature context.
- Conversion audit: Verovio `6.3.0` -> MIDI -> Orbitune Compound tokenizer.

The census uses a deliberately conservative *audit* predicate for underlying-composition age evidence: the source record must contain bounded historical person dates in MARC `100$d` whose latest year is <= 1955. Missing/unknown dates and birth-only open-ended dates fail closed. This predicate is a feasibility gate, not the final production legal policy.

## CI census result

GitHub Actions run `33790195053` completed successfully.

- RISM source records: **1,619,445**
- Records with musical incipits: **1,169,323**
- Musical incipits (`031$p`): **2,107,304**
- Incipits with clef: **2,107,256**
- Incipits with person dates: **1,677,676**
- Conservative PD-safe incipits before exact-PAE dedup: **1,576,994**
- Conservative PD-safe unique incipits: **1,454,156**
- Exact duplicate incipits removed by `(clef, keysig, timesig, PAE)` fingerprint: **122,838**
- PD-safe unique incipits missing clef: **29**

The unique conservative subset is about **69.0%** of all exported musical incipits.

## Orbitune conversion sample

A deterministic reservoir sample of 5,000 conservative unique incipits was converted with Verovio and then parsed/tokenized by Orbitune.

- Attempted: **5,000**
- Parsed: **5,000**
- Failures: **0**
- Parse success rate: **100%**
- Mean active Compound next-event pairs per incipit: **16.5578**
- Median: **15**
- P90: **25**
- Min / max: **2 / 98**

Verovio emitted non-fatal warnings for some legacy/irregular Plaine & Easie encodings. Those warnings are a reason to retain conversion diagnostics in the exact build, but they did not produce a sample parse failure.

## Scale projection

Using the conservative unique count multiplied by the sampled mean Orbitune active-event count gives:

**~24,077,624 active events before Orbitune cross-source dedup**

This number is **not exact** and must not be inserted into the production corpus census as if measured. It is a feasibility estimate only.

The result is large enough to justify an exact local pass. RISM is still short-form material (incipits, not complete works), so it should remain a separate low/medium-weight diversity supplement rather than replacing full-piece sources.

## Required local exact pass before production admission

1. Reuse the pinned 2026-08-01 export and verify the SHA1.
2. Define and test the final fail-closed underlying-composition admission policy. Do not silently broaden the audit predicate.
3. Stream all admitted unique PAE incipits through a pinned Verovio version.
4. Record all conversion warnings/failures and reject failed/empty outputs.
5. Convert to Orbitune Compound and count exact records / active next-event pairs.
6. Cross-deduplicate against the complete commercial-v4 candidate corpus using Orbitune fingerprints.
7. Keep RISM provenance and CC BY attribution in the manifest/build report.
8. Keep RISM in a distinct source/tier so sampling weight can reflect that it is short-form melodic material.
9. Only after the exact post-dedup census should RISM be promoted into the production registry.

No LR calibration or production training is authorized by this audit.
