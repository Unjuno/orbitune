# RISM commercial-v4 exact local pass

Status: **LOCAL EXACT MEASUREMENT ONLY — NOT A PRODUCTION REGISTRY SOURCE**

This pass must run only after the clean commercial-v4 baseline has been built from `configs/pretrain_corpus_commercial_v4.json`. Its purpose is to measure RISM after exact conversion, intra-source normalized dedup, and cross-source normalized dedup against that completed baseline.

## Fixed source and policy

- RISM export: `archive/source-2026-08-01.xml.gz`
- SHA1: `69261a6a6d30fa28139147287b6fcc060fd78edc`
- export license: CC BY 3.0
- Verovio: exactly `6.3.0`
- admission evidence: MARC `031$p` exists, clef exists, and MARC `100$d` is a bounded historical person-date string whose latest year is `<= 1955`
- missing/unknown dates and birth-only open-ended dates fail closed
- anonymous/unknown-composer broadening is not enabled

The 1955 rule is the same conservative audit predicate used by the feasibility census. It is pinned by the exact runner and cannot be widened with a command-line option.

## Required baseline

Build the current GREEN v4 first and keep both its final `manifest.jsonl` and `build_report.json`. The exact runner refuses to perform cross-source dedup unless all of the following agree:

- registry name is exactly `orbitune-commercial-safe-v4`;
- build report contains exactly the v4 source set;
- `accepted_after_cross_dedup` equals the manifest row count;
- train/validation/test index metadata all carry the same SHA256 as the supplied manifest;
- every manifest row has a valid normalized fingerprint.

Typical Windows layout:

```powershell
$V4 = "C:\ov4"
$PY = ".\venv_cuda\Scripts\python.exe"
```

The baseline build remains the normal Orbitune build:

```powershell
& $PY scripts\build_pretrain_corpus.py `
  --config configs\pretrain_corpus_commercial_v4.json `
  --root $V4
```

Do not run RISM against a partial v4 manifest. Record the baseline manifest SHA256 and v4 corpus identity before interpreting the RISM result.

## Exact RISM command

Reuse the already pinned RISM archive when possible. Do not materialize 1.45M permanent MIDI files.

```powershell
& $PY tools\rism_exact.py `
  --archive "C:\ov4\audit\rism\source-2026-08-01.xml.gz" `
  --baseline-manifest "C:\ov4\manifest.jsonl" `
  --baseline-build-report "C:\ov4\build_report.json" `
  --workers 8 `
  --report "C:\ov4\audit\rism\rism_exact_report.json" `
  --entries-output "C:\ov4\audit\rism\rism_exact_retained.jsonl.gz"
```

Each worker owns its own Verovio toolkit and temporary MIDI path. Temporary MIDI is discarded when the worker exits. The retained JSONL is provenance/fingerprint metadata, not a MIDI corpus. SHA-256 dedup keys are held as 32-byte digests rather than hexadecimal strings to keep the full 1.45M-candidate pass memory-bounded; retained provenance remains human-readable hex. The compressed provenance stream uses low gzip compression so compression does not dominate conversion time.

For a smoke run only:

```powershell
& $PY tools\rism_exact.py `
  --archive "C:\ov4\audit\rism\source-2026-08-01.xml.gz" `
  --baseline-manifest "C:\ov4\manifest.jsonl" `
  --baseline-build-report "C:\ov4\build_report.json" `
  --workers 4 `
  --limit 1000
```

A run using `--limit` is marked `is_exact_full_export=false` and must never be used for production admission.

## Exact dedup semantics

The order is fixed:

1. exact `(clef, key signature, time signature, PAE)` fingerprint dedup before conversion;
2. PAE -> Verovio 6.3.0 -> temporary MIDI;
3. production `midi_fingerprints()` normalized fingerprint and `composition_fingerprint_v1` calculation;
4. normalized-fingerprint dedup inside RISM;
5. normalized-fingerprint cross-dedup against the completed v4 baseline manifest;
6. `composition_fingerprint_v1` is recorded for split grouping only and is **not** used to collapse materially distinct arrangements.

The runner preserves input order when collecting parallel worker results, so the representative retained for an intra-RISM normalized duplicate is deterministic.

## Required report fields

The exact report includes at least:

- `RISM_PAE_UNIQUE`
- `RISM_CONVERSION_SUCCESS`
- `RISM_CONVERSION_FAILURE`
- `RISM_NORMALIZED_UNIQUE`
- `RISM_INTRA_SOURCE_DUPLICATES`
- `RISM_CROSS_V4_DUPLICATES`
- `RISM_RETAINED_AFTER_CROSS_DEDUP`
- `RISM_EXACT_ACTIVE_EVENTS_POST_DEDUP`
- Verovio warning/log counts and bounded diagnostic examples
- baseline manifest SHA256 and validated v4 build-report identity
- retained provenance JSONL SHA256

Retained rows carry RISM record id, person-date admission evidence, PAE fingerprint, rendered-MIDI SHA256, normalized fingerprint, composition fingerprint, active event count, source export SHA1, license, and Verovio version.

## Promotion gate

If and only if a **full** exact run retains a meaningful contribution after v4 cross-dedup (roughly more than 15M active next-event pairs), RISM becomes eligible for a separate production-source implementation. Eligibility is not automatic admission.

If promoted later:

- use a new additive registry revision;
- keep RISM in its own short-form tier;
- give it a lower sampling/quality weight than complete-score sources unless measurements justify otherwise;
- preserve RISM CC BY 3.0 attribution and exact export identity;
- do not use composition fingerprint as the duplicate criterion;
- rerun the complete final corpus build/census before training.

Until that review is complete:

- `TRAINING_STARTED = NO`
- `LR_CALIBRATION_STARTED = NO`
