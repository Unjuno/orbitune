# Training Data Card

## `RESEARCH_NC_ARIA_GIGAMIDI_V1`

| Property | Value |
|----------|-------|
| Corpus ID | `RESEARCH_NC_ARIA_GIGAMIDI_V1` |
| Rights class | `RESEARCH_NC` |
| License policy | `research-nc` |
| Distribution scope | `noncommercial` |
| Commercial eligible | `false` |
| Corpus identity SHA-256 | `c354afc28617eb344ef9326fdeb2f199c5208ac9c6d3f82a96676d6900b9acc9` |
| Manifest | `C:\orbitune-research-nc\manifests\aria_gigamidi_manifest.jsonl` |
| Manifest SHA-256 | `3dbfdf7f4c2266ac9c880d199d718fa1abb86eb5ff6ba0ef1a88638893f17ee1` |
| Record dtype | `uint8` (width 12) |

---

## Data Sources

### Aria-MIDI

| Property | Value |
|----------|-------|
| Source ID | `aria_midi` |
| License | CC-BY-NC-SA-4.0 |
| Rights class | RESEARCH_NC |
| Total files scanned | 820,944 |
| Parse failures | 0 |
| Quality rejected | 0 |
| Intra-source dedup removed | 11 |
| Cross-commercial dedup removed | 0 |
| **Accepted after dedup** | **820,933** |
| Min events threshold | 8 |
| Manifest SHA-256 | `e2bdeb1b7b7b8424edf817d85171c51537a61b6a1076e5294ed80307e3815599` |

#### Splits

| Split | Songs | Events |
|-------|-------|--------|
| Train | 815,984 | 1,745,990,770 |
| Validation | 4,129 | 8,904,792 |
| Test | 820 | 1,800,058 |

### GigaMIDI

| Property | Value |
|----------|-------|
| Source ID | `gigamidi` |
| License | Research-use only |
| Rights class | RESEARCH_NC |
| Total parquet rows | 2,136,218 |
| Train shards | 366 (all verified done, 0 incomplete) |
| SHA failures | 0 |
| Unaccounted | 0 |

#### Splits

| Split | Songs | Events |
|-------|-------|--------|
| Train | 1,447,871 | 2,325,410,458 |
| Validation | ? | ? |
| Test | ? | ? |

#### GigaMIDI shard audit

| Metric | Value |
|--------|-------|
| Total shards | 366 |
| Verified done | 366 |
| SHA failures | 0 |
| Incomplete shards | 0 |
| Unaccounted total | 0 |

#### GigaMIDI rejection breakdown (train split)

| Rejection reason | Count |
|-----------------|-------|
| `known_missing_input` (file missing during processing) | 366,056 |
| `parse_failure:OverflowError` | 12,342 |
| Dedup timing discrepancy | 126 |
| **Total rejected** | **378,524** |

The 126-song discrepancy between the GigaMIDI manifest's train count (1,826,143) and the shard build's expected input (1,826,269) is a **bounded accounting difference**. Dedup timing is the leading attribution — files that entered shard building were not present in the final combined manifest split assignment — but this is **not independently proven** at the per-song level. The difference is below the 0.01% threshold of total corpus songs and all 378,398 shard-build rejections are fully explained (366,056 missing-input + 12,342 parse_failure). Across 366 shards, unaccounted = 0.

---

## Combined Corpus

### Source breakdown

| Source | Manifest Train Songs | Indexed Train Songs | Songs Lost | Events Lost |
|--------|---------------------|---------------------|------------|-------------|
| Aria-MIDI | 815,984 | 815,984 | 0 | 0 |
| GigaMIDI | 1,826,143 | 1,447,871 | 378,272 | 645,211,002 |
| **Combined** | **2,642,127** | **2,263,855** | **378,272** | **645,211,002** |

### Corpus identity verification

```
TRAIN_RECORDS (4,071,401,228) − TRAIN_SONGS (2,263,855) = 4,069,137,373
combined_research_train_1x_active_events (indexed) = 4,069,137,373
difference = 0  ✓ PASS
```

This identity holds as a corpus-level property: the sum of `(record_count − 1)` across all songs equals the total next-event training targets. It is independent of sampler type.

### Split distribution (combined)

| Split | Songs | Active Events |
|-------|-------|---------------|
| Train | 2,263,855 | 4,069,137,373 |
| Validation | ? | ? |
| Test | ? | ? |

### Sampling mixture

| Source | Weight |
|--------|--------|
| Aria | 1.0 |
| GigaMIDI | 1.0 |
| Commercial replay | 0.0 (disabled) |

### 50K census sample

| Metric | Value |
|--------|-------|
| Sample size | 50,000 |
| Parse OK | 50,000 |
| Parse failed | 0 |
| Quality accepted | 50,000 |
| Quality rejected | 0 |
| Parse rate | 100% |
| Active events total | 89,746,960 |
| Events/file (mean) | 1,794.94 |
| Events/file (median) | 523.0 |
| Events/file (p95) | 7,162.1 |
| Events/file (p99) | 15,215.05 |
| Intra-source normalized duplicates | 0 |
| Cross-commercial normalized duplicates | 0 |
| Cross-Aria normalized duplicates | 0 |
| Post-quality + dedup files | 50,000 |
| Post-quality + dedup active events | 89,746,960 |

---

## Data Quality Gates

- **Parse failures**: 0 across Aria-MIDI; 12,342 `OverflowError` in GigaMIDI (handled by rejection)
- **Quality rejection rate**: 0% (0 quality-rejected files across both sources in census)
- **Duplicate removal**: 11 intra-source Aria duplicates removed; 0 cross-source duplicates detected
- **Shard completeness**: All 366 GigaMIDI shards verified complete with SHA-256 match
- **Missing input files**: 366,056 files reported as missing during shard build (attributed to concurrent deletion by another coordinator process)

---

## Notes

- The `record_dtype` for GigaMIDI indexed training data is `uint8` (vs. `int32` for the Aria-only index). The combined sampler handles both transparently.
- Events lost (645,211,002) are attributable to rejected/missing songs but cannot be independently verified per-song because the rejected files were either missing from disk or failed to parse. The average of ~1,705 events/song is consistent with the census mean of ~1,794.94 events/file.
- No shard-level SHA failures; shard accounting is complete (0 unaccounted across 366 shards).
