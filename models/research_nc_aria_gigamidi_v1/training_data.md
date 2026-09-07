# Training data and accounting

This card summarizes **reported local-run evidence**. Dataset archives, source manifests, indexes, and per-shard audit files are not distributed here. The unchanged [historical manifest](manifest.json) is retained for provenance; it is not independent evidence that every source input was preserved.

## Sources and use policy

The run used retained Aria-MIDI and GigaMIDI data, with no commercial replay reported for the combined stage. Both are treated as restricted sources in the public availability record. Aria is recorded as CC-BY-NC-SA-4.0; the committed GigaMIDI summaries use the broader label "research-use only". Exact upstream revision/license evidence must accompany any subsequent weight/data release; this cleanup does not re-admit or download datasets.

## Reported final train corpus

| Metric | Value | Meaning |
| --- | ---: | --- |
| Indexed songs | 2,263,855 | Accepted train song entries |
| Indexed records | 4,071,401,228 | Serialized record count |
| Available active next-event pairs | 4,069,137,373 | Reported corpus-level target-pair capacity |
| Historical pre-index manifest songs | 2,642,127 | Input-stage count; not the accepted corpus |
| Historical pre-index manifest events | 4,716,612,230 | Different-stage metric; do not relabel as actual training exposure |

Subtracting the reported song count from the record count reproduces the reported target-pair count. That arithmetic check alone does not prove input completeness, lossless encoding, split isolation, or a completed sampler epoch. Counts are dimensionless integers (SI unit 1); songs, records, and active pairs are different counting units.

The earlier data card reports 815,984 Aria and 1,447,871 GigaMIDI indexed train songs. Its per-source train record values, 1,745,990,770 and 2,325,410,458, add to the combined record count. The historical model README instead listed 2,325,401,045 for GigaMIDI, a 9,413-record discrepancy. Source-level index evidence is required to resolve this discrepancy; the combined total is retained as a reported value rather than silently amending historical evidence.

## Exclusions and unresolved completeness

| Reported item | Count | Interpretation |
| --- | ---: | --- |
| `known_missing_input` | 366,056 | File absent during indexing; **not** proof of a legitimate quality exclusion |
| `parse_failure:OverflowError` | 12,342 | Reported parse/indexing failure; exact causes require per-file evidence |
| Sum of those rejection counters | 378,398 | Shard-report accounting total |
| Manifest-to-index song-count decrease | 378,272 | Difference between reported stage totals |
| Difference between rejection total and stage decrease | 126 | Bounded discrepancy; "dedup timing" is not independently proven here |

The 126 is a discrepancy, not a third rejection category to add. The earlier source card's total of 378,524 incorrectly added it to 378,398. Records reduced by 645,211,002 across the reported stages, but a per-file attribution is not available in the public evidence. Do not equate the pre-index event metric with indexed records without checking their definitions.

Prior reports described simultaneous index coordinators and source deletion. The label `known_missing_input` balances a counter but does not close that data-loss question. A completeness claim requires expected input IDs, retained IDs, supported exclusion decisions, and stable source/index identities. No shard rebuild or checkpoint mutation is performed by this documentation change.

## Representation and sampling

The run reports an existing Aria int32 index combined with GigaMIDI uint8 shards, using 12-field Compound records. The filename `records.i32` is not sufficient to determine dtype. A future reproduction must validate declared dtype, field ranges, offsets, and roundtrip/losslessness using the actual indexes; the public cleanup cannot perform that check without them.

`IndexedTensorSampler` samples with replacement. Sequence length changes song eligibility, and batch geometry changes random draws. Cross-geometry ordering equality is not a supported invariant. Corpus capacity must not be described as the number of distinct events learned.

The historic mixture entries are Aria weight 1.0, GigaMIDI weight 1.0, commercial replay 0.0. Equal configured weights do not by themselves prove equal event exposure; inspect the actual sampler and per-source draw counters.

## Deduplication and evaluation limits

The summaries report 11 Aria intra-source duplicates removed and zero exact normalized-fingerprint cross-source matches in the audited comparison. This is not evidence of zero composition overlap or no validation leakage. Public per-source final validation/test index evidence is not provided here; those totals should remain unverified rather than be invented.

See [reproducibility.md](reproducibility.md) and [publication status](../../docs/PUBLICATION.md) for the evidence still required before a stronger release claim.
