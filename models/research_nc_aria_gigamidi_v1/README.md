# Orbitune Research-NC Aria+GigaMIDI V1

**Availability: documentation only. Trained weights are not included in Git and no public download is recorded.** Compound ONNX/Web export is not published. Follow [usage.md](usage.md) only after obtaining the checkpoint independently from a trusted, authorized source.

## Reported model identity

| Field | Recorded value |
| --- | --- |
| Model ID | `research-nc-aria-gigamidi-v1` |
| Architecture | `orbitune-compound-hierarchical-gpt-v1` |
| Tokenizer | `orbitune-compound-v0-experimental` |
| Parameters | 8,857,250 |
| Final global step | 100,000 |
| Additional steps in the combined-data stage | 50,000, from parent step 50,000 |
| Checkpoint size | 106,583,212 bytes |
| Recorded checkpoint license | CC-BY-NC-SA-4.0 |
| Project use policy | Noncommercial / `research-nc`; not commercially eligible |
| Public Base-registry eligibility | No: experimental Compound ABI, no ONNX artifact, and checkpoint exceeds the existing 95 MiB Base-artifact limit |

Recorded checkpoint SHA-256:

```text
8bf20a1198c4f5ee086ca13fd89521af6cfaa1fb28dd6602b1eb8f0b104629dc
```

These are reported-run facts retained from [manifest.json](manifest.json). The checkpoint and original local audit files were not available to this publication cleanup for independent verification. [publication.json](publication.json) is a separate, validated availability record and does not change the checkpoint bytes or grant publication rights.

## Lineage

| Stage | Step | Recorded SHA-256 |
| --- | ---: | --- |
| Commercial Base V1 | 20,000 | `1c33e63b3e9e4207f1695fb4b235f9867da44cdf9dd63d74f00f882af59c8178` |
| Aria research-NC V1 | 50,000 | `fb0b86398cd84b0394b323cfe927ca442ac7a93bb3d1d5f1ae0fc4989f1e8369` |
| Aria+GigaMIDI research-NC V1 | 100,000 | `8bf20a1198c4f5ee086ca13fd89521af6cfaa1fb28dd6602b1eb8f0b104629dc` |

## Training scale and interpretation

The historical record reports 2,263,855 indexed train songs, 4,071,401,228 records, and 4,069,137,373 available next-event pairs. These are **corpus capacity**, not unique events consumed or an epoch-completion claim. The combined stage ran 50,000 additional steps with batch size 16 and sequence length 256; the checkpoint record carries a cumulative `events_seen` counter of 409,600,000. Neither counter establishes unique coverage under random-replacement sampling.

See the [data card](training_data.md) for missing-input exclusions, unresolved source-accounting details, and the distinction between source notes, indexed records, and sampler draws.

## Reported evaluation

The committed historical manifest summarizes 30 generated MIDI files across three checkpoints, five seeds, and two temperatures. All 30 were reported parse-valid. It reports mean note counts of 258.9 (Commercial), 483.5 (Aria), and 443.7 (combined), and final combined validation loss of approximately -1.193.

The underlying `quality_validation_3way.json` and generated samples are **local audit artifacts, not bundled public evidence**. MIDI parse validity is a technical smoke test, not proof of musical quality, expressive phrasing, generalization, or superiority. Losses on different validation corpora must not be directly ranked.

## Known limitations

- The frozen continuation has a documented sampler-local RNG restoration defect. Current code contains a prospective fix; the frozen run must not be described as bit-exact. See [reproducibility.md](reproducibility.md).
- Missing GigaMIDI inputs and a 126-song accounting discrepancy remain important qualifications. A hash matching a generated report does not establish completeness of the intended input set.
- Exact normalized-fingerprint nonmatches do not establish absence of composition overlap, near-duplicates, or train/validation leakage.
- No listening-panel benchmark, public weight download, or Compound browser deployment is established here.

## Rights

The project's research-NC restriction remains unchanged. The historical manifest declares CC-BY-NC-SA-4.0 for the checkpoint; this page is not a new relicensing decision. The [Apache-2.0 source-code license](../../LICENSE) does not supersede data or checkpoint terms. Do not assume every generated MIDI automatically inherits one uniform license: assess applicable rights and actual release terms separately. [Creative Commons' AI guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) explains the need to consider license applicability and conditions.

[Usage](usage.md) | [Data card](training_data.md) | [Reproducibility](reproducibility.md) | [Publication checklist](../../docs/PUBLICATION.md)
