# Research-NC Aria+GigaMIDI V1

> **Orbitune Research-NC Aria+GigaMIDI V1** — a frozen, non-commercial Compound Hierarchical GPT model.

| Field | Value |
|-------|-------|
| **ID** | `research-nc-aria-gigamidi-v1` |
| **Architecture** | `orbitune-compound-hierarchical-gpt-v1` |
| **Tokenizer** | `orbitune-compound-v0-experimental` |
| **Parameters** | 8,857,250 |
| **Checkpoint SHA-256** | `8bf20a1198c4f5ee086ca13fd89521af6cfaa1fb28dd6602b1eb8f0b104629dc` |
| **Checkpoint (bytes)** | 106,583,212 (≈101.6 MiB) |
| **Frozen at step** | 100,000 |
| **Final val loss** | −1.1929529 |
| **Distribution** | Non-commercial only |
| **License** | CC-BY-NC-SA-4.0 (model output); training data: Aria-MIDI CC-BY-NC-SA-4.0 + GigaMIDI research-only |
| **Lineage** | COMMERCIAL_BASE_V1 → RESEARCH_NC_CKPT_ARIA_V1 → **RESEARCH_NC_ARIA_GIGAMIDI_V1** |

---

## Overview

This model is a **continuation** of the frozen Aria research-NC checkpoint, trained for 50,000 additional steps (from step 50,000 to step 100,000) on the combined Aria + GigaMIDI indexed corpus. It is frozen and immutable — no further training has been performed since the freeze.

The model uses Orbitune's Compound Transformer architecture, a hierarchical multi-scale transformer with local, medium, and global attention streams plus a recurrent memory module.

---

## Lineage Chain

```
COMMERCIAL_BASE_V1 (step 20,000)
    │  SHA256: 1c33e63b3e9e4207f1695fb4b235f9867da44cdf9dd63d74f00f882af59c8178
    │  Frozen: YES
    ↓
RESEARCH_NC_CKPT_ARIA_V1 (step 50,000)
    │  SHA256: fb0b86398cd84b0394b323cfe927ca442ac7a93bb3d1d5f1ae0fc4989f1e8369
    │  Frozen: YES
    ↓
RESEARCH_NC_ARIA_GIGAMIDI_V1 (step 100,000)  ← YOU ARE HERE
    │  SHA256: 8bf20a1198c4f5ee086ca13fd89521af6cfaa1fb28dd6602b1eb8f0b104629dc
    │  Frozen: YES (immutable)
```

### Training configuration

| Parameter | Value |
|-----------|-------|
| Seed | 3 (`random.Random(3 + 7919 = 7922)`) |
| Steps | 100,000 (50,000 new from step 50,000) |
| Batch size | 16 |
| Sequence length | 256 events |
| Precision | bfloat16 |
| Optimizer | Fused AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
| Config | `configs/compound_hierarchical_9m.json` |

### Model config (`configs/compound_hierarchical_9m.json`)

```json
{
  "d_model": 224,
  "n_head": 8,
  "local_layers": 4,
  "medium_layers": 2,
  "global_layers": 2,
  "intra_layers": 2,
  "ff_mult": 4,
  "dropout": 0.1,
  "local_window": 64,
  "medium_stride": 8,
  "medium_window": 64,
  "global_stride": 4,
  "global_window": 64,
  "fast_decay": 0.9,
  "medium_decay": 0.97,
  "slow_decay": 0.997
}
```

---

## Training Data

### Sources

| Source | License | Train Songs | Train Records |
|--------|---------|-------------|---------------|
| Aria-MIDI | CC-BY-NC-SA-4.0 | 815,984 | 1,745,990,770 |
| GigaMIDI | Research-use only | 1,447,871 | 2,325,401,045 (uint8 shards) |
| **Combined** | — | **2,263,855** | **4,071,401,228** |

### Corpus accounting

```
TRAIN_RECORDS (4,071,401,228) − TRAIN_SONGS (2,263,855) = 4,069,137,373
                                       = combined_research_train_1x_active_events (PASS)
```

- 378,272 GigaMIDI songs rejected during shard build: 366,056 missing-input + 12,342 parse_failure:OverflowError + 126 dedup-timing discrepancy
- 645,211,002 events lost from rejected/missing songs (attributable but not per-song verifiable)
- 366 training shards, all verified complete with SHA256 match; 0 unaccounted

See `training_data.md` for the full data card.

---

## Quality Validation

30 MIDI files generated (5 seeds × 2 temperatures × 3 model checkpoints). **All 30 are parse-valid.**

| Tier | Parse-Valid | Mean Notes | Mean Poly | Mean Events | Pitch Range |
|------|-------------|-----------|-----------|-------------|-------------|
| Commercial Base V1 | 10/10 | 258.9 | 4.1 | 498.7 | 33–93 |
| Aria research-NC V1 | 10/10 | 483.5 | 8.0 | 498.8 | 26–93 |
| **Aria+GigaMIDI V1 (this model)** | **10/10** | **443.7** | **3.8** | **500.0** | **33–89** |

Full results: `evidence/quality_validation_3way.json`

---

## Sampler / RNG Notes

- **Sampler type**: `IndexedTensorSampler` (random replacement, with replacement per draw)
- **RNG seed**: 7922 (`random.Random(3 + 7919)`)
- **Order invariance**: NOT order-invariant across batch sizes — batch size and seq_len affect the RNG stream by design
- **Known resume-fidelity bug**: During the 50,000→100,000 continuation, the local training RNG was not restored from checkpoint (the original code saved `sampler_rng_state=None`). Model weights, optimizer state, and validation plan are correct. The RNG fix has been implemented and applied prospectively.

See `reproducibility.md` for the full bug analysis and fix.

---

## Usage

```bash
# Generate a 512-event MIDI file
orbitune-compound generate \
  --checkpoint "C:\path\to\runs\research_nc_aria_gigamidi_v1\model.pt" \
  --out output.mid \
  --events 512 \
  --device cpu \
  --seed 100 \
  --temperature 0.85
```

See `usage.md` for full instructions including resume, priming, and GPU generation.

---

## License

- **Source code**: Apache-2.0
- **Model checkpoint**: CC-BY-NC-SA-4.0 (non-commercial use only)
- **Training data**: Aria-MIDI (CC-BY-NC-SA-4.0) + GigaMIDI (research-use only)
- **Generated output**: CC-BY-NC-SA-4.0

Commercial use is **prohibited**. See `LICENSE` for the full source-code license and `training_data.md` for data-license details.

---

## Evidence Files

| File | Description |
|------|-------------|
| `evidence/lineage_freeze_aria_gigamidi_v1.json` | Lineage chain and SHA256s |
| `evidence/index_freeze.json` | Corpus index accounting |
| `evidence/quality_validation_3way.json` | 30-file quality validation |
| `evidence/sampler_measurement_aria_gigamidi_v1.json` | Sampler RNG measurements |
| `evidence/sampler_resume_fix_audit_aria_gigamidi_v1.json` | RNG bug audit + fix |
| `evidence/gigamidi_train_audit.json` | 366-shard GigaMIDI audit |
| `evidence/combined_corpus_report.json` | Corpus composition report |
| `evidence/aria_midi_ingest_report.json` | Aria-MIDI ingest report |
