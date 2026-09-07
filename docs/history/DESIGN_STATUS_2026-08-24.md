# Orbitune Design Status

This file records current engineering decisions separately from experiments. A decision may be ACCEPTED, CANDIDATE, REJECTED, or OPEN. See `docs/AUDIT_2026-08-23.md` for the latest cross-repository audit and unresolved risks.

## ACCEPTED

- Product concept: local/on-device symbolic MIDI generation with a shared Base model and distributable LoRA adapters.
- Representation direction: Hybrid Compound Events; one musical event consumes one Transformer step.
- Base architecture family: causal decoder-only Transformer.
- Stable extension boundaries: `LinearBackend`, `ControlField`, `MemoryPolicy`.
- NOTE intra-event decoder direction: lightweight autoregressive MLP cascade rather than GRU/mini-attention.
- MIDI ingestion canonicalization: deterministic merge of same-onset/channel/pitch duplicates and truncation on overlapping same-channel/same-pitch retrigger before tokenization.
- Same-step Compound ordering is semantic state-before-note: Time Signature → Tempo → Bank → Program → controls → NOTE.
- Continuous attributes: factorized coarse + residual prediction.
- Long-running generation requires sparse historical context in addition to a recent sliding window.
- Training data must pass provenance/license, parsing, quality, deduplication, and split gates before pretraining.
- Post-training/policy learning is **not mandatory by default**. It is an experiment-gated branch evaluated only after Base rollout quality is measured.
- Published Base ids are immutable compatibility lineages bound to exact checkpoint SHA-256 values.

## IMPLEMENTED EXPERIMENTAL ABI

- `orbitune.compound` defines `orbitune-compound-v0-experimental` primitives for Compound Event types, deterministic MIDI-1 canonicalization, 96/qn timing constants, 7-coarse + 16-residual timing factorization, and explicit same-step semantic ordering.
- `orbitune.compound_midi` is a parallel MIDI type-0/1 parser that preserves the current production-candidate musical event scope without changing legacy `read_midi()` behavior.
- `orbitune.quantization` implements shared 8-coarse + 8-residual unsigned quantization for CC values, pitch bend and pressure-like controls.
- `CompoundEventTokenizer` maps one canonical event to one 12-field training record. The record width is a single tokenizer-owned ABI constant.
- DELTA and NOTE duration are factorized; continuous controls use the shared factorized pair.
- `orbitune.compound_dataset` writes tokenizer-ABI-tagged, record-width-tagged, song-preserving train/validation JSONL splits. Exact MIDI byte duplicates are grouped by SHA-256 and cannot cross splits.
- `orbitune.compound_training` rejects ABI/record-width mismatches and malformed factor indices before sampling song-local next-event windows.
- `scripts/prepare_compound_corpus.py` exposes the experimental corpus preparation path without replacing legacy CLI commands.
- Scheduled `continuous-train.yml` is explicitly gated as **legacy Theory-REMI reference training** and cannot be armed by corpus files alone.
- This Compound ABI is intentionally experimental. It must not be used as an immutable public Base compatibility target until the blockers below are closed.

## CANDIDATE (reference settings)

- Reference Base size: ~10M parameters. Proxy scale sweep: 5.0M/9.8M/20.1M validation loss 0.711/0.509/0.320 at increasing CPU forward cost; 10M remains the mobile-oriented middle point until real-MIDI validation.
- Timing resolution: 96 steps per quarter note.
- DELTA/DURATION: 7 coarse ranges + 16 residual levels per attribute for values in the currently tested range. Synthetic and local-MIDI tests support this as a knee point, but the >1536-step case is unresolved.
- NOTE scheduled sampling: curriculum ending near 0.25.
- Control vector: ~6 dimensions, 16 scalar levels per dimension as reference. Synthetic ControlField proxy MSE: 4-level 0.010433; 8-level 0.001936; 16-level 0.000425; 32-level 0.000100.
- ControlField implementation: musical-time Adaptive Gaussian RBF, about 12 bases.
- Infinite-memory policy: recent dense window + deterministic long-range anchors (16/32/64/128/256 bars) plus optional randomized historical samples.
- Quantized deployment: packed ternary native kernel preferred if available, INT8 fallback, FP16 fallback.
- LoRA: runtime adapter over fixed Base; initial ranks 4-8 after Compound target modules are frozen.

## REJECTED / DOWNRANKED

- Mandatory 3M Base size.
- Pure flat event stream as the production representation (kept only as a baseline).
- Independent Compound heads without intra-event conditioning.
- GRU or mini-attention NOTE decoders for the current four-attribute NOTE schema; they increased cost without improving the proxy task.
- Token-position ControlField as the primary time coordinate.
- Training-time hard grammar masks combined with scheduled sampling.
- PyTorch STE ternary inference as a production runtime.
- Recent-only sliding context for infinite generation when long-range structural dependencies matter.
- Adding DPO/RL merely because those methods exist; post-training must first pass a necessity gate.
- Treating the historical hidden-240 ~3M configuration as the current reference ABI.

## OPEN / BLOCKING VALIDATION

1. **Long-time representation:** current experimental timing factorization silently clips values above 1536 steps. Official Compound training cannot freeze until long DELTA/DURATION are represented without silent truncation.
2. **Sequence boundaries:** define BOS/EOS/start-of-unconditional-generation and continuation-stop semantics.
3. **Field schema:** freeze event-specific active fields, cardinalities, loss masks, inference masks and intra-event conditioning order from a single source of truth.
4. **External real-MIDI timing validation:** validate 96/qn and 7+16 against nearby alternatives on the production-like corpus.
5. **Expressive pedal:** measure CC64 half-pedal distribution and decide binary versus continuous/factorized PEDAL representation.
6. **Semantic MIDI scope:** decide/document canonical BANK state, integer-BPM tempo precision, dangling-note repair/rejection, and intentionally ignored meta/SysEx/port semantics.
7. **Production training corpus:** finalize legal/provenance policy, quality gates and **near-duplicate/composition-aware** deduplication. Exact-byte SHA grouping is already implemented but is insufficient by itself.
8. **Compound Base model:** implement factorized embeddings, causal Transformer, event-conditioned lightweight cascade and masked losses; pass synthetic and tiny-real overfit tests.
9. **Real-MIDI 5M/10M/20M scale sweep** before permanently fixing Base size.
10. **Real-device/Web runtime benchmark** for INT8 vs packed ternary; current Theory-REMI ONNX ABI is not a Compound runtime ABI.
11. **Real-MIDI long-rollout comparison** of plain sliding vs anchored/dilated memory.
12. **Real-corpus ControlField quantization and control-adherence validation.**
13. **Compound manifest/Adapter ABI:** define structured architecture/runtime compatibility before community Compound Bases/Adapters are accepted.
14. **Post-training necessity gate after Base pretraining.** Compare Base against high-quality SFT using held-out continuation-fit, standalone quality, diversity, and control-adherence judgments. DPO is tested only if a real rollout-selection gap remains; reward-based RL is research-only until reward validity and anti-collapse conditions are demonstrated. See `docs/POST_TRAINING_RESEARCH.md`.

## Latest proxy results

### Compound tokenizer property test
Randomized heterogeneous event files of 32, 128, and 512 raw events achieved 100% canonicalized encode/decode roundtrip across 450 files in the tested range. At 512 events the average canonicalization drop was 0.0013% in the stress distribution. This result **does not cover time values that exceed the current 1536-step cap**.

### Model scale proxy
| Target | Params | Validation loss | CPU forward (batch=6, seq=32, 5 threads) |
| --- | ---: | ---: | ---: |
| ~5M | 5.016M | 0.7107 | 33.14 ms |
| ~10M | 9.775M | 0.5087 | 56.38 ms |
| ~20M | 20.095M | 0.3197 | 84.18 ms |

### Long-memory proxy
With a fixed 32-bar context budget and dependencies extending to 256 bars, true source inclusion was ~44.5% for plain sliding and 100% for anchored/dilated policies. Target motif visibility rose from ~68.1% to ~92-93%.

### Control quantization proxy
| Scheme | Bits / 6D vector | Field MSE |
| --- | ---: | ---: |
| 4 levels/dim | 12 | 0.010433 |
| 8 levels/dim | 18 | 0.001936 |
| 16 levels/dim | 24 | 0.000425 |
| 32 levels/dim | 30 | 0.000100 |

## Post-training evidence summary

Current external evidence does not justify making policy learning mandatory. Symbolic-music work reports gains from supervised/preference post-training in some settings, while reward optimization can also reduce diversity or exploit imperfect rewards. Orbitune therefore treats post-training as an empirical branch, not an architectural assumption.

The next critical path is to close the Compound ABI blockers and implement the Base model, then validate on external real MIDI. Additional architecture ideation and premature policy-learning infrastructure are lower priority.
