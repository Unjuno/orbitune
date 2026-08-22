# Orbitune Design Status

This file records current engineering decisions separately from experiments. A decision may be ACCEPTED, CANDIDATE, REJECTED, or OPEN.

## ACCEPTED

- Product concept: local/on-device symbolic MIDI generation with a shared Base model and distributable LoRA adapters.
- Representation direction: Hybrid Compound Events; one musical event consumes one Transformer step.
- Base architecture family: causal decoder-only Transformer.
- Stable extension boundaries: `LinearBackend`, `ControlField`, `MemoryPolicy`.
- NOTE intra-event decoder direction: lightweight autoregressive MLP cascade rather than GRU/mini-attention.
- MIDI ingestion canonicalization: deterministic merge of same-onset/channel/pitch duplicates and truncation on overlapping same-channel/same-pitch retrigger before tokenization.
- Continuous attributes: factorized coarse + residual prediction.
- Long-running generation requires sparse historical context in addition to a recent sliding window.
- Training data must pass provenance/license, parsing, quality, deduplication, and split gates before pretraining.

## CANDIDATE (reference settings)

- Reference Base size: ~10M parameters. Proxy scale sweep: 5.0M/9.8M/20.1M validation loss 0.711/0.509/0.320 at increasing CPU forward cost; 10M remains the mobile-oriented middle point until real-MIDI validation.
- Timing resolution: 96 steps per quarter note.
- DELTA/DURATION: 7 coarse ranges + 16 residual levels per attribute (23-way total factorized vocabulary per time attribute). Synthetic and local-MIDI tests support this as the current knee point.
- NOTE scheduled sampling: curriculum ending near 0.25.
- Control vector: ~6 dimensions, 16 scalar levels per dimension as reference. Synthetic ControlField proxy MSE: 4-level 0.010433; 8-level 0.001936; 16-level 0.000425; 32-level 0.000100.
- ControlField implementation: musical-time Adaptive Gaussian RBF, about 12 bases.
- Infinite-memory policy: recent dense window + deterministic long-range anchors (16/32/64/128/256 bars) plus optional randomized historical samples.
- Quantized deployment: packed ternary native kernel preferred if available, INT8 fallback, FP16 fallback.
- LoRA: runtime adapter over fixed Base; initial ranks 4-8.

## REJECTED / DOWNRANKED

- Mandatory 3M Base size.
- Pure flat event stream as the production representation (kept only as a baseline).
- Independent compound heads without intra-event conditioning.
- GRU or mini-attention NOTE decoders for the current four-attribute NOTE schema; they increased cost without improving the proxy task.
- Token-position ControlField as the primary time coordinate.
- Training-time hard grammar masks combined with scheduled sampling.
- PyTorch STE ternary inference as a production runtime.
- Recent-only sliding context for infinite generation when long-range structural dependencies matter.

## OPEN / BLOCKING VALIDATION

1. External real-MIDI validation of 96/qn and 7+16 DELTA/DURATION quantization.
2. Production training corpus composition and legal/provenance policy. Strong commercial-compatible candidates include PDMX `no_license_conflict` and Slakh2100-redux; Groove/E-GMD is useful as a drum supplement. Lakh-derived corpora require extra copyright caution despite dataset-level license statements.
3. Real-MIDI 5M/10M/20M scale sweep before permanently fixing Base size.
4. Real-device/Web runtime benchmark for INT8 vs packed ternary.
5. Real-MIDI long-rollout comparison of plain sliding vs anchored/dilated memory.
6. Real-corpus ControlField quantization and control-adherence validation.

## Latest proxy results

### Compound tokenizer property test
Randomized heterogeneous event files of 32, 128, and 512 raw events achieved 100% canonicalized encode/decode roundtrip across 450 files. At 512 events the average canonicalization drop was 0.0013% in the stress distribution.

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

The next gating milestone is external real-MIDI validation, not additional architecture ideation.
