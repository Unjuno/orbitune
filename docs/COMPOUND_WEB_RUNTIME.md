# Compound browser runtime

## Status

Orbitune now has a browser-side runtime contract for the native-stream Compound architecture, but **the research model binaries are not published by this repository**. The checked-in Web code is therefore useful for ABI review, tests, and future deployment; it does not make the documented research checkpoint downloadable.

Current publication boundaries remain in [PUBLICATION.md](PUBLICATION.md). The model remains research-NC/noncommercial, and redistribution review is still pending.

## Native-generation V2 ABI

The production browser contract is a two-graph design:

1. **Stream graph**: accepted 12-field Compound record + tensorized stream state → context vector `[1,224]` + updated state.
2. **Decoder-prefix graph**: context + the already sampled intra-event prefix → all decoder heads over the fixed eight slot stages.

Generation executes one stream call and eight decoder-prefix calls per new Compound event. Categorical top-p sampling, Gaussian sampling, event-type masks, quantization, record assembly, and MIDI conversion run in JavaScript.

The machine-readable interface is [compound-inference-contract-v2.json](../web/compound-inference-contract-v2.json).

The older fixed-S64 teacher-forced export must not be used as the browser generation ABI. Local handoff testing found that it reset native hierarchical/recurrent context and teacher-forced later intra-event slots. The V2 handoff instead reported native stream parity over 512 records, exact native-vs-Web greedy record parity over 512 generated events, and a successful onnxruntime-web/WASM smoke. Those are locally reported validation results; the ONNX files and the private checkpoint are not bundled as public evidence here.

## Bit-exact JavaScript rules

The runtime intentionally mirrors current Python semantics:

- Python `round()` uses half-to-even (banker's) rounding; `Math.round()` is not equivalent at `.5` boundaries.
- `quantize_time` selects the first coarse interval whose upper edge is greater than or equal to the integer value, then uses the same half-to-even residual rounding.
- `quantize_unsigned` uses the current normalized coarse/residual formula.
- The intra-event decoder receives sampled normalized delta/velocity/duration values before quantization.
- TEMPO and TIME_SIGNATURE force channel 0; event-type-specific `a1`/`a2` masks and `_build_record` zeroing remain authoritative.

`web/compound-runtime.test.mjs` covers these contracts without requiring model files.

## Browser files

- `web/compound-runtime.mjs` — V2 stream/decoder orchestration and sampling.
- `web/compound-midi.mjs` — Compound record decode, canonicalization, and Standard MIDI serialization.
- `web/compound-player.mjs` — lightweight WebAudio note preview.
- `web/compound-variant.mjs` — fail-closed release, ABI, Base-SHA, URL, and variant validation.
- `web/compound.html` / `web/compound-app.mjs` — separate Compound UI.
- `web/compound-runtime-config.json` — publication/variant state. It intentionally contains no model URLs while redistribution remains unresolved.

The legacy `web/orbitune-runtime.mjs` remains Theory-REMI-specific and is not reused for Compound generation.

## Model and LoRA variants

The Compound page treats a deployable entry as a **variant** containing a matched stream graph and decoder-prefix graph. Initially, LoRA should use pre-merged model variants rather than the legacy dynamic Theory-REMI adapter ABI.

A future available variant must declare at least the following contract:

```json
{
  "id": "research-nc-aria-gigamidi-v1-web",
  "display_name": "Orbitune Research-NC Aria+GigaMIDI V1",
  "kind": "base",
  "available": true,
  "architecture": "orbitune-compound-hierarchical-gpt-v1",
  "tokenizer": "orbitune-compound-v0-experimental",
  "runtime_abi": "native-stream-state+decoder-prefix-v2",
  "base_checkpoint_sha256": "<exact frozen Base checkpoint SHA-256>",
  "stream": {"url": "https://.../stream.onnx", "sha256": "<64 hex>"},
  "decoder": {"url": "https://.../decoder_prefix.onnx", "sha256": "<64 hex>"},
  "execution_providers": ["wasm"]
}
```

A `kind: "lora-premerged"` variant additionally requires an `adapter_id`. The runtime rejects a variant whose architecture, tokenizer, runtime ABI, or Base checkpoint SHA does not exactly match the top-level model binding. Executable/insecure URL schemes are rejected; currently the validated execution provider is WASM only.

The release gate is also fail-closed: an `available: true` variant is invalid unless `redistribution_review` is `completed` **and** `publication_status` is `runtime_model_published`. Conversely, the current unpublished status cannot expose an available variant. These are source-level safety gates, not a substitute for the separate legal/rights review itself.

No model variant is configured today. Once redistribution is explicitly approved, a separate release task can add verified URLs and SHA-256 values without committing large ONNX binaries to Git.

## Listening

After generation, Compound records are decoded using the tokenizer-equivalent factorized time/control rules, canonicalized, and serialized to Standard MIDI. The page offers:

- a lightweight oscillator-based WebAudio preview (`Play` / `Stop`), and
- the authoritative generated Standard MIDI download.

The preview is a convenience synth, not a claim about rendered audio quality or instrumentation fidelity.

## Tests

The existing Web workflow automatically runs all `web/*.test.mjs` files under Node 22:

```bash
node --test web/*.test.mjs
```

Tests cover the native numeric/mask/runtime contract, publication-safe empty configuration, variant/Base binding, release-gate behavior, URL scheme restrictions, MIDI conversion, and preview timing. Model-dependent WASM parity remains an artifact-release gate because the two ONNX files are deliberately absent from Git.
