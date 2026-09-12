# Compound browser runtime

## Status

Orbitune's native-stream Compound V2 browser runtime is deployed on GitHub Pages with the immutable A2-512 Base. Large ONNX binaries are generated and validated during the main Pages build rather than committed to Git.

Public application:

```text
https://unjuno.github.io/orbitune/
```

The checked-in `web/compound-runtime-config.json` remains intentionally fail-closed for source/PR safety. The main Pages build generates the published runtime config only after reproducing exact reviewed graph bytes and rerunning parity validation.

A2 Web release identity is recorded in `models/research_nc_aria_gigamidi_a2_512_v1/web_release.json`. The Web graphs retain the Base's CC-BY-NC-SA-4.0 / research-noncommercial scope.

## Native-generation V2 ABI

The production browser contract is a two-graph design:

1. **Stream graph**: accepted 12-field Compound record + tensorized fixed-capacity stream state → context vector `[1,224]` + updated state.
2. **Decoder-prefix graph**: context + already sampled intra-event prefix → all decoder heads over the fixed eight slot stages.

Generation executes one stream call and eight decoder-prefix calls per new Compound event. JavaScript performs top-p/greedy sampling, Gaussian sampling, event-type masks, Python-compatible rounding, quantization, record assembly, MIDI conversion and streaming playback scheduling.

The machine-readable interface is [compound-inference-contract-v2.json](../web/compound-inference-contract-v2.json).

The older fixed-S64 teacher-forced export is not the browser generation ABI because it reset hierarchical/recurrent stream context and teacher-forced later intra-event slots.

## Published A2 graph identity

```text
Base checkpoint SHA-256
  e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0

stream.onnx
  bytes   27,241,782
  SHA256  27be3d6a4db726f52d7fc7e8df2e7a24be04ab5bd76da243bd8dd92712f2e107

decoder_prefix.onnx
  bytes   8,620,888
  SHA256  abb8326680222aff32d6b2fcb45356c748c523216cb0d75d6a755e615f419225
```

Two independent GitHub-hosted exports produced identical graph SHA-256 values. The reviewed deterministic greedy fixture contains the seed plus 48 generated records and has record-list SHA-256 `668972140fb8acaea2955453a882be7512209774b2daa2074649f1c8104b22a1`.

The release gate validates:

- exact A2 checkpoint SHA;
- tensor-wrapper/native streaming parity;
- tensor-wrapper/native decoder-prefix parity;
- Python ONNX Runtime parity;
- exact graph byte size + SHA against the reviewed release record;
- exact native-vs-`onnxruntime-web`/WASM greedy record equality.

The Pages deploy is skipped on any mismatch.

## Bit-exact JavaScript rules

The runtime intentionally mirrors Python semantics:

- Python `round()` is half-to-even; `Math.round()` is not used as an equivalent at `.5` boundaries.
- `quantize_time` and `quantize_unsigned` mirror the trained tokenizer rules.
- sampled normalized delta/velocity/duration values are fed back into later intra-event decoder slots before quantization.
- TEMPO and TIME_SIGNATURE force channel 0; event-specific `a1`/`a2` masks and record zeroing remain authoritative.
- Compound TEMPO remains `1..999 BPM`; Standard MIDI export explicitly rejects unrepresentable `1..3 BPM` in both Python and Web paths.

## Browser/PWA files

- `web/compound-runtime.mjs` — V2 stream/decoder orchestration and sampling.
- `web/compound-stream.mjs` — fixed-state continuous generation session.
- `web/compound-midi.mjs` — Compound decode/canonicalization/Standard MIDI serialization.
- `web/compound-player.mjs` — finite preview player.
- `web/compound-live-player.mjs` — bounded-lookahead live player.
- `web/compound-variant.mjs` — release, ABI, Base-SHA, URL and variant validation.
- `web/model-cache.mjs` — SHA-verified offline graph cache.
- `web/manifest.webmanifest`, `web/sw.js`, `web/pwa.mjs` — installable/offline application shell.
- `web/compound.html`, `web/compound-app.mjs` — current product UI.
- `web/legacy.html` — preserved Theory-REMI runtime.

## Base and LoRA variants

The deployed A2 Base is `kind: "base"`. The same runtime accepts future `kind: "lora-premerged"` variants, each of which must include an explicit `adapter_id`, bind to the exact A2 checkpoint SHA, and provide its own reviewed stream/decoder graph hashes.

The runtime does not hot-swap a different variant into an existing recurrent stream state. Selecting another Base/LoRA variant starts a new stream.

No production Compound LoRA artifact is claimed by the A2 Base Web release. A future LoRA release must repeat merge → V2 export → exact-byte identity → native/Python ORT/WASM parity before publication.

## Listening and continuous generation

Finite generation provides WebAudio preview and Standard MIDI download. Infinite-stream mode generates small batches only when bounded scheduled lookahead falls below its target. The full generated event history is not retained, so model/playback state does not grow with listening duration.

Stopping/resetting discards the compressed stream state. Past music is not reconstructable from that state unless separately recorded.

The WebAudio preview uses a lightweight oscillator synth and is not a claim of rendered instrument fidelity.

## Performance evidence

On two GitHub-hosted Ubuntu x86_64 runs using Node 22.23.2, `onnxruntime-web` 1.29.0, WASM and one thread, the 48-new-event greedy validation observed 19.317–22.978 generated events/s and model-load time 605–844 ms. This is bounded CI evidence, not a guarantee for mobile devices or every browser.

## Tests

Source Web tests run under Node 22:

```bash
node --test web/*.test.mjs
```

The separate `compound-web-export` workflow additionally validates the real A2 checkpoint and exported model artifacts. The main Pages workflow repeats the exact artifact gate before deployment.
