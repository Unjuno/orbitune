# Compound Web/PWA streaming application

## Status

The public GitHub Pages product surface is the installable Compound PWA at `https://unjuno.github.io/orbitune/` (the root redirects to `compound.html`). The immutable A2-512 Base is published as a reviewed native-stream V2 browser variant. The legacy Theory-REMI page is retained separately at `legacy.html`.

The A2 Base checkpoint identity is unchanged by the Web release:

```text
model id        orbitune-a2-512-research-nc
checkpoint SHA  e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0
license         CC-BY-NC-SA-4.0
distribution    research-noncommercial
```

## Generation modes

- **Infinite stream** — generate small event batches into a bounded playback lookahead while carrying only the fixed-size Compound stream state. Full generated history is deliberately not retained.
- **Finite MIDI export** — generate a bounded record sequence, preview it, and export Standard MIDI.

Stopping an infinite stream discards the carried state. The previous musical history cannot be reconstructed from that compressed state unless a separate recorder retained it.

## Streaming architecture

```text
immutable Base / reviewed pre-merged LoRA
        ↓
stream.onnx + decoder_prefix.onnx
        ↓
CompoundStreamSession
        ↓ small batches
bounded WebAudio lookahead
        ↓
speakers
```

`CompoundStreamSession` keeps the native stream state and current decoder context but no unbounded record list. The UI generates more records only when scheduled audio lookahead falls below its bounded target. This is the operational meaning of continuous/infinite generation: model state and playback buffering remain bounded as elapsed listening time grows.

Variant changes are not hot-swapped into an existing stream. Changing Base/LoRA identity resets generation state because recurrent/stream state compatibility across independently exported variants is not assumed.

## Published A2 Web graphs

The reviewed Web release identity is recorded in `models/research_nc_aria_gigamidi_a2_512_v1/web_release.json`.

```text
stream.onnx
  bytes   27,241,782
  SHA256  27be3d6a4db726f52d7fc7e8df2e7a24be04ab5bd76da243bd8dd92712f2e107

decoder_prefix.onnx
  bytes   8,620,888
  SHA256  abb8326680222aff32d6b2fcb45356c748c523216cb0d75d6a755e615f419225
```

Large ONNX binaries are not committed to Git. On every main Pages deployment, CI downloads the pinned A2 checkpoint, verifies its SHA-256, deterministically re-exports the graph pair, requires exact equality with the reviewed graph hashes/sizes, reruns Python ONNX Runtime parity and exact `onnxruntime-web`/WASM greedy parity, and only then places the graph bytes in the Pages artifact under `models/a2-v1/`.

The deterministic 48-new-event greedy reference contains 49 records including the seed and has record-list SHA-256 `668972140fb8acaea2955453a882be7512209774b2daa2074649f1c8104b22a1`. The WebAssembly rollout must match those records exactly.

## Bounded performance evidence

Two independent GitHub-hosted Ubuntu x86_64 export/validation runs using Node 22.23.2, `onnxruntime-web` 1.29.0, WASM, one thread, and the 48-new-event greedy protocol observed approximately 19.317–22.978 generated events/s. This is CI evidence only; it is not a guarantee for phones, tablets, Safari, or every browser/device combination.

## PWA and offline storage

`manifest.webmanifest` and `sw.js` provide the installable application shell. The service worker caches same-origin shell assets and pinned ONNX Runtime Web dependencies. Model graphs are deliberately excluded from the shell cache.

The explicit **Download for offline use** action:

1. requests persistent browser storage when supported;
2. caches pinned ONNX Runtime Web assets;
3. downloads the selected stream/decoder graph pair;
4. verifies both configured SHA-256 values before persistence;
5. stores verified graph bytes in the dedicated model cache for later offline inference.

Browser storage can still be evicted under platform pressure; installing the PWA does not itself guarantee indefinite model-byte retention. The UI reports whether the selected model pair is cached.

## Source config versus deployed config

The checked-in `web/compound-runtime-config.json` intentionally remains fail-closed: it binds to A2 but has no available variants. Pull requests and source checkouts therefore cannot accidentally claim a model publication.

The main Pages build generates the public runtime config only after the exact reviewed graph bytes have been reproduced and validated. The deployed config has `publication_status=runtime_model_published`, `redistribution_review=completed`, and one available A2 Base variant.

## LoRA status

The PWA and variant contract already support `kind: "lora-premerged"`, but no production Compound LoRA artifact is claimed by the A2 Base Web release. Each future LoRA variant must be bound to the exact A2 Base and repeat the same merge → V2 export → exact-byte hash → native/ORT/WASM parity gate before it appears in the selector.
