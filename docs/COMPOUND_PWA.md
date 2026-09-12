# Compound Web/PWA streaming application

## Product boundary

The Compound page is intended to become the installable local application for the immutable A2-512 Base and compatible reviewed LoRA variants. The Web/PWA path is separate from Base training: the Base checkpoint identity is not changed by this work.

The application has two generation modes:

- **Infinite stream** — generate small event batches into a bounded playback lookahead while carrying only the fixed-size Compound stream state. Full generated history is deliberately not retained.
- **Finite MIDI export** — generate a bounded record sequence, preview it, and export Standard MIDI.

Stopping an infinite stream discards the carried state. The previous musical history cannot be reconstructed from the compressed stream state unless a separate recorder retained it.

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

`CompoundStreamSession` keeps the native stream state and current decoder context but no unbounded record list. The UI produces more records only when the scheduled audio lookahead falls below its bounded target. This is the operational meaning of continuous/infinite generation: model state and playback buffering remain bounded as elapsed listening time grows.

Variant changes are not hot-swapped into an existing stream. Changing Base/LoRA identity resets generation state because recurrent/stream state compatibility across independently exported variants is not assumed.

## PWA and offline storage

`manifest.webmanifest` and `sw.js` provide an installable application shell. The service worker caches the same-origin runtime shell and ONNX Runtime Web dependencies. Model graphs are stored separately in Cache Storage only after their configured SHA-256 values are verified.

The explicit **Download for offline use** action:

1. requests persistent browser storage when supported;
2. caches the pinned ONNX Runtime Web assets;
3. downloads the selected stream/decoder graph pair;
4. verifies both configured SHA-256 values before persistence;
5. reuses those exact bytes for later inference when offline.

The browser may still evict site storage under platform pressure; PWA installation is not itself proof that model bytes remain resident forever. The UI exposes whether the currently selected model pair is cached.

## Publication gate

The application remains fail-closed until `web/compound-runtime-config.json` contains at least one reviewed `available: true` model variant. Current source work must not invent ONNX URLs or mark redistribution complete.

A Base or `lora-premerged` variant must continue to satisfy the existing exact Base-SHA, architecture, tokenizer, runtime-ABI, URL and research-NC lineage checks before the UI enables generation.

## Remaining artifact gate

Source/PWA functionality can be merged independently of model publication. To make the public page actually generate A2-512 music, a later artifact release must provide the exact A2-512 V2 `stream.onnx` and `decoder_prefix.onnx` bytes, hashes, CORS-capable immutable URLs and native/Web parity evidence. LoRA variants repeat the same gate on their own exact merged bytes.
