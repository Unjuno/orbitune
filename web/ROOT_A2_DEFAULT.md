# Root GitHub Pages runtime

`web/index.html` is the production entry point for the latest reviewed non-commercial Compound Base runtime.

Current default:

- Model: `orbitune-a2-512-research-nc`
- Checkpoint SHA-256: `e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0`
- Browser runtime: `native-stream-state+decoder-prefix-v2`
- ONNX Runtime Web stable release: `1.29.0`

The root page must use `compound-app.mjs`; the legacy Theory-REMI `app.mjs` must not be the root runtime. `compound.html` remains a compatibility URL for existing links.

The Pages workflow still owns publication of the reviewed A2 ONNX graphs and overwrites the tracked fail-closed `compound-runtime-config.json` only inside the deployment artifact.
