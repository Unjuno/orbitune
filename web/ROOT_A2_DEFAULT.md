# Root GitHub Pages runtime

`web/index.html` is the production entry point for the latest reviewed non-commercial Compound Base runtime.

Current default:

- Model: `orbitune-a2-512-research-nc`
- Checkpoint SHA-256: `25d43bd663501b5daf1b9708efe31706fd86a344f036ebe7be7a66df683eedda`
- Browser runtime: `native-stream-state+decoder-prefix-v2`
- ONNX Runtime Web stable release: `1.29.0`

The root page must use `compound-app.mjs`; the legacy Theory-REMI `app.mjs` must not be the root runtime. `compound.html` remains a compatibility URL for existing links.

The Pages workflow still owns publication of the reviewed A2 ONNX graphs and overwrites the tracked fail-closed `compound-runtime-config.json` only inside the deployment artifact.
