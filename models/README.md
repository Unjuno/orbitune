# Models

This directory holds research documentation. It is not the legacy `bases/` deployment registry and does not contain the documented trained weights. Candidate checkpoints may be kept locally under `models/`; their binaries remain ignored. Legacy Base and Adapter contributions follow their existing, separately reviewed artifact paths.

| Model | Availability | Use policy |
| --- | --- | --- |
| [Research-NC Aria+GigaMIDI V1](research_nc_aria_gigamidi_v1/README.md) | Documentation only; checkpoint download not published in this repository | Research / noncommercial |

The model card records the Commercial -> Aria -> Aria+GigaMIDI ancestry. An ancestor with a different use policy does not make a research descendant commercially eligible.

For this model, `publication.json` is a schema-validated availability record. `manifest.json` is an unchanged historical training report, not a public Base-registry manifest. See [publication status](../docs/PUBLICATION.md) before preparing a release.
