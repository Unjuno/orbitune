# Design status

> **Historical pointer:** the previous contents of this file were an engineering snapshot from 2026-08-24, before the hierarchical Compound Base was trained and before the native V2 browser runtime existed. The snapshot is preserved at [history/DESIGN_STATUS_2026-08-24.md](history/DESIGN_STATUS_2026-08-24.md).

Do not use the old snapshot's `OPEN`, `NEXT`, proxy architecture, or pre-implementation blocker list as current project status.

## Current sources of truth

- [Repository overview](../README.md)
- [Current architecture](ARCHITECTURE.md)
- [Compound Base](COMPOUND_BASE.md)
- [Current roadmap](ROADMAP.md)
- [Compound browser runtime](COMPOUND_WEB_RUNTIME.md)
- [Compound LoRA policy](COMPOUND_LORA_POLICY.md)
- [Publication status](PUBLICATION.md)
- [Research model card](../models/research_nc_aria_gigamidi_v1/README.md)

Current high-level state:

```text
DONE     hierarchical Compound Base implementation
DONE     frozen Research-NC Aria+GigaMIDI V1 at step 100k
DONE     native V2 Compound Base Web runtime source + Pages deployment
ACTIVE   local full-parameter Base continuation toward step 1M; latest recorded local evidence is in ROADMAP.md
THEN     freeze/evaluate a later Base candidate if produced
THEN     define and validate a new Compound Adapter ABI
GATE     model/ONNX publication remains subject to redistribution review
```

The active continuation is mutable local training state, not a public model release and not a public Adapter compatibility target. The historical 100k V1 identity remains frozen and unchanged.

The existing V2 parity evidence validates the Base browser ABI described by its handoff; it does not pre-validate a future LoRA-merged artifact.

Historical experiments remain useful as evidence for decisions made at the time; they are not silently converted into evidence for later model versions.
