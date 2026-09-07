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
DONE  hierarchical Compound Base implementation
DONE  frozen Research-NC Aria+GigaMIDI V1 at step 100k
DONE  native V2 Compound Web runtime source + Pages deployment
NOW   long-run full-parameter Base continuation / evaluation work
NEXT  freeze the next selected Base, then define a new Compound Adapter ABI
GATE  model/ONNX publication remains subject to redistribution review
```

Historical experiments remain useful as evidence for decisions made at the time; they are not silently converted into evidence for later model versions.
