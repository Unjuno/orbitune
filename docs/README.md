# Documentation index

## Current user-facing documentation

- [Repository overview and installation](../README.md)
- [Models and availability](../models/README.md)
- [Research model card](../models/research_nc_aria_gigamidi_v1/README.md)
- [Generation and checkpoint verification](../models/research_nc_aria_gigamidi_v1/usage.md)
- [Training-data accounting](../models/research_nc_aria_gigamidi_v1/training_data.md)
- [Reproducibility and known limitations](../models/research_nc_aria_gigamidi_v1/reproducibility.md)
- [Compound Base architecture and CLI](COMPOUND_BASE.md)
- [Compound browser runtime status](COMPOUND_WEB_RUNTIME.md)
- [Compound LoRA / Adapter policy](COMPOUND_LORA_POLICY.md)
- [Publication status and release checklist](PUBLICATION.md)
- [Current roadmap](ROADMAP.md)

## Current developer entry points

- [Compound Base](COMPOUND_BASE.md)
- [Compound browser runtime](COMPOUND_WEB_RUNTIME.md)
- [Compound LoRA policy](COMPOUND_LORA_POLICY.md)
- [Contributing](../CONTRIBUTING.md)
- [Legacy Base contribution contract](../CONTRIBUTING_BASES.md)
- [Legacy/Compound Adapter contribution boundary](../CONTRIBUTING_ADAPTERS.md)
- [Security and safe artifact handling](../SECURITY.md)

## Historical and superseded engineering records

Audit, benchmark, experiment and handoff documents are retained because they identify particular revisions and evidence. Some older files describe Compound as unimplemented or the Web ABI as undefined; those statements are historical and are not the current project status.

In particular, older design/handoff snapshots should be read together with the current documents above. Do not use an old `NEXT`, `OPEN` or `P0` list as the present release checklist without verifying that the item is still open.

The current source-of-truth hierarchy for public readers is:

```text
README.md
→ docs/COMPOUND_BASE.md
→ docs/COMPOUND_WEB_RUNTIME.md
→ docs/COMPOUND_LORA_POLICY.md
→ docs/PUBLICATION.md
→ model-specific README/publication.json
```

Historical files remain in place rather than being silently rewritten or deleted. Their presence is evidence/history, not a claim that their project-status section is current.
