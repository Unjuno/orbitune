# Orbitune Post-Training Research

Status: **UNDEFINED / EXPERIMENT-GATED**

Orbitune does not currently assume that preference optimization, reinforcement learning, or other policy-learning stages are mandatory. The Base is first and foremost a pretrained symbolic-MIDI continuation model. Post-training is only justified if measured rollout behavior exposes a gap that cannot be solved more cleanly through representation, data quality, sampling, memory policy, or lightweight supervised adaptation.

## Why this remains open

Recent symbolic-music work gives evidence in both directions:

- NotaGen (IJCAI 2025) reports gains from a pipeline combining pretraining, high-quality fine-tuning, and CLaMP-DPO.
- A 2026 on-device piano autocomplete project reports large pairwise-preference gains after DPO, with continuation quality used as the primary preference criterion rather than isolated musical quality.
- SMART (2025) reports that reward optimization can improve subjective ratings, but aggressive reward optimization can sharply reduce diversity.
- Other symbolic-music RL experiments report sensitivity to reward design, prompt distribution, and training stability.

Therefore the engineering question is not "which alignment algorithm should Orbitune use?" but "does Orbitune exhibit a measurable post-training problem that requires one?"

## Training-stage taxonomy

Orbitune separates four stages. Only stage 1 is mandatory today.

1. **Base pretraining — REQUIRED**
   - Objective: next Compound Event / attribute prediction on quality-gated MIDI.
   - Goal: general symbolic music modeling and continuation capability.

2. **High-quality supervised fine-tuning — OPTIONAL / GATED**
   - Objective: train on a smaller, cleaner or task-specific subset.
   - Trigger: Base has broad capability but continuation quality, conditioning behavior, or event discipline improves reliably on a high-quality supervised subset.

3. **Preference optimization (e.g. DPO) — OPTIONAL / GATED**
   - Objective: prefer better rollouts without defining a differentiable scalar reward.
   - Trigger: pairwise judgments are reliable and the Base/SFT model systematically knows music but chooses worse continuations.

4. **Reward-based policy optimization — RESEARCH ONLY**
   - Objective: optimize explicit symbolic/audio-domain reward signals.
   - Trigger: reward validity is demonstrated against held-out human judgments and diversity collapse is controlled.

LoRA style adapters are orthogonal to this taxonomy. They may use supervised adapter training without implying that the shared Base requires policy optimization.

## Decision experiment P0 — Is post-training needed at all?

### H
The pretrained Base already produces continuation quality close enough to a high-quality SFT candidate that policy learning is unnecessary.

### T
Use a fixed held-out prompt set stratified by event density, instrumentation, tempo, harmonic regime, and prompt length. Compare:

- pretrained Base;
- high-quality SFT checkpoint with the same architecture;
- optional preference-optimized checkpoint only after the first comparison.

For each prompt generate multiple continuations under matched sampling settings. Evaluate both automatic rollout metrics and blinded pairwise judgments.

Primary pairwise criteria must be separated:

1. **continuation fit** — does the output follow the prompt naturally?
2. **standalone musical quality** — does the continuation sound coherent in isolation?
3. **diversity/non-collapse** — does the model retain multiple plausible continuations?
4. **control adherence** — only when ControlField conditioning is present.

### D
- **PASS / no post-training required:** SFT preference win rate over Base is <= 55% with confidence interval crossing 50%, and no clinically relevant improvement in rollout pathologies.
- **FAIL / post-training candidate:** SFT wins >= 60% on continuation fit on held-out prompts with no material diversity regression.
- **UNCERTAIN:** 55–60% or evaluator disagreement/seed variance is comparable to the model difference.

Thresholds are provisional and must be frozen before the production experiment.

### C
Apparent SFT/DPO gains may actually come from better data curation, sampler changes, or evaluator bias rather than a need for policy learning.

### U
Major uncertainty sources: prompt distribution, listening evaluator reliability, renderer/instrument choice, stochastic sampling variance, preference-model bias, and duplicate/near-duplicate training leakage.

## Decision experiment P1 — DPO only if P0 fails

If supervised post-training gives a meaningful gain but rollout selection still appears suboptimal, construct pairwise preference data from multiple samples per prompt.

Requirements:

- mirror A/B presentation to reduce position bias;
- keep continuation-fit and standalone-quality labels separate;
- maintain a consensus subset for noisy automated judges;
- reserve human-rated prompts that are never used to generate preference training pairs;
- sweep conservative regularization strengths rather than assuming a single DPO beta;
- compare against an SFT-only checkpoint with identical sampling settings.

DPO is accepted only if held-out pairwise quality improves while pitch/rhythm entropy, novelty, and inter-sample diversity remain inside predefined bounds.

## Decision experiment P2 — Reward optimization only if reward validity is proven

Audio-domain or symbolic rewards are not allowed to become training objectives merely because they correlate with intuitive quality metrics. Before RL/GRPO-style optimization:

1. establish correlation with held-out human pairwise judgments;
2. adversarially search for reward hacking examples;
3. set diversity and validity stop conditions;
4. run short-horizon optimization first;
5. reject any reward that improves its scalar score while degrading blinded listening preference.

## Roadmap interaction

Post-training research must not block the current critical path:

```text
Production Compound Tokenizer
→ provenance-gated real MIDI corpus
→ Base pretraining
→ real-MIDI scale/runtime/long-memory/control validation
→ Base rollout evaluation
→ P0: determine whether post-training is needed
→ optional SFT
→ optional DPO
→ RL only after reward validation
```

The roadmap is deliberately adaptive. If Base pretraining plus data curation already meets continuation, diversity, control, and long-rollout requirements, the post-training branch closes with **NOT REQUIRED** and no policy-learning machinery is added.
