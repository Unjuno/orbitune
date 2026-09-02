# Compound Results (tracked artifacts)

Small artifacts from the fixed-window Compound Base training run
(see `docs/COMPOUND_FINAL_REPORT.md` for the full report).

## Files

* `full_val_*.json` — output of `tools/full_validation_eval.py`
  * `total_events` = 369,664 Compound MIDI events
  * `total_scalar_fields` = 4,435,968 (= events × 12)
  * `mean_loss_per_event` — the trainer-style mean loss, sum of 7 head means divided by 7
  * `per_component.{name}.mean_per_event` — per-head mean
* `sample-*.mid` — generated MIDI samples (256 events each, seed 0, T=1.0, top_p=0.9)
  * `sample-step-0500.mid`, `sample-step-1000.mid`, `sample-step-2000.mid` — Stage 1 snapshots
  * `sample-step-1900-best.mid` — the SHIP checkpoint (full-val best)
  * `sample-step-3000.mid` — Stage 3 1e-4 final

## Re-generating

* Full validation: `tools/full_validation_eval.py --checkpoint <ckpt> --validation-jsonl data/real_midi/val.jsonl --seq-len 256 --batch-size 32 --device cuda --out <out.json>`
* Single sample: `tools/generate_one_sample.py <ckpt> <out.mid> [label]`

The MAESTRO 2004 JSONLs in `data/real_midi/` are gitignored (large); they are
generated from `maestro-v2.0.0-midi.zip` (also gitignored) by the corpus
preparation pipeline. The `runs/compound/*.pt` checkpoints are also gitignored.
