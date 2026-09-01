"""Split synthetic_compound.jsonl into train/val halves."""
from __future__ import annotations

import json
import random
from pathlib import Path

src = Path("data/continuous/synthetic_compound.jsonl")
records = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
rng = random.Random(0)
rng.shuffle(records)
half = len(records) // 2
train, val = records[:half], records[half:]

train_out = Path("data/continuous/synthetic_compound_train.jsonl")
val_out = Path("data/continuous/synthetic_compound_val.jsonl")
train_out.write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
val_out.write_text("\n".join(json.dumps(r) for r in val) + "\n", encoding="utf-8")
print(f"train songs: {len(train)}  val songs: {len(val)}")