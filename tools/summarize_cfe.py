"""Summarize the initial CFE sweep."""
from __future__ import annotations

import json
from pathlib import Path

records = json.loads(Path("runs/cfe_initial.json").read_text())
results = records["results"]
safe = [r for r in results if r["status"] == "ok" and r["peak_reserved_fraction"] <= 0.92]
print(f"total results: {len(results)}  safe<=0.92: {len(safe)}")
print()
print("frontier (top 10 by events/sec, safe):")
frontier = sorted(safe, key=lambda r: (-r["events_per_sec"], r["peak_reserved_fraction"]))[:10]
print(f"{'rank':>4} {'n_head':>6} {'head_dim':>8} {'fast':>5} {'seq':>4} {'bs':>4} "
      f"{'events/s':>10} {'peak_res%':>10} {'loss':>8}")
for i, r in enumerate(frontier, 1):
    print(f"{i:>4} {r['n_head']:>6} {r['head_dim']:>8} {str(r['causal_fastpath']):>5} "
          f"{r['seq_len']:>4} {r['batch_size']:>4} {r['events_per_sec']:>10.1f} "
          f"{r['peak_reserved_fraction']*100:>9.1f}% {r['loss']:>8.4f}")

print()
print("over-budget or OOM rows (>0.92 reserved):")
over = [r for r in results if r["status"] == "ok" and r["peak_reserved_fraction"] > 0.92]
for r in over:
    print(f"  n_head={r['n_head']} head_dim={r['head_dim']} fast={r['causal_fastpath']} "
          f"seq={r['seq_len']} bs={r['batch_size']} ev/s={r['events_per_sec']:.1f} "
          f"peak_res={r['peak_reserved_fraction']*100:.1f}%")
oom = [r for r in results if r["status"] == "oom"]
for r in oom:
    print(f"  OOM: n_head={r['n_head']} head_dim={r['head_dim']} fast={r['causal_fastpath']} "
          f"seq={r['seq_len']} bs={r['batch_size']}")