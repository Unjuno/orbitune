"""Aggregate training telemetry from the 2000-step log."""
import re
import json
from pathlib import Path

raw = Path('runs/compound/stage1.log').read_bytes()
text = raw.decode('utf-16')

# Each log line is a JSON object. They can contain nested {} (components, cuda, runtime).
# Use a non-greedy match that handles balanced braces.
def find_log_objects(text):
    out = []
    i = 0
    while i < len(text):
        j = text.find('{"components"', i)
        if j < 0:
            break
        depth = 0
        k = j
        while k < len(text):
            c = text[k]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    out.append(text[j:k+1])
                    i = k + 1
                    break
            k += 1
        else:
            break
    return out

raws = find_log_objects(text)
print(f'log objects: {len(raws)}')
parsed = []
for r in raws:
    try:
        obj = json.loads(r)
        parsed.append(obj)
    except Exception:
        pass
print(f'parsed objects: {len(parsed)}')

# Filter to per-step training log rows
train_rows = [p for p in parsed if 'events_per_sec' in p and 'grad_norm' in p]
print(f'train rows: {len(train_rows)}')
if train_rows:
    peak_ev = max(r['events_per_sec'] for r in train_rows)
    mean_ev = sum(r['events_per_sec'] for r in train_rows) / len(train_rows)
    mean_grad = sum(r['grad_norm'] for r in train_rows) / len(train_rows)
    max_grad = max(r['grad_norm'] for r in train_rows)
    print(f'mean events/sec: {mean_ev:.0f}')
    print(f'peak events/sec: {peak_ev:.0f}')
    print(f'mean grad_norm: {mean_grad:.3f}')
    print(f'max grad_norm: {max_grad:.3f}')
    print(f'\nfirst 3:')
    for r in train_rows[:3]:
        print(f'  step {r["step"]:4d}  events/sec={r["events_per_sec"]:.0f}  grad={r["grad_norm"]:.2f}')
    print(f'\nlast 3:')
    for r in train_rows[-3:]:
        print(f'  step {r["step"]:4d}  events/sec={r["events_per_sec"]:.0f}  grad={r["grad_norm"]:.2f}')

# Train loss at key steps
last_by_step = {}
for p in parsed:
    if 'loss' in p and 'step' in p:
        last_by_step[p['step']] = p['loss']
print('\nTrain loss at key steps:')
for target in (25, 100, 500, 1000, 1500, 1975, 2000):
    if target in last_by_step:
        print(f'  step {target:4d}  loss={last_by_step[target]:.4f}')

# Components at key steps
print('\nLoss components at key steps:')
seen = set()
for p in parsed:
    if 'components' in p and p['step'] in (100, 500, 1000, 1500, 2000) and p['step'] not in seen:
        seen.add(p['step'])
        c = p['components']
        print(f'  step {p["step"]:4d}  ' + '  '.join(f'{k}={v:.3f}' for k, v in sorted(c.items())))

# Validation rows
val_rows = [p for p in parsed if 'validation_loss' in p and 'validation_window_hash' in p]
print(f'\nvalidation rows: {len(val_rows)}')

# Elapsed time approx
if train_rows:
    total_steps = 2000
    bs = 144
    sl = 256
    total_events = total_steps * bs * sl
    approx_seconds = total_events / mean_ev
    print(f'\napprox elapsed time: {approx_seconds/60:.1f} min ({approx_seconds:.0f} s)')

# CUDA telemetry from any parsed row that has cuda
cuda_rows = [p for p in parsed if 'cuda' in p and isinstance(p['cuda'], dict) and 'temperature_c' in p['cuda']]
if cuda_rows:
    peak_alloc = max(r['cuda'].get('peak_allocated_gib', 0) for r in cuda_rows)
    peak_res = max(r['cuda'].get('peak_reserved_gib', 0) for r in cuda_rows)
    max_util = max(r['cuda'].get('utilization', 0) for r in cuda_rows)
    max_temp = max(r['cuda'].get('temperature_c', 0) for r in cuda_rows)
    max_pow = max(r['cuda'].get('power_draw_watts', 0) for r in cuda_rows)
    print(f'\nCUDA telemetry (max across run):')
    print(f'  peak allocated VRAM: {peak_alloc:.2f} GB')
    print(f'  peak reserved VRAM: {peak_res:.2f} GB')
    print(f'  max utilization: {max_util}%')
    print(f'  max temperature: {max_temp} C')
    print(f'  max power: {max_pow} W')
