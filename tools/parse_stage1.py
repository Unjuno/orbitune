"""Parse the stage1.log validation history."""
import re
import json
from pathlib import Path

raw = Path('runs/compound/stage1.log').read_bytes()
# PowerShell Tee-Object writes UTF-16 LE with BOM. Try UTF-16 first, then
# fall back to UTF-8.
try:
    text = raw.decode('utf-16')
except UnicodeDecodeError:
    text = raw.decode('utf-8', errors='replace')
print(f'log size: {len(text)} chars (encoding: utf-16 if first char was ASCII)')

# Find every "{" ..."validation_window_hash": "H"}"  sequence
pat = re.compile(r'\{[^{}]*?"validation_window_hash"\s*:\s*"([0-9a-f]+)"[^{}]*?\}')
rows = []
for m in pat.finditer(text):
    block = m.group(0)
    # collapse internal whitespace (CR/LF) from PowerShell Tee-Object
    block_compact = re.sub(r'\s+', ' ', block)
    try:
        obj = json.loads(block_compact)
        rows.append((int(obj['step']), float(obj['validation_loss']), obj['validation_window_hash']))
    except Exception as e:
        pass
print(f'parsed rows: {len(rows)}')
hashes = set(h for _, _, h in rows)
print(f'unique validation_window_hash values: {len(hashes)}')
for h in hashes:
    print(f'  {h}')

best = None
best_step = None
print()
print('step | val_loss | best?')
for s, v, _ in rows:
    if best is None or v < best:
        best = v
        best_step = s
        mark = ' *'
    else:
        mark = ''
    print(f'{s:5d} | {v:9.4f} |{mark}')
print()
if best is not None:
    print(f'best validation loss: {best:.4f} at step {best_step}')
    last = rows[-1]
    print(f'latest validation: step {last[0]} val {last[1]:.4f} (delta vs best: {last[1]-best:+.4f})')
