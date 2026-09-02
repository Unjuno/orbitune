"""Corpus sanity check for the prepared real-MIDI JSONL split."""
import json
from pathlib import Path

train_path = Path('data/real_midi/train.jsonl')
val_path = Path('data/real_midi/val.jsonl')

def stats(p):
    songs = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            songs.append(json.loads(line))
    n = len(songs)
    total = sum(len(s['records']) for s in songs)
    lens = sorted(len(s['records']) for s in songs)
    return n, total, lens[0], lens[len(lens) // 2], lens[-1], [s['sha256'] for s in songs]

tn, tt, tmin, tmed, tmax, tshas = stats(train_path)
vn, vt, vmin, vmed, vmax, vshas = stats(val_path)
overlap = set(tshas) & set(vshas)
print(f'train: songs={tn} events={tt} min={tmin} median={tmed} max={tmax}')
print(f'val:   songs={vn} events={vt} min={vmin} median={vmed} max={vmax}')
print(f'train/val SHA overlap: {len(overlap)} (must be 0)')

short_train = sum(1 for line in train_path.read_text(encoding='utf-8').splitlines()
                  if line.strip() and len(json.loads(line)['records']) < 257)
short_val = sum(1 for line in val_path.read_text(encoding='utf-8').splitlines()
                if line.strip() and len(json.loads(line)['records']) < 257)
print(f'train songs with < 257 events: {short_train} / {tn}')
print(f'val   songs with < 257 events: {short_val} / {vn}')

# Confirm ABI fields
first = json.loads(train_path.read_text(encoding='utf-8').splitlines()[0])
print(f"first train row tokenizer_abi={first.get('tokenizer_abi')} record_width={first.get('record_width')}")
