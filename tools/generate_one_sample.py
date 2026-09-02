"""Generate one MIDI sample from a checkpoint and report its structure."""
import sys
import os
from collections import Counter
from orbitune.compound_midi import read_compound_midi
from orbitune.compound_base import CompoundHierarchicalGPT, write_compound_midi
from orbitune.tokenizer.compound_event import CompoundEventTokenizer
import torch

ckpt_path = sys.argv[1]
out_path = sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(ckpt_path)

model, payload = CompoundHierarchicalGPT.load_checkpoint(ckpt_path, map_location='cuda')
model.to('cuda').eval()
tokenizer = CompoundEventTokenizer()
torch.manual_seed(0)
records = model.generate_records(primer=[], max_new_events=256, temperature=1.0, top_p=0.9)
write_compound_midi(out_path, tokenizer.decode_records(records))

print(f'=== {label}  ({out_path}) ===')
print(f'  size: {os.path.getsize(out_path)} bytes')
ev = read_compound_midi(out_path)
print(f'  parsed events: {len(ev)}')
types = Counter(int(e.type) for e in ev)
TYPE_NAMES = {0: 'NOTE', 1: 'CC', 2: 'PROGRAM', 3: 'BANK', 4: 'TEMPO',
              5: 'PEDAL', 6: 'PITCH_BEND', 7: 'CHANNEL_PRESSURE',
              8: 'POLY_PRESSURE', 9: 'TIME_SIGNATURE'}
for t, c in sorted(types.items()):
    print(f'    {TYPE_NAMES.get(t, t):14s} {c}')
notes = [e for e in ev if int(e.type) == 0]
if notes:
    pitches = [e.a1 for e in notes]
    pc = Counter(pitches)
    mc, mcc = pc.most_common(1)[0]
    print(f'  NOTE count: {len(notes)}, most common pitch: {mc} ({mcc}/{len(notes)} = {mcc/len(notes)*100:.1f}%)')
    print(f'  unique pitches: {len(set(pitches))}, range: {min(pitches)}..{max(pitches)}')
    durs = [e.a2 for e in notes]
    print(f'  duration range: {min(durs)}..{max(durs)} ticks')
print()
