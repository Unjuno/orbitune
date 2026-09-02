from orbitune.compound_midi import read_compound_midi
import os
events = read_compound_midi('runs/compound/real-smoke.mid')
print(f'parsed events: {len(events)}')
types = {}
for e in events:
    t = str(e.type).split('.')[-1]
    types[t] = types.get(t, 0) + 1
print('event types:', types)
note_count = sum(1 for e in events if 'NOTE' in str(e.type))
print(f'NOTE events: {note_count}')
print('file size:', os.path.getsize('runs/compound/real-smoke.mid'), 'bytes')

# stuck note / collapse checks
from collections import Counter
notes = [e for e in events if 'NOTE' in str(e.type)]
if notes:
    pitches = [e.a1 for e in notes]
    pcounts = Counter(pitches)
    most_common, mc_count = pcounts.most_common(1)[0]
    print(f'most common pitch: {most_common} ({mc_count}/{len(notes)} = {mc_count/len(notes)*100:.1f}%)')
    durations = [e.a2 for e in notes]
    if durations:
        dcounts = Counter(durations)
        md, mdc = dcounts.most_common(1)[0]
        print(f'most common duration: {md} ({mdc}/{len(notes)} = {mdc/len(notes)*100:.1f}%)')
    deltas = [e.a1 for e in events]  # not strictly delta, but check
    print(f'unique event types: {len(types)}')
