"""Quality sanity check for all generated samples."""
import os
from collections import Counter
from orbitune.compound_midi import read_compound_midi

TYPE_NAMES = {0: 'NOTE', 1: 'CC', 2: 'PROGRAM', 3: 'BANK', 4: 'TEMPO',
              5: 'PEDAL', 6: 'PITCH_BEND', 7: 'CHANNEL_PRESSURE',
              8: 'POLY_PRESSURE', 9: 'TIME_SIGNATURE'}

files = [
    ('runs/compound/samples/step-0500.mid', 'step-0500'),
    ('runs/compound/samples/step-1000.mid', 'step-1000'),
    ('runs/compound/samples/step-2000.mid', 'step-2000'),
    ('runs/compound/samples/best.mid',      'best (step-1900)'),
]

for path, label in files:
    print(f'=== {label}  ({path}) ===')
    if not os.path.exists(path):
        print('  MISSING')
        continue
    print(f'  size: {os.path.getsize(path)} bytes')
    try:
        events = read_compound_midi(path)
    except Exception as e:
        print(f'  parse FAILED: {e}')
        continue
    n = len(events)
    print(f'  parsed events: {n}')
    types = Counter(int(e.type) for e in events)
    for t, c in sorted(types.items()):
        print(f'    {TYPE_NAMES.get(t, t):14s} {c}')
    notes = [e for e in events if int(e.type) == 0]
    if notes:
        pitches = [e.a1 for e in notes]
        pc = Counter(pitches)
        mc, mcc = pc.most_common(1)[0]
        print(f'  NOTE count: {len(notes)}, most common pitch: {mc} ({mcc}/{len(notes)} = {mcc/len(notes)*100:.1f}%)')
        print(f'  unique pitches: {len(set(pitches))}, range: {min(pitches)}..{max(pitches)}')
        durs = [e.a2 for e in notes]
        print(f'  duration range: {min(durs)}..{max(durs)} ticks')
    # stuck note / collapse checks
    # A "stuck note" is a NOTE that is never paired with a NOTE_OFF (the
    # MIDI writer emits off messages per active note; the canonicalize step
    # truncates them at retrigger or end). We only check structure here.
    if len(types) < 2:
        print('  WARNING: type collapse (only one event type used)')
    if notes and len(set(pitches)) < 5:
        print('  WARNING: pitch collapse (fewer than 5 unique pitches)')
    # Delta sanity: every event has a non-negative delta.
    bad_delta = [e for e in events if int(e.type) == 0 and e.step < 0]
    if bad_delta:
        print(f'  WARNING: {len(bad_delta)} NOTE events with negative step')
    print()
