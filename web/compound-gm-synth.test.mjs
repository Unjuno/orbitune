import assert from 'node:assert/strict';
import test from 'node:test';

import { CompoundEventType } from './compound-runtime.mjs';
import {
  annotateNotesWithGmState,
  createGmChannelState,
  gmProgramFamily,
} from './compound-gm-synth.mjs';

test('GM program families cover the 128-program space', () => {
  assert.equal(gmProgramFamily(0), 'piano');
  assert.equal(gmProgramFamily(24), 'guitar');
  assert.equal(gmProgramFamily(32), 'bass');
  assert.equal(gmProgramFamily(40), 'strings');
  assert.equal(gmProgramFamily(56), 'brass');
  assert.equal(gmProgramFamily(80), 'synth-lead');
  assert.equal(gmProgramFamily(127), 'sound-fx');
});

test('PROGRAM and BANK state annotate later notes on the same channel', () => {
  const resolved = annotateNotesWithGmState([
    { type: CompoundEventType.BANK, step: 0, channel: 2, a1: 1, a2: 4, a3: 0, a4: 0 },
    { type: CompoundEventType.PROGRAM, step: 0, channel: 2, a1: 32, a2: 0, a3: 0, a4: 0 },
    { type: CompoundEventType.NOTE, step: 0, channel: 2, a1: 40, a2: 96, a3: 100, a4: 0 },
  ]);
  assert.equal(resolved.notes.length, 1);
  assert.equal(resolved.notes[0].program, 32);
  assert.equal(resolved.notes[0].bankMsb, 1);
  assert.equal(resolved.notes[0].bankLsb, 4);
  assert.equal(resolved.notes[0].family, 'bass');
  assert.equal(resolved.notes[0].percussion, false);
});

test('channel 10 is rendered as percussion independent of melodic program state', () => {
  const resolved = annotateNotesWithGmState([
    { type: CompoundEventType.PROGRAM, step: 0, channel: 9, a1: 48, a2: 0, a3: 0, a4: 0 },
    { type: CompoundEventType.NOTE, step: 0, channel: 9, a1: 38, a2: 24, a3: 110, a4: 0 },
  ]);
  assert.equal(resolved.notes[0].percussion, true);
  assert.equal(resolved.notes[0].family, 'drums');
});

test('channel state can be carried across bounded streaming chunks', () => {
  const first = annotateNotesWithGmState([
    { type: CompoundEventType.PROGRAM, step: 0, channel: 1, a1: 40, a2: 0, a3: 0, a4: 0 },
  ], { initialState: createGmChannelState() });
  const second = annotateNotesWithGmState([
    { type: CompoundEventType.NOTE, step: 48, channel: 1, a1: 67, a2: 48, a3: 90, a4: 0 },
  ], { initialState: first.finalState });
  assert.equal(second.notes[0].program, 40);
  assert.equal(second.notes[0].family, 'strings');
});
