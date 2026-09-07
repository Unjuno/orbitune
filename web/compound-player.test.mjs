import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPreviewSchedule } from './compound-player.mjs';

test('preview matches Python MIDI ordering for same-step tempo changes', () => {
  const schedule = buildPreviewSchedule([
    { type: 4, step: 0, channel: 0, a1: 60, a2: 0, a3: 0, a4: 0 },
    { type: 4, step: 0, channel: 0, a1: 120, a2: 0, a3: 0, a4: 0 },
    { type: 0, step: 0, channel: 0, a1: 60, a2: 96, a3: 100, a4: 0 }
  ]);
  // write_compound_midi sorts the 120 BPM tempo message first and the 60 BPM
  // message last at tick 0, so the note lasts one second in the preview too.
  assert.equal(schedule.length, 1);
  assert.equal(schedule[0].start, 0);
  assert.equal(schedule[0].end, 1);
});
