import assert from 'node:assert/strict';
import test from 'node:test';

import { buildStreamingChunkTiming } from './compound-live-player.mjs';
import { CompoundEventType } from './compound-runtime.mjs';

test('streaming chunk timing carries the effective tempo into the next chunk', () => {
  const first = buildStreamingChunkTiming([
    { type: CompoundEventType.TEMPO, step: 0, channel: 0, a1: 60, a2: 0, a3: 0, a4: 0 },
    { type: CompoundEventType.NOTE, step: 96, channel: 0, a1: 60, a2: 96, a3: 90, a4: 0 },
  ]);
  assert.equal(first.finalBpm, 60);
  assert.equal(first.spanSeconds, 1);
  assert.equal(first.notes[0].start, 1);

  const second = buildStreamingChunkTiming([
    { type: CompoundEventType.NOTE, step: 96, channel: 0, a1: 64, a2: 96, a3: 90, a4: 0 },
  ], { defaultBpm: first.finalBpm });
  assert.equal(second.notes[0].start, 1);
});

test('streaming chunk timing does not require full generated history', () => {
  const timing = buildStreamingChunkTiming([
    { type: CompoundEventType.NOTE, step: 48, channel: 0, a1: 67, a2: 48, a3: 80, a4: 0 },
  ], { defaultBpm: 120 });
  assert.equal(timing.spanStep, 48);
  assert.equal(timing.spanSeconds, 0.25);
});
