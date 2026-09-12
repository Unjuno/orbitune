import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BoundedRecordQueue,
  createStreamingGenerator,
  InfiniteCompoundStream,
} from './compound-stream.mjs';

const NOTE_RECORD = Object.freeze([0, 0, 1, 0, 60, 0, 96, 0, 1, 0, 0, 0]);

function fakeRuntime() {
  const calls = { advance: 0, sample: 0 };
  return {
    calls,
    async advance(_record, state) {
      calls.advance += 1;
      return {
        ctx: new Float32Array(224).fill(calls.advance),
        state: { ...state, steps: Number(state.steps || 0) + 1 },
      };
    },
    async sampleNextRecord() {
      calls.sample += 1;
      return [...NOTE_RECORD];
    },
  };
}

test('streaming generator carries state without retaining generated history', async () => {
  const runtime = fakeRuntime();
  const generator = await createStreamingGenerator(runtime);
  assert.equal(runtime.calls.advance, 1, 'default seed must initialize the stream state');
  for (let index = 0; index < 20; index += 1) await generator.next({ temperature: 0 });
  assert.equal(runtime.calls.sample, 20);
  assert.equal(runtime.calls.advance, 21);
  assert.equal(generator.state.steps, 21);
  assert.equal('records' in generator, false, 'generator must not accumulate an unbounded record history');
});

test('bounded record queue retains only the configured recent window', () => {
  const queue = new BoundedRecordQueue(4);
  for (let index = 0; index < 10; index += 1) queue.push([index]);
  assert.equal(queue.size, 4);
  assert.deepEqual(queue.snapshot().map((record) => record[0]), [6, 7, 8, 9]);
});

test('infinite stream pump remains bounded while generation count grows', async () => {
  const runtime = fakeRuntime();
  const scheduled = [];
  const sink = {
    lookaheadSeconds: () => 0,
    async enqueue(record) { scheduled.push([...record]); },
  };
  const stream = new InfiniteCompoundStream({ runtime, sink, targetLookaheadSeconds: 1, maxRecentRecords: 3 });
  stream.generator = await createStreamingGenerator(runtime);
  for (let index = 0; index < 12; index += 1) await stream.pumpOnce();
  assert.equal(stream.generated, 12);
  assert.equal(stream.recentRecords().length, 3);
  assert.equal(scheduled.length, 12);
});
