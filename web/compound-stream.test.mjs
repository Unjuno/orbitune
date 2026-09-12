import assert from 'node:assert/strict';
import test from 'node:test';

import { DEFAULT_SEED_RECORD } from './compound-runtime.mjs';
import { CompoundStreamSession } from './compound-stream.mjs';

class FakeRuntime {
  constructor() { this.advanced = []; this.sampled = 0; }
  async advance(record, state) {
    this.advanced.push(Array.from(record));
    return { ctx: new Float32Array([this.advanced.length]), state: { ...state, steps: (state.steps || 0) + 1 } };
  }
  async sampleNextRecord(ctx) {
    this.sampled += 1;
    return [4, 0, 0, 0, 120 + this.sampled, 0, 0, 0, 0, 0, 0, 0];
  }
}

test('stream session carries state without retaining generated history', async () => {
  const runtime = new FakeRuntime();
  const session = new CompoundStreamSession(runtime, { temperature: 0, topP: 1 });
  await session.start();
  assert.deepEqual(runtime.advanced[0], DEFAULT_SEED_RECORD);
  const batch = await session.generateBatch(3);
  assert.equal(batch.records.length, 3);
  assert.equal(batch.generated, 3);
  assert.equal(batch.state.steps, 4);
  assert.equal(runtime.advanced.length, 4);
  assert.equal(session.records, undefined);
});

test('stream session supports an explicit primer', async () => {
  const runtime = new FakeRuntime();
  const session = new CompoundStreamSession(runtime);
  const primer = [[4, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 110, 0, 0, 0, 0, 0, 0, 0]];
  const started = await session.start({ primerRecords: primer });
  assert.equal(started.state.steps, 2);
  assert.deepEqual(runtime.advanced, primer);
});

test('stream session aborts between records', async () => {
  const runtime = new FakeRuntime();
  const session = new CompoundStreamSession(runtime);
  await session.start();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(() => session.generateBatch(2, { signal: controller.signal }), { name: 'AbortError' });
});
