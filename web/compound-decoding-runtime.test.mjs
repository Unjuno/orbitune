import assert from 'node:assert/strict';
import test from 'node:test';

import { CompoundDecodingRuntime, sampleCategoricalAdvanced } from './compound-decoding-runtime.mjs';
import { CompoundEventType } from './compound-runtime.mjs';

test('top-k filtering never samples outside the retained k candidates', () => {
  const logits = new Float32Array([6, 5, 4, 3]);
  for (const random of [() => 0, () => 0.4, () => 0.999999]) {
    const id = sampleCategoricalAdvanced(logits, [0, 1, 2, 3], {
      strategy: 'top-k', temperature: 1, topP: 1, topK: 2, minP: 0, random,
    });
    assert.ok([0, 1].includes(id));
  }
});

test('min-p filtering removes candidates far below the most likely token', () => {
  const logits = new Float32Array([8, 7.8, 1, 0]);
  const id = sampleCategoricalAdvanced(logits, [0, 1, 2, 3], {
    strategy: 'min-p', temperature: 1, topP: 1, topK: 0, minP: 0.5, random: () => 0.999999,
  });
  assert.ok([0, 1].includes(id));
});

test('decoding adapter applies structural temperature to PROGRAM decisions independently', async () => {
  const eventType = new Float32Array(10);
  eventType[CompoundEventType.PROGRAM] = 10;
  const channel = new Float32Array(32);
  channel[16 + 3] = 10;
  const a1 = new Float32Array(4096);
  a1[3 * 1024 + 40] = 10;
  const scalar = new Float32Array(8);
  scalar[2] = 0;
  const fakeBase = {
    advance: async () => { throw new Error('not used'); },
    decodePrefix: async () => ({
      event_type_logits: { data: eventType },
      channel_logits: { data: channel },
      a1_logits: { data: a1 },
      delta_mean: { data: scalar },
      delta_log_scale: { data: scalar },
      velocity_mean: { data: scalar },
      velocity_log_scale: { data: scalar },
      duration_mean: { data: scalar },
      duration_log_scale: { data: scalar },
      control_mean: { data: scalar },
      control_log_scale: { data: scalar },
    }),
  };
  const runtime = new CompoundDecodingRuntime(fakeBase, {
    strategy: 'nucleus', temperature: 0.05, structureTemperature: 0, topP: 0.9, topK: 0, minP: 0,
  });
  const record = await runtime.sampleNextRecord(new Float32Array(224), { random: () => 0.5 });
  assert.equal(record[0], CompoundEventType.PROGRAM);
  assert.equal(record[1], 3);
  assert.equal(record[4], 40);
});
