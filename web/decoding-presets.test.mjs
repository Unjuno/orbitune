import assert from 'node:assert/strict';
import test from 'node:test';
import { DECODING_PRESETS, decodingPreset } from './decoding-presets.mjs';

test('decoding presets separate structural exploration from note temperature', () => {
  assert.equal(DECODING_PRESETS.conservative.temperature, 0.58);
  assert.ok(DECODING_PRESETS.conservative.structureTemperature > DECODING_PRESETS.conservative.temperature);
  assert.equal(DECODING_PRESETS.topk.strategy, 'top-k');
  assert.equal(DECODING_PRESETS.minp.strategy, 'min-p');
});

test('decodingPreset returns a mutable copy', () => {
  const preset = decodingPreset('balanced');
  preset.temperature = 0.1;
  assert.equal(DECODING_PRESETS.balanced.temperature, 0.85);
});
