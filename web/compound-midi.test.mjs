import test from 'node:test';
import assert from 'node:assert/strict';
import { compoundEventsToMidiBytes } from './compound-midi.mjs';
import { CompoundEventType } from './compound-runtime.mjs';

function tempo(bpm) {
  return { type: CompoundEventType.TEMPO, step: 0, channel: 0, a1: bpm, a2: 0, a3: 0, a4: 0 };
}

test('MIDI tempo serialization rejects 1-3 BPM instead of wrapping three-byte payload', () => {
  for (const bpm of [1, 2, 3]) {
    assert.throws(() => compoundEventsToMidiBytes([tempo(bpm)]), /exceeds the three-byte MIDI tempo field/);
  }
});

test('MIDI tempo serialization accepts the first representable BPM', () => {
  const midi = compoundEventsToMidiBytes([tempo(4)]);
  assert.equal(new TextDecoder().decode(midi.slice(0, 4)), 'MThd');
  // 4 BPM = 15,000,000 microseconds/qn = 0xE4E1C0, which fits in 3 bytes.
  const pattern = [0xff, 0x51, 0x03, 0xe4, 0xe1, 0xc0];
  const data = Array.from(midi);
  assert.ok(data.some((_, index) => pattern.every((byte, offset) => data[index + offset] === byte)));
});
