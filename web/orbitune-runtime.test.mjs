import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ORBITUNE_V0,
  TOKEN_TO_ID,
  VOCAB,
  allowedNextTokenIds,
  emptyAdapterInputs,
  eventsToMidiBytes,
  packAdapterSafetensors,
  parseSafetensors,
  sampleAllowedLogits,
  tokenIdsToEvents,
} from './orbitune-runtime.mjs';

function makeSafetensors() {
  const metadata = {
    format: 'orbitune-lora-v0',
    rank: '4',
    alpha: '8.0',
    dropout: '0.0',
    target_modules: '["q_proj","v_proj"]',
  };
  const header = { __metadata__: metadata };
  let offset = 0;
  for (let layer = 0; layer < 4; layer += 1) {
    for (const target of ['q_proj', 'v_proj']) {
      const aBytes = 4 * 240 * 4;
      const bBytes = 240 * 4 * 4;
      header[`blocks.${layer}.attn.${target}.lora_a`] = { dtype: 'F32', shape: [4, 240], data_offsets: [offset, offset + aBytes] };
      offset += aBytes;
      header[`blocks.${layer}.attn.${target}.lora_b`] = { dtype: 'F32', shape: [240, 4], data_offsets: [offset, offset + bBytes] };
      offset += bBytes;
    }
  }
  const encodedHeader = new TextEncoder().encode(JSON.stringify(header));
  const buffer = new ArrayBuffer(8 + encodedHeader.length + offset);
  const view = new DataView(buffer);
  view.setBigUint64(0, BigInt(encodedHeader.length), true);
  new Uint8Array(buffer, 8, encodedHeader.length).set(encodedHeader);
  return buffer;
}

function noteAt(position, pitch = 60) {
  return [
    TOKEN_TO_ID.get(`POSITION_${position}`),
    TOKEN_TO_ID.get(`NOTE_PITCH_${pitch}`),
    TOKEN_TO_ID.get('NOTE_DURATION_4'),
    TOKEN_TO_ID.get('VELOCITY_16'),
  ];
}

test('v0 vocabulary and empty adapter shapes are fixed', () => {
  assert.equal(VOCAB.length, 204);
  assert.equal(TOKEN_TO_ID.get('BAR'), 3);
  assert.equal(ORBITUNE_V0.maxNotesPerPosition, 8);
  const adapter = emptyAdapterInputs();
  assert.equal(adapter.loraA.length, ORBITUNE_V0.layers * 2 * ORBITUNE_V0.rank * ORBITUNE_V0.hidden);
  assert.equal(adapter.loraB.length, ORBITUNE_V0.layers * 2 * ORBITUNE_V0.hidden * ORBITUNE_V0.rank);
});

test('Safetensors metadata and fixed rank-4 adapter packing work', () => {
  const buffer = makeSafetensors();
  const parsed = parseSafetensors(buffer);
  assert.equal(parsed.metadata.format, 'orbitune-lora-v0');
  assert.equal(parsed.tensors.size, 16);
  const packed = packAdapterSafetensors(buffer);
  assert.equal(packed.scale[0], 2);
  assert.equal(packed.loraA.length, 4 * 2 * 4 * 240);
  assert.equal(packed.loraB.length, 4 * 2 * 240 * 4);
});

test('grammar allows bounded chords and requires full requested bars', () => {
  const bos = TOKEN_TO_ID.get('BOS');
  const bar = TOKEN_TO_ID.get('BAR');
  const eos = TOKEN_TO_ID.get('EOS');
  let ids = [bos];
  assert.deepEqual(allowedNextTokenIds(ids, 1), [bar]);
  ids.push(bar, ...noteAt(0, 60));
  const afterZero = allowedNextTokenIds(ids, 1);
  assert(!afterZero.includes(eos));
  assert(afterZero.includes(TOKEN_TO_ID.get('POSITION_0')));
  assert(afterZero.includes(TOKEN_TO_ID.get('POSITION_1')));

  const choosingSamePosition = [...ids, TOKEN_TO_ID.get('POSITION_0')];
  const pitchChoices = allowedNextTokenIds(choosingSamePosition, 1);
  assert(!pitchChoices.includes(TOKEN_TO_ID.get('NOTE_PITCH_60')));
  assert(pitchChoices.includes(TOKEN_TO_ID.get('NOTE_PITCH_64')));

  ids.push(...noteAt(12, 64));
  const finalBar = allowedNextTokenIds(ids, 1);
  assert(finalBar.includes(eos));
  assert(finalBar.includes(TOKEN_TO_ID.get('POSITION_12')));
  assert(finalBar.includes(TOKEN_TO_ID.get('POSITION_13')));

  const twoBars = [bos, bar, ...noteAt(12)];
  const afterFirstBar = allowedNextTokenIds(twoBars, 2);
  assert(afterFirstBar.includes(bar));
  assert(!afterFirstBar.includes(eos));
});

test('grammar caps simultaneous notes at one position', () => {
  const ids = [TOKEN_TO_ID.get('BOS'), TOKEN_TO_ID.get('BAR')];
  for (let index = 0; index < ORBITUNE_V0.maxNotesPerPosition; index += 1) {
    ids.push(...noteAt(12, 60 + index));
  }
  const allowed = allowedNextTokenIds(ids, 1);
  assert(!allowed.includes(TOKEN_TO_ID.get('POSITION_12')));
  assert(allowed.includes(TOKEN_TO_ID.get('POSITION_13')));
  assert(allowed.includes(TOKEN_TO_ID.get('EOS')));
});

test('allowed-logit sampling never chooses a masked token', () => {
  const logits = new Float32Array(VOCAB.length).fill(-100);
  logits[TOKEN_TO_ID.get('NOTE_PITCH_60')] = 100;
  logits[TOKEN_TO_ID.get('NOTE_PITCH_64')] = 50;
  logits[TOKEN_TO_ID.get('BAR')] = 1000;
  const allowed = [TOKEN_TO_ID.get('NOTE_PITCH_60'), TOKEN_TO_ID.get('NOTE_PITCH_64')];
  const sampled = sampleAllowedLogits(logits, allowed, { temperature: 1, topP: 1, random: () => 0.1 });
  assert.equal(sampled, TOKEN_TO_ID.get('NOTE_PITCH_60'));
});

test('generated token IDs convert chords to playable MIDI bytes', () => {
  const ids = [
    TOKEN_TO_ID.get('BOS'), TOKEN_TO_ID.get('BAR'),
    ...noteAt(0, 60), ...noteAt(0, 64), ...noteAt(12, 67), TOKEN_TO_ID.get('EOS'),
  ];
  const events = tokenIdsToEvents(ids);
  assert.equal(events.length, 3);
  assert.equal(events[0].position, 0);
  assert.equal(events[1].position, 0);
  const midi = eventsToMidiBytes(events, 84);
  assert.equal(new TextDecoder().decode(midi.slice(0, 4)), 'MThd');
  assert(new TextDecoder().decode(midi).includes('MTrk'));
});
