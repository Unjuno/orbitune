import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  bankersRound, quantizeTime, dequantizeTime, quantizeUnsigned, dequantizeUnsigned,
  allowedA1Ids, allowedA2Ids, CompoundEventType, buildRecord, validateCompoundRecord,
  sampleCategorical, createInitialStreamState,
} from './compound-runtime.mjs';
import { canonicalizeCompoundEvents, compoundEventsToMidiBytes, decodeCompoundRecords } from './compound-midi.mjs';
import { buildPreviewSchedule, midiPitchToFrequency } from './compound-player.mjs';

test('Python-compatible bank half-to-even rounding', () => {
  assert.equal(bankersRound(0.5), 0); assert.equal(bankersRound(1.5), 2); assert.equal(bankersRound(2.5), 2); assert.equal(bankersRound(3.5), 4);
  assert.equal(bankersRound(-1.5), -2); assert.equal(bankersRound(-2.5), -2);
});

test('time quantization preserves Python edge rules', () => {
  assert.deepEqual(quantizeTime(24), { coarse: 0, residual: 15 });
  assert.deepEqual(quantizeTime(25), { coarse: 1, residual: 1 });
  assert.equal(dequantizeTime(quantizeTime(1536)), 1536);
  assert.deepEqual(quantizeTime(999999), { coarse: 6, residual: 15 });
});

test('unsigned factorization round-trips representative MIDI controls', () => {
  for (const maximum of [127, 16383]) for (const value of [0, 1, Math.floor(maximum / 2), maximum]) {
    const q = quantizeUnsigned(value, { maximum }); const decoded = dequantizeUnsigned(q, { maximum });
    assert.ok(decoded >= 0 && decoded <= maximum);
  }
});

test('native per-type categorical masks are represented', () => {
  assert.equal(allowedA1Ids(CompoundEventType.NOTE).length, 128);
  assert.deepEqual([...allowedA1Ids(CompoundEventType.PITCH_BEND)], [0]);
  assert.deepEqual([...allowedA1Ids(CompoundEventType.PEDAL)], [0, 1]);
  assert.equal(allowedA1Ids(CompoundEventType.TEMPO)[0], 1); assert.equal(allowedA1Ids(CompoundEventType.TEMPO).at(-1), 999);
  assert.deepEqual([...allowedA2Ids(CompoundEventType.TIME_SIGNATURE)], [1,2,4,8,16,32,64,128,256,512]);
});

test('temperature zero categorical sampling is deterministic argmax over allowed values', () => {
  const logits = new Float32Array([0, 5, 10, 2]);
  assert.equal(sampleCategorical(logits, [0, 1, 3], { temperature: 0 }), 1);
});

test('record builder applies native unused-field zeroing', () => {
  const delta = { coarse: 1, residual: 2 }; const duration = { coarse: 3, residual: 4 }; const control = { coarse: 5, residual: 6 };
  const note = buildRecord({ eventType: 0, channel: 2, delta, a1: 60, a2: 0, velocity: 100, duration, control });
  assert.deepEqual(note, [0,2,1,2,60,0,100,0,3,4,0,0]); assert.equal(validateCompoundRecord(note), true);
  const cc = buildRecord({ eventType: 1, channel: 2, delta, a1: 10, a2: 0, velocity: 100, duration, control });
  assert.deepEqual(cc, [1,2,1,2,10,0,0,0,0,0,5,6]);
});

test('initial V2 tensor state has exact fixed capacities', () => {
  const state = createInitialStreamState();
  assert.equal(state.loc.length, 64 * 12); assert.equal(state.mbuf.length, 8 * 224); assert.equal(state.mhist.length, 64 * 224);
  assert.equal(state.gbuf.length, 4 * 224); assert.equal(state.ghist.length, 64 * 224); assert.equal(state.memf.length, 224); assert.equal(state.steps, 0);
});

test('record decoding follows tokenizer factorized duration/control semantics', () => {
  const records = [
    [4,0,0,0,120,0,0,0,0,0,0,0],
    [0,0,0,0,60,0,100,0,1,0,0,0],
    [1,0,0,0,10,0,0,0,0,0,4,0],
  ];
  const events = decodeCompoundRecords(records);
  assert.equal(events[0].type, 4); assert.equal(events.find((e) => e.type === 0).a2, 24); assert.ok(events.find((e) => e.type === 1).a2 >= 0);
});

test('canonicalization merges duplicate notes and truncates retriggers', () => {
  const notes = canonicalizeCompoundEvents([
    { type:0, step:0, channel:0, a1:60, a2:30, a3:80, a4:0 },
    { type:0, step:0, channel:0, a1:60, a2:20, a3:100, a4:0 },
    { type:0, step:10, channel:0, a1:60, a2:20, a3:90, a4:0 },
  ]);
  assert.equal(notes.length, 2); assert.equal(notes[0].a2, 10); assert.equal(notes[0].a3, 100);
});

test('Compound MIDI writer emits deterministic SMF header and track', () => {
  const events = [
    { type:4, step:0, channel:0, a1:120, a2:0, a3:0, a4:0 },
    { type:0, step:0, channel:0, a1:69, a2:96, a3:100, a4:0 },
  ];
  const midi = compoundEventsToMidiBytes(events);
  assert.equal(new TextDecoder().decode(midi.slice(0, 4)), 'MThd'); assert.equal(new TextDecoder().decode(midi.slice(14, 18)), 'MTrk');
  assert.ok(midi.length > 30);
});

test('preview schedule honors tempo and pitch frequency', () => {
  const schedule = buildPreviewSchedule([
    { type:4, step:0, channel:0, a1:120, a2:0, a3:0, a4:0 },
    { type:0, step:0, channel:0, a1:69, a2:96, a3:100, a4:0 },
    { type:4, step:96, channel:0, a1:60, a2:0, a3:0, a4:0 },
    { type:0, step:96, channel:0, a1:60, a2:96, a3:100, a4:0 },
  ]);
  assert.equal(midiPitchToFrequency(69), 440); assert.equal(schedule.length, 2);
  assert.equal(schedule[0].start, 0); assert.equal(schedule[0].end, 0.5); assert.equal(schedule[1].start, 0.5); assert.equal(schedule[1].end, 1.5);
});

class FakeTensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}
const fakeOrt = { Tensor: FakeTensor };
function f32(size, value = 0) { const out = new Float32Array(size); out.fill(value); return out; }
function tensor(data, dims = []) { return { data, dims }; }

test('V2 stream adapter uses exact tensor names/shapes and carries returned state', async () => {
  const { CompoundBrowserRuntime } = await import('./compound-runtime.mjs');
  const runtime = new CompoundBrowserRuntime(fakeOrt); let captured = null;
  runtime.streamSession = { run: async (feeds) => {
    captured = feeds;
    return {
      ctx: tensor(f32(224, 1), [1,224]), loc_o: tensor(new BigInt64Array(64*12), [64,12]), lloc_o: tensor(new BigInt64Array([1n])),
      mbuf_o: tensor(f32(8*224), [8,224]), mblen_o: tensor(new BigInt64Array([1n])), mhist_o: tensor(f32(64*224), [64,224]), mhlen_o: tensor(new BigInt64Array([0n])),
      gbuf_o: tensor(f32(4*224), [4,224]), gblen_o: tensor(new BigInt64Array([0n])), ghist_o: tensor(f32(64*224), [64,224]), ghlen_o: tensor(new BigInt64Array([0n])),
      memf_o: tensor(f32(224), [1,224]), memm_o: tensor(f32(224), [1,224]), mems_o: tensor(f32(224), [1,224]), steps_o: tensor(new BigInt64Array([1n])),
    };
  }};
  const result = await runtime.advance([4,0,0,0,120,0,0,0,0,0,0,0], createInitialStreamState());
  assert.deepEqual(captured.rec.dims, [12]); assert.deepEqual(captured.loc.dims, [64,12]); assert.deepEqual(captured.lloc.dims, []);
  assert.deepEqual(captured.memf.dims, [1,224]); assert.equal(result.ctx.length, 224); assert.equal(result.state.steps, 1); assert.equal(result.state.lloc, 1);
});

test('decoder-prefix orchestration samples the native eight stages and preserves pre-quantization scalars', async () => {
  const { CompoundBrowserRuntime } = await import('./compound-runtime.mjs');
  const runtime = new CompoundBrowserRuntime(fakeOrt); let calls = 0; const prefixes = [];
  runtime.decoderSession = { run: async (feeds) => {
    prefixes.push({ et:Number(feeds.et.data[0]), ch:Number(feeds.ch.data[0]), dn:feeds.dn.data[0], a1:Number(feeds.a1.data[0]), a2:Number(feeds.a2.data[0]), vn:feeds.vn.data[0], dur:feeds.dur.data[0] });
    calls += 1;
    const event = f32(8*10, -10); event[0*10 + CompoundEventType.NOTE] = 10;
    const channel = f32(8*16, -10); channel[1*16 + 2] = 10;
    const a1 = f32(8*1024, -10); a1[3*1024 + 60] = 10;
    const a2 = f32(8*1024, -10);
    const dm=f32(8); dm[2]=0.5; const dl=f32(8,-5);
    const vm=f32(8); vm[5]=0.5; const vl=f32(8,-5);
    const durm=f32(8); durm[6]=0.5; const durl=f32(8,-5);
    const cm=f32(8); const cl=f32(8,-5);
    return { event_type_logits:tensor(event), channel_logits:tensor(channel), delta_mean:tensor(dm), delta_log_scale:tensor(dl), a1_logits:tensor(a1), a2_logits:tensor(a2), velocity_mean:tensor(vm), velocity_log_scale:tensor(vl), duration_mean:tensor(durm), duration_log_scale:tensor(durl), control_mean:tensor(cm), control_log_scale:tensor(cl) };
  }};
  const record = await runtime.sampleNextRecord(f32(224), { temperature:0 });
  assert.equal(calls, 8); assert.deepEqual(record, [0,2,5,15,60,0,64,0,5,15,0,0]);
  assert.equal(prefixes[3].dn, 0.5); assert.equal(prefixes[6].vn, 0.5); assert.equal(prefixes[7].dur, 0.5);
});

test('checked-in Compound runtime config does not publish model URLs', async () => {
  const config = JSON.parse(await readFile(new URL('./compound-runtime-config.json', import.meta.url), 'utf8'));
  assert.equal(config.publication_status, 'runtime_ready_model_unpublished');
  assert.equal(config.redistribution_review, 'pending');
  assert.deepEqual(config.variants, []);
  assert.equal(config.commercial_eligible, false);
});

test('checked-in V2 contract matches runtime ABI cardinalities and graph boundaries', async () => {
  const contract = JSON.parse(await readFile(new URL('./compound-inference-contract-v2.json', import.meta.url), 'utf8'));
  assert.equal(contract.strategy, 'native-stream-state + decoder-prefix (two graphs)');
  assert.deepEqual(contract.field_cardinalities, [10,16,7,16,1024,1024,128,256,7,16,8,8]);
  assert.equal(contract.stream_advance.inputs.length, 15);
  assert.deepEqual(contract.decoder_prefix.stage_order, ['event_type','channel','delta','a1','a2','velocity','duration','control']);
  assert.equal(contract.decoder_prefix.outputs.length, 12);
});
