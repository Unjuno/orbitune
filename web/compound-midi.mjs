import {
  bankersRound,
  CompoundEventType,
  dequantizeTime,
  dequantizeUnsigned,
  validateCompoundRecord,
} from './compound-runtime.mjs';

export const TEMPORAL_RESOLUTION = 96;
const SORT_PRIORITY = Object.freeze({
  [CompoundEventType.TIME_SIGNATURE]: 0,
  [CompoundEventType.TEMPO]: 1,
  [CompoundEventType.BANK]: 2,
  [CompoundEventType.PROGRAM]: 3,
  [CompoundEventType.CC]: 4,
  [CompoundEventType.PEDAL]: 5,
  [CompoundEventType.PITCH_BEND]: 6,
  [CompoundEventType.CHANNEL_PRESSURE]: 7,
  [CompoundEventType.POLY_PRESSURE]: 8,
  [CompoundEventType.NOTE]: 9,
});

function requireZero(name, fields) {
  const nonzero = Object.entries(fields).filter(([, value]) => value !== 0).map(([key]) => key);
  if (nonzero.length) throw new Error(`${name} unused fields must be zero: ${nonzero.join(', ')}`);
}

export function validateCompoundEvent(event) {
  if (!Number.isInteger(event.step) || event.step < 0) throw new Error('step must be a non-negative integer');
  if (!Number.isInteger(event.channel) || event.channel < 0 || event.channel > 15) throw new Error('channel must be in 0..15');
  for (const name of ['a1', 'a2', 'a3', 'a4']) if (!Number.isInteger(event[name]) || event[name] < 0) throw new Error(`${name} must be a non-negative integer`);
  switch (event.type) {
    case CompoundEventType.NOTE:
      if (event.a1 > 127 || event.a2 <= 0 || event.a3 < 1 || event.a3 > 127) throw new Error('invalid NOTE event');
      requireZero('NOTE', { a4: event.a4 }); break;
    case CompoundEventType.CC:
      if (event.a1 > 127 || event.a2 > 127) throw new Error('invalid CC event');
      requireZero('CC', { a3: event.a3, a4: event.a4 }); break;
    case CompoundEventType.PROGRAM:
      if (event.a1 > 127) throw new Error('invalid PROGRAM event');
      requireZero('PROGRAM', { a2: event.a2, a3: event.a3, a4: event.a4 }); break;
    case CompoundEventType.BANK:
      if (event.a1 > 127 || event.a2 > 127) throw new Error('invalid BANK event');
      requireZero('BANK', { a3: event.a3, a4: event.a4 }); break;
    case CompoundEventType.TEMPO:
      if (event.channel !== 0 || event.a1 < 1 || event.a1 > 999) throw new Error('invalid TEMPO event');
      requireZero('TEMPO', { a2: event.a2, a3: event.a3, a4: event.a4 }); break;
    case CompoundEventType.PEDAL:
      if (![0, 1].includes(event.a1)) throw new Error('invalid PEDAL event');
      requireZero('PEDAL', { a2: event.a2, a3: event.a3, a4: event.a4 }); break;
    case CompoundEventType.PITCH_BEND:
      if (event.a1 > 16383) throw new Error('invalid PITCH_BEND event');
      requireZero('PITCH_BEND', { a2: event.a2, a3: event.a3, a4: event.a4 }); break;
    case CompoundEventType.CHANNEL_PRESSURE:
      if (event.a1 > 127) throw new Error('invalid CHANNEL_PRESSURE event');
      requireZero('CHANNEL_PRESSURE', { a2: event.a2, a3: event.a3, a4: event.a4 }); break;
    case CompoundEventType.POLY_PRESSURE:
      if (event.a1 > 127 || event.a2 > 127) throw new Error('invalid POLY_PRESSURE event');
      requireZero('POLY_PRESSURE', { a3: event.a3, a4: event.a4 }); break;
    case CompoundEventType.TIME_SIGNATURE:
      if (event.channel !== 0 || event.a1 < 1 || event.a1 > 255 || event.a2 <= 0 || !Number.isSafeInteger(event.a2) || (event.a2 & (event.a2 - 1)) !== 0) throw new Error('invalid TIME_SIGNATURE event');
      requireZero('TIME_SIGNATURE', { a3: event.a3, a4: event.a4 }); break;
    default: throw new Error(`unknown Compound event type ${event.type}`);
  }
  return true;
}

export function decodeCompoundRecords(records) {
  const events = []; let step = 0;
  for (const raw of records) {
    const record = Array.from(raw, Number); validateCompoundRecord(record);
    step += dequantizeTime({ coarse: record[2], residual: record[3] });
    const type = record[0]; let a1 = record[4]; let a2 = record[5];
    const continuous = { coarse: record[10], residual: record[11] };
    if (type === CompoundEventType.NOTE) a2 = Math.max(1, dequantizeTime({ coarse: record[8], residual: record[9] }));
    else if (type === CompoundEventType.CC) a2 = dequantizeUnsigned(continuous, { maximum: 127 });
    else if (type === CompoundEventType.PITCH_BEND) a1 = dequantizeUnsigned(continuous, { maximum: 16383 });
    else if (type === CompoundEventType.CHANNEL_PRESSURE) a1 = dequantizeUnsigned(continuous, { maximum: 127 });
    else if (type === CompoundEventType.POLY_PRESSURE) a2 = dequantizeUnsigned(continuous, { maximum: 127 });
    events.push({ type, step, channel: record[1], a1, a2, a3: record[6], a4: record[7] });
  }
  return canonicalizeCompoundEvents(events);
}

function eventSort(a, b) {
  return a.step - b.step || SORT_PRIORITY[a.type] - SORT_PRIORITY[b.type] || a.channel - b.channel || a.type - b.type || a.a1 - b.a1 || a.a2 - b.a2 || a.a3 - b.a3 || a.a4 - b.a4;
}

export function canonicalizeCompoundEvents(events) {
  const checked = events.map((event) => ({ ...event }));
  for (const event of checked) validateCompoundEvent(event);
  checked.sort(eventSort);
  const merged = new Map(); const others = [];
  for (const event of checked) {
    if (event.type !== CompoundEventType.NOTE) { others.push(event); continue; }
    const key = `${event.step}:${event.channel}:${event.a1}`;
    const previous = merged.get(key);
    if (!previous) merged.set(key, event);
    else merged.set(key, { type: CompoundEventType.NOTE, step: event.step, channel: event.channel, a1: event.a1, a2: Math.max(previous.a2, event.a2), a3: Math.max(previous.a3, event.a3), a4: 0 });
  }
  const notes = [...merged.values()].sort((a, b) => a.step - b.step || a.channel - b.channel || a.a1 - b.a1);
  const fixed = []; const active = new Map();
  for (const event of notes) {
    const key = `${event.channel}:${event.a1}`; const previousIndex = active.get(key);
    if (previousIndex !== undefined) {
      const previous = fixed[previousIndex];
      if (previous.step + previous.a2 > event.step) fixed[previousIndex] = { ...previous, a2: Math.max(1, event.step - previous.step) };
    }
    active.set(key, fixed.length); fixed.push(event);
  }
  return [...fixed, ...others].sort(eventSort);
}

function writeVlq(value) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error('VLQ value must be a non-negative safe integer');
  const bytes = [value % 128]; value = Math.floor(value / 128);
  while (value) { bytes.push(0x80 | (value % 128)); value = Math.floor(value / 128); }
  return bytes.reverse();
}
function pushU16(array, value) { array.push(Math.floor(value / 256) & 0xff, value & 0xff); }
function pushU32(array, value) { array.push(Math.floor(value / 16777216) & 0xff, Math.floor(value / 65536) & 0xff, Math.floor(value / 256) & 0xff, value & 0xff); }
function compareBytes(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

export function compoundEventsToMidiBytes(events, { division = TEMPORAL_RESOLUTION } = {}) {
  if (!Number.isInteger(division) || division <= 0 || division > 65535) throw new Error('division must be in 1..65535');
  const timeline = [];
  for (const event of canonicalizeCompoundEvents(events)) {
    const tick = bankersRound(event.step * division / TEMPORAL_RESOLUTION);
    if (event.type === CompoundEventType.NOTE) {
      timeline.push({ tick, priority: 20, message: [0x90 | event.channel, event.a1, event.a3] });
      timeline.push({ tick: bankersRound((event.step + event.a2) * division / TEMPORAL_RESOLUTION), priority: 10, message: [0x80 | event.channel, event.a1, 0] });
    } else if (event.type === CompoundEventType.CC) timeline.push({ tick, priority: 5, message: [0xB0 | event.channel, event.a1, event.a2] });
    else if (event.type === CompoundEventType.PROGRAM) timeline.push({ tick, priority: 4, message: [0xC0 | event.channel, event.a1] });
    else if (event.type === CompoundEventType.BANK) {
      timeline.push({ tick, priority: 2, message: [0xB0 | event.channel, 0, event.a1] });
      timeline.push({ tick, priority: 3, message: [0xB0 | event.channel, 32, event.a2] });
    } else if (event.type === CompoundEventType.TEMPO) {
      const micros = Math.max(1, bankersRound(60_000_000 / event.a1));
      timeline.push({ tick, priority: 0, message: [0xff, 0x51, 0x03, Math.floor(micros / 65536) & 0xff, Math.floor(micros / 256) & 0xff, micros & 0xff] });
    } else if (event.type === CompoundEventType.PEDAL) timeline.push({ tick, priority: 5, message: [0xB0 | event.channel, 64, event.a1 ? 127 : 0] });
    else if (event.type === CompoundEventType.PITCH_BEND) timeline.push({ tick, priority: 6, message: [0xE0 | event.channel, event.a1 & 0x7f, Math.floor(event.a1 / 128) & 0x7f] });
    else if (event.type === CompoundEventType.CHANNEL_PRESSURE) timeline.push({ tick, priority: 6, message: [0xD0 | event.channel, event.a1] });
    else if (event.type === CompoundEventType.POLY_PRESSURE) timeline.push({ tick, priority: 6, message: [0xA0 | event.channel, event.a1, event.a2] });
    else if (event.type === CompoundEventType.TIME_SIGNATURE) timeline.push({ tick, priority: 0, message: [0xff, 0x58, 0x04, event.a1, bankersRound(Math.log2(event.a2)), 24, 8] });
  }
  timeline.sort((a, b) => a.tick - b.tick || a.priority - b.priority || compareBytes(a.message, b.message));
  const track = []; let previous = 0;
  for (const item of timeline) { track.push(...writeVlq(Math.max(0, item.tick - previous)), ...item.message); previous = item.tick; }
  track.push(0x00, 0xff, 0x2f, 0x00);
  const out = [...new TextEncoder().encode('MThd')]; pushU32(out, 6); pushU16(out, 0); pushU16(out, 1); pushU16(out, division);
  out.push(...new TextEncoder().encode('MTrk')); pushU32(out, track.length); out.push(...track);
  return new Uint8Array(out);
}
