import { createVerifiedModelSession } from './model-loader.mjs';

export const COMPOUND_ARCHITECTURE = 'orbitune-compound-hierarchical-gpt-v1';
export const COMPOUND_TOKENIZER = 'orbitune-compound-v0-experimental';
export const COMPOUND_RECORD_WIDTH = 12;
export const COMPOUND_D_MODEL = 224;
export const COMPOUND_CONTEXT_ABI = 'native-stream-state+decoder-prefix-v2';
export const FIELD_CARDINALITIES = Object.freeze([10, 16, 7, 16, 1024, 1024, 128, 256, 7, 16, 8, 8]);
export const TIME_COARSE_EDGES = Object.freeze([0, 24, 48, 96, 192, 384, 768, 1536]);
export const TIME_RESIDUAL_LEVELS = 16;
export const CONTINUOUS_COARSE_LEVELS = 8;
export const CONTINUOUS_RESIDUAL_LEVELS = 8;

export const CompoundEventType = Object.freeze({
  NOTE: 0,
  CC: 1,
  PROGRAM: 2,
  BANK: 3,
  TEMPO: 4,
  PEDAL: 5,
  PITCH_BEND: 6,
  CHANNEL_PRESSURE: 7,
  POLY_PRESSURE: 8,
  TIME_SIGNATURE: 9,
});

export const DEFAULT_SEED_RECORD = Object.freeze([4, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0]);
const ALL_EVENT_TYPES = Object.freeze(Array.from({ length: 10 }, (_, i) => i));
const ALL_CHANNELS = Object.freeze(Array.from({ length: 16 }, (_, i) => i));
const RANGE_128 = Object.freeze(Array.from({ length: 128 }, (_, i) => i));
const TEMPO_A1 = Object.freeze(Array.from({ length: 999 }, (_, i) => i + 1));
const TIME_SIGNATURE_A1 = Object.freeze(Array.from({ length: 255 }, (_, i) => i + 1));
const PEDAL_A1 = Object.freeze([0, 1]);
const ZERO_ONLY = Object.freeze([0]);
const TIME_SIGNATURE_A2 = Object.freeze([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]);
const CONTROL_TYPES = new Set([
  CompoundEventType.CC,
  CompoundEventType.PITCH_BEND,
  CompoundEventType.CHANNEL_PRESSURE,
  CompoundEventType.POLY_PRESSURE,
]);

export function bankersRound(value) {
  if (!Number.isFinite(value)) throw new Error('bankersRound requires a finite number');
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

export function quantizeTime(value) {
  if (!Number.isFinite(value)) throw new Error('time value must be finite');
  value = Math.trunc(value);
  if (value < 0) throw new Error('time value must be non-negative');
  const clipped = Math.min(value, TIME_COARSE_EDGES.at(-1));
  let coarse = TIME_COARSE_EDGES.length - 2;
  for (let index = 0; index < TIME_COARSE_EDGES.length - 1; index += 1) {
    if (clipped <= TIME_COARSE_EDGES[index + 1]) { coarse = index; break; }
  }
  const lo = TIME_COARSE_EDGES[coarse];
  const hi = TIME_COARSE_EDGES[coarse + 1];
  const width = Math.max(1, hi - lo);
  let residual = bankersRound((clipped - lo) / width * (TIME_RESIDUAL_LEVELS - 1));
  residual = Math.max(0, Math.min(TIME_RESIDUAL_LEVELS - 1, residual));
  return { coarse, residual };
}

export function dequantizeTime({ coarse, residual }) {
  if (!Number.isInteger(coarse) || coarse < 0 || coarse >= TIME_COARSE_EDGES.length - 1) throw new Error('coarse index is outside the reference range');
  if (!Number.isInteger(residual) || residual < 0 || residual >= TIME_RESIDUAL_LEVELS) throw new Error('residual index is outside the reference range');
  const lo = TIME_COARSE_EDGES[coarse];
  const hi = TIME_COARSE_EDGES[coarse + 1];
  return bankersRound(lo + residual / (TIME_RESIDUAL_LEVELS - 1) * (hi - lo));
}

export function quantizeUnsigned(value, { maximum, coarseLevels = CONTINUOUS_COARSE_LEVELS, residualLevels = CONTINUOUS_RESIDUAL_LEVELS } = {}) {
  if (!(maximum > 0)) throw new Error('maximum must be positive');
  value = Math.trunc(value);
  if (value < 0 || value > maximum) throw new Error(`value must be in 0..${maximum}`);
  if (coarseLevels < 2 || residualLevels < 2) throw new Error('quantization levels must be at least 2');
  const normalized = value / maximum;
  const coarse = Math.min(coarseLevels - 1, Math.trunc(normalized * coarseLevels));
  const lo = coarse / coarseLevels;
  const hi = (coarse + 1) / coarseLevels;
  const local = (normalized - lo) / Math.max(1e-12, hi - lo);
  let residual = bankersRound(local * (residualLevels - 1));
  residual = Math.max(0, Math.min(residualLevels - 1, residual));
  return { coarse, residual };
}

export function dequantizeUnsigned({ coarse, residual }, { maximum, coarseLevels = CONTINUOUS_COARSE_LEVELS, residualLevels = CONTINUOUS_RESIDUAL_LEVELS } = {}) {
  if (!(maximum > 0)) throw new Error('maximum must be positive');
  if (!Number.isInteger(coarse) || coarse < 0 || coarse >= coarseLevels) throw new Error('coarse index is outside range');
  if (!Number.isInteger(residual) || residual < 0 || residual >= residualLevels) throw new Error('residual index is outside range');
  const lo = coarse / coarseLevels;
  const hi = (coarse + 1) / coarseLevels;
  const normalized = lo + residual / (residualLevels - 1) * (hi - lo);
  return Math.max(0, Math.min(maximum, bankersRound(normalized * maximum)));
}

export function allowedA1Ids(eventType) {
  if ([CompoundEventType.NOTE, CompoundEventType.CC, CompoundEventType.PROGRAM, CompoundEventType.BANK, CompoundEventType.POLY_PRESSURE].includes(eventType)) return RANGE_128;
  if (eventType === CompoundEventType.TEMPO) return TEMPO_A1;
  if (eventType === CompoundEventType.PEDAL) return PEDAL_A1;
  if (eventType === CompoundEventType.TIME_SIGNATURE) return TIME_SIGNATURE_A1;
  return ZERO_ONLY;
}

export function allowedA2Ids(eventType) {
  if (eventType === CompoundEventType.BANK) return RANGE_128;
  if (eventType === CompoundEventType.TIME_SIGNATURE) return TIME_SIGNATURE_A2;
  return ZERO_ONLY;
}

function argmaxAllowed(logits, allowedIds) {
  if (!allowedIds.length) throw new Error('no allowed categorical values');
  let best = allowedIds[0];
  let bestValue = Number(logits[best]);
  for (let i = 1; i < allowedIds.length; i += 1) {
    const id = allowedIds[i]; const value = Number(logits[id]);
    if (value > bestValue) { best = id; bestValue = value; }
  }
  return best;
}

export function sampleCategorical(logits, allowedIds, { temperature = 0.85, topP = 0.92, random = Math.random } = {}) {
  if (!(topP > 0 && topP <= 1)) throw new Error('topP must be in (0, 1]');
  if (temperature <= 0) return argmaxAllowed(logits, allowedIds);
  const entries = allowedIds.map((id) => ({ id, value: Number(logits[id]) / Math.max(temperature, 1e-5) }));
  const maxValue = Math.max(...entries.map((entry) => entry.value));
  let total = 0;
  for (const entry of entries) { entry.probability = Math.exp(entry.value - maxValue); total += entry.probability; }
  if (!(total > 0) || !Number.isFinite(total)) throw new Error('categorical probabilities are invalid');
  for (const entry of entries) entry.probability /= total;
  entries.sort((a, b) => b.probability - a.probability || a.id - b.id);
  if (topP < 1) {
    let cumulative = 0; let keep = 0;
    while (keep < entries.length) {
      cumulative += entries[keep].probability;
      keep += 1;
      if (cumulative >= topP) break;
    }
    entries.splice(keep);
    const keptTotal = entries.reduce((sum, entry) => sum + entry.probability, 0);
    for (const entry of entries) entry.probability /= keptTotal;
  }
  let threshold = random();
  for (const entry of entries) {
    threshold -= entry.probability;
    if (threshold <= 0) return entry.id;
  }
  return entries.at(-1).id;
}

export function normalFromRandom(random = Math.random) {
  let u1 = 0; let u2 = 0;
  while (u1 <= Number.EPSILON) u1 = random();
  u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function sampleGaussian(mean, logScale, temperature, { random = Math.random, normal = null } = {}) {
  mean = Number(mean); logScale = Number(logScale);
  if (!Number.isFinite(mean) || !Number.isFinite(logScale)) throw new Error('Gaussian head returned non-finite values');
  if (temperature <= 0) return Math.max(0, Math.min(1, mean));
  const z = normal ? normal() : normalFromRandom(random);
  const value = mean + z * Math.exp(logScale) * temperature;
  return Math.max(0, Math.min(1, value));
}

export function buildRecord({ eventType, channel, delta, a1, a2, velocity, duration, control }) {
  const note = eventType === CompoundEventType.NOTE;
  const controlActive = CONTROL_TYPES.has(eventType);
  const keepA2 = [CompoundEventType.NOTE, CompoundEventType.BANK, CompoundEventType.TIME_SIGNATURE, CompoundEventType.POLY_PRESSURE].includes(eventType);
  return [
    eventType,
    channel,
    delta.coarse,
    delta.residual,
    a1,
    keepA2 ? a2 : 0,
    note ? velocity : 0,
    0,
    note ? duration.coarse : 0,
    note ? duration.residual : 0,
    controlActive ? control.coarse : 0,
    controlActive ? control.residual : 0,
  ];
}

export function validateCompoundRecord(record) {
  if (!Array.isArray(record) && !ArrayBuffer.isView(record)) throw new Error('record must be array-like');
  if (record.length !== COMPOUND_RECORD_WIDTH) throw new Error('Compound record must contain 12 fields');
  for (let i = 0; i < record.length; i += 1) {
    const value = Number(record[i]);
    if (!Number.isInteger(value) || value < 0 || value >= FIELD_CARDINALITIES[i]) throw new Error(`record field ${i}=${value} outside cardinality ${FIELD_CARDINALITIES[i]}`);
  }
  return true;
}

export function createInitialStreamState() {
  return {
    loc: new BigInt64Array(64 * 12), lloc: 0,
    mbuf: new Float32Array(8 * COMPOUND_D_MODEL), mblen: 0,
    mhist: new Float32Array(64 * COMPOUND_D_MODEL), mhlen: 0,
    gbuf: new Float32Array(4 * COMPOUND_D_MODEL), gblen: 0,
    ghist: new Float32Array(64 * COMPOUND_D_MODEL), ghlen: 0,
    memf: new Float32Array(COMPOUND_D_MODEL), memm: new Float32Array(COMPOUND_D_MODEL), mems: new Float32Array(COMPOUND_D_MODEL),
    steps: 0,
  };
}

function scalarI64(ort, value) { return new ort.Tensor('int64', new BigInt64Array([BigInt(value)]), []); }
function scalarF32(ort, value) { return new ort.Tensor('float32', new Float32Array([Number(value)]), []); }
function tensorI64(ort, values, dims) {
  const data = values instanceof BigInt64Array ? values : new BigInt64Array(Array.from(values, (value) => BigInt(value)));
  return new ort.Tensor('int64', data, dims);
}
function tensorF32(ort, values, dims) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  return new ort.Tensor('float32', data, dims);
}
function copyTyped(data) { return new data.constructor(data); }
function scalarNumber(tensor) { return Number(tensor.data[0]); }

function categoricalAt(result, name, stage, width) {
  const tensor = result[name];
  if (!tensor) throw new Error(`decoder output ${name} is missing`);
  const offset = stage * width;
  return tensor.data.slice(offset, offset + width);
}
function scalarHeadAt(result, name, stage) {
  const tensor = result[name];
  if (!tensor) throw new Error(`decoder output ${name} is missing`);
  return Number(tensor.data[stage]);
}

export class CompoundBrowserRuntime {
  constructor(ortNamespace) {
    if (!ortNamespace?.Tensor) throw new Error('ONNX Runtime Web namespace is required');
    this.ort = ortNamespace;
    this.streamSession = null;
    this.decoderSession = null;
  }

  async loadModels({ streamUrl, streamSha256, decoderUrl, decoderSha256, executionProviders = ['wasm'], fetchImpl = globalThis.fetch }) {
    if (!streamUrl || !decoderUrl) throw new Error('both Compound model URLs are required');
    [this.streamSession, this.decoderSession] = await Promise.all([
      createVerifiedModelSession(this.ort, streamUrl, { expectedSha256: streamSha256, executionProviders, fetchImpl }),
      createVerifiedModelSession(this.ort, decoderUrl, { expectedSha256: decoderSha256, executionProviders, fetchImpl }),
    ]);
  }

  async advance(record, state) {
    if (!this.streamSession) throw new Error('Compound stream model is not loaded');
    validateCompoundRecord(record);
    const ort = this.ort;
    const feeds = {
      rec: tensorI64(ort, record, [12]),
      loc: tensorI64(ort, state.loc, [64, 12]), lloc: scalarI64(ort, state.lloc),
      mbuf: tensorF32(ort, state.mbuf, [8, 224]), mblen: scalarI64(ort, state.mblen),
      mhist: tensorF32(ort, state.mhist, [64, 224]), mhlen: scalarI64(ort, state.mhlen),
      gbuf: tensorF32(ort, state.gbuf, [4, 224]), gblen: scalarI64(ort, state.gblen),
      ghist: tensorF32(ort, state.ghist, [64, 224]), ghlen: scalarI64(ort, state.ghlen),
      memf: tensorF32(ort, state.memf, [1, 224]), memm: tensorF32(ort, state.memm, [1, 224]), mems: tensorF32(ort, state.mems, [1, 224]),
      steps: scalarI64(ort, state.steps),
    };
    const out = await this.streamSession.run(feeds);
    if (!out.ctx) throw new Error('Compound stream output ctx is missing');
    const nextState = {
      loc: copyTyped(out.loc_o.data), lloc: scalarNumber(out.lloc_o),
      mbuf: copyTyped(out.mbuf_o.data), mblen: scalarNumber(out.mblen_o),
      mhist: copyTyped(out.mhist_o.data), mhlen: scalarNumber(out.mhlen_o),
      gbuf: copyTyped(out.gbuf_o.data), gblen: scalarNumber(out.gblen_o),
      ghist: copyTyped(out.ghist_o.data), ghlen: scalarNumber(out.ghlen_o),
      memf: copyTyped(out.memf_o.data), memm: copyTyped(out.memm_o.data), mems: copyTyped(out.mems_o.data),
      steps: scalarNumber(out.steps_o),
    };
    return { ctx: copyTyped(out.ctx.data), state: nextState };
  }

  async decodePrefix(ctx, prefix) {
    if (!this.decoderSession) throw new Error('Compound decoder model is not loaded');
    if (!ctx || ctx.length !== COMPOUND_D_MODEL) throw new Error('ctx must contain 224 float values');
    const ort = this.ort;
    return this.decoderSession.run({
      ctx: tensorF32(ort, ctx, [1, 224]),
      et: scalarI64(ort, prefix.et ?? 0),
      ch: scalarI64(ort, prefix.ch ?? 0),
      dn: scalarF32(ort, prefix.dn ?? 0),
      a1: scalarI64(ort, prefix.a1 ?? 0),
      a2: scalarI64(ort, prefix.a2 ?? 0),
      vn: scalarF32(ort, prefix.vn ?? 0),
      dur: scalarF32(ort, prefix.dur ?? 0),
    });
  }

  async sampleNextRecord(ctx, { temperature = 0.85, topP = 0.92, random = Math.random, normal = null } = {}) {
    const prefix = { et: 0, ch: 0, dn: 0, a1: 0, a2: 0, vn: 0, dur: 0 };

    let heads = await this.decodePrefix(ctx, prefix);
    const eventType = sampleCategorical(categoricalAt(heads, 'event_type_logits', 0, 10), ALL_EVENT_TYPES, { temperature, topP, random });
    prefix.et = eventType;

    heads = await this.decodePrefix(ctx, prefix);
    const channel = [CompoundEventType.TEMPO, CompoundEventType.TIME_SIGNATURE].includes(eventType)
      ? 0
      : sampleCategorical(categoricalAt(heads, 'channel_logits', 1, 16), ALL_CHANNELS, { temperature, topP, random });
    prefix.ch = channel;

    heads = await this.decodePrefix(ctx, prefix);
    const deltaNorm = sampleGaussian(scalarHeadAt(heads, 'delta_mean', 2), scalarHeadAt(heads, 'delta_log_scale', 2), temperature, { random, normal });
    prefix.dn = deltaNorm;
    const delta = quantizeTime(Math.max(0, bankersRound(deltaNorm * 1536)));

    heads = await this.decodePrefix(ctx, prefix);
    const a1 = sampleCategorical(categoricalAt(heads, 'a1_logits', 3, 1024), allowedA1Ids(eventType), { temperature, topP, random });
    prefix.a1 = a1;

    heads = await this.decodePrefix(ctx, prefix);
    const a2Allowed = allowedA2Ids(eventType);
    const a2 = a2Allowed.length === 1 ? a2Allowed[0] : sampleCategorical(categoricalAt(heads, 'a2_logits', 4, 1024), a2Allowed, { temperature, topP, random });
    prefix.a2 = a2;

    heads = await this.decodePrefix(ctx, prefix);
    let velocityNorm = 0; let velocity = 0;
    if (eventType === CompoundEventType.NOTE) {
      velocityNorm = sampleGaussian(scalarHeadAt(heads, 'velocity_mean', 5), scalarHeadAt(heads, 'velocity_log_scale', 5), temperature, { random, normal });
      velocity = Math.max(1, Math.min(127, bankersRound(velocityNorm * 127)));
    }
    prefix.vn = velocityNorm;

    heads = await this.decodePrefix(ctx, prefix);
    let durationNorm = 0; let duration = { coarse: 0, residual: 0 };
    if (eventType === CompoundEventType.NOTE) {
      durationNorm = sampleGaussian(scalarHeadAt(heads, 'duration_mean', 6), scalarHeadAt(heads, 'duration_log_scale', 6), temperature, { random, normal });
      duration = quantizeTime(Math.max(1, bankersRound(durationNorm * 1536)));
    }
    prefix.dur = durationNorm;

    heads = await this.decodePrefix(ctx, prefix);
    let control = { coarse: 0, residual: 0 };
    if (CONTROL_TYPES.has(eventType)) {
      const controlNorm = sampleGaussian(scalarHeadAt(heads, 'control_mean', 7), scalarHeadAt(heads, 'control_log_scale', 7), temperature, { random, normal });
      control = quantizeUnsigned(bankersRound(controlNorm * 16383), { maximum: 16383 });
    }

    const record = buildRecord({ eventType, channel, delta, a1, a2, velocity, duration, control });
    validateCompoundRecord(record);
    return record;
  }

  async generate({ primerRecords = [], maxNewEvents = 256, temperature = 0.85, topP = 0.92, random = Math.random, normal = null, onProgress = null } = {}) {
    if (!Number.isInteger(maxNewEvents) || maxNewEvents < 0) throw new Error('maxNewEvents must be a non-negative integer');
    let state = createInitialStreamState();
    const records = (primerRecords.length ? primerRecords : [DEFAULT_SEED_RECORD]).map((record) => Array.from(record, Number));
    let ctx = null;
    for (const record of records) {
      const advanced = await this.advance(record, state); ctx = advanced.ctx; state = advanced.state;
    }
    if (!ctx) throw new Error('generation requires at least one seed record');
    for (let i = 0; i < maxNewEvents; i += 1) {
      const record = await this.sampleNextRecord(ctx, { temperature, topP, random, normal });
      records.push(record);
      const advanced = await this.advance(record, state); ctx = advanced.ctx; state = advanced.state;
      if (typeof onProgress === 'function') onProgress({ generated: i + 1, total: maxNewEvents, record });
    }
    return { records, state, ctx };
  }
}
