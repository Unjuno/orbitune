import {
  CompoundEventType,
  DEFAULT_SEED_RECORD,
  allowedA1Ids,
  allowedA2Ids,
  bankersRound,
  buildRecord,
  createInitialStreamState,
  quantizeTime,
  quantizeUnsigned,
  sampleGaussian,
  validateCompoundRecord,
} from './compound-runtime.mjs';

const ALL_EVENT_TYPES = Object.freeze(Array.from({ length: 10 }, (_, i) => i));
const ALL_CHANNELS = Object.freeze(Array.from({ length: 16 }, (_, i) => i));
const CONTROL_TYPES = new Set([
  CompoundEventType.CC,
  CompoundEventType.PITCH_BEND,
  CompoundEventType.CHANNEL_PRESSURE,
  CompoundEventType.POLY_PRESSURE,
]);

export const DECODING_STRATEGIES = Object.freeze(['nucleus', 'top-k', 'min-p', 'hybrid']);

function validateOptions({ strategy, temperature, structureTemperature, topP, topK, minP }) {
  if (!DECODING_STRATEGIES.includes(strategy)) throw new Error(`unsupported decoding strategy: ${strategy}`);
  if (!(temperature >= 0)) throw new Error('temperature must be non-negative');
  if (!(structureTemperature >= 0)) throw new Error('structureTemperature must be non-negative');
  if (!(topP > 0 && topP <= 1)) throw new Error('topP must be in (0, 1]');
  if (!Number.isInteger(topK) || topK < 0) throw new Error('topK must be a non-negative integer');
  if (!(minP >= 0 && minP < 1)) throw new Error('minP must be in [0, 1)');
}

function argmax(entries) {
  if (!entries.length) throw new Error('no allowed categorical values');
  let best = entries[0];
  for (const entry of entries.slice(1)) if (entry.value > best.value) best = entry;
  return best.id;
}

export function sampleCategoricalAdvanced(
  logits,
  allowedIds,
  {
    strategy = 'nucleus',
    temperature = 0.85,
    topP = 0.92,
    topK = 0,
    minP = 0,
    random = Math.random,
  } = {},
) {
  validateOptions({ strategy, temperature, structureTemperature: temperature, topP, topK, minP });
  if (!allowedIds.length) throw new Error('no allowed categorical values');
  const entries = allowedIds.map((id) => ({ id, value: Number(logits[id]) }));
  if (temperature <= 0) return argmax(entries);
  for (const entry of entries) entry.value /= Math.max(temperature, 1e-5);
  const maxValue = Math.max(...entries.map((entry) => entry.value));
  let total = 0;
  for (const entry of entries) {
    entry.probability = Math.exp(entry.value - maxValue);
    total += entry.probability;
  }
  if (!(total > 0) || !Number.isFinite(total)) throw new Error('categorical probabilities are invalid');
  for (const entry of entries) entry.probability /= total;
  entries.sort((a, b) => b.probability - a.probability || a.id - b.id);

  let kept = entries;
  if ((strategy === 'top-k' || strategy === 'hybrid') && topK > 0) kept = kept.slice(0, Math.max(1, Math.min(topK, kept.length)));
  if ((strategy === 'min-p' || strategy === 'hybrid') && minP > 0) {
    const threshold = entries[0].probability * minP;
    kept = kept.filter((entry) => entry.probability >= threshold);
    if (!kept.length) kept = [entries[0]];
  }
  if ((strategy === 'nucleus' || strategy === 'hybrid') && topP < 1) {
    let cumulative = 0;
    let count = 0;
    while (count < kept.length) {
      cumulative += kept[count].probability;
      count += 1;
      if (cumulative >= topP) break;
    }
    kept = kept.slice(0, Math.max(1, count));
  }
  const keptTotal = kept.reduce((sum, entry) => sum + entry.probability, 0);
  for (const entry of kept) entry.probability /= keptTotal;
  let threshold = random();
  for (const entry of kept) {
    threshold -= entry.probability;
    if (threshold <= 0) return entry.id;
  }
  return kept.at(-1).id;
}

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

function structuralA1(eventType) {
  return eventType === CompoundEventType.PROGRAM || eventType === CompoundEventType.BANK;
}

export class CompoundDecodingRuntime {
  constructor(baseRuntime, options = {}) {
    if (!baseRuntime?.advance || !baseRuntime?.decodePrefix) throw new Error('base Compound runtime is required');
    this.baseRuntime = baseRuntime;
    this.configure(options);
  }

  configure(options = {}) {
    const merged = {
      strategy: 'nucleus',
      temperature: 0.85,
      structureTemperature: 1.05,
      topP: 0.92,
      topK: 0,
      minP: 0,
      ...(this.options || {}),
      ...options,
    };
    validateOptions(merged);
    this.options = merged;
    return this;
  }

  loadModels(options) { return this.baseRuntime.loadModels(options); }
  advance(record, state) { return this.baseRuntime.advance(record, state); }
  decodePrefix(ctx, prefix) { return this.baseRuntime.decodePrefix(ctx, prefix); }

  async sampleNextRecord(ctx, options = {}) {
    const cfg = { ...this.options, ...options };
    cfg.structureTemperature = options.structureTemperature ?? this.options.structureTemperature;
    cfg.strategy = options.strategy ?? this.options.strategy;
    cfg.topK = options.topK ?? this.options.topK;
    cfg.minP = options.minP ?? this.options.minP;
    validateOptions(cfg);
    const random = cfg.random || Math.random;
    const normal = cfg.normal || null;
    const prefix = { et: 0, ch: 0, dn: 0, a1: 0, a2: 0, vn: 0, dur: 0 };
    const pick = (logits, allowed, temperature) => sampleCategoricalAdvanced(logits, allowed, {
      strategy: cfg.strategy,
      temperature,
      topP: cfg.topP,
      topK: cfg.topK,
      minP: cfg.minP,
      random,
    });

    let heads = await this.decodePrefix(ctx, prefix);
    const eventType = pick(categoricalAt(heads, 'event_type_logits', 0, 10), ALL_EVENT_TYPES, cfg.structureTemperature);
    prefix.et = eventType;

    heads = await this.decodePrefix(ctx, prefix);
    const channel = [CompoundEventType.TEMPO, CompoundEventType.TIME_SIGNATURE].includes(eventType)
      ? 0
      : pick(categoricalAt(heads, 'channel_logits', 1, 16), ALL_CHANNELS, cfg.structureTemperature);
    prefix.ch = channel;

    heads = await this.decodePrefix(ctx, prefix);
    const deltaNorm = sampleGaussian(scalarHeadAt(heads, 'delta_mean', 2), scalarHeadAt(heads, 'delta_log_scale', 2), cfg.temperature, { random, normal });
    prefix.dn = deltaNorm;
    const delta = quantizeTime(Math.max(0, bankersRound(deltaNorm * 1536)));

    heads = await this.decodePrefix(ctx, prefix);
    const a1Temperature = structuralA1(eventType) ? cfg.structureTemperature : cfg.temperature;
    const a1 = pick(categoricalAt(heads, 'a1_logits', 3, 1024), allowedA1Ids(eventType), a1Temperature);
    prefix.a1 = a1;

    heads = await this.decodePrefix(ctx, prefix);
    const a2Allowed = allowedA2Ids(eventType);
    const a2 = a2Allowed.length === 1 ? a2Allowed[0] : pick(categoricalAt(heads, 'a2_logits', 4, 1024), a2Allowed, cfg.structureTemperature);
    prefix.a2 = a2;

    heads = await this.decodePrefix(ctx, prefix);
    let velocityNorm = 0; let velocity = 0;
    if (eventType === CompoundEventType.NOTE) {
      velocityNorm = sampleGaussian(scalarHeadAt(heads, 'velocity_mean', 5), scalarHeadAt(heads, 'velocity_log_scale', 5), cfg.temperature, { random, normal });
      velocity = Math.max(1, Math.min(127, bankersRound(velocityNorm * 127)));
    }
    prefix.vn = velocityNorm;

    heads = await this.decodePrefix(ctx, prefix);
    let durationNorm = 0; let duration = { coarse: 0, residual: 0 };
    if (eventType === CompoundEventType.NOTE) {
      durationNorm = sampleGaussian(scalarHeadAt(heads, 'duration_mean', 6), scalarHeadAt(heads, 'duration_log_scale', 6), cfg.temperature, { random, normal });
      duration = quantizeTime(Math.max(1, bankersRound(durationNorm * 1536)));
    }
    prefix.dur = durationNorm;

    heads = await this.decodePrefix(ctx, prefix);
    let control = { coarse: 0, residual: 0 };
    if (CONTROL_TYPES.has(eventType)) {
      const controlNorm = sampleGaussian(scalarHeadAt(heads, 'control_mean', 7), scalarHeadAt(heads, 'control_log_scale', 7), cfg.temperature, { random, normal });
      control = quantizeUnsigned(bankersRound(controlNorm * 16383), { maximum: 16383 });
    }

    const record = buildRecord({ eventType, channel, delta, a1, a2, velocity, duration, control });
    validateCompoundRecord(record);
    return record;
  }

  async generate({ primerRecords = [], maxNewEvents = 256, onProgress = null, ...options } = {}) {
    if (!Number.isInteger(maxNewEvents) || maxNewEvents < 0) throw new Error('maxNewEvents must be a non-negative integer');
    let state = createInitialStreamState();
    const records = (primerRecords.length ? primerRecords : [DEFAULT_SEED_RECORD]).map((record) => Array.from(record, Number));
    let ctx = null;
    for (const record of records) {
      const advanced = await this.advance(record, state); ctx = advanced.ctx; state = advanced.state;
    }
    if (!ctx) throw new Error('generation requires at least one seed record');
    for (let i = 0; i < maxNewEvents; i += 1) {
      const record = await this.sampleNextRecord(ctx, options);
      records.push(record);
      const advanced = await this.advance(record, state); ctx = advanced.ctx; state = advanced.state;
      if (typeof onProgress === 'function') onProgress({ generated: i + 1, total: maxNewEvents, record });
    }
    return { records, state, ctx };
  }
}
