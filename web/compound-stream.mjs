import { createInitialStreamState, DEFAULT_SEED_RECORD } from './compound-runtime.mjs';

function checkAbort(signal) {
  if (signal?.aborted) throw new DOMException('Streaming generation aborted', 'AbortError');
}

export class CompoundStreamSession {
  constructor(runtime, { temperature = 0.85, topP = 0.92, random = Math.random, normal = null } = {}) {
    if (!runtime?.advance || !runtime?.sampleNextRecord) throw new Error('Compound runtime with advance/sampleNextRecord is required');
    this.runtime = runtime;
    this.temperature = temperature;
    this.topP = topP;
    this.random = random;
    this.normal = normal;
    this.reset();
  }

  reset() {
    this.state = createInitialStreamState();
    this.ctx = null;
    this.started = false;
    this.generated = 0;
  }

  async start({ primerRecords = [], signal = null } = {}) {
    this.reset();
    const records = (primerRecords.length ? primerRecords : [DEFAULT_SEED_RECORD]).map((record) => Array.from(record, Number));
    for (const record of records) {
      checkAbort(signal);
      const advanced = await this.runtime.advance(record, this.state);
      this.ctx = advanced.ctx;
      this.state = advanced.state;
    }
    if (!this.ctx) throw new Error('streaming generation requires at least one seed record');
    this.started = true;
    return this.snapshot();
  }

  snapshot() {
    return { state: this.state, ctx: this.ctx, generated: this.generated, started: this.started };
  }

  async generateOne({ signal = null } = {}) {
    if (!this.started || !this.ctx) throw new Error('streaming session is not started');
    checkAbort(signal);
    const record = await this.runtime.sampleNextRecord(this.ctx, {
      temperature: this.temperature,
      topP: this.topP,
      random: this.random,
      normal: this.normal,
    });
    checkAbort(signal);
    const advanced = await this.runtime.advance(record, this.state);
    this.ctx = advanced.ctx;
    this.state = advanced.state;
    this.generated += 1;
    return record;
  }

  async generateBatch(count = 16, { signal = null, onRecord = null } = {}) {
    if (!Number.isInteger(count) || count <= 0) throw new Error('streaming batch count must be a positive integer');
    const records = [];
    for (let index = 0; index < count; index += 1) {
      const record = await this.generateOne({ signal });
      records.push(record);
      if (typeof onRecord === 'function') onRecord({ record, generated: this.generated, batchIndex: index });
    }
    return { records, ...this.snapshot() };
  }
}
