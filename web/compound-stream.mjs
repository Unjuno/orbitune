import {
  CompoundEventType,
  DEFAULT_SEED_RECORD,
  createInitialStreamState,
  dequantizeTime,
  validateCompoundRecord,
} from './compound-runtime.mjs';
import { midiPitchToFrequency } from './compound-player.mjs';

const TICKS_PER_QUARTER = 96;

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

export async function createStreamingGenerator(runtime, { primerRecords = [] } = {}) {
  if (!runtime?.advance || !runtime?.sampleNextRecord) throw new Error('runtime with advance() and sampleNextRecord() is required');
  let state = createInitialStreamState();
  const seeds = (primerRecords.length ? primerRecords : [DEFAULT_SEED_RECORD]).map((record) => Array.from(record, Number));
  let ctx = null;
  for (const record of seeds) {
    const advanced = await runtime.advance(record, state);
    ctx = advanced.ctx;
    state = advanced.state;
  }
  if (!ctx) throw new Error('streaming generation requires at least one seed record');
  return {
    get state() { return state; },
    get ctx() { return ctx; },
    async next(options = {}) {
      const record = await runtime.sampleNextRecord(ctx, options);
      const advanced = await runtime.advance(record, state);
      ctx = advanced.ctx;
      state = advanced.state;
      return Array.from(record, Number);
    },
  };
}

export class BoundedRecordQueue {
  constructor(maxRecords = 512) {
    if (!Number.isInteger(maxRecords) || maxRecords <= 0) throw new Error('maxRecords must be a positive integer');
    this.maxRecords = maxRecords;
    this.records = [];
  }

  push(record) {
    this.records.push(Array.from(record, Number));
    if (this.records.length > this.maxRecords) this.records.splice(0, this.records.length - this.maxRecords);
  }

  clear() { this.records.length = 0; }
  snapshot() { return this.records.map((record) => [...record]); }
  get size() { return this.records.length; }
}

export class CompoundStreamingPreviewSink {
  constructor({
    audioContextFactory = () => new (globalThis.AudioContext || globalThis.webkitAudioContext)(),
    leadSeconds = 0.20,
    maxActiveOscillators = 256,
  } = {}) {
    this.audioContextFactory = audioContextFactory;
    this.leadSeconds = leadSeconds;
    this.maxActiveOscillators = maxActiveOscillators;
    this.context = null;
    this.nodes = new Set();
    this.cursorTime = 0;
    this.bpm = 120;
  }

  async start() {
    this.stop();
    this.context ||= this.audioContextFactory();
    if (this.context.state === 'suspended') await this.context.resume();
    this.cursorTime = this.context.currentTime + this.leadSeconds;
    this.bpm = 120;
  }

  lookaheadSeconds() {
    if (!this.context) return 0;
    return Math.max(0, this.cursorTime - this.context.currentTime);
  }

  async enqueue(record) {
    if (!this.context) throw new Error('streaming preview sink is not started');
    validateCompoundRecord(record);
    const deltaTicks = dequantizeTime({ coarse: Number(record[2]), residual: Number(record[3]) });
    const deltaSeconds = deltaTicks / TICKS_PER_QUARTER * 60 / this.bpm;
    const minimumTime = this.context.currentTime + 0.02;
    this.cursorTime = Math.max(this.cursorTime + deltaSeconds, minimumTime);

    const type = Number(record[0]);
    if (type === CompoundEventType.TEMPO) {
      this.bpm = Number(record[4]);
      return;
    }
    if (type !== CompoundEventType.NOTE) return;

    if (this.nodes.size >= this.maxActiveOscillators) {
      const oldest = this.nodes.values().next().value;
      if (oldest) {
        try { oldest.stop(); } catch {}
        try { oldest.disconnect(); } catch {}
        this.nodes.delete(oldest);
      }
    }

    const pitch = Number(record[4]);
    const velocity = Number(record[6]);
    const durationTicks = Math.max(1, dequantizeTime({ coarse: Number(record[8]), residual: Number(record[9]) }));
    const durationSeconds = Math.max(0.02, durationTicks / TICKS_PER_QUARTER * 60 / this.bpm);
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.type = 'triangle';
    osc.frequency.value = midiPitchToFrequency(pitch);
    const amplitude = Math.min(0.12, Math.max(0.005, velocity / 127 * 0.08));
    const start = this.cursorTime;
    const end = start + durationSeconds;
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(amplitude, start + 0.01);
    gain.gain.setValueAtTime(amplitude, Math.max(start + 0.011, end - 0.03));
    gain.gain.exponentialRampToValueAtTime(0.0001, end);
    osc.connect(gain);
    gain.connect(this.context.destination);
    osc.onended = () => {
      try { osc.disconnect(); } catch {}
      try { gain.disconnect(); } catch {}
      this.nodes.delete(osc);
    };
    osc.start(start);
    osc.stop(end + 0.01);
    this.nodes.add(osc);
  }

  stop() {
    for (const node of this.nodes) {
      try { node.stop(); } catch {}
      try { node.disconnect(); } catch {}
    }
    this.nodes.clear();
    if (this.context) this.cursorTime = this.context.currentTime;
  }
}

export class InfiniteCompoundStream {
  constructor({
    runtime,
    sink,
    targetLookaheadSeconds = 3,
    maxRecentRecords = 512,
    idleDelayMs = 20,
    onRecord = null,
  } = {}) {
    if (!runtime?.advance || !runtime?.sampleNextRecord) throw new Error('Compound runtime is required');
    if (!sink?.enqueue || !sink?.lookaheadSeconds) throw new Error('stream sink is required');
    if (!(targetLookaheadSeconds > 0)) throw new Error('targetLookaheadSeconds must be positive');
    this.runtime = runtime;
    this.sink = sink;
    this.targetLookaheadSeconds = targetLookaheadSeconds;
    this.idleDelayMs = idleDelayMs;
    this.onRecord = onRecord;
    this.recent = new BoundedRecordQueue(maxRecentRecords);
    this.generator = null;
    this.running = false;
    this.loopPromise = null;
    this.generated = 0;
    this.options = {};
  }

  async start({ primerRecords = [], temperature = 0.85, topP = 0.92, random = Math.random, normal = null } = {}) {
    if (this.running) return;
    this.recent.clear();
    this.generated = 0;
    this.options = { temperature, topP, random, normal };
    this.generator = await createStreamingGenerator(this.runtime, { primerRecords });
    if (this.sink.start) await this.sink.start();
    this.running = true;
    this.loopPromise = this.#loop();
  }

  async pumpOnce() {
    if (!this.generator) throw new Error('stream is not initialized');
    if (this.sink.lookaheadSeconds() >= this.targetLookaheadSeconds) return false;
    const record = await this.generator.next(this.options);
    this.recent.push(record);
    await this.sink.enqueue(record);
    this.generated += 1;
    if (typeof this.onRecord === 'function') {
      this.onRecord({
        generated: this.generated,
        record: [...record],
        lookaheadSeconds: this.sink.lookaheadSeconds(),
        recentRecords: this.recent.size,
      });
    }
    return true;
  }

  async #loop() {
    let burst = 0;
    while (this.running) {
      try {
        const generated = await this.pumpOnce();
        burst = generated ? burst + 1 : 0;
        if (!generated || burst >= 32) {
          burst = 0;
          await sleep(generated ? 0 : this.idleDelayMs);
        }
      } catch (error) {
        this.running = false;
        if (typeof this.onRecord === 'function') this.onRecord({ error });
        break;
      }
    }
  }

  async stop() {
    this.running = false;
    if (this.loopPromise) await this.loopPromise;
    this.loopPromise = null;
    if (this.sink.stop) this.sink.stop();
  }

  recentRecords() { return this.recent.snapshot(); }
}
