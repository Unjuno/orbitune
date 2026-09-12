import { canonicalizeCompoundEvents } from './compound-midi.mjs';
import { CompoundEventType } from './compound-runtime.mjs';
import { midiPitchToFrequency } from './compound-player.mjs';

export function buildStreamingChunkTiming(events, { defaultBpm = 120 } = {}) {
  if (!(defaultBpm > 0)) throw new Error('defaultBpm must be positive');
  const canonical = canonicalizeCompoundEvents(events);
  const tempos = canonical
    .filter((event) => event.type === CompoundEventType.TEMPO)
    .sort((a, b) => a.step - b.step || b.a1 - a.a1);
  const segments = [{ step: 0, seconds: 0, bpm: defaultBpm }];
  for (const tempo of tempos) {
    const previous = segments.at(-1);
    const seconds = previous.seconds + (tempo.step - previous.step) / 96 * 60 / previous.bpm;
    if (tempo.step === previous.step) previous.bpm = tempo.a1;
    else segments.push({ step: tempo.step, seconds, bpm: tempo.a1 });
  }
  function secondsAt(step) {
    let segment = segments[0];
    for (const candidate of segments) { if (candidate.step > step) break; segment = candidate; }
    return segment.seconds + (step - segment.step) / 96 * 60 / segment.bpm;
  }
  const notes = canonical.filter((event) => event.type === CompoundEventType.NOTE).map((event) => ({
    start: secondsAt(event.step),
    end: secondsAt(event.step + event.a2),
    channel: event.channel,
    pitch: event.a1,
    velocity: event.a3,
    frequency: midiPitchToFrequency(event.a1),
  }));
  const spanStep = canonical.reduce((maximum, event) => Math.max(maximum, event.step), 0);
  return { notes, spanSeconds: secondsAt(spanStep), finalBpm: segments.at(-1).bpm, spanStep };
}

export class CompoundStreamingPreviewPlayer {
  constructor({ audioContextFactory = () => new (globalThis.AudioContext || globalThis.webkitAudioContext)(), leadSeconds = 0.12 } = {}) {
    this.audioContextFactory = audioContextFactory;
    this.leadSeconds = leadSeconds;
    this.context = null;
    this.cursorTime = null;
    this.scheduled = [];
  }

  async ensureStarted() {
    this.context ||= this.audioContextFactory();
    if (this.context.state === 'suspended') await this.context.resume();
    if (this.cursorTime == null) this.cursorTime = this.context.currentTime + this.leadSeconds;
    return this.context;
  }

  prune() {
    if (!this.context) return;
    const cutoff = this.context.currentTime - 0.25;
    this.scheduled = this.scheduled.filter((entry) => entry.end > cutoff);
  }

  bufferedSeconds() {
    if (!this.context || this.cursorTime == null) return 0;
    return Math.max(0, this.cursorTime - this.context.currentTime);
  }

  async append(events, { defaultBpm = 120 } = {}) {
    const context = await this.ensureStarted();
    this.prune();
    const timing = buildStreamingChunkTiming(events, { defaultBpm });
    const origin = Math.max(this.cursorTime ?? 0, context.currentTime + 0.04);
    for (const note of timing.notes) {
      const osc = context.createOscillator(); const gain = context.createGain();
      osc.type = 'triangle'; osc.frequency.value = note.frequency;
      const amplitude = Math.min(0.12, Math.max(0.005, note.velocity / 127 * 0.08));
      const start = origin + note.start; const end = Math.max(start + 0.02, origin + note.end);
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(amplitude, start + 0.01);
      gain.gain.setValueAtTime(amplitude, Math.max(start + 0.011, end - 0.03));
      gain.gain.exponentialRampToValueAtTime(0.0001, end);
      osc.connect(gain); gain.connect(context.destination); osc.start(start); osc.stop(end + 0.01);
      this.scheduled.push({ node: osc, end: end + 0.01 });
    }
    this.cursorTime = origin + timing.spanSeconds;
    return { ...timing, scheduledNotes: timing.notes.length, bufferedSeconds: this.bufferedSeconds() };
  }

  async pause() { if (this.context?.state === 'running') await this.context.suspend(); }
  async resume() { if (this.context?.state === 'suspended') await this.context.resume(); }

  stop() {
    for (const entry of this.scheduled) { try { entry.node.stop(); } catch {} try { entry.node.disconnect(); } catch {} }
    this.scheduled = [];
    this.cursorTime = null;
  }
}
