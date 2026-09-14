import { canonicalizeCompoundEvents } from './compound-midi.mjs';
import { CompoundEventType } from './compound-runtime.mjs';
import {
  annotateNotesWithGmState,
  midiPitchToFrequency,
  scheduleGmVoice,
} from './compound-gm-synth.mjs';

export { midiPitchToFrequency };

export function buildPreviewSchedule(events, { defaultBpm = 120 } = {}) {
  if (!(defaultBpm > 0)) throw new Error('defaultBpm must be positive');
  const canonical = canonicalizeCompoundEvents(events);
  // Python's MIDI writer sorts same-tick TEMPO meta messages by encoded bytes.
  // Faster tempos (smaller microseconds/qn) come first, so the slowest BPM is
  // the final effective tempo at that tick. Mirror that for preview timing.
  const tempoEvents = canonical.filter((event) => event.type === CompoundEventType.TEMPO).sort((a, b) => a.step - b.step || b.a1 - a.a1);
  const segments = [{ step: 0, seconds: 0, bpm: defaultBpm }];
  for (const tempo of tempoEvents) {
    const previous = segments.at(-1);
    const seconds = previous.seconds + (tempo.step - previous.step) / 96 * 60 / previous.bpm;
    if (tempo.step === previous.step) { previous.bpm = tempo.a1; }
    else segments.push({ step: tempo.step, seconds, bpm: tempo.a1 });
  }
  function secondsAt(step) {
    let segment = segments[0];
    for (const candidate of segments) { if (candidate.step > step) break; segment = candidate; }
    return segment.seconds + (step - segment.step) / 96 * 60 / segment.bpm;
  }
  const { notes } = annotateNotesWithGmState(canonical);
  return notes.map((event) => ({
    start: secondsAt(event.step),
    end: secondsAt(event.step + event.a2),
    channel: event.channel,
    pitch: event.a1,
    velocity: event.a3,
    frequency: midiPitchToFrequency(event.a1),
    program: event.program,
    bankMsb: event.bankMsb,
    bankLsb: event.bankLsb,
    family: event.family,
    percussion: event.percussion,
  }));
}

export class CompoundPreviewPlayer {
  constructor({ audioContextFactory = () => new (globalThis.AudioContext || globalThis.webkitAudioContext)() } = {}) {
    this.audioContextFactory = audioContextFactory;
    this.context = null;
    this.nodes = [];
  }

  async play(events) {
    this.stop();
    const schedule = buildPreviewSchedule(events);
    if (!schedule.length) throw new Error('generated MIDI contains no NOTE events to preview');
    this.context ||= this.audioContextFactory();
    if (this.context.state === 'suspended') await this.context.resume();
    const origin = this.context.currentTime + 0.05;
    for (const note of schedule.slice(0, 4096)) {
      const start = origin + note.start;
      const end = Math.max(start + 0.02, origin + note.end);
      this.nodes.push(...scheduleGmVoice(this.context, note, { start, end }));
    }
    return schedule.length;
  }

  stop() {
    for (const node of this.nodes) {
      try { node.stop?.(); } catch {}
      try { node.disconnect?.(); } catch {}
    }
    this.nodes = [];
  }
}
