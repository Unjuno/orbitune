import { canonicalizeCompoundEvents } from './compound-midi.mjs';
import { CompoundEventType } from './compound-runtime.mjs';
import {
  annotateNotesWithGmState,
  midiPitchToFrequency,
} from './compound-gm-synth.mjs';
import { SampledSoundFontSynth } from './compound-soundfont-player.mjs';

export { midiPitchToFrequency };

export function buildPreviewSchedule(events, { defaultBpm = 120 } = {}) {
  if (!(defaultBpm > 0)) throw new Error('defaultBpm must be positive');
  const canonical = canonicalizeCompoundEvents(events);
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
  constructor({ soundFontSynth = null, soundFontSynthFactory = () => new SampledSoundFontSynth() } = {}) {
    this.soundFontSynth = soundFontSynth;
    this.soundFontSynthFactory = soundFontSynthFactory;
  }

  get synth() {
    this.soundFontSynth ||= this.soundFontSynthFactory();
    return this.soundFontSynth;
  }

  async prepare() {
    await this.synth.ensureStarted();
    return this.synth.release;
  }

  async play(events) {
    const noteCount = buildPreviewSchedule(events).length;
    if (!noteCount) throw new Error('generated MIDI contains no NOTE events to preview');
    this.stop();
    const result = await this.synth.schedule(events);
    return result.noteCount;
  }

  stop() {
    if (!this.soundFontSynth) return;
    this.soundFontSynth.stop({ hard: true });
  }

  destroy() {
    if (!this.soundFontSynth) return;
    this.soundFontSynth.destroy();
    this.soundFontSynth = null;
  }
}
