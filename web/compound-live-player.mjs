import { canonicalizeCompoundEvents } from './compound-midi.mjs';
import { CompoundEventType } from './compound-runtime.mjs';
import {
  annotateNotesWithGmState,
  createGmChannelState,
  midiPitchToFrequency,
} from './compound-gm-synth.mjs';
import { SampledSoundFontSynth } from './compound-soundfont-player.mjs';

export function buildStreamingChunkTiming(events, { defaultBpm = 120, channelState = null } = {}) {
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
  const resolved = annotateNotesWithGmState(canonical, { initialState: channelState || createGmChannelState() });
  const notes = resolved.notes.map((event) => ({
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
  const spanStep = canonical.reduce((maximum, event) => Math.max(maximum, event.step), 0);
  return {
    notes,
    spanSeconds: secondsAt(spanStep),
    finalBpm: segments.at(-1).bpm,
    spanStep,
    finalChannelState: resolved.finalState,
  };
}

export class CompoundStreamingPreviewPlayer {
  constructor({
    soundFontSynth = null,
    soundFontSynthFactory = () => new SampledSoundFontSynth(),
    leadSeconds = 0.12,
  } = {}) {
    this.soundFontSynth = soundFontSynth;
    this.soundFontSynthFactory = soundFontSynthFactory;
    this.leadSeconds = leadSeconds;
    this.cursorTime = null;
    this.channelState = createGmChannelState();
  }

  get synth() {
    this.soundFontSynth ||= this.soundFontSynthFactory();
    return this.soundFontSynth;
  }

  get context() { return this.soundFontSynth?.context || null; }

  async ensureStarted() {
    await this.synth.ensureStarted();
    if (this.cursorTime == null) this.cursorTime = this.synth.context.currentTime + this.leadSeconds;
    return this.synth.context;
  }

  prune() {}

  bufferedSeconds() {
    const context = this.context;
    if (!context || this.cursorTime == null) return 0;
    return Math.max(0, this.cursorTime - context.currentTime);
  }

  async append(events, { defaultBpm = 120 } = {}) {
    const context = await this.ensureStarted();
    const timing = buildStreamingChunkTiming(events, {
      defaultBpm,
      channelState: this.channelState,
    });
    this.channelState = timing.finalChannelState;
    const origin = Math.max(this.cursorTime ?? 0, context.currentTime + 0.04);
    const sampled = await this.synth.schedule(events, { origin, defaultBpm });
    this.cursorTime = origin + timing.spanSeconds;
    return {
      ...timing,
      scheduledNotes: sampled.noteCount,
      bufferedSeconds: this.bufferedSeconds(),
      soundFont: sampled.release?.display_name || sampled.release?.id || null,
    };
  }

  async pause() { await this.soundFontSynth?.pause(); }
  async resume() { await this.soundFontSynth?.resume(); }

  stop() {
    this.soundFontSynth?.stop({ hard: true });
    this.cursorTime = null;
    this.channelState = createGmChannelState();
  }

  destroy() {
    this.soundFontSynth?.destroy();
    this.soundFontSynth = null;
    this.cursorTime = null;
    this.channelState = createGmChannelState();
  }
}
