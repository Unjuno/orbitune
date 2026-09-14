import { CompoundEventType } from './compound-runtime.mjs';

export const GM_PERCUSSION_CHANNEL = 9;
export const GM_FAMILY_NAMES = Object.freeze([
  'piano', 'chromatic-percussion', 'organ', 'guitar',
  'bass', 'strings', 'ensemble', 'brass',
  'reed', 'pipe', 'synth-lead', 'synth-pad',
  'synth-fx', 'ethnic', 'percussive', 'sound-fx',
]);

export function midiPitchToFrequency(pitch) {
  return 440 * 2 ** ((Number(pitch) - 69) / 12);
}

export function gmProgramFamily(program = 0) {
  const normalized = Math.max(0, Math.min(127, Number(program) || 0));
  return GM_FAMILY_NAMES[Math.floor(normalized / 8)];
}

export function createGmChannelState() {
  return Array.from({ length: 16 }, () => ({ program: 0, bankMsb: 0, bankLsb: 0 }));
}

export function cloneGmChannelState(state = createGmChannelState()) {
  if (!Array.isArray(state) || state.length !== 16) throw new Error('GM channel state must contain 16 channels');
  return state.map((entry) => ({
    program: Math.max(0, Math.min(127, Number(entry?.program) || 0)),
    bankMsb: Math.max(0, Math.min(127, Number(entry?.bankMsb) || 0)),
    bankLsb: Math.max(0, Math.min(127, Number(entry?.bankLsb) || 0)),
  }));
}

export function annotateNotesWithGmState(events, { initialState } = {}) {
  const state = cloneGmChannelState(initialState || createGmChannelState());
  const notes = [];
  for (const event of events) {
    const channel = Math.max(0, Math.min(15, Number(event.channel) || 0));
    if (event.type === CompoundEventType.BANK) {
      state[channel] = { ...state[channel], bankMsb: event.a1, bankLsb: event.a2 };
    } else if (event.type === CompoundEventType.PROGRAM) {
      state[channel] = { ...state[channel], program: event.a1 };
    } else if (event.type === CompoundEventType.NOTE) {
      const voice = state[channel];
      notes.push({
        ...event,
        program: voice.program,
        bankMsb: voice.bankMsb,
        bankLsb: voice.bankLsb,
        percussion: channel === GM_PERCUSSION_CHANNEL,
        family: channel === GM_PERCUSSION_CHANNEL ? 'drums' : gmProgramFamily(voice.program),
      });
    }
  }
  return { notes, finalState: state };
}

function track(nodes, node) {
  if (node) nodes.push(node);
  return node;
}

function makeGain(context, nodes, destination) {
  const gain = track(nodes, context.createGain());
  gain.connect(destination || context.destination);
  return gain;
}

function scheduleEnvelope(gain, start, end, peak, { attack = 0.01, release = 0.05, sustain = 0.75 } = {}) {
  const safeEnd = Math.max(start + 0.025, end);
  const attackEnd = Math.min(safeEnd - 0.01, start + Math.max(0.002, attack));
  const releaseStart = Math.max(attackEnd + 0.001, safeEnd - Math.max(0.01, release));
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, peak), attackEnd);
  gain.gain.setValueAtTime(Math.max(0.0002, peak * sustain), releaseStart);
  gain.gain.exponentialRampToValueAtTime(0.0001, safeEnd);
  return safeEnd;
}

function addOscillator(context, nodes, destination, {
  type = 'sine', frequency, detune = 0, gain = 0.05, start, end,
  attack = 0.01, release = 0.05, sustain = 0.75,
} = {}) {
  const osc = track(nodes, context.createOscillator());
  const amp = makeGain(context, nodes, destination);
  osc.type = type;
  osc.frequency.setValueAtTime(frequency, start);
  if (osc.detune?.setValueAtTime) osc.detune.setValueAtTime(detune, start);
  const safeEnd = scheduleEnvelope(amp, start, end, gain, { attack, release, sustain });
  osc.connect(amp);
  osc.start(start);
  osc.stop(safeEnd + 0.02);
  return osc;
}

function createNoiseSource(context, nodes, durationSeconds = 1) {
  if (!context.createBuffer || !context.createBufferSource) return null;
  const length = Math.max(1, Math.ceil(context.sampleRate * Math.max(0.02, durationSeconds)));
  const buffer = context.createBuffer(1, length, context.sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < channel.length; i += 1) channel[i] = Math.random() * 2 - 1;
  const source = track(nodes, context.createBufferSource());
  source.buffer = buffer;
  return source;
}

function addNoise(context, nodes, destination, {
  start, end, gain = 0.04, attack = 0.002, release = 0.04,
  highpass = null, lowpass = null,
} = {}) {
  const source = createNoiseSource(context, nodes, end - start + 0.1);
  if (!source) return null;
  let output = source;
  if (context.createBiquadFilter && highpass != null) {
    const filter = track(nodes, context.createBiquadFilter());
    filter.type = 'highpass';
    filter.frequency.setValueAtTime(highpass, start);
    output.connect(filter); output = filter;
  }
  if (context.createBiquadFilter && lowpass != null) {
    const filter = track(nodes, context.createBiquadFilter());
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(lowpass, start);
    output.connect(filter); output = filter;
  }
  const amp = makeGain(context, nodes, destination);
  const safeEnd = scheduleEnvelope(amp, start, end, gain, { attack, release, sustain: 0.35 });
  output.connect(amp);
  source.start(start);
  source.stop(safeEnd + 0.02);
  return source;
}

function scheduleDrum(context, note, start, end, destination) {
  const nodes = [];
  const pitch = note.pitch;
  const velocity = Math.max(1, Math.min(127, note.velocity || 80));
  const level = velocity / 127;
  const shortEnd = Math.min(end, start + 0.45);

  if (pitch === 35 || pitch === 36) {
    const osc = track(nodes, context.createOscillator());
    const amp = makeGain(context, nodes, destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(145, start);
    if (osc.frequency.exponentialRampToValueAtTime) osc.frequency.exponentialRampToValueAtTime(48, start + 0.12);
    const safeEnd = scheduleEnvelope(amp, start, Math.min(shortEnd, start + 0.28), 0.18 * level, { attack: 0.002, release: 0.12, sustain: 0.25 });
    osc.connect(amp); osc.start(start); osc.stop(safeEnd + 0.02);
  } else if ([38, 40].includes(pitch)) {
    addNoise(context, nodes, destination, { start, end: Math.min(shortEnd, start + 0.22), gain: 0.11 * level, highpass: 900, lowpass: 9000 });
    addOscillator(context, nodes, destination, { type: 'triangle', frequency: 185, gain: 0.045 * level, start, end: Math.min(shortEnd, start + 0.16), attack: 0.002, release: 0.08, sustain: 0.2 });
  } else if ([42, 44, 46, 49, 51, 52, 55, 57, 59].includes(pitch)) {
    const duration = pitch === 46 || pitch >= 49 ? 0.45 : 0.12;
    addNoise(context, nodes, destination, { start, end: Math.min(end, start + duration), gain: 0.075 * level, highpass: 5000 });
  } else if (pitch >= 41 && pitch <= 50) {
    const tomFrequency = 70 + (pitch - 41) * 14;
    addOscillator(context, nodes, destination, { type: 'sine', frequency: tomFrequency, gain: 0.11 * level, start, end: Math.min(shortEnd, start + 0.3), attack: 0.002, release: 0.13, sustain: 0.25 });
  } else {
    addNoise(context, nodes, destination, { start, end: Math.min(shortEnd, start + 0.16), gain: 0.055 * level, highpass: 1200 });
  }
  return nodes;
}

function scheduleProgramVoice(context, note, start, end, destination) {
  const nodes = [];
  const frequency = note.frequency || midiPitchToFrequency(note.pitch);
  const velocity = Math.max(1, Math.min(127, note.velocity || 80));
  const level = velocity / 127;
  const family = note.family || gmProgramFamily(note.program);

  const recipes = {
    piano: [
      { type: 'triangle', ratio: 1, gain: 0.085, attack: 0.003, release: 0.16, sustain: 0.42 },
      { type: 'sine', ratio: 2, gain: 0.028, attack: 0.003, release: 0.11, sustain: 0.18 },
    ],
    'chromatic-percussion': [
      { type: 'sine', ratio: 1, gain: 0.08, attack: 0.002, release: 0.22, sustain: 0.24 },
      { type: 'sine', ratio: 3, gain: 0.03, attack: 0.002, release: 0.1, sustain: 0.08 },
    ],
    organ: [
      { type: 'sine', ratio: 1, gain: 0.065, attack: 0.018, release: 0.07, sustain: 0.95 },
      { type: 'square', ratio: 2, gain: 0.018, attack: 0.018, release: 0.07, sustain: 0.9 },
    ],
    guitar: [
      { type: 'triangle', ratio: 1, gain: 0.075, attack: 0.002, release: 0.14, sustain: 0.32 },
      { type: 'sine', ratio: 2, gain: 0.022, attack: 0.002, release: 0.09, sustain: 0.12 },
    ],
    bass: [
      { type: 'square', ratio: 1, gain: 0.055, attack: 0.008, release: 0.09, sustain: 0.72 },
      { type: 'sine', ratio: 0.5, gain: 0.035, attack: 0.008, release: 0.11, sustain: 0.75 },
    ],
    strings: [
      { type: 'sawtooth', ratio: 1, gain: 0.04, attack: 0.11, release: 0.22, sustain: 0.78 },
      { type: 'triangle', ratio: 1, detune: 7, gain: 0.028, attack: 0.13, release: 0.25, sustain: 0.82 },
    ],
    ensemble: [
      { type: 'sawtooth', ratio: 1, detune: -5, gain: 0.035, attack: 0.09, release: 0.2, sustain: 0.75 },
      { type: 'sawtooth', ratio: 1, detune: 6, gain: 0.035, attack: 0.09, release: 0.2, sustain: 0.75 },
    ],
    brass: [
      { type: 'sawtooth', ratio: 1, gain: 0.055, attack: 0.045, release: 0.11, sustain: 0.72 },
      { type: 'square', ratio: 1, gain: 0.018, attack: 0.04, release: 0.1, sustain: 0.55 },
    ],
    reed: [
      { type: 'square', ratio: 1, gain: 0.045, attack: 0.035, release: 0.09, sustain: 0.7 },
      { type: 'sine', ratio: 2, gain: 0.015, attack: 0.04, release: 0.08, sustain: 0.5 },
    ],
    pipe: [
      { type: 'sine', ratio: 1, gain: 0.07, attack: 0.025, release: 0.1, sustain: 0.88 },
      { type: 'sine', ratio: 2, gain: 0.014, attack: 0.025, release: 0.1, sustain: 0.75 },
    ],
    'synth-lead': [
      { type: 'sawtooth', ratio: 1, gain: 0.05, attack: 0.012, release: 0.08, sustain: 0.82 },
      { type: 'square', ratio: 1, detune: 9, gain: 0.02, attack: 0.012, release: 0.08, sustain: 0.68 },
    ],
    'synth-pad': [
      { type: 'triangle', ratio: 1, detune: -8, gain: 0.035, attack: 0.2, release: 0.35, sustain: 0.82 },
      { type: 'sine', ratio: 1, detune: 8, gain: 0.035, attack: 0.24, release: 0.38, sustain: 0.86 },
    ],
    'synth-fx': [
      { type: 'sawtooth', ratio: 1, detune: -14, gain: 0.035, attack: 0.08, release: 0.2, sustain: 0.65 },
      { type: 'triangle', ratio: 2, detune: 14, gain: 0.02, attack: 0.1, release: 0.2, sustain: 0.55 },
    ],
    ethnic: [
      { type: 'triangle', ratio: 1, gain: 0.065, attack: 0.003, release: 0.16, sustain: 0.35 },
      { type: 'sine', ratio: 2, gain: 0.02, attack: 0.003, release: 0.1, sustain: 0.16 },
    ],
    percussive: [
      { type: 'square', ratio: 1, gain: 0.055, attack: 0.002, release: 0.12, sustain: 0.22 },
      { type: 'sine', ratio: 3, gain: 0.018, attack: 0.002, release: 0.07, sustain: 0.08 },
    ],
    'sound-fx': [
      { type: 'sawtooth', ratio: 1, detune: -25, gain: 0.03, attack: 0.02, release: 0.18, sustain: 0.5 },
      { type: 'square', ratio: 0.5, detune: 18, gain: 0.025, attack: 0.03, release: 0.2, sustain: 0.45 },
    ],
  };

  for (const recipe of recipes[family] || recipes.piano) {
    addOscillator(context, nodes, destination, {
      type: recipe.type,
      frequency: frequency * recipe.ratio,
      detune: recipe.detune || 0,
      gain: recipe.gain * level,
      start,
      end,
      attack: recipe.attack,
      release: recipe.release,
      sustain: recipe.sustain,
    });
  }
  return nodes;
}

export function scheduleGmVoice(context, note, { start, end, destination = context.destination } = {}) {
  if (!(end > start)) end = start + 0.03;
  return note.percussion
    ? scheduleDrum(context, note, start, end, destination)
    : scheduleProgramVoice(context, note, start, end, destination);
}
