import { CompoundEventType } from './compound-runtime.mjs';
import { TEMPORAL_RESOLUTION } from './compound-midi.mjs';
import { GM_PROGRAM_NAMES } from './compound-event-monitor.mjs';

const SVG_NS = 'http://www.w3.org/2000/svg';
const DEFAULT_WINDOW_BEATS = 16;
const DEFAULT_NOTE_LIMIT = 320;
const DEFAULT_MARKER_LIMIT = 160;
const DEFAULT_HISTORY_BEATS = 96;

const EVENT_NAMES = Object.freeze({
  [CompoundEventType.NOTE]: 'NOTE',
  [CompoundEventType.CC]: 'CC',
  [CompoundEventType.PROGRAM]: 'PROGRAM',
  [CompoundEventType.BANK]: 'BANK',
  [CompoundEventType.TEMPO]: 'TEMPO',
  [CompoundEventType.PEDAL]: 'PEDAL',
  [CompoundEventType.PITCH_BEND]: 'PITCH_BEND',
  [CompoundEventType.CHANNEL_PRESSURE]: 'CHANNEL_PRESSURE',
  [CompoundEventType.POLY_PRESSURE]: 'POLY_PRESSURE',
  [CompoundEventType.TIME_SIGNATURE]: 'TIME_SIGNATURE',
});

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function safeChannel(value) { return clamp(Number.isFinite(Number(value)) ? Math.trunc(Number(value)) : 0, 0, 15); }
function safeProgram(value) { return clamp(Number.isFinite(Number(value)) ? Math.trunc(Number(value)) : 0, 0, 127); }
function svgElement(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

export function midiNoteName(pitch) {
  pitch = clamp(Math.trunc(Number(pitch) || 0), 0, 127);
  const names = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
  return `${names[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

export function createDawState({ noteLimit = DEFAULT_NOTE_LIMIT, markerLimit = DEFAULT_MARKER_LIMIT } = {}) {
  return {
    channels: Array.from({ length: 16 }, (_, channel) => ({
      channel,
      program: 0,
      bankMsb: 0,
      bankLsb: 0,
      notes: 0,
      programExplicit: false,
    })),
    notes: [],
    markers: [],
    eventCounts: Object.fromEntries(Object.values(EVENT_NAMES).map((name) => [name, 0])),
    totalEvents: 0,
    maxStep: 0,
    relativeBaseStep: 0,
    nextNoteId: 1,
    selectedChannel: null,
    selectedNoteId: null,
    follow: true,
    viewportEndStep: 0,
    windowBeats: DEFAULT_WINDOW_BEATS,
    noteLimit,
    markerLimit,
  };
}

function instrumentFor(channel, program) {
  return channel === 9 ? 'Drum kit' : (GM_PROGRAM_NAMES[safeProgram(program)] || `Program ${safeProgram(program)}`);
}

function pruneBounded(state) {
  if (state.notes.length > state.noteLimit) state.notes.splice(0, state.notes.length - state.noteLimit);
  if (state.markers.length > state.markerLimit) state.markers.splice(0, state.markers.length - state.markerLimit);
  const keepAfter = Math.max(0, state.maxStep - DEFAULT_HISTORY_BEATS * TEMPORAL_RESOLUTION);
  while (state.notes.length && state.notes[0].step + state.notes[0].duration < keepAfter && state.notes.length > 64) state.notes.shift();
  while (state.markers.length && state.markers[0].step < keepAfter && state.markers.length > 32) state.markers.shift();
  if (state.selectedNoteId != null && !state.notes.some((note) => note.id === state.selectedNoteId)) state.selectedNoteId = null;
}

export function consumeDawEvents(state, events, { relative = false } = {}) {
  if (!Array.isArray(events)) throw new Error('events must be an array');
  const baseStep = relative ? state.relativeBaseStep : 0;
  let chunkMax = 0;
  for (const raw of events) {
    const localStep = Math.max(0, Math.trunc(Number(raw.step) || 0));
    const step = baseStep + localStep;
    chunkMax = Math.max(chunkMax, localStep);
    const type = Number(raw.type);
    const channel = safeChannel(raw.channel);
    const voice = state.channels[channel];
    const typeName = EVENT_NAMES[type] || `TYPE_${type}`;
    state.totalEvents += 1;
    state.eventCounts[typeName] = (state.eventCounts[typeName] || 0) + 1;
    state.maxStep = Math.max(state.maxStep, step);

    if (type === CompoundEventType.PROGRAM) {
      voice.program = safeProgram(raw.a1);
      voice.programExplicit = true;
      state.markers.push({ kind: 'PROGRAM', step, channel, value: voice.program, label: instrumentFor(channel, voice.program) });
    } else if (type === CompoundEventType.BANK) {
      voice.bankMsb = clamp(Math.trunc(Number(raw.a1) || 0), 0, 127);
      voice.bankLsb = clamp(Math.trunc(Number(raw.a2) || 0), 0, 127);
      state.markers.push({ kind: 'BANK', step, channel, value: `${voice.bankMsb}:${voice.bankLsb}`, label: `Bank ${voice.bankMsb}:${voice.bankLsb}` });
    } else if (type === CompoundEventType.NOTE) {
      const pitch = clamp(Math.trunc(Number(raw.a1) || 0), 0, 127);
      const duration = Math.max(1, Math.trunc(Number(raw.a2) || 1));
      const velocity = clamp(Math.trunc(Number(raw.a3) || 1), 1, 127);
      voice.notes += 1;
      const note = {
        id: state.nextNoteId++,
        step,
        duration,
        channel,
        pitch,
        velocity,
        program: voice.program,
        programExplicit: voice.programExplicit,
        bankMsb: voice.bankMsb,
        bankLsb: voice.bankLsb,
        instrument: instrumentFor(channel, voice.program),
      };
      state.notes.push(note);
      state.maxStep = Math.max(state.maxStep, step + duration);
    } else if ([CompoundEventType.CC, CompoundEventType.PEDAL, CompoundEventType.PITCH_BEND, CompoundEventType.TEMPO, CompoundEventType.TIME_SIGNATURE].includes(type)) {
      state.markers.push({ kind: typeName, step, channel, value: raw.a1, label: typeName });
    }
  }
  if (relative && events.length) state.relativeBaseStep = baseStep + chunkMax;
  else if (!relative) state.relativeBaseStep = state.maxStep;
  if (state.follow) state.viewportEndStep = state.maxStep;
  pruneBounded(state);
  return state;
}

export function selectedDawNote(state) {
  return state.notes.find((note) => note.id === state.selectedNoteId) || null;
}

export function dawSnapshot(state) {
  const activeChannels = state.channels.filter((channel) => channel.notes > 0 || channel.programExplicit);
  return {
    totalEvents: state.totalEvents,
    eventCounts: { ...state.eventCounts },
    maxStep: state.maxStep,
    relativeBaseStep: state.relativeBaseStep,
    selectedChannel: state.selectedChannel,
    selectedNote: selectedDawNote(state),
    follow: state.follow,
    windowBeats: state.windowBeats,
    notes: state.notes.slice(),
    markers: state.markers.slice(),
    channels: activeChannels.map((channel) => ({ ...channel, instrument: instrumentFor(channel.channel, channel.program) })),
  };
}

export function computeDawViewport(state) {
  const windowSteps = Math.max(TEMPORAL_RESOLUTION * 4, Math.trunc(state.windowBeats * TEMPORAL_RESOLUTION));
  const end = state.follow
    ? Math.max(windowSteps, state.maxStep + Math.round(TEMPORAL_RESOLUTION * 0.5))
    : clamp(Math.max(windowSteps, state.viewportEndStep), windowSteps, Math.max(windowSteps, state.maxStep + TEMPORAL_RESOLUTION));
  return { start: Math.max(0, end - windowSteps), end, span: windowSteps };
}

function visibleNotes(state, viewport) {
  return state.notes.filter((note) => {
    if (state.selectedChannel != null && note.channel !== state.selectedChannel) return false;
    return note.step + note.duration >= viewport.start && note.step <= viewport.end;
  });
}

function pitchBounds(notes) {
  if (!notes.length) return { min: 48, max: 84 };
  let min = Math.min(...notes.map((note) => note.pitch));
  let max = Math.max(...notes.map((note) => note.pitch));
  min = clamp(min - 2, 0, 127);
  max = clamp(max + 2, 0, 127);
  if (max - min < 23) {
    const pad = Math.ceil((23 - (max - min)) / 2);
    min = clamp(min - pad, 0, 127);
    max = clamp(min + 23, 0, 127);
    min = clamp(max - 23, 0, 127);
  }
  return { min, max };
}

function trackLabel(channel) {
  if (channel.channel === 9) return `Ch 10 · Drums`;
  const explicit = channel.programExplicit ? '' : ' (default)';
  return `Ch ${channel.channel + 1} · ${channel.program} ${channel.instrument}${explicit}`;
}

export class CompoundDawMonitor {
  constructor({
    summaryElement = null,
    tracksElement = null,
    rollElement = null,
    velocityElement = null,
    inspectorElement = null,
    followInput = null,
    zoomSelect = null,
    scrubInput = null,
  } = {}) {
    this.summaryElement = summaryElement;
    this.tracksElement = tracksElement;
    this.rollElement = rollElement;
    this.velocityElement = velocityElement;
    this.inspectorElement = inspectorElement;
    this.followInput = followInput;
    this.zoomSelect = zoomSelect;
    this.scrubInput = scrubInput;
    this.state = createDawState();
    this._resizeObserver = null;
    this._wireControls();
    if (typeof ResizeObserver === 'function' && this.rollElement) {
      this._resizeObserver = new ResizeObserver(() => this.render());
      this._resizeObserver.observe(this.rollElement);
    }
  }

  _wireControls() {
    this.followInput?.addEventListener('change', () => {
      this.state.follow = Boolean(this.followInput.checked);
      if (this.state.follow) this.state.viewportEndStep = this.state.maxStep;
      this.render();
    });
    this.zoomSelect?.addEventListener('change', () => {
      const beats = Number(this.zoomSelect.value);
      if (Number.isFinite(beats) && beats >= 4 && beats <= 128) this.state.windowBeats = beats;
      this.render();
    });
    this.scrubInput?.addEventListener('input', () => {
      this.state.follow = false;
      if (this.followInput) this.followInput.checked = false;
      this.state.viewportEndStep = Number(this.scrubInput.value) || this.state.maxStep;
      this.render();
    });
  }

  reset() {
    const windowBeats = this.state.windowBeats;
    this.state = createDawState();
    this.state.windowBeats = windowBeats;
    if (this.followInput) this.followInput.checked = true;
    this.render();
  }

  consume(events, { relative = false } = {}) {
    consumeDawEvents(this.state, events, { relative });
    this.render();
    return this.snapshot();
  }

  snapshot() { return dawSnapshot(this.state); }

  selectChannel(channel) {
    this.state.selectedChannel = channel == null ? null : safeChannel(channel);
    this.state.selectedNoteId = null;
    this.render();
  }

  selectNote(id) {
    this.state.selectedNoteId = Number(id);
    const note = selectedDawNote(this.state);
    if (note) this.state.selectedChannel = note.channel;
    this.render();
  }

  _renderSummary(snapshot) {
    if (!this.summaryElement) return;
    const c = snapshot.eventCounts;
    const active = snapshot.channels.length;
    this.summaryElement.textContent = `events ${snapshot.totalEvents} · notes ${c.NOTE || 0} · tracks ${active} · programs ${c.PROGRAM || 0} · ${Math.max(0, snapshot.maxStep / TEMPORAL_RESOLUTION).toFixed(1)} beats`;
  }

  _renderTracks(snapshot) {
    if (!this.tracksElement) return;
    this.tracksElement.replaceChildren();
    const all = document.createElement('button');
    all.type = 'button';
    all.className = `daw-track ${this.state.selectedChannel == null ? 'is-selected' : ''}`;
    all.innerHTML = `<span class="daw-track-dot daw-all"></span><span><strong>All tracks</strong><small>${snapshot.notes.length} retained notes</small></span>`;
    all.addEventListener('click', () => this.selectChannel(null));
    this.tracksElement.appendChild(all);

    const channels = snapshot.channels.length ? snapshot.channels : [{ channel: 0, program: 0, instrument: 'No notes yet', notes: 0, programExplicit: false }];
    for (const channel of channels) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `daw-track daw-channel-${channel.channel} ${this.state.selectedChannel === channel.channel ? 'is-selected' : ''}`;
      button.setAttribute('aria-pressed', String(this.state.selectedChannel === channel.channel));
      const label = channel.instrument === 'No notes yet' ? channel.instrument : trackLabel(channel);
      button.innerHTML = `<span class="daw-track-dot"></span><span><strong>${label}</strong><small>${channel.notes} notes</small></span>`;
      button.addEventListener('click', () => this.selectChannel(channel.channel));
      this.tracksElement.appendChild(button);
    }
  }

  _renderRoll() {
    if (!this.rollElement) return;
    const viewport = computeDawViewport(this.state);
    const notes = visibleNotes(this.state, viewport);
    const pitch = pitchBounds(notes);
    const width = Math.max(560, Math.round(this.rollElement.clientWidth || 840));
    const height = Math.max(300, Math.round(this.rollElement.clientHeight || 360));
    const rulerWidth = 54;
    const top = 24;
    const plotWidth = width - rulerWidth;
    const plotHeight = height - top;
    const pitchCount = pitch.max - pitch.min + 1;
    const rowHeight = plotHeight / pitchCount;
    const xFor = (step) => rulerWidth + (step - viewport.start) / viewport.span * plotWidth;
    const yFor = (value) => top + (pitch.max - value) * rowHeight;

    this.rollElement.setAttribute('viewBox', `0 0 ${width} ${height}`);
    this.rollElement.replaceChildren();

    const background = svgElement('rect', { x: 0, y: 0, width, height, class: 'daw-roll-bg' });
    this.rollElement.appendChild(background);

    for (let p = pitch.min; p <= pitch.max; p += 1) {
      const y = yFor(p) + rowHeight;
      const line = svgElement('line', { x1: rulerWidth, y1: y, x2: width, y2: y, class: p % 12 === 0 ? 'daw-grid-pitch daw-grid-octave' : 'daw-grid-pitch' });
      this.rollElement.appendChild(line);
      if (p % 12 === 0) {
        const text = svgElement('text', { x: 5, y: yFor(p) + rowHeight * 0.72, class: 'daw-pitch-label' });
        text.textContent = midiNoteName(p);
        this.rollElement.appendChild(text);
      }
    }

    const firstBeat = Math.floor(viewport.start / TEMPORAL_RESOLUTION);
    const lastBeat = Math.ceil(viewport.end / TEMPORAL_RESOLUTION);
    for (let beat = firstBeat; beat <= lastBeat; beat += 1) {
      const x = xFor(beat * TEMPORAL_RESOLUTION);
      const line = svgElement('line', { x1: x, y1: top, x2: x, y2: height, class: beat % 4 === 0 ? 'daw-grid-time daw-grid-bar' : 'daw-grid-time' });
      this.rollElement.appendChild(line);
      if (beat % 4 === 0) {
        const text = svgElement('text', { x: x + 4, y: 16, class: 'daw-time-label' });
        text.textContent = `${beat}b`;
        this.rollElement.appendChild(text);
      }
    }

    for (const marker of this.state.markers) {
      if (marker.step < viewport.start || marker.step > viewport.end) continue;
      if (this.state.selectedChannel != null && marker.channel !== this.state.selectedChannel && marker.kind !== 'TEMPO' && marker.kind !== 'TIME_SIGNATURE') continue;
      if (!['PROGRAM', 'BANK'].includes(marker.kind)) continue;
      const x = xFor(marker.step);
      const line = svgElement('line', { x1: x, y1: top, x2: x, y2: height, class: `daw-marker-line daw-channel-${marker.channel}` });
      this.rollElement.appendChild(line);
      const label = svgElement('text', { x: x + 3, y: top + 12, class: 'daw-marker-label' });
      label.textContent = marker.kind === 'PROGRAM' ? `P${marker.value}` : 'BANK';
      this.rollElement.appendChild(label);
    }

    for (const note of notes) {
      const x = xFor(note.step);
      const endX = xFor(note.step + note.duration);
      const y = yFor(note.pitch) + 1;
      const rect = svgElement('rect', {
        x: Math.max(rulerWidth, x),
        y,
        width: Math.max(3, Math.min(width, endX) - Math.max(rulerWidth, x)),
        height: Math.max(3, rowHeight - 2),
        rx: 2,
        class: `daw-note daw-channel-${note.channel} ${this.state.selectedNoteId === note.id ? 'is-selected' : ''}`,
        tabindex: 0,
        role: 'button',
        'data-note-id': note.id,
        'aria-label': `${midiNoteName(note.pitch)}, channel ${note.channel + 1}, ${note.instrument}, velocity ${note.velocity}, duration ${(note.duration / TEMPORAL_RESOLUTION).toFixed(2)} beats`,
      });
      rect.addEventListener('click', () => this.selectNote(note.id));
      rect.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); this.selectNote(note.id); }
      });
      this.rollElement.appendChild(rect);
    }

    const headX = xFor(this.state.maxStep);
    if (headX >= rulerWidth && headX <= width) {
      this.rollElement.appendChild(svgElement('line', { x1: headX, y1: 0, x2: headX, y2: height, class: 'daw-generation-head' }));
      const label = svgElement('text', { x: Math.min(width - 92, headX + 5), y: 16, class: 'daw-head-label' });
      label.textContent = 'generation head';
      this.rollElement.appendChild(label);
    }
  }

  _renderVelocity() {
    if (!this.velocityElement) return;
    const viewport = computeDawViewport(this.state);
    const notes = visibleNotes(this.state, viewport);
    const width = Math.max(560, Math.round(this.velocityElement.clientWidth || 840));
    const height = 92;
    const rulerWidth = 54;
    const plotWidth = width - rulerWidth;
    const xFor = (step) => rulerWidth + (step - viewport.start) / viewport.span * plotWidth;
    this.velocityElement.setAttribute('viewBox', `0 0 ${width} ${height}`);
    this.velocityElement.replaceChildren();
    this.velocityElement.appendChild(svgElement('rect', { x: 0, y: 0, width, height, class: 'daw-velocity-bg' }));
    const label = svgElement('text', { x: 6, y: 16, class: 'daw-velocity-label' });
    label.textContent = 'VEL';
    this.velocityElement.appendChild(label);
    for (const value of [32, 64, 96, 127]) {
      const y = height - (value / 127) * (height - 10);
      this.velocityElement.appendChild(svgElement('line', { x1: rulerWidth, y1: y, x2: width, y2: y, class: 'daw-velocity-grid' }));
    }
    for (const note of notes) {
      const x = xFor(note.step);
      const barHeight = Math.max(2, note.velocity / 127 * (height - 10));
      const rect = svgElement('rect', {
        x: Math.max(rulerWidth, x),
        y: height - barHeight,
        width: 3,
        height: barHeight,
        class: `daw-velocity-bar daw-channel-${note.channel} ${this.state.selectedNoteId === note.id ? 'is-selected' : ''}`,
      });
      rect.addEventListener('click', () => this.selectNote(note.id));
      this.velocityElement.appendChild(rect);
    }
  }

  _renderInspector() {
    if (!this.inspectorElement) return;
    const note = selectedDawNote(this.state);
    if (!note) {
      this.inspectorElement.innerHTML = '<strong>Select a note</strong><span>Click a note block to inspect pitch, velocity, duration and the Program that produced it.</span>';
      return;
    }
    const startBeat = note.step / TEMPORAL_RESOLUTION;
    const durationBeats = note.duration / TEMPORAL_RESOLUTION;
    this.inspectorElement.innerHTML = [
      `<strong>${midiNoteName(note.pitch)} · MIDI ${note.pitch}</strong>`,
      `<span>Ch ${note.channel + 1} · ${note.channel === 9 ? note.instrument : `Program ${note.program} · ${note.instrument}`}</span>`,
      `<span>start ${startBeat.toFixed(2)} beats · duration ${durationBeats.toFixed(2)} beats · velocity ${note.velocity}</span>`,
      `<span>bank ${note.bankMsb}:${note.bankLsb}${note.programExplicit ? '' : ' · default Program 0'}</span>`,
    ].join('');
  }

  _renderScrubber() {
    if (!this.scrubInput) return;
    const min = Math.max(0, this.state.maxStep - DEFAULT_HISTORY_BEATS * TEMPORAL_RESOLUTION);
    const max = Math.max(TEMPORAL_RESOLUTION, this.state.maxStep);
    this.scrubInput.min = String(min);
    this.scrubInput.max = String(max);
    this.scrubInput.step = String(Math.max(1, Math.round(TEMPORAL_RESOLUTION / 4)));
    this.scrubInput.value = String(this.state.follow ? max : clamp(this.state.viewportEndStep, min, max));
    this.scrubInput.disabled = this.state.maxStep <= 0;
  }

  render() {
    const snapshot = this.snapshot();
    if (this.followInput) this.followInput.checked = this.state.follow;
    if (this.zoomSelect) this.zoomSelect.value = String(this.state.windowBeats);
    this._renderSummary(snapshot);
    this._renderTracks(snapshot);
    this._renderRoll();
    this._renderVelocity();
    this._renderInspector();
    this._renderScrubber();
  }

  destroy() { this._resizeObserver?.disconnect(); }
}
