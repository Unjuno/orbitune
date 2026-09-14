import { CompoundEventType } from './compound-runtime.mjs';
import { TEMPORAL_RESOLUTION } from './compound-midi.mjs';

export const GM_PROGRAM_NAMES = Object.freeze([
  'Acoustic Grand Piano','Bright Acoustic Piano','Electric Grand Piano','Honky-tonk Piano','Electric Piano 1','Electric Piano 2','Harpsichord','Clavinet',
  'Celesta','Glockenspiel','Music Box','Vibraphone','Marimba','Xylophone','Tubular Bells','Dulcimer',
  'Drawbar Organ','Percussive Organ','Rock Organ','Church Organ','Reed Organ','Accordion','Harmonica','Tango Accordion',
  'Acoustic Guitar (nylon)','Acoustic Guitar (steel)','Electric Guitar (jazz)','Electric Guitar (clean)','Electric Guitar (muted)','Overdriven Guitar','Distortion Guitar','Guitar Harmonics',
  'Acoustic Bass','Electric Bass (finger)','Electric Bass (pick)','Fretless Bass','Slap Bass 1','Slap Bass 2','Synth Bass 1','Synth Bass 2',
  'Violin','Viola','Cello','Contrabass','Tremolo Strings','Pizzicato Strings','Orchestral Harp','Timpani',
  'String Ensemble 1','String Ensemble 2','Synth Strings 1','Synth Strings 2','Choir Aahs','Voice Oohs','Synth Voice','Orchestra Hit',
  'Trumpet','Trombone','Tuba','Muted Trumpet','French Horn','Brass Section','Synth Brass 1','Synth Brass 2',
  'Soprano Sax','Alto Sax','Tenor Sax','Baritone Sax','Oboe','English Horn','Bassoon','Clarinet',
  'Piccolo','Flute','Recorder','Pan Flute','Blown Bottle','Shakuhachi','Whistle','Ocarina',
  'Lead 1 (square)','Lead 2 (sawtooth)','Lead 3 (calliope)','Lead 4 (chiff)','Lead 5 (charang)','Lead 6 (voice)','Lead 7 (fifths)','Lead 8 (bass + lead)',
  'Pad 1 (new age)','Pad 2 (warm)','Pad 3 (polysynth)','Pad 4 (choir)','Pad 5 (bowed)','Pad 6 (metallic)','Pad 7 (halo)','Pad 8 (sweep)',
  'FX 1 (rain)','FX 2 (soundtrack)','FX 3 (crystal)','FX 4 (atmosphere)','FX 5 (brightness)','FX 6 (goblins)','FX 7 (echoes)','FX 8 (sci-fi)',
  'Sitar','Banjo','Shamisen','Koto','Kalimba','Bag Pipe','Fiddle','Shanai',
  'Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum','Reverse Cymbal',
  'Guitar Fret Noise','Breath Noise','Seashore','Bird Tweet','Telephone Ring','Helicopter','Applause','Gunshot',
]);

export const EVENT_TYPE_NAMES = Object.freeze({
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

const SVG_NS = 'http://www.w3.org/2000/svg';
const DEFAULT_NOTE_LIMIT = 320;
const DEFAULT_MARKER_LIMIT = 160;
const DEFAULT_HISTORY_BEATS = 96;
const DEFAULT_WINDOW_BEATS = 16;

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function clampProgram(value) { return clamp(Math.trunc(Number(value) || 0), 0, 127); }
function safeChannel(value) { return clamp(Math.trunc(Number(value) || 0), 0, 15); }
function instrumentFor(channel, program) { return channel === 9 ? 'Drum kit' : GM_PROGRAM_NAMES[clampProgram(program)]; }
function svgElement(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}
function appendText(parent, tag, text, className = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  parent.appendChild(node);
  return node;
}

export function midiNoteName(pitch) {
  pitch = clamp(Math.trunc(Number(pitch) || 0), 0, 127);
  const names = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
  return `${names[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

export function createMonitorState() {
  return {
    channels: Array.from({ length: 16 }, (_, channel) => ({ channel, program: 0, bankMsb: 0, bankLsb: 0, notes: 0, programExplicit: false })),
    eventCounts: Object.fromEntries(Object.values(EVENT_TYPE_NAMES).map((name) => [name, 0])),
    recent: [],
    notes: [],
    markers: [],
    totalEvents: 0,
    maxStep: 0,
    relativeBaseStep: 0,
    consumeCalls: 0,
    nextNoteId: 1,
    selectedChannel: null,
    selectedNoteId: null,
    follow: true,
    viewportEndStep: 0,
    windowBeats: DEFAULT_WINDOW_BEATS,
  };
}

export function describeCompoundEvent(event, state) {
  const typeName = EVENT_TYPE_NAMES[event.type] || `TYPE_${event.type}`;
  const channel = safeChannel(event.channel);
  const voice = state.channels[channel];
  if (event.type === CompoundEventType.PROGRAM) return `PROGRAM · ch ${channel + 1} · ${clampProgram(event.a1)} ${GM_PROGRAM_NAMES[clampProgram(event.a1)]}`;
  if (event.type === CompoundEventType.BANK) return `BANK · ch ${channel + 1} · MSB ${event.a1} · LSB ${event.a2}`;
  if (event.type === CompoundEventType.NOTE) {
    const instrument = instrumentFor(channel, voice.program);
    return `NOTE · ch ${channel + 1} · ${midiNoteName(event.a1)} · vel ${event.a3} · ${instrument}`;
  }
  if (event.type === CompoundEventType.CC) return `CC · ch ${channel + 1} · #${event.a1}=${event.a2}`;
  if (event.type === CompoundEventType.TEMPO) return `TEMPO · ${event.a1} BPM`;
  if (event.type === CompoundEventType.PEDAL) return `PEDAL · ch ${channel + 1} · ${event.a1 ? 'down' : 'up'}`;
  return `${typeName} · ch ${channel + 1}`;
}

function pruneMonitorState(state, { noteLimit = DEFAULT_NOTE_LIMIT, markerLimit = DEFAULT_MARKER_LIMIT } = {}) {
  if (state.notes.length > noteLimit) state.notes.splice(0, state.notes.length - noteLimit);
  if (state.markers.length > markerLimit) state.markers.splice(0, state.markers.length - markerLimit);
  const keepAfter = Math.max(0, state.maxStep - DEFAULT_HISTORY_BEATS * TEMPORAL_RESOLUTION);
  while (state.notes.length > 64 && state.notes[0].step + state.notes[0].duration < keepAfter) state.notes.shift();
  while (state.markers.length > 32 && state.markers[0].step < keepAfter) state.markers.shift();
  if (state.selectedNoteId != null && !state.notes.some((note) => note.id === state.selectedNoteId)) state.selectedNoteId = null;
}

export function consumeMonitorEvents(state, events, { recentLimit = 36, relative = false, noteLimit = DEFAULT_NOTE_LIMIT, markerLimit = DEFAULT_MARKER_LIMIT } = {}) {
  const baseStep = relative ? state.relativeBaseStep : 0;
  let chunkMax = 0;
  for (const event of events) {
    const localStep = Math.max(0, Math.trunc(Number(event.step) || 0));
    const step = baseStep + localStep;
    chunkMax = Math.max(chunkMax, localStep);
    const channel = safeChannel(event.channel);
    const voice = state.channels[channel];
    const description = describeCompoundEvent(event, state);
    const typeName = EVENT_TYPE_NAMES[event.type] || `TYPE_${event.type}`;
    state.eventCounts[typeName] = (state.eventCounts[typeName] || 0) + 1;
    state.totalEvents += 1;
    state.maxStep = Math.max(state.maxStep, step);

    if (event.type === CompoundEventType.PROGRAM) {
      voice.program = clampProgram(event.a1);
      voice.programExplicit = true;
      state.markers.push({ kind: 'PROGRAM', step, channel, value: voice.program, label: instrumentFor(channel, voice.program) });
    } else if (event.type === CompoundEventType.BANK) {
      voice.bankMsb = clamp(Math.trunc(Number(event.a1) || 0), 0, 127);
      voice.bankLsb = clamp(Math.trunc(Number(event.a2) || 0), 0, 127);
      state.markers.push({ kind: 'BANK', step, channel, value: `${voice.bankMsb}:${voice.bankLsb}`, label: `Bank ${voice.bankMsb}:${voice.bankLsb}` });
    } else if (event.type === CompoundEventType.NOTE) {
      const duration = Math.max(1, Math.trunc(Number(event.a2) || 1));
      const note = {
        id: state.nextNoteId++,
        step,
        duration,
        channel,
        pitch: clamp(Math.trunc(Number(event.a1) || 0), 0, 127),
        velocity: clamp(Math.trunc(Number(event.a3) || 1), 1, 127),
        program: voice.program,
        bankMsb: voice.bankMsb,
        bankLsb: voice.bankLsb,
        programExplicit: voice.programExplicit,
        instrument: instrumentFor(channel, voice.program),
      };
      voice.notes += 1;
      state.notes.push(note);
      state.maxStep = Math.max(state.maxStep, step + duration);
    } else if ([CompoundEventType.CC, CompoundEventType.PEDAL, CompoundEventType.PITCH_BEND, CompoundEventType.TEMPO, CompoundEventType.TIME_SIGNATURE].includes(event.type)) {
      state.markers.push({ kind: typeName, step, channel, value: event.a1, label: typeName });
    }

    state.recent.push({ type: typeName, channel, description });
    if (state.recent.length > recentLimit) state.recent.splice(0, state.recent.length - recentLimit);
  }
  state.relativeBaseStep = baseStep + chunkMax;
  state.consumeCalls += 1;
  if (state.follow) state.viewportEndStep = state.maxStep;
  pruneMonitorState(state, { noteLimit, markerLimit });
  return state;
}

export function monitorSnapshot(state) {
  const activeChannels = state.channels.filter((channel) => channel.notes > 0 || channel.programExplicit);
  return {
    totalEvents: state.totalEvents,
    eventCounts: { ...state.eventCounts },
    channels: activeChannels.map((channel) => ({ ...channel, instrument: instrumentFor(channel.channel, channel.program) })),
    recent: state.recent.slice(),
    notes: state.notes.slice(),
    markers: state.markers.slice(),
    maxStep: state.maxStep,
    selectedChannel: state.selectedChannel,
    selectedNote: state.notes.find((note) => note.id === state.selectedNoteId) || null,
    follow: state.follow,
    windowBeats: state.windowBeats,
  };
}

export function computeMonitorViewport(state) {
  const span = Math.max(TEMPORAL_RESOLUTION * 4, Math.trunc(state.windowBeats * TEMPORAL_RESOLUTION));
  const end = state.follow
    ? Math.max(span, state.maxStep + Math.round(TEMPORAL_RESOLUTION * 0.5))
    : clamp(Math.max(span, state.viewportEndStep), span, Math.max(span, state.maxStep + TEMPORAL_RESOLUTION));
  return { start: Math.max(0, end - span), end, span };
}

function visibleNotes(state, viewport) {
  return state.notes.filter((note) => {
    if (state.selectedChannel != null && note.channel !== state.selectedChannel) return false;
    return note.step + note.duration >= viewport.start && note.step <= viewport.end;
  });
}

function pitchBounds(notes) {
  if (!notes.length) return { min: 48, max: 84 };
  let min = clamp(Math.min(...notes.map((note) => note.pitch)) - 2, 0, 127);
  let max = clamp(Math.max(...notes.map((note) => note.pitch)) + 2, 0, 127);
  if (max - min < 23) {
    const center = Math.round((min + max) / 2);
    min = clamp(center - 12, 0, 104);
    max = min + 23;
  }
  return { min, max };
}

export class CompoundEventMonitor {
  constructor({ channelsElement = null, recentElement = null, summaryElement = null } = {}) {
    this.channelsElement = channelsElement;
    this.recentElement = recentElement;
    this.summaryElement = summaryElement;
    this.state = createMonitorState();
    this.rollElement = null;
    this.velocityElement = null;
    this.inspectorElement = null;
    this.followInput = null;
    this.zoomSelect = null;
    this.scrubInput = null;
    this.resizeObserver = null;
    this._upgradeSurface();
  }

  _upgradeSurface() {
    if (typeof document === 'undefined' || !this.summaryElement || !this.channelsElement || !this.recentElement) return;
    const body = this.summaryElement.parentElement;
    if (!body) return;
    const title = document.getElementById('event-monitor-title');
    if (title) title.textContent = 'Generated MIDI · piano roll';
    const subtitle = title?.parentElement?.querySelector('p');
    if (subtitle) subtitle.textContent = 'DAW-style view of channel, Program, pitch, duration and velocity. Click a track or note to inspect it.';

    if (!document.querySelector('link[data-orbitune-daw]')) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = './daw-monitor.css';
      link.dataset.orbituneDaw = 'true';
      document.head.appendChild(link);
    }

    const workspace = document.createElement('div');
    workspace.className = 'daw-workspace';

    const toolbar = document.createElement('div');
    toolbar.className = 'daw-toolbar';
    const zoomField = document.createElement('div');
    zoomField.className = 'field';
    const zoomLabel = document.createElement('label');
    zoomLabel.htmlFor = 'compound-daw-zoom';
    zoomLabel.textContent = 'Visible timeline';
    this.zoomSelect = document.createElement('select');
    this.zoomSelect.id = 'compound-daw-zoom';
    for (const beats of [8, 16, 32, 64]) this.zoomSelect.add(new Option(`${beats} beats`, String(beats), false, beats === DEFAULT_WINDOW_BEATS));
    zoomField.append(zoomLabel, this.zoomSelect);
    const followLabel = document.createElement('label');
    followLabel.className = 'daw-follow';
    this.followInput = document.createElement('input');
    this.followInput.type = 'checkbox';
    this.followInput.checked = true;
    followLabel.append(this.followInput, document.createTextNode(' Follow generation'));
    toolbar.append(zoomField, followLabel);

    this.summaryElement.className = 'daw-summary';
    this.channelsElement.className = 'daw-tracks';
    this.channelsElement.setAttribute('aria-label', 'Generated MIDI tracks');
    this.recentElement.className = 'daw-roll-stack';
    this.recentElement.replaceChildren();

    const editor = document.createElement('div');
    editor.className = 'daw-editor';
    const rollScroller = document.createElement('div');
    rollScroller.className = 'daw-roll-scroller';
    this.rollElement = svgElement('svg', { id: 'compound-daw-roll', class: 'daw-roll', role: 'img', 'aria-label': 'Generated MIDI piano roll' });
    this.velocityElement = svgElement('svg', { id: 'compound-daw-velocity', class: 'daw-velocity', role: 'img', 'aria-label': 'Generated MIDI velocity lane' });
    rollScroller.append(this.rollElement, this.velocityElement);
    const scrubRow = document.createElement('div');
    scrubRow.className = 'daw-scrub-row';
    const scrubLabel = document.createElement('label');
    scrubLabel.htmlFor = 'compound-daw-scrub';
    scrubLabel.textContent = 'Timeline';
    this.scrubInput = document.createElement('input');
    this.scrubInput.id = 'compound-daw-scrub';
    this.scrubInput.type = 'range';
    this.scrubInput.min = '0'; this.scrubInput.max = '96'; this.scrubInput.value = '0'; this.scrubInput.disabled = true;
    scrubRow.append(scrubLabel, this.scrubInput);
    this.recentElement.append(rollScroller, scrubRow);
    editor.append(this.channelsElement, this.recentElement);

    this.inspectorElement = document.createElement('div');
    this.inspectorElement.id = 'compound-daw-inspector';
    this.inspectorElement.className = 'daw-inspector';
    this.inspectorElement.setAttribute('aria-live', 'polite');

    workspace.append(toolbar, this.summaryElement, editor, this.inspectorElement);
    body.replaceChildren(workspace);

    this.followInput.addEventListener('change', () => {
      this.state.follow = this.followInput.checked;
      if (this.state.follow) this.state.viewportEndStep = this.state.maxStep;
      this.render();
    });
    this.zoomSelect.addEventListener('change', () => {
      const beats = Number(this.zoomSelect.value);
      if (Number.isFinite(beats) && beats >= 4 && beats <= 128) this.state.windowBeats = beats;
      this.render();
    });
    this.scrubInput.addEventListener('input', () => {
      this.state.follow = false;
      this.followInput.checked = false;
      this.state.viewportEndStep = Number(this.scrubInput.value) || this.state.maxStep;
      this.render();
    });
    if (typeof ResizeObserver === 'function') {
      this.resizeObserver = new ResizeObserver(() => this.render());
      this.resizeObserver.observe(rollScroller);
    }
  }

  reset() {
    const windowBeats = this.state.windowBeats;
    this.state = createMonitorState();
    this.state.windowBeats = windowBeats;
    if (this.followInput) this.followInput.checked = true;
    this.render();
  }

  consume(events) {
    const relative = this.state.consumeCalls > 0;
    consumeMonitorEvents(this.state, events, { relative });
    this.render();
    return monitorSnapshot(this.state);
  }

  snapshot() { return monitorSnapshot(this.state); }

  selectChannel(channel) {
    this.state.selectedChannel = channel == null ? null : safeChannel(channel);
    this.state.selectedNoteId = null;
    this.render();
  }

  selectNote(id) {
    this.state.selectedNoteId = Number(id);
    const note = this.state.notes.find((item) => item.id === this.state.selectedNoteId);
    if (note) this.state.selectedChannel = note.channel;
    this.render();
  }

  _renderSummary(snapshot) {
    if (!this.summaryElement) return;
    const c = snapshot.eventCounts;
    this.summaryElement.textContent = `events ${snapshot.totalEvents} · notes ${c.NOTE || 0} · tracks ${snapshot.channels.length} · programs ${c.PROGRAM || 0} · ${(snapshot.maxStep / TEMPORAL_RESOLUTION).toFixed(1)} beats`;
  }

  _renderTracks(snapshot) {
    if (!this.channelsElement) return;
    this.channelsElement.replaceChildren();
    const all = document.createElement('button');
    all.type = 'button';
    all.className = `daw-track ${this.state.selectedChannel == null ? 'is-selected' : ''}`;
    const allDot = document.createElement('span'); allDot.className = 'daw-track-dot daw-all';
    const allText = document.createElement('span');
    appendText(allText, 'strong', 'All tracks'); appendText(allText, 'small', `${snapshot.notes.length} retained notes`);
    all.append(allDot, allText);
    all.addEventListener('click', () => this.selectChannel(null));
    this.channelsElement.appendChild(all);

    const channels = snapshot.channels.length ? snapshot.channels : [{ channel: 0, program: 0, instrument: 'No notes yet', notes: 0, programExplicit: false }];
    for (const channel of channels) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `daw-track daw-channel-${channel.channel} ${this.state.selectedChannel === channel.channel ? 'is-selected' : ''}`;
      button.setAttribute('aria-pressed', String(this.state.selectedChannel === channel.channel));
      const dot = document.createElement('span'); dot.className = 'daw-track-dot';
      const text = document.createElement('span');
      const label = channel.instrument === 'No notes yet'
        ? channel.instrument
        : channel.channel === 9
          ? `Ch 10 · Drums`
          : `Ch ${channel.channel + 1} · ${channel.program} ${channel.instrument}${channel.programExplicit ? '' : ' (default)'}`;
      appendText(text, 'strong', label); appendText(text, 'small', `${channel.notes} notes`);
      button.append(dot, text);
      button.addEventListener('click', () => this.selectChannel(channel.channel));
      this.channelsElement.appendChild(button);
    }
  }

  _renderRoll() {
    if (!this.rollElement) return;
    const viewport = computeMonitorViewport(this.state);
    const notes = visibleNotes(this.state, viewport);
    const pitch = pitchBounds(notes);
    const width = Math.max(560, Math.round(this.rollElement.clientWidth || 840));
    const height = Math.max(300, Math.round(this.rollElement.clientHeight || 360));
    const rulerWidth = 54, top = 24, plotWidth = width - rulerWidth, plotHeight = height - top;
    const pitchCount = pitch.max - pitch.min + 1, rowHeight = plotHeight / pitchCount;
    const xFor = (step) => rulerWidth + (step - viewport.start) / viewport.span * plotWidth;
    const yFor = (value) => top + (pitch.max - value) * rowHeight;
    this.rollElement.setAttribute('viewBox', `0 0 ${width} ${height}`);
    this.rollElement.replaceChildren(svgElement('rect', { x: 0, y: 0, width, height, class: 'daw-roll-bg' }));

    for (let p = pitch.min; p <= pitch.max; p += 1) {
      const y = yFor(p) + rowHeight;
      this.rollElement.appendChild(svgElement('line', { x1: rulerWidth, y1: y, x2: width, y2: y, class: p % 12 === 0 ? 'daw-grid-pitch daw-grid-octave' : 'daw-grid-pitch' }));
      if (p % 12 === 0) {
        const text = svgElement('text', { x: 5, y: yFor(p) + rowHeight * 0.72, class: 'daw-pitch-label' });
        text.textContent = midiNoteName(p); this.rollElement.appendChild(text);
      }
    }
    const firstBeat = Math.floor(viewport.start / TEMPORAL_RESOLUTION), lastBeat = Math.ceil(viewport.end / TEMPORAL_RESOLUTION);
    for (let beat = firstBeat; beat <= lastBeat; beat += 1) {
      const x = xFor(beat * TEMPORAL_RESOLUTION);
      this.rollElement.appendChild(svgElement('line', { x1: x, y1: top, x2: x, y2: height, class: beat % 4 === 0 ? 'daw-grid-time daw-grid-bar' : 'daw-grid-time' }));
      if (beat % 4 === 0) {
        const text = svgElement('text', { x: x + 4, y: 16, class: 'daw-time-label' }); text.textContent = `${beat}b`; this.rollElement.appendChild(text);
      }
    }
    for (const marker of this.state.markers) {
      if (marker.step < viewport.start || marker.step > viewport.end || !['PROGRAM', 'BANK'].includes(marker.kind)) continue;
      if (this.state.selectedChannel != null && marker.channel !== this.state.selectedChannel) continue;
      const x = xFor(marker.step);
      this.rollElement.appendChild(svgElement('line', { x1: x, y1: top, x2: x, y2: height, class: `daw-marker-line daw-channel-${marker.channel}` }));
      const label = svgElement('text', { x: x + 3, y: top + 12, class: 'daw-marker-label' }); label.textContent = marker.kind === 'PROGRAM' ? `P${marker.value}` : 'BANK'; this.rollElement.appendChild(label);
    }
    for (const note of notes) {
      const x = Math.max(rulerWidth, xFor(note.step));
      const endX = Math.min(width, xFor(note.step + note.duration));
      const rect = svgElement('rect', {
        x, y: yFor(note.pitch) + 1, width: Math.max(3, endX - x), height: Math.max(3, rowHeight - 2), rx: 2,
        class: `daw-note daw-channel-${note.channel} ${this.state.selectedNoteId === note.id ? 'is-selected' : ''}`,
        tabindex: 0, role: 'button', 'data-note-id': note.id,
        'aria-label': `${midiNoteName(note.pitch)}, channel ${note.channel + 1}, ${note.instrument}, velocity ${note.velocity}, duration ${(note.duration / TEMPORAL_RESOLUTION).toFixed(2)} beats`,
      });
      rect.addEventListener('click', () => this.selectNote(note.id));
      rect.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); this.selectNote(note.id); } });
      this.rollElement.appendChild(rect);
    }
    const headX = xFor(this.state.maxStep);
    if (headX >= rulerWidth && headX <= width) {
      this.rollElement.appendChild(svgElement('line', { x1: headX, y1: 0, x2: headX, y2: height, class: 'daw-generation-head' }));
      const label = svgElement('text', { x: Math.min(width - 92, headX + 5), y: 16, class: 'daw-head-label' }); label.textContent = 'generation head'; this.rollElement.appendChild(label);
    }
  }

  _renderVelocity() {
    if (!this.velocityElement) return;
    const viewport = computeMonitorViewport(this.state), notes = visibleNotes(this.state, viewport);
    const width = Math.max(560, Math.round(this.velocityElement.clientWidth || 840)), height = 92, rulerWidth = 54, plotWidth = width - rulerWidth;
    const xFor = (step) => rulerWidth + (step - viewport.start) / viewport.span * plotWidth;
    this.velocityElement.setAttribute('viewBox', `0 0 ${width} ${height}`);
    this.velocityElement.replaceChildren(svgElement('rect', { x: 0, y: 0, width, height, class: 'daw-velocity-bg' }));
    const label = svgElement('text', { x: 6, y: 16, class: 'daw-velocity-label' }); label.textContent = 'VEL'; this.velocityElement.appendChild(label);
    for (const value of [32, 64, 96, 127]) {
      const y = height - (value / 127) * (height - 10);
      this.velocityElement.appendChild(svgElement('line', { x1: rulerWidth, y1: y, x2: width, y2: y, class: 'daw-velocity-grid' }));
    }
    for (const note of notes) {
      const barHeight = Math.max(2, note.velocity / 127 * (height - 10));
      const rect = svgElement('rect', { x: Math.max(rulerWidth, xFor(note.step)), y: height - barHeight, width: 3, height: barHeight, class: `daw-velocity-bar daw-channel-${note.channel} ${this.state.selectedNoteId === note.id ? 'is-selected' : ''}` });
      rect.addEventListener('click', () => this.selectNote(note.id)); this.velocityElement.appendChild(rect);
    }
  }

  _renderInspector(snapshot) {
    if (!this.inspectorElement) return;
    this.inspectorElement.replaceChildren();
    const note = snapshot.selectedNote;
    if (!note) {
      appendText(this.inspectorElement, 'strong', 'Select a note');
      appendText(this.inspectorElement, 'span', 'Click a note block to inspect pitch, velocity, duration and the Program that produced it.');
      return;
    }
    appendText(this.inspectorElement, 'strong', `${midiNoteName(note.pitch)} · MIDI ${note.pitch}`);
    appendText(this.inspectorElement, 'span', `Ch ${note.channel + 1} · ${note.channel === 9 ? note.instrument : `Program ${note.program} · ${note.instrument}`}`);
    appendText(this.inspectorElement, 'span', `start ${(note.step / TEMPORAL_RESOLUTION).toFixed(2)} beats · duration ${(note.duration / TEMPORAL_RESOLUTION).toFixed(2)} beats · velocity ${note.velocity}`);
    appendText(this.inspectorElement, 'span', `bank ${note.bankMsb}:${note.bankLsb}${note.programExplicit ? '' : ' · default Program 0'}`);
  }

  _renderScrubber() {
    if (!this.scrubInput) return;
    const min = Math.max(0, this.state.maxStep - DEFAULT_HISTORY_BEATS * TEMPORAL_RESOLUTION), max = Math.max(TEMPORAL_RESOLUTION, this.state.maxStep);
    this.scrubInput.min = String(min); this.scrubInput.max = String(max); this.scrubInput.step = String(Math.max(1, Math.round(TEMPORAL_RESOLUTION / 4)));
    this.scrubInput.value = String(this.state.follow ? max : clamp(this.state.viewportEndStep, min, max)); this.scrubInput.disabled = this.state.maxStep <= 0;
  }

  render() {
    const snapshot = this.snapshot();
    if (this.followInput) this.followInput.checked = this.state.follow;
    if (this.zoomSelect) this.zoomSelect.value = String(this.state.windowBeats);
    this._renderSummary(snapshot);
    if (!this.rollElement) {
      if (this.channelsElement) {
        this.channelsElement.replaceChildren();
        for (const channel of snapshot.channels) appendText(this.channelsElement, 'div', `Ch ${channel.channel + 1} · ${channel.instrument} · notes ${channel.notes}`, 'event-monitor-channel');
      }
      if (this.recentElement) {
        this.recentElement.replaceChildren();
        for (const item of snapshot.recent.slice().reverse()) appendText(this.recentElement, 'div', item.description, `event-monitor-event event-${item.type.toLowerCase().replaceAll('_', '-')}`);
      }
      return;
    }
    this._renderTracks(snapshot); this._renderRoll(); this._renderVelocity(); this._renderInspector(snapshot); this._renderScrubber();
  }

  destroy() { this.resizeObserver?.disconnect(); }
}
