import { CompoundEventMonitor } from './compound-event-monitor.mjs';
import { TEMPORAL_RESOLUTION } from './compound-midi.mjs';

const ORIGINAL_SELECT_NOTE = CompoundEventMonitor.prototype.selectNote;
const ORIGINAL_RENDER = CompoundEventMonitor.prototype.render;
const ENHANCED = Symbol('orbitune-daw-ux');

export function selectNotePreservingTrack(state, id) {
  state.selectedNoteId = Number(id);
  if (!state.notes.some((note) => note.id === state.selectedNoteId)) state.selectedNoteId = null;
  return state.selectedNoteId;
}

export function scrubBeat(value) {
  const step = Math.max(0, Number(value) || 0);
  return step / TEMPORAL_RESOLUTION;
}

function ensureUxStyles() {
  if (typeof document === 'undefined' || document.querySelector('link[data-orbitune-daw-ux]')) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = './daw-ux.css';
  link.dataset.orbituneDawUx = 'true';
  document.head.appendChild(link);
}

function relocatePanel() {
  if (typeof document === 'undefined') return null;
  const title = document.getElementById('event-monitor-title');
  const panel = title?.closest('.panel');
  const layout = panel?.closest('.app-layout');
  if (!panel || !layout) return panel || null;
  panel.classList.add('daw-panel-full');
  if (layout.nextElementSibling !== panel) layout.insertAdjacentElement('afterend', panel);
  const kicker = panel.querySelector('.status-kicker');
  if (kicker) kicker.textContent = 'live MIDI editor';
  if (title) title.textContent = 'Generated MIDI editor';
  const subtitle = title?.parentElement?.querySelector('p');
  if (subtitle) subtitle.textContent = 'Track view, piano roll, velocity and note details. Track selection filters; note selection only inspects.';
  return panel;
}

function addToolbarActions(monitor) {
  const toolbar = monitor.summaryElement?.parentElement?.querySelector('.daw-toolbar');
  if (!toolbar || toolbar.querySelector('[data-daw-ux-actions]')) return;
  const actions = document.createElement('div');
  actions.className = 'daw-ux-actions';
  actions.dataset.dawUxActions = 'true';

  const latest = document.createElement('button');
  latest.type = 'button';
  latest.className = 'secondary daw-ux-button';
  latest.textContent = 'Latest';
  latest.addEventListener('click', () => {
    monitor.state.follow = true;
    monitor.state.viewportEndStep = monitor.state.maxStep;
    if (monitor.followInput) monitor.followInput.checked = true;
    monitor.render();
  });

  const clearFilter = document.createElement('button');
  clearFilter.type = 'button';
  clearFilter.className = 'secondary daw-ux-button';
  clearFilter.textContent = 'All tracks';
  clearFilter.addEventListener('click', () => {
    monitor.state.selectedChannel = null;
    monitor.state.selectedNoteId = null;
    monitor.render();
  });

  actions.append(latest, clearFilter);
  toolbar.appendChild(actions);
}

function addRollInteractions(monitor) {
  const roll = monitor.rollElement;
  if (!roll || roll.dataset.dawUxBound === 'true') return;
  roll.dataset.dawUxBound = 'true';
  roll.addEventListener('click', (event) => {
    if (event.target?.classList?.contains('daw-roll-bg')) {
      monitor.state.selectedNoteId = null;
      monitor.render();
    }
  });
}

function tuneRenderedSurface(monitor) {
  const panel = relocatePanel();
  if (!panel) return;
  addToolbarActions(monitor);
  addRollInteractions(monitor);

  const scrub = monitor.scrubInput;
  const scrubLabel = scrub?.parentElement?.querySelector('label');
  if (scrub && scrubLabel) scrubLabel.textContent = `View end · ${scrubBeat(scrub.value).toFixed(1)} beats`;

  const selectedTrack = panel.querySelector('.daw-track.is-selected strong');
  const summary = panel.querySelector('.daw-summary');
  if (summary) {
    const track = monitor.state.selectedChannel == null ? 'all tracks' : `channel ${monitor.state.selectedChannel + 1}`;
    summary.dataset.view = track;
    summary.setAttribute('aria-label', `${summary.textContent}. Viewing ${track}.`);
  }
  if (selectedTrack) selectedTrack.closest('.daw-track')?.setAttribute('title', 'Click a track to filter the piano roll');

  const labels = [...panel.querySelectorAll('.daw-marker-label')];
  if (labels.length > 10) {
    const stride = Math.max(2, Math.ceil(labels.length / 8));
    labels.forEach((label, index) => { label.style.display = index % stride === 0 || index === labels.length - 1 ? '' : 'none'; });
  }

  for (const note of panel.querySelectorAll('.daw-note')) {
    const aria = note.getAttribute('aria-label');
    if (aria) note.setAttribute('title', aria);
  }
}

export function installDawUx() {
  if (CompoundEventMonitor.prototype[ENHANCED]) return;
  Object.defineProperty(CompoundEventMonitor.prototype, ENHANCED, { value: true });
  ensureUxStyles();

  CompoundEventMonitor.prototype.selectNote = function selectNote(id) {
    selectNotePreservingTrack(this.state, id);
    this.render();
  };

  CompoundEventMonitor.prototype.render = function render(...args) {
    const result = ORIGINAL_RENDER.apply(this, args);
    if (typeof document !== 'undefined') tuneRenderedSurface(this);
    return result;
  };

  // compound-app creates the monitor before this enhancement module executes.
  // Relocate its already-rendered panel immediately; future renders are handled
  // by the prototype hook above.
  if (typeof document !== 'undefined') {
    ensureUxStyles();
    relocatePanel();
  }
}

if (typeof document !== 'undefined') installDawUx();

// Keep the original reference alive for debugging without invoking it. This
// also makes the intentional behavior change explicit: note inspection no
// longer calls the old method, because that method implicitly soloed a track.
void ORIGINAL_SELECT_NOTE;
