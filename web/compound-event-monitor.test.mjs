import assert from 'node:assert/strict';
import test from 'node:test';
import {
  computeMonitorViewport,
  consumeMonitorEvents,
  createMonitorState,
  midiNoteName,
  monitorSnapshot,
} from './compound-event-monitor.mjs';
import { CompoundEventType } from './compound-runtime.mjs';
import { TEMPORAL_RESOLUTION } from './compound-midi.mjs';

test('event monitor resolves program state before later notes', () => {
  const state = createMonitorState();
  consumeMonitorEvents(state, [
    { type: CompoundEventType.PROGRAM, step: 0, channel: 2, a1: 40, a2: 0, a3: 0, a4: 0 },
    { type: CompoundEventType.NOTE, step: 24, channel: 2, a1: 64, a2: 96, a3: 100, a4: 0 },
  ]);
  const snapshot = monitorSnapshot(state);
  assert.equal(snapshot.channels[0].program, 40);
  assert.equal(snapshot.channels[0].instrument, 'Violin');
  assert.match(snapshot.recent.at(-1).description, /Violin/);
  assert.equal(snapshot.notes[0].instrument, 'Violin');
  assert.equal(snapshot.notes[0].duration, 96);
  assert.equal(snapshot.notes[0].velocity, 100);
});

test('event monitor keeps only bounded recent events and note history', () => {
  const state = createMonitorState();
  const events = Array.from({ length: 500 }, (_, i) => ({ type: CompoundEventType.NOTE, step: i * 24, channel: 0, a1: 60 + (i % 12), a2: 24, a3: 90, a4: 0 }));
  consumeMonitorEvents(state, events, { recentLimit: 12, noteLimit: 80 });
  assert.equal(state.recent.length, 12);
  assert.ok(state.notes.length <= 80);
  assert.equal(state.totalEvents, 500);
});

test('relative live chunks continue on one absolute DAW timeline', () => {
  const state = createMonitorState();
  consumeMonitorEvents(state, [
    { type: CompoundEventType.NOTE, step: 96, channel: 0, a1: 60, a2: 48, a3: 80, a4: 0 },
  ]);
  consumeMonitorEvents(state, [
    { type: CompoundEventType.NOTE, step: 24, channel: 0, a1: 64, a2: 48, a3: 90, a4: 0 },
  ], { relative: true });
  const snapshot = monitorSnapshot(state);
  assert.deepEqual(snapshot.notes.map((note) => note.step), [96, 120]);
  assert.equal(state.relativeBaseStep, 120);
});

test('DAW viewport follows generation and respects the selected beat span', () => {
  const state = createMonitorState();
  state.windowBeats = 8;
  state.maxStep = 20 * TEMPORAL_RESOLUTION;
  const viewport = computeMonitorViewport(state);
  assert.equal(viewport.span, 8 * TEMPORAL_RESOLUTION);
  assert.ok(viewport.end >= state.maxStep);
  assert.equal(viewport.end - viewport.start, viewport.span);
});

test('MIDI note labels use standard octave naming', () => {
  assert.equal(midiNoteName(60), 'C4');
  assert.equal(midiNoteName(69), 'A4');
});
