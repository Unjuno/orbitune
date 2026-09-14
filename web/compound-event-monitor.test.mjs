import assert from 'node:assert/strict';
import test from 'node:test';
import { consumeMonitorEvents, createMonitorState, monitorSnapshot } from './compound-event-monitor.mjs';
import { CompoundEventType } from './compound-runtime.mjs';

test('event monitor resolves program state before later notes', () => {
  const state = createMonitorState();
  consumeMonitorEvents(state, [
    { type: CompoundEventType.PROGRAM, channel: 2, a1: 40, a2: 0, a3: 0, a4: 0 },
    { type: CompoundEventType.NOTE, channel: 2, a1: 64, a2: 96, a3: 100, a4: 0 },
  ]);
  const snapshot = monitorSnapshot(state);
  assert.equal(snapshot.channels[0].program, 40);
  assert.equal(snapshot.channels[0].instrument, 'Violin');
  assert.match(snapshot.recent.at(-1).description, /Violin/);
});

test('event monitor keeps only bounded recent events', () => {
  const state = createMonitorState();
  const events = Array.from({ length: 50 }, (_, i) => ({ type: CompoundEventType.NOTE, channel: 0, a1: 60 + (i % 12), a2: 24, a3: 90, a4: 0 }));
  consumeMonitorEvents(state, events, { recentLimit: 12 });
  assert.equal(state.recent.length, 12);
  assert.equal(state.totalEvents, 50);
});
