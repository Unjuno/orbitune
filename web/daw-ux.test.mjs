import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { scrubBeat, selectNotePreservingTrack } from './daw-ux.mjs';

test('note inspection preserves the current track filter', () => {
  const state = {
    selectedChannel: null,
    selectedNoteId: null,
    notes: [{ id: 7, channel: 3 }],
  };
  assert.equal(selectNotePreservingTrack(state, 7), 7);
  assert.equal(state.selectedChannel, null);
  state.selectedChannel = 3;
  assert.equal(selectNotePreservingTrack(state, 7), 7);
  assert.equal(state.selectedChannel, 3);
});

test('invalid note selection clears only note inspection', () => {
  const state = { selectedChannel: 5, selectedNoteId: 2, notes: [] };
  assert.equal(selectNotePreservingTrack(state, 999), null);
  assert.equal(state.selectedChannel, 5);
});

test('scrub position is reported in beats', () => {
  assert.equal(scrubBeat(0), 0);
  assert.equal(scrubBeat(96), 1);
  assert.equal(scrubBeat(384), 4);
});

test('both public entry points load the DAW UX enhancement after the Compound app', async () => {
  for (const filename of ['index.html', 'compound.html']) {
    const html = await readFile(new URL(`./${filename}`, import.meta.url), 'utf8');
    const app = html.indexOf('src="./compound-app.mjs"');
    const ux = html.indexOf('src="./daw-ux.mjs"');
    assert.ok(app >= 0, `${filename} missing compound app module`);
    assert.ok(ux > app, `${filename} must load DAW UX after compound app`);
  }
});

test('PWA shell caches DAW UX module and stylesheet', async () => {
  const source = await readFile(new URL('./sw.js', import.meta.url), 'utf8');
  assert.ok(source.includes("'./daw-ux.mjs'"));
  assert.ok(source.includes("'./daw-ux.css'"));
});
