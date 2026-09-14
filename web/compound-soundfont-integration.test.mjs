import assert from 'node:assert/strict';
import test from 'node:test';

import { CompoundPreviewPlayer } from './compound-player.mjs';
import { CompoundStreamingPreviewPlayer } from './compound-live-player.mjs';
import { CompoundEventType } from './compound-runtime.mjs';

function note(step = 0) {
  return { type: CompoundEventType.NOTE, step, channel: 0, a1: 60, a2: 48, a3: 100, a4: 0 };
}

class FakeSampledSynth {
  constructor() {
    this.context = { currentTime: 1 };
    this.release = { id: 'fake-sf2', display_name: 'Fake Sampled SF2' };
    this.started = 0;
    this.calls = [];
    this.stops = [];
  }
  async ensureStarted() { this.started += 1; return this; }
  async schedule(events, options = {}) {
    this.calls.push({ events, options });
    return {
      noteCount: events.filter((event) => event.type === CompoundEventType.NOTE).length,
      release: this.release,
    };
  }
  async pause() { this.paused = true; }
  async resume() { this.paused = false; }
  stop(options = {}) { this.stops.push(options); }
  destroy() { this.destroyed = true; }
}

test('finite preview delegates audible playback to sampled SoundFont synth', async () => {
  const synth = new FakeSampledSynth();
  const player = new CompoundPreviewPlayer({ soundFontSynth: synth });
  const count = await player.play([note()]);
  assert.equal(count, 1);
  assert.equal(synth.calls.length, 1);
  assert.deepEqual(synth.calls[0].events, [note()]);
  assert.deepEqual(synth.stops[0], { hard: true });
});

test('infinite preview schedules control-only chunks through the persistent sampled synth', async () => {
  const synth = new FakeSampledSynth();
  const player = new CompoundStreamingPreviewPlayer({ soundFontSynth: synth, leadSeconds: 0.1 });
  await player.ensureStarted();
  const controlOnly = [
    { type: CompoundEventType.BANK, step: 0, channel: 1, a1: 0, a2: 0, a3: 0, a4: 0 },
    { type: CompoundEventType.PROGRAM, step: 0, channel: 1, a1: 33, a2: 0, a3: 0, a4: 0 },
  ];
  const result = await player.append(controlOnly, { defaultBpm: 120 });
  assert.equal(result.scheduledNotes, 0);
  assert.equal(result.soundFont, 'Fake Sampled SF2');
  assert.equal(synth.calls.length, 1);
  assert.deepEqual(synth.calls[0].events, controlOnly);
  player.stop();
  assert.deepEqual(synth.stops.at(-1), { hard: true });
});
