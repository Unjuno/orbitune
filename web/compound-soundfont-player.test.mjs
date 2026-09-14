import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  buildSampledPlaybackSchedule,
  validateSoundFontRelease,
} from './compound-soundfont-player.mjs';
import { CompoundEventType } from './compound-runtime.mjs';

const event = (type, step, channel, a1, a2 = 0, a3 = 0, a4 = 0) => ({ type, step, channel, a1, a2, a3, a4 });

test('checked-in sampled SoundFont release is fully pinned', () => {
  const release = validateSoundFontRelease(JSON.parse(fs.readFileSync(new URL('./soundfont-release.json', import.meta.url), 'utf8')));
  assert.equal(release.id, 'generaluser-gs-2.0.3');
  assert.equal(release.filename, 'GeneralUser-GS.sf2');
  assert.equal(release.bytes, 32319396);
  assert.equal(release.sha256, 'c278464b823daf9c52106c0957f752817da0e52964817ff682fe3a8d2f8446ce');
  assert.equal(release.source_commit, '684543d5e5efaef08d02be50dcda8d552478fa60');
  assert.equal(release.synth_engine.package, 'spessasynth_lib');
  assert.equal(release.synth_engine.version, '4.3.14');
});

test('sampled playback sends bank and program before a same-step note', () => {
  const schedule = buildSampledPlaybackSchedule([
    event(CompoundEventType.NOTE, 0, 2, 60, 96, 100),
    event(CompoundEventType.PROGRAM, 0, 2, 48),
    event(CompoundEventType.BANK, 0, 2, 1, 5),
  ]);
  assert.deepEqual(schedule.timeline.slice(0, 4).map((item) => item.kind), ['bank-msb', 'bank-lsb', 'program', 'note-on']);
  assert.deepEqual(schedule.timeline[0].message, [0xB2, 0, 1]);
  assert.deepEqual(schedule.timeline[1].message, [0xB2, 32, 5]);
  assert.deepEqual(schedule.timeline[2].message, [0xC2, 48]);
  assert.deepEqual(schedule.timeline[3].message, [0x92, 60, 100]);
  assert.deepEqual(schedule.timeline.at(-1).message, [0x82, 60, 0]);
  assert.equal(schedule.noteCount, 1);
});

test('sampled playback preserves MIDI channel 10 drum messages', () => {
  const schedule = buildSampledPlaybackSchedule([
    event(CompoundEventType.NOTE, 0, 9, 36, 24, 110),
    event(CompoundEventType.NOTE, 24, 9, 38, 24, 96),
  ]);
  const ons = schedule.timeline.filter((item) => item.kind === 'note-on');
  assert.equal(ons.length, 2);
  assert.equal(ons[0].message[0], 0x99);
  assert.equal(ons[1].message[0], 0x99);
});

test('control-only streaming chunks remain schedulable', () => {
  const schedule = buildSampledPlaybackSchedule([
    event(CompoundEventType.BANK, 0, 1, 0, 0),
    event(CompoundEventType.PROGRAM, 0, 1, 33),
    event(CompoundEventType.CC, 12, 1, 7, 100),
  ]);
  assert.equal(schedule.noteCount, 0);
  assert.deepEqual(schedule.timeline.map((item) => item.kind), ['bank-msb', 'bank-lsb', 'program', 'cc']);
});

test('tempo mapping is retained for sampled playback scheduling', () => {
  const schedule = buildSampledPlaybackSchedule([
    event(CompoundEventType.TEMPO, 0, 0, 60),
    event(CompoundEventType.NOTE, 96, 0, 60, 96, 90),
  ]);
  const noteOn = schedule.timeline.find((item) => item.kind === 'note-on');
  const noteOff = schedule.timeline.find((item) => item.kind === 'note-off');
  assert.equal(noteOn.time, 1);
  assert.equal(noteOff.time, 2);
  assert.equal(schedule.finalBpm, 60);
});
