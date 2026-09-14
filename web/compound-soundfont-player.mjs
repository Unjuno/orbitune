import { canonicalizeCompoundEvents } from './compound-midi.mjs';
import { CompoundEventType } from './compound-runtime.mjs';
import { sha256Hex } from './model-loader.mjs';

export const SOUNDFONT_RELEASE_URL = './soundfont-release.json';
export const SOUNDFONT_ASSET_URL = './soundfonts/GeneralUser-GS.sf2';
export const SPESSASYNTH_BUNDLE_URL = './vendor/spessasynth-bundle.mjs';
export const SPESSASYNTH_WORKLET_URL = './vendor/spessasynth_processor.min.js';
export const SOUNDFONT_CACHE_NAME = 'orbitune-soundfont-v1';

function normalizeSha256(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(normalized)) throw new Error('SoundFont SHA-256 must be 64 hexadecimal characters');
  return normalized;
}

export function validateSoundFontRelease(release) {
  if (!release || typeof release !== 'object') throw new Error('SoundFont release metadata is required');
  if (release.schema !== 'orbitune-web-soundfont-v1') throw new Error(`unsupported SoundFont release schema: ${release.schema}`);
  if (!release.id || !release.filename) throw new Error('SoundFont release id and filename are required');
  if (release.format !== 'sf2') throw new Error(`unsupported SoundFont format: ${release.format}`);
  if (!Number.isInteger(release.bytes) || release.bytes <= 0) throw new Error('SoundFont release byte count must be positive');
  normalizeSha256(release.sha256);
  if (release.synth_engine?.package !== 'spessasynth_lib') throw new Error('SoundFont release must bind the SpessaSynth browser engine');
  if (!release.synth_engine?.version) throw new Error('SpessaSynth version is required');
  return release;
}

export async function loadSoundFontRelease({ fetchImpl = globalThis.fetch, releaseUrl = SOUNDFONT_RELEASE_URL } = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable for SoundFont metadata');
  const response = await fetchImpl(releaseUrl, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`SoundFont release metadata failed: HTTP ${response.status}`);
  return validateSoundFontRelease(await response.json());
}

async function verifySoundFontBytes(bytes, release) {
  if (!(bytes instanceof ArrayBuffer)) throw new Error('SoundFont bytes must be an ArrayBuffer');
  if (bytes.byteLength !== release.bytes) throw new Error(`SoundFont byte-size mismatch: ${bytes.byteLength} != ${release.bytes}`);
  const actual = await sha256Hex(bytes);
  const expected = normalizeSha256(release.sha256);
  if (actual !== expected) throw new Error(`SoundFont SHA-256 mismatch: ${actual} != ${expected}`);
  return actual;
}

export async function loadVerifiedSoundFontBytes({
  fetchImpl = globalThis.fetch,
  cacheStorage = globalThis.caches,
  release = null,
  releaseUrl = SOUNDFONT_RELEASE_URL,
  assetUrl = SOUNDFONT_ASSET_URL,
  persist = true,
} = {}) {
  const metadata = release || await loadSoundFontRelease({ fetchImpl, releaseUrl });
  validateSoundFontRelease(metadata);
  let cache = null;
  if (cacheStorage?.open) cache = await cacheStorage.open(SOUNDFONT_CACHE_NAME);
  if (cache) {
    const cached = await cache.match(assetUrl);
    if (cached) {
      const bytes = await cached.arrayBuffer();
      try {
        await verifySoundFontBytes(bytes, metadata);
        return { bytes, release: metadata, source: 'cache' };
      } catch {
        await cache.delete(assetUrl);
      }
    }
  }
  if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable for SoundFont bytes');
  const response = await fetchImpl(assetUrl, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`SoundFont download failed: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  await verifySoundFontBytes(bytes, metadata);
  if (cache && persist) await cache.put(assetUrl, new Response(bytes, { headers: { 'content-type': 'audio/sf2' } }));
  return { bytes, release: metadata, source: 'network' };
}

export async function warmOfflineSoundFont(options = {}) {
  const loaded = await loadVerifiedSoundFontBytes({ ...options, persist: true });
  return {
    id: loaded.release.id,
    displayName: loaded.release.display_name || loaded.release.id,
    bytes: loaded.bytes.byteLength,
    source: loaded.source,
    sha256: loaded.release.sha256,
  };
}

function compareBytes(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

function buildTempoMap(canonical, defaultBpm) {
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
  const secondsAt = (step) => {
    let segment = segments[0];
    for (const candidate of segments) {
      if (candidate.step > step) break;
      segment = candidate;
    }
    return segment.seconds + (step - segment.step) / 96 * 60 / segment.bpm;
  };
  return { secondsAt, finalBpm: segments.at(-1).bpm };
}

function pushEventMessages(timeline, event) {
  const channel = event.channel & 0x0f;
  if (event.type === CompoundEventType.NOTE) {
    timeline.push({ step: event.step, priority: 20, kind: 'note-on', message: [0x90 | channel, event.a1, event.a3] });
    timeline.push({ step: event.step + event.a2, priority: 10, kind: 'note-off', message: [0x80 | channel, event.a1, 0] });
  } else if (event.type === CompoundEventType.CC) {
    timeline.push({ step: event.step, priority: 5, kind: 'cc', message: [0xB0 | channel, event.a1, event.a2] });
  } else if (event.type === CompoundEventType.PROGRAM) {
    timeline.push({ step: event.step, priority: 4, kind: 'program', message: [0xC0 | channel, event.a1] });
  } else if (event.type === CompoundEventType.BANK) {
    timeline.push({ step: event.step, priority: 2, kind: 'bank-msb', message: [0xB0 | channel, 0, event.a1] });
    timeline.push({ step: event.step, priority: 3, kind: 'bank-lsb', message: [0xB0 | channel, 32, event.a2] });
  } else if (event.type === CompoundEventType.PEDAL) {
    timeline.push({ step: event.step, priority: 5, kind: 'pedal', message: [0xB0 | channel, 64, event.a1 ? 127 : 0] });
  } else if (event.type === CompoundEventType.PITCH_BEND) {
    timeline.push({ step: event.step, priority: 6, kind: 'pitch-bend', message: [0xE0 | channel, event.a1 & 0x7f, Math.floor(event.a1 / 128) & 0x7f] });
  } else if (event.type === CompoundEventType.CHANNEL_PRESSURE) {
    timeline.push({ step: event.step, priority: 6, kind: 'channel-pressure', message: [0xD0 | channel, event.a1] });
  } else if (event.type === CompoundEventType.POLY_PRESSURE) {
    timeline.push({ step: event.step, priority: 6, kind: 'poly-pressure', message: [0xA0 | channel, event.a1, event.a2] });
  }
}

export function buildSampledPlaybackSchedule(events, { defaultBpm = 120 } = {}) {
  if (!(defaultBpm > 0)) throw new Error('defaultBpm must be positive');
  const canonical = canonicalizeCompoundEvents(events);
  const { secondsAt, finalBpm } = buildTempoMap(canonical, defaultBpm);
  const raw = [];
  for (const event of canonical) pushEventMessages(raw, event);
  raw.sort((a, b) => a.step - b.step || a.priority - b.priority || compareBytes(a.message, b.message));
  const timeline = raw.map((item) => ({ ...item, time: secondsAt(item.step) }));
  const noteCount = timeline.reduce((count, item) => count + (item.kind === 'note-on' ? 1 : 0), 0);
  const spanStep = canonical.reduce((maximum, event) => Math.max(maximum, event.step), 0);
  return { timeline, noteCount, spanStep, spanSeconds: secondsAt(spanStep), finalBpm };
}

export class SampledSoundFontSynth {
  constructor({
    audioContextFactory = () => new (globalThis.AudioContext || globalThis.webkitAudioContext)(),
    fetchImpl = globalThis.fetch,
    cacheStorage = globalThis.caches,
    moduleLoader = () => import(SPESSASYNTH_BUNDLE_URL),
    workletUrl = SPESSASYNTH_WORKLET_URL,
    soundFontUrl = SOUNDFONT_ASSET_URL,
    releaseUrl = SOUNDFONT_RELEASE_URL,
  } = {}) {
    this.audioContextFactory = audioContextFactory;
    this.fetchImpl = fetchImpl;
    this.cacheStorage = cacheStorage;
    this.moduleLoader = moduleLoader;
    this.workletUrl = workletUrl;
    this.soundFontUrl = soundFontUrl;
    this.releaseUrl = releaseUrl;
    this.context = null;
    this.synth = null;
    this.release = null;
    this.initializing = null;
    this.workletLoaded = false;
  }

  async ensureStarted() {
    if (this.synth) {
      if (this.context?.state === 'suspended') await this.context.resume();
      return this;
    }
    if (this.initializing) return this.initializing;
    this.initializing = (async () => {
      const context = this.context ||= this.audioContextFactory();
      if (!context?.audioWorklet?.addModule) throw new Error('AudioWorklet is required for sampled SoundFont playback');
      if (context.state === 'suspended') await context.resume();
      const workletPromise = this.workletLoaded
        ? Promise.resolve()
        : context.audioWorklet.addModule(this.workletUrl).then(() => { this.workletLoaded = true; });
      const [{ WorkletSynthesizer }, loaded] = await Promise.all([
        this.moduleLoader(),
        loadVerifiedSoundFontBytes({
          fetchImpl: this.fetchImpl,
          cacheStorage: this.cacheStorage,
          releaseUrl: this.releaseUrl,
          assetUrl: this.soundFontUrl,
        }),
        workletPromise,
      ]);
      if (typeof WorkletSynthesizer !== 'function') throw new Error('SpessaSynth WorkletSynthesizer export is unavailable');
      const synth = new WorkletSynthesizer(context);
      synth.connect(context.destination);
      await synth.soundBankManager.addSoundBank(loaded.bytes, loaded.release.id);
      await synth.isReady;
      synth.reset();
      try { synth.midiChannels?.[9]?.setDrums?.(true); } catch {}
      this.synth = synth;
      this.release = loaded.release;
      this.initializing = null;
      return this;
    })().catch((error) => {
      this.initializing = null;
      throw error;
    });
    return this.initializing;
  }

  async schedule(events, { origin = null, defaultBpm = 120 } = {}) {
    await this.ensureStarted();
    const schedule = buildSampledPlaybackSchedule(events, { defaultBpm });
    const start = origin ?? (this.context.currentTime + 0.05);
    for (const item of schedule.timeline) {
      this.synth.sendMessage(item.message, 0, { time: start + item.time });
    }
    return { ...schedule, origin: start, release: this.release };
  }

  async pause() {
    if (this.context?.state === 'running') await this.context.suspend();
  }

  async resume() {
    if (this.context?.state === 'suspended') await this.context.resume();
  }

  stop({ hard = false } = {}) {
    try { this.synth?.stopAll?.(true); } catch {}
    if (hard) {
      try { this.synth?.destroy?.(); } catch {}
      this.synth = null;
      this.release = null;
      this.initializing = null;
      return;
    }
    try { this.synth?.reset?.(); } catch {}
    try { this.synth?.midiChannels?.[9]?.setDrums?.(true); } catch {}
  }

  destroy() {
    this.stop({ hard: true });
  }
}
