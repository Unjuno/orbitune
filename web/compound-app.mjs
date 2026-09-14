import { CompoundBrowserRuntime } from './compound-runtime.mjs';
import { CompoundDecodingRuntime } from './compound-decoding-runtime.mjs';
import { DECODING_PRESETS, decodingPreset } from './decoding-presets.mjs';
import { CompoundEventMonitor } from './compound-event-monitor.mjs';
import { compoundEventsToMidiBytes, decodeCompoundRecords } from './compound-midi.mjs';
import { CompoundPreviewPlayer } from './compound-player.mjs';
import { CompoundStreamingPreviewPlayer } from './compound-live-player.mjs';
import { warmOfflineSoundFont } from './compound-soundfont-player.mjs';
import { CompoundStreamSession } from './compound-stream.mjs';
import { availableCompoundVariants, validateCompoundRuntimeConfig } from './compound-variant.mjs';
import { cacheVariantArtifacts, createPersistentModelFetch, getVariantCacheStatus } from './model-cache.mjs';
import { PwaInstallController, registerOrbituneServiceWorker, warmOfflineRuntime } from './pwa.mjs';

const variantSelect = document.getElementById('compound-variant');
const modelMeta = document.getElementById('compound-model-meta');
const eventCount = document.getElementById('compound-events');
const decodingPresetSelect = document.getElementById('compound-decoding-preset');
const decodingStrategy = document.getElementById('compound-decoding-strategy');
const temperature = document.getElementById('compound-temperature');
const temperatureValue = document.getElementById('compound-temperature-value');
const structureTemperature = document.getElementById('compound-structure-temperature');
const structureTemperatureValue = document.getElementById('compound-structure-temperature-value');
const topP = document.getElementById('compound-top-p');
const topPValue = document.getElementById('compound-top-p-value');
const topK = document.getElementById('compound-top-k');
const topKValue = document.getElementById('compound-top-k-value');
const minP = document.getElementById('compound-min-p');
const minPValue = document.getElementById('compound-min-p-value');
const generateButton = document.getElementById('compound-generate');
const playButton = document.getElementById('compound-play');
const stopButton = document.getElementById('compound-stop');
const downloadLink = document.getElementById('compound-download');
const liveStartButton = document.getElementById('compound-live-start');
const livePauseButton = document.getElementById('compound-live-pause');
const liveStopButton = document.getElementById('compound-live-stop');
const offlineButton = document.getElementById('compound-offline');
const installButton = document.getElementById('compound-install');
const storageStatus = document.getElementById('compound-storage-status');
const status = document.getElementById('compound-status');

let config = null;
let runtime = null;
let loadedVariant = null;
let objectUrl = null;
let generatedEvents = [];
let liveSession = null;
let liveAbort = null;
let liveRunning = false;
let livePaused = false;
let liveGenerated = 0;
let liveBpm = 120;
const finitePlayer = new CompoundPreviewPlayer();
const livePlayer = new CompoundStreamingPreviewPlayer();
const persistentModelFetch = createPersistentModelFetch();
const installController = new PwaInstallController();
const eventMonitor = new CompoundEventMonitor({
  channelsElement: document.getElementById('compound-channel-monitor'),
  recentElement: document.getElementById('compound-event-log'),
  summaryElement: document.getElementById('compound-event-summary'),
});

function sleep(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }
function setStatus(message) { status.textContent = message; }
function updateLabels() {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
  structureTemperatureValue.textContent = Number(structureTemperature.value).toFixed(2);
  topPValue.textContent = Number(topP.value).toFixed(2);
  topKValue.textContent = String(Number(topK.value));
  minPValue.textContent = Number(minP.value).toFixed(2);
}
function currentDecodingOptions() {
  return {
    strategy: decodingStrategy.value,
    temperature: Number(temperature.value),
    structureTemperature: Number(structureTemperature.value),
    topP: Number(topP.value),
    topK: Number(topK.value),
    minP: Number(minP.value),
  };
}
function applyDecodingPreset(name) {
  if (name === 'custom') return;
  const preset = decodingPreset(name);
  decodingStrategy.value = preset.strategy;
  temperature.value = String(preset.temperature);
  structureTemperature.value = String(preset.structureTemperature);
  topP.value = String(preset.topP);
  topK.value = String(preset.topK);
  minP.value = String(preset.minP);
  updateLabels();
  runtime?.configure?.(currentDecodingOptions());
}
function markDecodingCustom() {
  decodingPresetSelect.value = 'custom';
  updateLabels();
  runtime?.configure?.(currentDecodingOptions());
}
async function loadJson(url) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}
function availableVariants() { return config ? availableCompoundVariants(config) : []; }
function selectedVariant() { return availableVariants().find((variant) => variant.id === variantSelect.value) || null; }

async function ensureRuntime() {
  const variant = selectedVariant();
  if (!variant) throw new Error('No published Compound model variant is available');
  if (!globalThis.ort) throw new Error('ONNX Runtime Web failed to load');
  if (runtime && loadedVariant === variant.id) {
    runtime.configure(currentDecodingOptions());
    return runtime;
  }
  const baseRuntime = new CompoundBrowserRuntime(globalThis.ort);
  runtime = new CompoundDecodingRuntime(baseRuntime, currentDecodingOptions());
  setStatus(`Loading and SHA-256 verifying ${variant.display_name || variant.id}…`);
  await runtime.loadModels({
    streamUrl: variant.stream.url,
    streamSha256: variant.stream.sha256,
    decoderUrl: variant.decoder.url,
    decoderSha256: variant.decoder.sha256,
    executionProviders: variant.execution_providers || ['wasm'],
    fetchImpl: persistentModelFetch,
  });
  loadedVariant = variant.id;
  return runtime;
}

async function updateStorageStatus() {
  const variant = selectedVariant();
  if (!variant) {
    storageStatus.textContent = 'Offline package: unavailable until a model variant is published.';
    offlineButton.disabled = true;
    return;
  }
  const cached = await getVariantCacheStatus(variant);
  let suffix = cached.ready ? 'model ready' : cached.supported ? 'model not downloaded' : 'unsupported by this browser';
  if (navigator.storage?.estimate) {
    const estimate = await navigator.storage.estimate();
    if (estimate.usage != null && estimate.quota) suffix += ` · storage ${(estimate.usage / 1048576).toFixed(0)}/${(estimate.quota / 1048576).toFixed(0)} MiB`;
  }
  storageStatus.textContent = `Offline package: ${suffix}. SoundFont is verified and cached on first playback or explicit offline preparation.`;
  offlineButton.disabled = !cached.supported;
}

async function prepareOffline() {
  const variant = selectedVariant();
  if (!variant) return;
  offlineButton.disabled = true;
  try {
    if (navigator.storage?.persist) await navigator.storage.persist();
    setStatus('Preparing ONNX Runtime for offline use…');
    await warmOfflineRuntime({ onAsset: ({ status: state }) => setStatus(`Preparing ONNX Runtime for offline use… ${state}`) });
    setStatus(`Downloading and verifying ${variant.display_name || variant.id} for offline use…`);
    await cacheVariantArtifacts(variant, {
      onArtifact: ({ name, status: state }) => setStatus(`Offline model ${name}: ${state}`),
    });
    setStatus('Downloading and SHA-256 verifying GeneralUser GS 2.0.3 sampled SoundFont…');
    const soundFont = await warmOfflineSoundFont();
    await updateStorageStatus();
    setStatus(`Offline preparation complete. Model, runtime and ${soundFont.displayName} (${(soundFont.bytes / 1048576).toFixed(1)} MiB) are cached locally.`);
  } catch (error) {
    setStatus(`Offline preparation failed: ${error.message}`);
  } finally {
    offlineButton.disabled = false;
  }
}

function setLiveControls() {
  liveStartButton.disabled = liveRunning || !availableVariants().length;
  livePauseButton.disabled = !liveRunning;
  liveStopButton.disabled = !liveRunning;
  livePauseButton.textContent = livePaused ? 'Resume stream' : 'Pause stream';
  variantSelect.disabled = liveRunning;
  generateButton.disabled = liveRunning || !availableVariants().length;
}

async function runLiveLoop() {
  try {
    while (liveRunning && !liveAbort.signal.aborted) {
      if (livePaused) { await sleep(100); continue; }
      const buffered = livePlayer.bufferedSeconds();
      if (buffered > 12) { await sleep(200); continue; }
      runtime.configure(currentDecodingOptions());
      liveSession.temperature = Number(temperature.value);
      liveSession.topP = Number(topP.value);
      const batch = await liveSession.generateBatch(16, { signal: liveAbort.signal });
      const localEvents = decodeCompoundRecords(batch.records);
      eventMonitor.consume(localEvents);
      const playback = await livePlayer.append(localEvents, { defaultBpm: liveBpm });
      liveBpm = playback.finalBpm;
      liveGenerated += batch.records.length;
      const decoding = currentDecodingOptions();
      setStatus([
        'Live stream running locally with sampled SoundFont playback.',
        `variant=${selectedVariant()?.id || 'unknown'}`,
        `decoding=${decoding.strategy}`,
        `temperature=${decoding.temperature.toFixed(2)}`,
        `structure_temperature=${decoding.structureTemperature.toFixed(2)}`,
        `soundfont=${playback.soundFont || 'GeneralUser GS 2.0.3'}`,
        `generated_events=${liveGenerated}`,
        `buffered_seconds=${playback.bufferedSeconds.toFixed(1)}`,
        `state_steps=${batch.state.steps}`,
        'Decoding controls affect future events without resetting model state.',
      ].join('\n'));
    }
  } catch (error) {
    if (error?.name !== 'AbortError') setStatus(`Live stream failed: ${error.message}`);
  } finally {
    if (liveRunning) await stopLive({ keepStatus: true });
  }
}

async function startLive() {
  if (liveRunning) return;
  finitePlayer.stop();
  eventMonitor.reset();
  liveAbort = new AbortController();
  liveGenerated = 0; liveBpm = 120; livePaused = false;
  try {
    setStatus('Loading SHA-256 verified GeneralUser GS 2.0.3 sampled SoundFont…');
    await livePlayer.ensureStarted();
    const activeRuntime = await ensureRuntime();
    activeRuntime.configure(currentDecodingOptions());
    liveSession = new CompoundStreamSession(activeRuntime, {
      temperature: Number(temperature.value),
      topP: Number(topP.value),
    });
    await liveSession.start({ signal: liveAbort.signal });
    liveRunning = true;
    setLiveControls();
    runLiveLoop();
  } catch (error) {
    livePlayer.stop();
    liveAbort = null;
    setStatus(`Live stream failed to start: ${error.message}`);
    setLiveControls();
  }
}

async function toggleLivePause() {
  if (!liveRunning) return;
  livePaused = !livePaused;
  if (livePaused) await livePlayer.pause(); else await livePlayer.resume();
  setLiveControls();
}

async function stopLive({ keepStatus = false } = {}) {
  if (liveAbort) liveAbort.abort();
  liveRunning = false; livePaused = false; liveSession = null; liveAbort = null;
  livePlayer.stop();
  setLiveControls();
  if (!keepStatus) setStatus(`Stream stopped after ${liveGenerated} generated events. Starting again creates a new stream and sampled-synth state.`);
}

async function initialize() {
  for (const [name, preset] of Object.entries(DECODING_PRESETS)) {
    if (![...decodingPresetSelect.options].some((option) => option.value === name)) decodingPresetSelect.add(new Option(preset.label, name));
  }
  applyDecodingPreset(decodingPresetSelect.value || 'balanced');
  eventMonitor.render();
  registerOrbituneServiceWorker().catch((error) => console.warn('Service worker registration failed', error));
  installController.subscribe((available) => { installButton.hidden = !available; });
  try {
    config = await loadJson('./compound-runtime-config.json');
    validateCompoundRuntimeConfig(config);
  } catch (error) {
    generateButton.disabled = true;
    liveStartButton.disabled = true;
    setStatus(`Compound runtime config failed closed: ${error.message}`);
    return;
  }
  modelMeta.textContent = `${config.model_id} · ${config.architecture} · ${config.distribution_scope}`;
  const variants = availableVariants();
  variantSelect.replaceChildren();
  if (!variants.length) {
    variantSelect.appendChild(new Option('Model artifacts not published', ''));
    generateButton.disabled = true;
    liveStartButton.disabled = true;
    offlineButton.disabled = true;
    setStatus('The PWA and infinite-stream runtime are ready, but no reviewed ONNX model variant is published yet.');
    await updateStorageStatus();
    return;
  }
  for (const variant of variants) variantSelect.appendChild(new Option(variant.display_name || variant.id, variant.id));
  setLiveControls();
  await updateStorageStatus();
  setStatus('Ready. Generation uses A2-512 locally with selectable decoding; playback uses the SHA-256 verified GeneralUser GS 2.0.3 sampled SoundFont.');
}

async function generate() {
  generateButton.disabled = true; playButton.disabled = true; stopButton.disabled = true; downloadLink.hidden = true; finitePlayer.stop();
  eventMonitor.reset();
  if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = null; generatedEvents = [];
  try {
    const activeRuntime = await ensureRuntime();
    const count = Number(eventCount.value);
    const decoding = currentDecodingOptions();
    activeRuntime.configure(decoding);
    const started = performance.now();
    const result = await activeRuntime.generate({ maxNewEvents: count, ...decoding, onProgress: ({ generated, total }) => {
      if (generated === 1 || generated % 16 === 0 || generated === total) setStatus(`Generating… ${generated}/${total} events`);
    }});
    generatedEvents = decodeCompoundRecords(result.records);
    eventMonitor.consume(generatedEvents);
    const midi = compoundEventsToMidiBytes(generatedEvents);
    objectUrl = URL.createObjectURL(new Blob([midi], { type: 'audio/midi' }));
    downloadLink.href = objectUrl; downloadLink.download = `orbitune-${selectedVariant()?.id || 'compound'}-${count}events.mid`; downloadLink.hidden = false;
    playButton.disabled = !generatedEvents.some((event) => event.type === 0); stopButton.disabled = false;
    setStatus([
      'Generation complete.',
      `variant=${selectedVariant()?.id || 'unknown'}`,
      `decoding=${decoding.strategy}`,
      `new_events=${count}`,
      `decoded_events=${generatedEvents.length}`,
      `temperature=${decoding.temperature.toFixed(2)}`,
      `structure_temperature=${decoding.structureTemperature.toFixed(2)}`,
      `top_p=${decoding.topP.toFixed(2)}`,
      `top_k=${decoding.topK}`,
      `min_p=${decoding.minP.toFixed(2)}`,
      `elapsed_ms=${(performance.now() - started).toFixed(0)}`,
    ].join('\n'));
  } catch (error) { setStatus(`Generation failed: ${error.message}`); }
  finally { setLiveControls(); }
}

decodingPresetSelect.addEventListener('change', () => applyDecodingPreset(decodingPresetSelect.value));
for (const control of [decodingStrategy, temperature, structureTemperature, topP, topK, minP]) control.addEventListener('input', markDecodingCustom);
variantSelect.addEventListener('change', async () => { await stopLive({ keepStatus: true }); runtime = null; loadedVariant = null; eventMonitor.reset(); await updateStorageStatus(); });
generateButton.addEventListener('click', generate);
playButton.addEventListener('click', async () => {
  try {
    setStatus('Loading SHA-256 verified GeneralUser GS 2.0.3 sampled SoundFont…');
    const notes = await finitePlayer.play(generatedEvents);
    setStatus(`Sampled preview running.\npreview_notes=${notes}\nsoundfont=GeneralUser GS 2.0.3\nsynth=SpessaSynth 4.3.14`);
  } catch (error) { setStatus(`Playback failed: ${error.message}`); }
});
stopButton.addEventListener('click', () => finitePlayer.stop());
liveStartButton.addEventListener('click', startLive);
livePauseButton.addEventListener('click', toggleLivePause);
liveStopButton.addEventListener('click', () => stopLive());
offlineButton.addEventListener('click', prepareOffline);
installButton.addEventListener('click', async () => { try { await installController.prompt(); } catch (error) { setStatus(`Install failed: ${error.message}`); } });
window.addEventListener('beforeunload', () => {
  if (liveAbort) liveAbort.abort();
  livePlayer.destroy(); finitePlayer.destroy();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
});
initialize();
