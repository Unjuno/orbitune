import { CompoundBrowserRuntime } from './compound-runtime.mjs';
import { compoundEventsToMidiBytes, decodeCompoundRecords } from './compound-midi.mjs';
import { CompoundPreviewPlayer } from './compound-player.mjs';
import { CompoundStreamingPreviewSink, InfiniteCompoundStream } from './compound-stream.mjs';
import { availableCompoundVariants, validateCompoundRuntimeConfig } from './compound-variant.mjs';

const variantSelect = document.getElementById('compound-variant');
const modelMeta = document.getElementById('compound-model-meta');
const eventCount = document.getElementById('compound-events');
const temperature = document.getElementById('compound-temperature');
const temperatureValue = document.getElementById('compound-temperature-value');
const topP = document.getElementById('compound-top-p');
const topPValue = document.getElementById('compound-top-p-value');
const generateButton = document.getElementById('compound-generate');
const playButton = document.getElementById('compound-play');
const stopButton = document.getElementById('compound-stop');
const streamStartButton = document.getElementById('compound-stream-start');
const streamStopButton = document.getElementById('compound-stream-stop');
const installButton = document.getElementById('compound-install');
const pwaStatus = document.getElementById('compound-pwa-status');
const streamStats = document.getElementById('compound-stream-stats');
const downloadLink = document.getElementById('compound-download');
const status = document.getElementById('compound-status');

let config = null;
let runtime = null;
let loadedVariant = null;
let objectUrl = null;
let generatedEvents = [];
let infiniteStream = null;
let deferredInstallPrompt = null;
const player = new CompoundPreviewPlayer();
const streamingSink = new CompoundStreamingPreviewSink();

function setStatus(message) { status.textContent = message; }
function updateLabels() {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
  topPValue.textContent = Number(topP.value).toFixed(2);
}
async function loadJson(url) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}
function availableVariants() { return config ? availableCompoundVariants(config) : []; }
function selectedVariant() { return availableVariants().find((variant) => variant.id === variantSelect.value) || null; }

async function requestPersistentStorage() {
  if (!navigator.storage?.persist) return false;
  try { return await navigator.storage.persist(); }
  catch { return false; }
}

async function ensureRuntime() {
  const variant = selectedVariant();
  if (!variant) throw new Error('No published Compound model variant is available');
  if (!globalThis.ort) throw new Error('ONNX Runtime Web failed to load');
  if (runtime && loadedVariant === variant.id) return runtime;
  runtime = new CompoundBrowserRuntime(globalThis.ort);
  await requestPersistentStorage();
  setStatus(`Loading and SHA-256 verifying ${variant.display_name || variant.id}…\nVerified model bytes are persisted locally when IndexedDB is available.`);
  await runtime.loadModels({
    streamUrl: variant.stream.url, streamSha256: variant.stream.sha256,
    decoderUrl: variant.decoder.url, decoderSha256: variant.decoder.sha256,
    executionProviders: variant.execution_providers || ['wasm'],
  });
  loadedVariant = variant.id;
  return runtime;
}

async function registerPwa() {
  if (!('serviceWorker' in navigator)) {
    pwaStatus.textContent = 'PWA service workers are not supported by this browser.';
    return;
  }
  try {
    await navigator.serviceWorker.register('./sw.js', { scope: './' });
    pwaStatus.textContent = navigator.serviceWorker.controller
      ? 'PWA shell active. Verified model files can be reused from local persistent storage.'
      : 'PWA shell installed. Reload once to activate offline control.';
  } catch (error) {
    pwaStatus.textContent = `PWA registration failed: ${error.message}`;
  }
}

async function initialize() {
  updateLabels();
  registerPwa();
  try {
    config = await loadJson('./compound-runtime-config.json');
    validateCompoundRuntimeConfig(config);
  } catch (error) {
    generateButton.disabled = true;
    streamStartButton.disabled = true;
    setStatus(`Compound runtime config failed closed: ${error.message}`);
    return;
  }
  modelMeta.textContent = `${config.model_id} · ${config.architecture} · ${config.distribution_scope}`;
  const variants = availableVariants();
  variantSelect.replaceChildren();
  if (!variants.length) {
    variantSelect.appendChild(new Option('Model artifacts not published', ''));
    generateButton.disabled = true;
    streamStartButton.disabled = true;
    setStatus('The native Compound browser runtime is implemented, but no verified browser model variant is published yet. The UI remains fail-closed until exact ONNX URLs and SHA-256 values pass the release gate.');
    return;
  }
  for (const variant of variants) variantSelect.appendChild(new Option(variant.display_name || variant.id, variant.id));
  generateButton.disabled = false;
  streamStartButton.disabled = false;
  setStatus('Ready. Batch generation and continuous generation run locally through ONNX Runtime Web.');
}

async function generate() {
  generateButton.disabled = true; playButton.disabled = true; stopButton.disabled = true; downloadLink.hidden = true; player.stop();
  if (infiniteStream) await stopInfinite();
  if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = null; generatedEvents = [];
  try {
    const activeRuntime = await ensureRuntime(); const count = Number(eventCount.value); const temp = Number(temperature.value); const p = Number(topP.value);
    const started = performance.now();
    const result = await activeRuntime.generate({ maxNewEvents: count, temperature: temp, topP: p, onProgress: ({ generated, total }) => {
      if (generated === 1 || generated % 16 === 0 || generated === total) setStatus(`Generating… ${generated}/${total} events`);
    }});
    generatedEvents = decodeCompoundRecords(result.records);
    const midi = compoundEventsToMidiBytes(generatedEvents);
    objectUrl = URL.createObjectURL(new Blob([midi], { type: 'audio/midi' }));
    downloadLink.href = objectUrl; downloadLink.download = `orbitune-${selectedVariant()?.id || 'compound'}-${count}events.mid`; downloadLink.hidden = false;
    playButton.disabled = !generatedEvents.some((event) => event.type === 0); stopButton.disabled = false;
    setStatus([`Generation complete.`, `variant=${selectedVariant()?.id || 'unknown'}`, `new_events=${count}`, `decoded_events=${generatedEvents.length}`, `temperature=${temp.toFixed(2)}`, `top_p=${p.toFixed(2)}`, `elapsed_ms=${(performance.now() - started).toFixed(0)}`].join('\n'));
  } catch (error) { setStatus(`Generation failed: ${error.message}`); }
  finally {
    try { generateButton.disabled = !availableVariants().length; }
    catch { generateButton.disabled = true; }
  }
}

async function startInfinite() {
  if (infiniteStream?.running) return;
  player.stop();
  try {
    const activeRuntime = await ensureRuntime();
    const temp = Number(temperature.value); const p = Number(topP.value);
    const stream = new InfiniteCompoundStream({
      runtime: activeRuntime,
      sink: streamingSink,
      targetLookaheadSeconds: 3,
      maxRecentRecords: 512,
      onRecord: ({ generated, lookaheadSeconds, recentRecords, error }) => {
        if (error) {
          setStatus(`Continuous generation stopped: ${error.message}`);
          streamStartButton.disabled = !availableVariants().length;
          streamStopButton.disabled = true;
          variantSelect.disabled = false;
          generateButton.disabled = !availableVariants().length;
          return;
        }
        if (generated === 1 || generated % 16 === 0) {
          streamStats.textContent = `generated=${generated} · buffered≈${lookaheadSeconds.toFixed(2)}s · retained_recent=${recentRecords}/512`;
        }
      },
    });
    infiniteStream = stream;
    await stream.start({ temperature: temp, topP: p });
    streamStartButton.disabled = true;
    streamStopButton.disabled = false;
    variantSelect.disabled = true;
    generateButton.disabled = true;
    setStatus(`Continuous generation active.\nvariant=${selectedVariant()?.id || 'unknown'}\nNo complete-song history is retained; model state and playback look-ahead stay bounded.`);
  } catch (error) {
    infiniteStream = null;
    setStatus(`Continuous generation failed: ${error.message}`);
  }
}

async function stopInfinite() {
  const stream = infiniteStream;
  infiniteStream = null;
  if (stream) await stream.stop();
  streamStartButton.disabled = !availableVariants().length;
  streamStopButton.disabled = true;
  variantSelect.disabled = false;
  generateButton.disabled = !availableVariants().length;
  if (stream) streamStats.textContent = `stopped · generated=${stream.generated} · retained_recent=${stream.recentRecords().length}`;
}

temperature.addEventListener('input', updateLabels); topP.addEventListener('input', updateLabels);
variantSelect.addEventListener('change', async () => { await stopInfinite(); runtime = null; loadedVariant = null; });
generateButton.addEventListener('click', generate);
streamStartButton.addEventListener('click', startInfinite);
streamStopButton.addEventListener('click', stopInfinite);
playButton.addEventListener('click', async () => { try { const notes = await player.play(generatedEvents); setStatus(`${status.textContent}\npreview_notes=${notes}`); } catch (error) { setStatus(`Playback failed: ${error.message}`); } });
stopButton.addEventListener('click', () => player.stop());
installButton.addEventListener('click', async () => {
  if (!deferredInstallPrompt) return;
  await deferredInstallPrompt.prompt();
  deferredInstallPrompt = null;
  installButton.hidden = true;
});
window.addEventListener('beforeinstallprompt', (event) => {
  event.preventDefault();
  deferredInstallPrompt = event;
  installButton.hidden = false;
});
window.addEventListener('appinstalled', () => {
  deferredInstallPrompt = null;
  installButton.hidden = true;
  pwaStatus.textContent = 'Orbitune is installed as an app.';
});
window.addEventListener('beforeunload', () => {
  player.stop();
  streamingSink.stop();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
});
initialize();
