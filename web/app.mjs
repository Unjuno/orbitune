import { OrbituneBrowserRuntime, eventsToMidiBytes } from './orbitune-runtime.mjs';
import { createVerifiedModelSession } from './model-loader.mjs';
import { assertAdapterBaseSha256 } from './adapter-compat.mjs';

const adapterSelect = document.getElementById('adapter');
const adapterMeta = document.getElementById('adapter-meta');
const bpmInput = document.getElementById('bpm');
const barsSelect = document.getElementById('bars');
const temperature = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const generateButton = document.getElementById('generate');
const downloadLink = document.getElementById('download');
const status = document.getElementById('status');

let registry = { adapters: [] };
let runtimeConfig = { model_url: '', model_sha256: '', base_sha256: '', execution_providers: ['wasm'] };
let runtime = null;
let objectUrl = null;

function setStatus(message) { status.textContent = message; }
function selectedAdapter() { return registry.adapters.find((item) => item.id === adapterSelect.value) || null; }
function updateTemperatureLabel() { temperatureValue.textContent = Number(temperature.value).toFixed(2); }

function applyAdapterDefaults() {
  const adapter = selectedAdapter();
  if (!adapter) { adapterMeta.textContent = 'Base model only'; return; }
  const defaults = adapter.generation_defaults || {};
  if (defaults.bpm) bpmInput.value = defaults.bpm;
  if (defaults.bars) barsSelect.value = defaults.bars;
  if (defaults.temperature) temperature.value = defaults.temperature;
  updateTemperatureLabel();
  adapterMeta.textContent = [adapter.source, adapter.family, ...(adapter.tags || [])].filter(Boolean).join(' · ') || adapter.id;
}

async function loadJson(url, fallback) {
  try {
    const response = await fetch(url, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn(`Failed to load ${url}`, error);
    return fallback;
  }
}

async function initialize() {
  updateTemperatureLabel();
  registry = await loadJson('./data/adapters.json', { adapters: [] });
  runtimeConfig = await loadJson('./runtime-config.json', runtimeConfig);
  for (const adapter of registry.adapters || []) {
    const option = document.createElement('option');
    option.value = adapter.id;
    option.textContent = adapter.display_name || adapter.id;
    adapterSelect.appendChild(option);
  }
  adapterMeta.textContent = registry.adapters?.length ? `${registry.adapters.length} bundled adapter(s)` : 'No bundled adapters yet — base model only';
  if (!runtimeConfig.model_url) {
    generateButton.disabled = true;
    setStatus('Browser runtime is wired, but the immutable Orbitune Base has not been published yet.');
    return;
  }
  if (!globalThis.ort) {
    generateButton.disabled = true;
    setStatus('ONNX Runtime Web failed to load.');
    return;
  }
  runtime = new OrbituneBrowserRuntime(globalThis.ort);
  setStatus('Downloading and verifying Orbitune Base…');
  try {
    runtime.session = await createVerifiedModelSession(globalThis.ort, runtimeConfig.model_url, {
      expectedSha256: runtimeConfig.model_sha256 || '',
      executionProviders: runtimeConfig.execution_providers || ['wasm'],
    });
    generateButton.disabled = false;
    setStatus('Base model verified. Ready to generate locally in this browser.');
  } catch (error) {
    generateButton.disabled = true;
    setStatus(`Failed to load Base model: ${error.message}`);
  }
}

async function loadSelectedAdapter() {
  const adapter = selectedAdapter();
  runtime.clearAdapter();
  if (!adapter) return;
  const response = await fetch(adapter.adapter_url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`adapter download failed: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  assertAdapterBaseSha256(bytes, runtimeConfig.base_sha256);
  runtime.loadAdapter(bytes);
}

async function generate() {
  if (!runtime) return;
  generateButton.disabled = true;
  downloadLink.hidden = true;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  const bpm = Number(bpmInput.value);
  const bars = Number(barsSelect.value);
  const temp = Number(temperature.value);
  const adapter = selectedAdapter();
  setStatus(`Loading ${adapter?.display_name || 'Base only'} and generating ${bars} bars…`);
  try {
    await loadSelectedAdapter();
    const started = performance.now();
    const result = await runtime.generate({ bars, temperature: temp, topP: 0.92 });
    const elapsed = performance.now() - started;
    const midi = eventsToMidiBytes(result.events, bpm);
    const blob = new Blob([midi], { type: 'audio/midi' });
    objectUrl = URL.createObjectURL(blob);
    downloadLink.href = objectUrl;
    downloadLink.download = `orbitune-${adapter?.id || 'base'}-${bars}bars.mid`;
    downloadLink.hidden = false;
    setStatus(['Generation complete.', `adapter=${adapter?.id || 'base-only'}`, `bars=${bars}`, `notes=${result.events.length}`, `temperature=${temp.toFixed(2)}`, `elapsed_ms=${elapsed.toFixed(0)}`].join('\n'));
  } catch (error) {
    setStatus(`Generation failed: ${error.message}`);
  } finally {
    generateButton.disabled = false;
  }
}

temperature.addEventListener('input', updateTemperatureLabel);
adapterSelect.addEventListener('change', applyAdapterDefaults);
generateButton.addEventListener('click', generate);
initialize();
