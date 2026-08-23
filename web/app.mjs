import { OrbituneBrowserRuntime, eventsToMidiBytes } from './orbitune-runtime.mjs';
import { createVerifiedModelSession } from './model-loader.mjs';
import { assertAdapterBaseSha256 } from './adapter-compat.mjs';

const baseSelect = document.getElementById('base');
const baseMeta = document.getElementById('base-meta');
const adapterSelect = document.getElementById('adapter');
const adapterMeta = document.getElementById('adapter-meta');
const bpmInput = document.getElementById('bpm');
const barsSelect = document.getElementById('bars');
const temperature = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const generateButton = document.getElementById('generate');
const downloadLink = document.getElementById('download');
const status = document.getElementById('status');

let bases = { bases: [] };
let adapters = { adapters: [] };
let runtime = null;
let loadedBaseId = null;
let objectUrl = null;

function setStatus(message) { status.textContent = message; }
function selectedBase() { return bases.bases.find((item) => item.id === baseSelect.value) || null; }
function selectedAdapter() { return adapters.adapters.find((item) => item.id === adapterSelect.value) || null; }
function updateTemperatureLabel() { temperatureValue.textContent = Number(temperature.value).toFixed(2); }

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

function refreshAdapterOptions() {
  const current = adapterSelect.value;
  adapterSelect.replaceChildren(new Option('Base only', ''));
  const base = selectedBase();
  for (const adapter of adapters.adapters || []) {
    if (!base || adapter.base_model !== base.id) continue;
    if (adapter.base_sha256?.toLowerCase() !== base.checkpoint_sha256?.toLowerCase()) continue;
    adapterSelect.appendChild(new Option(adapter.display_name || adapter.id, adapter.id));
  }
  if ([...adapterSelect.options].some((option) => option.value === current)) adapterSelect.value = current;
  adapterMeta.textContent = `${adapterSelect.options.length - 1} compatible adapter(s)`;
}

function applyBaseMetadata() {
  const base = selectedBase();
  if (!base) { baseMeta.textContent = 'No Base selected'; return; }
  baseMeta.textContent = `${base.parameter_count.toLocaleString()} params · ${base.architecture} · ${base.license}`;
  refreshAdapterOptions();
}

function applyAdapterDefaults() {
  const adapter = selectedAdapter();
  if (!adapter) { adapterMeta.textContent = `${adapterSelect.options.length - 1} compatible adapter(s) · Base only`; return; }
  if (adapter.base_model !== baseSelect.value) {
    baseSelect.value = adapter.base_model;
    applyBaseMetadata();
    adapterSelect.value = adapter.id;
  }
  const defaults = adapter.generation_defaults || {};
  if (defaults.bpm) bpmInput.value = defaults.bpm;
  if (defaults.bars) barsSelect.value = defaults.bars;
  if (defaults.temperature) temperature.value = defaults.temperature;
  updateTemperatureLabel();
  adapterMeta.textContent = [adapter.source, adapter.family, ...(adapter.tags || [])].filter(Boolean).join(' · ') || adapter.id;
}

async function ensureBaseLoaded() {
  const base = selectedBase();
  if (!base) throw new Error('No Base is available');
  if (base.web_runtime_compatible !== true) throw new Error(`Base ${base.id} is not compatible with the current Theory-REMI Web runtime`);
  if (!globalThis.ort) throw new Error('ONNX Runtime Web failed to load');
  if (runtime && loadedBaseId === base.id) return base;
  runtime = new OrbituneBrowserRuntime(globalThis.ort);
  setStatus(`Downloading and verifying Base ${base.display_name || base.id}…`);
  runtime.session = await createVerifiedModelSession(globalThis.ort, base.web_onnx_url, {
    expectedSha256: base.web_onnx_sha256,
    executionProviders: ['wasm'],
  });
  loadedBaseId = base.id;
  runtime.clearAdapter();
  return base;
}

async function initialize() {
  updateTemperatureLabel();
  [bases, adapters] = await Promise.all([
    loadJson('./data/bases.json', { bases: [] }),
    loadJson('./data/adapters.json', { adapters: [] }),
  ]);
  const allBases = Array.isArray(bases.bases) ? bases.bases : [];
  const compatibleBases = allBases.filter((base) => base.web_runtime_compatible === true);
  bases = { ...bases, bases: compatibleBases };
  for (const base of compatibleBases) baseSelect.appendChild(new Option(base.display_name || base.id, base.id));
  if (!compatibleBases.length) {
    generateButton.disabled = true;
    baseMeta.textContent = 'No compatible Base models are committed yet.';
    adapterMeta.textContent = 'Adapters become available with their compatible Base.';
    const hidden = allBases.length - compatibleBases.length;
    setStatus(hidden > 0
      ? `Repository contains ${hidden} Base model(s), but none match the current Theory-REMI Web runtime ABI.`
      : 'Repository runtime is ready, but no Base model has been published in bases/.');
    return;
  }
  applyBaseMetadata();
  generateButton.disabled = false;
  setStatus('Select a Base or Adapter. The compatible Base is resolved automatically.');
}

async function loadSelectedAdapter(base) {
  const adapter = selectedAdapter();
  runtime.clearAdapter();
  if (!adapter) return;
  if (adapter.base_model !== base.id) throw new Error(`Adapter requires Base ${adapter.base_model}`);
  if (adapter.base_sha256?.toLowerCase() !== base.checkpoint_sha256?.toLowerCase()) {
    throw new Error('Adapter registry Base hash does not match the selected Base checkpoint');
  }
  const response = await fetch(adapter.adapter_url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`adapter download failed: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  assertAdapterBaseSha256(bytes, base.checkpoint_sha256);
  runtime.loadAdapter(bytes);
}

async function generate() {
  generateButton.disabled = true;
  downloadLink.hidden = true;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  const bpm = Number(bpmInput.value); const bars = Number(barsSelect.value); const temp = Number(temperature.value); const adapter = selectedAdapter();
  try {
    const base = await ensureBaseLoaded();
    await loadSelectedAdapter(base);
    setStatus(`Generating ${bars} bars with ${adapter?.display_name || base.display_name || base.id}…`);
    const started = performance.now();
    const result = await runtime.generate({ bars, temperature: temp, topP: 0.92 });
    const elapsed = performance.now() - started;
    const midi = eventsToMidiBytes(result.events, bpm);
    objectUrl = URL.createObjectURL(new Blob([midi], { type: 'audio/midi' }));
    downloadLink.href = objectUrl;
    downloadLink.download = `orbitune-${adapter?.id || base.id}-${bars}bars.mid`;
    downloadLink.hidden = false;
    setStatus([`Generation complete.`, `base=${base.id}`, `adapter=${adapter?.id || 'none'}`, `bars=${bars}`, `notes=${result.events.length}`, `temperature=${temp.toFixed(2)}`, `elapsed_ms=${elapsed.toFixed(0)}`].join('\n'));
  } catch (error) {
    setStatus(`Generation failed: ${error.message}`);
  } finally {
    generateButton.disabled = !bases.bases?.length;
  }
}

temperature.addEventListener('input', updateTemperatureLabel);
baseSelect.addEventListener('change', () => { loadedBaseId = null; runtime = null; applyBaseMetadata(); adapterSelect.value = ''; });
adapterSelect.addEventListener('change', applyAdapterDefaults);
generateButton.addEventListener('click', generate);
initialize();
