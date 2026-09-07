import { CompoundBrowserRuntime } from './compound-runtime.mjs';
import { compoundEventsToMidiBytes, decodeCompoundRecords } from './compound-midi.mjs';
import { CompoundPreviewPlayer } from './compound-player.mjs';
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
const downloadLink = document.getElementById('compound-download');
const status = document.getElementById('compound-status');

let config = null; let runtime = null; let loadedVariant = null; let objectUrl = null; let generatedEvents = [];
const player = new CompoundPreviewPlayer();

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

async function ensureRuntime() {
  const variant = selectedVariant();
  if (!variant) throw new Error('No published Compound model variant is available');
  if (!globalThis.ort) throw new Error('ONNX Runtime Web failed to load');
  if (runtime && loadedVariant === variant.id) return runtime;
  runtime = new CompoundBrowserRuntime(globalThis.ort);
  setStatus(`Downloading and SHA-256 verifying ${variant.display_name || variant.id}…`);
  await runtime.loadModels({
    streamUrl: variant.stream.url, streamSha256: variant.stream.sha256,
    decoderUrl: variant.decoder.url, decoderSha256: variant.decoder.sha256,
    executionProviders: variant.execution_providers || ['wasm'],
  });
  loadedVariant = variant.id;
  return runtime;
}

async function initialize() {
  updateLabels();
  try {
    config = await loadJson('./compound-runtime-config.json');
    validateCompoundRuntimeConfig(config);
  } catch (error) {
    generateButton.disabled = true;
    setStatus(`Compound runtime config failed closed: ${error.message}`);
    return;
  }
  modelMeta.textContent = `${config.model_id} · ${config.architecture} · ${config.distribution_scope}`;
  const variants = availableVariants();
  variantSelect.replaceChildren();
  if (!variants.length) {
    variantSelect.appendChild(new Option('Model artifacts not published', ''));
    generateButton.disabled = true;
    setStatus('The native Compound browser runtime is implemented, but model/ONNX redistribution is still frozen pending review. No model URL is configured.');
    return;
  }
  for (const variant of variants) variantSelect.appendChild(new Option(variant.display_name || variant.id, variant.id));
  generateButton.disabled = false;
  setStatus('Ready. Generation runs locally in your browser through ONNX Runtime Web/WASM.');
}

async function generate() {
  generateButton.disabled = true; playButton.disabled = true; stopButton.disabled = true; downloadLink.hidden = true; player.stop();
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

temperature.addEventListener('input', updateLabels); topP.addEventListener('input', updateLabels);
variantSelect.addEventListener('change', () => { runtime = null; loadedVariant = null; });
generateButton.addEventListener('click', generate);
playButton.addEventListener('click', async () => { try { const notes = await player.play(generatedEvents); setStatus(`${status.textContent}\npreview_notes=${notes}`); } catch (error) { setStatus(`Playback failed: ${error.message}`); } });
stopButton.addEventListener('click', () => player.stop());
window.addEventListener('beforeunload', () => { player.stop(); if (objectUrl) URL.revokeObjectURL(objectUrl); });
initialize();
