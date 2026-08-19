export const ORBITUNE_V0 = Object.freeze({
  layers: 4,
  hidden: 448,
  rank: 4,
  positionsPerBar: 16,
  context: 1024,
  velocityBins: 32,
  maxNotesPerPosition: 8,
});

export function buildVocab() {
  const tokens = ['PAD', 'BOS', 'EOS', 'BAR'];
  for (let i = 0; i < 16; i += 1) tokens.push(`POSITION_${i}`);
  for (let p = 21; p <= 108; p += 1) tokens.push(`NOTE_PITCH_${p}`);
  for (let d = 1; d <= 64; d += 1) tokens.push(`NOTE_DURATION_${d}`);
  for (let v = 1; v <= 32; v += 1) tokens.push(`VELOCITY_${v}`);
  return tokens;
}

export const VOCAB = Object.freeze(buildVocab());
export const TOKEN_TO_ID = new Map(VOCAB.map((token, id) => [token, id]));

function idsWithPrefix(prefix) { return VOCAB.map((token, id) => token.startsWith(prefix) ? id : -1).filter((id) => id >= 0); }
function readFloat32LittleEndian(view, byteOffset, count) { const values = new Float32Array(count); for (let i = 0; i < count; i += 1) values[i] = view.getFloat32(byteOffset + i * 4, true); return values; }

export function parseSafetensors(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (view.byteLength < 8) throw new Error('Safetensors file is too small');
  const headerLength = Number(view.getBigUint64(0, true));
  const headerStart = 8; const headerEnd = headerStart + headerLength;
  if (headerEnd > view.byteLength) throw new Error('Safetensors header is truncated');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, headerStart, headerLength)));
  const tensors = new Map();
  for (const [name, spec] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (spec.dtype !== 'F32') throw new Error(`Unsupported tensor dtype ${spec.dtype} for ${name}; Orbitune v0 expects F32 adapters`);
    const [relativeStart, relativeEnd] = spec.data_offsets; const bytes = relativeEnd - relativeStart;
    if (bytes < 0 || bytes % 4 !== 0) throw new Error(`Invalid tensor byte range for ${name}`);
    const absoluteStart = headerEnd + relativeStart; const absoluteEnd = headerEnd + relativeEnd;
    if (absoluteEnd > view.byteLength) throw new Error(`Tensor data is truncated for ${name}`);
    tensors.set(name, { shape: spec.shape, data: readFloat32LittleEndian(view, absoluteStart, bytes / 4) });
  }
  return { metadata: header.__metadata__ || {}, tensors };
}

function copyInto(destination, destinationOffset, tensor, expectedShape, name) {
  if (JSON.stringify(tensor.shape) !== JSON.stringify(expectedShape)) throw new Error(`${name} shape ${JSON.stringify(tensor.shape)} != ${JSON.stringify(expectedShape)}`);
  destination.set(tensor.data, destinationOffset);
}

export function emptyAdapterInputs() {
  return { loraA: new Float32Array(ORBITUNE_V0.layers * 2 * ORBITUNE_V0.rank * ORBITUNE_V0.hidden), loraB: new Float32Array(ORBITUNE_V0.layers * 2 * ORBITUNE_V0.hidden * ORBITUNE_V0.rank), scale: new Float32Array([1]) };
}

export function packAdapterSafetensors(arrayBuffer) {
  const { metadata, tensors } = parseSafetensors(arrayBuffer);
  if (metadata.format !== 'orbitune-lora-v0') throw new Error('Not an Orbitune LoRA v0 adapter');
  const rank = Number(metadata.rank); if (rank !== ORBITUNE_V0.rank) throw new Error(`Orbitune v0 browser runtime requires rank ${ORBITUNE_V0.rank}`);
  const targets = JSON.parse(metadata.target_modules || '[]'); if (JSON.stringify(targets) !== JSON.stringify(['q_proj', 'v_proj'])) throw new Error('Orbitune v0 browser runtime requires q_proj/v_proj targets');
  const alpha = Number(metadata.alpha); if (!Number.isFinite(alpha) || alpha <= 0) throw new Error('Adapter alpha metadata is missing or invalid');
  const packed = emptyAdapterInputs();
  for (let layer = 0; layer < ORBITUNE_V0.layers; layer += 1) {
    for (let targetIndex = 0; targetIndex < 2; targetIndex += 1) {
      const target = targets[targetIndex]; const prefix = `blocks.${layer}.attn.${target}`; const tensorA = tensors.get(`${prefix}.lora_a`); const tensorB = tensors.get(`${prefix}.lora_b`);
      if (!tensorA || !tensorB) throw new Error(`Adapter is missing ${prefix} LoRA tensors`);
      const aOffset = (layer * 2 + targetIndex) * ORBITUNE_V0.rank * ORBITUNE_V0.hidden; const bOffset = (layer * 2 + targetIndex) * ORBITUNE_V0.hidden * ORBITUNE_V0.rank;
      copyInto(packed.loraA, aOffset, tensorA, [ORBITUNE_V0.rank, ORBITUNE_V0.hidden], `${prefix}.lora_a`); copyInto(packed.loraB, bOffset, tensorB, [ORBITUNE_V0.hidden, ORBITUNE_V0.rank], `${prefix}.lora_b`);
    }
  }
  packed.scale[0] = alpha / rank; return packed;
}

function generationState(tokenIds) {
  let barCount = 0; let lastPosition = null; let notesAtPosition = 0; let pitchesAtPosition = new Set(); let pendingPosition = null;
  for (const id of tokenIds) {
    const token = VOCAB[id];
    if (token === 'BAR') { barCount += 1; lastPosition = null; notesAtPosition = 0; pitchesAtPosition = new Set(); pendingPosition = null; }
    else if (token?.startsWith('POSITION_')) { const position = Number(token.slice('POSITION_'.length)); if (position !== lastPosition) { lastPosition = position; notesAtPosition = 0; pitchesAtPosition = new Set(); } pendingPosition = position; }
    else if (token?.startsWith('NOTE_PITCH_') && pendingPosition === lastPosition) { pitchesAtPosition.add(Number(token.slice('NOTE_PITCH_'.length))); notesAtPosition += 1; pendingPosition = null; }
  }
  return { barCount, lastPosition, notesAtPosition, pitchesAtPosition };
}

export function allowedNextTokenIds(tokenIds, requestedBars = 8) {
  if (!(requestedBars > 0)) throw new Error('requestedBars must be positive');
  const last = VOCAB[tokenIds[tokenIds.length - 1]] || 'BOS'; const state = generationState(tokenIds);
  if (last === 'EOS') return []; if (last === 'BOS') return [TOKEN_TO_ID.get('BAR')]; if (last === 'BAR') return idsWithPrefix('POSITION_');
  if (last?.startsWith('POSITION_')) return idsWithPrefix('NOTE_PITCH_').filter((id) => !state.pitchesAtPosition.has(Number(VOCAB[id].slice('NOTE_PITCH_'.length))));
  if (last?.startsWith('NOTE_PITCH_')) return idsWithPrefix('NOTE_DURATION_'); if (last?.startsWith('NOTE_DURATION_')) return idsWithPrefix('VELOCITY_');
  if (last?.startsWith('VELOCITY_')) {
    if (state.lastPosition === null) throw new Error('velocity token encountered before a position token');
    const nextPositions = []; if (state.notesAtPosition < ORBITUNE_V0.maxNotesPerPosition) nextPositions.push(TOKEN_TO_ID.get(`POSITION_${state.lastPosition}`));
    for (let i = state.lastPosition + 1; i < ORBITUNE_V0.positionsPerBar; i += 1) nextPositions.push(TOKEN_TO_ID.get(`POSITION_${i}`));
    const canCloseBar = state.lastPosition >= 12;
    if (state.barCount >= requestedBars) return canCloseBar ? [...nextPositions, TOKEN_TO_ID.get('EOS')] : nextPositions;
    return canCloseBar ? [...nextPositions, TOKEN_TO_ID.get('BAR')] : nextPositions;
  }
  return [TOKEN_TO_ID.get('BAR')];
}

export function sampleAllowedLogits(logits, allowedIds, { temperature = 0.85, topP = 0.92, random = Math.random } = {}) {
  if (!(temperature > 0)) throw new Error('temperature must be > 0'); if (!(topP > 0 && topP <= 1)) throw new Error('topP must be in (0, 1]'); if (!allowedIds.length) throw new Error('no allowed tokens to sample');
  const entries = allowedIds.map((id) => ({ id, value: logits[id] / temperature })); const maxValue = Math.max(...entries.map((item) => item.value)); let sum = 0;
  for (const item of entries) { item.probability = Math.exp(item.value - maxValue); sum += item.probability; } for (const item of entries) item.probability /= sum; entries.sort((a, b) => b.probability - a.probability);
  if (topP < 1) { let cumulative = 0; const kept = []; for (const item of entries) { kept.push(item); cumulative += item.probability; if (cumulative >= topP) break; } entries.splice(0, entries.length, ...kept); const keptSum = entries.reduce((value, item) => value + item.probability, 0); for (const item of entries) item.probability /= keptSum; }
  let threshold = random(); for (const item of entries) { threshold -= item.probability; if (threshold <= 0) return item.id; } return entries[entries.length - 1].id;
}

export function tokenIdsToEvents(tokenIds) {
  const events = []; let bar = -1; let i = 0;
  while (i < tokenIds.length) {
    const token = VOCAB[tokenIds[i]]; if (token === 'BAR') { bar += 1; i += 1; continue; } if (!token?.startsWith('POSITION_') || i + 3 >= tokenIds.length) { i += 1; continue; }
    const position = Number(token.slice('POSITION_'.length)); const pitchToken = VOCAB[tokenIds[i + 1]]; const durationToken = VOCAB[tokenIds[i + 2]]; const velocityToken = VOCAB[tokenIds[i + 3]];
    if (!pitchToken?.startsWith('NOTE_PITCH_') || !durationToken?.startsWith('NOTE_DURATION_') || !velocityToken?.startsWith('VELOCITY_')) { i += 1; continue; }
    events.push({ bar: Math.max(0, bar), position, pitch: Number(pitchToken.slice('NOTE_PITCH_'.length)), duration: Number(durationToken.slice('NOTE_DURATION_'.length)), velocity: Math.max(1, Math.min(127, Math.round(Number(velocityToken.slice('VELOCITY_'.length)) / 32 * 127))) }); i += 4;
  }
  return events;
}

function variableLengthQuantity(value) { const bytes = [value & 0x7f]; value >>>= 7; while (value) { bytes.unshift((value & 0x7f) | 0x80); value >>>= 7; } return bytes; }
function pushU16(array, value) { array.push((value >>> 8) & 0xff, value & 0xff); }
function pushU32(array, value) { array.push((value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff); }

export function eventsToMidiBytes(events, bpm = 84) {
  const ticksPerBeat = 480; const ticksPerPosition = 120; const scheduled = []; const tempo = Math.round(60000000 / bpm); scheduled.push({ time: 0, data: [0xff, 0x51, 0x03, (tempo >>> 16) & 0xff, (tempo >>> 8) & 0xff, tempo & 0xff] });
  for (const event of events) { const start = (event.bar * 16 + event.position) * ticksPerPosition; const end = start + Math.max(1, event.duration) * ticksPerPosition; scheduled.push({ time: start, data: [0x90, event.pitch, event.velocity] }); scheduled.push({ time: end, data: [0x80, event.pitch, 0] }); }
  scheduled.sort((a, b) => a.time - b.time || a.data[0] - b.data[0]); const track = []; let previous = 0; for (const event of scheduled) { track.push(...variableLengthQuantity(event.time - previous), ...event.data); previous = event.time; } track.push(0x00, 0xff, 0x2f, 0x00);
  const out = [...new TextEncoder().encode('MThd')]; pushU32(out, 6); pushU16(out, 0); pushU16(out, 1); pushU16(out, ticksPerBeat); out.push(...new TextEncoder().encode('MTrk')); pushU32(out, track.length); out.push(...track); return new Uint8Array(out);
}

export class OrbituneBrowserRuntime {
  constructor(ortNamespace) { if (!ortNamespace) throw new Error('ONNX Runtime Web namespace is required'); this.ort = ortNamespace; this.session = null; this.adapter = emptyAdapterInputs(); }
  async loadModel(url, { executionProviders = ['wasm'] } = {}) { this.session = await this.ort.InferenceSession.create(url, { executionProviders }); }
  loadAdapter(arrayBuffer) { this.adapter = packAdapterSafetensors(arrayBuffer); }
  clearAdapter() { this.adapter = emptyAdapterInputs(); }
  async logitsFor(tokenIds) {
    if (!this.session) throw new Error('model is not loaded');
    const context = tokenIds.slice(-ORBITUNE_V0.context); const inputIds = new BigInt64Array(context.map((id) => BigInt(id)));
    const feeds = { input_ids: new this.ort.Tensor('int64', inputIds, [1, context.length]), lora_a: new this.ort.Tensor('float32', this.adapter.loraA, [ORBITUNE_V0.layers, 2, ORBITUNE_V0.rank, ORBITUNE_V0.hidden]), lora_b: new this.ort.Tensor('float32', this.adapter.loraB, [ORBITUNE_V0.layers, 2, ORBITUNE_V0.hidden, ORBITUNE_V0.rank]), lora_scale: new this.ort.Tensor('float32', this.adapter.scale, [1]) };
    const result = await this.session.run(feeds); const logits = result.logits; const offset = (logits.dims[1] - 1) * VOCAB.length; return logits.data.slice(offset, offset + VOCAB.length);
  }
  async generate({ bars = 8, temperature = 0.85, topP = 0.92, maxNewTokens = 2048 } = {}) {
    const ids = [TOKEN_TO_ID.get('BOS')];
    for (let step = 0; step < maxNewTokens; step += 1) { const allowed = allowedNextTokenIds(ids, bars); if (!allowed.length) break; const logits = await this.logitsFor(ids); const next = sampleAllowedLogits(logits, allowed, { temperature, topP }); ids.push(next); if (VOCAB[next] === 'EOS') break; }
    if (VOCAB[ids[ids.length - 1]] !== 'EOS') throw new Error('generation hit maxNewTokens before completing the requested bars');
    return { tokenIds: ids, events: tokenIdsToEvents(ids) };
  }
}
