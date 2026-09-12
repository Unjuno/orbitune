import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import process from 'node:process';

import * as ort from 'onnxruntime-web';
import { CompoundBrowserRuntime, validateCompoundRecord } from '../web/compound-runtime.mjs';

const [streamPath, decoderPath, goldenPath] = process.argv.slice(2);
if (!streamPath || !decoderPath || !goldenPath) {
  throw new Error('usage: node scripts/validate_compound_web_wasm.mjs STREAM_ONNX DECODER_ONNX GOLDEN_JSON');
}

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const golden = JSON.parse(await readFile(goldenPath, 'utf8'));
assert.equal(golden.runtime_abi, 'native-stream-state+decoder-prefix-v2');
assert.equal(golden.temperature, 0);
assert.equal(golden.top_p, 1);
assert.ok(Array.isArray(golden.records) && golden.records.length === golden.new_events + 1);
for (const record of golden.records) validateCompoundRecord(record);

const runtime = new CompoundBrowserRuntime(ort);
const started = performance.now();
runtime.streamSession = await ort.InferenceSession.create(new Uint8Array(await readFile(streamPath)), {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});
runtime.decoderSession = await ort.InferenceSession.create(new Uint8Array(await readFile(decoderPath)), {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});
const loadMs = performance.now() - started;

const generationStarted = performance.now();
const result = await runtime.generate({
  primerRecords: [golden.records[0]],
  maxNewEvents: golden.new_events,
  temperature: 0,
  topP: 1,
});
const generationMs = performance.now() - generationStarted;
assert.deepEqual(result.records, golden.records, 'ONNX Runtime Web/WASM greedy records diverge from native golden');
const canonical = JSON.stringify(result.records);
const recordsSha256 = createHash('sha256').update(canonical).digest('hex');
assert.equal(recordsSha256, golden.records_sha256);

console.log(JSON.stringify({
  status: 'PASS',
  runtime: 'onnxruntime-web/wasm',
  records: result.records.length,
  new_events: golden.new_events,
  records_sha256: recordsSha256,
  load_ms: Math.round(loadMs),
  generation_ms: Math.round(generationMs),
  events_per_second: Number((golden.new_events / Math.max(0.001, generationMs / 1000)).toFixed(3)),
}, null, 2));
