import test from 'node:test';
import assert from 'node:assert/strict';

import { createVerifiedModelSession, sha256Hex } from './model-loader.mjs';

const MODEL_BYTES = new TextEncoder().encode('orbitune-model').buffer;
const MODEL_SHA256 = 'e489c189f402f5bf9deadf97bb6346e4c75dc95a853eddcd1fbc77eb2f0a2ee4';

function fakeOrt(calls) {
  return {
    InferenceSession: {
      async create(source, options) {
        calls.push({ source, options });
        return { source, options };
      },
    },
  };
}

test('sha256Hex uses Web Crypto SHA-256', async () => {
  assert.equal(await sha256Hex(MODEL_BYTES), MODEL_SHA256);
});

test('verified model loader hashes bytes before creating a session', async () => {
  const calls = [];
  const fetchCalls = [];
  const session = await createVerifiedModelSession(fakeOrt(calls), 'https://example.test/model.onnx', {
    expectedSha256: MODEL_SHA256,
    executionProviders: ['wasm'],
    fetchImpl: async (url, options) => {
      fetchCalls.push({ url, options });
      return { ok: true, status: 200, arrayBuffer: async () => MODEL_BYTES };
    },
  });
  assert.equal(fetchCalls.length, 1);
  assert.equal(calls.length, 1);
  assert(calls[0].source instanceof ArrayBuffer);
  assert.deepEqual(calls[0].options.executionProviders, ['wasm']);
  assert(session.source instanceof ArrayBuffer);
  assert.deepEqual(session.options, calls[0].options);
});

test('verified model loader rejects a hash mismatch before ONNX Runtime sees bytes', async () => {
  const calls = [];
  await assert.rejects(
    createVerifiedModelSession(fakeOrt(calls), 'https://example.test/model.onnx', {
      expectedSha256: '0'.repeat(64),
      fetchImpl: async () => ({ ok: true, status: 200, arrayBuffer: async () => MODEL_BYTES }),
    }),
    /SHA-256 mismatch/,
  );
  assert.equal(calls.length, 0);
});

test('model loader preserves direct URL loading when no hash is configured', async () => {
  const calls = [];
  await createVerifiedModelSession(fakeOrt(calls), 'https://example.test/model.onnx');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].source, 'https://example.test/model.onnx');
});
