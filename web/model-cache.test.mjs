import assert from 'node:assert/strict';
import test from 'node:test';

import { cacheVariantArtifacts, createPersistentModelFetch, getVariantCacheStatus } from './model-cache.mjs';

class MemoryCache {
  constructor() { this.values = new Map(); }
  async match(key) { const value = this.values.get(String(key)); return value ? value.clone() : undefined; }
  async put(key, response) { this.values.set(String(key), response.clone()); }
  async delete(key) { return this.values.delete(String(key)); }
}
class MemoryCaches {
  constructor() { this.cache = new MemoryCache(); }
  async open() { return this.cache; }
}

const variant = {
  stream: { url: 'https://example.test/stream.onnx', sha256: '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824' },
  decoder: { url: 'https://example.test/decoder.onnx', sha256: '486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7' },
};

test('variant artifacts are verified, persisted, and reused offline', async () => {
  const cacheStorage = new MemoryCaches(); let calls = 0;
  const fetchImpl = async (url) => {
    calls += 1;
    if (String(url).includes('stream')) return new Response(new TextEncoder().encode('hello'));
    return new Response(new TextEncoder().encode('world'));
  };
  const status = await cacheVariantArtifacts(variant, { cacheStorage, fetchImpl });
  assert.deepEqual(status, { supported: true, stream: true, decoder: true, ready: true });
  assert.equal(calls, 2);

  const cachedFetch = createPersistentModelFetch({ cacheStorage, fetchImpl: async () => { throw new Error('network must not be used'); } });
  assert.equal(new TextDecoder().decode(await (await cachedFetch(variant.stream.url)).arrayBuffer()), 'hello');
  assert.equal(new TextDecoder().decode(await (await cachedFetch(variant.decoder.url)).arrayBuffer()), 'world');
});

test('corrupt artifacts fail before entering persistent cache', async () => {
  const cacheStorage = new MemoryCaches();
  await assert.rejects(
    () => cacheVariantArtifacts(variant, { cacheStorage, fetchImpl: async () => new Response(new TextEncoder().encode('bad')) }),
    /SHA-256 mismatch/,
  );
  assert.deepEqual(await getVariantCacheStatus(variant, { cacheStorage }), { supported: true, stream: false, decoder: false, ready: false });
});
