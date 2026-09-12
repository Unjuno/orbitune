import { sha256Hex } from './model-loader.mjs';

export const MODEL_CACHE_NAME = 'orbitune-compound-models-v1';

function requireArtifact(spec, label) {
  if (!spec?.url || !spec?.sha256) throw new Error(`${label} artifact URL/SHA-256 is required`);
  return { url: String(spec.url), sha256: String(spec.sha256).toLowerCase() };
}

function cacheStorageOrNull(cacheStorage) {
  return cacheStorage && typeof cacheStorage.open === 'function' ? cacheStorage : null;
}

async function getCache(cacheStorage, cacheName) {
  const storage = cacheStorageOrNull(cacheStorage);
  return storage ? storage.open(cacheName) : null;
}

async function verifiedResponse(url, expectedSha256, fetchImpl) {
  const response = await fetchImpl(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  const actual = await sha256Hex(bytes);
  if (actual !== expectedSha256) throw new Error(`model artifact SHA-256 mismatch: ${actual} != ${expectedSha256}`);
  return new Response(bytes, {
    status: 200,
    headers: { 'content-type': 'application/octet-stream', 'x-orbitune-sha256': actual },
  });
}

export async function cacheVariantArtifacts(
  variant,
  { cacheStorage = globalThis.caches, fetchImpl = globalThis.fetch, cacheName = MODEL_CACHE_NAME, onArtifact = null } = {},
) {
  if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable for model caching');
  const cache = await getCache(cacheStorage, cacheName);
  if (!cache) throw new Error('Cache Storage is unavailable; persistent offline model storage is not supported');
  const artifacts = [
    ['stream', requireArtifact(variant.stream, 'stream')],
    ['decoder', requireArtifact(variant.decoder, 'decoder')],
  ];
  for (const [name, spec] of artifacts) {
    const cached = await cache.match(spec.url);
    if (cached) {
      const bytes = await cached.arrayBuffer();
      if (await sha256Hex(bytes) === spec.sha256) {
        if (typeof onArtifact === 'function') onArtifact({ name, status: 'cached' });
        continue;
      }
      await cache.delete(spec.url);
    }
    if (typeof onArtifact === 'function') onArtifact({ name, status: 'downloading' });
    const response = await verifiedResponse(spec.url, spec.sha256, fetchImpl);
    await cache.put(spec.url, response);
    if (typeof onArtifact === 'function') onArtifact({ name, status: 'stored' });
  }
  return getVariantCacheStatus(variant, { cacheStorage, cacheName });
}

export async function getVariantCacheStatus(variant, { cacheStorage = globalThis.caches, cacheName = MODEL_CACHE_NAME } = {}) {
  const cache = await getCache(cacheStorage, cacheName);
  if (!cache) return { supported: false, stream: false, decoder: false, ready: false };
  const stream = variant?.stream?.url ? Boolean(await cache.match(variant.stream.url)) : false;
  const decoder = variant?.decoder?.url ? Boolean(await cache.match(variant.decoder.url)) : false;
  return { supported: true, stream, decoder, ready: stream && decoder };
}

export async function removeVariantArtifacts(variant, { cacheStorage = globalThis.caches, cacheName = MODEL_CACHE_NAME } = {}) {
  const cache = await getCache(cacheStorage, cacheName);
  if (!cache) return false;
  const removed = [];
  for (const spec of [variant?.stream, variant?.decoder]) if (spec?.url) removed.push(await cache.delete(spec.url));
  return removed.some(Boolean);
}

export function createPersistentModelFetch({ cacheStorage = globalThis.caches, fetchImpl = globalThis.fetch, cacheName = MODEL_CACHE_NAME } = {}) {
  return async (url, init = {}) => {
    const cache = await getCache(cacheStorage, cacheName);
    const cached = cache ? await cache.match(url) : null;
    if (cached) return cached.clone();
    if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable and model artifact is not cached');
    return fetchImpl(url, init);
  };
}
