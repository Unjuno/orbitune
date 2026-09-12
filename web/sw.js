const CACHE_PREFIX = 'orbitune-pwa';
const SHELL_CACHE = `${CACHE_PREFIX}-shell-v1`;
const RUNTIME_CACHE = `${CACHE_PREFIX}-runtime-v1`;
const SHELL_ASSETS = [
  './compound.html',
  './compound-app.mjs',
  './compound-runtime.mjs',
  './compound-midi.mjs',
  './compound-player.mjs',
  './compound-stream.mjs',
  './compound-variant.mjs',
  './model-loader.mjs',
  './model-store.mjs',
  './compound-runtime-config.json',
  './manifest.webmanifest',
  './orbitune-icon.svg',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    await cache.addAll(SHELL_ASSETS);
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((key) => key.startsWith(CACHE_PREFIX) && ![SHELL_CACHE, RUNTIME_CACHE].includes(key)).map((key) => caches.delete(key)));
    await self.clients.claim();
  })());
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok || response.type === 'opaque') await cache.put(request, response.clone());
  return response;
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  const ownOrigin = url.origin === self.location.origin;
  const ortRuntimeAsset = url.hostname === 'cdn.jsdelivr.net' && url.pathname.includes('/onnxruntime-web@1.29.0/');
  if (ownOrigin) {
    event.respondWith(cacheFirst(request, SHELL_CACHE).catch(() => caches.match('./compound.html')));
    return;
  }
  if (ortRuntimeAsset) event.respondWith(cacheFirst(request, RUNTIME_CACHE));
});

self.addEventListener('message', (event) => {
  if (event.data === 'SKIP_WAITING') self.skipWaiting();
});
