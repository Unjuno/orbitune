const SHELL_CACHE = 'orbitune-shell-v1';
const RUNTIME_CACHE = 'orbitune-runtime-deps-v1';
const SHELL_ASSETS = [
  './index.html',
  './compound.html',
  './legacy.html',
  './compound-app.mjs',
  './compound-runtime.mjs',
  './compound-stream.mjs',
  './compound-midi.mjs',
  './compound-player.mjs',
  './compound-live-player.mjs',
  './compound-variant.mjs',
  './model-loader.mjs',
  './model-cache.mjs',
  './pwa.mjs',
  './compound-runtime-config.json',
  './manifest.webmanifest',
  './orbitune.svg',
  './orbitune-192.png',
  './orbitune-512.png',
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
    const names = await caches.keys();
    await Promise.all(names.filter((name) => name.startsWith('orbitune-shell-') && name !== SHELL_CACHE).map((name) => caches.delete(name)));
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

async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) return cached;
    throw error;
  }
}

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin === self.location.origin) {
    // Model bytes are intentionally not placed in the shell cache. The app
    // verifies their declared SHA-256 first and persists reviewed bytes in the
    // dedicated model cache only after explicit user action.
    if (url.pathname.includes('/models/')) return;
    const dynamic = event.request.mode === 'navigate' || url.pathname.endsWith('/compound-runtime-config.json');
    event.respondWith(dynamic ? networkFirst(event.request, SHELL_CACHE) : cacheFirst(event.request, SHELL_CACHE));
    return;
  }
  if (url.hostname === 'cdn.jsdelivr.net' && url.pathname.includes('/onnxruntime-web@1.29.0/')) {
    event.respondWith(cacheFirst(event.request, RUNTIME_CACHE));
  }
});
