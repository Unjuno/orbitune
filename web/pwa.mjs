export const ORT_VERSION = '1.29.0';
export const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const ORT_OFFLINE_ASSETS = Object.freeze([
  `${ORT_BASE}ort.min.js`,
  `${ORT_BASE}ort-wasm-simd-threaded.mjs`,
  `${ORT_BASE}ort-wasm-simd-threaded.wasm`,
]);
export const RUNTIME_CACHE_NAME = 'orbitune-runtime-deps-v1';

export async function registerOrbituneServiceWorker({ navigatorObject = globalThis.navigator } = {}) {
  if (!navigatorObject?.serviceWorker?.register) return { supported: false, registration: null };
  const registration = await navigatorObject.serviceWorker.register('./sw.js', { scope: './' });
  return { supported: true, registration };
}

export async function warmOfflineRuntime({ cacheStorage = globalThis.caches, fetchImpl = globalThis.fetch, onAsset = null } = {}) {
  if (!cacheStorage?.open) throw new Error('Cache Storage is unavailable');
  if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable');
  const cache = await cacheStorage.open(RUNTIME_CACHE_NAME);
  let stored = 0;
  for (const url of ORT_OFFLINE_ASSETS) {
    const existing = await cache.match(url);
    if (existing) {
      if (typeof onAsset === 'function') onAsset({ url, status: 'cached' });
      continue;
    }
    if (typeof onAsset === 'function') onAsset({ url, status: 'downloading' });
    const response = await fetchImpl(url, { cache: 'no-cache', mode: 'cors' });
    if (!response.ok) throw new Error(`ONNX Runtime offline asset failed: ${url} HTTP ${response.status}`);
    await cache.put(url, response.clone());
    stored += 1;
    if (typeof onAsset === 'function') onAsset({ url, status: 'stored' });
  }
  return { stored, total: ORT_OFFLINE_ASSETS.length };
}

export class PwaInstallController {
  constructor({ windowObject = globalThis.window } = {}) {
    this.windowObject = windowObject;
    this.promptEvent = null;
    this.listeners = new Set();
    if (windowObject?.addEventListener) {
      windowObject.addEventListener('beforeinstallprompt', (event) => {
        event.preventDefault();
        this.promptEvent = event;
        this.emit();
      });
      windowObject.addEventListener('appinstalled', () => {
        this.promptEvent = null;
        this.emit();
      });
    }
  }
  get available() { return Boolean(this.promptEvent); }
  subscribe(listener) { this.listeners.add(listener); listener(this.available); return () => this.listeners.delete(listener); }
  emit() { for (const listener of this.listeners) listener(this.available); }
  async prompt() {
    if (!this.promptEvent) throw new Error('PWA install prompt is not currently available');
    const event = this.promptEvent;
    await event.prompt();
    const choice = await event.userChoice;
    if (choice?.outcome === 'accepted') this.promptEvent = null;
    this.emit();
    return choice;
  }
}
