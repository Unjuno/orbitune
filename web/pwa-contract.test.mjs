import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { ORT_OFFLINE_ASSETS, ORT_VERSION } from './pwa.mjs';

test('PWA manifest launches the Compound streaming application', async () => {
  const manifest = JSON.parse(await readFile(new URL('./manifest.webmanifest', import.meta.url), 'utf8'));
  assert.equal(manifest.start_url, './compound.html');
  assert.equal(manifest.display, 'standalone');
  assert.ok(manifest.icons.some((icon) => icon.src === './orbitune-192.png' && icon.sizes === '192x192'));
  assert.ok(manifest.icons.some((icon) => icon.src === './orbitune-512.png' && icon.sizes === '512x512'));
});

test('service worker precaches the streaming shell', async () => {
  const source = await readFile(new URL('./sw.js', import.meta.url), 'utf8');
  for (const asset of ['./compound.html', './compound-app.mjs', './compound-stream.mjs', './compound-live-player.mjs', './model-cache.mjs', './orbitune-192.png', './orbitune-512.png']) {
    assert.ok(source.includes(`'${asset}'`), `${asset} missing from service-worker shell`);
  }
});

test('offline runtime assets are pinned to the same ORT version as the page', async () => {
  const html = await readFile(new URL('./compound.html', import.meta.url), 'utf8');
  assert.equal(ORT_VERSION, '1.29.0');
  assert.ok(html.includes(`onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`));
  assert.ok(ORT_OFFLINE_ASSETS.every((url) => url.includes(`onnxruntime-web@${ORT_VERSION}/dist/`)));
});
