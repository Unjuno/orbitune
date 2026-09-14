import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { ORT_OFFLINE_ASSETS, ORT_VERSION } from './pwa.mjs';

test('PWA manifest launches the root Compound A2 streaming application', async () => {
  const manifest = JSON.parse(await readFile(new URL('./manifest.webmanifest', import.meta.url), 'utf8'));
  assert.equal(manifest.id, './');
  assert.equal(manifest.start_url, './');
  assert.equal(manifest.display, 'standalone');
  assert.ok(manifest.description.includes('A2-512'));
  assert.ok(manifest.icons.some((icon) => icon.src === './orbitune-192.png' && icon.sizes === '192x192'));
  assert.ok(manifest.icons.some((icon) => icon.src === './orbitune-512.png' && icon.sizes === '512x512'));
});

test('service worker precaches decoding, event monitor and sampled playback shell', async () => {
  const source = await readFile(new URL('./sw.js', import.meta.url), 'utf8');
  for (const asset of [
    './',
    './index.html',
    './compound.html',
    './compound-app.mjs',
    './compound-decoding-runtime.mjs',
    './decoding-presets.mjs',
    './compound-event-monitor.mjs',
    './event-monitor.css',
    './compound-stream.mjs',
    './compound-soundfont-player.mjs',
    './vendor/spessasynth-bundle.mjs',
    './vendor/spessasynth_processor.min.js',
    './soundfont-release.json',
    './model-cache.mjs',
    './orbitune-192.png',
    './orbitune-512.png',
  ]) assert.ok(source.includes(`'${asset}'`), `${asset} missing from service-worker shell`);
  assert.ok(source.includes("url.pathname.includes('/soundfonts/')"), 'SoundFont bytes must bypass the shell cache');
});

test('offline runtime assets are pinned to the same latest stable ORT Web version as both public entry points', async () => {
  const rootHtml = await readFile(new URL('./index.html', import.meta.url), 'utf8');
  const compatibilityHtml = await readFile(new URL('./compound.html', import.meta.url), 'utf8');
  assert.equal(ORT_VERSION, '1.29.0');
  assert.ok(rootHtml.includes(`onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`));
  assert.ok(compatibilityHtml.includes(`onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`));
  assert.ok(ORT_OFFLINE_ASSETS.every((url) => url.includes(`onnxruntime-web@${ORT_VERSION}/dist/`)));
});

test('Pages workflow pins sampled playback dependencies and verifies SF2 against release metadata', async () => {
  const workflow = await readFile(new URL('../.github/workflows/pages.yml', import.meta.url), 'utf8');
  assert.ok(workflow.includes('spessasynth_lib@4.3.14'));
  assert.ok(workflow.includes('spessasynth_core@4.3.22'));
  assert.ok(workflow.includes('esbuild@0.28.2'));
  assert.ok(workflow.includes("JSON.parse(fs.readFileSync('web/soundfont-release.json','utf8'))"));
  assert.ok(workflow.includes('SF_URL='));
  assert.ok(workflow.includes('SF_BYTES='));
  assert.ok(workflow.includes('SF_SHA256='));
  assert.ok(workflow.includes('684543d5e5efaef08d02be50dcda8d552478fa60'));
  assert.ok(workflow.includes('ACTUAL_SF_BYTES='));
  assert.ok(workflow.includes('ACTUAL_SF_SHA256='));
  assert.ok(workflow.includes('test "$ACTUAL_SF_SHA256" = "$SF_SHA256"'));
});
