import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const manifestUrl = new URL('./manifest.webmanifest', import.meta.url);
const htmlUrl = new URL('./compound.html', import.meta.url);
const appUrl = new URL('./compound-app.mjs', import.meta.url);
const swUrl = new URL('./sw.js', import.meta.url);

async function text(url) { return readFile(url, 'utf8'); }

test('Compound page declares an installable standalone PWA manifest', async () => {
  const manifest = JSON.parse(await text(manifestUrl));
  assert.equal(manifest.display, 'standalone');
  assert.equal(manifest.start_url, './compound.html');
  assert.equal(manifest.scope, './');
  assert(manifest.icons.some((icon) => icon.src === './orbitune-icon.svg'));
  assert.match(await text(htmlUrl), /rel="manifest" href="\.\/manifest\.webmanifest"/);
});

test('Compound app registers the scoped service worker', async () => {
  const app = await text(appUrl);
  assert.match(app, /serviceWorker\.register\('\.\/sw\.js', \{ scope: '\.\/' \}\)/);
});

test('PWA shell precaches code but never embeds model artifact URLs', async () => {
  const sw = await text(swUrl);
  for (const asset of ['compound.html', 'compound-app.mjs', 'compound-stream.mjs', 'model-loader.mjs', 'model-store.mjs']) {
    assert(sw.includes(`./${asset}`), `${asset} must be part of the shell cache`);
  }
  assert(!sw.includes('huggingface.co/'), 'large model artifacts must stay outside the shell cache');
});
