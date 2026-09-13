import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (name) => fs.readFileSync(new URL(name, import.meta.url), 'utf8');

function assertIds(html, ids) {
  for (const id of ids) {
    assert.match(html, new RegExp(`id=["']${id}["']`), `missing required DOM id: ${id}`);
  }
}

test('landing page preserves the legacy runtime DOM contract and shared product stylesheet', () => {
  const html = read('./index.html');
  assert.match(html, /href=["']\.\/orbitune-ui\.css["']/);
  assertIds(html, [
    'base',
    'base-meta',
    'adapter',
    'adapter-meta',
    'bpm',
    'bars',
    'temperature',
    'temperature-value',
    'generate',
    'download',
    'status',
  ]);
  assert.match(html, /href=["']\.\/compound\.html["']/);
});

test('Infinite MIDI page preserves the Compound application DOM contract and accessibility hooks', () => {
  const html = read('./compound.html');
  assert.match(html, /rel=["']manifest["']\s+href=["']\.\/manifest\.webmanifest["']/);
  assert.match(html, /href=["']\.\/orbitune-ui\.css["']/);
  assertIds(html, [
    'compound-variant',
    'compound-model-meta',
    'compound-events',
    'compound-temperature',
    'compound-temperature-value',
    'compound-top-p',
    'compound-top-p-value',
    'compound-generate',
    'compound-play',
    'compound-stop',
    'compound-download',
    'compound-live-start',
    'compound-live-pause',
    'compound-live-stop',
    'compound-offline',
    'compound-install',
    'compound-storage-status',
    'compound-status',
  ]);
  assert.match(html, /id=["']compound-status["'][^>]*aria-live=["']polite["']/);
});

test('PWA shell revisions and caches the shared UI asset', () => {
  const serviceWorker = read('./sw.js');
  assert.match(serviceWorker, /orbitune-shell-v2/);
  assert.match(serviceWorker, /'\.\/orbitune-ui\.css'/);
  assert.match(serviceWorker, /'\.\/index\.html'/);
  assert.match(serviceWorker, /'\.\/compound\.html'/);
});
