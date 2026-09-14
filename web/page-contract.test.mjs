import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (name) => fs.readFileSync(new URL(name, import.meta.url), 'utf8');

function assertIds(html, ids) {
  for (const id of ids) {
    assert.match(html, new RegExp(`id=["']${id}["']`), `missing required DOM id: ${id}`);
  }
}

const compoundIds = [
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
];

function assertCompoundAppPage(html) {
  assert.match(html, /rel=["']manifest["']\s+href=["']\.\/manifest\.webmanifest["']/);
  assert.match(html, /href=["']\.\/orbitune-ui\.css["']/);
  assert.match(html, /onnxruntime-web@1\.29\.0\/dist\/ort\.min\.js/);
  assert.match(html, /src=["']\.\/compound-app\.mjs["']/);
  assert.match(html, /GeneralUser GS 2\.0\.3/);
  assert.match(html, /SpessaSynth 4\.3\.14/);
  assertIds(html, compoundIds);
  assert.match(html, /id=["']compound-status["'][^>]*aria-live=["']polite["']/);
}

test('root Pages entry point runs the latest published A2-512 Compound app, not the legacy runtime', () => {
  const html = read('./index.html');
  assertCompoundAppPage(html);
  assert.match(html, /A2-512/);
  assert.doesNotMatch(html, /src=["']\.\/app\.mjs["']/);
  assert.doesNotMatch(html, /Theory-REMI/);
});

test('compatibility Compound URL preserves the same sampled application contract', () => {
  const html = read('./compound.html');
  assertCompoundAppPage(html);
});

test('PWA installs and starts at the root A2 app', () => {
  const manifest = JSON.parse(read('./manifest.webmanifest'));
  assert.equal(manifest.id, './');
  assert.equal(manifest.start_url, './');
  assert.match(manifest.description, /A2-512/);
});

test('PWA shell revisions and packages the sampled playback engine without shell-caching the SF2', () => {
  const serviceWorker = read('./sw.js');
  assert.match(serviceWorker, /orbitune-shell-v5/);
  for (const asset of [
    './',
    './orbitune-ui.css',
    './index.html',
    './compound.html',
    './compound-soundfont-player.mjs',
    './vendor/spessasynth-bundle.mjs',
    './vendor/spessasynth_processor.min.js',
    './soundfont-release.json',
    './third_party/GeneralUser-GS-LICENSE.txt',
  ]) assert.ok(serviceWorker.includes(`'${asset}'`), `${asset} missing from service-worker shell`);
  assert.match(serviceWorker, /url\.pathname\.includes\('\/soundfonts\/'\)/);
  assert.doesNotMatch(serviceWorker, /'\.\/soundfonts\/GeneralUser-GS\.sf2'/);
});
