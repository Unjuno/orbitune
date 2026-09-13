import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const index = fs.readFileSync(new URL('./index.html', import.meta.url), 'utf8');

test('root entrypoint is the Compound A2 application', () => {
  assert.match(index, /A2-512/);
  assert.match(index, /src=["']\.\/compound-app\.mjs["']/);
  assert.match(index, /onnxruntime-web@1\.29\.0\/dist\/ort\.min\.js/);
  assert.doesNotMatch(index, /src=["']\.\/app\.mjs["']/);
  assert.doesNotMatch(index, /Theory-REMI/);
});
