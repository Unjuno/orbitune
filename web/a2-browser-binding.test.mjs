import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const A2_ID = 'orbitune-a2-512-research-nc';
const A2_SHA256 = 'e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0';

test('checked-in browser runtime is bound to the immutable A2-512 Base', async () => {
  const config = JSON.parse(await readFile(new URL('./compound-runtime-config.json', import.meta.url), 'utf8'));
  assert.equal(config.model_id, A2_ID);
  assert.equal(config.checkpoint_sha256, A2_SHA256);
  assert.equal(config.publication_status, 'runtime_ready_model_unpublished');
  assert.deepEqual(config.variants, []);
});
