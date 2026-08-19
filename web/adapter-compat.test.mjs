import test from 'node:test';
import assert from 'node:assert/strict';

import { assertAdapterBaseSha256 } from './adapter-compat.mjs';

function makeSafetensorsMetadata(metadata) {
  const header = JSON.stringify({ __metadata__: metadata });
  const headerBytes = new TextEncoder().encode(header);
  const buffer = new ArrayBuffer(8 + headerBytes.length);
  const view = new DataView(buffer);
  view.setBigUint64(0, BigInt(headerBytes.length), true);
  new Uint8Array(buffer, 8).set(headerBytes);
  return buffer;
}

test('browser accepts adapter bound to the exact Base checkpoint', () => {
  const sha = 'a'.repeat(64);
  const bytes = makeSafetensorsMetadata({ base_sha256: sha });
  assert.equal(assertAdapterBaseSha256(bytes, sha), sha);
});

test('browser rejects adapter bound to another Base checkpoint', () => {
  const bytes = makeSafetensorsMetadata({ base_sha256: 'a'.repeat(64) });
  assert.throws(() => assertAdapterBaseSha256(bytes, 'b'.repeat(64)), /different Base checkpoint/);
});
