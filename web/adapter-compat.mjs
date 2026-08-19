import { parseSafetensors } from './orbitune-runtime.mjs';

function isSha256(value) {
  return typeof value === 'string' && /^[0-9a-f]{64}$/i.test(value);
}

export function assertAdapterBaseSha256(arrayBuffer, expectedBaseSha256) {
  if (!isSha256(expectedBaseSha256)) {
    throw new Error('runtime Base SHA-256 is not configured');
  }
  const { metadata } = parseSafetensors(arrayBuffer);
  const actual = String(metadata.base_sha256 || '').toLowerCase();
  if (!isSha256(actual)) {
    throw new Error('adapter is missing a valid base_sha256');
  }
  if (actual !== expectedBaseSha256.toLowerCase()) {
    throw new Error('adapter was trained for a different Base checkpoint');
  }
  return actual;
}
