function normalizeSha256(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(normalized)) throw new Error('expected SHA-256 must be 64 lowercase/uppercase hex characters');
  return normalized;
}

export async function sha256Hex(arrayBuffer) {
  if (!globalThis.crypto?.subtle) throw new Error('Web Crypto SHA-256 is unavailable in this browser');
  const digest = await globalThis.crypto.subtle.digest('SHA-256', arrayBuffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function createVerifiedModelSession(
  ortNamespace,
  url,
  {
    expectedSha256 = '',
    executionProviders = ['wasm'],
    fetchImpl = globalThis.fetch,
  } = {},
) {
  if (!ortNamespace?.InferenceSession?.create) throw new Error('ONNX Runtime Web namespace is invalid');
  if (!url) throw new Error('model URL is required');
  if (!expectedSha256) {
    return ortNamespace.InferenceSession.create(url, { executionProviders });
  }
  if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable for verified model loading');

  const expected = normalizeSha256(expectedSha256);
  const response = await fetchImpl(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Base model download failed: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  const actual = await sha256Hex(bytes);
  if (actual !== expected) throw new Error(`Base model SHA-256 mismatch: ${actual} != ${expected}`);
  return ortNamespace.InferenceSession.create(bytes, { executionProviders });
}
