import test from 'node:test';
import assert from 'node:assert/strict';
import { availableCompoundVariants, validateCompoundRuntimeConfig } from './compound-variant.mjs';

const BASE_SHA = 'e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0';
const STREAM_SHA = '27be3d6a4db726f52d7fc7e8df2e7a24be04ab5bd76da243bd8dd92712f2e107';
const DECODER_SHA = 'abb8326680222aff32d6b2fcb45356c748c523216cb0d75d6a755e615f419225';

function config(overrides = {}) {
  return {
    schema_version: '0.2.0', runtime_abi: 'native-stream-state+decoder-prefix-v2',
    publication_status: 'runtime_ready_model_unpublished', model_id: 'orbitune-a2-512-research-nc',
    checkpoint_sha256: BASE_SHA, architecture: 'orbitune-compound-hierarchical-gpt-v1', tokenizer: 'orbitune-compound-v0-experimental',
    commercial_eligible: false, distribution_scope: 'research-noncommercial', license_policy: 'research-nc', redistribution_review: 'pending', variants: [],
    ...overrides,
  };
}

function baseVariant(overrides = {}) {
  return {
    id: 'orbitune-a2-512-web-v1', display_name: 'Orbitune A2-512 Base', kind: 'base', available: true,
    architecture: 'orbitune-compound-hierarchical-gpt-v1', tokenizer: 'orbitune-compound-v0-experimental',
    runtime_abi: 'native-stream-state+decoder-prefix-v2', base_checkpoint_sha256: BASE_SHA,
    stream: { url: './models/a2-v1/stream.onnx', sha256: STREAM_SHA },
    decoder: { url: './models/a2-v1/decoder_prefix.onnx', sha256: DECODER_SHA }, execution_providers: ['wasm'],
    ...overrides,
  };
}

test('current unpublished A2 config validates with zero available variants', () => {
  assert.equal(validateCompoundRuntimeConfig(config()), true);
  assert.deepEqual(availableCompoundVariants(config()), []);
});

test('legacy generic noncommercial scope is rejected in favor of canonical A2 scope', () => {
  assert.throws(() => validateCompoundRuntimeConfig(config({ distribution_scope: 'noncommercial' })), /research-NC\/noncommercial lineage/);
});

test('an available Base variant requires both completed review and published status', () => {
  const variant = baseVariant();
  assert.throws(() => validateCompoundRuntimeConfig(config({ variants: [variant] })), /completed redistribution review/);
  assert.throws(() => validateCompoundRuntimeConfig(config({ redistribution_review: 'completed', variants: [variant] })), /runtime_model_published/);
  const published = config({ redistribution_review: 'completed', publication_status: 'runtime_model_published', variants: [variant] });
  assert.equal(validateCompoundRuntimeConfig(published), true);
  assert.equal(availableCompoundVariants(published).length, 1);
});

test('published status cannot be set while no model variant is available', () => {
  assert.throws(() => validateCompoundRuntimeConfig(config({ redistribution_review: 'completed', publication_status: 'runtime_model_published' })), /at least one available variant/);
});

test('variant is bound to exact Compound architecture tokenizer runtime and Base SHA', () => {
  const published = (variant) => config({ redistribution_review: 'completed', publication_status: 'runtime_model_published', variants: [variant] });
  assert.throws(() => validateCompoundRuntimeConfig(published(baseVariant({ architecture: 'other' }))), /architecture/);
  assert.throws(() => validateCompoundRuntimeConfig(published(baseVariant({ tokenizer: 'other' }))), /tokenizer/);
  assert.throws(() => validateCompoundRuntimeConfig(published(baseVariant({ runtime_abi: 'other' }))), /runtime_abi/);
  assert.throws(() => validateCompoundRuntimeConfig(published(baseVariant({ base_checkpoint_sha256: 'a'.repeat(64) }))), /different Base checkpoint/);
});

test('variant artifact URLs reject executable, insecure, and protocol-relative schemes', () => {
  const published = (url) => config({ redistribution_review: 'completed', publication_status: 'runtime_model_published', variants: [baseVariant({ stream: { url, sha256: STREAM_SHA } })] });
  for (const url of ['javascript:alert(1)', 'data:application/octet-stream,x', 'blob:https://example.invalid/id', 'http://example.invalid/model.onnx', '//example.invalid/model.onnx']) {
    assert.throws(() => validateCompoundRuntimeConfig(published(url)));
  }
});

test('runtime currently admits WASM only', () => {
  const published = config({ redistribution_review: 'completed', publication_status: 'runtime_model_published', variants: [baseVariant({ execution_providers: ['webgpu'] })] });
  assert.throws(() => validateCompoundRuntimeConfig(published), /\["wasm"\]/);
});

test('pre-merged LoRA variant requires an explicit adapter identity', () => {
  const lora = baseVariant({ id: 'demo-lora', kind: 'lora-premerged' });
  const published = (variant) => config({ redistribution_review: 'completed', publication_status: 'runtime_model_published', variants: [variant] });
  assert.throws(() => validateCompoundRuntimeConfig(published(lora)), /requires adapter_id/);
  assert.equal(validateCompoundRuntimeConfig(published({ ...lora, adapter_id: 'demo-style-v1' })), true);
});

test('duplicate variant IDs are rejected', () => {
  const first = baseVariant(); const second = baseVariant({ stream: { url: './other-stream.onnx', sha256: STREAM_SHA } });
  const published = config({ redistribution_review: 'completed', publication_status: 'runtime_model_published', variants: [first, second] });
  assert.throws(() => validateCompoundRuntimeConfig(published), /duplicate Compound variant id/);
});
