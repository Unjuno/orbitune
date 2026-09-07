import { COMPOUND_ARCHITECTURE, COMPOUND_CONTEXT_ABI, COMPOUND_TOKENIZER } from './compound-runtime.mjs';

const SHA256_RE = /^[0-9a-f]{64}$/i;
const ID_RE = /^[a-z0-9][a-z0-9-]*$/;
const ALLOWED_KINDS = new Set(['base', 'lora-premerged']);
const ALLOWED_EXECUTION_PROVIDERS = new Set(['wasm']);

function requireSha256(value, label) {
  if (typeof value !== 'string' || !SHA256_RE.test(value)) throw new Error(`${label} must be a 64-character SHA-256`);
  return value.toLowerCase();
}

function requireArtifact(spec, label) {
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) throw new Error(`${label} must be an object`);
  const keys = Object.keys(spec).sort();
  if (JSON.stringify(keys) !== JSON.stringify(['sha256', 'url'])) throw new Error(`${label} must contain exactly url and sha256`);
  if (typeof spec.url !== 'string' || !spec.url.trim()) throw new Error(`${label}.url is required`);
  const url = spec.url.trim();
  if (/^(?:javascript|data|blob):/i.test(url)) throw new Error(`${label}.url uses a forbidden scheme`);
  if (/^[a-z][a-z0-9+.-]*:/i.test(url) && !/^https:/i.test(url)) throw new Error(`${label}.url must use HTTPS or a same-site relative URL`);
  if (url.startsWith('//')) throw new Error(`${label}.url must not be protocol-relative`);
  return { url, sha256: requireSha256(spec.sha256, `${label}.sha256`) };
}

export function validateCompoundRuntimeConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) throw new Error('Compound runtime config must be an object');
  if (config.runtime_abi !== COMPOUND_CONTEXT_ABI) throw new Error(`Compound runtime ABI mismatch: ${config.runtime_abi}`);
  if (config.architecture !== COMPOUND_ARCHITECTURE) throw new Error(`Compound architecture mismatch: ${config.architecture}`);
  if (config.tokenizer !== COMPOUND_TOKENIZER) throw new Error(`Compound tokenizer mismatch: ${config.tokenizer}`);
  const checkpointSha = requireSha256(config.checkpoint_sha256, 'checkpoint_sha256');
  if (config.commercial_eligible !== false || config.distribution_scope !== 'noncommercial' || config.license_policy !== 'research-nc') {
    throw new Error('Compound runtime config must preserve research-NC/noncommercial lineage');
  }
  if (!Array.isArray(config.variants)) throw new Error('Compound runtime config variants must be an array');
  const seen = new Set();
  for (const variant of config.variants) {
    validateCompoundVariant(variant, { config, checkpointSha });
    if (seen.has(variant.id)) throw new Error(`duplicate Compound variant id: ${variant.id}`);
    seen.add(variant.id);
  }
  return true;
}

export function validateCompoundVariant(variant, { config, checkpointSha = null } = {}) {
  if (!variant || typeof variant !== 'object' || Array.isArray(variant)) throw new Error('Compound variant must be an object');
  if (typeof variant.id !== 'string' || !ID_RE.test(variant.id)) throw new Error('Compound variant id must match ^[a-z0-9][a-z0-9-]*$');
  if (variant.kind && !ALLOWED_KINDS.has(variant.kind)) throw new Error(`Compound variant ${variant.id} has unsupported kind ${variant.kind}`);
  if (variant.available !== true && variant.available !== false) throw new Error(`Compound variant ${variant.id} available must be boolean`);
  if (!config) throw new Error('Compound variant validation requires the runtime config');
  if (variant.architecture !== config.architecture) throw new Error(`Compound variant ${variant.id} architecture does not match runtime config`);
  if (variant.tokenizer !== config.tokenizer) throw new Error(`Compound variant ${variant.id} tokenizer does not match runtime config`);
  if (variant.runtime_abi !== config.runtime_abi) throw new Error(`Compound variant ${variant.id} runtime_abi does not match runtime config`);
  const expectedBaseSha = checkpointSha || requireSha256(config.checkpoint_sha256, 'checkpoint_sha256');
  if (requireSha256(variant.base_checkpoint_sha256, `Compound variant ${variant.id} base_checkpoint_sha256`) !== expectedBaseSha) {
    throw new Error(`Compound variant ${variant.id} is bound to a different Base checkpoint`);
  }
  if (variant.kind === 'lora-premerged' && (typeof variant.adapter_id !== 'string' || !variant.adapter_id.trim())) {
    throw new Error(`Compound pre-merged LoRA variant ${variant.id} requires adapter_id`);
  }
  if (variant.kind === 'base' && variant.adapter_id != null) throw new Error(`Compound Base variant ${variant.id} must not declare adapter_id`);
  const providers = variant.execution_providers ?? ['wasm'];
  if (!Array.isArray(providers) || !providers.length || providers.some((provider) => !ALLOWED_EXECUTION_PROVIDERS.has(provider))) {
    throw new Error(`Compound variant ${variant.id} execution_providers must currently be ["wasm"]`);
  }
  if (variant.available) {
    requireArtifact(variant.stream, `Compound variant ${variant.id} stream`);
    requireArtifact(variant.decoder, `Compound variant ${variant.id} decoder`);
  } else {
    for (const [name, spec] of [['stream', variant.stream], ['decoder', variant.decoder]]) {
      if (spec != null) requireArtifact(spec, `Compound variant ${variant.id} ${name}`);
    }
  }
  return true;
}

export function availableCompoundVariants(config) {
  validateCompoundRuntimeConfig(config);
  return config.variants.filter((variant) => variant.available === true);
}
