export const DECODING_PRESETS = Object.freeze({
  balanced: Object.freeze({
    label: 'Balanced nucleus',
    strategy: 'nucleus',
    temperature: 0.85,
    instrumentTemperature: 1.05,
    topP: 0.92,
    topK: 0,
    minP: 0,
  }),
  conservative: Object.freeze({
    label: 'Stable notes / open instruments',
    strategy: 'hybrid',
    temperature: 0.58,
    instrumentTemperature: 1.05,
    topP: 0.92,
    topK: 24,
    minP: 0.02,
  }),
  exploratory: Object.freeze({
    label: 'Exploratory',
    strategy: 'hybrid',
    temperature: 0.98,
    instrumentTemperature: 1.18,
    topP: 0.96,
    topK: 48,
    minP: 0.01,
  }),
  topk: Object.freeze({
    label: 'Top-k',
    strategy: 'top-k',
    temperature: 0.85,
    instrumentTemperature: 1.05,
    topP: 1,
    topK: 32,
    minP: 0,
  }),
  minp: Object.freeze({
    label: 'Min-p',
    strategy: 'min-p',
    temperature: 0.85,
    instrumentTemperature: 1.05,
    topP: 1,
    topK: 0,
    minP: 0.04,
  }),
});

export function decodingPreset(name = 'balanced') {
  const preset = DECODING_PRESETS[name];
  if (!preset) throw new Error(`unknown decoding preset: ${name}`);
  return { ...preset };
}
