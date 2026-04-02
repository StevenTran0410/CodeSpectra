import type { ProviderKind } from './electron'

/** Provider kinds that require an API key and send data to external servers. */
export const CLOUD_KINDS: ReadonlySet<ProviderKind> = new Set([
  'openai',
  'anthropic',
  'gemini',
  'deepseek',
])
