/** Shared types mirroring Python Pydantic models. Source of truth: backend/domain/<name>/types.py */

export interface Workspace {
  id: string
  name: string
  created_at: string
  updated_at: string
  settings: Record<string, unknown>
}

export type ProviderKind = 'ollama' | 'lmstudio'

export interface ProviderCapabilities {
  streaming: boolean
  embeddings: boolean
  max_context_tokens: number
  supports_system_prompt: boolean
}

export interface ProviderConfig {
  id: string
  kind: ProviderKind
  display_name: string
  base_url: string
  model_id: string
  capabilities: ProviderCapabilities
  extra: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface CreateProviderRequest {
  kind: ProviderKind
  display_name: string
  base_url: string
  model_id: string
  capabilities?: Partial<ProviderCapabilities>
}

export interface UpdateProviderRequest {
  display_name?: string
  base_url?: string
  model_id?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Git clone
// ──────────────────────────────────────────────────────────────────────────────

export interface CloneFromUrlRequest {
  url: string
  dest_path: string
}
