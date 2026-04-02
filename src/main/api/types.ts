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
// GitHub integration
// ──────────────────────────────────────────────────────────────────────────────

export interface DeviceFlowStart {
  device_code: string
  user_code: string
  verification_uri: string
  expires_in: number
  interval: number
}

export interface DeviceFlowPollResult {
  /** "pending" | "success" | "expired" | "denied" | "slow_down" | "error" */
  status: string
  account?: GitHubAccount
}

export interface GitHubAccount {
  id: string
  login: string
  display_name: string | null
  avatar_url: string | null
  created_at: string
  updated_at: string
}

export interface GitHubRepo {
  id: number
  full_name: string
  name: string
  owner_login: string
  is_private: boolean
  description: string | null
  default_branch: string
  html_url: string
  ssh_url: string
  clone_url: string
  updated_at: string
}

export interface GitHubRepoListResponse {
  repos: GitHubRepo[]
  page: number
  has_more: boolean
}
