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

declare global {
  interface Window {
    api: {
      workspace: {
        list: () => Promise<Workspace[]>
        create: (name: string) => Promise<Workspace>
        rename: (id: string, name: string) => Promise<Workspace>
        delete: (id: string) => Promise<void>
      }
      provider: {
        list: () => Promise<ProviderConfig[]>
        create: (req: CreateProviderRequest) => Promise<ProviderConfig>
        update: (id: string, req: UpdateProviderRequest) => Promise<ProviderConfig>
        delete: (id: string) => Promise<void>
        test: (id: string) => Promise<{ ok: boolean; message: string }>
        models: (id: string) => Promise<{ models: string[] }>
      }
      app: {
        getVersion: () => Promise<string>
        getUserDataPath: () => Promise<string>
        getLogsPath: () => Promise<string>
      }
    }
  }
}
