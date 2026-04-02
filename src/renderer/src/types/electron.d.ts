export interface Workspace {
  id: string
  name: string
  created_at: string
  updated_at: string
  settings: Record<string, unknown>
}

export type RepoSourceType = 'github' | 'bitbucket' | 'local_folder'

export interface LocalRepo {
  id: string
  path: string
  name: string
  source_type: RepoSourceType
  is_git_repo: boolean
  git_branch: string | null
  git_head_hash: string | null
  git_remote_url: string | null
  has_size_warning: boolean
  added_at: string
  last_validated_at: string
}

export interface ValidateFolderResponse {
  path: string
  name: string
  exists: boolean
  is_directory: boolean
  is_git_repo: boolean
  git_branch: string | null
  git_head_hash: string | null
  git_remote_url: string | null
  has_size_warning: boolean
  size_warning_reason: string | null
}

export type ProviderKind = 'ollama' | 'lmstudio' | 'openai' | 'anthropic' | 'gemini' | 'deepseek'
export const CLOUD_KINDS: ReadonlySet<ProviderKind> = new Set(['openai', 'anthropic', 'gemini', 'deepseek'])

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
  api_key?: string
}

export interface UpdateProviderRequest {
  display_name?: string
  base_url?: string
  model_id?: string
  api_key?: string
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
        test: (id: string) => Promise<{ ok: boolean; message: string; warning?: string }>
        models: (id: string) => Promise<{ models: string[] }>
      }
      consent: {
        checkCloud: () => Promise<{ given: boolean }>
        giveCloud: (given: boolean) => Promise<{ given: boolean }>
      }
      folder: {
        pick: () => Promise<string | null>
        validate: (path: string) => Promise<ValidateFolderResponse>
        list: () => Promise<LocalRepo[]>
        add: (path: string) => Promise<LocalRepo>
        remove: (id: string) => Promise<void>
        revalidate: (id: string) => Promise<LocalRepo>
      }
      app: {
        getVersion: () => Promise<string>
        getUserDataPath: () => Promise<string>
        getLogsPath: () => Promise<string>
      }
    }
  }
}
