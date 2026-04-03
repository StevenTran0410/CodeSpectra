// ── Job / Analysis pipeline ───────────────────────────────────────────────────

export type JobStatus = 'pending' | 'running' | 'done' | 'failed' | 'cancelled'
export type StepStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped'

export interface StepState {
  status: StepStatus
  progress: number       // 0-100
  message: string | null
}

export interface Job {
  id: string
  type: string
  repo_id: string | null
  status: JobStatus
  steps: Record<string, StepState>
  current_step: string | null
  error: string | null
  started_at: string
  finished_at: string | null
}

// ── Workspace ─────────────────────────────────────────────────────────────────

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
  git_branch: string | null      // actual HEAD branch at last validation
  git_head_hash: string | null
  git_remote_url: string | null
  has_size_warning: boolean
  selected_branch: string | null // user-chosen analysis branch (null = use HEAD)
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

// ── Providers ─────────────────────────────────────────────────────────────────

export type ProviderKind = 'ollama' | 'lmstudio' | 'openai' | 'anthropic' | 'gemini' | 'deepseek'

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
        branches: (id: string) => Promise<string[]>
        setBranch: (id: string, branch: string) => Promise<LocalRepo>
        cloneFromUrl: (url: string) => Promise<LocalRepo>
      }
      git: {
        getConfig: () => Promise<{ ssh_key_path: string | null }>
        setConfig: (sshKeyPath: string | null) => Promise<{ ssh_key_path: string | null }>
        pickSshKey: () => Promise<string | null>
      }
      job: {
        get: (id: string) => Promise<Job>
        cancel: (id: string) => Promise<Job>
        listForRepo: (repoId: string) => Promise<Job[]>
        listRecent: () => Promise<Job[]>
      }
      app: {
        getVersion: () => Promise<string>
        getUserDataPath: () => Promise<string>
        getLogsPath: () => Promise<string>
      }
    }
  }
}
