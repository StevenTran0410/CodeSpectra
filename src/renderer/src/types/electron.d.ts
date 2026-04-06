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
export type SyncMode = 'latest' | 'pinned'
export type ClonePolicy = 'full' | 'shallow' | 'partial'

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
  active_snapshot_id: string | null
  sync_mode: SyncMode
  pinned_ref: string | null
  ignore_overrides: string[]
  detect_submodules: boolean
  added_at: string
  last_validated_at: string
}

export interface RepoSnapshot {
  id: string
  local_repo_id: string
  branch: string | null
  commit_hash: string | null
  local_path: string
  status: 'pending' | 'syncing' | 'ready' | 'failed'
  error: string | null
  clone_policy: ClonePolicy
  manual_ignores: string[]
  synced_at: string
  created_at: string
}

export interface EstimateFileCountResponse {
  estimated_file_count: number
  workspace_default_ignores: string[]
  repo_ignore_overrides: string[]
  effective_ignores: string[]
}

export interface AnalysisEstimateResponse {
  repo_id: string
  snapshot_id: string
  file_count: number
  estimated_tokens: number
}

export interface AnalysisReportSummary {
  id: string
  job_id: string
  repo_id: string
  repo_name: string | null
  snapshot_id: string
  branch: string | null
  provider_id: string
  model_id: string
  scan_mode: 'quick' | 'full'
  privacy_mode: 'strict_local' | 'byok_cloud'
  created_at: string
}

export interface AnalysisReport {
  summary: AnalysisReportSummary
  report: {
    sections?: Array<{
      section: string
      content: string
      confidence: 'high' | 'medium' | 'low' | string
      evidence_files: string[]
      blind_spots: string[]
      details?: Record<string, unknown>
    }>
    confidence_summary?: {
      high: number
      medium: number
      low: number
    }
  }
}

export interface ManifestTreeNode {
  path: string
  is_dir: boolean
}

export interface ManifestTreeResponse {
  snapshot_id: string
  nodes: ManifestTreeNode[]
}

export interface ManifestFileContentResponse {
  snapshot_id: string
  rel_path: string
  content: string
  truncated: boolean
}

export type SymbolKind =
  | 'class'
  | 'function'
  | 'method'
  | 'interface'
  | 'enum'
  | 'type'
  | 'variable'
  | 'module'

export interface SymbolRecord {
  id: string
  snapshot_id: string
  rel_path: string
  language: string | null
  name: string
  kind: SymbolKind
  line_start: number
  line_end: number
  signature: string | null
  parent_name: string | null
  extract_source: 'ast' | 'lexical'
}

export interface RepoMapSummary {
  snapshot_id: string
  total_symbols: number
  files_indexed: number
  parse_failures: number
  extract_mode: 'lexical' | 'hybrid'
  language_breakdown: Record<string, number>
  kind_breakdown: Record<string, number>
  generated_at: string
}

export interface GraphNodeScore {
  rel_path: string
  indegree: number
  outdegree: number
  score: number
}

export interface StructuralGraphSummary {
  snapshot_id: string
  total_nodes: number
  total_edges: number
  external_edges: number
  entrypoints: string[]
  top_central_files: GraphNodeScore[]
  generated_at: string
  native_toolchain: string | null
}

export interface GraphEdge {
  snapshot_id: string
  src_path: string
  dst_path: string
  edge_type: string
  is_external: boolean
}

export interface GraphNeighborsResponse {
  snapshot_id: string
  seed_path: string
  hops: number
  nodes: string[]
  edges: GraphEdge[]
}

export type RetrievalMode = 'hybrid' | 'vectorless'
export type RetrievalSection =
  | 'architecture'
  | 'conventions'
  | 'feature_map'
  | 'important_files'
  | 'glossary'

export interface RetrievalEvidence {
  chunk_id: string
  rel_path: string
  chunk_index: number
  reason_codes: string[]
  score: number
  token_estimate: number
  excerpt: string
}

export interface RetrievalBundle {
  snapshot_id: string
  mode: RetrievalMode
  section: RetrievalSection
  query: string
  budget_tokens: number
  used_tokens: number
  evidences: RetrievalEvidence[]
}

export interface RetrievalCompareResponse {
  snapshot_id: string
  section: RetrievalSection
  query: string
  baseline: RetrievalBundle
  vectorless: RetrievalBundle
  precision_at_5_delta: number
  evidence_hit_rate_delta: number
  token_cost_delta: number
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
        branches: (id: string, refresh?: boolean) => Promise<string[]>
        setBranch: (id: string, branch: string) => Promise<LocalRepo>
        setActiveSnapshot: (id: string, snapshotId: string | null) => Promise<LocalRepo>
        updateSettings: (
          id: string,
          settings: {
            sync_mode: SyncMode
            pinned_ref: string | null
            ignore_overrides: string[]
            detect_submodules: boolean
          }
        ) => Promise<LocalRepo>
        estimateFileCount: (id: string) => Promise<EstimateFileCountResponse>
        cloneFromUrl: (url: string) => Promise<LocalRepo>
      }
      sync: {
        prepare: (body: {
          local_repo_id: string
          branch?: string | null
          clone_policy?: ClonePolicy
        }) => Promise<RepoSnapshot>
        listForRepo: (repoId: string) => Promise<RepoSnapshot[]>
        getSnapshot: (snapshotId: string) => Promise<RepoSnapshot>
        deleteSnapshot: (snapshotId: string) => Promise<void>
      }
      manifest: {
        build: (snapshotId: string, manualIgnores?: string[]) => Promise<{
          snapshot_id: string
          total_files: number
          new_files: number
          changed_files: number
          unchanged_files: number
          ignored_files: number
        }>
        tree: (snapshotId: string) => Promise<ManifestTreeResponse>
        file: (snapshotId: string, relPath: string) => Promise<ManifestFileContentResponse>
      }
      repomap: {
        build: (snapshotId: string, forceRebuild?: boolean) => Promise<{ summary: RepoMapSummary }>
        summary: (snapshotId: string) => Promise<RepoMapSummary>
        symbols: (snapshotId: string, limit?: number, pathPrefix?: string) => Promise<{
          snapshot_id: string
          symbols: SymbolRecord[]
        }>
        search: (snapshotId: string, q: string, limit?: number) => Promise<{
          snapshot_id: string
          symbols: SymbolRecord[]
        }>
        exportCsv: (snapshotId: string, excludeTests?: boolean) => Promise<{
          saved: boolean
          file_path: string | null
          row_count: number
        }>
      }
      graph: {
        build: (snapshotId: string, forceRebuild?: boolean) => Promise<{ summary: StructuralGraphSummary }>
        summary: (snapshotId: string) => Promise<StructuralGraphSummary | null>
        edges: (snapshotId: string, limit?: number) => Promise<{
          snapshot_id: string
          edges: GraphEdge[]
        }>
        neighbors: (
          snapshotId: string,
          seedPath: string,
          hops?: number,
          limit?: number
        ) => Promise<GraphNeighborsResponse>
      }
      retrieval: {
        buildIndex: (snapshotId: string, forceRebuild?: boolean) => Promise<{
          snapshot_id: string
          chunk_count: number
          files_indexed: number
          generated_at: string
        }>
        retrieve: (body: {
          snapshot_id: string
          query: string
          section: RetrievalSection
          mode?: RetrievalMode
          max_results?: number
        }) => Promise<RetrievalBundle>
        compare: (body: {
          snapshot_id: string
          query: string
          section: RetrievalSection
          max_results?: number
        }) => Promise<RetrievalCompareResponse>
      }
      analysis: {
        estimate: (repoId: string, snapshotId: string) => Promise<AnalysisEstimateResponse>
        start: (body: {
          repo_id: string
          snapshot_id: string
          scan_mode: 'quick' | 'full'
          privacy_mode: 'strict_local' | 'byok_cloud'
          provider_id: string
          model_id: string
        }) => Promise<Job>
        listReports: (repoId?: string, limit?: number) => Promise<AnalysisReportSummary[]>
        getReport: (reportId: string) => Promise<AnalysisReport>
        getReportByJob: (jobId: string) => Promise<AnalysisReport>
        deleteReport: (reportId: string) => Promise<void>
        exportReportMarkdown: (reportId: string) => Promise<{
          saved: boolean
          file_path: string | null
        }>
        deleteRepot: (reportId: string) => Promise<void>
        deleterepot: (reportId: string) => Promise<void>
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
        getDiagnostics: () => Promise<{
          python_version: string
          native_module_loaded: boolean
          native_functions: Array<{
            name: string
            available: boolean
            description: string
          }>
        }>
      }
    }
  }
}
