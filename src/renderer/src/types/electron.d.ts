// ── Job / Analysis pipeline ───────────────────────────────────────────────────

export interface SectionDoneEvent {
  type?: string
  section: string
  status: 'done' | 'error'
  duration_ms?: number
  data?: unknown
  error?: string | null
}

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
  description?: string
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
  include_tests: boolean
  mode: 'code_analysis' | 'aeh'
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
  commit_hash: string | null
  provider_id: string
  model_id: string
  scan_mode: 'quick' | 'full'
  privacy_mode: 'strict_local' | 'byok_cloud'
  created_at: string
}

export interface ReportSectionDiff {
  letter: string
  changed: boolean
  skipped_by_hash: boolean
  confidence_delta: string | null
  content_word_delta_pct: number | null
  list_added: string[]
  list_removed: string[]
  section_score_changes: Record<string, string>
  improvement: boolean | null
}

export interface ReportDiffResult {
  report_id_a: string
  report_id_b: string
  quality_trend: string
  sections_changed: number
  identical: boolean
  section_diffs: Record<string, ReportSectionDiff>
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

export interface CommunityInfo {
  community_id: number
  member_count: number
  hub_paths: string[]
  modularity_contribution: number
  neighbor_community_ids: number[]
  is_singleton: boolean
  llm_summary: string | null
  generated_at: string
}

export interface GraphCommunitiesResponse {
  snapshot_id: string
  total_communities: number
  communities: CommunityInfo[]
  node_index: Record<string, number>
}

export interface NodeCommunityResponse {
  snapshot_id: string
  node_path: string
  community_id: number
  members: string[]
}

export interface CyclesResponse {
  snapshot_id: string
  cycles: string[][]
}

// ── CS-250: function-level symbol edge drill-down (graph viewer) ─────────────

export interface SymbolEdgeInfo {
  src_symbol: string
  dst_symbol: string
  edge_type: string
  confidence_score: number
  resolution_method: string
}

export interface FileSymbolEdgesResponse {
  snapshot_id: string
  file_path: string
  defined_symbols: string[]
  outgoing: SymbolEdgeInfo[]
  incoming: SymbolEdgeInfo[]
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

export interface TwoStageCandidate {
  chunk_id: string
  rel_path: string
  chunk_index: number
  bm25_score: number
  token_estimate: number
  excerpt: string
}

export interface TwoStageExpansion {
  seed_path: string
  symbol_refs: string[]
  community_members: string[]
  neighbor_files: string[]
  net_new_count: number
}

export interface TwoStageRankedChunk {
  chunk_id: string
  rel_path: string
  chunk_index: number
  score: number
  bm25_component: number
  symbol_bonus: number
  module_bonus: number
  centrality_bonus: number
  token_estimate: number
  excerpt: string
}

export interface SignalRankEntry {
  chunk_id: string
  rel_path: string
  rank: number
  raw_score: number
  signal_name: string
  token_estimate?: number
}

export interface FusedRankEntry {
  chunk_id: string
  rel_path: string
  fused_score: number
  per_signal_ranks: Record<string, number>
  excerpt: string
  token_estimate?: number
}

export interface RerankedEntry {
  chunk_id: string
  rel_path: string
  fused_score: number
  rerank_score: number
  fused_rank: number
  excerpt: string
  token_estimate?: number
}

export interface RrfFusionDebugBundle {
  snapshot_id: string
  query: string
  section: string
  bm25_signal: SignalRankEntry[]
  graph_signal: SignalRankEntry[]
  module_signal: SignalRankEntry[]
  category_signal: SignalRankEntry[]
  fused: FusedRankEntry[]
  reranked?: RerankedEntry[]
  reranker_status?: 'ok' | 'no_gpu' | 'model_load_failed' | 'disabled'
  final?: FusedRankEntry[]
}

export interface GpuRerankerStatus {
  enabled: boolean
  gpu_available: boolean
  vram_gb: number | null
}

export interface TwoStageDebugBundle {
  snapshot_id: string
  query: string
  section: string
  stage1: { candidates: TwoStageCandidate[] }
  stage2: { expansions: TwoStageExpansion[] }
  stage3: {
    ranked: TwoStageRankedChunk[]
    used_tokens: number
    budget_tokens: number
    used_cpp_ranker: boolean
  }
}

export interface StalenessResult {
  stale: boolean
  old_commit?: string
  current_commit?: string
  changed_files_count?: number
  insertions?: number
  deletions?: number
  sections_affected?: string[]
  recommend_new_snapshot?: boolean
}

export interface QACitation {
  file: string
  line_start: number | null
  line_end: number | null
  snippet: string
}

export interface QAResponse {
  answer: string
  citations: QACitation[]
  confidence: 'high' | 'medium' | 'low'
  unknowns: string[]
  suggested_files: string[]
  deep_research_recommended: boolean
  retrieval_debug: Record<string, unknown> | null
}

export interface ResearchStepResult {
  step_number: number
  description: string
  files_involved: string[]
  finding: string
  graph_path: string[] | null
}

export interface DeepResearchResponse {
  summary: string
  reasoning_chain: ResearchStepResult[]
  files_explored: string[]
  confidence: 'high' | 'medium' | 'low'
  unknowns: string[]
  elapsed_ms: number
  research_debug: Record<string, unknown> | null
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
        create: (name: string, description?: string) => Promise<Workspace>
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
      gpuReranker: {
        status: () => Promise<GpuRerankerStatus>
        setEnabled: (enabled: boolean) => Promise<GpuRerankerStatus>
      }
      folder: {
        pick: () => Promise<string | null>
        validate: (path: string) => Promise<ValidateFolderResponse>
        list: (workspaceId?: string, mode?: string) => Promise<LocalRepo[]>
        add: (path: string, workspaceId?: string, mode?: string) => Promise<LocalRepo>
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
            include_tests: boolean
          }
        ) => Promise<LocalRepo>
        estimateFileCount: (id: string) => Promise<EstimateFileCountResponse>
        cloneFromUrl: (url: string, workspaceId?: string, mode?: string) => Promise<LocalRepo>
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
        edges: (snapshotId: string, limit?: number, internalOnly?: boolean) => Promise<{
          snapshot_id: string
          edges: GraphEdge[]
        }>
        neighbors: (
          snapshotId: string,
          seedPath: string,
          hops?: number,
          limit?: number
        ) => Promise<GraphNeighborsResponse>
        communities: (snapshotId: string) => Promise<GraphCommunitiesResponse>
        communityForNode: (snapshotId: string, path: string) => Promise<NodeCommunityResponse>
        cycles: (snapshotId: string) => Promise<CyclesResponse>
        symbolEdges: (snapshotId: string, filePath: string) => Promise<FileSymbolEdgesResponse>
        exportData: (snapshotId: string) => Promise<{
          nodes: string[]
          edges: Array<{ src: string; dst: string; external: boolean }>
          communities: Record<string, number>
          community_groups: Record<string, string[]>
          cycles: string[][]
          test_files: string[]
          generated_at: string
        }>
        exportJson: (snapshotId: string) => Promise<{ saved: boolean; file_path: string | null }>
      }
      query: {
        exportCsv: (csv: string, defaultName: string) => Promise<{ saved: boolean; file_path: string | null }>
      }
      retrieval: {
        buildIndex: (snapshotId: string, forceRebuild?: boolean) => Promise<{
          snapshot_id: string
          chunk_count: number
          files_indexed: number
          generated_at: string
        }>
        summary: (snapshotId: string) => Promise<{
          snapshot_id: string
          chunk_count: number
          has_bm25_stats: boolean
          built: boolean
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
        retrieveTwoStage: (body: {
          snapshot_id: string
          query: string
          section: RetrievalSection
          budget?: number
        }) => Promise<TwoStageDebugBundle>
        retrieveRrfFusion: (body: {
          snapshot_id: string
          query: string
          section: RetrievalSection
          budget?: number
        }) => Promise<RrfFusionDebugBundle>
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
          force_rerun?: boolean
          large_codebase_mode?: boolean
          skip_synthesis?: boolean
        }) => Promise<Job>
        listReports: (repoId?: string, limit?: number, workspaceId?: string) => Promise<AnalysisReportSummary[]>
        getReport: (reportId: string) => Promise<AnalysisReport>
        getReportByJob: (jobId: string) => Promise<AnalysisReport>
        deleteReport: (reportId: string) => Promise<void>
        exportReportMarkdown: (reportId: string) => Promise<{
          saved: boolean
          file_path: string | null
        }>
        exportAuditSection: (reportId: string) => Promise<{
          saved: boolean
          file_path: string | null
        }>
        rerunSection: (body: {
          report_id: string
          section: string
          provider_id: string
          model_id: string
        }) => Promise<{ section: string; data: Record<string, unknown>; duration_ms: number }>
        compareReports: (body: {
          report_id_a: string
          report_id_b: string
        }) => Promise<ReportDiffResult>
        getSectionSources: (reportId: string, sectionId: string) => Promise<unknown>
        pollEvents: (jobId: string, fromIdx?: number) => Promise<{ events: SectionDoneEvent[]; job_done: boolean }>
        getStaleness: (reportId: string) => Promise<StalenessResult>
        onSectionDone: (cb: (event: unknown, data: SectionDoneEvent) => void) => void
        offSectionDone: (cb: (event: unknown, data: SectionDoneEvent) => void) => void

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
      qa: {
        ask: (body: {
          snapshot_id: string
          question: string
          provider_id: string
          model_id: string
          report_id?: string
          include_debug?: boolean
        }) => Promise<QAResponse>
        askStream: (body: {
          snapshot_id: string
          question: string
          provider_id: string
          model_id: string
          report_id?: string
          include_debug?: boolean
        }) => void
        classifyIntent: (body: { question: string }) => Promise<{ deep_research: boolean }>
        classifier: {
          status: () => Promise<{ trained: boolean; backend: string; builtin_examples: number; user_examples: number }>
          examples: () => Promise<{ id: string; text: string; is_deep_research: boolean; created_at: string }[]>
          addExample: (body: { text: string; is_deep_research: boolean }) => Promise<{ id: string; text: string; is_deep_research: boolean; created_at: string }>
          deleteExample: (id: string) => Promise<void>
          retrain: () => Promise<{ trained: boolean; backend: string; builtin_examples: number; user_examples: number }>
        }
        deepResearch: (body: {
          snapshot_id: string
          question: string
          provider_id: string
          model_id: string
          report_id?: string
          max_hops?: number
          include_debug?: boolean
        }) => Promise<DeepResearchResponse>
        deepResearchStream: (body: {
          snapshot_id: string
          question: string
          provider_id: string
          model_id: string
          report_id?: string
          max_hops?: number
          include_debug?: boolean
        }) => void
        onStreamEvent: (cb: (event: unknown, data: unknown) => void) => void
        offStreamEvent: (cb: (event: unknown, data: unknown) => void) => void
      }
      impact: {
        blastRadius: (body: {
          snapshot_id: string
          changed_files: string[]
          report_id?: string | null
          max_hops?: number
          include_call_chains?: boolean
        }) => Promise<unknown>
        plan: (body: {
          snapshot_id: string
          task_description: string
          report_id: string
          provider_id?: string | null
          model_id?: string | null
        }) => Promise<unknown>
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
            module: string
          }>
        }>
      }
      aeh: {
        start: () => Promise<number>
        listRuns: () => Promise<AEHRunListItem[]>
        runDetail: (runId: string) => Promise<AEHRunDetailResponse>
        componentEvaluations: (runId: string, componentId: string) => Promise<AEHEvaluationDetailItem[]>
        traceDetail: (traceId: string) => Promise<AEHTraceDetailResponse>
        datasetCases: (datasetId: string) => Promise<unknown[]>
        providers: () => Promise<AEHProviderSummary[]>
        rerun: (runId: string, body: { model_overrides: Record<string, string>; active_defects: string[] }) => Promise<{ run_id: string }>
        startDiscovery: (body: {
          repo_ref: string
          snapshot_id: string
          provider_id?: string | null
          model_id?: string | null
          backend_url?: string | null
          backend_token?: string | null
        }) => Promise<{ session_id: string }>
        listDiscoverySessions: (repoRef?: string, snapshotId?: string) => Promise<AEHDiscoverySession[]>
        getDiscoverySession: (sessionId: string) => Promise<AEHDiscoverySession>
        resumeDiscoverySession: (sessionId: string, body?: { provider_id?: string | null; model_id?: string | null }) => Promise<{ success: boolean }>
        listDiscoveryCandidates: (sessionId: string) => Promise<AEHDiscoveryCandidate[]>
        updateDiscoveryCandidateVerdict: (candidateId: string, verdict: 'proposed' | 'confirmed' | 'rejected') => Promise<{ success: boolean }>
        updateDiscoveryCandidateExcludedFiles: (candidateId: string, excludedFiles: string[]) => Promise<{ success: boolean }>
        startExpansion: (
          candidateId: string,
          body: {
            provider_id?: string | null
            model_id?: string | null
            node_budget?: number
            hop_cap?: number
            classify_provider_id?: string | null
            classify_model_id?: string | null
          }
        ) => Promise<{ session_id: string }>
        getExpansionSession: (sessionId: string) => Promise<AEHExpansionSession>
        listExpansionSessions: (candidateId: string) => Promise<AEHExpansionSession[]>
        getExpansionMap: (sessionId: string) => Promise<AEHSystemMap>
        updateExpansionMap: (sessionId: string, map: AEHSystemMap) => Promise<{ success: boolean }>
        generatePlan: (
          sessionId: string,
          body: {
            provider_id?: string | null
            model_id?: string | null
          }
        ) => Promise<AEHPlanSuite>
        getPlan: (sessionId: string) => Promise<AEHPlanSuite>
        updatePlan: (sessionId: string, body: { entries: AEHPlanEntry[] }) => Promise<{ success: boolean }>
        advanceSession: (
          sessionId: string,
          body: {
            confirmed_candidates?: string[] | null
            confirmed_map_session_id?: string | null
            confirmed_plan?: boolean | null
            provider_id?: string | null
            model_id?: string | null
          }
        ) => Promise<{ pipeline_stage: string }>
      }
    }
  }

  export interface AEHRunListItem {
    id: string
    target_system_id: string
    eval_plan_id: string | null
    started_at: string
    finished_at: string | null
    status: string
    map_path: string | null
    active_defects: string[]
    pass_rate: number
    judge_cost: number
  }

  export interface AEHComponentAggregate {
    total: number
    passed: number
  }

  export interface AEHSystemMapComponent {
    id: string
    role: string
    model: string | null
    entry_point: string | null
    file?: string
    constraints: Array<{ name: string; value: any; source: string }>
    upstream: string[]
    downstream: string[]
  }

  export interface AEHSystemMap {
    target_system_id: string
    discrepancies: string[]
    components: AEHSystemMapComponent[]
  }

  export interface AEHRunDetailResponse {
    id: string
    target_system_id: string
    eval_plan_id: string | null
    started_at: string
    finished_at: string | null
    status: string
    map_path: string | null
    active_defects: string[]
    system_map: AEHSystemMap
    component_aggregates: Record<string, AEHComponentAggregate>
    overall_pass_rate: number
    target?: string | null
    suite_path?: string | null
    parent_run_id?: string | null
    model_overrides?: Record<string, string>
  }

  export interface AEHEvaluationDetailItem {
    id: string
    metric_name: string
    metric_class: string
    score: number | null
    passed: boolean | null
    details: Record<string, any>
    evaluator: string | null
    cost_tokens: number | null
    trace_id: string | null
    span_id: string | null
    root_input: string | null
    final_output: string | null
    trace_tokens: number | null
    trace_latency: number | null
  }

  export interface AEHTraceSpan {
    id: string
    trace_id: string
    parent_span_id: string | null
    component_id: string | null
    span_type: string
    input_json: string | null
    output_json: string | null
    model: string | null
    tokens_in: number | null
    tokens_out: number | null
    latency_ms: number | null
    started_at: string
    details_json: string
  }

  export interface AEHTraceDetailResponse {
    trace: {
      id: string
      run_id: string
      dataset_case_id: string | null
      root_input: string
      final_output: string | null
      total_tokens: number
      total_latency_ms: number
    } | null
    spans: AEHTraceSpan[]
  }

  export interface AEHProviderSummary {
    provider_id: string
    display_name: string
    model_id: string | null
  }

  export interface AEHDiscoverySession {
    id: string
    repo_ref: string
    snapshot_id: string
    status: 'running' | 'completed' | 'failed' | 'paused_rate_limit'
    error: string | null
    created_at: string
    finished_at: string | null
    pause_info: { reason: string; provider_id: string; model_id: string | null } | null
    pipeline_stage: string
  }

  export interface AEHDiscoveryCandidate {
    id: string
    session_id: string
    name: string
    frameworks: string[]
    entry_points: string[]
    evidence: any[]
    confidence: 'high' | 'medium' | 'low'
    needs_human: boolean
    verdict: 'proposed' | 'confirmed' | 'rejected'
    community_id: string | null
    cluster_files: string[]
    hub_paths: string[]
    wiring_block: {
      nodes: Array<{ alias: string; class_name: string; source_hint_file: string }>
      edges: Array<{ src: string; dst: string }>
      framework: string
      source: 'static' | 'llm_fallback'
    } | null
    excluded_files: string[]
    matched_files?: string[]
    file_provenance?: Record<string, string>
  }

  export interface AEHExpansionSession {
    id: string
    candidate_id: string
    snapshot_id: string
    status: 'running' | 'completed' | 'failed'
    error: string | null
    map_path: string | null
    accepted: string[]
    boundary: string[]
    accepted_edges: { src: string; dst: string }[]
    stop_reason: string | null
    created_at: string
    finished_at: string | null
    plan_path: string | null
  }

  export interface AEHDatasetRef {
    ref?: string | null
    required?: Record<string, any> | null
    waived?: string | null
  }

  export interface AEHPlanEntry {
    id: string
    component: string
    metric: string
    metric_class: 'assertion' | 'classifier' | 'llm_judge'
    dataset?: AEHDatasetRef | null
    params?: Record<string, any>
    rationale?: string
    provenance?: 'rule' | 'human_added' | 'llm_suggested'
    status?: string | null
  }

  export interface AEHPlanSuite {
    entries: AEHPlanEntry[]
  }
}
