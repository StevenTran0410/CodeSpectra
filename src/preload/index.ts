import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type {
  Workspace,
  ProviderConfig,
  CreateProviderRequest,
  UpdateProviderRequest,
} from '../main/api/types'

const api = {
  workspace: {
    list: (): Promise<Workspace[]> => ipcRenderer.invoke('workspace:list'),
    create: (name: string, description?: string): Promise<Workspace> => ipcRenderer.invoke('workspace:create', name, description),
    rename: (id: string, name: string): Promise<Workspace> =>
      ipcRenderer.invoke('workspace:rename', id, name),
    delete: (id: string): Promise<void> => ipcRenderer.invoke('workspace:delete', id)
  },
  provider: {
    list: (): Promise<ProviderConfig[]> => ipcRenderer.invoke('provider:list'),
    create: (req: CreateProviderRequest): Promise<ProviderConfig> =>
      ipcRenderer.invoke('provider:create', req),
    update: (id: string, req: UpdateProviderRequest): Promise<ProviderConfig> =>
      ipcRenderer.invoke('provider:update', id, req),
    delete: (id: string): Promise<void> => ipcRenderer.invoke('provider:delete', id),
    test: (id: string): Promise<{ ok: boolean; message: string; warning?: string }> =>
      ipcRenderer.invoke('provider:test', id),
    models: (id: string): Promise<{ models: string[] }> =>
      ipcRenderer.invoke('provider:models', id)
  },
  consent: {
    checkCloud: (): Promise<{ given: boolean }> => ipcRenderer.invoke('consent:cloud:check'),
    giveCloud: (given: boolean): Promise<{ given: boolean }> =>
      ipcRenderer.invoke('consent:cloud:give', given)
  },
  gpuReranker: {
    status: (): Promise<unknown> => ipcRenderer.invoke('gpuReranker:status'),
    setEnabled: (enabled: boolean): Promise<unknown> =>
      ipcRenderer.invoke('gpuReranker:setEnabled', enabled),
    download: (): Promise<unknown> => ipcRenderer.invoke('gpuReranker:download')
  },
  localEmbedding: {
    status: (): Promise<unknown> => ipcRenderer.invoke('localEmbedding:status'),
    setEnabled: (enabled: boolean): Promise<unknown> =>
      ipcRenderer.invoke('localEmbedding:setEnabled', enabled),
    download: (): Promise<unknown> => ipcRenderer.invoke('localEmbedding:download')
  },
  folder: {
    pick: (): Promise<string | null> => ipcRenderer.invoke('folder:pick'),
    validate: (path: string) => ipcRenderer.invoke('folder:validate', path),
    list: (workspaceId?: string, mode?: string) => ipcRenderer.invoke('folder:list', workspaceId, mode),
    add: (path: string, workspaceId?: string, mode?: string) => ipcRenderer.invoke('folder:add', path, workspaceId, mode),
    remove: (id: string): Promise<void> => ipcRenderer.invoke('folder:remove', id),
    revalidate: (id: string) => ipcRenderer.invoke('folder:revalidate', id),
    branches: (id: string, refresh = false): Promise<string[]> =>
      ipcRenderer.invoke('folder:branches', id, refresh),
    setBranch: (id: string, branch: string) => ipcRenderer.invoke('folder:setBranch', id, branch),
    setActiveSnapshot: (id: string, snapshotId: string | null) =>
      ipcRenderer.invoke('folder:setActiveSnapshot', id, snapshotId),
    updateSettings: (
      id: string,
      settings: {
        sync_mode: 'latest' | 'pinned'
        pinned_ref: string | null
        ignore_overrides: string[]
        detect_submodules: boolean
        include_tests: boolean
      }
    ) => ipcRenderer.invoke('folder:updateSettings', id, settings),
    estimateFileCount: (id: string) => ipcRenderer.invoke('folder:estimateFileCount', id),
    cloneFromUrl: (url: string, workspaceId?: string, mode?: string) => ipcRenderer.invoke('folder:cloneFromUrl', url, workspaceId, mode)
  },
  sync: {
    prepare: (body: {
      local_repo_id: string
      branch?: string | null
      clone_policy?: 'full' | 'shallow' | 'partial'
    }) => ipcRenderer.invoke('sync:prepare', body),
    listForRepo: (repoId: string) => ipcRenderer.invoke('sync:listForRepo', repoId),
    getSnapshot: (snapshotId: string) => ipcRenderer.invoke('sync:getSnapshot', snapshotId),
    deleteSnapshot: (snapshotId: string): Promise<void> => ipcRenderer.invoke('sync:deleteSnapshot', snapshotId),
  },
  manifest: {
    build: (snapshotId: string, manualIgnores?: string[]) =>
      ipcRenderer.invoke('manifest:build', snapshotId, manualIgnores),
    tree: (snapshotId: string) => ipcRenderer.invoke('manifest:tree', snapshotId),
    file: (snapshotId: string, relPath: string) =>
      ipcRenderer.invoke('manifest:file', snapshotId, relPath),
  },
  repomap: {
    build: (snapshotId: string, forceRebuild = true) =>
      ipcRenderer.invoke('repomap:build', snapshotId, forceRebuild),
    summary: (snapshotId: string) => ipcRenderer.invoke('repomap:summary', snapshotId),
    symbols: (snapshotId: string, limit = 300, pathPrefix?: string) =>
      ipcRenderer.invoke('repomap:symbols', snapshotId, limit, pathPrefix),
    search: (snapshotId: string, q: string, limit = 120) =>
      ipcRenderer.invoke('repomap:search', snapshotId, q, limit),
    exportCsv: (snapshotId: string, excludeTests = true) =>
      ipcRenderer.invoke('repomap:exportCsv', snapshotId, excludeTests),
  },
  graph: {
    build: (snapshotId: string, forceRebuild = true) =>
      ipcRenderer.invoke('graph:build', snapshotId, forceRebuild),
    summary: (snapshotId: string) => ipcRenderer.invoke('graph:summary', snapshotId),
    edges: (snapshotId: string, limit = 2000, internalOnly = false) => ipcRenderer.invoke('graph:edges', snapshotId, limit, internalOnly),
    neighbors: (snapshotId: string, seedPath: string, hops = 1, limit = 300) =>
      ipcRenderer.invoke('graph:neighbors', snapshotId, seedPath, hops, limit),
    communities: (snapshotId: string) => ipcRenderer.invoke('graph:communities', snapshotId),
    communityForNode: (snapshotId: string, path: string) =>
      ipcRenderer.invoke('graph:communityForNode', snapshotId, path),
    cycles: (snapshotId: string) => ipcRenderer.invoke('graph:cycles', snapshotId),
    symbolEdges: (snapshotId: string, filePath: string) =>
      ipcRenderer.invoke('graph:symbolEdges', snapshotId, filePath),
    exportData: (snapshotId: string) => ipcRenderer.invoke('graph:exportData', snapshotId),
    exportJson: (snapshotId: string) => ipcRenderer.invoke('graph:exportJson', snapshotId),
  },
  query: {
    exportCsv: (csv: string, defaultName: string) =>
      ipcRenderer.invoke('query:exportCsv', csv, defaultName),
  },
  retrieval: {
    buildIndex: (snapshotId: string, forceRebuild = true) =>
      ipcRenderer.invoke('retrieval:buildIndex', snapshotId, forceRebuild),
    summary: (snapshotId: string) => ipcRenderer.invoke('retrieval:summary', snapshotId),
    retrieve: (body: {
      snapshot_id: string
      query: string
      section: 'architecture' | 'conventions' | 'feature_map' | 'important_files' | 'glossary'
      mode?: 'hybrid' | 'vectorless'
      max_results?: number
    }) => ipcRenderer.invoke('retrieval:retrieve', body),
    compare: (body: {
      snapshot_id: string
      query: string
      section: 'architecture' | 'conventions' | 'feature_map' | 'important_files' | 'glossary'
      max_results?: number
    }) => ipcRenderer.invoke('retrieval:compare', body),
    retrieveTwoStage: (body: {
      snapshot_id: string
      query: string
      section: 'architecture' | 'conventions' | 'feature_map' | 'important_files' | 'glossary'
      budget?: number
    }) => ipcRenderer.invoke('retrieval:retrieveTwoStage', body),
    retrieveRrfFusion: (body: {
      snapshot_id: string
      query: string
      section: 'architecture' | 'conventions' | 'feature_map' | 'important_files' | 'glossary'
      budget?: number
    }) => ipcRenderer.invoke('retrieval:retrieveRrfFusion', body),
  },
  analysis: {
    start: (body: {
      repo_id: string
      snapshot_id: string
      scan_mode: 'quick' | 'full'
      privacy_mode: 'strict_local' | 'byok_cloud'
      provider_id: string
      model_id: string
      force_rerun?: boolean
      large_codebase_mode?: boolean
    }): Promise<{
      id: string
      warning?: { code: string; message: string; severity: string } | null
    }> => ipcRenderer.invoke('analysis:start', body),
    listReports: (repoId?: string, limit = 30, workspaceId?: string) =>
      ipcRenderer.invoke('analysis:listReports', repoId, limit, workspaceId),
    getReport: (reportId: string) =>
      ipcRenderer.invoke('analysis:getReport', reportId),
    getReportByJob: (jobId: string) =>
      ipcRenderer.invoke('analysis:getReportByJob', jobId),
    deleteReport: (reportId: string) =>
      ipcRenderer.invoke('analysis:deleteReport', reportId),
    exportReportMarkdown: (reportId: string) =>
      ipcRenderer.invoke('analysis:exportReportMarkdown', reportId),
    exportAuditSection: (reportId: string) =>
      ipcRenderer.invoke('analysis:exportAuditSection', reportId),
    rerunSection: (body: {
      report_id: string
      section: string
      provider_id: string
      model_id: string
    }) => ipcRenderer.invoke('analysis:rerunSection', body),
    compareReports: (body: { report_id_a: string; report_id_b: string }) =>
      ipcRenderer.invoke('analysis:compareReports', body),
    getSectionSources: (reportId: string, sectionId: string) =>
      ipcRenderer.invoke('analysis:getSectionSources', reportId, sectionId),
    getStaleness: (reportId: string) =>
      ipcRenderer.invoke('analysis:getStaleness', reportId),
    pollEvents: (jobId: string, fromIdx?: number) =>
      ipcRenderer.invoke('analysis:pollEvents', jobId, fromIdx ?? 0),
    onSectionDone: (cb: (event: unknown, data: unknown) => void) => {
      ipcRenderer.on('analysis:section_done', cb)
    },
    offSectionDone: (cb: (event: unknown, data: unknown) => void) => {
      ipcRenderer.removeListener('analysis:section_done', cb)
    },
  },
  git: {
    getConfig: (): Promise<{ ssh_key_path: string | null }> =>
      ipcRenderer.invoke('git:getConfig'),
    setConfig: (sshKeyPath: string | null): Promise<{ ssh_key_path: string | null }> =>
      ipcRenderer.invoke('git:setConfig', sshKeyPath),
    pickSshKey: (): Promise<string | null> =>
      ipcRenderer.invoke('git:pickSshKey'),
  },
  job: {
    get: (id: string) => ipcRenderer.invoke('job:get', id),
    cancel: (id: string) => ipcRenderer.invoke('job:cancel', id),
    listForRepo: (repoId: string) => ipcRenderer.invoke('job:listForRepo', repoId),
    listRecent: () => ipcRenderer.invoke('job:listRecent'),
  },
  qa: {
    ask: (body: {
      snapshot_id: string
      question: string
      provider_id: string
      model_id: string
      report_id?: string
      include_debug?: boolean
    }) => ipcRenderer.invoke('qa:ask', body),
    askStream: (body: {
      snapshot_id: string
      question: string
      provider_id: string
      model_id: string
      report_id?: string
      include_debug?: boolean
    }) => ipcRenderer.send('qa:ask:stream', body),
    classifyIntent: (body: { question: string }) => ipcRenderer.invoke('qa:classifyIntent', body),
    classifier: {
      status: (): Promise<{ trained: boolean; backend: string; builtin_examples: number; user_examples: number }> =>
        ipcRenderer.invoke('qa:classifier:status'),
      examples: (): Promise<{ id: string; text: string; is_deep_research: boolean; created_at: string }[]> =>
        ipcRenderer.invoke('qa:classifier:examples'),
      addExample: (body: { text: string; is_deep_research: boolean }) =>
        ipcRenderer.invoke('qa:classifier:addExample', body),
      deleteExample: (id: string): Promise<void> => ipcRenderer.invoke('qa:classifier:deleteExample', id),
      retrain: (): Promise<{ trained: boolean; backend: string; builtin_examples: number; user_examples: number }> =>
        ipcRenderer.invoke('qa:classifier:retrain'),
    },
    deepResearch: (body: {
      snapshot_id: string
      question: string
      provider_id: string
      model_id: string
      report_id?: string
      max_hops?: number
      include_debug?: boolean
    }) => ipcRenderer.invoke('qa:deepResearch', body),
    deepResearchStream: (body: {
      snapshot_id: string
      question: string
      provider_id: string
      model_id: string
      report_id?: string
      max_hops?: number
      include_debug?: boolean
    }) => ipcRenderer.send('qa:deepResearch:stream', body),
    onStreamEvent: (cb: (event: unknown, data: unknown) => void) => {
      ipcRenderer.on('qa:stream-event', cb)
    },
    offStreamEvent: (cb: (event: unknown, data: unknown) => void) => {
      ipcRenderer.removeListener('qa:stream-event', cb)
    },
  },
  impact: {
    blastRadius: (body: {
      snapshot_id: string
      changed_files: string[]
      report_id?: string | null
      max_hops?: number
      include_call_chains?: boolean
    }) => ipcRenderer.invoke('impact:blastRadius', body),
    plan: (body: {
      snapshot_id: string
      task_description: string
      report_id: string
      provider_id?: string | null
      model_id?: string | null
    }) => ipcRenderer.invoke('impact:plan', body),
  },
  app: {
    getVersion: (): Promise<string> => ipcRenderer.invoke('app:get-version'),
    getUserDataPath: (): Promise<string> => ipcRenderer.invoke('app:get-user-data-path'),
    getLogsPath: (): Promise<string> => ipcRenderer.invoke('app:get-logs-path'),
    getDiagnostics: () => ipcRenderer.invoke('app:get-diagnostics'),
    retryBackend: (): Promise<void> => ipcRenderer.invoke('app:retry-backend'),
  },
  aeh: {
    start: (): Promise<number> => ipcRenderer.invoke('aeh:start'),
    listRuns: () => ipcRenderer.invoke('aeh:listRuns'),
    runDetail: (runId: string) => ipcRenderer.invoke('aeh:runDetail', runId),
    componentEvaluations: (runId: string, componentId: string) =>
      ipcRenderer.invoke('aeh:componentEvaluations', runId, componentId),
    traceDetail: (traceId: string) => ipcRenderer.invoke('aeh:traceDetail', traceId),
    providers: () => ipcRenderer.invoke('aeh:providers'),
    rerun: (runId: string, body: unknown) => ipcRenderer.invoke('aeh:rerun', runId, body),
    startDiscovery: (body: unknown) => ipcRenderer.invoke('aeh:startDiscovery', body),
    listDiscoverySessions: (repoRef?: string, snapshotId?: string) => ipcRenderer.invoke('aeh:listDiscoverySessions', repoRef, snapshotId),
    getDiscoverySession: (sessionId: string) => ipcRenderer.invoke('aeh:getDiscoverySession', sessionId),
    resumeDiscoverySession: (
      sessionId: string,
      body?: {
        provider_id?: string | null
        model_id?: string | null
        reasoning_effort?: string | null
        thinking_budget?: number | null
      }
    ) => ipcRenderer.invoke('aeh:resumeDiscoverySession', sessionId, body ?? {}),
    listDiscoveryCandidates: (sessionId: string) =>
      ipcRenderer.invoke('aeh:listDiscoveryCandidates', sessionId),
    updateDiscoveryCandidateVerdict: (candidateId: string, verdict: string) =>
      ipcRenderer.invoke('aeh:updateDiscoveryCandidateVerdict', candidateId, verdict),
    updateDiscoveryCandidateExcludedFiles: (candidateId: string, excludedFiles: string[]) =>
      ipcRenderer.invoke('aeh:updateDiscoveryCandidateExcludedFiles', candidateId, excludedFiles),
    startExpansion: (candidateId: string, body: unknown) =>
      ipcRenderer.invoke('aeh:startExpansion', candidateId, body),
    getExpansionSession: (sessionId: string) =>
      ipcRenderer.invoke('aeh:getExpansionSession', sessionId),
    listExpansionSessions: (candidateId: string) =>
      ipcRenderer.invoke('aeh:listExpansionSessions', candidateId),
    getExpansionMap: (sessionId: string) =>
      ipcRenderer.invoke('aeh:getExpansionMap', sessionId),
    updateExpansionMap: (sessionId: string, map: unknown) =>
      ipcRenderer.invoke('aeh:updateExpansionMap', sessionId, map),
    generatePlan: (sessionId: string, body: unknown) =>
      ipcRenderer.invoke('aeh:generatePlan', sessionId, body),
    getPlan: (sessionId: string) =>
      ipcRenderer.invoke('aeh:getPlan', sessionId),
    updatePlan: (sessionId: string, body: unknown) =>
      ipcRenderer.invoke('aeh:updatePlan', sessionId, body),
    getPlanReport: (sessionId: string) =>
      ipcRenderer.invoke('aeh:getPlanReport', sessionId),
    generateAgentFlowMap: (sessionId: string, body: unknown) =>
      ipcRenderer.invoke('aeh:generateAgentFlowMap', sessionId, body),
    getAgentFlowMap: (sessionId: string) =>
      ipcRenderer.invoke('aeh:getAgentFlowMap', sessionId),
    advanceSession: (sessionId: string, body: unknown) =>
      ipcRenderer.invoke('aeh:advanceSession', sessionId, body),
    fulfillDatasets: (sessionId: string, body: unknown) =>
      ipcRenderer.invoke('aeh:fulfillDatasets', sessionId, body),
    listDatasets: () => ipcRenderer.invoke('aeh:listDatasets'),
    getDatasetCases: (datasetId: string) =>
      ipcRenderer.invoke('aeh:getDatasetCases', datasetId),
    caseVerdict: (caseId: string, body: unknown) =>
      ipcRenderer.invoke('aeh:caseVerdict', caseId, body)
  }
}

// contextBridge.exposeInMainWorld doesn't support Proxy objects — they get stripped across the context isolation boundary, so window.api.* would be undefined in the renderer.
if (process.contextIsolated) {
  contextBridge.exposeInMainWorld('electron', electronAPI)
  contextBridge.exposeInMainWorld('api', api)
} else {
  // @ts-expect-error fallback
  window.electron = electronAPI
  // @ts-expect-error fallback
  window.api = api
}
