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
    create: (name: string): Promise<Workspace> => ipcRenderer.invoke('workspace:create', name),
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
  folder: {
    pick: (): Promise<string | null> => ipcRenderer.invoke('folder:pick'),
    validate: (path: string) => ipcRenderer.invoke('folder:validate', path),
    list: () => ipcRenderer.invoke('folder:list'),
    add: (path: string) => ipcRenderer.invoke('folder:add', path),
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
      }
    ) => ipcRenderer.invoke('folder:updateSettings', id, settings),
    estimateFileCount: (id: string) => ipcRenderer.invoke('folder:estimateFileCount', id),
    cloneFromUrl: (url: string) => ipcRenderer.invoke('folder:cloneFromUrl', url)
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
    edges: (snapshotId: string, limit = 2000) => ipcRenderer.invoke('graph:edges', snapshotId, limit),
    neighbors: (snapshotId: string, seedPath: string, hops = 1, limit = 300) =>
      ipcRenderer.invoke('graph:neighbors', snapshotId, seedPath, hops, limit),
  },
  retrieval: {
    buildIndex: (snapshotId: string, forceRebuild = true) =>
      ipcRenderer.invoke('retrieval:buildIndex', snapshotId, forceRebuild),
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
  },
  analysis: {
    estimate: (repoId: string, snapshotId: string) =>
      ipcRenderer.invoke('analysis:estimate', repoId, snapshotId),
    start: (body: {
      repo_id: string
      snapshot_id: string
      scan_mode: 'quick' | 'full'
      privacy_mode: 'strict_local' | 'byok_cloud'
      provider_id: string
      model_id: string
    }) => ipcRenderer.invoke('analysis:start', body),
    listReports: (repoId?: string, limit = 30) =>
      ipcRenderer.invoke('analysis:listReports', repoId, limit),
    getReport: (reportId: string) =>
      ipcRenderer.invoke('analysis:getReport', reportId),
    getReportByJob: (jobId: string) =>
      ipcRenderer.invoke('analysis:getReportByJob', jobId),
    deleteReport: (reportId: string) =>
      ipcRenderer.invoke('analysis:deleteReport', reportId),
    exportReportMarkdown: (reportId: string) =>
      ipcRenderer.invoke('analysis:exportReportMarkdown', reportId),
    onSectionDone: (cb: (event: unknown, data: unknown) => void) => {
      ipcRenderer.on('analysis:section_done', cb)
    },
    offSectionDone: (cb: (event: unknown, data: unknown) => void) => {
      ipcRenderer.removeListener('analysis:section_done', cb)
    },
    // backward-compat typo alias
    deleteRepot: (reportId: string) =>
      ipcRenderer.invoke('analysis:deleteRepot', reportId),
    // lowercase alias for ad-hoc console calls
    deleterepot: (reportId: string) =>
      ipcRenderer.invoke('analysis:deleteRepot', reportId),
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
  app: {
    getVersion: (): Promise<string> => ipcRenderer.invoke('app:get-version'),
    getUserDataPath: (): Promise<string> => ipcRenderer.invoke('app:get-user-data-path'),
    getLogsPath: (): Promise<string> => ipcRenderer.invoke('app:get-logs-path'),
    getDiagnostics: () => ipcRenderer.invoke('app:get-diagnostics')
  }
}

if (process.contextIsolated) {
  contextBridge.exposeInMainWorld('electron', electronAPI)
  contextBridge.exposeInMainWorld('api', api)
} else {
  // @ts-expect-error fallback
  window.electron = electronAPI
  // @ts-expect-error fallback
  window.api = api
}
