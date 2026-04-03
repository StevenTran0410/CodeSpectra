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
    branches: (id: string): Promise<string[]> => ipcRenderer.invoke('folder:branches', id),
    setBranch: (id: string, branch: string) => ipcRenderer.invoke('folder:setBranch', id, branch),
    cloneFromUrl: (url: string) => ipcRenderer.invoke('folder:cloneFromUrl', url)
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
    getLogsPath: (): Promise<string> => ipcRenderer.invoke('app:get-logs-path')
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
